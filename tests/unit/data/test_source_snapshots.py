"""Integrity, receipt timing, deduplication, and secret-exclusion contracts."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from regime_allocation.data import source_snapshots as snapshots
from regime_allocation.data.source_snapshots import SnapshotIntegrityError, SnapshotStore


def _metadata() -> dict:
    return {
        "provider_id": "fred_api",
        "series_id": "PAYEMS",
        "source_url": "https://fred.stlouisfed.org/series/PAYEMS",
        "query": {"series_id": "PAYEMS", "output_type": 2, "vintage_dates": ["2020-01-01"]},
        "retrieved_at": datetime(2020, 1, 2, 10, tzinfo=timezone(timedelta(hours=-5))),
        "retrieval_time_status": "observed",
        "source_definition": {"version": "m03_sources_v1", "definition": "Payroll level"},
    }


def test_round_trip_preserves_raw_bytes_and_separate_utc_clocks(tmp_path: Path) -> None:
    raw = b"\x00\xfforiginal\r\nprovider bytes"
    before = datetime.now(UTC)
    store = SnapshotStore(tmp_path / "store")
    receipt = store.capture(raw, **_metadata())
    after = datetime.now(UTC)
    assert receipt["retrieved_at"] == "2020-01-02T15:00:00Z"
    ingested = datetime.fromisoformat(receipt["ingested_at"])
    assert before <= ingested <= after
    assert receipt["content_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["content_size_bytes"] == len(raw)
    assert store.read(receipt) == raw
    persisted = json.loads((store.root / receipt["receipt_path"]).read_bytes())
    assert persisted == receipt
    assert store.read(persisted) == raw


def test_deduplicated_objects_have_distinct_immutable_receipts(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "store")
    metadata = _metadata()
    first = store.capture(b"same payload", **metadata)
    object_path = store.root / first["object_path"]
    original_mtime = object_path.stat().st_mtime_ns
    second = store.capture(b"same payload", **metadata)
    assert first["object_path"] == second["object_path"]
    assert first["receipt_id"] != second["receipt_id"]
    assert first["receipt_path"] != second["receipt_path"]
    assert object_path.stat().st_mtime_ns == original_mtime
    metadata["query"]["vintage_dates"].append("2021-01-01")
    metadata["source_definition"]["definition"] = "Changed after capture"
    assert first["query"]["vintage_dates"] == ["2020-01-01"]
    assert store.read(first) == store.read(second) == b"same payload"


def test_concurrent_identical_captures_share_only_the_object(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "store")
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _: store.capture(b"shared", **_metadata()), range(8)))
    assert len({item["object_path"] for item in receipts}) == 1
    assert len({item["receipt_path"] for item in receipts}) == 8
    assert all(store.read(item) == b"shared" for item in receipts)
    assert not list(store.root.rglob("*.tmp"))


def test_imported_cache_does_not_invent_a_retrieval_time(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata.update(retrieved_at=None, retrieval_time_status="unknown_legacy_cache")
    store = SnapshotStore(tmp_path / "store")
    receipt = store.capture(b"old cache", **metadata)
    assert receipt["retrieved_at"] is None
    assert receipt["ingested_at"].endswith("Z")
    assert store.read(receipt) == b"old cache"


@pytest.mark.parametrize(
    "changes",
    [
        {"retrieved_at": datetime(2020, 1, 1)},  # noqa: DTZ001 - reject a deliberately naive clock
        {"retrieved_at": datetime.now(UTC) + timedelta(days=1)},
        {"retrieved_at": None},
        {"retrieval_time_status": "unknown_legacy_cache"},
        {"retrieval_time_status": "inferred_from_file_mtime"},
        {"source_definition": {"definition": "No version"}},
    ],
)
def test_invalid_capture_metadata_fails_before_storage(tmp_path: Path, changes: dict) -> None:
    metadata = _metadata() | changes
    store = SnapshotStore(tmp_path / "store")
    with pytest.raises(ValueError):
        store.capture(b"data", **metadata)
    assert not store.root.exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"source_url": "https://example.com/data?api_key=PRIVATE_VALUE"},
        {"source_url": "https://example.com/data?API%255fKEY=PRIVATE_VALUE"},
        {"source_url": "https://user:PRIVATE_VALUE@example.com/data"},
        {"source_url": "https://example.com/data#PRIVATE_VALUE"},
        {"query": {"nested": [{"Authorization": "PRIVATE_VALUE"}]}},
        {"query": {"request": "https://example.com/data?token=PRIVATE_VALUE"}},
        {"query": {"headers": ["Bearer PRIVATE_VALUE"]}},
        {"query": {"headers": ["Basic PRIVATE_VALUE"]}},
        {"query": {"headers": ["Cookie: PRIVATE_VALUE"]}},
        {"query": {"api-key": "PRIVATE_VALUE"}},
        {"query": {"private_key": "PRIVATE_VALUE"}},
        {"source_url": "https://example.com/data?sig=PRIVATE_VALUE"},
        {"source_definition": {"version": "v1", "client_secret": "PRIVATE_VALUE"}},
    ],
)
def test_credentials_are_rejected_without_echo_or_partial_files(
    tmp_path: Path, changes: dict
) -> None:
    store = SnapshotStore(tmp_path / "store")
    with pytest.raises(ValueError) as caught:
        store.capture(b"data", **(_metadata() | changes))
    assert "PRIVATE_VALUE" not in str(caught.value)
    assert not store.root.exists()


def test_corrupt_object_is_rejected_on_read_and_recapture(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "store")
    receipt = store.capture(b"original", **_metadata())
    path = store.root / receipt["object_path"]
    path.write_bytes(b"corrupted")
    with pytest.raises(SnapshotIntegrityError, match="hash"):
        store.read(receipt)
    with pytest.raises(SnapshotIntegrityError, match="collision or corruption"):
        store.capture(b"original", **_metadata())
    assert path.read_bytes() == b"corrupted"
    assert len(list((store.root / "receipts").glob("*.json"))) == 1


def test_receipt_metadata_and_persisted_receipt_are_verified(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "store")
    receipt = store.capture(b"original", **_metadata())
    modified = deepcopy(receipt)
    modified["query"]["output_type"] = 4
    with pytest.raises(SnapshotIntegrityError, match="metadata hash"):
        store.read(modified)
    (store.root / receipt["receipt_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(SnapshotIntegrityError, match="persisted receipt"):
        store.read(receipt)


def test_existing_receipt_cannot_be_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SnapshotStore(tmp_path / "store")
    receipt = store.capture(b"original", **_metadata())
    persisted = (store.root / receipt["receipt_path"]).read_bytes()
    monkeypatch.setattr(snapshots, "_utc", lambda _: receipt["ingested_at"])

    class FixedIdentifier:
        hex = receipt["receipt_id"]

    monkeypatch.setattr(snapshots, "uuid4", FixedIdentifier)
    metadata = _metadata()
    # Match both serialized clocks so a reused identifier collides exactly.
    original_clock = receipt["retrieved_at"]
    monkeypatch.setattr(
        snapshots,
        "_utc",
        lambda value: original_clock if value.year == 2020 else receipt["ingested_at"],
    )
    with pytest.raises(SnapshotIntegrityError, match="collision or corruption"):
        store.capture(b"original", **metadata)
    assert (store.root / receipt["receipt_path"]).read_bytes() == persisted


def test_tampered_paths_and_symlink_escape_are_rejected(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "store")
    receipt = store.capture(b"original", **_metadata())
    tampered = dict(receipt, object_path="../outside.bin")
    with pytest.raises(SnapshotIntegrityError, match="paths"):
        store.read(tampered)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_store = tmp_path / "linked_store"
    linked_store.mkdir()
    try:
        (linked_store / "objects").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable for this Windows account")
    with pytest.raises(SnapshotIntegrityError, match="escapes|symbolic"):
        SnapshotStore(linked_store).capture(b"blocked", **_metadata())
    assert list(outside.iterdir()) == []
