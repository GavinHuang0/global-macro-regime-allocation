"""Acquisition provenance must constrain the M03 information set and missingness."""

import json
import shutil
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd
import pytest

from regime_allocation.data import m03_source_audit as audit
from regime_allocation.data.providers import FredApiConfigurationError
from regime_allocation.data.providers.vintage_matrix import (
    DownloadedVintageMatrix,
    encode_vintage_matrix,
)
from regime_allocation.data.source_registry import load_source_registry
from regime_allocation.data.source_snapshots import SnapshotStore

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = datetime(2020, 3, 10, 12, tzinfo=UTC)


@pytest.fixture
def registry():
    return load_source_registry(ROOT / "configs/data/m03_sources.yaml")


@pytest.fixture
def project(tmp_path):
    target = tmp_path / "configs/data/m03_sources.yaml"
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/data/m03_sources.yaml", target)
    return tmp_path


def _source(registry, series="PAYEMS"):
    return next(item.copy() for item in registry["sources"] if item["series_id"] == series)


def _matrix(series="PAYEMS", *, missing_february=False):
    columns = {f"{series}_20200207": [100.0, 101.0, None]}
    if not missing_february:
        columns[f"{series}_20200306"] = [900.0, 102.0, 103.0]
    return pd.DataFrame(columns, index=pd.to_datetime(["2019-12-01", "2020-01-01", "2020-02-01"]))


def _capture(tmp_path, source, *, retrieved_at=None, matrix=None, query_updates=None):
    series = source["series_id"]
    content = encode_vintage_matrix(_matrix(series) if matrix is None else matrix, series)
    query = {
        "observation_start": "2019-12-01",
        "observation_end": "2020-03-31",
        "vintage_start": "2020-01-01",
        "vintage_end": "2020-03-09",
    }
    query.update(query_updates or {})
    store = SnapshotStore(tmp_path / "snapshots")
    receipt = store.capture(
        content,
        provider_id="fred_api",
        series_id=series,
        source_url=source["source_url"],
        query=query,
        retrieved_at=retrieved_at,
        retrieval_time_status="observed" if retrieved_at else "unknown_legacy_cache",
        source_definition=source,
    )
    acquired = {series: {"receipt": receipt, "content_format": "vintage_matrix_zip"}}
    return store, acquired


def _ledgers(registry, source, store, acquired, **kwargs):
    options = {
        "reference_start": date(2020, 1, 1),
        "reference_end": date(2020, 3, 1),
        "as_of": CUTOFF,
        "availability_mode": "archive",
        "defaults": registry["defaults"],
    }
    options.update(kwargs)
    return audit._ledgers([source], acquired, store, **options)


def test_observed_capture_after_cutoff_exposes_no_values(tmp_path, registry):
    source = _source(registry)
    store, acquired = _capture(tmp_path, source, retrieved_at=datetime(2020, 3, 10, 13, tzinfo=UTC))
    features, _, _ = _ledgers(registry, source, store, acquired, availability_mode="captured")
    assert features.status.eq("excluded").all()
    assert features.reason.eq("snapshot_not_retrieved_by_asof").all()
    for name in (
        "current_value",
        "previous_value_as_of_release",
        "transformed_value",
        "archive_vintage_date",
        "archive_available_at",
        "information_available_at",
    ):
        assert features[name].isna().all(), name


@pytest.mark.parametrize(
    ("retrieved_at", "expected"),
    [
        (datetime(2020, 2, 7, 17, tzinfo=UTC), "2020-02-08T05:00:00Z"),
        (datetime(2020, 2, 9, 13, tzinfo=UTC), "2020-02-09T13:00:00Z"),
    ],
)
def test_captured_information_clock_is_later_of_archive_and_retrieval(
    tmp_path, registry, retrieved_at, expected
):
    source = _source(registry)
    store, acquired = _capture(
        tmp_path, source, retrieved_at=retrieved_at, matrix=_matrix(missing_february=True)
    )
    captured, _, _ = _ledgers(registry, source, store, acquired, availability_mode="captured")
    january = captured.set_index("reference_date").loc[pd.Timestamp("2020-01-01")]
    assert january.status == "retained"
    assert january.archive_available_at == pd.Timestamp("2020-02-08T05:00:00Z")
    assert january.information_available_at == pd.Timestamp(expected)
    assert pd.isna(january.publication_timestamp)
    archive, _, _ = _ledgers(registry, source, store, acquired)
    january_archive = archive.set_index("reference_date").loc[pd.Timestamp("2020-01-01")]
    assert january_archive.information_available_at == january_archive.archive_available_at


def test_unqueried_reference_and_expired_archive_query_have_distinct_reasons(tmp_path, registry):
    source = _source(registry)
    query = {"observation_end": "2020-02-29", "vintage_end": "2020-02-15"}
    store, acquired = _capture(
        tmp_path, source, matrix=_matrix(missing_february=True), query_updates=query
    )
    features, _, (boundaries, _) = _ledgers(
        registry, source, store, acquired, reference_start=date(2019, 11, 1)
    )
    rows = features.set_index("reference_date")
    assert rows.loc["2019-11-01", "reason"] == "outside_acquired_observation_range"
    assert rows.loc["2020-03-01", "reason"] == "outside_acquired_observation_range"
    assert rows.loc["2020-02-01", "reason"] == "archive_query_ended_before_asof"
    assert rows.loc["2020-01-01", "status"] == "retained"
    for field, value in acquired["PAYEMS"]["receipt"]["query"].items():
        assert features[f"acquired_{field}"].eq(value).all()
    assert boundaries[0]["acquisition_query"] == acquired["PAYEMS"]["receipt"]["query"]
    assert rows.loc[["2019-11-01", "2020-02-01", "2020-03-01"], "current_value"].isna().all()


def test_current_archive_query_keeps_absent_and_not_yet_visible_distinct(tmp_path, registry):
    source = _source(registry)
    store, acquired = _capture(tmp_path, source, matrix=_matrix(missing_february=True))
    features, _, _ = _ledgers(registry, source, store, acquired)
    rows = features.set_index("reference_date")
    assert rows.loc["2020-02-01", "reason"] == "unavailable_by_asof"
    assert rows.loc["2020-03-01", "reason"] == "absent_reference"


@pytest.mark.parametrize(
    "query_update",
    [
        {"observation_start": "2020-01-01"},
        {"observation_end": "2020-01-31"},
        {"vintage_start": "2020-02-08"},
        {"vintage_end": "2020-02-29"},
    ],
)
def test_payload_outside_declared_query_cannot_supply_a_feature(tmp_path, registry, query_update):
    source = _source(registry)
    store, acquired = _capture(tmp_path, source, query_updates=query_update)
    # Even an out-of-window warmup prior must not silently support an in-window
    # feature, nor may undeclared earlier/later vintages set its first appearance.
    with pytest.raises(ValueError, match="query"):
        _ledgers(registry, source, store, acquired)


def test_duplicate_raw_vintage_header_is_rejected_before_pandas_mangles_it():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "PAYEMS.csv",
            "observation_date,PAYEMS_20200207,PAYEMS_20200207\n"
            "2019-12-01,100,999\n2020-01-01,101,999\n",
        )
    with pytest.raises(ValueError, match="duplicate"):
        audit._load_matrix(buffer.getvalue(), "PAYEMS", "vintage_matrix_zip")


def test_evidence_native_levels_honor_configured_active_support(tmp_path, registry):
    source = _source(registry, "RSAFS")
    source.update(active_start="2020-01-01", active_end="2020-01-01")
    store, acquired = _capture(tmp_path, source)
    _, observations, _ = _ledgers(
        registry, source, store, acquired, reference_start=date(2019, 12, 1)
    )
    rows = observations.set_index("reference_date")
    assert rows.loc["2020-01-01", "status"] == "retained"
    assert rows.loc["2020-01-01", "current_value"] == 101.0
    assert rows.drop(pd.Timestamp("2020-01-01")).reason.eq("outside_active_support").all()
    assert observations.transformed_value.isna().all()


def test_fresh_observed_capture_can_be_replayed_offline_at_later_cutoff(project, monkeypatch):
    retrieved_at = datetime(2020, 3, 10, 13, tzinfo=UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return retrieved_at.astimezone(tz) if tz else retrieved_at.replace(tzinfo=None)

    calls = []

    class Provider:
        def download_level_matrix(self, series, **query):
            calls.append((series, query))
            return DownloadedVintageMatrix(
                series_id=series,
                selected_vintage_dates=(date(2020, 2, 7), date(2020, 3, 6)),
                content=encode_vintage_matrix(_matrix(series), series),
                source_url=f"https://fred.stlouisfed.org/series/{series}",
                provider_id="fred_api",
            )

    monkeypatch.setattr(audit, "datetime", Clock)
    monkeypatch.setattr(
        audit, "select_vintage_provider", lambda *args, **kwargs: SimpleNamespace(client=Provider())
    )
    options = {
        "project_root": project,
        "as_of": CUTOFF,
        "reference_start": date(2020, 1, 1),
        "reference_end": date(2020, 2, 29),
        "series_ids": ("PAYEMS",),
        "availability_mode": "captured",
    }
    original = audit.build_source_audit(output_dir=Path("outputs/fresh"), **options)
    original_manifest = json.loads(original.read_bytes())
    receipt = original_manifest["sources"]["PAYEMS"]["receipt"]
    assert receipt["retrieval_time_status"] == "observed"
    assert pd.Timestamp(receipt["retrieved_at"]) == pd.Timestamp(retrieved_at)
    assert len(calls) == 1
    assert calls[0][1]["vintage_end"] == date(2020, 3, 9)
    assert calls[0][1]["observation_end"] == date(2020, 2, 29)
    assert "chunk_cache_dir" not in calls[0][1]
    blocked = pd.read_csv(original.parent / "feature_ledger.csv")
    assert blocked.reason.eq("snapshot_not_retrieved_by_asof").all()

    def unavailable(*args, **kwargs):
        raise AssertionError("offline replay must not initialize a provider")

    monkeypatch.setattr(audit, "select_vintage_provider", unavailable)
    options["as_of"] = datetime(2020, 3, 10, 14, tzinfo=UTC)
    replay = audit.build_source_audit(
        output_dir=Path("outputs/replay"), replay_manifest=original.relative_to(project), **options
    )
    replay_manifest = json.loads(replay.read_bytes())
    assert replay_manifest["sources"]["PAYEMS"]["receipt"] == receipt
    ledger = pd.read_csv(replay.parent / "feature_ledger.csv")
    assert ledger.status.eq("retained").all()
    assert ledger.current_value.tolist() == [101.0, 103.0]
    assert ledger.previous_value_as_of_release.tolist() == [100.0, 102.0]
    assert pd.to_datetime(ledger.information_available_at, utc=True).eq(retrieved_at).all()


def test_missing_fred_key_fails_before_creating_store_or_output(project):
    with pytest.raises(FredApiConfigurationError, match="FRED_API_KEY"):
        audit.build_source_audit(
            project_root=project,
            output_dir=Path("outputs/missing-key"),
            as_of=CUTOFF,
            reference_start=date(2020, 1, 1),
            reference_end=date(2020, 2, 29),
            series_ids=("PAYEMS",),
            environ={},
        )
    assert not (project / "data").exists()
    assert not (project / "outputs").exists()
