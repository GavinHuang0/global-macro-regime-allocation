"""Receipt eligibility and immutable Yahoo replay through the complete M03 wrapper."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from regime_allocation.data.m03_source_audit import build_source_audit
from regime_allocation.data.providers.vintage_matrix import encode_vintage_matrix
from regime_allocation.data.source_registry import load_source_registry
from regime_allocation.data.source_snapshots import SnapshotStore

ROOT = Path(__file__).resolve().parents[3]
AS_OF = datetime(2020, 3, 10, 13, tzinfo=UTC)
OBSERVED = datetime(2020, 3, 9, 22, tzinfo=UTC)
AFTER_AS_OF = datetime(2020, 3, 11, 13, tzinfo=UTC)


def _price_content(closes=(100, 101, 102, 103)) -> bytes:
    timestamps = [int(datetime(2020, 3, day, 14, tzinfo=UTC).timestamp()) for day in (5, 6, 9, 10)]
    return json.dumps(
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {
                            "symbol": "TEST",
                            "currency": "USD",
                            "dataGranularity": "1d",
                            "exchangeTimezoneName": "America/New_York",
                        },
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [{"close": list(closes)}],
                            "adjclose": [{"adjclose": list(closes)}],
                        },
                        "events": {},
                    }
                ],
            }
        }
    ).encode()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Use the real registry and receipt format; no orchestration functions are mocked."""
    registry_path = tmp_path / "configs/data/m03_sources.yaml"
    registry_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/data/m03_sources.yaml", registry_path)
    source = next(
        item
        for item in load_source_registry(registry_path)["sources"]
        if item["series_id"] == "PAYEMS"
    )
    matrix = pd.DataFrame(
        {"PAYEMS_20200207": [100.0, 101.0]},
        index=pd.to_datetime(["2019-12-01", "2020-01-01"]),
    )
    receipt = SnapshotStore(tmp_path / "data/raw/m03_sources").capture(
        encode_vintage_matrix(matrix, "PAYEMS"),
        provider_id="fred_api",
        series_id="PAYEMS",
        source_url="https://fred.stlouisfed.org/series/PAYEMS",
        query={
            "observation_start": "2019-12-01",
            "observation_end": "2020-01-01",
            "vintage_start": "2020-02-07",
            "vintage_end": "2020-03-10",
        },
        retrieved_at=datetime(2020, 3, 1, tzinfo=UTC),
        retrieval_time_status="observed",
        source_definition=source,
    )
    (tmp_path / "macro_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return tmp_path


def _input_manifest(
    project: Path,
    name: str,
    *,
    retrieved_at: datetime | None,
    content: bytes | None = None,
    prior_run_as_of: datetime = AS_OF,
) -> Path:
    content = _price_content() if content is None else content
    mutable = project / f"{name}_TEST.json"
    mutable.write_bytes(content)
    price_receipt = SnapshotStore(project / "data/raw/m03_sources").capture(
        content,
        provider_id="yahoo_chart",
        series_id="TEST",
        source_url="https://query1.finance.yahoo.com/v8/finance/chart/TEST",
        query={"import_file": mutable.name},
        retrieved_at=retrieved_at,
        retrieval_time_status="observed" if retrieved_at is not None else "unknown_legacy_cache",
        source_definition={
            "source_definition_version": "yahoo_chart_m03_v1",
            "definition": "Yahoo daily chart prices and corporate actions",
            "independent_provider_verification": False,
        },
    )
    macro_receipt = json.loads((project / "macro_receipt.json").read_bytes())
    manifest = {
        "as_of": prior_run_as_of.isoformat(),
        "sources": {"PAYEMS": {"receipt": macro_receipt, "content_format": "vintage_matrix_zip"}},
        "prices": {"TEST": price_receipt},
    }
    path = project / f"{name}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path.relative_to(project)


def _build(
    project: Path,
    manifest: Path,
    *,
    name="audit",
    mode="captured",
    previous: Path | None = None,
    as_of: datetime = AS_OF,
) -> tuple[Path, dict]:
    output_manifest = build_source_audit(
        project_root=project,
        output_dir=Path("outputs") / name,
        as_of=as_of,
        reference_start=date(2020, 1, 1),
        reference_end=date(2020, 1, 1),
        series_ids=("PAYEMS",),
        replay_manifest=manifest,
        previous_manifest=previous,
        availability_mode=mode,
        environ={},
    )
    # The macro pipeline also runs normally; these tests do not substitute empty ledgers.
    ledger = pd.read_csv(output_manifest.parent / "feature_ledger.csv")
    assert ledger["status"].tolist() == ["retained"]
    return output_manifest, json.loads((output_manifest.parent / "source_audit.json").read_bytes())


@pytest.mark.parametrize("old_retrieved_at", [None, AFTER_AS_OF], ids=["unknown", "future"])
def test_captured_comparison_blocks_ineligible_previous_receipt(
    project: Path,
    old_retrieved_at: datetime | None,
) -> None:
    old = _input_manifest(project, "old", retrieved_at=old_retrieved_at)
    current = _input_manifest(
        project, "current", retrieved_at=OBSERVED, content=_price_content((100, 104, 102, 103))
    )
    _, report = _build(project, current, previous=old)
    assert report["price_action_audits"][0]["status"] == "passed"
    assert report["price_snapshot_comparisons"] == [
        {
            "ticker": "TEST",
            "status": "previous_snapshot_not_retrieved_by_asof",
        }
    ]
    assert "price_changes" not in report["price_snapshot_comparisons"][0]


@pytest.mark.parametrize("current_retrieved_at", [None, AFTER_AS_OF], ids=["unknown", "future"])
def test_captured_mode_blocks_ineligible_current_receipt_and_comparison(
    project: Path,
    current_retrieved_at: datetime | None,
) -> None:
    old = _input_manifest(project, "old", retrieved_at=OBSERVED)
    current = _input_manifest(project, "current", retrieved_at=current_retrieved_at)
    _, report = _build(project, current, previous=old)
    assert report["price_action_audits"] == [
        {
            "ticker": "TEST",
            "status": "snapshot_not_demonstrably_retrieved_by_asof",
        }
    ]
    assert report.get("price_snapshot_comparisons", []) == []


def test_receipts_retrieved_exactly_at_cutoff_are_eligible(project: Path) -> None:
    old = _input_manifest(project, "old", retrieved_at=AS_OF)
    current = _input_manifest(project, "current", retrieved_at=AS_OF)
    _, report = _build(project, current, previous=old)
    assert report["price_action_audits"][0]["status"] == "passed"
    assert report["price_snapshot_comparisons"][0]["status"] == "unchanged_common_support"


def test_archive_comparison_allows_unknown_and_later_retrieval(project: Path) -> None:
    old = _input_manifest(project, "old", retrieved_at=None)
    current = _input_manifest(
        project, "current", retrieved_at=AFTER_AS_OF, content=_price_content((100, 104, 102, 103))
    )
    _, report = _build(project, current, previous=old, mode="archive")
    assert report["price_action_audits"][0]["status"] == "passed"
    comparison = report["price_snapshot_comparisons"][0]
    assert comparison["status"] == "changed_common_support"
    assert comparison["coverage"]["common_dates"] == 3
    assert {(row["date"], row["field"]) for row in comparison["price_changes"]} == {
        ("2020-03-06", "close"),
        ("2020-03-06", "adjusted_close"),
    }
    assert comparison["independent_source_verified"] is False


def test_price_replay_preserves_receipt_and_ignores_mutable_original(project: Path) -> None:
    original = _input_manifest(project, "current", retrieved_at=OBSERVED)
    first_manifest, first_report = _build(project, original, name="first")
    # A later cache refresh cannot affect an immutable replay.
    (project / "current_TEST.json").write_bytes(b"changed mutable Yahoo response")
    second_manifest, second_report = _build(
        project,
        first_manifest.relative_to(project),
        name="replay",
    )
    first = json.loads(first_manifest.read_bytes())
    second = json.loads(second_manifest.read_bytes())
    assert second["prices"] == first["prices"]
    assert second_report["price_action_audits"] == first_report["price_action_audits"]
    assert first_report["price_action_audits"][0]["as_of"] == "2020-03-09"
    assert first_report["price_action_audits"][0]["summary"]["compared_intervals"] == 2


def test_price_comparison_uses_current_cutoff_for_both_snapshots(project: Path) -> None:
    old = _input_manifest(
        project, "old", retrieved_at=OBSERVED, prior_run_as_of=datetime(2020, 3, 6, 13, tzinfo=UTC)
    )
    current = _input_manifest(
        project, "current", retrieved_at=OBSERVED, content=_price_content((100, 101, 102, -1))
    )
    _, report = _build(project, current, previous=old)
    comparison = report["price_snapshot_comparisons"][0]
    assert comparison["as_of"] == "2020-03-09"
    assert comparison["coverage"]["common_dates"] == 3
    assert comparison["status"] == "unchanged_common_support"
    assert comparison["price_changes"] == []
    assert report["price_action_audits"][0]["status"] == "passed"


def test_price_cutoff_uses_new_york_day_before_utc_midnight_boundary(project: Path) -> None:
    # 02:00 UTC Mar10 is still Mar9 in New York: Mar9 bars must be excluded.
    current = _input_manifest(
        project, "current", retrieved_at=OBSERVED, content=_price_content((100, 101, -1, -1))
    )
    _, report = _build(project, current, as_of=datetime(2020, 3, 10, 2, tzinfo=UTC))
    audit = report["price_action_audits"][0]
    assert audit["status"] == "passed"
    assert audit["as_of"] == "2020-03-08"
    assert audit["coverage"]["last_observation"] == "2020-03-06"
    assert audit["coverage"]["observations_after_cutoff_excluded"] == 2
    assert audit["summary"]["compared_intervals"] == 1
