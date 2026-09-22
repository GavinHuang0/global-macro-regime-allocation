"""End-to-end source audits use preserved bytes, not mutable caches or future data."""

import hashlib
import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from regime_allocation.data.m03_source_audit import build_source_audit
from regime_allocation.data.providers.vintage_matrix import encode_vintage_matrix
from regime_allocation.data.source_audits import compare_feature_ledgers, ppi_splice_audit

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = datetime(2020, 3, 10, tzinfo=UTC)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "configs/data").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "configs/data/m03_sources.yaml", tmp_path / "configs/data/m03_sources.yaml"
    )
    matrix = pd.DataFrame(
        {
            "PAYEMS_20200207": [100.0, 101.0, None],
            "PAYEMS_20200306": [900.0, 102.0, 103.0],
            "PAYEMS_20200501": [9999.0, 9999.0, 9999.0],
        },
        index=pd.to_datetime(["2019-12-01", "2020-01-01", "2020-02-01"]),
    )
    content = encode_vintage_matrix(matrix, "PAYEMS")
    (tmp_path / "source.zip").write_bytes(content)
    manifest = {
        "acquired_files": [
            {
                "series_id": "PAYEMS",
                "purpose": "defining",
                "provider": "fred_api",
                "path": "source.zip",
                "sha256": hashlib.sha256(content).hexdigest(),
                "source_url": "https://fred.stlouisfed.org/series/PAYEMS",
                "observation_start": "1989-01-01",
                "observation_end": "2020-02-01",
                "vintage_start": "1990-01-01",
                "vintage_end": "2020-05-01",
            }
        ]
    }
    (tmp_path / "import.json").write_text(json.dumps(manifest))
    return tmp_path


def build(project, **kwargs):
    args = {
        "project_root": project,
        "output_dir": Path("outputs/audit"),
        "as_of": CUTOFF,
        "reference_start": date(2020, 1, 1),
        "reference_end": date(2020, 3, 1),
        "series_ids": ("PAYEMS",),
        "import_m02_manifest": Path("import.json"),
    }
    args.update(kwargs)
    return build_source_audit(**args)


def test_import_replay_is_independent_of_mutable_original(project):
    manifest = build(project)
    original = pd.read_csv(manifest.parent / "feature_ledger.csv")
    assert original.status.tolist() == ["retained", "retained", "excluded"]
    assert original.current_value.iloc[:2].tolist() == [101.0, 103.0]
    assert original.previous_value_as_of_release.iloc[:2].tolist() == [100.0, 102.0]
    assert original.retrieved_at.isna().all()
    (project / "source.zip").write_bytes(b"changed mutable source")
    replay = build(
        project,
        output_dir=Path("outputs/replay"),
        import_m02_manifest=None,
        replay_manifest=manifest.relative_to(project),
    )
    assert (manifest.parent / "feature_ledger.csv").read_bytes() == (
        replay.parent / "feature_ledger.csv"
    ).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        build(project)


def test_unknown_retrieval_cannot_enter_captured_information_set(project):
    manifest = build(project, availability_mode="captured")
    ledger = pd.read_csv(manifest.parent / "feature_ledger.csv")
    assert ledger.status.eq("excluded").all()
    assert ledger.reason.eq("snapshot_retrieval_time_unknown").all()
    assert ledger.current_value.isna().all()
    assert ledger.archive_vintage_date.isna().all()


def test_source_corruption_and_output_paths_fail_closed(project):
    (project / "source.zip").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        build(project)
    assert not (project / "outputs/audit").exists()
    for output in ("README.md", "results/published/m03", "../outside", "outputs"):
        with pytest.raises(ValueError):
            build(project, output_dir=Path(output))


def test_same_cutoff_comparison_recomputes_old_snapshot(project):
    old = build(project, as_of=datetime(2020, 2, 10, tzinfo=UTC))
    new = build(project, output_dir=Path("outputs/new"), previous_manifest=old.relative_to(project))
    report = json.loads((new.parent / "source_audit.json").read_bytes())
    diff = report["previous_snapshot_comparison"]
    assert diff["common_retained_rows"] == 2
    assert diff["added_coverage"] == []
    assert diff["changed_values_on_common_support"] == []


def test_invalid_first_value_is_not_coerced_into_later_release(project):
    matrix = pd.DataFrame(
        {"PAYEMS_20200207": [100.0, "invalid"], "PAYEMS_20200306": [100.0, 101.0]},
        index=pd.to_datetime(["2019-12-01", "2020-01-01"]),
    )
    content = encode_vintage_matrix(matrix, "PAYEMS")
    (project / "source.zip").write_bytes(content)
    payload = json.loads((project / "import.json").read_bytes())
    payload["acquired_files"][0]["sha256"] = hashlib.sha256(content).hexdigest()
    (project / "import.json").write_text(json.dumps(payload))
    manifest = build(project)
    ledger = pd.read_csv(manifest.parent / "feature_ledger.csv")
    assert ledger.reason.iloc[0] == "invalid_current_value"


def test_splice_audit_uses_common_reference_support():
    frame = pd.DataFrame(
        {
            "series_id": ["PPILFE", "PPILFE", "WPSFD4131", "WPSFD4131"],
            "reference_month": pd.to_datetime(
                ["2015-04-01", "2015-05-01", "2015-05-01", "2015-06-01"]
            ),
            "status": ["retained"] * 4,
            "transformed_value": [1.0, 2.0, 2.5, 3.0],
            "archive_available_at": pd.to_datetime(
                ["2015-05-01", "2015-06-01", "2015-06-03", "2015-07-01"], utc=True
            ),
        }
    )
    audit = ppi_splice_audit(frame)
    assert audit["common_months"] == 1
    assert audit["old_only_months"] == audit["new_only_months"] == 1
    assert audit["mean_difference"] == 0.5
    assert audit["pairs"][0]["jointly_available_at"].startswith("2015-06-03")


def test_coverage_changes_are_distinct_from_value_changes():
    old = pd.DataFrame(
        {
            "series_id": ["A", "A"],
            "reference_month": ["2020-01-01", "2020-02-01"],
            "status": ["retained", "retained"],
            "transformed_value": [1.0, 2.0],
            "archive_available_at": ["2020-02-01Z", "2020-03-01Z"],
        }
    )
    old["archive_available_at"] = ["2020-02-01T00:00:00Z", "2020-03-01T00:00:00Z"]
    new = old.copy()
    new.loc[0, "status"] = "excluded"
    new.loc[1, "transformed_value"] = 2.5
    report = compare_feature_ledgers(old, new)
    assert report["common_retained_rows"] == 1
    assert len(report["lost_coverage"]) == len(report["changed_values_on_common_support"]) == 1
