"""Registry coverage, inherited acquisition policy, and definition validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from regime_allocation.data.source_registry import load_source_registry

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs/data/m03_sources.yaml"


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_source_registry_preserves_m02_score_definitions_and_splice_policy() -> None:
    registry = load_source_registry(REGISTRY)
    sources = {source["series_id"]: source for source in registry["sources"]}
    score = yaml.safe_load((ROOT / "configs/models/m02_soft_composite.yaml").read_text())
    expected = set()
    for component, declaration in score["components"].items():
        for configured in declaration["sources"]:
            series = configured["series_id"]
            expected.add(series)
            actual = sources[series]
            assert actual["role"] == "score"
            assert actual["component"] == component
            assert actual["transform"] == declaration["transform"]
            assert actual["release_id"] == configured["release_id"]
            assert actual["axis"] == declaration["axis"]
            assert actual["vintage_start"] == str(
                configured.get("vintage_start", score["data"]["vintage_start"])
            )
            for field in ("vintage_end", "active_start", "active_end"):
                assert actual[field] == (str(configured[field]) if field in configured else None)
    assert len(expected) == 9
    assert {
        source["series_id"] for source in sources.values() if source["role"] == "score"
    } == expected
    assert set(sources) - expected == {"ICSA", "NEWORDER", "ANXAVS", "RSAFS", "RRSFS"}
    assert all(source["verified_archive_start"] is None for source in sources.values())
    assert all(
        source["vintage_start_status"] == "configured_policy_boundary"
        for source in sources.values()
    )


def test_evidence_boundaries_match_fred_live_policy_not_alfred_overrides() -> None:
    sources = {source["series_id"]: source for source in load_source_registry(REGISTRY)["sources"]}
    old = yaml.safe_load(
        (ROOT / "configs/models/m01_non_defining_release_evidence.yaml").read_text()
    )
    retail = yaml.safe_load((ROOT / "configs/models/m02_inference_sensitivities.yaml").read_text())[
        "retail_sensitivities"
    ]["real_decomposition"]
    for series in ("ICSA", "NEWORDER"):
        assert sources[series]["vintage_start"] == str(old["data"]["vintage_start"])
        assert sources[series]["observation_start"] == str(old["data"]["observation_start"])
    for series in ("RSAFS", "RRSFS"):
        assert sources[series]["vintage_start"] == str(retail["vintage_start"])
        assert sources[series]["observation_start"] == str(retail["observation_start"])
        assert sources[series]["release_id"] == retail["release_id"]
    assert sources["ANXAVS"]["vintage_start"] == "2011-05-01"
    assert sources["ICSA"]["frequency"] == "weekly"


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "unknown"),
        ("frequency", "daily"),
        ("release_id", True),
        ("release_id", 0),
        ("transform", "future_revision_backfill"),
        ("vintage_start", "2000-02-30"),
        ("vintage_start", "2000-01-01T08:30:00Z"),
        ("vintage_end", "1980-01-01"),
        ("active_start", "2010-01-15"),
        ("vintage_start_status", "verified_archive_boundary"),
        ("verified_archive_start", "1990-01-01"),
        ("release_timestamp_status", "assumed_0830"),
        ("source_definition_version", "different_version"),
        ("source_url", "https://example.com?api_key=PRIVATE_VALUE"),
    ],
)
def test_invalid_source_metadata_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    registry = deepcopy(load_source_registry(REGISTRY))
    registry["sources"][0][field] = value
    with pytest.raises(ValueError):
        load_source_registry(_write(tmp_path, registry))


def test_duplicate_series_missing_fields_and_unsupported_schema_are_rejected(
    tmp_path: Path,
) -> None:
    for mutate in (
        lambda data: data["sources"].append(deepcopy(data["sources"][0])),
        lambda data: data["sources"][0].pop("definition"),
        lambda data: data.update(schema_version=2),
        lambda data: data.update(schema_version=True),
        lambda data: data["defaults"].update(max_release_lag_days=-1),
    ):
        registry = load_source_registry(REGISTRY)
        mutate(registry)
        with pytest.raises(ValueError):
            load_source_registry(_write(tmp_path, registry))


def test_duplicate_yaml_keys_fail_instead_of_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique string keys"):
        load_source_registry(path)
