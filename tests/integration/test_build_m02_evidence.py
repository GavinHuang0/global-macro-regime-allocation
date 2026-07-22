"""Exercise the Model 02 evidence build with fully synthetic providers.

The test creates a hash-pinned Model 01 event artifact, replaces all network
acquisition with deterministic FRED-shaped data, and verifies reuse lineage,
new-series scope, same-vintage retail derivation, housing controls,
asynchronous inflation events, causal warmups, and credential hygiene.
"""

from __future__ import annotations

from datetime import date, timedelta
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from regime_allocation.cli import build_m02_evidence
from regime_allocation.data.dataset_acquisition import MatrixAcquisition, sha256
from regime_allocation.data.providers import ProviderSelection
from regime_allocation.data.providers.vintage_matrix import (
    DownloadedFirstReleaseObservations,
    FirstReleaseObservation,
    encode_vintage_matrix,
)
from regime_allocation.features.release_evidence import (
    EVENT_TABLE_COLUMNS,
    build_claims_release_events,
    build_monthly_release_events,
    validate_event_table,
)


def _monthly_records(
    series_id: str,
    months: pd.DatetimeIndex,
    *,
    block: str,
    transform: str,
    release_offset: int,
) -> pd.DataFrame:
    position = np.arange(len(months), dtype=float)
    if series_id == "RSAFS":
        current = 240.0 + 2.4 * position
        previous = current - 2.4
    elif series_id == "RSFSXMV":
        current = 180.0 + 1.5 * position
        previous = current - 1.5
    elif series_id.startswith("JTS"):
        current = 4.0 + 0.15 * np.sin(position / 3.0) + position * 0.002
        previous = 4.0 + 0.15 * np.sin((position - 1.0) / 3.0) + (position - 1) * 0.002
    else:
        current = 100.0 * np.exp(0.003 * position + 0.01 * np.sin(position / 4.0))
        previous = 100.0 * np.exp(
            0.003 * (position - 1.0) + 0.01 * np.sin((position - 1.0) / 4.0)
        )
    release_dates = [
        (month.to_period("M") + 1).to_timestamp() + pd.Timedelta(days=release_offset)
        for month in months
    ]
    if transform == "difference":
        transformed = current - previous
    else:
        transformed = 100.0 * np.log(current / previous)
    return pd.DataFrame(
        {
            "reference_month": months,
            "component": block,
            "series_id": series_id,
            "release_date": release_dates,
            "current_value": current,
            "previous_value_as_of_release": previous,
            "transform": transform,
            "transformed_value": transformed,
            "release_lag_days": [
                int((release - month.to_period("M").end_time.normalize()).days)
                for month, release in zip(months, release_dates, strict=True)
            ],
        }
    )


def _write_m01_inputs(root: Path, months: pd.DatetimeIndex) -> tuple[Path, Path]:
    frames: list[pd.DataFrame] = []
    monthly_contract = {
        "JTSJOR": ("jolts", "job_openings_rate_change", "difference", 8),
        "JTSHIR": ("jolts", "hires_rate_change", "difference", 8),
        "JTSQUR": ("jolts", "quits_rate_change", "difference", 8),
        "JTSLDR": ("jolts", "layoffs_discharges_rate_change", "difference", 8),
        "RSAFS": ("retail_sales", "retail_sales_log_change", "log_difference", 14),
        "RSFSXMV": (
            "retail_sales",
            "retail_sales_ex_motor_vehicles_log_change",
            "log_difference",
            14,
        ),
        "HOUST": ("housing", "housing_starts_log_change", "log_difference", 17),
        "PERMIT": ("housing", "building_permits_log_change", "log_difference", 17),
        "NEWORDER": (
            "durable_goods",
            "core_capital_goods_orders_log_change",
            "log_difference",
            25,
        ),
    }
    for series_id, (block, feature, transform, offset) in monthly_contract.items():
        raw = _monthly_records(
            series_id,
            months,
            block=block,
            transform=transform,
            release_offset=offset,
        )
        frames.append(
            build_monthly_release_events(
                raw,
                release_block=block,
                feature_name=feature,
                frequency="monthly",
                provider_id="alfred_web",
                source_url=f"https://fred.stlouisfed.org/series/{series_id}",
                min_standardization_history=3,
            )
        )
    weekly_references = pd.date_range("2017-01-07", "2024-12-28", freq="7D")
    claims = tuple(
        FirstReleaseObservation(
            reference_date=timestamp.date(),
            release_date=(timestamp + pd.Timedelta(days=5)).date(),
            value=250_000.0
            * math.exp(0.04 * math.sin(position / 5.0) + position * 0.0002),
        )
        for position, timestamp in enumerate(weekly_references)
    )
    frames.append(
        build_claims_release_events(
            claims,
            release_block="weekly_claims",
            feature_name="initial_claims_innovation",
            series_id="ICSA",
            provider_id="alfred_web",
            source_url="https://fred.stlouisfed.org/series/ICSA",
            min_ar_history=5,
            min_standardization_history=3,
        )
    )
    events = pd.concat(frames, ignore_index=True).sort_values(
        ["release_date", "event_group_id", "reference_date", "feature_name"]
    ).reset_index(drop=True)
    validate_event_table(events)
    event_path = (
        root
        / "data"
        / "processed"
        / "m01_non_defining_release_evidence"
        / "non_defining_release_events.csv"
    )
    event_path.parent.mkdir(parents=True)
    events.to_csv(event_path, index=False)
    manifest_path = root / "data" / "manifests" / "m01_evidence.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "m01_deterministic_composite",
                "provider_selected": "alfred_web",
                "processed_files": [
                    {
                        "path": event_path.relative_to(root).as_posix(),
                        "rows": len(events),
                        "sha256": sha256(event_path.read_bytes()),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path, event_path


def _new_matrix(series_id: str, months: pd.DatetimeIndex) -> bytes:
    offsets = {"ANXAVS": 25, "EXPINF1YR": 5, "WPSID61": 10}
    position = np.arange(len(months), dtype=float)
    if series_id == "EXPINF1YR":
        levels = 2.0 + 0.15 * np.sin(position / 4.0) + 0.002 * position
    else:
        levels = 100.0 * np.exp(0.003 * position + 0.01 * np.sin(position / 5.0))
    matrix = pd.DataFrame(index=months)
    for index, month in enumerate(months):
        if series_id == "EXPINF1YR":
            release = month + pd.Timedelta(days=offsets[series_id])
        else:
            release = (
                (month.to_period("M") + 1).to_timestamp()
                + pd.Timedelta(days=offsets[series_id])
            )
        values = np.full(len(months), np.nan)
        values[: index + 1] = levels[: index + 1]
        matrix[f"{series_id}_{release:%Y%m%d}"] = values
    matrix.index.name = "reference_month"
    return encode_vintage_matrix(matrix, series_id)


class _FakeFred:
    provider_id = "fred_api"
    provider_description = "synthetic authenticated FRED provider"
    cache_namespace = "fred_api"

    @staticmethod
    def series_page_url(series_id: str) -> str:
        return f"https://fred.stlouisfed.org/series/{series_id}"

    def list_first_release_observations(
        self, series_id: str, **_: object
    ) -> DownloadedFirstReleaseObservations:
        assert series_id == "MORTGAGE30US"
        weeks = pd.date_range("2017-12-07", "2024-12-26", freq="7D")
        observations = tuple(
            FirstReleaseObservation(
                reference_date=week.date(),
                release_date=(week + pd.Timedelta(days=1)).date(),
                value=4.0 + 0.005 * position + 0.1 * math.sin(position / 8.0),
            )
            for position, week in enumerate(weeks)
        )
        return DownloadedFirstReleaseObservations(
            series_id=series_id,
            observations=observations,
            source_url=self.series_page_url(series_id),
            provider_id=self.provider_id,
        )


def _test_config(
    manifest_path: Path,
    event_path: Path,
    root: Path,
) -> dict[str, object]:
    config = yaml.safe_load(
        Path("configs/models/m02_release_evidence.yaml").read_text(encoding="utf-8")
    )
    config["inputs"] = {
        "m01_evidence_manifest": manifest_path.relative_to(root).as_posix(),
        "m01_events": event_path.relative_to(root).as_posix(),
    }
    config["data"].update(
        {
            "feature_start": "2018-01-01",
            "observation_start": "2017-01-01",
            "observation_end": "2024-12-31",
            "vintage_start": "2017-01-01",
            "vintage_end": "2025-02-28",
        }
    )
    config["features"].update(
        {
            "default_monthly_min_standardization_history": 3,
            "short_history_min_standardization_history": 3,
            "mortgage_control_min_standardization_history": 3,
        }
    )
    config["blocks"]["consumer_demand"]["derived_sources"][0][
        "min_standardization_history"
    ] = 3
    config["blocks"]["housing_activity"]["controls"][0].update(
        {"vintage_start": "2017-01-01", "min_standardization_history": 3}
    )
    for block in ("business_investment", "inflation_pressure"):
        for source in config["blocks"][block].get("new_sources", []):
            source["vintage_start"] = "2017-01-01"
            source["min_standardization_history"] = 3
    config["outputs"] = {
        "raw_dir": "data/raw/m02_test_evidence",
        "processed_dir": "data/processed/m02_test_evidence",
        "events_file": "events.csv",
        "features_file": "features.csv",
        "new_first_release_observations_file": "observations.csv",
        "manifest": "data/manifests/m02_test_evidence.json",
    }
    return config


def test_build_m02_evidence_reuses_lineage_and_acquires_only_new_series(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    months = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
    m01_manifest, m01_events = _write_m01_inputs(tmp_path, months)
    config = _test_config(m01_manifest, m01_events, tmp_path)
    config_path = tmp_path / "configs" / "m02_evidence.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    fake = _FakeFred()
    monkeypatch.setattr(
        build_m02_evidence,
        "select_vintage_provider",
        lambda *_args, **_kwargs: ProviderSelection("fred", "fred_api", fake),
    )
    downloaded: list[str] = []

    def fake_download(
        *, series_id: str, raw_dir: Path, **_: object
    ) -> MatrixAcquisition:
        downloaded.append(series_id)
        payload = _new_matrix(series_id, months)
        path = raw_dir / f"{series_id}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return MatrixAcquisition(
            path=path,
            content=payload,
            provider_id="fred_api",
            cache_origin="downloaded",
            source_url=fake.series_page_url(series_id),
        )

    monkeypatch.setattr(
        build_m02_evidence,
        "download_or_load_vintage_matrix",
        fake_download,
    )
    secret = "s" * 32
    outputs = build_m02_evidence.build_evidence_dataset(
        project_root=tmp_path,
        config_path=config_path,
        provider="fred",
        environ={"FRED_API_KEY": secret},
    )

    assert set(outputs) == {"manifest", "events", "features", "new_observations"}
    assert sorted(downloaded) == ["ANXAVS", "EXPINF1YR", "WPSID61"]
    events = pd.read_csv(
        outputs["events"],
        parse_dates=["release_date", "reference_date", "reference_month"],
    )
    assert tuple(events.columns) == EVENT_TABLE_COLUMNS
    assert set(events["release_block"]) == build_m02_evidence._EXPECTED_BLOCKS
    assert "CCSA" not in set(events["series_id"])
    assert set(events.loc[events["release_block"] == "consumer_demand", "series_id"]) == {
        "RSFSXMV",
        "RSAFS_MINUS_RSFSXMV",
    }
    assert "RSAFS" not in set(events["series_id"])

    motor = events.loc[events["series_id"] == "RSAFS_MINUS_RSFSXMV"].iloc[0]
    expected_current = (240.0 - 180.0)
    expected_previous = ((240.0 - 2.4) - (180.0 - 1.5))
    assert motor["current_value"] == pytest.approx(expected_current)
    assert motor["transformed_value"] == pytest.approx(
        100.0 * math.log(expected_current / expected_previous)
    )
    assert motor["transform"] == "same_vintage_level_difference_then_log_difference"

    housing_features = set(
        events.loc[events["release_block"] == "housing_activity", "feature_name"]
    )
    assert {
        "housing_starts_log_change",
        "building_permits_log_change",
        "mortgage_rate_monthly_average_change_control",
        "mortgage_rate_post_2022_11_17_methodology",
    }.issubset(housing_features)
    dummy = events.loc[
        events["feature_name"] == "mortgage_rate_post_2022_11_17_methodology"
    ]
    assert set(dummy["feature_value"]) == {0.0, 1.0}
    assert dummy.loc[dummy["reference_month"] < "2022-11-01", "feature_value"].eq(0).all()
    assert dummy.loc[dummy["reference_month"] > "2022-11-01", "feature_value"].eq(1).all()

    inflation = events.loc[events["release_block"] == "inflation_pressure"]
    by_month = inflation.groupby("reference_month")["release_date"].nunique()
    assert (by_month >= 2).any()
    assert inflation.groupby("reference_month")["event_group_id"].nunique().max() >= 2
    expectations = inflation.loc[inflation["series_id"] == "EXPINF1YR"]
    assert len(expectations) == len(months) - 1
    assert (expectations["release_lag_days"] < 0).all()
    assert (
        expectations["release_date"]
        <= expectations["reference_month"].dt.to_period("M").dt.end_time
    ).all()
    assert expectations["feature_status"].eq("available").sum() == len(months) - 4

    observations = pd.read_csv(outputs["new_observations"])
    assert set(observations["series_id"]) == {
        "MORTGAGE30US",
        "ANXAVS",
        "EXPINF1YR",
        "WPSID61",
    }
    expectations_observations = observations.loc[
        observations["series_id"] == "EXPINF1YR"
    ]
    assert expectations_observations["eligible_for_feature"].all()
    assert (
        expectations_observations["feature_eligibility_status"] == "eligible"
    ).all()
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["mixed_provenance"] is True
    assert manifest["new_series_provider_selected"] == "fred_api"
    assert manifest["new_series_failure_policy"] == "stop_without_fallback"
    assert manifest["m01_provider_selected"] == "alfred_web"
    assert manifest["input_lineage"]["m01_evidence"]["event_sha256"] == sha256(
        m01_events.read_bytes()
    )
    assert {item["series_id"] for item in manifest["raw_acquisitions_new_series_only"]} == {
        "MORTGAGE30US",
        "ANXAVS",
        "EXPINF1YR",
        "WPSID61",
    }
    serialized = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in outputs.values()
    )
    assert secret not in serialized


def test_m02_evidence_rejects_stale_m01_artifact(tmp_path: Path) -> None:
    months = pd.date_range("2018-01-01", periods=8, freq="MS")
    manifest, events = _write_m01_inputs(tmp_path, months)
    config = _test_config(manifest, events, tmp_path)
    config_path = tmp_path / "m02.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    events.write_text(events.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        build_m02_evidence._load_verified_m01_events(tmp_path, config["inputs"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("release_id", 1),
        ("release_timing", "retrospective_after_reference_month_end"),
    ],
)
def test_m02_evidence_configuration_rejects_source_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config = yaml.safe_load(
        Path("configs/models/m02_release_evidence.yaml").read_text(encoding="utf-8")
    )
    config["blocks"]["inflation_pressure"]["new_sources"][0][field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="new-source contract"):
        build_m02_evidence._load_config(path)
