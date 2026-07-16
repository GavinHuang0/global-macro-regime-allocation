"""Synthetic, no-network integration test for model 01 publication."""

from __future__ import annotations

from datetime import date
from io import BytesIO
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest
import yaml

from regime_allocation.cli import build_m01_dataset
from regime_allocation.features.composites import ALL_COMPONENTS


SERIES_IDS = (
    "PAYEMS",
    "INDPRO",
    "PCEC96",
    "UNRATE",
    "CPILFESL",
    "PCEPILFE",
    "PPILFE",
    "WPSFD4131",
    "AHETPI",
)


def _synthetic_levels(series_id: str, periods: int) -> list[float]:
    """Build positive, non-constant levels suitable for every transform."""
    series_number = SERIES_IDS.index(series_id) + 1
    if series_id == "PAYEMS":
        values = [100_000.0]
        for offset in range(1, periods):
            step = 80.0 + 7.0 * series_number + 9.0 * ((offset + 1) % 5)
            values.append(values[-1] + step)
        return values
    if series_id == "UNRATE":
        values = [7.0]
        for offset in range(1, periods):
            step = 0.015 * (((offset + series_number) % 5) - 2)
            values.append(values[-1] + step)
        return values

    values = [80.0 + 4.0 * series_number]
    for offset in range(1, periods):
        log_change_percent = (
            0.08
            + 0.012 * series_number
            + 0.025 * (((offset + series_number) % 6) - 2.5)
        )
        values.append(values[-1] * math.exp(log_change_percent / 100.0))
    return values


def _vintage_zip(series_id: str, months: pd.DatetimeIndex) -> bytes:
    """Create an observation-by-vintage ZIP with one release per month."""
    levels = _synthetic_levels(series_id, len(months))
    matrix = pd.DataFrame(index=months)
    for position, reference_month in enumerate(months):
        release_month = (reference_month.to_period("M") + 1).to_timestamp()
        release_date = release_month + pd.Timedelta(days=14)
        column = f"{series_id}_{release_date:%Y%m%d}"
        values = [float("nan")] * len(months)
        values[: position + 1] = levels[: position + 1]
        matrix[column] = values
    matrix.index.name = "observation_date"

    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{series_id}_levels_by_vintage.csv",
            matrix.to_csv(na_rep="."),
        )
    return buffer.getvalue()


def _model_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_id": "m01_deterministic_composite",
        "data": {
            "reference_start": "2000-01-01",
            "reference_end": "2000-12-01",
            "expected_reference_months": 12,
            "observation_start": "1999-01-01",
            "vintage_start": "1999-01-01",
            "vintage_end": "2001-02-01",
            "max_release_lag_days": 92,
        },
        "features": {
            "min_history_months": 3,
            "standard_deviation_ddof": 1,
            "smoothing_window_months": 2,
            "component_weight": 0.25,
            "zero_tie_policy": "up",
        },
        "components": {
            "payrolls": {
                "axis": "growth",
                "transform": "difference",
                "sources": [{"series_id": "PAYEMS", "release_id": 50}],
            },
            "industrial_production": {
                "axis": "growth",
                "transform": "log_difference",
                "sources": [{"series_id": "INDPRO", "release_id": 13}],
            },
            "consumer_activity": {
                "axis": "growth",
                "transform": "log_difference",
                "sources": [{"series_id": "PCEC96", "release_id": 54}],
            },
            "unemployment_rate": {
                "axis": "growth",
                "transform": "negative_difference",
                "sources": [{"series_id": "UNRATE", "release_id": 50}],
            },
            "core_cpi": {
                "axis": "inflation",
                "transform": "log_difference",
                "sources": [{"series_id": "CPILFESL", "release_id": 10}],
            },
            "core_pce": {
                "axis": "inflation",
                "transform": "log_difference",
                "sources": [{"series_id": "PCEPILFE", "release_id": 54}],
            },
            "producer_prices": {
                "axis": "inflation",
                "transform": "log_difference",
                "sources": [
                    {
                        "series_id": "PPILFE",
                        "release_id": 46,
                        "active_end": "2000-06-01",
                    },
                    {
                        "series_id": "WPSFD4131",
                        "release_id": 46,
                        "active_start": "2000-07-01",
                    },
                ],
            },
            "average_hourly_earnings": {
                "axis": "inflation",
                "transform": "log_difference",
                "sources": [{"series_id": "AHETPI", "release_id": 50}],
            },
        },
        "outputs": {
            "raw_dir": "data/raw/model_01",
            "processed_dir": "data/processed/model_01",
            "manifest": "data/manifests/model_01.json",
            "published_dir": "results/published/model_01",
        },
    }


def test_build_dataset_end_to_end_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    months = pd.date_range("1999-01-01", "2000-12-01", freq="MS")
    payloads = {
        series_id: _vintage_zip(series_id, months) for series_id in SERIES_IDS
    }
    downloaded: list[str] = []

    def fake_download_or_load(
        *,
        series_id: str,
        raw_dir: Path,
        **_: object,
    ) -> build_m01_dataset._MatrixAcquisition:
        downloaded.append(series_id)
        raw_path = raw_dir / f"{series_id}_synthetic.zip"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payloads[series_id])
        return build_m01_dataset._MatrixAcquisition(
            path=raw_path,
            content=payloads[series_id],
            provider_id="fred_api",
            cache_origin="downloaded",
            source_url=f"https://fred.stlouisfed.org/series/{series_id}",
        )

    monkeypatch.setattr(
        build_m01_dataset,
        "_download_or_load",
        fake_download_or_load,
    )

    config_path = tmp_path / "configs" / "model_01.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(_model_config(), sort_keys=False),
        encoding="utf-8",
    )

    published_dir = tmp_path / "results" / "published" / "model_01"
    published_dir.mkdir(parents=True)
    (published_dir / "regime_history.csv").write_text(
        "stale history that must be atomically replaced\n", encoding="utf-8"
    )
    (published_dir / "latest_confirmed.json").write_text(
        '{"stale": true}\n', encoding="utf-8"
    )

    outputs = build_m01_dataset.build_dataset(
        project_root=tmp_path,
        config_path=config_path,
        provider="fred",
        environ={"FRED_API_KEY": "a" * 32},
    )

    assert set(outputs) == {"manifest", "components", "features", "history", "latest"}
    assert downloaded == list(SERIES_IDS)
    assert all(path.exists() for path in outputs.values())
    assert not list(tmp_path.rglob("*.tmp"))

    components = pd.read_csv(
        outputs["components"],
        parse_dates=["reference_month", "release_date"],
    )
    assert list(components.columns) == [
        "reference_month",
        "component",
        "series_id",
        "release_date",
        "current_value",
        "previous_value_as_of_release",
        "transform",
        "transformed_value",
        "release_lag_days",
        "source_url",
    ]
    assert set(components["component"]) == set(ALL_COMPONENTS)
    producer_sources = components.loc[
        components["component"] == "producer_prices",
        ["reference_month", "series_id"],
    ]
    before_splice = producer_sources["reference_month"] <= pd.Timestamp("2000-06-01")
    after_splice = producer_sources["reference_month"] >= pd.Timestamp("2000-07-01")
    assert set(producer_sources.loc[before_splice, "series_id"]) == {"PPILFE"}
    assert set(producer_sources.loc[after_splice, "series_id"]) == {"WPSFD4131"}

    features = pd.read_csv(outputs["features"], parse_dates=["reference_month"])
    expected_feature_columns = ["reference_month"]
    for component in ALL_COMPONENTS:
        expected_feature_columns.extend(
            [f"{component}_transformed", f"{component}_z"]
        )
    expected_feature_columns.extend(
        [
            "growth_raw",
            "growth_smoothed",
            "inflation_raw",
            "inflation_smoothed",
            "label_available_at",
            "regime_id",
            "regime_label",
        ]
    )
    assert list(features.columns) == expected_feature_columns
    expected_months = pd.date_range("2000-01-01", "2000-12-01", freq="MS")
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(features["reference_month"]),
        expected_months,
        check_names=False,
    )
    transformed_columns = [
        f"{component}_transformed" for component in ALL_COMPONENTS
    ]
    assert not features[transformed_columns].isna().any().any()
    assert not features[["regime_id", "regime_label", "label_available_at"]].isna().any().any()

    history = pd.read_csv(
        outputs["history"],
        parse_dates=["reference_month", "label_available_at"],
    )
    assert list(history.columns) == [
        "reference_month",
        "growth_raw",
        "inflation_raw",
        "growth_smoothed",
        "inflation_smoothed",
        "regime_id",
        "regime_label",
        "label_available_at",
        "data_status",
    ]
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(history["reference_month"]),
        expected_months,
        check_names=False,
    )
    assert set(history["data_status"]) == {"classified"}

    latest = json.loads(outputs["latest"].read_text(encoding="utf-8"))
    assert set(latest) == {
        "model_id",
        "reference_month",
        "label_available_at",
        "growth_score",
        "inflation_score",
        "regime_id",
        "regime_label",
        "status",
    }
    last_history_row = history.iloc[-1]
    assert latest["reference_month"] == last_history_row["reference_month"].date().isoformat()
    assert latest["label_available_at"] == last_history_row["label_available_at"].date().isoformat()
    assert latest["growth_score"] == pytest.approx(last_history_row["growth_smoothed"])
    assert latest["inflation_score"] == pytest.approx(
        last_history_row["inflation_smoothed"]
    )
    assert latest["regime_id"] == last_history_row["regime_id"]
    assert latest["regime_label"] == last_history_row["regime_label"]

    manifest_text = outputs["manifest"].read_text(encoding="utf-8")
    assert "a" * 32 not in manifest_text
    assert "api_key" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["schema_version"] == 2
    assert manifest["provider_policy"] == (
        "fred_api_preferred_when_FRED_API_KEY_is_present"
    )
    assert manifest["provider_requested"] == "fred"
    assert manifest["provider_selected"] == "fred_api"
    assert manifest["providers_used"] == ["fred_api"]
    assert manifest["reference_months"] == 12
    assert manifest["history_months"] == 12
    assert manifest["classified_months"] == 12
    assert manifest["unavailable_months"] == []
    assert manifest["first_classified_month"] == "2000-01-01"
    assert manifest["latest_classified_month"] == "2000-12-01"
    assert len(manifest["raw_files"]) == len(SERIES_IDS)
    assert {item["provider"] for item in manifest["raw_files"]} == {"fred_api"}
    assert {item["cache_origin"] for item in manifest["raw_files"]} == {
        "downloaded"
    }
