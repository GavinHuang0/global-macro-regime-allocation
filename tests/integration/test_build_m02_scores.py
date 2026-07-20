"""Exercise the distinct Model 02 score builder without network access.

Synthetic same-vintage archives pass through the production first-release and
publication pipeline.  The test locks the percentage-payroll transform,
unsmoothed score schema, provider provenance, atomic outputs, and absence of
quadrant labels or probabilities.
"""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest
import yaml

from regime_allocation.cli import build_m02_scores
from regime_allocation.data.dataset_acquisition import MatrixAcquisition
from regime_allocation.models.m02_soft_composite.scores import ALL_COMPONENTS


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


def _levels(series_id: str, periods: int) -> list[float]:
    position = SERIES_IDS.index(series_id) + 1
    values = [100_000.0 if series_id == "PAYEMS" else 80.0 + 3.0 * position]
    if series_id == "UNRATE":
        values[0] = 6.0
    for offset in range(1, periods):
        if series_id == "UNRATE":
            values.append(values[-1] + 0.01 * (((offset + position) % 5) - 2))
        else:
            change = 0.05 + 0.01 * position + 0.02 * ((offset % 5) - 2)
            values.append(values[-1] * math.exp(change / 100.0))
    return values


def _vintage_zip(series_id: str, months: pd.DatetimeIndex) -> bytes:
    levels = _levels(series_id, len(months))
    matrix = pd.DataFrame(index=months)
    for position, reference_month in enumerate(months):
        release_date = (
            (reference_month.to_period("M") + 1).to_timestamp()
            + pd.Timedelta(days=14)
        )
        values = [float("nan")] * len(months)
        values[: position + 1] = levels[: position + 1]
        matrix[f"{series_id}_{release_date:%Y%m%d}"] = values
    matrix.index.name = "observation_date"
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{series_id}_levels_by_vintage.csv", matrix.to_csv(na_rep=".")
        )
    return buffer.getvalue()


def _config() -> dict[str, object]:
    components = {
        "payrolls": ("growth", "log_difference", "PAYEMS", 50),
        "industrial_production": ("growth", "log_difference", "INDPRO", 13),
        "consumer_activity": ("growth", "log_difference", "PCEC96", 54),
        "unemployment_rate": ("growth", "negative_difference", "UNRATE", 50),
        "core_cpi": ("inflation", "log_difference", "CPILFESL", 10),
        "core_pce": ("inflation", "log_difference", "PCEPILFE", 54),
        "producer_prices": ("inflation", "log_difference", "PPILFE", 46),
        "average_hourly_earnings": (
            "inflation",
            "log_difference",
            "AHETPI",
            50,
        ),
    }
    configured_components: dict[str, object] = {}
    for component, (axis, transform, series_id, release_id) in components.items():
        sources: list[dict[str, object]] = [
            {"series_id": series_id, "release_id": release_id}
        ]
        if component == "producer_prices":
            sources = [
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
            ]
        configured_components[component] = {
            "axis": axis,
            "transform": transform,
            "sources": sources,
        }
    return {
        "schema_version": 1,
        "model_id": "m02_soft_composite",
        "data": {
            "reference_start": "2000-01-01",
            "reference_end": "2000-12-01",
            "expected_reference_months": 12,
            "observation_start": "1999-01-01",
            "vintage_start": "1999-01-01",
            "vintage_end": "2001-02-01",
            "max_release_lag_days": 92,
            "expected_missing_component_months": {},
        },
        "features": {
            "min_history_months": 3,
            "standard_deviation_ddof": 1,
            "component_weight": 0.25,
            "apply_trailing_smoothing": False,
        },
        "components": configured_components,
        "outputs": {
            "raw_dir": "data/raw/m02_soft_composite",
            "processed_dir": "data/processed/m02_soft_composite",
            "manifest": "data/manifests/m02_soft_composite.json",
            "published_dir": "results/published/m02_soft_composite",
        },
    }


def test_score_availability_carries_a_delayed_prerequisite() -> None:
    months = pd.date_range("2020-01-01", periods=3, freq="MS")
    releases = pd.DataFrame(
        pd.Timestamp("2020-02-15"), index=months, columns=list(ALL_COMPONENTS)
    )
    releases.loc[months[1]:, :] = pd.Timestamp("2020-03-15")
    releases.loc[months[2]:, :] = pd.Timestamp("2020-04-15")
    releases.loc[months[0], "payrolls"] = pd.Timestamp("2020-04-01")

    availability = build_m02_scores._cumulative_score_availability(
        releases, scores_available=pd.Series(True, index=months)
    )

    assert availability.loc[months[0]] == pd.Timestamp("2020-04-01")
    assert availability.loc[months[1]] == pd.Timestamp("2020-04-01")
    assert availability.loc[months[2]] == pd.Timestamp("2020-04-15")


def test_build_scores_end_to_end_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    months = pd.date_range("1999-01-01", "2000-12-01", freq="MS")
    payloads = {series_id: _vintage_zip(series_id, months) for series_id in SERIES_IDS}
    downloaded: list[str] = []

    def fake_download(
        *, series_id: str, raw_dir: Path, **_: object
    ) -> MatrixAcquisition:
        downloaded.append(series_id)
        path = raw_dir / f"{series_id}_synthetic.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[series_id])
        return MatrixAcquisition(
            path=path,
            content=payloads[series_id],
            provider_id="fred_api",
            cache_origin="downloaded",
            source_url=f"https://fred.stlouisfed.org/series/{series_id}",
        )

    monkeypatch.setattr(build_m02_scores, "download_or_load_vintage_matrix", fake_download)
    config_path = tmp_path / "configs" / "m02_soft_composite.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")

    outputs = build_m02_scores.build_scores(
        project_root=tmp_path,
        config_path=config_path,
        provider="fred",
        environ={"FRED_API_KEY": "a" * 32},
    )

    assert set(outputs) == {"manifest", "components", "scores", "history", "latest"}
    assert downloaded == list(SERIES_IDS)
    assert all(path.exists() for path in outputs.values())
    assert not list(tmp_path.rglob("*.tmp"))

    components = pd.read_csv(outputs["components"], parse_dates=["reference_month"])
    payroll = components.loc[components["component"] == "payrolls"].iloc[0]
    expected_payroll_growth = 100.0 * math.log(
        payroll["current_value"] / payroll["previous_value_as_of_release"]
    )
    assert payroll["transformed_value"] == pytest.approx(expected_payroll_growth)
    assert payroll["transform"] == "log_difference"

    scores = pd.read_csv(outputs["scores"], parse_dates=["reference_month"])
    assert not any("smoothed" in column for column in scores)
    assert not any("regime" in column or "probability" in column for column in scores)
    assert scores[["growth_score", "inflation_score"]].notna().all().all()

    latest = json.loads(outputs["latest"].read_text(encoding="utf-8"))
    assert latest["regime_probabilities_computed"] is False
    assert "regime_id" not in latest

    manifest_text = outputs["manifest"].read_text(encoding="utf-8")
    assert "a" * 32 not in manifest_text
    assert "api_key" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["provider_selected"] == "fred_api"
    assert manifest["providers_used"] == ["fred_api"]
    assert manifest["score_definition"]["trailing_smoothing"] is False
    for record in manifest["generated_file_hashes"]:
        path = tmp_path / record["path"]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
