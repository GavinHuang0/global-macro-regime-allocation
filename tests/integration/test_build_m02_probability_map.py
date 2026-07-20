"""Exercise the complete Model 02 probability-map publication stage.

Synthetic score artifacts and exact-vintage matrices pass through revision
measurement, causal covariance estimation, bivariate Gaussian integration,
manifest hashing, and public serialization without network access.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from regime_allocation.cli import build_m02_probability_map
from regime_allocation.data.dataset_acquisition import MatrixAcquisition
from regime_allocation.data.providers.vintage_matrix import encode_vintage_matrix
from regime_allocation.models.m02_soft_composite.scores import (
    ALL_COMPONENTS,
    build_composite_scores,
)


SERIES_BY_COMPONENT = {
    "payrolls": ("PAYEMS", "log_difference"),
    "industrial_production": ("INDPRO", "log_difference"),
    "consumer_activity": ("PCEC96", "log_difference"),
    "unemployment_rate": ("UNRATE", "negative_difference"),
    "core_cpi": ("CPILFESL", "log_difference"),
    "core_pce": ("PCEPILFE", "log_difference"),
    "producer_prices": ("PPILFE", "log_difference"),
    "average_hourly_earnings": ("AHETPI", "log_difference"),
}


def _write_score_inputs(root: Path) -> tuple[Path, Path, Path, int]:
    history = pd.date_range("1999-01-01", "2002-12-01", freq="MS")
    transformed = pd.DataFrame(index=history)
    long_rows: list[dict[str, object]] = []
    for component_position, component in enumerate(ALL_COMPONENTS, start=1):
        values = (
            np.sin(np.arange(len(history)) / (2.0 + component_position / 4.0))
            + 0.02 * component_position * np.arange(len(history))
        )
        transformed[component] = values
        series_id, transform = SERIES_BY_COMPONENT[component]
        for month, value in zip(history, values):
            release_date = month + pd.offsets.MonthEnd(1) + pd.Timedelta(days=15)
            long_rows.append(
                {
                    "reference_month": month,
                    "component": component,
                    "series_id": series_id,
                    "release_date": release_date,
                    "current_value": 100.0,
                    "previous_value_as_of_release": 99.0,
                    "transform": transform,
                    "transformed_value": float(value),
                    "release_lag_days": 15,
                    "source_url": f"https://fred.stlouisfed.org/series/{series_id}",
                }
            )
    scores = build_composite_scores(transformed, min_history=2, ddof=1)
    publication = scores.loc["2000-01-01":].copy()
    publication["growth_score_available_at"] = (
        publication.index + pd.offsets.MonthEnd(1) + pd.Timedelta(days=15)
    )
    publication["inflation_score_available_at"] = publication[
        "growth_score_available_at"
    ]
    publication["score_available_at"] = publication["growth_score_available_at"]
    publication["data_status"] = "score_available"
    publication.index.name = "reference_month"

    components_path = (
        root / "data/processed/m02_soft_composite/first_release_components_long.csv"
    )
    scores_path = root / "data/processed/m02_soft_composite/composite_scores.csv"
    components_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(long_rows).to_csv(components_path, index=False, date_format="%Y-%m-%d")
    publication.reset_index().to_csv(scores_path, index=False, date_format="%Y-%m-%d")

    score_config = {
        "model_id": "m02_soft_composite",
        "data": {
            "observation_start": "1998-01-01",
            "reference_end": "2002-12-01",
        },
        "features": {
            "min_history_months": 2,
            "standard_deviation_ddof": 1,
        },
    }
    score_config_path = root / "configs/models/m02_soft_composite.yaml"
    score_config_path.parent.mkdir(parents=True, exist_ok=True)
    score_config_path.write_text(
        yaml.safe_dump(score_config, sort_keys=False), encoding="utf-8"
    )
    complete = int(
        publication[["growth_score", "inflation_score"]].notna().all(axis=1).sum()
    )
    score_manifest = {
        "model_id": "m02_soft_composite",
        "stage": "deterministic_composite_score_definition",
        "complete_score_months": complete,
        "configuration": score_config_path.relative_to(root).as_posix(),
        "configuration_sha256": hashlib.sha256(score_config_path.read_bytes()).hexdigest(),
        "generated_file_hashes": [
            {
                "path": components_path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(components_path.read_bytes()).hexdigest(),
            },
            {
                "path": scores_path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
            },
        ],
    }
    score_manifest_path = root / "data/manifests/m02_soft_composite.json"
    score_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    score_manifest_path.write_text(
        json.dumps(score_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return components_path, scores_path, score_manifest_path, complete


def _probability_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_id": "m02_soft_composite",
        "stage_id": "m02_probability_map",
        "inputs": {
            "score_manifest": "data/manifests/m02_soft_composite.json",
            "first_release_components": (
                "data/processed/m02_soft_composite/"
                "first_release_components_long.csv"
            ),
            "score_features": "data/processed/m02_soft_composite/composite_scores.csv",
        },
        "data": {
            "knowledge_cutoff": "2005-01-01",
            "revision_raw_dir": "data/raw/alfred/m02_probability_map",
        },
        "mapping": {
            "distribution": "bivariate_gaussian",
            "perturbation_mean": [0.0, 0.0],
            "revision_horizons_months": [3, 12],
            "baseline_revision_horizon_months": 12,
            "training_cutoff": "strictly_before_score_availability",
            "disagreement": {
                "estimator": (
                    "expanding_mean_delete_one_component_jackknife_variance"
                ),
                "covariance_structure": "diagonal",
                "minimum_complete_months": 2,
            },
            "revision": {
                "estimator": "expanding_centered_sample_covariance",
                "comparison": "later_minus_first_release",
                "standardization": "frozen_first_release_expanding_scale",
                "minimum_complete_months": 2,
            },
            "expected_coverage": [
                {
                    "revision_horizon_months": 3,
                    "complete_revision_errors": 36,
                    "first_revision_error_month": "2000-01-01",
                    "latest_revision_error_month": "2002-12-01",
                    "available_probability_months": 31,
                    "first_probability_month": "2000-06-01",
                    "latest_probability_month": "2002-12-01",
                },
                {
                    "revision_horizon_months": 12,
                    "complete_revision_errors": 36,
                    "first_revision_error_month": "2000-01-01",
                    "latest_revision_error_month": "2002-12-01",
                    "available_probability_months": 22,
                    "first_probability_month": "2001-03-01",
                    "latest_probability_month": "2002-12-01",
                },
            ],
        },
        "outputs": {
            "processed_dir": "data/processed/m02_soft_composite/probability_map",
            "manifest": "data/manifests/m02_probability_map.json",
            "published_dir": "results/published/m02_soft_composite/probability_map",
        },
    }


def test_probability_map_builds_exact_revision_and_public_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_score_inputs(tmp_path)
    config_path = tmp_path / "configs/models/m02_probability_map.yaml"
    config_path.write_text(
        yaml.safe_dump(_probability_config(), sort_keys=False), encoding="utf-8"
    )

    def fake_acquisition(
        *,
        series_id: str,
        vintage_dates: tuple[object, ...],
        raw_dir: Path,
        **_: object,
    ) -> MatrixAcquisition:
        dates = tuple(pd.Timestamp(item) for item in vintage_dates)
        months = pd.date_range("1998-01-01", "2002-12-01", freq="MS")
        matrix = pd.DataFrame(index=months)
        month_position = np.arange(len(months), dtype=float)
        for vintage_position, vintage in enumerate(dates, start=1):
            matrix[f"{series_id}_{vintage:%Y%m%d}"] = (
                100.0
                + month_position
                + 0.0001 * vintage_position * np.square(month_position)
            )
        payload = encode_vintage_matrix(matrix, series_id)
        path = raw_dir / f"{series_id}_synthetic.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return MatrixAcquisition(
            path=path,
            content=payload,
            provider_id="fred_api",
            cache_origin="downloaded",
            source_url=f"https://fred.stlouisfed.org/series/{series_id}",
        )

    monkeypatch.setattr(
        build_m02_probability_map,
        "download_or_load_exact_vintage_matrix",
        fake_acquisition,
    )
    outputs = build_m02_probability_map.build_probability_map(
        project_root=tmp_path,
        config_path=config_path,
        provider="fred",
        environ={"FRED_API_KEY": "a" * 32},
    )

    assert set(outputs) == {
        "manifest",
        "baseline_history",
        "sensitivity_history",
        "latest",
        "summary",
    }
    baseline = pd.read_csv(outputs["baseline_history"])
    available = baseline[baseline["mapping_status"] == "available"]
    probability_columns = [
        column
        for column in baseline
        if column.startswith("probability_") and column != "probability_sum"
    ]
    np.testing.assert_allclose(available[probability_columns].sum(axis=1), 1.0)

    latest = json.loads(outputs["latest"].read_text(encoding="utf-8"))
    assert latest["baseline"]["revision_horizon_months"] == 12
    assert "revision_error_sample_mean" in latest["baseline"]
    assert "centered_revision_error_mean" not in latest["baseline"]
    assert len(latest["baseline"]["quadrants"]) == 4
    manifest_text = outputs["manifest"].read_text(encoding="utf-8")
    assert "a" * 32 not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["providers_used"] == ["fred_api"]
    assert manifest["runtime"]["python"]
    assert set(manifest["runtime"]["packages"]) == {
        "numpy",
        "pandas",
        "PyYAML",
        "scikit-learn",
        "scipy",
    }
    for record in manifest["generated_file_hashes"]:
        path = tmp_path / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
