"""Exercise Model 02's complete VAR(1) transition publication stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from regime_allocation.cli.build_m02_transition import build_transition


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(root: Path) -> tuple[Path, Path]:
    months = pd.date_range("2000-01-01", periods=36, freq="MS")
    position = np.arange(len(months), dtype=float)
    growth = np.sin(position / 3.0) + 0.12 * np.cos(position / 1.7)
    inflation = 0.55 * np.cos(position / 4.0) + 0.18 * np.sin(position / 2.3)
    available_at = months + pd.offsets.MonthEnd(1) + pd.Timedelta(days=12)
    scores = pd.DataFrame(
        {
            "reference_month": months,
            "growth_score": growth,
            "inflation_score": inflation,
            "score_available_at": available_at,
        }
    )
    score_path = root / "data/processed/m02_soft_composite/composite_scores.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(score_path, index=False, date_format="%Y-%m-%d")

    score_manifest_path = root / "data/manifests/m02_soft_composite.json"
    score_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    score_manifest = {
        "model_id": "m02_soft_composite",
        "generated_file_hashes": [
            {
                "path": score_path.relative_to(root).as_posix(),
                "sha256": _sha256(score_path),
            }
        ],
    }
    score_manifest_path.write_text(
        json.dumps(score_manifest, indent=2) + "\n", encoding="utf-8"
    )

    mapping = scores.copy()
    mapping["revision_horizon_months"] = 12
    mapping["is_baseline"] = True
    mapping["mapping_status"] = "available"
    mapping["growth_map_variance"] = 0.30
    mapping["growth_inflation_map_covariance"] = 0.04
    mapping["inflation_map_variance"] = 0.20
    mapping_path = (
        root
        / "results/published/m02_soft_composite/probability_map/"
        "quadrant_probabilities.csv"
    )
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_path, index=False, date_format="%Y-%m-%d")

    probability_manifest_path = root / "data/manifests/m02_probability_map.json"
    probability_manifest = {
        "model_id": "m02_soft_composite",
        "stage_id": "m02_probability_map",
        "score_manifest_sha256": _sha256(score_manifest_path),
        "generated_file_hashes": [
            {
                "path": mapping_path.relative_to(root).as_posix(),
                "sha256": _sha256(mapping_path),
            }
        ],
    }
    probability_manifest_path.write_text(
        json.dumps(probability_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return score_path, mapping_path


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_id": "m02_soft_composite",
        "stage_id": "m02_var1_transition",
        "inputs": {
            "score_manifest": "data/manifests/m02_soft_composite.json",
            "score_features": "data/processed/m02_soft_composite/composite_scores.csv",
            "probability_map_manifest": "data/manifests/m02_probability_map.json",
            "baseline_mapping": (
                "results/published/m02_soft_composite/probability_map/"
                "quadrant_probabilities.csv"
            ),
        },
        "transition": {
            "state_order": ["growth_score", "inflation_score"],
            "lag_order": 1,
            "estimator": "expanding_multivariate_ols",
            "include_intercept": True,
            "include_time_trend": False,
            "regularization": "none",
            "minimum_training_pairs": 12,
            "require_consecutive_reference_months": True,
            "include_pair_ending_at_source_month": True,
            "forecast_issue_policy": "source_score_availability_date",
            "operational_start_of_month_prior": False,
            "availability_cutoff": (
                "pair_available_at_on_or_before_prior_available_at"
            ),
            "reference_cutoff": "response_month_on_or_before_source_month",
            "innovation_covariance": "residual_cross_product_over_n_minus_3",
            "parameter_uncertainty": "excluded_from_baseline",
            "source_state": "exact_released_composite_score_center",
            "forecast_mean": "intercept_plus_A_times_exact_source_score",
            "latent_forecast_covariance": "Q",
            "target_mapping_covariance_proxy": (
                "latest_causally_available_baseline_map_at_source_cutoff"
            ),
            "reporting_forecast_covariance": "Q_plus_Omega_map_proxy",
            "mapping_noise_independent_of_var_innovation": True,
            "force_stationarity": False,
            "expected_coverage": {
                "score_history_rows": 36,
                "candidate_adjacent_pairs": 35,
                "eligible_consecutive_pairs_at_latest_cutoff": 35,
                "baseline_mapping_months": 36,
                "available_prior_months": 24,
                "first_source_month": "2001-01-01",
                "first_target_month": "2001-02-01",
                "latest_source_month": "2002-12-01",
                "latest_target_month": "2003-01-01",
                "latest_training_pairs": 35,
                "unstable_prior_fits": 4,
                "priors_available_by_target_start": 0,
                "priors_available_by_target_end": 24,
                "priors_available_after_target_end": 0,
            },
        },
        "outputs": {
            "pair_audit": (
                "data/processed/m02_soft_composite/var1_transition/pair_audit.csv"
            ),
            "published_history": (
                "results/published/m02_soft_composite/transition/"
                "next_month_priors.csv"
            ),
            "published_latest": (
                "results/published/m02_soft_composite/transition/"
                "latest_next_month_prior.json"
            ),
            "published_summary": (
                "results/published/m02_soft_composite/transition/"
                "transition_summary.json"
            ),
            "manifest": "data/manifests/m02_var1_transition.json",
        },
    }


def test_transition_builder_publishes_causal_normalized_priors(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    config_path = tmp_path / "configs/models/m02_var1_transition.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")

    outputs = build_transition(project_root=tmp_path, config_path=config_path)

    assert set(outputs) == {"manifest", "pair_audit", "history", "latest", "summary"}
    history = pd.read_csv(outputs["history"])
    available = history[history["prior_status"] == "available"]
    probabilities = [
        column
        for column in history
        if column.startswith("probability_") and column != "probability_sum"
    ]
    assert len(available) == 24
    np.testing.assert_allclose(available[probabilities].sum(axis=1), 1.0)

    latest = json.loads(outputs["latest"].read_text(encoding="utf-8"))
    assert latest["target_reference_month"] == "2003-01-01"
    assert latest["prior_kind"] == "as_of_exact_source_score_release"
    assert latest["fit"]["training_pairs"] == 35
    assert latest["source_state"]["score_is_exact"] is True
    np.testing.assert_allclose(
        latest["next_month_latent_prior"]["covariance"],
        latest["fit"]["innovation_covariance"],
    )
    latent = np.asarray(latest["next_month_latent_prior"]["covariance"])
    mapping_proxy = np.asarray(
        latest["target_mapping_covariance_proxy"]["covariance"]
    )
    reporting = np.asarray(latest["next_month_prior"]["covariance"])
    np.testing.assert_allclose(reporting, latent + mapping_proxy)
    assert len(latest["next_month_prior"]["quadrants"]) == 4

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["stage_id"] == "m02_var1_transition"
    for record in manifest["generated_file_hashes"]:
        assert _sha256(tmp_path / record["path"]) == record["sha256"]


def test_transition_builder_rejects_stale_probability_map_lineage(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    probability_manifest_path = tmp_path / "data/manifests/m02_probability_map.json"
    probability_manifest = json.loads(
        probability_manifest_path.read_text(encoding="utf-8")
    )
    probability_manifest["score_manifest_sha256"] = "0" * 64
    probability_manifest_path.write_text(
        json.dumps(probability_manifest, indent=2) + "\n", encoding="utf-8"
    )
    config_path = tmp_path / "configs/models/m02_var1_transition.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="lineage"):
        build_transition(project_root=tmp_path, config_path=config_path)


def test_transition_builder_rejects_four_pair_minimum(tmp_path: Path) -> None:
    config = _config()
    config["transition"]["minimum_training_pairs"] = 4
    config_path = tmp_path / "configs/models/m02_var1_transition.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="at least five"):
        build_transition(project_root=tmp_path, config_path=config_path)


def test_transition_builder_rejects_cross_input_score_mismatch(
    tmp_path: Path,
) -> None:
    score_path, _ = _write_inputs(tmp_path)
    scores = pd.read_csv(score_path)
    scores.loc[10, "growth_score"] += 0.25
    scores.to_csv(score_path, index=False, date_format="%Y-%m-%d")

    score_manifest_path = tmp_path / "data/manifests/m02_soft_composite.json"
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    score_manifest["generated_file_hashes"][0]["sha256"] = _sha256(score_path)
    score_manifest_path.write_text(
        json.dumps(score_manifest, indent=2) + "\n", encoding="utf-8"
    )
    probability_manifest_path = tmp_path / "data/manifests/m02_probability_map.json"
    probability_manifest = json.loads(
        probability_manifest_path.read_text(encoding="utf-8")
    )
    probability_manifest["score_manifest_sha256"] = _sha256(score_manifest_path)
    probability_manifest_path.write_text(
        json.dumps(probability_manifest, indent=2) + "\n", encoding="utf-8"
    )

    config_path = tmp_path / "configs/models/m02_var1_transition.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="disagree on growth_score"):
        build_transition(project_root=tmp_path, config_path=config_path)


def test_transition_builder_rejects_missing_revision_horizon(
    tmp_path: Path,
) -> None:
    _, mapping_path = _write_inputs(tmp_path)
    mapping = pd.read_csv(mapping_path)
    mapping.loc[5, "revision_horizon_months"] = np.nan
    mapping.to_csv(mapping_path, index=False, date_format="%Y-%m-%d")

    probability_manifest_path = tmp_path / "data/manifests/m02_probability_map.json"
    probability_manifest = json.loads(
        probability_manifest_path.read_text(encoding="utf-8")
    )
    probability_manifest["generated_file_hashes"][0]["sha256"] = _sha256(
        mapping_path
    )
    probability_manifest_path.write_text(
        json.dumps(probability_manifest, indent=2) + "\n", encoding="utf-8"
    )

    config_path = tmp_path / "configs/models/m02_var1_transition.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="only the 12-month baseline"):
        build_transition(project_root=tmp_path, config_path=config_path)
