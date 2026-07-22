"""Build Model 02's causal expanding VAR(1) next-month priors.

The stage fits a two-score VAR with an intercept at each historical source
cutoff. It forecasts the next exact score center with covariance ``Q``, adds the
latest causally available mapping covariance as a target-reporting proxy, and
integrates that reporting distribution over the four growth/inflation
quadrants. It does not process daily release evidence or construct a
four-month joint path posterior.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from regime_allocation.data.dataset_acquisition import (
    sha256,
    write_csv,
    write_json,
)
from regime_allocation.models.m02_soft_composite.probability_map import (
    REGIME_LABELS,
    REGIME_ORDER,
)
from regime_allocation.models.m02_soft_composite.var_transition import (
    build_var_pair_audit,
    causal_var1_prior_history,
    select_causal_var_pairs,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "m02_var1_transition"


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load and freeze the transition-stage architecture contract."""

    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("VAR transition configuration must be a mapping")
    if config.get("schema_version") != 1:
        raise ValueError("VAR transition configuration schema_version must be 1")
    if str(config.get("model_id")) != MODEL_ID:
        raise ValueError(f"model_id must be {MODEL_ID!r}")
    if str(config.get("stage_id")) != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID!r}")
    for section in ("inputs", "transition", "outputs"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"VAR transition {section} must be a mapping")
    transition = config["transition"]
    frozen = {
        "state_order": ["growth_score", "inflation_score"],
        "lag_order": 1,
        "estimator": "expanding_multivariate_ols",
        "include_intercept": True,
        "include_time_trend": False,
        "regularization": "none",
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
    }
    for key, expected in frozen.items():
        if transition.get(key) != expected:
            raise ValueError(f"unexpected VAR transition setting for {key}")
    if int(transition.get("minimum_training_pairs", 0)) < 5:
        raise ValueError("minimum_training_pairs must be at least five")
    for output_path in config["outputs"].values():
        if "m02" not in str(output_path):
            raise ValueError("every Model 02 transition output must use an m02 namespace")
    return config, raw


def _manifest_hash(manifest: Mapping[str, Any], relative_path: str) -> str:
    matches = [
        str(item["sha256"])
        for item in manifest["generated_file_hashes"]
        if str(item["path"]) == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"upstream manifest does not identify {relative_path}")
    return matches[0]


def _verified_bytes(project_root: Path, relative_path: str, digest: str) -> bytes:
    payload = (project_root / relative_path).read_bytes()
    if sha256(payload) != digest:
        raise ValueError(f"input hash mismatch: {relative_path}")
    return payload


def _quadrants(row: pd.Series) -> list[dict[str, object]]:
    return [
        {
            "regime_id": regime,
            "regime_label": REGIME_LABELS[regime],
            "weight": float(row[f"weight_{regime}"]),
            "probability": float(row[f"probability_{regime}"]),
        }
        for regime in REGIME_ORDER
    ]


def _latest_payload(row: pd.Series) -> dict[str, object]:
    transition = [
        [
            float(row["a_growth_from_growth"]),
            float(row["a_growth_from_inflation"]),
        ],
        [
            float(row["a_inflation_from_growth"]),
            float(row["a_inflation_from_inflation"]),
        ],
    ]
    return {
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "prior_kind": str(row["prior_kind"]),
        "source_reference_month": row["source_reference_month"].date().isoformat(),
        "target_reference_month": row["target_reference_month"].date().isoformat(),
        "prior_available_at": row["prior_available_at"].date().isoformat(),
        "operational_timing": {
            "target_month_end": row["target_month_end"].date().isoformat(),
            "delay_from_target_start_days": int(
                row["prior_delay_from_target_start_days"]
            ),
            "available_by_target_start": bool(row["available_by_target_start"]),
            "available_by_target_end": bool(row["available_by_target_end"]),
        },
        "prior_status": str(row["prior_status"]),
        "interpretation": (
            "One-step marginal quadrant prior from the reporting distribution "
            "around the latent score VAR forecast; this is not a daily "
            "posterior or a four-month joint path."
        ),
        "fit": {
            "training_pairs": int(row["training_pairs"]),
            "latest_response_month": row[
                "fit_latest_destination_month"
            ].date().isoformat(),
            "residual_degrees_of_freedom": int(
                row["residual_degrees_of_freedom"]
            ),
            "intercept": [
                float(row["growth_intercept"]),
                float(row["inflation_intercept"]),
            ],
            "transition_matrix": transition,
            "innovation_covariance": [
                [
                    float(row["growth_innovation_variance"]),
                    float(row["growth_inflation_innovation_covariance"]),
                ],
                [
                    float(row["growth_inflation_innovation_covariance"]),
                    float(row["inflation_innovation_variance"]),
                ],
            ],
            "spectral_radius": float(row["spectral_radius"]),
            "is_stable": bool(row["is_stable"]),
            "design_condition_number": float(row["design_condition_number"]),
            "r_squared": {
                "growth": float(row["growth_r_squared"]),
                "inflation": float(row["inflation_r_squared"]),
            },
        },
        "source_state": {
            "score_is_exact": True,
            "score": [
                float(row["source_growth_score"]),
                float(row["source_inflation_score"]),
            ],
        },
        "target_mapping_covariance_proxy": {
            "method": (
                "latest_causally_available_baseline_map_at_source_cutoff"
            ),
            "proxy_reference_month": row[
                "mapping_proxy_reference_month"
            ].date().isoformat(),
            "proxy_available_at": row[
                "mapping_proxy_available_at"
            ].date().isoformat(),
            "covariance": [
                [
                    float(row["growth_mapping_proxy_variance"]),
                    float(row["growth_inflation_mapping_proxy_covariance"]),
                ],
                [
                    float(row["growth_inflation_mapping_proxy_covariance"]),
                    float(row["inflation_mapping_proxy_variance"]),
                ],
            ],
            "interpretation": (
                "Reporting-only approximation for the unavailable target-month "
                "mapping noise; it is not uncertainty in the exact source state."
            ),
        },
        "next_month_latent_prior": {
            "mean": [
                float(row["growth_prior_mean"]),
                float(row["inflation_prior_mean"]),
            ],
            "covariance": [
                [
                    float(row["growth_latent_prior_variance"]),
                    float(row["growth_inflation_latent_prior_covariance"]),
                ],
                [
                    float(row["growth_inflation_latent_prior_covariance"]),
                    float(row["inflation_latent_prior_variance"]),
                ],
            ],
        },
        "next_month_prior": {
            "state": "reporting_score_for_quadrant_integration",
            "mean": [
                float(row["growth_prior_mean"]),
                float(row["inflation_prior_mean"]),
            ],
            "covariance": [
                [
                    float(row["growth_prior_variance"]),
                    float(row["growth_inflation_prior_covariance"]),
                ],
                [
                    float(row["growth_inflation_prior_covariance"]),
                    float(row["inflation_prior_variance"]),
                ],
            ],
            "correlation": float(row["prior_correlation"]),
            "entropy": float(row["entropy"]),
            "quadrants": _quadrants(row),
            "quadrant_weight_sum": float(row["quadrant_weight_sum"]),
            "probability_sum": float(row["probability_sum"]),
        },
    }


def _coverage(priors: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, object]:
    available = priors[priors["prior_status"] == "available"]
    mapped = priors[priors["prior_status"] != "source_map_unavailable"]
    eligible_pairs = pairs[pairs["eligible_at_latest_cutoff"]]
    unstable = available[~available["is_stable"].astype(bool)]
    return {
        "score_history_rows": len(priors),
        "candidate_adjacent_pairs": len(pairs),
        "eligible_consecutive_pairs_at_latest_cutoff": len(eligible_pairs),
        "baseline_mapping_months": len(mapped),
        "available_prior_months": len(available),
        "first_source_month": available["source_reference_month"]
        .min()
        .date()
        .isoformat(),
        "first_target_month": available["target_reference_month"]
        .min()
        .date()
        .isoformat(),
        "latest_source_month": available["source_reference_month"]
        .max()
        .date()
        .isoformat(),
        "latest_target_month": available["target_reference_month"]
        .max()
        .date()
        .isoformat(),
        "latest_training_pairs": int(available.iloc[-1]["training_pairs"]),
        "unstable_prior_fits": len(unstable),
        "priors_available_by_target_start": int(
            available["available_by_target_start"].astype(bool).sum()
        ),
        "priors_available_by_target_end": int(
            available["available_by_target_end"].astype(bool).sum()
        ),
        "priors_available_after_target_end": int(
            (~available["available_by_target_end"].astype(bool)).sum()
        ),
    }


def _validate_coverage(actual: Mapping[str, object], expected: Mapping[str, object]) -> None:
    normalized = {
        key: value.isoformat() if hasattr(value, "isoformat") else value
        for key, value in expected.items()
    }
    if dict(actual) != normalized:
        raise ValueError(
            "VAR transition coverage differs from the frozen contract; "
            f"actual={actual}, expected={normalized}"
        )


def build_transition(
    *,
    project_root: Path,
    config_path: Path,
) -> dict[str, Path]:
    """Build, validate, and publish Model 02's VAR(1) transition priors."""

    config, config_bytes = _load_config(config_path)
    inputs = config["inputs"]
    transition_config = config["transition"]
    outputs = config["outputs"]

    score_manifest_path = project_root / inputs["score_manifest"]
    probability_manifest_path = project_root / inputs["probability_map_manifest"]
    score_manifest_bytes = score_manifest_path.read_bytes()
    probability_manifest_bytes = probability_manifest_path.read_bytes()
    score_manifest = json.loads(score_manifest_bytes)
    probability_manifest = json.loads(probability_manifest_bytes)
    if score_manifest.get("model_id") != MODEL_ID:
        raise ValueError("VAR transition received the wrong score manifest")
    if probability_manifest.get("model_id") != MODEL_ID:
        raise ValueError("VAR transition received the wrong mapping manifest")
    if probability_manifest.get("stage_id") != "m02_probability_map":
        raise ValueError("VAR transition requires the Model 02 probability map")
    expected_score_manifest_hash = probability_manifest.get(
        "score_manifest_sha256"
    )
    actual_score_manifest_hash = sha256(score_manifest_bytes)
    if expected_score_manifest_hash != actual_score_manifest_hash:
        raise ValueError(
            "probability-map lineage does not match the supplied score manifest"
        )

    score_relative = str(inputs["score_features"])
    mapping_relative = str(inputs["baseline_mapping"])
    score_bytes = _verified_bytes(
        project_root,
        score_relative,
        _manifest_hash(score_manifest, score_relative),
    )
    mapping_bytes = _verified_bytes(
        project_root,
        mapping_relative,
        _manifest_hash(probability_manifest, mapping_relative),
    )
    scores = pd.read_csv(
        BytesIO(score_bytes),
        parse_dates=["reference_month", "score_available_at"],
    ).set_index("reference_month")
    mapping = pd.read_csv(
        BytesIO(mapping_bytes),
        parse_dates=["reference_month", "score_available_at"],
    )
    revision_horizons = mapping["revision_horizon_months"]
    if revision_horizons.isna().any() or set(revision_horizons.astype(int)) != {12}:
        raise ValueError("transition input must contain only the 12-month baseline map")
    baseline_flags = mapping["is_baseline"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    if not baseline_flags.all():
        raise ValueError("transition input contains a nonbaseline mapping row")

    pair_audit = build_var_pair_audit(scores)
    priors = causal_var1_prior_history(
        scores,
        mapping,
        pair_audit,
        minimum_training_pairs=int(transition_config["minimum_training_pairs"]),
    )
    available = priors[priors["prior_status"] == "available"]
    if available.empty:
        raise ValueError("VAR transition produced no available next-month priors")

    probability_columns = [f"probability_{regime}" for regime in REGIME_ORDER]
    if not np.allclose(
        available[probability_columns].sum(axis=1),
        1.0,
        atol=1.0e-10,
        rtol=1.0e-10,
    ):
        raise ValueError("VAR next-month probabilities do not sum to one")
    if (available[probability_columns] < 0.0).any().any():
        raise ValueError("VAR next-month probabilities contain negative values")

    latest = available.iloc[-1]
    latest_month = pd.Timestamp(latest["source_reference_month"])
    latest_cutoff = pd.Timestamp(latest["prior_available_at"])
    latest_pairs = select_causal_var_pairs(
        pair_audit,
        source_reference_month=latest_month,
        forecast_available_at=latest_cutoff,
    )
    pair_audit["eligible_at_latest_cutoff"] = (
        (pair_audit["pair_status"] == "eligible")
        & (pair_audit["destination_reference_month"] <= latest_month)
        & (pair_audit["pair_available_at"] <= latest_cutoff)
    )
    if int(pair_audit["eligible_at_latest_cutoff"].sum()) != len(latest_pairs):
        raise ValueError("latest pair-audit eligibility is internally inconsistent")

    for row in available.itertuples(index=False):
        mapping_proxy_covariance = np.asarray(
            [
                [
                    row.growth_mapping_proxy_variance,
                    row.growth_inflation_mapping_proxy_covariance,
                ],
                [
                    row.growth_inflation_mapping_proxy_covariance,
                    row.inflation_mapping_proxy_variance,
                ],
            ],
            dtype=float,
        )
        innovation = np.asarray(
            [
                [
                    row.growth_innovation_variance,
                    row.growth_inflation_innovation_covariance,
                ],
                [
                    row.growth_inflation_innovation_covariance,
                    row.inflation_innovation_variance,
                ],
            ],
            dtype=float,
        )
        published = np.asarray(
            [
                [row.growth_prior_variance, row.growth_inflation_prior_covariance],
                [row.growth_inflation_prior_covariance, row.inflation_prior_variance],
            ],
            dtype=float,
        )
        latent_published = np.asarray(
            [
                [
                    row.growth_latent_prior_variance,
                    row.growth_inflation_latent_prior_covariance,
                ],
                [
                    row.growth_inflation_latent_prior_covariance,
                    row.inflation_latent_prior_variance,
                ],
            ],
            dtype=float,
        )
        if not np.allclose(
            latent_published, innovation, atol=1.0e-10, rtol=1.0e-10
        ):
            raise ValueError("published latent VAR covariance must equal Q")
        expected = innovation + mapping_proxy_covariance
        if not np.allclose(published, expected, atol=1.0e-10, rtol=1.0e-10):
            raise ValueError(
                "published reporting covariance must equal Q plus mapping proxy"
            )

    coverage = _coverage(priors, pair_audit)
    _validate_coverage(coverage, transition_config["expected_coverage"])
    latest_payload = _latest_payload(latest)
    unstable = available[~available["is_stable"].astype(bool)]
    march_2020 = available[
        available["source_reference_month"] == pd.Timestamp("2020-03-01")
    ]
    april_2020 = available[
        available["source_reference_month"] == pd.Timestamp("2020-04-01")
    ]
    summary_payload = {
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "coverage": coverage,
        "status_counts": {
            str(key): int(value)
            for key, value in priors["prior_status"].value_counts().items()
        },
        "unstable_source_months": [
            value.date().isoformat()
            for value in unstable["source_reference_month"].tolist()
        ],
        "operational_timing": {
            "forecast_issue_policy": "source_score_availability_date",
            "available_by_target_start": int(
                available["available_by_target_start"].astype(bool).sum()
            ),
            "available_by_target_end": int(
                available["available_by_target_end"].astype(bool).sum()
            ),
            "available_after_target_end_source_months": [
                value.date().isoformat()
                for value in available.loc[
                    ~available["available_by_target_end"].astype(bool),
                    "source_reference_month",
                ].tolist()
            ],
        },
        "covid_outlier_diagnostic": {
            "pre_shock_source_month": "2020-03-01",
            "pre_shock_growth_innovation_variance": (
                float(march_2020.iloc[0]["growth_innovation_variance"])
                if not march_2020.empty
                else None
            ),
            "shock_source_month": "2020-04-01",
            "shock_spectral_radius": (
                float(april_2020.iloc[0]["spectral_radius"])
                if not april_2020.empty
                else None
            ),
            "shock_growth_prior_mean": (
                float(april_2020.iloc[0]["growth_prior_mean"])
                if not april_2020.empty
                else None
            ),
            "interpretation": (
                "The literal unbounded-score OLS baseline is retained; no "
                "winsorization or stationarity projection is applied."
            ),
        },
        "latest": {
            "source_reference_month": latest_month.date().isoformat(),
            "target_reference_month": pd.Timestamp(
                latest["target_reference_month"]
            ).date().isoformat(),
            "training_pairs": int(latest["training_pairs"]),
            "spectral_radius": float(latest["spectral_radius"]),
            "growth_innovation_variance": float(
                latest["growth_innovation_variance"]
            ),
            "inflation_innovation_variance": float(
                latest["inflation_innovation_variance"]
            ),
            "mapping_proxy_reference_month": pd.Timestamp(
                latest["mapping_proxy_reference_month"]
            ).date().isoformat(),
            "reporting_covariance_convention": "Q_plus_Omega_map_proxy",
        },
    }

    pair_path = project_root / outputs["pair_audit"]
    history_path = project_root / outputs["published_history"]
    latest_path = project_root / outputs["published_latest"]
    summary_path = project_root / outputs["published_summary"]
    manifest_path = project_root / outputs["manifest"]
    write_csv(pair_audit, pair_path)
    write_csv(priors, history_path)
    write_json(latest_payload, latest_path)
    write_json(summary_payload, summary_path)

    generated_paths = [pair_path, history_path, latest_path, summary_path]
    generated_hashes = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
        for path in generated_paths
    ]
    snapshot_hash = hashlib.sha256()
    for payload in (score_manifest_bytes, probability_manifest_bytes, score_bytes, mapping_bytes):
        snapshot_hash.update(sha256(payload).encode("ascii"))
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(project_root).as_posix(),
        "configuration_sha256": sha256(config_bytes),
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                package: version(package)
                for package in ("numpy", "pandas", "PyYAML", "scipy")
            },
        },
        "upstream_manifests": [
            {
                "path": score_manifest_path.relative_to(project_root).as_posix(),
                "sha256": sha256(score_manifest_bytes),
            },
            {
                "path": probability_manifest_path.relative_to(project_root).as_posix(),
                "sha256": sha256(probability_manifest_bytes),
            },
        ],
        "input_files": [
            {"path": score_relative, "sha256": sha256(score_bytes)},
            {"path": mapping_relative, "sha256": sha256(mapping_bytes)},
        ],
        "data_snapshot_sha256": snapshot_hash.hexdigest(),
        "transition_definition": {
            key: value
            for key, value in transition_config.items()
            if key != "expected_coverage"
        },
        "coverage": coverage,
        "processed_files": [pair_path.relative_to(project_root).as_posix()],
        "published_files": [
            path.relative_to(project_root).as_posix()
            for path in (history_path, latest_path, summary_path)
        ],
        "generated_file_hashes": generated_hashes,
    }
    write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "pair_audit": pair_path,
        "history": history_path,
        "latest": latest_path,
        "summary": summary_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m02_var1_transition.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    """Parse command-line arguments and build Model 02 transition priors."""

    args = _parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    outputs = build_transition(
        project_root=project_root,
        config_path=config_path.resolve(),
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
