"""Build Model 02 Gaussian quadrant weights from deterministic scores.

This stage acquires exact three- and twelve-month as-of snapshots, measures
later-minus-first revisions in the frozen first-release score scale, estimates
causal expanding mapping uncertainty, and integrates a bivariate Gaussian over
the four growth/inflation quadrants.  It does not fit transition dynamics,
process leading-release likelihoods, or run an allocation backtest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from regime_allocation.data.dataset_acquisition import (
    MatrixAcquisition,
    as_date,
    download_or_load_exact_vintage_matrix,
    provider_raw_dir,
    sha256,
    write_csv,
    write_json,
)
from regime_allocation.data.providers import select_vintage_provider
from regime_allocation.data.providers.vintage_matrix import load_vintage_matrix
from regime_allocation.models.m02_soft_composite.probability_map import (
    REGIME_LABELS,
    REGIME_ORDER,
    causal_quadrant_mapping_history,
    component_disagreement_history,
)
from regime_allocation.models.m02_soft_composite.revisions import (
    aggregate_axis_revision_errors,
    component_revision_errors,
    first_release_standardization_scales,
    required_exact_vintages,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "m02_probability_map"


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load and validate the probability-map contract."""

    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("probability-map configuration must be a mapping")
    if str(config.get("model_id")) != MODEL_ID:
        raise ValueError(f"model_id must be {MODEL_ID!r}")
    if str(config.get("stage_id")) != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID!r}")

    mapping = config.get("mapping", {})
    if mapping.get("distribution") != "bivariate_gaussian":
        raise ValueError("Model 02 probability map must be bivariate Gaussian")
    if [float(item) for item in mapping.get("perturbation_mean", [])] != [0.0, 0.0]:
        raise ValueError("Model 02 mapping perturbation must have zero mean")
    horizons = tuple(int(value) for value in mapping.get("revision_horizons_months", []))
    if horizons != (3, 12):
        raise ValueError("Model 02 requires separate 3- and 12-month revisions")
    if int(mapping.get("baseline_revision_horizon_months", 0)) != 12:
        raise ValueError("the production probability map must use 12-month revisions")
    if mapping.get("training_cutoff") != "strictly_before_score_availability":
        raise ValueError("historical mapping fits must use a strict causal cutoff")

    disagreement = mapping.get("disagreement", {})
    if disagreement.get("estimator") != (
        "expanding_mean_delete_one_component_jackknife_variance"
    ):
        raise ValueError("unexpected disagreement estimator")
    if disagreement.get("covariance_structure") != "diagonal":
        raise ValueError("disagreement covariance must be diagonal")
    revision = mapping.get("revision", {})
    if revision.get("estimator") != "expanding_centered_sample_covariance":
        raise ValueError("unexpected revision covariance estimator")
    if revision.get("comparison") != "later_minus_first_release":
        raise ValueError("revision comparison must be later minus first release")
    if revision.get("standardization") != "frozen_first_release_expanding_scale":
        raise ValueError("revision errors must use the frozen first-release scale")

    for output_path in config.get("outputs", {}).values():
        if "m02" not in str(output_path):
            raise ValueError("every Model 02 output must use an m02 namespace")
    return config, raw


def _verified_bytes(project_root: Path, relative_path: str, digest: str) -> bytes:
    path = project_root / relative_path
    payload = path.read_bytes()
    if sha256(payload) != digest:
        raise ValueError(f"input hash mismatch: {relative_path}")
    return payload


def _generated_hash(score_manifest: Mapping[str, Any], relative_path: str) -> str:
    matches = [
        str(item["sha256"])
        for item in score_manifest["generated_file_hashes"]
        if str(item["path"]) == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"score manifest does not uniquely identify {relative_path}")
    return matches[0]


def _probability_entries(row: pd.Series) -> list[dict[str, object]]:
    return [
        {
            "regime_id": regime,
            "regime_label": REGIME_LABELS[regime],
            "weight": float(row[f"weight_{regime}"]),
            "probability": float(row[f"probability_{regime}"]),
        }
        for regime in REGIME_ORDER
    ]


def _mapping_payload(row: pd.Series) -> dict[str, object]:
    return {
        "specification_id": str(row["specification_id"]),
        "revision_horizon_months": int(row["revision_horizon_months"]),
        "mapping_status": str(row["mapping_status"]),
        "training_counts": {
            "disagreement_months": int(row["disagreement_training_months"]),
            "revision_months": int(row["revision_training_months"]),
        },
        "omega_disagreement": [
            [float(row["growth_disagreement_variance"]), 0.0],
            [0.0, float(row["inflation_disagreement_variance"])],
        ],
        "omega_revision": [
            [
                float(row["growth_revision_variance"]),
                float(row["growth_inflation_revision_covariance"]),
            ],
            [
                float(row["growth_inflation_revision_covariance"]),
                float(row["inflation_revision_variance"]),
            ],
        ],
        "omega_map": [
            [
                float(row["growth_map_variance"]),
                float(row["growth_inflation_map_covariance"]),
            ],
            [
                float(row["growth_inflation_map_covariance"]),
                float(row["inflation_map_variance"]),
            ],
        ],
        "map_correlation": float(row["map_correlation"]),
        "revision_error_sample_mean": [
            float(row["growth_revision_mean"]),
            float(row["inflation_revision_mean"]),
        ],
        "entropy": float(row["entropy"]),
        "quadrants": _probability_entries(row),
        "quadrant_weight_sum": float(row["quadrant_weight_sum"]),
        "probability_sum": float(row["probability_sum"]),
    }


def _coverage_summary(
    mapping_history: pd.DataFrame,
    axis_errors: pd.DataFrame,
    *,
    horizon: int,
    baseline_horizon: int,
) -> dict[str, object]:
    subset = mapping_history[
        mapping_history["revision_horizon_months"] == horizon
    ]
    available = subset[subset["mapping_status"] == "available"]
    mature_errors = axis_errors[
        (axis_errors["horizon_months"] == horizon)
        & (axis_errors["revision_status"] == "available")
    ]
    return {
        "revision_horizon_months": horizon,
        "is_baseline": horizon == baseline_horizon,
        "complete_revision_errors": len(mature_errors),
        "first_revision_error_month": (
            mature_errors["reference_month"].min().date().isoformat()
            if not mature_errors.empty
            else None
        ),
        "latest_revision_error_month": (
            mature_errors["reference_month"].max().date().isoformat()
            if not mature_errors.empty
            else None
        ),
        "available_probability_months": len(available),
        "first_probability_month": (
            available["reference_month"].min().date().isoformat()
            if not available.empty
            else None
        ),
        "latest_probability_month": (
            available["reference_month"].max().date().isoformat()
            if not available.empty
            else None
        ),
    }


def _validate_coverage_contract(
    actual: list[dict[str, object]], expected: list[dict[str, object]]
) -> None:
    ignored = {"is_baseline"}
    normalized_actual = [
        {key: value for key, value in item.items() if key not in ignored}
        for item in actual
    ]
    normalized_expected = [
        {
            key: (value.isoformat() if hasattr(value, "isoformat") else value)
            for key, value in item.items()
        }
        for item in expected
    ]
    if normalized_actual != normalized_expected:
        raise ValueError(
            "probability-map coverage differs from the frozen contract; "
            f"actual={normalized_actual}, expected={normalized_expected}"
        )


def build_probability_map(
    *,
    project_root: Path,
    config_path: Path,
    refresh: bool = False,
    provider: str = "auto",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Build, audit, and publish causal Model 02 quadrant probabilities."""

    config, config_bytes = _load_config(config_path)
    inputs = config["inputs"]
    data_config = config["data"]
    mapping_config = config["mapping"]
    output_config = config["outputs"]

    score_manifest_path = project_root / inputs["score_manifest"]
    score_manifest_bytes = score_manifest_path.read_bytes()
    score_manifest = json.loads(score_manifest_bytes)
    if score_manifest.get("model_id") != MODEL_ID:
        raise ValueError("probability map received the wrong score manifest")
    if score_manifest.get("stage") != "deterministic_composite_score_definition":
        raise ValueError("probability map requires the deterministic score stage")

    components_relative = str(inputs["first_release_components"])
    scores_relative = str(inputs["score_features"])
    components_bytes = _verified_bytes(
        project_root,
        components_relative,
        _generated_hash(score_manifest, components_relative),
    )
    scores_bytes = _verified_bytes(
        project_root,
        scores_relative,
        _generated_hash(score_manifest, scores_relative),
    )
    score_config_relative = str(score_manifest["configuration"])
    score_config_bytes = _verified_bytes(
        project_root,
        score_config_relative,
        str(score_manifest["configuration_sha256"]),
    )
    score_config = yaml.safe_load(score_config_bytes)

    first_release = pd.read_csv(
        project_root / components_relative,
        parse_dates=["reference_month", "release_date"],
    )
    score_features = pd.read_csv(
        project_root / scores_relative,
        parse_dates=[
            "reference_month",
            "growth_score_available_at",
            "inflation_score_available_at",
            "score_available_at",
        ],
    ).set_index("reference_month")
    if score_features.index.has_duplicates:
        raise ValueError("score feature history contains duplicate months")
    complete_score_months = score_features.index[
        score_features[["growth_score", "inflation_score"]].notna().all(axis=1)
    ]
    if len(complete_score_months) != int(score_manifest["complete_score_months"]):
        raise ValueError("complete score count differs from its source manifest")

    feature_config = score_config["features"]
    scales = first_release_standardization_scales(
        first_release,
        min_history=int(feature_config["min_history_months"]),
        ddof=int(feature_config["standard_deviation_ddof"]),
    )
    horizons = tuple(
        int(value) for value in mapping_config["revision_horizons_months"]
    )
    knowledge_cutoff = pd.Timestamp(data_config["knowledge_cutoff"]).normalize()
    required_vintages = required_exact_vintages(
        first_release,
        reference_months=complete_score_months,
        horizons=horizons,
        knowledge_cutoff=knowledge_cutoff,
    )

    selection = select_vintage_provider(provider, environ=environ)
    client = selection.client
    legacy_revision_raw_dir = project_root / data_config["revision_raw_dir"]
    revision_raw_dir = provider_raw_dir(
        legacy_revision_raw_dir, client.cache_namespace
    )
    observation_start = as_date(score_config["data"]["observation_start"])
    observation_end = as_date(score_config["data"]["reference_end"])
    exact_matrices: dict[str, pd.DataFrame] = {}
    raw_files: list[dict[str, object]] = []
    for series_id, vintage_dates in sorted(required_vintages.items()):
        acquisition: MatrixAcquisition = download_or_load_exact_vintage_matrix(
            client=client,
            series_id=series_id,
            raw_dir=revision_raw_dir,
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_dates=tuple(item.date() for item in vintage_dates),
            refresh=refresh,
        )
        matrix = load_vintage_matrix(acquisition.content, series_id)
        exact_matrices[series_id] = matrix
        raw_files.append(
            {
                "series_id": series_id,
                "provider": acquisition.provider_id,
                "source_url": acquisition.source_url,
                "cache_origin": acquisition.cache_origin,
                "path": acquisition.path.relative_to(project_root).as_posix(),
                "sha256": sha256(acquisition.content),
                "bytes": len(acquisition.content),
                "vintage_count": len(vintage_dates),
                "first_vintage": vintage_dates[0].date().isoformat(),
                "last_vintage": vintage_dates[-1].date().isoformat(),
            }
        )

    component_errors = component_revision_errors(
        first_release,
        scales=scales,
        exact_vintage_matrices=exact_matrices,
        reference_months=complete_score_months,
        horizons=horizons,
        knowledge_cutoff=knowledge_cutoff,
    )
    axis_errors = aggregate_axis_revision_errors(
        component_errors,
        reference_months=complete_score_months,
        horizons=horizons,
    )
    disagreement = component_disagreement_history(score_features)
    mapping_history = causal_quadrant_mapping_history(
        score_features,
        disagreement,
        axis_errors,
        horizons=horizons,
        baseline_horizon=int(mapping_config["baseline_revision_horizon_months"]),
        minimum_disagreement_months=int(
            mapping_config["disagreement"]["minimum_complete_months"]
        ),
        minimum_revision_months=int(
            mapping_config["revision"]["minimum_complete_months"]
        ),
    )

    baseline_horizon = int(mapping_config["baseline_revision_horizon_months"])
    baseline_history = mapping_history[
        mapping_history["revision_horizon_months"] == baseline_horizon
    ].copy()
    sensitivity_history = mapping_history[
        mapping_history["revision_horizon_months"] != baseline_horizon
    ].copy()
    available_baseline = baseline_history[
        baseline_history["mapping_status"] == "available"
    ]
    if available_baseline.empty:
        raise ValueError("no baseline quadrant probabilities were produced")
    probability_columns = [f"probability_{regime}" for regime in REGIME_ORDER]
    probability_sums = available_baseline[probability_columns].sum(axis=1)
    if not np.allclose(probability_sums, 1.0, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("published quadrant probabilities do not sum to one")
    if (available_baseline[probability_columns] < 0.0).any().any():
        raise ValueError("published quadrant probabilities contain negative values")

    latest_reference_month = complete_score_months.max()
    latest_rows = mapping_history[
        (mapping_history["reference_month"] == latest_reference_month)
        & (mapping_history["mapping_status"] == "available")
    ].sort_values("revision_horizon_months")
    if set(latest_rows["revision_horizon_months"]) != set(horizons):
        raise ValueError("latest score lacks a complete horizon sensitivity map")
    latest_baseline = latest_rows[
        latest_rows["revision_horizon_months"] == baseline_horizon
    ].iloc[0]
    latest_sensitivities = latest_rows[
        latest_rows["revision_horizon_months"] != baseline_horizon
    ]

    processed_dir = project_root / output_config["processed_dir"]
    published_dir = project_root / output_config["published_dir"]
    manifest_path = project_root / output_config["manifest"]
    component_errors_path = processed_dir / "component_revision_errors.csv"
    axis_errors_path = processed_dir / "axis_revision_errors.csv"
    disagreement_path = processed_dir / "component_disagreement.csv"
    full_history_path = processed_dir / "probability_map_all_specifications.csv"
    baseline_path = published_dir / "quadrant_probabilities.csv"
    sensitivity_path = published_dir / "revision_horizon_sensitivity.csv"
    latest_path = published_dir / "latest_quadrant_probabilities.json"
    summary_path = published_dir / "uncertainty_summary.json"

    latest_payload = {
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "reference_month": latest_reference_month.date().isoformat(),
        "score_available_at": pd.Timestamp(
            latest_baseline["score_available_at"]
        ).date().isoformat(),
        "deterministic_score": {
            "growth": float(latest_baseline["growth_score"]),
            "inflation": float(latest_baseline["inflation_score"]),
        },
        "interpretation": (
            "Gaussian quadrant mapping of the observed composite score; these "
            "weights are not yet transition forecasts or event-updated posteriors."
        ),
        "baseline": _mapping_payload(latest_baseline),
        "sensitivities": [
            _mapping_payload(row)
            for _, row in latest_sensitivities.iterrows()
        ],
    }

    specification_summaries = [
        _coverage_summary(
            mapping_history,
            axis_errors,
            horizon=horizon,
            baseline_horizon=baseline_horizon,
        )
        for horizon in horizons
    ]
    _validate_coverage_contract(
        specification_summaries,
        list(mapping_config["expected_coverage"]),
    )
    summary_payload = {
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "knowledge_cutoff": knowledge_cutoff.date().isoformat(),
        "minimum_disagreement_months": int(
            mapping_config["disagreement"]["minimum_complete_months"]
        ),
        "minimum_revision_months": int(
            mapping_config["revision"]["minimum_complete_months"]
        ),
        "specifications": specification_summaries,
        "latest": {
            "reference_month": latest_reference_month.date().isoformat(),
            "baseline_revision_horizon_months": baseline_horizon,
            "omega_disagreement": latest_payload["baseline"]["omega_disagreement"],
            "omega_revision": latest_payload["baseline"]["omega_revision"],
            "omega_map": latest_payload["baseline"]["omega_map"],
            "map_correlation": latest_payload["baseline"]["map_correlation"],
        },
    }

    write_csv(component_errors, component_errors_path)
    write_csv(axis_errors, axis_errors_path)
    write_csv(disagreement.reset_index(), disagreement_path)
    write_csv(mapping_history, full_history_path)
    write_csv(baseline_history, baseline_path)
    write_csv(sensitivity_history, sensitivity_path)
    write_json(latest_payload, latest_path)
    write_json(summary_payload, summary_path)

    generated_paths = [
        component_errors_path,
        axis_errors_path,
        disagreement_path,
        full_history_path,
        baseline_path,
        sensitivity_path,
        latest_path,
        summary_path,
    ]
    generated_hashes = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
        for path in generated_paths
    ]
    combined_snapshot = hashlib.sha256()
    combined_snapshot.update(sha256(score_manifest_bytes).encode("ascii"))
    for item in sorted(raw_files, key=lambda value: str(value["series_id"])):
        combined_snapshot.update(str(item["sha256"]).encode("ascii"))
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
                for package in (
                    "numpy",
                    "pandas",
                    "PyYAML",
                    "scikit-learn",
                    "scipy",
                )
            },
        },
        "score_manifest": score_manifest_path.relative_to(project_root).as_posix(),
        "score_manifest_sha256": sha256(score_manifest_bytes),
        "input_files": [
            {"path": components_relative, "sha256": sha256(components_bytes)},
            {"path": scores_relative, "sha256": sha256(scores_bytes)},
            {"path": score_config_relative, "sha256": sha256(score_config_bytes)},
        ],
        "provider_requested": selection.requested,
        "provider_selected": selection.selected,
        "providers_used": sorted({str(item["provider"]) for item in raw_files}),
        "provider_output": "exact fixed-horizon as-of level snapshots",
        "knowledge_cutoff": knowledge_cutoff.date().isoformat(),
        "data_snapshot_sha256": combined_snapshot.hexdigest(),
        "mapping_definition": {
            "distribution": "bivariate_gaussian",
            "perturbation_mean": [0.0, 0.0],
            "omega_map": "omega_disagreement_plus_omega_revision",
            "disagreement": (
                "diagonal expanding mean of monthly delete-one-component "
                "jackknife variances"
            ),
            "revision": "expanding centered sample covariance in score units",
            "baseline_revision_horizon_months": baseline_horizon,
            "sensitivity_revision_horizons_months": [
                value for value in horizons if value != baseline_horizon
            ],
            "training_cutoff": "strictly_before_score_availability",
        },
        "coverage": specification_summaries,
        "raw_files": raw_files,
        "processed_files": [
            path.relative_to(project_root).as_posix()
            for path in [
                component_errors_path,
                axis_errors_path,
                disagreement_path,
                full_history_path,
            ]
        ],
        "published_files": [
            path.relative_to(project_root).as_posix()
            for path in [baseline_path, sensitivity_path, latest_path, summary_path]
        ],
        "generated_file_hashes": generated_hashes,
    }
    write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "baseline_history": baseline_path,
        "sensitivity_history": sensitivity_path,
        "latest": latest_path,
        "summary": summary_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m02_probability_map.yaml"),
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--provider",
        choices=("auto", "fred", "alfred"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    """Parse command-line arguments and build the probability map."""

    args = _parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    outputs = build_probability_map(
        project_root=project_root,
        config_path=config_path.resolve(),
        refresh=args.refresh,
        provider=args.provider,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
