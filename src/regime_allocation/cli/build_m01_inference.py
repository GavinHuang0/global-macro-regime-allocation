"""Fit release likelihoods and replay Model 01's causal Bayesian filter.

The command reads deterministic labels, leading-evidence events, and inference
configuration; estimates release-block likelihoods at successive historical
cutoffs; updates the joint four-month regime path; and publishes event
posteriors, forecast checkpoints, metrics, calibration tables, sensitivities,
the latest state, and a hash manifest.

Every likelihood fit uses events and target labels available strictly before
the forecast event. Forecast scoring also requires the checkpoint to precede
the realized label's cumulative availability date.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from regime_allocation.models.m01_deterministic_composite.inference import (
    STATE_IDS,
    path_marginals,
    probability_entropy,
)
from regime_allocation.models.m01_deterministic_composite.pipeline import (
    REGIME_LABELS,
    REGIME_ORDER,
)
from regime_allocation.models.m01_deterministic_composite.transition import (
    estimate_transition_matrix,
    propagate_regime_marginal,
)
from regime_allocation.models.m01_deterministic_composite.walkforward import (
    LikelihoodSpecification,
    attach_forecast_targets,
    build_causal_training_cache,
    evaluate_forecast_table,
    normalize_regime_history,
    prepare_block_vectors,
    run_walk_forward_filter,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d",
        compression=compression,
    )
    temporary.replace(path)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("Bayesian-filter configuration must be a mapping")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported Bayesian-filter configuration schema")
    if config.get("model_id") != "m01_deterministic_composite":
        raise ValueError("Bayesian-filter configuration has the wrong model_id")
    if config.get("stage_id") != "event_driven_bayesian_filter":
        raise ValueError("Bayesian-filter configuration has the wrong stage_id")
    state = config.get("state", {})
    if state.get("regime_order") != list(STATE_IDS):
        raise ValueError("configuration regime order is not canonical")
    if state.get("path_months") != 4 or state.get("path_cardinality") != 256:
        raise ValueError("Model 01 inference requires a four-month, 256-path state")
    filter_config = config.get("filter", {})
    if filter_config.get("replay_end_policy") != "latest_available_information_date":
        raise ValueError(
            "filter replay_end_policy must be latest_available_information_date"
        )
    blocks = config.get("blocks")
    if not isinstance(blocks, dict) or set(blocks) != {
        "weekly_claims",
        "jolts",
        "retail_sales",
        "housing",
        "durable_goods",
    }:
        raise ValueError("configuration must contain the five frozen evidence blocks")
    claims = blocks["weekly_claims"]
    if claims.get("source_series") != ["ICSA"]:
        raise ValueError("the initial weekly-claims likelihood must use ICSA only")
    likelihood = config.get("likelihood", {})
    if likelihood.get("family") != "multivariate_student_t":
        raise ValueError("baseline likelihood must be multivariate Student-t")
    if float(likelihood.get("degrees_of_freedom", float("nan"))) != 7.0:
        raise ValueError("baseline degrees_of_freedom must equal 7")
    if float(likelihood.get("regime_means", {}).get("kappa", float("nan"))) != 5.0:
        raise ValueError("baseline regime-mean kappa must equal 5")
    covariance = likelihood.get("covariance", {})
    if covariance.get("estimator") != "ledoit_wolf":
        raise ValueError("baseline covariance estimator must be Ledoit-Wolf")
    if not math.isclose(
        float(covariance.get("t_shape_multiplier", float("nan"))),
        5.0 / 7.0,
    ):
        raise ValueError("Student-t shape multiplier must equal 5/7")
    outputs = config.get("outputs")
    required_outputs = {
        "processed_dir",
        "checkpoint_index",
        "joint_path_checkpoints",
        "marginal_checkpoints",
        "event_update_audit",
        "likelihood_fit_audit",
        "forecast_predictions",
        "evaluation_metrics",
        "calibration_bins",
        "sensitivity_specifications",
        "sensitivity_metrics",
        "manifest",
        "published_dir",
        "latest_posterior",
        "evaluation_summary",
        "published_sensitivity_metrics",
    }
    if not isinstance(outputs, dict) or not required_outputs.issubset(outputs):
        raise ValueError("configuration is missing required inference output paths")
    return config, raw


def _block_config(config: dict[str, Any]) -> dict[str, dict[str, object]]:
    return {
        block_id: {
            "feature_names": tuple(str(item) for item in value["feature_names"]),
            "source_series": tuple(str(item) for item in value["source_series"]),
        }
        for block_id, value in config["blocks"].items()
    }


def _sensitivity_specifications(
    config: dict[str, Any],
) -> tuple[LikelihoodSpecification, ...]:
    likelihood = config["likelihood"]
    minimum = int(likelihood["minimum_complete_vectors"])
    baseline = LikelihoodSpecification(
        specification_id="baseline",
        degrees_of_freedom=float(likelihood["degrees_of_freedom"]),
        kappa=float(likelihood["regime_means"]["kappa"]),
        covariance_method="ledoit_wolf",
        scale_multiplier=1.0,
        minimum_complete_vectors=minimum,
    )
    specifications = [baseline]
    sensitivity = config["sensitivity"]["parameters"]
    for value in sensitivity["degrees_of_freedom"]:
        degrees = None if str(value).lower() == "gaussian" else float(value)
        if degrees == baseline.degrees_of_freedom:
            continue
        name = "gaussian" if degrees is None else f"df_{degrees:g}"
        specifications.append(
            LikelihoodSpecification(
                specification_id=f"sensitivity_{name}",
                degrees_of_freedom=degrees,
                kappa=baseline.kappa,
                minimum_complete_vectors=minimum,
            )
        )
    for value in sensitivity["regime_mean_kappa"]:
        kappa = float(value)
        if kappa == baseline.kappa:
            continue
        specifications.append(
            LikelihoodSpecification(
                specification_id=f"sensitivity_kappa_{kappa:g}",
                degrees_of_freedom=baseline.degrees_of_freedom,
                kappa=kappa,
                minimum_complete_vectors=minimum,
            )
        )
    for value in sensitivity["covariance_estimator"]:
        name = str(value)
        if name == "ledoit_wolf":
            continue
        if name == "empirical":
            method = "empirical"
            fixed = None
        elif name.startswith("fixed_spherical_"):
            method = "fixed_spherical"
            fixed = float(name.rsplit("_", maxsplit=1)[1])
        else:
            raise ValueError(f"unknown covariance sensitivity: {name}")
        specifications.append(
            LikelihoodSpecification(
                specification_id=f"sensitivity_covariance_{name}",
                degrees_of_freedom=baseline.degrees_of_freedom,
                kappa=baseline.kappa,
                covariance_method=method,
                fixed_shrinkage=fixed,
                minimum_complete_vectors=minimum,
            )
        )
    for value in sensitivity["scale_standard_deviation_multiplier"]:
        multiplier = float(value)
        if multiplier == 1.0:
            continue
        specifications.append(
            LikelihoodSpecification(
                specification_id=f"sensitivity_scale_{multiplier:g}",
                degrees_of_freedom=baseline.degrees_of_freedom,
                kappa=baseline.kappa,
                scale_multiplier=multiplier,
                minimum_complete_vectors=minimum,
            )
        )
    identifiers = [item.specification_id for item in specifications]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("sensitivity specification identifiers are not unique")
    return tuple(specifications)


def _latest_payload(
    baseline: Any,
    history: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    targets = normalize_regime_history(history).set_index("reference_month")
    path_probabilities = path_marginals(baseline.latest_path)
    anchor = baseline.latest_anchor_month
    months = tuple(
        (anchor.to_period("M") - lag).to_timestamp() for lag in range(3, -1, -1)
    )
    marginals = []
    for axis, reference_month in enumerate(months):
        probabilities = path_probabilities[axis]
        target = targets.loc[reference_month] if reference_month in targets.index else None
        confirmed = bool(
            target is not None
            and pd.notna(target["regime_id"])
            and pd.notna(target["label_available_at"])
            and pd.Timestamp(target["label_available_at"]) <= as_of_date
        )
        marginals.append(
            {
                "reference_month": reference_month.date().isoformat(),
                "relative_month": axis - 3,
                "status": "confirmed" if confirmed else "filtered_posterior",
                "entropy": probability_entropy(probabilities),
                "probability_sum": float(probabilities.sum()),
                "probabilities": [
                    {
                        "regime_id": state_id,
                        "regime_label": REGIME_LABELS[REGIME_ORDER[position]],
                        "probability": float(probabilities[position]),
                    }
                    for position, state_id in enumerate(STATE_IDS)
                ],
            }
        )
    next_probabilities = propagate_regime_marginal(
        path_probabilities[-1], baseline.latest_transition_matrix
    )
    return {
        "schema_version": 1,
        "model_id": "m01_deterministic_composite",
        "stage_id": "event_driven_bayesian_filter",
        "status": "research_nowcast_not_investment_advice",
        "as_of_date": as_of_date.date().isoformat(),
        "checkpoint_type": "latest",
        "baseline_likelihood": baseline.specification.to_record(),
        "state_order": [
            {
                "regime_id": state_id,
                "regime_label": REGIME_LABELS[REGIME_ORDER[position]],
            }
            for position, state_id in enumerate(STATE_IDS)
        ],
        "path_anchor_month": anchor.date().isoformat(),
        "joint_path_probability_sum": float(baseline.latest_path.sum()),
        "path_marginals": marginals,
        "next_month_transition_forecast": {
            "reference_month": (
                (anchor.to_period("M") + 1).to_timestamp().date().isoformat()
            ),
            "probability_sum": float(next_probabilities.sum()),
            "transition_matrix_policy": "frozen_at_latest_calendar_month_roll",
            "probabilities": [
                {
                    "regime_id": state_id,
                    "regime_label": REGIME_LABELS[REGIME_ORDER[position]],
                    "probability": float(next_probabilities[position]),
                }
                for position, state_id in enumerate(STATE_IDS)
            ],
        },
        "interpretation": (
            "Event-driven posterior using leading releases and causal deterministic "
            "label confirmations. Unconfirmed months remain probability distributions."
        ),
    }


def build_inference(
    *,
    project_root: Path,
    config_path: Path,
) -> dict[str, Path]:
    """Run, evaluate, and atomically publish the complete inference stage."""

    config, config_bytes = _load_config(config_path)
    sources = config["sources"]
    history_path = project_root / str(sources["regime_history"])
    event_path = project_root / str(sources["leading_event_table"])
    history_bytes = history_path.read_bytes()
    event_bytes = event_path.read_bytes()
    history = pd.read_csv(history_path)
    events = pd.read_csv(event_path)
    block_vectors = prepare_block_vectors(
        events,
        history,
        block_configuration=_block_config(config),
    )
    filter_start = pd.Timestamp(
        config["filter"]["replay_start_reference_month"]
    ).normalize()
    event_end = pd.to_datetime(events["release_date"], errors="raise").max().normalize()
    label_end = pd.to_datetime(
        history["label_available_at"], errors="coerce"
    ).max()
    filter_end = max(event_end, pd.Timestamp(label_end).normalize())
    alpha = float(config["filter"]["transition"]["dirichlet_alpha"])
    specifications = _sensitivity_specifications(config)
    transition_roll_dates = pd.date_range(
        start=filter_start,
        end=filter_end,
        freq="MS",
    )
    transition_estimates = {
        pd.Timestamp(roll_date): estimate_transition_matrix(
            history,
            knowledge_cutoff=pd.Timestamp(roll_date) - pd.Timedelta(days=1),
            alpha=alpha,
        )
        for roll_date in transition_roll_dates
    }
    causal_training_cache = build_causal_training_cache(
        block_vectors,
        filter_start=filter_start,
        filter_end=filter_end,
        minimum_complete_vectors=specifications[0].minimum_complete_vectors,
    )

    baseline = run_walk_forward_filter(
        history,
        block_vectors,
        specification=specifications[0],
        filter_start=filter_start,
        filter_end=filter_end,
        transition_alpha=alpha,
        apply_evidence=True,
        store_detailed_paths=True,
        transition_estimates=transition_estimates,
        causal_training_cache=causal_training_cache,
    )
    transition_spec = LikelihoodSpecification(
        specification_id="transition_only",
        degrees_of_freedom=7.0,
        kappa=5.0,
        minimum_complete_vectors=specifications[0].minimum_complete_vectors,
    )
    forecast_frames = [baseline.forecasts]
    transition_run = run_walk_forward_filter(
        history,
        block_vectors,
        specification=transition_spec,
        filter_start=filter_start,
        filter_end=filter_end,
        transition_alpha=alpha,
        apply_evidence=False,
        store_detailed_paths=False,
        store_checkpoint_artifacts=False,
        store_audits=False,
        transition_estimates=transition_estimates,
        causal_training_cache=causal_training_cache,
    )
    forecast_frames.append(transition_run.forecasts)
    for specification in specifications[1:]:
        sensitivity_run = run_walk_forward_filter(
            history,
            block_vectors,
            specification=specification,
            filter_start=filter_start,
            filter_end=filter_end,
            transition_alpha=alpha,
            apply_evidence=True,
            store_detailed_paths=False,
            store_checkpoint_artifacts=False,
            store_audits=False,
            transition_estimates=transition_estimates,
            causal_training_cache=causal_training_cache,
        )
        forecast_frames.append(sensitivity_run.forecasts)
    forecasts = attach_forecast_targets(
        pd.concat(forecast_frames, ignore_index=True),
        history,
        evaluation_start=config["evaluation"]["start_reference_month"],
    )
    evaluated = evaluate_forecast_table(
        forecasts,
        transition_only_specification_id="transition_only",
        calibration_bin_count=int(config["evaluation"]["calibration_bins"]),
    )

    outputs = config["outputs"]
    processed_dir = project_root / str(outputs["processed_dir"])
    published_dir = project_root / str(outputs["published_dir"])
    paths = {
        "checkpoints": processed_dir / str(outputs["checkpoint_index"]),
        "joint_paths": processed_dir / str(outputs["joint_path_checkpoints"]),
        "marginals": processed_dir / str(outputs["marginal_checkpoints"]),
        "event_audit": processed_dir / str(outputs["event_update_audit"]),
        "fit_audit": processed_dir / str(outputs["likelihood_fit_audit"]),
        "forecasts": processed_dir / str(outputs["forecast_predictions"]),
        "metrics": processed_dir / str(outputs["evaluation_metrics"]),
        "calibration": processed_dir / str(outputs["calibration_bins"]),
        "specifications": processed_dir / str(outputs["sensitivity_specifications"]),
        "sensitivity": processed_dir / str(outputs["sensitivity_metrics"]),
        "manifest": project_root / str(outputs["manifest"]),
        "latest": published_dir / str(outputs["latest_posterior"]),
        "evaluation": published_dir / str(outputs["evaluation_summary"]),
        "published_sensitivity": (
            published_dir / str(outputs["published_sensitivity_metrics"])
        ),
    }
    specification_frame = pd.DataFrame.from_records(
        [item.to_record() for item in specifications]
        + [
            {
                **transition_spec.to_record(),
                "distribution": "not_applied",
            }
        ]
    )
    sensitivity_metrics = evaluated.metrics.loc[
        evaluated.metrics["specification_id"] != "baseline"
    ].copy()
    _write_csv(baseline.checkpoints, paths["checkpoints"])
    _write_csv(baseline.joint_paths, paths["joint_paths"])
    _write_csv(baseline.marginals, paths["marginals"])
    _write_csv(baseline.event_audit, paths["event_audit"])
    _write_csv(baseline.likelihood_fit_audit, paths["fit_audit"])
    _write_csv(forecasts, paths["forecasts"])
    _write_csv(evaluated.metrics, paths["metrics"])
    _write_csv(evaluated.calibration_bins, paths["calibration"])
    _write_csv(specification_frame, paths["specifications"])
    _write_csv(sensitivity_metrics, paths["sensitivity"])
    _write_csv(sensitivity_metrics, paths["published_sensitivity"])

    latest_payload = _latest_payload(
        baseline,
        history,
        as_of_date=filter_end,
    )
    _write_json(latest_payload, paths["latest"])
    fixed_checkpoint_types = {"month_start", "month_end", "pre_confirmation"}
    summary_metrics = evaluated.metrics.loc[
        evaluated.metrics["specification_id"].isin(
            ["baseline", "transition_only"]
        )
        & evaluated.metrics["checkpoint_type"].isin(fixed_checkpoint_types)
    ]
    evaluation_payload = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "stage_id": config["stage_id"],
        "evaluation_start_reference_month": config["evaluation"][
            "start_reference_month"
        ],
        "metric_definitions": {
            "negative_log_likelihood": (
                "Mean -log probability assigned to the realized regime; lower is "
                "better and exact zero probability has infinite loss."
            ),
            "multiclass_brier_score": (
                "Mean unscaled sum of squared four-class probability errors, "
                "bounded from 0 to 2; lower is better."
            ),
            "map_accuracy": "Share of maximum-probability regime calls that are correct.",
            "balanced_accuracy": (
                "Mean recall across realized regimes, giving rare regimes equal weight."
            ),
            "macro_f1": (
                "Mean harmonic precision-recall score across active realized or "
                "predicted regimes."
            ),
            "expected_calibration_error": (
                "Bin-weighted gap between forecast probabilities and realized "
                "frequencies; reported for top-label and classwise views."
            ),
            "axis_brier_scores": (
                "Binary Brier scores after aggregating quadrants into growth-up and "
                "inflation-up probabilities."
            ),
            "posterior_entropy": (
                "Average uncertainty/sharpness in nats; lower is sharper but not "
                "necessarily more accurate."
            ),
            "brier_skill_vs_transition_only": (
                "One minus model Brier divided by transition-only Brier; positive "
                "values improve on the causal transition baseline."
            ),
        },
        "fixed_checkpoint_metrics": summary_metrics.to_dict(orient="records"),
        "warnings": [
            "Post-confirmation checkpoints are never scored as forecasts.",
            "JOLTS releases arriving after their target is confirmed are recorded "
            "as no-op evidence.",
            "Sensitivity results are descriptive and were not used to select the "
            "baseline on the evaluation sample.",
        ],
    }
    _write_json(evaluation_payload, paths["evaluation"])

    processed_files = []
    for name in (
        "checkpoints",
        "joint_paths",
        "marginals",
        "event_audit",
        "fit_audit",
        "forecasts",
        "metrics",
        "calibration",
        "specifications",
        "sensitivity",
    ):
        path = paths[name]
        processed_files.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path.read_bytes()),
            }
        )
    manifest = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "stage_id": config["stage_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_start": filter_start.date().isoformat(),
        "filter_end": filter_end.date().isoformat(),
        "configuration": config_path.relative_to(project_root).as_posix(),
        "configuration_sha256": _sha256(config_bytes),
        "inputs": {
            "regime_history": history_path.relative_to(project_root).as_posix(),
            "regime_history_sha256": _sha256(history_bytes),
            "leading_event_table": event_path.relative_to(project_root).as_posix(),
            "leading_event_table_sha256": _sha256(event_bytes),
        },
        "baseline_specification": specifications[0].to_record(),
        "sensitivity_specification_count": len(specifications) - 1,
        "transition_only_baseline_included": True,
        "checkpoint_count": len(baseline.checkpoints),
        "joint_path_rows": len(baseline.joint_paths),
        "marginal_rows": len(baseline.marginals),
        "event_audit_rows": len(baseline.event_audit),
        "likelihood_fit_rows": len(baseline.likelihood_fit_audit),
        "forecast_rows_all_specifications": len(forecasts),
        "evaluation_metric_rows": len(evaluated.metrics),
        "processed_files": processed_files,
        "published_files": [
            paths["latest"].relative_to(project_root).as_posix(),
            paths["evaluation"].relative_to(project_root).as_posix(),
            paths["published_sensitivity"].relative_to(project_root).as_posix(),
        ],
        "credential_policy": (
            "Inference consumes processed, credential-free inputs and never reads "
            "or serializes FRED_API_KEY."
        ),
    }
    _write_json(manifest, paths["manifest"])
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m01_event_driven_bayesian_filter.yaml"),
        help="Inference configuration path, relative to project root by default.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments and publish Model 01 inference artifacts."""
    args = _parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    outputs = build_inference(
        project_root=project_root,
        config_path=config_path.resolve(),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
