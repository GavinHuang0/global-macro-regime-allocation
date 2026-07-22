"""Build Model 02's event-driven four-month Gaussian Bayesian filter.

The command verifies the score, probability-map, and release-evidence
manifests before reading any modeled input.  It then prepares release-block
vectors, fits every observation equation with a strict expanding information
cutoff, replays an evidence filter and an otherwise identical transition-only
baseline, conditions completed composite scores exactly, and maps the latent
score posterior into soft growth/inflation quadrants only at reporting time.

Detailed checkpoints, Gaussian moments, event decisions, causal predictive
residuals, model fits, evaluation rows, and dependence diagnostics are retained
under ``data/processed``.  Compact histories and the latest current-month
posterior are published under ``results/published``.  This stage performs no
network access and never reads an API credential.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
from io import BytesIO
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from regime_allocation.data.dataset_acquisition import sha256
from regime_allocation.models.m02_soft_composite.dependence_diagnostics import (
    INDEPENDENCE_CAVEAT,
    ResidualColumnSpec,
    build_release_block_dependence_report,
)
from regime_allocation.models.m02_soft_composite.probability_map import (
    REGIME_LABELS,
    REGIME_ORDER,
)
from regime_allocation.models.m02_soft_composite.walkforward import (
    GaussianWalkForwardResult,
    prepare_observation_data,
    run_event_driven_filter,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "event_driven_linear_gaussian_filter"
EXPECTED_OBSERVATION_MODELS = {
    "weekly_labor_stress",
    "monthly_labor_demand",
    "consumer_demand",
    "housing_activity",
    "business_investment",
    "inflation_expectations",
    "inflation_input_costs",
}


def _project_path(project_root: Path, configured: object) -> Path:
    """Resolve a configured path and reject paths outside the project root."""

    root = project_root.resolve()
    candidate = (root / str(configured)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"configured path leaves the project root: {configured}")
    return candidate


def _output_path(namespace: Path, configured: object) -> Path:
    """Resolve an output filename without allowing it to leave its namespace."""

    root = namespace.resolve()
    candidate = (root / str(configured)).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"configured output leaves its namespace: {configured}")
    return candidate


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load and validate the frozen inference-stage architecture contract."""

    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("Model 02 inference configuration must be a mapping")
    if config.get("schema_version") != 1:
        raise ValueError("Model 02 inference schema_version must be 1")
    if config.get("model_id") != MODEL_ID or config.get("stage_id") != STAGE_ID:
        raise ValueError("Model 02 inference model_id or stage_id is unexpected")
    for section in (
        "sources",
        "state",
        "transition",
        "mapping",
        "emissions",
        "observation_models",
        "calendar",
        "evaluation",
        "dependence",
        "outputs",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Model 02 inference {section} must be a mapping")
    if set(config["observation_models"]) != EXPECTED_OBSERVATION_MODELS:
        raise ValueError("Model 02 observation-model set has changed")
    state = config["state"]
    if (
        list(state.get("axes", [])) != ["growth_score", "inflation_score"]
        or int(state.get("path_months", 0)) != 4
        or int(state.get("dimension", 0)) != 8
        or state.get("exact_score_policy")
        != "zero_noise_end_of_day_conditioning"
    ):
        raise ValueError("Model 02 continuous-state contract has changed")
    transition = config["transition"]
    if (
        transition.get("model") != "expanding_var1_ols"
        or int(transition.get("minimum_training_pairs", 0)) < 5
        or transition.get("fit_cutoff")
        != "pair_available_strictly_before_month_roll"
    ):
        raise ValueError("Model 02 transition contract has changed")
    emissions = config["emissions"]
    grid = [float(value) for value in emissions.get("lambda_grid", [])]
    if (
        emissions.get("family") != "multivariate_gaussian"
        or emissions.get("state_reference_policy")
        != "event_reference_month_without_silent_retargeting"
        or not grid
        or min(grid) <= 0.0
        or any(right <= left for left, right in zip(grid, grid[1:]))
    ):
        raise ValueError("Model 02 emission contract has changed")
    if int(config["mapping"].get("joint_path_samples", 0)) < 256:
        raise ValueError("joint_path_samples must be at least 256")
    samples = int(config["mapping"]["joint_path_samples"])
    if samples & (samples - 1):
        raise ValueError("joint_path_samples must be a power of two")
    dependence = config["dependence"]
    if (
        dependence.get("residual_source")
        != "causal_shared_pre_release_whitened_predictive_innovations"
        or dependence.get("weekly_claims_monthly_aggregation") != "mean"
        or list(dependence.get("cross_block_statistics", []))
        != ["pearson", "spearman"]
        or dependence.get("multiple_testing") != "benjamini_hochberg"
        or list(dependence.get("weekly_serial_lags", [])) != [1, 4, 8]
        or list(dependence.get("monthly_serial_lags", [])) != [1, 3, 6]
    ):
        raise ValueError("Model 02 dependence-diagnostic contract has changed")
    weekly_ids = dependence.get("weekly_response_ids", [])
    if (
        isinstance(weekly_ids, (str, bytes))
        or not weekly_ids
        or any(not str(value).strip() for value in weekly_ids)
    ):
        raise ValueError("weekly_response_ids must contain nonempty identifiers")
    cross_lags = [int(value) for value in dependence.get("cross_block_lags", [])]
    if (
        not cross_lags
        or len(cross_lags) != len(set(cross_lags))
        or any(int(raw) != raw for raw in dependence.get("cross_block_lags", []))
    ):
        raise ValueError("cross_block_lags must contain distinct integer offsets")
    if int(dependence.get("minimum_pair_observations", 0)) < 3:
        raise ValueError("minimum_pair_observations must be at least three")
    significance = float(dependence.get("serial_significance_level", math.nan))
    if not math.isfinite(significance) or not 0.0 < significance < 1.0:
        raise ValueError("serial_significance_level must lie strictly between zero and one")
    evaluation = config["evaluation"]
    if (
        evaluation.get("primary_score_at")
        != "start_of_availability_day_after_calendar_roll_but_before_any_same_day_release"
        or evaluation.get("sensitivity_score_at")
        != "after_same_day_release_blocks_before_exact_score"
    ):
        raise ValueError("Model 02 evaluation-timing contract has changed")
    required_outputs = {
        "processed_dir",
        "checkpoint_index",
        "joint_gaussian_checkpoints",
        "marginal_checkpoints",
        "joint_path_checkpoints",
        "event_update_audit",
        "observation_preparation_audit",
        "exact_score_audit",
        "transition_fit_audit",
        "emission_fit_audit",
        "evaluation_rows",
        "evaluation_summary",
        "predictive_residuals",
        "dependence_monthly_residuals",
        "dependence_pairwise",
        "dependence_serial",
        "manifest",
        "published_dir",
        "latest_posterior",
        "latest_joint_path_probabilities",
        "monthly_current_posteriors",
        "published_emission_summary",
        "published_evaluation_summary",
        "published_dependence_pairwise",
        "published_dependence_serial",
        "published_dependence_summary",
        "published_filter_summary",
    }
    if not required_outputs.issubset(config["outputs"]):
        missing = required_outputs.difference(config["outputs"])
        raise ValueError("Model 02 inference outputs omit: " + ", ".join(sorted(missing)))
    for key in ("processed_dir", "manifest", "published_dir"):
        if "m02" not in str(config["outputs"][key]):
            raise ValueError("every Model 02 inference namespace must contain m02")
    return config, raw


def _manifest_file_hash(manifest: Mapping[str, Any], relative_path: str) -> str:
    """Return one generated-file hash across the repository manifest schemas."""

    candidates: list[Mapping[str, Any]] = []
    for key in ("generated_file_hashes", "generated_files"):
        values = manifest.get(key, [])
        if isinstance(values, list):
            candidates.extend(item for item in values if isinstance(item, Mapping))
    matches = [
        str(item["sha256"])
        for item in candidates
        if str(item.get("path")) == relative_path and item.get("sha256")
    ]
    if len(matches) != 1:
        raise ValueError(f"upstream manifest does not uniquely identify {relative_path}")
    return matches[0]


def _verified_file(project_root: Path, relative_path: str, digest: str) -> bytes:
    """Read one upstream file only after its manifest hash matches."""

    path = _project_path(project_root, relative_path)
    payload = path.read_bytes()
    if sha256(payload) != digest:
        raise ValueError(f"input hash mismatch: {relative_path}")
    return payload


def _write_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    """Atomically write CSV, including explicit gzip for ``.csv.gz`` outputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d",
        compression=compression,
    )
    temporary.replace(path)


def _write_jsonl(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write one stable JSON object per fit-audit row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(
                json.dumps(
                    _json_safe(record),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    temporary.replace(path)


def _json_safe(value: object) -> object:
    """Recursively convert model output into strict, portable JSON values."""

    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(payload: Mapping[str, object], path: Path) -> None:
    """Atomically write standards-compliant JSON with no NaN literals."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _dependence_residuals(residuals: pd.DataFrame) -> pd.DataFrame:
    """Build the diagnostics schema without ambiguous duplicate renames."""

    output_columns = [
        "block_id",
        "model_id",
        "response_id",
        "reference_month",
        "observation_date",
        "validation_available_at",
        "standardized_residual",
    ]
    required = {
        "economic_block",
        "observation_model_id",
        "response_id",
        "reference_month",
        "release_date",
        "whitened_innovation",
    }
    if residuals.empty:
        return pd.DataFrame(columns=output_columns)
    missing = required.difference(residuals.columns)
    if missing:
        raise ValueError(
            "predictive residuals omit diagnostics columns: "
            + ", ".join(sorted(missing))
        )
    return pd.DataFrame(
        {
            "block_id": residuals["economic_block"],
            "model_id": residuals["observation_model_id"],
            "response_id": residuals["response_id"],
            "reference_month": residuals["reference_month"],
            "observation_date": (
                residuals["observation_date"]
                if "observation_date" in residuals
                else residuals["reference_month"]
            ),
            "validation_available_at": residuals["release_date"],
            "standardized_residual": residuals["whitened_innovation"],
        }
    )


def _evaluation_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired pre-exact score and quadrant forecast diagnostics."""

    if rows.empty:
        return pd.DataFrame()
    if "evaluation_checkpoint" not in rows:
        raise ValueError("evaluation rows omit the information-timing checkpoint")
    records: list[dict[str, object]] = []
    metric_columns = (
        "score_center_negative_log_predictive_density",
        "quadrant_cross_entropy_to_exact_score_map",
        "quadrant_brier_distance_to_exact_score_map",
        "quadrant_kl_divergence_to_exact_score_map",
    )
    for (checkpoint, variant), group in rows.groupby(
        ["evaluation_checkpoint", "filter_variant"], sort=True
    ):
        record: dict[str, object] = {
            "evaluation_checkpoint": str(checkpoint),
            "comparison": str(variant),
            "evaluation_months": int(group["reference_month"].nunique()),
            "evaluation_rows": int(len(group)),
            "growth_score_rmse": float(math.sqrt(group["growth_squared_error"].mean())),
            "inflation_score_rmse": float(
                math.sqrt(group["inflation_squared_error"].mean())
            ),
        }
        for column in metric_columns:
            if column in group:
                record[f"mean_{column}"] = float(group[column].mean())
                record[f"median_{column}"] = float(group[column].median())
                record[f"nonmissing_{column}"] = int(group[column].notna().sum())
        records.append(record)
    base_records = list(records)
    for checkpoint, checkpoint_rows in pd.DataFrame.from_records(base_records).groupby(
        "evaluation_checkpoint", sort=True
    ):
        if not {"evidence_filter", "transition_only"}.issubset(
            set(checkpoint_rows["comparison"])
        ):
            continue
        indexed = checkpoint_rows.set_index("comparison")
        evidence = indexed.loc["evidence_filter"]
        baseline = indexed.loc["transition_only"]
        paired: dict[str, object] = {
            "evaluation_checkpoint": str(checkpoint),
            "comparison": "evidence_filter_minus_transition_only",
            "evaluation_months": int(evidence["evaluation_months"]),
            "evaluation_rows": int(evidence["evaluation_rows"]),
        }
        for column in checkpoint_rows.columns:
            if column in {"comparison", "evaluation_checkpoint"} or column.startswith(
                "nonmissing_"
            ):
                continue
            if column in paired or column in {"evaluation_months", "evaluation_rows"}:
                continue
            if pd.notna(evidence.get(column)) and pd.notna(baseline.get(column)):
                paired[column] = float(evidence[column] - baseline[column])
        records.append(paired)
    return pd.DataFrame.from_records(records)


def _emission_summary(fits: pd.DataFrame) -> pd.DataFrame:
    """Flatten the latest causal fit for each observation model and response."""

    if fits.empty:
        return pd.DataFrame()
    latest = fits.sort_values(["fit_date", "fit_id"]).groupby(
        "observation_model_id", sort=True, as_index=False
    ).tail(1)
    records: list[dict[str, object]] = []
    for row in latest.itertuples(index=False):
        audit = json.loads(row.fit_audit_json)
        loadings = audit["state_loadings"]
        controls = audit["control_loadings"]
        covariance = audit["residual_covariance"]
        for response in audit["specification"]["response_names"]:
            records.append(
                {
                    "observation_model_id": row.observation_model_id,
                    "response_name": response,
                    "fit_date": row.fit_date,
                    "training_count": int(row.training_count),
                    "selected_base_lambda": float(row.selected_base_lambda),
                    "active_controls": row.active_controls,
                    "dropped_unidentified_controls": row.dropped_unidentified_controls,
                    "intercept": float(audit["intercept"][response]),
                    "growth_loading": float(loadings[response]["growth_score"]),
                    "inflation_loading": float(loadings[response]["inflation_score"]),
                    "control_loadings_json": json.dumps(
                        controls.get(response, {}), sort_keys=True
                    ),
                    "residual_variance": float(covariance[response][response]),
                    "ledoit_wolf_shrinkage": float(row.ledoit_wolf_shrinkage),
                }
            )
    return pd.DataFrame.from_records(records)


def _published_monthly_posteriors(result: GaussianWalkForwardResult) -> pd.DataFrame:
    """Return evidence-filter current-month marginals at month-end and latest."""

    keep = result.marginals.loc[
        result.marginals["filter_variant"].eq("evidence_filter")
        & result.marginals["relative_month"].eq(0)
        & result.marginals["checkpoint_type"].isin(["month_end", "latest"])
    ].copy()
    return keep.sort_values(["as_of_date", "checkpoint_type"]).reset_index(drop=True)


def _dependence_summary(report: Any) -> dict[str, object]:
    """Create a compact public interpretation of dependence diagnostics."""

    pairwise = report.cross_block_tests
    valid = pairwise.loc[pairwise["status"].eq("ok")].copy()
    significant = valid.loc[valid["q_value"] < 0.05] if "q_value" in valid else valid.iloc[0:0]
    serial = report.serial_tests
    serial_q = "q_value" if "q_value" in serial else "p_value"
    serial_valid = serial.loc[serial["status"].eq("ok")].copy()
    serial_significant = serial_valid.loc[serial_valid[serial_q] < 0.05]
    strongest: list[dict[str, object]] = []
    if not valid.empty:
        columns = [
            "left_block_id",
            "left_model_id",
            "left_response_id",
            "right_block_id",
            "right_model_id",
            "right_response_id",
            "lag_months",
            "statistic",
            "correlation",
            "sample_count",
            "p_value",
            "q_value",
        ]
        strongest = (
            valid.assign(abs_correlation=valid["correlation"].abs())
            .sort_values("abs_correlation", ascending=False)
            .head(10)
            .loc[:, [column for column in columns if column in valid]]
            .to_dict(orient="records")
        )
    return {
        "cross_model_test_rows": int(len(pairwise)),
        "valid_cross_model_tests": int(len(valid)),
        "bh_q_below_0_05": int(len(significant)),
        "serial_test_rows": int(len(serial)),
        "valid_serial_tests": int(len(serial_valid)),
        "serial_adjusted_q_below_0_05": int(len(serial_significant)),
        "strongest_absolute_correlations": strongest,
        "interpretation_caveat": INDEPENDENCE_CAVEAT,
        "p_value_caveat": (
            "Correlation p-values are exploratory when residual serial dependence "
            "is present; effect sizes and coverage remain primary."
        ),
    }


def _latest_payload(
    result: GaussianWalkForwardResult,
    dependence_summary: Mapping[str, object],
) -> dict[str, object]:
    """Serialize the latest authoritative Gaussian state and quadrant readout."""

    checkpoint = result.checkpoints.iloc[-1]
    checkpoint_id = str(checkpoint["checkpoint_id"])
    marginal_rows = result.marginals.loc[
        result.marginals["checkpoint_id"].eq(checkpoint_id)
    ]
    variants: dict[str, object] = {}
    for variant, group in marginal_rows.groupby("filter_variant", sort=True):
        state = result.latest_states[str(variant)]
        months: list[dict[str, object]] = []
        for row in group.sort_values("reference_month").itertuples(index=False):
            probabilities = [
                {
                    "regime_id": regime,
                    "regime_label": REGIME_LABELS[regime],
                    "probability": (
                        None
                        if not hasattr(row, f"probability_{regime}")
                        or pd.isna(getattr(row, f"probability_{regime}"))
                        else float(getattr(row, f"probability_{regime}"))
                    ),
                }
                for regime in REGIME_ORDER
            ]
            months.append(
                {
                    "reference_month": pd.Timestamp(row.reference_month).date().isoformat(),
                    "relative_month": int(row.relative_month),
                    "score_center_mean": [float(row.growth_mean), float(row.inflation_mean)],
                    "latent_covariance": [
                        [
                            float(row.growth_latent_variance),
                            float(row.growth_inflation_latent_covariance),
                        ],
                        [
                            float(row.growth_inflation_latent_covariance),
                            float(row.inflation_latent_variance),
                        ],
                    ],
                    "score_center_is_exact": bool(row.exact_score_center),
                    "quadrants": probabilities,
                }
            )
        variants[str(variant)] = {
            "reference_months": [month.date().isoformat() for month in state.reference_months],
            "joint_score_center_mean": state.mean.tolist(),
            "joint_score_center_covariance": state.covariance.tolist(),
            "exact_mask": list(state.exact_mask),
            "monthly_marginals": months,
        }
    latest_paths = result.joint_paths.loc[
        result.joint_paths["checkpoint_id"].eq(checkpoint_id)
        & result.joint_paths["filter_variant"].eq("evidence_filter")
    ].copy()
    top_paths = []
    if not latest_paths.empty:
        for row in latest_paths.nlargest(10, "probability").itertuples(index=False):
            top_paths.append(
                {
                    "regimes": [row.regime_0, row.regime_1, row.regime_2, row.regime_3],
                    "probability": float(row.probability),
                }
            )
    current_month_start = result.latest_states["evidence_filter"].reference_months[-1]
    recent_events = result.event_audit.loc[
        pd.to_datetime(result.event_audit["release_date"]).between(
            current_month_start, result.replay_end, inclusive="both"
        )
    ] if not result.event_audit.empty else result.event_audit
    return {
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "as_of_date": result.replay_end.date().isoformat(),
        "checkpoint_id": checkpoint_id,
        "checkpoint_type": str(checkpoint["checkpoint_type"]),
        "mapping": {
            "status": str(checkpoint["mapping_status"]),
            "proxy_reference_month": (
                None
                if pd.isna(checkpoint["mapping_reference_month"])
                else pd.Timestamp(checkpoint["mapping_reference_month"]).date().isoformat()
            ),
            "available_at": (
                None
                if pd.isna(checkpoint["mapping_available_at"])
                else pd.Timestamp(checkpoint["mapping_available_at"]).date().isoformat()
            ),
            "interpretation": (
                "Mapping covariance is reporting uncertainty for U=Z+epsilon_map; "
                "it is not propagated as uncertainty in an exact released score center."
            ),
        },
        "variants": variants,
        "top_joint_quadrant_paths": top_paths,
        "latest_joint_path_sobol_samples": (
            None if latest_paths.empty else int(latest_paths.iloc[0]["sobol_samples"])
        ),
        "latest_joint_path_maximum_marginal_error": (
            None
            if latest_paths.empty
            else float(latest_paths.iloc[0]["maximum_marginal_probability_error"])
        ),
        "current_month_event_status_counts": {
            str(key): int(value)
            for key, value in recent_events.get("update_status", pd.Series(dtype=str))
            .value_counts()
            .items()
        },
        "dependence_summary": dict(dependence_summary),
        "joint_path_assumption": (
            "Baseline Sobol path readout treats monthly mapping perturbations as "
            "independent; Gaussian score-center dynamics retain cross-month dependence."
        ),
    }


def _filter_summary(
    result: GaussianWalkForwardResult,
    prepared: Any,
    evaluation: pd.DataFrame,
    dependence: Mapping[str, object],
) -> dict[str, object]:
    """Return public coverage, timing, and model-readiness diagnostics."""

    status_counts = (
        {
            str(key): int(value)
            for key, value in result.event_audit["update_status"].value_counts().items()
        }
        if not result.event_audit.empty
        else {}
    )
    first_fits = {}
    if not result.emission_fit_audit.empty:
        first_fits = {
            str(key): pd.Timestamp(value).date().isoformat()
            for key, value in result.emission_fit_audit.groupby(
                "observation_model_id"
            )["fit_date"].min().items()
        }
    return {
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "replay_start": result.initial_date.date().isoformat(),
        "replay_end": result.replay_end.date().isoformat(),
        "checkpoint_rows": int(len(result.checkpoints)),
        "joint_gaussian_rows": int(len(result.joint_gaussians)),
        "marginal_rows": int(len(result.marginals)),
        "joint_path_rows": int(len(result.joint_paths)),
        "event_rows_scored_or_audited": int(len(result.event_audit)),
        "event_status_counts": status_counts,
        "exact_score_audit_rows": int(len(result.exact_score_audit)),
        "transition_fits": int(len(result.transition_fit_audit)),
        "emission_fits": int(len(result.emission_fit_audit)),
        "first_emission_fit_dates": first_fits,
        "observation_preparation": prepared.preparation_audit.to_dict(orient="records"),
        "evaluation": evaluation.to_dict(orient="records"),
        "dependence": dict(dependence),
        "jolts_timing_note": (
            "Monthly labor-demand releases retain their actual reference month. "
            "Many arrive after that month's exact score and are intentionally "
            "predictive diagnostics/no-ops rather than silently retargeted signals."
        ),
    }


def build_inference(
    *,
    project_root: Path,
    config_path: Path,
    replay_end: object | None = None,
) -> dict[str, Path]:
    """Verify inputs, run Model 02 inference, and publish audited artifacts."""

    project_root = project_root.resolve()
    config_path = _project_path(project_root, config_path)
    config, config_bytes = _load_config(config_path)
    sources = config["sources"]
    outputs = config["outputs"]

    manifest_specs = (
        ("score", sources["score_manifest"], MODEL_ID, None),
        ("probability_map", sources["probability_map_manifest"], MODEL_ID, "m02_probability_map"),
        ("evidence", sources["evidence_manifest"], MODEL_ID, "m02_release_evidence"),
    )
    manifests: dict[str, dict[str, Any]] = {}
    manifest_bytes: dict[str, bytes] = {}
    manifest_paths: dict[str, Path] = {}
    for label, configured, expected_model, expected_stage in manifest_specs:
        path = _project_path(project_root, configured)
        payload = path.read_bytes()
        manifest = json.loads(payload)
        if manifest.get("model_id") != expected_model:
            raise ValueError(f"{label} manifest has the wrong model_id")
        if expected_stage is not None and manifest.get("stage_id") != expected_stage:
            raise ValueError(f"{label} manifest has the wrong stage_id")
        manifests[label] = manifest
        manifest_bytes[label] = payload
        manifest_paths[label] = path
    expected_score_lineage = manifests["probability_map"].get("score_manifest_sha256")
    if expected_score_lineage != sha256(manifest_bytes["score"]):
        raise ValueError("probability-map lineage does not match the score manifest")

    input_specs = (
        ("score", str(sources["score_features"]), manifests["score"]),
        ("probability_map", str(sources["probability_map"]), manifests["probability_map"]),
        ("evidence", str(sources["evidence_events"]), manifests["evidence"]),
    )
    input_bytes: dict[str, bytes] = {}
    for label, relative, manifest in input_specs:
        digest = _manifest_file_hash(manifest, relative)
        input_bytes[label] = _verified_file(project_root, relative, digest)

    # Parse the exact bytes that were hash-verified above.  Reopening the files
    # here would create a time-of-check/time-of-use gap in the lineage contract.
    scores = pd.read_csv(BytesIO(input_bytes["score"]))
    mapping = pd.read_csv(BytesIO(input_bytes["probability_map"]))
    events = pd.read_csv(BytesIO(input_bytes["evidence"]))
    prepared = prepare_observation_data(events, scores, config)
    result = run_event_driven_filter(
        scores,
        mapping,
        prepared,
        config,
        replay_end=replay_end,
    )

    residuals = _dependence_residuals(result.predictive_residuals)
    dependence_config = config["dependence"]
    report = build_release_block_dependence_report(
        residuals,
        weekly_response_ids=tuple(dependence_config["weekly_response_ids"]),
        cross_block_lags=tuple(int(value) for value in dependence_config["cross_block_lags"]),
        minimum_pair_observations=int(
            dependence_config["minimum_pair_observations"]
        ),
        serial_significance_level=float(
            dependence_config["serial_significance_level"]
        ),
        columns=ResidualColumnSpec(observation_date="observation_date"),
    )
    evaluation = _evaluation_summary(result.evaluation_rows)
    emission_summary = _emission_summary(result.emission_fit_audit)
    dependence_summary = _dependence_summary(report)
    monthly = _published_monthly_posteriors(result)
    latest_payload = _latest_payload(result, dependence_summary)
    filter_summary = _filter_summary(
        result, prepared, evaluation, dependence_summary
    )

    processed_dir = _project_path(project_root, outputs["processed_dir"])
    published_dir = _project_path(project_root, outputs["published_dir"])
    processed_paths = {
        key: _output_path(processed_dir, outputs[key])
        for key in (
            "checkpoint_index",
            "joint_gaussian_checkpoints",
            "marginal_checkpoints",
            "joint_path_checkpoints",
            "event_update_audit",
            "observation_preparation_audit",
            "exact_score_audit",
            "transition_fit_audit",
            "emission_fit_audit",
            "evaluation_rows",
            "evaluation_summary",
            "predictive_residuals",
            "dependence_monthly_residuals",
            "dependence_pairwise",
            "dependence_serial",
        )
    }
    public_paths = {
        key: _output_path(published_dir, outputs[key])
        for key in (
            "latest_posterior",
            "latest_joint_path_probabilities",
            "monthly_current_posteriors",
            "published_emission_summary",
            "published_evaluation_summary",
            "published_dependence_pairwise",
            "published_dependence_serial",
            "published_dependence_summary",
            "published_filter_summary",
        )
    }
    manifest_path = _project_path(project_root, outputs["manifest"])
    generated = [*processed_paths.values(), *public_paths.values()]
    resolved_outputs = [path.resolve() for path in generated]
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise ValueError("configured inference outputs contain duplicate paths")
    if manifest_path in set(resolved_outputs):
        raise ValueError("manifest path collides with a generated data artifact")

    tables = {
        "checkpoint_index": result.checkpoints,
        "joint_gaussian_checkpoints": result.joint_gaussians,
        "marginal_checkpoints": result.marginals,
        "joint_path_checkpoints": result.joint_paths,
        "event_update_audit": result.event_audit,
        "observation_preparation_audit": prepared.preparation_audit,
        "exact_score_audit": result.exact_score_audit,
        "transition_fit_audit": result.transition_fit_audit,
        "evaluation_rows": result.evaluation_rows,
        "evaluation_summary": evaluation,
        "predictive_residuals": result.predictive_residuals,
        "dependence_monthly_residuals": report.monthly_residuals,
        "dependence_pairwise": report.cross_block_tests,
        "dependence_serial": report.serial_tests,
    }
    for key, frame in tables.items():
        compression = "gzip" if processed_paths[key].name.endswith(".gz") else None
        _write_csv(frame, processed_paths[key], compression=compression)
    _write_jsonl(result.emission_fit_audit, processed_paths["emission_fit_audit"])

    latest_checkpoint_id = str(result.checkpoints.iloc[-1]["checkpoint_id"])
    latest_paths = result.joint_paths.loc[
        result.joint_paths["checkpoint_id"].eq(latest_checkpoint_id)
        & result.joint_paths["filter_variant"].eq("evidence_filter")
    ].copy()
    _write_csv(latest_paths, public_paths["latest_joint_path_probabilities"])
    _write_csv(monthly, public_paths["monthly_current_posteriors"])
    _write_csv(emission_summary, public_paths["published_emission_summary"])
    _write_csv(evaluation, public_paths["published_evaluation_summary"])
    _write_csv(report.cross_block_tests, public_paths["published_dependence_pairwise"])
    _write_csv(report.serial_tests, public_paths["published_dependence_serial"])
    _write_json(latest_payload, public_paths["latest_posterior"])
    _write_json(dependence_summary, public_paths["published_dependence_summary"])
    _write_json(filter_summary, public_paths["published_filter_summary"])

    generated_hashes = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
        for path in generated
    ]
    snapshot = hashlib.sha256()
    for label in ("score", "probability_map", "evidence"):
        snapshot.update(sha256(manifest_bytes[label]).encode("ascii"))
        snapshot.update(sha256(input_bytes[label]).encode("ascii"))
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
                for package in ("numpy", "pandas", "PyYAML", "scipy", "scikit-learn")
            },
        },
        "upstream_manifests": [
            {
                "path": manifest_paths[label].relative_to(project_root).as_posix(),
                "sha256": sha256(manifest_bytes[label]),
            }
            for label in ("score", "probability_map", "evidence")
        ],
        "input_files": [
            {"path": relative, "sha256": sha256(input_bytes[label])}
            for label, relative, _ in input_specs
        ],
        "data_snapshot_sha256": snapshot.hexdigest(),
        "information_contract": {
            "transition_cutoff": "pair_available_at_strictly_before_month_roll",
            "emission_cutoff": "training_available_at_strictly_before_release",
            "exact_score_timing": "end_of_availability_day",
            "mapping_role": "reporting_readout_only",
            "event_reference_policy": "actual_event_reference_month",
            "same_day_order": "release_factors_then_exact_score",
            "primary_evaluation_cutoff": (
                "before_all_releases_on_exact_score_availability_date"
            ),
            "post_release_evaluation_role": "labeled_sensitivity_only",
        },
        "coverage": {
            "replay_start": result.initial_date.date().isoformat(),
            "replay_end": result.replay_end.date().isoformat(),
            "checkpoints": len(result.checkpoints),
            "events": len(result.event_audit),
            "exact_scores": len(result.exact_score_audit),
            "emission_fits": len(result.emission_fit_audit),
            "evaluation_rows": len(result.evaluation_rows),
        },
        "dependence_diagnostics": report.to_audit_dict(),
        "processed_files": [
            path.relative_to(project_root).as_posix() for path in processed_paths.values()
        ],
        "published_files": [
            path.relative_to(project_root).as_posix() for path in public_paths.values()
        ],
        "generated_file_hashes": generated_hashes,
        "credential_policy": "no network access and no credential read in inference stage",
    }
    _write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "latest": public_paths["latest_posterior"],
        "monthly_posteriors": public_paths["monthly_current_posteriors"],
        "evaluation": public_paths["published_evaluation_summary"],
        "dependence": public_paths["published_dependence_summary"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m02_event_driven_bayesian_filter.yaml"),
    )
    parser.add_argument(
        "--replay-end",
        default=None,
        help="Optional YYYY-MM-DD cutoff not later than available inputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments and build the complete Model 02 inference stage."""

    args = _parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    outputs = build_inference(
        project_root=project_root,
        config_path=config_path.resolve(),
        replay_end=args.replay_end,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
