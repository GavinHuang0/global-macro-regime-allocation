"""Run six causal evidence-block experiments around the frozen Model 02 baseline.

The command verifies every upstream artifact, appends candidate observation
models to an in-memory copy of the base filter configuration, and replays the
fixed ``student_t_7_combined`` specification.  Evidence-set allowlists keep the
published baseline byte-for-byte equivalent at its evaluation surface while
each candidate changes only the evidence rows declared for that experiment.

No network access or credential read occurs in this stage.  The preceding
evidence-acquisition command is the only stage that needs ``FRED_API_KEY``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from regime_allocation.cli.build_m02_inference_sensitivities import (
    _evaluation_subperiod_summary,
    _evaluation_summary,
    _latest_joint_state_rows,
    _load_yaml,
    _manifest_hash,
    _namespace_path,
    _paired_comparisons,
    _project_path,
    _verified_bytes,
    _write_csv,
    _write_json,
)
from regime_allocation.data.dataset_acquisition import sha256
from regime_allocation.models.m02_soft_composite.inference_sensitivities import (
    run_inference_sensitivities,
    variant_registry_from_config,
)
from regime_allocation.models.m02_soft_composite.dependence_diagnostics import (
    INDEPENDENCE_CAVEAT,
    ResidualColumnSpec,
    benjamini_hochberg,
    build_release_block_dependence_report,
)
from regime_allocation.models.m02_soft_composite.walkforward import (
    prepare_observation_data,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "m02_evidence_block_inference_experiments"
BASELINE_ID = "student_t_7_combined"
PRIORITY_VARIANTS = (
    "priority_01_empire_only",
    "priority_01_philadelphia_only",
    "priority_01_surveys",
    "priority_02_claims",
    "priority_03_consumer",
    "priority_04_housing",
    "priority_05_import_prices",
    "priority_06_backlog",
    "candidate_all_six",
)
LEGACY_MODELS = frozenset(
    {
        "weekly_labor_stress",
        "monthly_labor_demand",
        "consumer_demand",
        "housing_activity",
        "business_investment",
        "inflation_expectations",
        "inflation_input_costs",
    }
)

# These hashes disclose the precise implementation surface used by the replay.
IMPLEMENTATION_FILES = (
    "src/regime_allocation/cli/build_m02_evidence_experiment_inference.py",
    "src/regime_allocation/cli/build_m02_evidence_experiments.py",
    "src/regime_allocation/features/m02_evidence_experiments.py",
    "src/regime_allocation/models/m02_soft_composite/gaussian_emissions.py",
    "src/regime_allocation/models/m02_soft_composite/inference_sensitivities.py",
    "src/regime_allocation/models/m02_soft_composite/joint_filter.py",
    "src/regime_allocation/models/m02_soft_composite/partial_defining.py",
    "src/regime_allocation/models/m02_soft_composite/robust_emissions.py",
    "src/regime_allocation/models/m02_soft_composite/robust_var.py",
    "src/regime_allocation/models/m02_soft_composite/var_transition.py",
    "src/regime_allocation/models/m02_soft_composite/walkforward.py",
)


def _load_experiment_config(path: Path) -> tuple[dict[str, Any], bytes]:
    """Validate the immutable comparison contract before running anything."""

    config, raw = _load_yaml(path)
    if (
        config.get("schema_version") != 1
        or config.get("model_id") != MODEL_ID
        or config.get("stage_id") != STAGE_ID
    ):
        raise ValueError("unexpected evidence experiment configuration identity")
    for section in (
        "model_selection",
        "sources",
        "candidate_observation_models",
        "evidence_sets",
        "partial_defining_releases",
        "student_t_emissions",
        "robust_var",
        "retail_sensitivities",
        "dependence_diagnostics",
        "evaluation",
        "outputs",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"evidence inference config omits mapping section {section}")
    if not isinstance(config.get("variants"), list) or not config["variants"]:
        raise ValueError("evidence inference config requires variants")

    registry = variant_registry_from_config(config)
    actual = set(registry["variant_id"].astype(str))
    required = {"transition_only", "partial_only", BASELINE_ID, *PRIORITY_VARIANTS}
    if actual != required:
        raise ValueError("evidence experiment variant set has changed")
    selection = config["model_selection"]
    if selection.get("baseline") != BASELINE_ID:
        raise ValueError(f"frozen baseline must remain {BASELINE_ID}")
    if list(selection.get("major_benchmarks", ())) != [
        "transition_only",
        "partial_only",
    ]:
        raise ValueError("major benchmarks must remain transition_only and partial_only")

    baseline = registry.loc[registry["variant_id"].eq(BASELINE_ID)].iloc[0]
    if not (
        baseline["model_role"] == "baseline"
        and bool(baseline["non_defining_evidence"])
        and bool(baseline["partial_defining_releases"])
        and baseline["emission_family"] == "student_t_7"
        and baseline["var_method"] == "ols"
        and baseline["retail_method"] == "baseline_nominal"
        and baseline["evidence_set_id"] == "legacy_baseline"
    ):
        raise ValueError("the selected baseline specification is not frozen")
    fixed_columns = (
        "non_defining_evidence",
        "partial_defining_releases",
        "emission_family",
        "var_method",
        "retail_method",
    )
    for variant_id in PRIORITY_VARIANTS:
        candidate = registry.loc[registry["variant_id"].eq(variant_id)].iloc[0]
        if candidate["model_role"] != "sensitivity" or any(
            candidate[column] != baseline[column] for column in fixed_columns
        ):
            raise ValueError(
                f"{variant_id} must differ from the frozen baseline only by evidence set"
            )
    legacy = frozenset(
        str(value)
        for value in config["evidence_sets"]["legacy_baseline"]["observation_models"]
    )
    if legacy != LEGACY_MODELS:
        raise ValueError("legacy_baseline evidence membership has changed")
    degrees = [
        float(value)
        for value in config["student_t_emissions"]
        ["degrees_of_freedom_sensitivity"]["candidates"]
    ]
    if degrees != [7.0]:
        raise ValueError("this experiment must hold Student-t degrees of freedom at 7")

    required_outputs = {
        "processed_dir",
        "published_dir",
        "manifest",
        "evaluation_rows",
        "evaluation_summary",
        "evaluation_subperiod_summary",
        "paired_comparisons",
        "paired_block_bootstrap",
        "dependence_residuals",
        "dependence_monthly_residuals",
        "dependence_pairwise",
        "dependence_serial",
        "dependence_same_publication",
        "event_update_audit",
        "partial_defining_audit",
        "exact_score_audit",
        "transition_fit_audit",
        "transition_weight_audit",
        "emission_fit_audit",
        "tail_fold_scores",
        "hyperparameter_schedule",
        "latest_marginals",
        "model_registry",
        "profile_coverage",
        "baseline_invariance",
        "public_evaluation_summary",
        "public_evaluation_subperiod_summary",
        "public_paired_comparisons",
        "public_paired_block_bootstrap",
        "public_dependence_pairwise",
        "public_dependence_serial",
        "public_dependence_same_publication",
        "public_dependence_summary",
        "public_profile_coverage",
        "public_baseline_invariance",
        "public_model_registry",
        "public_method_summary",
    }
    missing = required_outputs.difference(config["outputs"])
    if missing:
        raise ValueError("evidence experiment outputs omit: " + ", ".join(sorted(missing)))
    return config, raw


def _load_manifest(
    path: Path,
    *,
    stage_id: str | None,
) -> tuple[dict[str, Any], bytes]:
    content = path.read_bytes()
    manifest = json.loads(content)
    wrong_stage = stage_id is not None and manifest.get("stage_id") != stage_id
    if manifest.get("model_id") != MODEL_ID or wrong_stage:
        raise ValueError(f"upstream manifest identity mismatch: {path}")
    return manifest, content


def _merge_observation_models(
    base: Mapping[str, Any],
    candidates: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an in-memory base config augmented only with new model IDs."""

    merged = deepcopy(dict(base))
    existing = dict(merged.get("observation_models", {}))
    collisions = set(existing).intersection(candidates)
    if collisions:
        raise ValueError(
            "candidate observation models overwrite the base configuration: "
            + ", ".join(sorted(collisions))
        )
    existing.update(deepcopy(dict(candidates)))
    merged["observation_models"] = existing
    return merged


def _normalize_comparison_frame(
    frame: pd.DataFrame,
    *,
    variant_column: str,
    variant_id: str,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    selected = frame.loc[frame[variant_column].astype(str).eq(variant_id)].copy()
    if selected.empty:
        raise ValueError(f"comparison artifact lacks {variant_id}")
    for column in selected.columns:
        if column.endswith("_date") or column.endswith("_month") or column.endswith("_at"):
            selected[column] = pd.to_datetime(selected[column], errors="coerce")
    return selected.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)


def _artifact_invariance(
    current: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    artifact: str,
    variant_column: str,
    key_columns: tuple[str, ...],
    ignored_columns: frozenset[str] = frozenset(),
    tolerance: float = 1.0e-12,
) -> list[dict[str, object]]:
    """Compare one candidate replay surface to the frozen selected baseline."""

    left = _normalize_comparison_frame(
        current,
        variant_column=variant_column,
        variant_id=BASELINE_ID,
        key_columns=key_columns,
    )
    right = _normalize_comparison_frame(
        frozen,
        variant_column=variant_column,
        variant_id=BASELINE_ID,
        key_columns=key_columns,
    )
    ignored = {variant_column, "model_role", *ignored_columns}
    left_columns = set(left.columns).difference(ignored)
    right_columns = set(right.columns).difference(ignored)
    rows: list[dict[str, object]] = [
        {
            "artifact": artifact,
            "check": "row_count",
            "column": "",
            "current": len(left),
            "frozen": len(right),
            "maximum_absolute_difference": 0.0,
            "passed": len(left) == len(right),
        },
        {
            "artifact": artifact,
            "check": "column_set",
            "column": "",
            "current": "|".join(sorted(left_columns)),
            "frozen": "|".join(sorted(right_columns)),
            "maximum_absolute_difference": 0.0,
            "passed": left_columns == right_columns,
        },
    ]
    if len(left) != len(right) or left_columns != right_columns:
        return rows

    for column in sorted(left_columns):
        first = left[column]
        second = right[column]
        if pd.api.types.is_numeric_dtype(first) and pd.api.types.is_numeric_dtype(second):
            a = pd.to_numeric(first, errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(second, errors="coerce").to_numpy(dtype=float)
            both_missing = np.isnan(a) & np.isnan(b)
            finite = np.isfinite(a) & np.isfinite(b)
            mismatch = ~(both_missing | finite)
            maximum = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            passed = not mismatch.any() and bool(
                np.allclose(a[finite], b[finite], atol=tolerance, rtol=tolerance)
            )
        elif pd.api.types.is_datetime64_any_dtype(first):
            maximum = 0.0
            a = first.to_numpy(dtype="datetime64[ns]")
            b = second.to_numpy(dtype="datetime64[ns]")
            passed = bool(np.array_equal(a, b))
        else:
            maximum = 0.0
            passed = first.fillna("<NA>").astype(str).equals(
                second.fillna("<NA>").astype(str)
            )
        rows.append(
            {
                "artifact": artifact,
                "check": "column_values",
                "column": column,
                "current": "",
                "frozen": "",
                "maximum_absolute_difference": maximum,
                "passed": passed,
            }
        )
    return rows


def _profile_coverage(
    prepared: Any,
    event_audit: pd.DataFrame,
    registry: pd.DataFrame,
    evidence_sets: Mapping[str, Any],
) -> pd.DataFrame:
    """Summarize data availability and realized updates by model and variant."""

    preparation = prepared.preparation_audit.copy()
    timing = (
        prepared.events.groupby("observation_model_id", as_index=False)
        .agg(
            first_live_release=("release_date", "min"),
            last_live_release=("release_date", "max"),
            first_target_month=("reference_month", "min"),
            last_target_month=("reference_month", "max"),
        )
    )
    preparation = preparation.merge(
        timing, on="observation_model_id", how="left", validate="one_to_one"
    )
    models_by_set = {
        str(set_id): frozenset(str(value) for value in declaration["observation_models"])
        for set_id, declaration in evidence_sets.items()
    }
    used_by = {}
    for model_id in preparation["observation_model_id"].astype(str):
        using = registry.loc[
            registry["evidence_set_id"].map(
                lambda set_id: model_id in models_by_set[str(set_id)]
            ),
            "variant_id",
        ].astype(str)
        used_by[model_id] = "|".join(using)
    preparation["variants_using_model"] = preparation["observation_model_id"].map(used_by)

    if event_audit.empty:
        preparation["audit_rows"] = 0
        preparation["applied_updates"] = 0
        return preparation
    audit = (
        event_audit.groupby(["variant_id", "observation_model_id"], as_index=False)
        .agg(
            audit_rows=("update_status", "size"),
            applied_updates=("update_status", lambda values: int(values.eq("applied").sum())),
            fit_errors=(
                "update_status",
                lambda values: int(values.astype(str).str.startswith("fit_error:").sum()),
            ),
            median_event_weight=("event_weight", "median"),
        )
    )
    return audit.merge(
        preparation,
        on="observation_model_id",
        how="left",
        validate="many_to_one",
    ).sort_values(["variant_id", "observation_model_id"], kind="mergesort")


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Return Holm family-wise-error adjusted p-values with monotonicity."""

    values = pd.to_numeric(p_values, errors="coerce")
    valid = values.dropna().sort_values(kind="mergesort")
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        candidate = min(1.0, float(value) * (count - rank))
        running = max(running, candidate)
        adjusted.loc[index] = running
    return adjusted


def _paired_block_bootstrap(
    evaluations: pd.DataFrame,
    *,
    initial_date: pd.Timestamp,
    burn_in_months: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Quantify paired monthly uncertainty using a circular block bootstrap.

    The observed delta is always ``candidate - baseline``. A negative delta is
    favorable for proper losses, whereas a positive delta is favorable for
    hard-quadrant accuracy. Twelve-month circular blocks preserve seasonality
    and short-run serial dependence without using future data for model fits.
    """

    bootstrap = config["evaluation"]["paired_block_bootstrap"]
    replications = int(bootstrap["replications"])
    block_length = int(bootstrap["block_length_months"])
    confidence = float(bootstrap["confidence_level"])
    seed = int(bootstrap["random_seed"])
    if replications < 1000 or block_length < 2 or not 0.0 < confidence < 1.0:
        raise ValueError("paired block-bootstrap settings are too weak or invalid")
    directions = {str(key): str(value) for key, value in bootstrap["metrics"].items()}
    if set(directions.values()).difference({"lower", "higher"}):
        raise ValueError("bootstrap metric direction must be lower or higher")

    frame = evaluations.copy()
    frame["reference_month"] = pd.to_datetime(frame["reference_month"])
    eligible = (initial_date.to_period("M") + int(burn_in_months)).to_timestamp()
    frame = frame.loc[frame["reference_month"].ge(eligible)].copy()
    sample_masks = {
        "full_sample": pd.Series(True, index=frame.index),
        "excluding_2020_03_through_2020_05": ~frame["reference_month"].between(
            "2020-03-01", "2020-05-01", inclusive="both"
        ),
    }
    requested_samples = tuple(str(value) for value in bootstrap["samples"])
    if set(requested_samples).difference(sample_masks):
        raise ValueError("paired bootstrap requests an unsupported sample")

    records: list[dict[str, object]] = []
    key = ["reference_month", "evaluation_checkpoint"]
    baseline = frame.loc[frame["filter_variant"].eq(BASELINE_ID)]
    alpha = (1.0 - confidence) / 2.0
    for sample_id in requested_samples:
        allowed = set(frame.loc[sample_masks[sample_id], "reference_month"])
        base_sample = baseline.loc[baseline["reference_month"].isin(allowed)]
        for variant_id in (str(value) for value in bootstrap["variants"]):
            candidate = frame.loc[
                frame["filter_variant"].eq(variant_id)
                & frame["reference_month"].isin(allowed)
            ]
            for checkpoint in config["evaluation"]["information_stage_checkpoints"]:
                left = candidate.loc[candidate["evaluation_checkpoint"].eq(checkpoint)]
                right = base_sample.loc[
                    base_sample["evaluation_checkpoint"].eq(checkpoint)
                ]
                for metric, direction in directions.items():
                    paired = left.loc[:, [*key, metric]].merge(
                        right.loc[:, [*key, metric]],
                        on=key,
                        how="inner",
                        suffixes=("_candidate", "_baseline"),
                        validate="one_to_one",
                    )
                    # Circular blocks must follow calendar order.  Upstream
                    # evaluation rows are currently chronological, but make
                    # that contract explicit so a future row-order change
                    # cannot silently turn the block bootstrap into an IID-like
                    # resample of arbitrarily ordered observations.
                    paired = paired.sort_values(key, kind="stable").reset_index(
                        drop=True
                    )
                    candidate_values = pd.to_numeric(
                        paired[f"{metric}_candidate"], errors="coerce"
                    ).astype(float)
                    baseline_values = pd.to_numeric(
                        paired[f"{metric}_baseline"], errors="coerce"
                    ).astype(float)
                    delta = (candidate_values - baseline_values).to_numpy(dtype=float)
                    delta = delta[np.isfinite(delta)]
                    if not len(delta):
                        continue
                    identity = (
                        f"{seed}|{sample_id}|{variant_id}|{checkpoint}|{metric}"
                    ).encode("utf-8")
                    local_seed = int.from_bytes(
                        hashlib.sha256(identity).digest()[:8], "little"
                    )
                    rng = np.random.default_rng(local_seed)
                    blocks = int(math.ceil(len(delta) / block_length))
                    starts = rng.integers(0, len(delta), size=(replications, blocks))
                    offsets = np.arange(block_length, dtype=int)
                    indices = (starts[:, :, None] + offsets[None, None, :]) % len(delta)
                    indices = indices.reshape(replications, -1)[:, : len(delta)]
                    boot_means = delta[indices].mean(axis=1)
                    observed = float(delta.mean())
                    centered = delta - observed
                    null_means = centered[indices].mean(axis=1)
                    p_value = float(
                        (1 + np.count_nonzero(np.abs(null_means) >= abs(observed)))
                        / (replications + 1)
                    )
                    probability_better = float(
                        np.mean(boot_means < 0.0)
                        if direction == "lower"
                        else np.mean(boot_means > 0.0)
                    )
                    records.append(
                        {
                            "evaluation_sample": sample_id,
                            "filter_variant": variant_id,
                            "comparison_reference": BASELINE_ID,
                            "evaluation_checkpoint": checkpoint,
                            "metric": metric,
                            "better_direction": direction,
                            "paired_months": len(delta),
                            "mean_candidate_minus_baseline": observed,
                            "bootstrap_confidence_level": confidence,
                            "bootstrap_lower": float(np.quantile(boot_means, alpha)),
                            "bootstrap_upper": float(
                                np.quantile(boot_means, 1.0 - alpha)
                            ),
                            "bootstrap_probability_candidate_better": probability_better,
                            "centered_two_sided_p_value": p_value,
                            "block_length_months": block_length,
                            "replications": replications,
                            "random_seed": seed,
                        }
                    )
    result = pd.DataFrame.from_records(records)
    family = frozenset(str(value) for value in bootstrap["holm_family"])
    result["holm_family_member"] = result["filter_variant"].isin(family)
    result["holm_adjusted_p_value"] = np.nan
    grouping = ["evaluation_sample", "evaluation_checkpoint", "metric"]
    for _, indices in result.loc[result["holm_family_member"]].groupby(grouping).groups.items():
        result.loc[indices, "holm_adjusted_p_value"] = _holm_adjust(
            result.loc[indices, "centered_two_sided_p_value"]
        )
    return result.sort_values(
        ["evaluation_sample", "evaluation_checkpoint", "metric", "filter_variant"],
        kind="mergesort",
    ).reset_index(drop=True)


def _same_publication_dependence(
    residuals: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Test ICSA/CCSA residual association on exact common publication dates."""

    pair = config["dependence_diagnostics"]["exact_same_publication_pair"]
    table = residuals.copy()
    table["validation_available_at"] = pd.to_datetime(
        table["validation_available_at"], errors="coerce"
    ).dt.normalize()

    def series(model_key: str, response_key: str, value_name: str) -> pd.DataFrame:
        selected = table.loc[
            table["model_id"].astype(str).eq(str(pair[model_key]))
            & table["response_id"].astype(str).eq(str(pair[response_key]))
        ].copy()
        if selected.empty:
            raise ValueError(f"same-publication dependence has no {model_key} rows")
        # A provider can publish a catch-up batch for multiple weekly reference
        # periods on one date. Its mean is the publication-level diagnostic unit.
        return (
            selected.groupby("validation_available_at", as_index=False)
            .agg(
                **{
                    value_name: ("standardized_residual", "mean"),
                    f"{value_name}_source_rows": ("standardized_residual", "size"),
                }
            )
            .sort_values("validation_available_at", kind="mergesort")
        )

    left = series("left_model_id", "left_response_id", "left_residual")
    right = series("right_model_id", "right_response_id", "right_residual")
    aligned = left.merge(
        right,
        on="validation_available_at",
        how="inner",
        validate="one_to_one",
    )
    records: list[dict[str, object]] = []
    for statistic in ("pearson", "spearman"):
        if len(aligned) < 12:
            coefficient = math.nan
            p_value = math.nan
            status = "insufficient_observations"
        elif statistic == "pearson":
            test = stats.pearsonr(aligned["left_residual"], aligned["right_residual"])
            coefficient = float(test.statistic)
            p_value = float(test.pvalue)
            status = "ok"
        else:
            test = stats.spearmanr(aligned["left_residual"], aligned["right_residual"])
            coefficient = float(test.statistic)
            p_value = float(test.pvalue)
            status = "ok"
        records.append(
            {
                "left_model_id": pair["left_model_id"],
                "left_response_id": pair["left_response_id"],
                "right_model_id": pair["right_model_id"],
                "right_response_id": pair["right_response_id"],
                "alignment": "exact_common_publication_date",
                "statistic": statistic,
                "correlation": coefficient,
                "sample_count": len(aligned),
                "first_common_publication_date": (
                    pd.NaT
                    if aligned.empty
                    else aligned["validation_available_at"].min()
                ),
                "last_common_publication_date": (
                    pd.NaT
                    if aligned.empty
                    else aligned["validation_available_at"].max()
                ),
                "p_value": p_value,
                "status": status,
                "inference_label": "exploratory_causal_fit_retrospective_target",
            }
        )
    result = pd.DataFrame.from_records(records)
    result["q_value"] = benjamini_hochberg(result["p_value"].to_numpy(dtype=float))
    result["interpretation_caveat"] = INDEPENDENCE_CAVEAT
    return result


def _dependence_summary(report: Any, same_publication: pd.DataFrame) -> dict[str, object]:
    """Return a compact, explicitly non-causal interpretation artifact."""

    pairwise = report.cross_block_tests
    valid = pairwise.loc[pairwise["status"].eq("ok")].copy()
    significant = valid.loc[valid["q_value"].lt(0.05)]
    serial = report.serial_tests
    serial_valid = serial.loc[serial["status"].eq("ok")].copy()
    serial_significant = serial_valid.loc[serial_valid["q_value"].lt(0.05)]
    strongest = []
    if not valid.empty:
        columns = [
            "left_model_id",
            "left_response_id",
            "right_model_id",
            "right_response_id",
            "lag_months",
            "statistic",
            "correlation",
            "sample_count",
            "q_value",
            "correlation_p_q_interpretation",
        ]
        strongest = (
            valid.assign(_absolute=valid["correlation"].abs())
            .sort_values("_absolute", ascending=False, kind="mergesort")
            .head(12)
            .loc[:, columns]
            .to_dict(orient="records")
        )
    return {
        "method": "causal_fit_retrospective_completed_first_release_target_residuals",
        "live_filter_usage": "none",
        "cross_model_test_rows": len(pairwise),
        "valid_cross_model_tests": len(valid),
        "cross_model_bh_q_below_0_05": len(significant),
        "serial_test_rows": len(serial),
        "valid_serial_tests": len(serial_valid),
        "serial_bh_q_below_0_05": len(serial_significant),
        "same_publication_claims_tests": json.loads(
            same_publication.to_json(orient="records", date_format="iso")
        ),
        "strongest_absolute_correlations": strongest,
        "interpretation_caveat": INDEPENDENCE_CAVEAT,
        "p_value_caveat": (
            "All p- and q-values are exploratory, especially where residual "
            "serial dependence remains; effect sizes and coverage are primary."
        ),
    }


def build_evidence_experiment_inference(
    *,
    project_root: Path,
    config_path: Path,
    replay_end: object | None = None,
) -> dict[str, Path]:
    """Verify inputs, replay all six tests, and publish comparison artifacts."""

    root = project_root.resolve()
    config_path = _project_path(root, config_path)
    experiment, experiment_bytes = _load_experiment_config(config_path)
    registry = variant_registry_from_config(experiment)
    sources = experiment["sources"]

    base_path = _project_path(root, sources["base_filter_config"])
    base, base_bytes = _load_yaml(base_path)
    if base.get("model_id") != MODEL_ID:
        raise ValueError("base filter configuration has wrong model_id")
    original_model_ids = frozenset(str(value) for value in base["observation_models"])
    if original_model_ids != LEGACY_MODELS:
        raise ValueError(
            "legacy_baseline must equal the exact base observation-model key set"
        )
    base = _merge_observation_models(base, experiment["candidate_observation_models"])

    manifest_paths = {
        "score": _project_path(root, sources["score_manifest"]),
        "mapping": _project_path(root, sources["mapping_manifest"]),
        "evidence": _project_path(root, sources["evidence_manifest"]),
        "frozen_baseline": _project_path(
            root, sources["baseline_sensitivity_manifest"]
        ),
    }
    stage_ids = {
        # The original score manifest predates explicit stage identifiers.
        "score": None,
        "mapping": "m02_probability_map",
        "evidence": "m02_evidence_block_experiments",
        "frozen_baseline": "partial_defining_and_robustness_sensitivities",
    }
    manifests: dict[str, dict[str, Any]] = {}
    manifest_bytes: dict[str, bytes] = {}
    for name, path in manifest_paths.items():
        manifests[name], manifest_bytes[name] = _load_manifest(
            path, stage_id=stage_ids[name]
        )

    frozen_manifest = manifests["frozen_baseline"]
    if frozen_manifest.get("schema_version") != 1:
        raise ValueError("frozen baseline manifest has an unsupported schema")
    frozen_selection = frozen_manifest.get("model_selection", {})
    if (
        frozen_selection.get("baseline") != BASELINE_ID
        or list(frozen_selection.get("major_benchmarks", ()))
        != ["transition_only", "partial_only"]
    ):
        raise ValueError("frozen baseline manifest has unexpected model selection")
    if frozen_manifest.get("base_configuration_sha256") != sha256(base_bytes):
        raise ValueError("base filter configuration differs from the frozen baseline")
    frozen_config_path = _project_path(root, frozen_manifest["configuration"])
    frozen_config_bytes = frozen_config_path.read_bytes()
    if sha256(frozen_config_bytes) != frozen_manifest.get("configuration_sha256"):
        raise ValueError("frozen baseline configuration hash mismatch")
    frozen_upstream = {
        str(row["path"]): str(row["sha256"])
        for row in frozen_manifest.get("upstream_manifests", ())
    }
    for name in ("score", "mapping"):
        relative = manifest_paths[name].relative_to(root).as_posix()
        if frozen_upstream.get(relative) != sha256(manifest_bytes[name]):
            raise ValueError(f"{name} lineage differs from the frozen baseline")
    evidence_manifest = manifests["evidence"]
    if evidence_manifest.get("experiment_id") != "six_priority_evidence_test_v1":
        raise ValueError("unexpected evidence experiment id")
    baseline_evidence = evidence_manifest.get("baseline_evidence", {})
    baseline_evidence_path = str(baseline_evidence.get("manifest", ""))
    if frozen_upstream.get(baseline_evidence_path) != baseline_evidence.get(
        "manifest_sha256"
    ):
        raise ValueError("extended evidence does not descend from the frozen evidence set")
    frozen_end = pd.Timestamp(frozen_manifest["coverage"]["replay_end"]).normalize()
    requested_end = pd.Timestamp(
        replay_end if replay_end is not None else base.get("calendar", {}).get("replay_end")
    ).normalize()
    if requested_end != frozen_end:
        raise ValueError(
            "baseline invariance requires replay_end to equal the frozen replay end "
            f"{frozen_end.date()}"
        )

    source_contract = {
        "score_features": "score",
        "defining_components": "score",
        "mapping_history": "mapping",
        "evidence_events": "evidence",
        "baseline_evaluation_rows": "frozen_baseline",
        "baseline_event_update_audit": "frozen_baseline",
        "baseline_latest_marginals": "frozen_baseline",
    }
    source_bytes: dict[str, bytes] = {}
    for source_key, manifest_name in source_contract.items():
        relative = str(sources[source_key])
        source_bytes[source_key] = _verified_bytes(
            root, relative, manifests[manifest_name]
        )

    scores = pd.read_csv(BytesIO(source_bytes["score_features"]))
    components = pd.read_csv(BytesIO(source_bytes["defining_components"]))
    mapping = pd.read_csv(BytesIO(source_bytes["mapping_history"]))
    evidence = pd.read_csv(BytesIO(source_bytes["evidence_events"]))
    frozen_evaluation = pd.read_csv(
        BytesIO(source_bytes["baseline_evaluation_rows"]), float_precision="round_trip"
    )
    frozen_event_audit = pd.read_csv(
        BytesIO(source_bytes["baseline_event_update_audit"]),
        compression="gzip",
        float_precision="round_trip",
    )
    frozen_latest = pd.read_csv(
        BytesIO(source_bytes["baseline_latest_marginals"]), float_precision="round_trip"
    )
    prepared = prepare_observation_data(evidence, scores, base)
    candidate_ids = set(experiment["candidate_observation_models"])
    candidate_preparation = prepared.preparation_audit.loc[
        prepared.preparation_audit["observation_model_id"].isin(candidate_ids)
    ]
    if set(candidate_preparation["observation_model_id"]) != candidate_ids:
        raise ValueError("not every candidate observation model reached preparation")
    unusable = candidate_preparation.loc[
        candidate_preparation["selected_long_rows"].le(0)
        | candidate_preparation["live_event_vectors"].le(0)
        | candidate_preparation["training_candidates"].le(0)
    ]
    if not unusable.empty:
        raise ValueError(
            "candidate observation model has no testable point-in-time sample: "
            + ", ".join(unusable["observation_model_id"].astype(str))
        )
    result = run_inference_sensitivities(
        scores,
        mapping,
        prepared,
        components,
        base,
        experiment,
        replay_end=replay_end,
    )
    dependence_residuals = result.dependence_residuals.copy()
    if dependence_residuals.empty:
        raise ValueError("combined candidate produced no dependence residuals")
    fit_cutoff = pd.to_datetime(dependence_residuals["fit_cutoff"], errors="coerce")
    last_training = pd.to_datetime(
        dependence_residuals["last_training_available_at"], errors="coerce"
    )
    if fit_cutoff.isna().any() or last_training.isna().any() or not bool(
        (last_training < fit_cutoff).all()
    ):
        raise ValueError("dependence residual lineage violates the strict fit cutoff")
    dependence_config = experiment["dependence_diagnostics"]
    dependence_report = build_release_block_dependence_report(
        dependence_residuals,
        weekly_response_ids=tuple(dependence_config["weekly_response_ids"]),
        cross_block_lags=tuple(
            int(value) for value in dependence_config["cross_block_lags"]
        ),
        minimum_pair_observations=int(
            dependence_config["minimum_pair_observations"]
        ),
        serial_significance_level=float(
            dependence_config["serial_significance_level"]
        ),
        columns=ResidualColumnSpec(observation_date="observation_date"),
    )
    same_publication = _same_publication_dependence(
        dependence_residuals, experiment
    )
    dependence_summary = _dependence_summary(
        dependence_report, same_publication
    )

    burn_in = int(experiment["evaluation"]["initialization_burn_in_months"])
    summary = _evaluation_summary(
        result.evaluations,
        initial_date=result.initial_date,
        burn_in_months=burn_in,
    )
    subperiod = _evaluation_subperiod_summary(
        result.evaluations,
        initial_date=result.initial_date,
        burn_in_months=burn_in,
    )
    paired = _paired_comparisons(
        result.evaluations,
        initial_date=result.initial_date,
        burn_in_months=burn_in,
        event_audit=result.event_audit,
    )
    paired_bootstrap = _paired_block_bootstrap(
        result.evaluations,
        initial_date=result.initial_date,
        burn_in_months=burn_in,
        config=experiment,
    )
    coverage = _profile_coverage(
        prepared,
        result.event_audit,
        registry,
        experiment["evidence_sets"],
    )

    invariance_records = _artifact_invariance(
        result.evaluations,
        frozen_evaluation,
        artifact="evaluation_rows",
        variant_column="filter_variant",
        key_columns=(
            "reference_month",
            "evaluation_checkpoint",
            "availability_date",
            "forecast_as_of_date",
        ),
    )
    invariance_records.extend(
        _artifact_invariance(
            result.event_audit,
            frozen_event_audit,
            artifact="event_update_audit",
            variant_column="variant_id",
            key_columns=(
                "release_date",
                "observation_model_id",
                "reference_month",
                "event_instance_id",
            ),
            # Fit IDs are sequential publication surrogates. Candidate-only
            # fits can change their numbering without changing any fitted
            # system, parameter, event weight, or posterior update.
            ignored_columns=frozenset({"fit_id"}),
        )
    )
    invariance_records.extend(
        _artifact_invariance(
            result.latest_marginals,
            frozen_latest,
            artifact="latest_marginals",
            variant_column="variant_id",
            key_columns=("reference_month", "relative_month", "as_of_date"),
        )
    )
    invariance = pd.DataFrame.from_records(invariance_records)
    if not bool(invariance["passed"].all()):
        failed = invariance.loc[~invariance["passed"], ["artifact", "check", "column"]]
        raise ValueError(
            "candidate experiment changed the frozen baseline surface: "
            + failed.to_dict(orient="records").__repr__()
        )

    outputs = experiment["outputs"]
    processed_dir = _project_path(root, outputs["processed_dir"])
    published_dir = _project_path(root, outputs["published_dir"])
    processed_keys = (
        "evaluation_rows",
        "evaluation_summary",
        "evaluation_subperiod_summary",
        "paired_comparisons",
        "paired_block_bootstrap",
        "dependence_residuals",
        "dependence_monthly_residuals",
        "dependence_pairwise",
        "dependence_serial",
        "dependence_same_publication",
        "event_update_audit",
        "partial_defining_audit",
        "exact_score_audit",
        "transition_fit_audit",
        "transition_weight_audit",
        "emission_fit_audit",
        "tail_fold_scores",
        "hyperparameter_schedule",
        "latest_marginals",
        "model_registry",
        "profile_coverage",
        "baseline_invariance",
    )
    processed_paths = {
        key: _namespace_path(processed_dir, outputs[key]) for key in processed_keys
    }
    tables = {
        "evaluation_rows": result.evaluations,
        "evaluation_summary": summary,
        "evaluation_subperiod_summary": subperiod,
        "paired_comparisons": paired,
        "paired_block_bootstrap": paired_bootstrap,
        "dependence_residuals": dependence_residuals,
        "dependence_monthly_residuals": dependence_report.monthly_residuals,
        "dependence_pairwise": dependence_report.cross_block_tests,
        "dependence_serial": dependence_report.serial_tests,
        "dependence_same_publication": same_publication,
        "event_update_audit": result.event_audit,
        "partial_defining_audit": result.partial_defining_audit,
        "exact_score_audit": result.exact_score_audit,
        "transition_fit_audit": result.transition_fit_audit,
        "transition_weight_audit": result.transition_weight_audit,
        "emission_fit_audit": result.emission_fit_audit,
        "tail_fold_scores": result.tail_fold_scores,
        "hyperparameter_schedule": result.hyperparameter_schedule,
        "latest_marginals": result.latest_marginals,
        "model_registry": registry,
        "profile_coverage": coverage,
        "baseline_invariance": invariance,
    }
    for key, frame in tables.items():
        _write_csv(frame, processed_paths[key])

    public_mapping = {
        "public_evaluation_summary": summary,
        "public_evaluation_subperiod_summary": subperiod,
        "public_paired_comparisons": paired,
        "public_paired_block_bootstrap": paired_bootstrap,
        "public_dependence_pairwise": dependence_report.cross_block_tests,
        "public_dependence_serial": dependence_report.serial_tests,
        "public_dependence_same_publication": same_publication,
        "public_profile_coverage": coverage,
        "public_baseline_invariance": invariance,
        "public_model_registry": registry,
    }
    public_paths = {
        key: _namespace_path(published_dir, outputs[key]) for key in public_mapping
    }
    for key, frame in public_mapping.items():
        _write_csv(frame, public_paths[key])

    primary_checkpoint = str(experiment["evaluation"]["primary_checkpoint"])
    primary = summary.loc[summary["evaluation_checkpoint"].eq(primary_checkpoint)]
    baseline_row = primary.loc[primary["filter_variant"].eq(BASELINE_ID)]
    if len(baseline_row) != 1:
        raise ValueError("summary does not contain exactly one selected baseline row")
    metric_columns = (
        "mean_score_nlpd",
        "mean_quadrant_cross_entropy",
        "mean_quadrant_brier",
        "hard_quadrant_accuracy",
    )
    candidate_metrics: dict[str, dict[str, float | None]] = {}
    for variant_id in PRIORITY_VARIANTS:
        row = primary.loc[primary["filter_variant"].eq(variant_id)]
        if len(row) != 1:
            raise ValueError(f"primary summary lacks {variant_id}")
        candidate_metrics[variant_id] = {
            column: (
                None if pd.isna(row.iloc[0][column]) else float(row.iloc[0][column])
            )
            for column in metric_columns
        }
    method_summary = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "replay_start": result.initial_date.date().isoformat(),
        "replay_end": result.replay_end.date().isoformat(),
        "selected_baseline": BASELINE_ID,
        "baseline_invariance_passed": True,
        "candidate_status": "research_sensitivities_only_no_promotion",
        "primary_checkpoint": primary_checkpoint,
        "priority_variants": list(PRIORITY_VARIANTS),
        "candidate_primary_metrics": candidate_metrics,
        "paired_uncertainty": {
            "method": "circular moving-block bootstrap of paired monthly deltas",
            "block_length_months": int(
                experiment["evaluation"]["paired_block_bootstrap"]
                ["block_length_months"]
            ),
            "replications": int(
                experiment["evaluation"]["paired_block_bootstrap"]["replications"]
            ),
            "multiple_testing": "Holm adjustment across the six isolated priorities",
        },
        "dependence_diagnostic": {
            "variant_id": dependence_config["variant_id"],
            "residual_rows": len(dependence_residuals),
            "valid_cross_model_tests": dependence_summary[
                "valid_cross_model_tests"
            ],
            "cross_model_bh_q_below_0_05": dependence_summary[
                "cross_model_bh_q_below_0_05"
            ],
            "serial_bh_q_below_0_05": dependence_summary[
                "serial_bh_q_below_0_05"
            ],
            "live_filter_usage": "none",
        },
        "causal_contract": {
            "emission_training": "training_available_at strictly before release_date",
            "transition_training": "score pair available strictly before month roll",
            "hyperparameter": "fixed Student-t nu=7; causal annual ridge schedule",
            "same_day_weights": "shared pre-release-day state",
            "defining_releases": "partial first-release updates then exact completed score",
        },
        "latest_joint_gaussian_states": _latest_joint_state_rows(result),
        "credential_policy": "no network access and no credential read in inference stage",
    }
    method_summary_path = _namespace_path(
        published_dir, outputs["public_method_summary"]
    )
    dependence_summary_path = _namespace_path(
        published_dir, outputs["public_dependence_summary"]
    )
    _write_json(dependence_summary, dependence_summary_path)
    _write_json(method_summary, method_summary_path)

    generated = [
        *processed_paths.values(),
        *public_paths.values(),
        dependence_summary_path,
        method_summary_path,
    ]
    implementation_hashes: list[dict[str, object]] = []
    implementation_bytes: list[bytes] = []
    for relative in IMPLEMENTATION_FILES:
        path = _project_path(root, relative)
        content = path.read_bytes()
        implementation_bytes.append(content)
        implementation_hashes.append(
            {
                "path": relative,
                "sha256": sha256(content),
                "bytes": len(content),
            }
        )
    input_hash = hashlib.sha256()
    for content in (
        experiment_bytes,
        base_bytes,
        frozen_config_bytes,
        *manifest_bytes.values(),
        *source_bytes.values(),
        *implementation_bytes,
    ):
        input_hash.update(sha256(content).encode("ascii"))
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(root).as_posix(),
        "configuration_sha256": sha256(experiment_bytes),
        "base_configuration": base_path.relative_to(root).as_posix(),
        "base_configuration_sha256": sha256(base_bytes),
        "data_snapshot_sha256": input_hash.hexdigest(),
        "upstream_manifests": [
            {
                "path": manifest_paths[name].relative_to(root).as_posix(),
                "sha256": sha256(manifest_bytes[name]),
            }
            for name in sorted(manifest_paths)
        ],
        "implementation_file_hashes": implementation_hashes,
        "baseline_invariance_passed": True,
        "model_selection": {
            "baseline": BASELINE_ID,
            "major_benchmarks": ["transition_only", "partial_only"],
            "candidate_status": "experiment_only_no_candidate_promoted",
            "sensitivities": list(PRIORITY_VARIANTS),
        },
        "information_contract": method_summary["causal_contract"],
        "coverage": {
            "replay_start": result.initial_date.date().isoformat(),
            "replay_end": result.replay_end.date().isoformat(),
            "evaluation_rows": len(result.evaluations),
            "event_audit_rows": len(result.event_audit),
            "emission_fits": len(result.emission_fit_audit),
        },
        "generated_file_hashes": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path.read_bytes()),
                "bytes": path.stat().st_size,
            }
            for path in generated
        ],
        "credential_policy": "no network access and no credential read in inference stage",
    }
    manifest_path = _project_path(root, outputs["manifest"])
    _write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "evaluation": public_paths["public_evaluation_summary"],
        "paired": public_paths["public_paired_comparisons"],
        "coverage": public_paths["public_profile_coverage"],
        "dependence": dependence_summary_path,
        "invariance": public_paths["public_baseline_invariance"],
        "registry": public_paths["public_model_registry"],
        "summary": method_summary_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m02_evidence_block_inference.yaml"),
    )
    parser.add_argument("--replay-end", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.project_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    outputs = build_evidence_experiment_inference(
        project_root=root,
        config_path=config,
        replay_end=args.replay_end,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
