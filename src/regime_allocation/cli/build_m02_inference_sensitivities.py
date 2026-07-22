"""Build Model 02 partial-release and robust-inference sensitivities.

The command verifies every frozen upstream artifact, prepares the baseline and
real-retail observation blocks, runs the multi-variant causal replay, and
publishes compact comparisons plus detailed event, fit, weight, and lineage
audits.  It performs no network access and never reads an API credential.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from regime_allocation.data.dataset_acquisition import sha256
from regime_allocation.models.m02_soft_composite.inference_sensitivities import (
    real_retail_spec,
    run_inference_sensitivities,
)
from regime_allocation.models.m02_soft_composite.partial_defining import (
    prepare_partial_defining_data,
)
from regime_allocation.models.m02_soft_composite.walkforward import (
    prepare_observation_data,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "partial_defining_and_robustness_sensitivities"

# Freeze the exact implementation surface needed to reproduce this stage.
# Upstream data artifacts are already protected by their own manifests; these
# hashes make code changes independently visible even when the inputs and YAML
# configuration remain byte-for-byte identical.
IMPLEMENTATION_FILES = (
    "src/regime_allocation/cli/build_m02_inference_sensitivities.py",
    "src/regime_allocation/cli/build_m02_probability_map.py",
    "src/regime_allocation/cli/build_m02_retail_sensitivity.py",
    "src/regime_allocation/cli/build_m02_scores.py",
    "src/regime_allocation/features/m02_retail_sensitivities.py",
    "src/regime_allocation/models/m02_soft_composite/gaussian_emissions.py",
    "src/regime_allocation/models/m02_soft_composite/inference_sensitivities.py",
    "src/regime_allocation/models/m02_soft_composite/joint_filter.py",
    "src/regime_allocation/models/m02_soft_composite/partial_defining.py",
    "src/regime_allocation/models/m02_soft_composite/probability_map.py",
    "src/regime_allocation/models/m02_soft_composite/revisions.py",
    "src/regime_allocation/models/m02_soft_composite/robust_emissions.py",
    "src/regime_allocation/models/m02_soft_composite/robust_var.py",
    "src/regime_allocation/models/m02_soft_composite/scores.py",
    "src/regime_allocation/models/m02_soft_composite/var_transition.py",
    "src/regime_allocation/models/m02_soft_composite/walkforward.py",
)


def _project_path(root: Path, configured: object) -> Path:
    base = root.resolve()
    candidate = (base / str(configured)).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"configured path leaves project root: {configured}")
    return candidate


def _namespace_path(namespace: Path, filename: object) -> Path:
    base = namespace.resolve()
    candidate = (base / str(filename)).resolve()
    if candidate == base or base not in candidate.parents:
        raise ValueError(f"configured output leaves namespace: {filename}")
    return candidate


def _load_yaml(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    return payload, raw


def _load_sensitivity_config(path: Path) -> tuple[dict[str, Any], bytes]:
    config, raw = _load_yaml(path)
    if (
        config.get("schema_version") != 1
        or config.get("model_id") != MODEL_ID
        or config.get("stage_id") != STAGE_ID
    ):
        raise ValueError("unexpected Model 02 sensitivity configuration identity")
    for section in (
        "sources",
        "partial_defining_releases",
        "student_t_emissions",
        "robust_var",
        "retail_sensitivities",
        "evaluation",
        "outputs",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"sensitivity config omits mapping section {section}")
    if not isinstance(config.get("variants"), list) or not config["variants"]:
        raise ValueError("sensitivity config requires variants")
    candidates = [float(value) for value in config["student_t_emissions"]
                  ["degrees_of_freedom_sensitivity"]["candidates"]]
    if candidates != [4.0, 5.0, 7.0, 10.0, math.inf]:
        raise ValueError("frozen heavy-tail grid has changed")
    if int(config["student_t_emissions"]["degrees_of_freedom_sensitivity"]
           ["minimum_predictive_events"]) < 24:
        raise ValueError("tail selection requires at least 24 predictive events")
    stress = config["retail_sensitivities"]["stress_interaction"]
    if stress.get("enabled", True) or not str(stress.get("status", "")).startswith(
        "not_estimable"
    ):
        raise ValueError("the empirically unidentified stress interaction must stay disabled")
    required_outputs = {
        "processed_dir",
        "published_dir",
        "manifest",
        "evaluation_rows",
        "evaluation_summary",
        "evaluation_subperiod_summary",
        "paired_comparisons",
        "event_update_audit",
        "partial_defining_audit",
        "partial_preparation_audit",
        "exact_score_audit",
        "transition_fit_audit",
        "transition_weight_audit",
        "emission_fit_audit",
        "tail_fold_scores",
        "tail_summary",
        "retail_stress_identification_audit",
        "hyperparameter_schedule",
        "latest_marginals",
        "public_evaluation_summary",
        "public_evaluation_subperiod_summary",
        "public_paired_comparisons",
        "public_tail_summary",
        "public_latest_marginals",
        "public_method_summary",
    }
    missing = required_outputs.difference(config["outputs"])
    if missing:
        raise ValueError("sensitivity outputs omit: " + ", ".join(sorted(missing)))
    return config, raw


def _manifest_hash(manifest: Mapping[str, Any], relative_path: str) -> str:
    rows = [
        *manifest.get("generated_file_hashes", ()),
        *manifest.get("generated_files", ()),
    ]
    matches = [
        row["sha256"]
        for row in rows
        if row.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest lacks unique hash for {relative_path}")
    return str(matches[0])


def _verified_bytes(
    root: Path,
    relative_path: str,
    manifest: Mapping[str, Any],
) -> bytes:
    path = _project_path(root, relative_path)
    content = path.read_bytes()
    expected = _manifest_hash(manifest, relative_path)
    if sha256(content) != expected:
        raise ValueError(f"artifact hash mismatch: {relative_path}")
    return content


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(path, index=False, compression=compression)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _evaluation_summary(
    evaluations: pd.DataFrame,
    *,
    initial_date: pd.Timestamp,
    burn_in_months: int,
) -> pd.DataFrame:
    frame = evaluations.copy()
    if frame.empty:
        return pd.DataFrame()
    eligible_month = (initial_date.to_period("M") + int(burn_in_months)).to_timestamp()
    frame["summary_eligible"] = frame["reference_month"].ge(eligible_month)
    frame = frame.loc[frame["summary_eligible"]].copy()
    metrics = {
        "reference_month": "count",
        "growth_squared_error": "mean",
        "inflation_squared_error": "mean",
        "score_center_negative_log_predictive_density": "mean",
        "quadrant_cross_entropy_to_exact_score_map": "mean",
        "quadrant_brier_distance_to_exact_score_map": "mean",
        "quadrant_kl_divergence_to_exact_score_map": "mean",
        "hard_quadrant_correct": "mean",
    }
    available = {name: aggregation for name, aggregation in metrics.items() if name in frame}
    summary = (
        frame.groupby(["filter_variant", "evaluation_checkpoint"], as_index=False)
        .agg(available)
        .rename(
            columns={
                "reference_month": "months",
                "score_center_negative_log_predictive_density": "mean_score_nlpd",
                "quadrant_cross_entropy_to_exact_score_map": "mean_quadrant_cross_entropy",
                "quadrant_brier_distance_to_exact_score_map": "mean_quadrant_brier",
                "quadrant_kl_divergence_to_exact_score_map": "mean_quadrant_kl",
                "hard_quadrant_correct": "hard_quadrant_accuracy",
            }
        )
    )
    summary["growth_rmse"] = np.sqrt(summary.pop("growth_squared_error"))
    summary["inflation_rmse"] = np.sqrt(summary.pop("inflation_squared_error"))
    medians = (
        frame.groupby(["filter_variant", "evaluation_checkpoint"], as_index=False)[
            "score_center_negative_log_predictive_density"
        ]
        .median()
        .rename(
            columns={
                "score_center_negative_log_predictive_density": "median_score_nlpd"
            }
        )
    )
    summary = summary.merge(
        medians,
        on=["filter_variant", "evaluation_checkpoint"],
        how="left",
        validate="one_to_one",
    )
    summary["burn_in_months"] = int(burn_in_months)
    return summary.sort_values(
        ["evaluation_checkpoint", "mean_score_nlpd", "filter_variant"],
        kind="mergesort",
    ).reset_index(drop=True)


def _evaluation_subperiod_summary(
    evaluations: pd.DataFrame,
    *,
    initial_date: pd.Timestamp,
    burn_in_months: int,
) -> pd.DataFrame:
    """Expose how much the three-month COVID target shock drives mean scores."""

    month = pd.to_datetime(evaluations["reference_month"], errors="coerce")
    periods = {
        "full_sample": pd.Series(True, index=evaluations.index),
        "excluding_2020_03_through_2020_05": ~month.between(
            "2020-03-01", "2020-05-01", inclusive="both"
        ),
        "pre_pandemic_through_2020_02": month.le(pd.Timestamp("2020-02-01")),
        "post_initial_shock_from_2020_06": month.ge(pd.Timestamp("2020-06-01")),
    }
    parts: list[pd.DataFrame] = []
    for label, mask in periods.items():
        summary = _evaluation_summary(
            evaluations.loc[mask].copy(),
            initial_date=initial_date,
            burn_in_months=burn_in_months,
        )
        summary.insert(0, "evaluation_sample", label)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True, sort=False)


def _paired_comparisons(
    evaluations: pd.DataFrame,
    *,
    initial_date: pd.Timestamp,
    burn_in_months: int,
    event_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if evaluations.empty:
        return pd.DataFrame()
    eligible_month = (initial_date.to_period("M") + int(burn_in_months)).to_timestamp()
    frame = evaluations.loc[evaluations["reference_month"].ge(eligible_month)].copy()
    keys = ["reference_month", "evaluation_checkpoint"]
    metrics = [
        "score_center_negative_log_predictive_density",
        "growth_squared_error",
        "inflation_squared_error",
        "quadrant_cross_entropy_to_exact_score_map",
        "quadrant_brier_distance_to_exact_score_map",
        "quadrant_kl_divergence_to_exact_score_map",
        "hard_quadrant_correct",
    ]
    metrics = [name for name in metrics if name in frame]
    rows: list[dict[str, object]] = []

    def add_comparison(
        variant_id: str,
        comparator_id: str,
        *,
        comparison_scope: str,
        allowed_months: set[pd.Timestamp] | None = None,
    ) -> None:
        variant = frame.loc[frame["filter_variant"].eq(variant_id)].copy()
        comparator = frame.loc[frame["filter_variant"].eq(comparator_id)].copy()
        if allowed_months is not None:
            variant = variant.loc[variant["reference_month"].isin(allowed_months)]
            comparator = comparator.loc[comparator["reference_month"].isin(allowed_months)]
        paired = variant[keys + metrics].merge(
            comparator[keys + metrics],
            on=keys,
            how="inner",
            suffixes=("_variant", "_comparator"),
            validate="one_to_one",
        )
        for checkpoint, group in paired.groupby("evaluation_checkpoint", sort=True):
            record: dict[str, object] = {
                "filter_variant": variant_id,
                "comparison_baseline": comparator_id,
                "comparison_scope": comparison_scope,
                "evaluation_checkpoint": checkpoint,
                "paired_months": int(len(group)),
                "first_paired_month": group["reference_month"].min(),
                "last_paired_month": group["reference_month"].max(),
            }
            for metric in metrics:
                variant_values = pd.to_numeric(
                    group[f"{metric}_variant"], errors="coerce"
                ).astype(float)
                transition_values = pd.to_numeric(
                    group[f"{metric}_comparator"], errors="coerce"
                ).astype(float)
                delta = variant_values - transition_values
                record[f"mean_delta_{metric}"] = float(delta.mean())
            rows.append(record)

    for variant_id in sorted(frame["filter_variant"].unique()):
        if variant_id != "transition_only":
            add_comparison(
                str(variant_id),
                "transition_only",
                comparison_scope="all_common_evaluation_months",
            )
    if {
        "retail_shrinkage_combined",
        "huber_var_combined",
    }.issubset(set(frame["filter_variant"])):
        add_comparison(
            "retail_shrinkage_combined",
            "huber_var_combined",
            comparison_scope="all_common_evaluation_months",
        )
    if {
        "retail_real_decomposition_combined",
        "huber_var_combined",
    }.issubset(set(frame["filter_variant"])) and event_audit is not None:
        direct_months = set(
            pd.to_datetime(
                event_audit.loc[
                    event_audit["variant_id"].eq(
                        "retail_real_decomposition_combined"
                    )
                    & event_audit["profile_id"].eq("retail_real")
                    & event_audit["update_status"].eq("applied"),
                    "reference_month",
                ],
                errors="coerce",
            ).dropna()
        )
        add_comparison(
            "retail_real_decomposition_combined",
            "huber_var_combined",
            comparison_scope="direct_real_retail_update_reference_months",
            allowed_months=direct_months,
        )
    return pd.DataFrame.from_records(rows)


def _tail_summary(
    folds: pd.DataFrame,
    *,
    profile_ids: Sequence[str] | None = None,
    degrees_grid: Sequence[float] = (4.0, 5.0, 7.0, 10.0, math.inf),
) -> pd.DataFrame:
    scored = folds.loc[folds["status"].eq("scored")].copy()
    if scored.empty and not profile_ids:
        return pd.DataFrame()
    grouped = (
        scored.groupby(
            ["profile_id", "observation_model_id", "base_lambda", "degrees_of_freedom"],
            as_index=False,
        )
        .agg(
            validation_folds=("validation_year", "nunique"),
            validation_observations=("validation_observations", "sum"),
            total_predictive_nll=("total_predictive_nll", "sum"),
        )
    )
    grouped["mean_predictive_nll"] = (
        grouped["total_predictive_nll"] / grouped["validation_observations"]
    )
    # For the requested nu sensitivity, compare each tail after choosing the
    # best ridge penalty on the same causal fold predictions.
    best = grouped.sort_values(
        ["profile_id", "degrees_of_freedom", "mean_predictive_nll", "base_lambda"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).drop_duplicates(["profile_id", "degrees_of_freedom"])
    best["best_tail_for_profile"] = best.groupby("profile_id")[
        "mean_predictive_nll"
    ].transform("min").pipe(
        lambda minimum: np.isclose(best["mean_predictive_nll"], minimum)
    )
    best["selection_role"] = (
        "full-history causal-fold diagnostic only; live schedule uses prior folds"
    )
    best["sensitivity_status"] = "evaluated"
    if profile_ids:
        missing_records: list[dict[str, object]] = []
        available_profiles = set(best["profile_id"])
        for profile_id in sorted(set(profile_ids).difference(available_profiles)):
            for degrees in degrees_grid:
                missing_records.append(
                    {
                        "profile_id": profile_id,
                        "observation_model_id": profile_id.removeprefix("base:"),
                        "base_lambda": math.nan,
                        "degrees_of_freedom": float(degrees),
                        "validation_folds": 0,
                        "validation_observations": 0,
                        "total_predictive_nll": math.nan,
                        "mean_predictive_nll": math.nan,
                        "best_tail_for_profile": False,
                        "selection_role": (
                            "not used; live schedule retains the declared nu=7 fallback"
                        ),
                        "sensitivity_status": (
                            "insufficient_causal_annual_origin_history"
                        ),
                    }
                )
        if missing_records:
            best = pd.concat(
                [best, pd.DataFrame.from_records(missing_records)],
                ignore_index=True,
                sort=False,
            )
    return best.sort_values(
        ["profile_id", "degrees_of_freedom"], kind="mergesort"
    ).reset_index(drop=True)


def _stress_identification_audit(events: pd.DataFrame) -> pd.DataFrame:
    """Align the latest strictly prior claims innovation to retail releases."""

    claims = events.loc[
        events["observation_model_id"].eq("weekly_labor_stress")
        & pd.to_numeric(events["initial_claims_innovation"], errors="coerce").notna(),
        ["release_date", "initial_claims_innovation", "event_instance_id"],
    ].copy()
    claims = claims.rename(
        columns={
            "release_date": "claims_release_date",
            "event_instance_id": "claims_event_instance_id",
        }
    ).sort_values("claims_release_date", kind="mergesort")
    retail = events.loc[
        events["observation_model_id"].eq("consumer_demand"),
        ["release_date", "reference_month", "event_instance_id"],
    ].copy().sort_values("release_date", kind="mergesort")
    aligned = pd.merge_asof(
        retail,
        claims,
        left_on="release_date",
        right_on="claims_release_date",
        direction="backward",
        allow_exact_matches=False,
    )
    innovation = pd.to_numeric(
        aligned["initial_claims_innovation"], errors="coerce"
    )
    aligned["declared_claims_stress"] = np.maximum(innovation - 2.0, 0.0)
    aligned["positive_stress"] = aligned["declared_claims_stress"].gt(0.0)
    aligned["identification_status"] = np.where(
        aligned["positive_stress"], "positive_interaction_design", "zero_interaction_design"
    )
    return aligned


def _latest_joint_state_rows(result: Any) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant_id, state in result.latest_states.items():
        record: dict[str, object] = {
            "variant_id": variant_id,
            "reference_months": [month.date().isoformat() for month in state.reference_months],
            "mean": state.mean.tolist(),
            "covariance": state.covariance.tolist(),
            "exact_mask": list(state.exact_mask),
        }
        rows.append(record)
    return rows


def build_inference_sensitivities(
    *,
    project_root: Path,
    config_path: Path,
    replay_end: object | None = None,
) -> dict[str, Path]:
    """Verify inputs, run the sensitivity replay, and publish its artifacts."""

    root = project_root.resolve()
    config_path = _project_path(root, config_path)
    sensitivity, sensitivity_bytes = _load_sensitivity_config(config_path)
    base_path = _project_path(root, sensitivity["sources"]["base_filter_config"])
    base, base_bytes = _load_yaml(base_path)
    if base.get("model_id") != MODEL_ID:
        raise ValueError("base filter configuration has wrong model_id")

    manifest_paths = {
        "score": _project_path(root, sensitivity["sources"]["score_manifest"]),
        "mapping": _project_path(root, sensitivity["sources"]["mapping_manifest"]),
        "evidence": _project_path(root, sensitivity["sources"]["evidence_manifest"]),
        "retail": _project_path(root, sensitivity["outputs"]["retail_manifest"]),
    }
    manifest_bytes = {name: path.read_bytes() for name, path in manifest_paths.items()}
    manifests = {name: json.loads(content) for name, content in manifest_bytes.items()}
    if manifests["score"].get("model_id") != MODEL_ID:
        raise ValueError("score manifest model_id mismatch")
    if manifests["mapping"].get("stage_id") != "m02_probability_map":
        raise ValueError("mapping manifest stage mismatch")
    if manifests["evidence"].get("stage_id") != "m02_release_evidence":
        raise ValueError("evidence manifest stage mismatch")
    if manifests["retail"].get("stage_id") != "m02_retail_real_decomposition_sensitivity":
        raise ValueError("retail manifest stage mismatch")
    if manifests["retail"].get("configuration_sha256") != sha256(sensitivity_bytes):
        raise ValueError(
            "retail manifest was built from a different sensitivity config; rerun retail stage"
        )

    source_manifest = {
        "score_features": "score",
        "defining_components": "score",
        "mapping_history": "mapping",
        "evidence_events": "evidence",
    }
    source_bytes: dict[str, bytes] = {}
    for source_key, manifest_name in source_manifest.items():
        relative = str(sensitivity["sources"][source_key])
        source_bytes[source_key] = _verified_bytes(
            root, relative, manifests[manifest_name]
        )
    retail_relative = str(sensitivity["outputs"]["retail_events"])
    retail_bytes = _verified_bytes(root, retail_relative, manifests["retail"])

    from io import BytesIO

    scores = pd.read_csv(BytesIO(source_bytes["score_features"]))
    components = pd.read_csv(BytesIO(source_bytes["defining_components"]))
    mapping = pd.read_csv(BytesIO(source_bytes["mapping_history"]))
    evidence = pd.read_csv(BytesIO(source_bytes["evidence_events"]))
    retail_events = pd.read_csv(BytesIO(retail_bytes))
    prepared = prepare_observation_data(evidence, scores, base)
    partial_prepared = prepare_partial_defining_data(components, scores)
    stress_audit = _stress_identification_audit(prepared.events)
    real_config = deepcopy(base)
    real_spec = real_retail_spec(base)
    real_config["observation_models"] = {
        real_spec.block_id: {
            "economic_block": "consumer_demand",
            "event_block": "consumer_demand_real_decomposition",
            "responses": list(real_spec.response_names),
            "controls": [],
            "loading_penalties": [list(row) for row in real_spec.state_loading_penalties],
            "exact_zero_mask": [list(row) for row in real_spec.exact_zero_mask],
            "minimum_training_samples": real_spec.minimum_training_samples,
            "validation_minimum_training_samples": (
                real_spec.validation_minimum_training_samples
            ),
            "minimum_validation_observations": real_spec.minimum_validation_observations,
        }
    }
    real_prepared = prepare_observation_data(retail_events, scores, real_config)
    result = run_inference_sensitivities(
        scores,
        mapping,
        prepared,
        components,
        base,
        sensitivity,
        real_retail_prepared=real_prepared,
        replay_end=replay_end,
    )

    burn_in = int(sensitivity["evaluation"]["initialization_burn_in_months"])
    evaluation_summary = _evaluation_summary(
        result.evaluations, initial_date=result.initial_date, burn_in_months=burn_in
    )
    evaluation_subperiod_summary = _evaluation_subperiod_summary(
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
    tail_summary = _tail_summary(
        result.tail_fold_scores,
        profile_ids=result.hyperparameter_schedule["profile_id"].unique(),
        degrees_grid=[
            float(value)
            for value in sensitivity["student_t_emissions"]
            ["degrees_of_freedom_sensitivity"]["candidates"]
        ],
    )
    outputs = sensitivity["outputs"]
    processed_dir = _project_path(root, outputs["processed_dir"])
    published_dir = _project_path(root, outputs["published_dir"])
    processed_keys = (
        "evaluation_rows",
        "evaluation_summary",
        "evaluation_subperiod_summary",
        "paired_comparisons",
        "event_update_audit",
        "partial_defining_audit",
        "partial_preparation_audit",
        "exact_score_audit",
        "transition_fit_audit",
        "transition_weight_audit",
        "emission_fit_audit",
        "tail_fold_scores",
        "tail_summary",
        "retail_stress_identification_audit",
        "hyperparameter_schedule",
        "latest_marginals",
    )
    processed_paths = {
        key: _namespace_path(processed_dir, outputs[key]) for key in processed_keys
    }
    public_keys = (
        "public_evaluation_summary",
        "public_evaluation_subperiod_summary",
        "public_paired_comparisons",
        "public_tail_summary",
        "public_latest_marginals",
        "public_method_summary",
    )
    public_paths = {
        key: _namespace_path(published_dir, outputs[key]) for key in public_keys
    }
    tables = {
        "evaluation_rows": result.evaluations,
        "evaluation_summary": evaluation_summary,
        "evaluation_subperiod_summary": evaluation_subperiod_summary,
        "paired_comparisons": paired,
        "event_update_audit": result.event_audit,
        "partial_defining_audit": result.partial_defining_audit,
        "partial_preparation_audit": partial_prepared.preparation_audit,
        "exact_score_audit": result.exact_score_audit,
        "transition_fit_audit": result.transition_fit_audit,
        "transition_weight_audit": result.transition_weight_audit,
        "emission_fit_audit": result.emission_fit_audit,
        "tail_fold_scores": result.tail_fold_scores,
        "tail_summary": tail_summary,
        "retail_stress_identification_audit": stress_audit,
        "hyperparameter_schedule": result.hyperparameter_schedule,
        "latest_marginals": result.latest_marginals,
    }
    for key, frame in tables.items():
        _write_csv(frame, processed_paths[key])
    _write_csv(evaluation_summary, public_paths["public_evaluation_summary"])
    _write_csv(
        evaluation_subperiod_summary,
        public_paths["public_evaluation_subperiod_summary"],
    )
    _write_csv(paired, public_paths["public_paired_comparisons"])
    _write_csv(tail_summary, public_paths["public_tail_summary"])
    _write_csv(result.latest_marginals, public_paths["public_latest_marginals"])

    stress = sensitivity["retail_sensitivities"]["stress_interaction"]
    method_summary = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "replay_start": result.initial_date.date().isoformat(),
        "replay_end": result.replay_end.date().isoformat(),
        "variants": sorted(result.latest_states),
        "partial_defining_updates_applied": int(
            result.partial_defining_audit.get("status", pd.Series(dtype=str))
            .eq("applied")
            .sum()
        ),
        "robust_events_applied": int(
            result.event_audit.get("update_status", pd.Series(dtype=str)).eq("applied").sum()
        ),
        "event_weight_summary": {
            "count": int(result.event_audit.get("event_weight", pd.Series(dtype=float)).count()),
            "minimum": (
                None
                if result.event_audit.get("event_weight", pd.Series(dtype=float)).dropna().empty
                else float(result.event_audit["event_weight"].min())
            ),
            "median": (
                None
                if result.event_audit.get("event_weight", pd.Series(dtype=float)).dropna().empty
                else float(result.event_audit["event_weight"].median())
            ),
        },
        "stress_interaction": {
            "status": stress["status"],
            "enabled": bool(stress["enabled"]),
            "reason": stress["identification_audit"],
            "aligned_retail_events": int(len(stress_audit)),
            "positive_stress_events": int(stress_audit["positive_stress"].sum()),
        },
        "same_day_weight_policy": "all robust weights frozen from shared pre-release-day state",
        "tail_selection_policy": sensitivity["student_t_emissions"]
        ["degrees_of_freedom_sensitivity"]["selection"],
        "latest_joint_gaussian_states": _latest_joint_state_rows(result),
        "credential_policy": "no network access and no credential read in this stage",
    }
    _write_json(method_summary, public_paths["public_method_summary"])

    generated = [*processed_paths.values(), *public_paths.values()]
    generated_hashes = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
        for path in generated
    ]
    implementation_hashes = []
    implementation_bytes = []
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
    manifest_path = _project_path(root, outputs["manifest"])
    input_hash = hashlib.sha256()
    for content in [
        sensitivity_bytes,
        base_bytes,
        *manifest_bytes.values(),
        *source_bytes.values(),
        retail_bytes,
        *implementation_bytes,
    ]:
        input_hash.update(sha256(content).encode("ascii"))
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(root).as_posix(),
        "configuration_sha256": sha256(sensitivity_bytes),
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
        "information_contract": {
            "transition_cutoff": "pair_available_at_strictly_before_month_roll",
            "component_fit_cutoff": "complete_score_available_strictly_before_release",
            "emission_fit_cutoff": "training_available_at_strictly_before_release",
            "same_day_student_t_weights": "shared_pre_release_day_state",
            "final_defining_block": "skipped_then_exact_score_conditioned_end_of_day",
            "tail_selection": "annual_causal_rolling_origin_prior_folds_only",
        },
        "coverage": {
            "replay_start": result.initial_date.date().isoformat(),
            "replay_end": result.replay_end.date().isoformat(),
            "evaluation_rows": len(result.evaluations),
            "evidence_event_audit_rows": len(result.event_audit),
            "partial_event_audit_rows": len(result.partial_defining_audit),
            "transition_fits": len(result.transition_fit_audit),
            "emission_fits": len(result.emission_fit_audit),
        },
        "generated_file_hashes": generated_hashes,
        "credential_policy": "no network access and no credential read in inference sensitivity stage",
    }
    _write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "evaluation": public_paths["public_evaluation_summary"],
        "paired": public_paths["public_paired_comparisons"],
        "tails": public_paths["public_tail_summary"],
        "latest": public_paths["public_latest_marginals"],
        "summary": public_paths["public_method_summary"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m02_inference_sensitivities.yaml"),
    )
    parser.add_argument("--replay-end", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.project_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    outputs = build_inference_sensitivities(
        project_root=root,
        config_path=config,
        replay_end=args.replay_end,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
