"""Attribute the selected Model 02 evidence blocks with causal ablations.

This command leaves every selected ``student_t_7_combined`` modeling choice
fixed and changes only the enabled non-defining observation-model allowlist.
Each existing block is added alone to ``partial_only`` and removed alone from
the selected baseline.  It also joins the prior evidence-replacement replay to
the matching reduced-core variants, so replacement blocks are evaluated
against a model with the legacy block genuinely absent.

The stage is offline: it verifies and reads frozen artifacts, performs no
network request, and never reads ``FRED_API_KEY`` or any other credential.
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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from regime_allocation.cli.build_m02_evidence_experiment_inference import (
    _load_manifest,
    _profile_coverage,
)
from regime_allocation.cli.build_m02_inference_sensitivities import (
    _evaluation_subperiod_summary,
    _evaluation_summary,
    _latest_joint_state_rows,
    _load_yaml,
    _namespace_path,
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
from regime_allocation.models.m02_soft_composite.walkforward import (
    prepare_observation_data,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "existing_block_attribution"
BASELINE_ID = "student_t_7_combined"
CONTROL_IDS = ("transition_only", "partial_only", BASELINE_ID)
METRICS = (
    "score_center_negative_log_predictive_density",
    "quadrant_cross_entropy_to_exact_score_map",
    "quadrant_brier_distance_to_exact_score_map",
    "hard_quadrant_correct",
)
LOWER_IS_BETTER = frozenset(METRICS[:-1])

# The input artifacts have their own manifests. These files are the complete
# code surface that turns those inputs into this experiment's outputs.
IMPLEMENTATION_FILES = (
    "src/regime_allocation/cli/build_m02_existing_block_attribution.py",
    "src/regime_allocation/cli/build_m02_evidence_experiment_inference.py",
    "src/regime_allocation/cli/build_m02_inference_sensitivities.py",
    "src/regime_allocation/models/m02_soft_composite/gaussian_emissions.py",
    "src/regime_allocation/models/m02_soft_composite/inference_sensitivities.py",
    "src/regime_allocation/models/m02_soft_composite/joint_filter.py",
    "src/regime_allocation/models/m02_soft_composite/partial_defining.py",
    "src/regime_allocation/models/m02_soft_composite/robust_emissions.py",
    "src/regime_allocation/models/m02_soft_composite/robust_var.py",
    "src/regime_allocation/models/m02_soft_composite/var_transition.py",
    "src/regime_allocation/models/m02_soft_composite/walkforward.py",
)


def _comparison_id(candidate_id: str, reference_id: str) -> str:
    return f"{candidate_id}__vs__{reference_id}"


def _load_attribution_config(path: Path) -> tuple[dict[str, Any], bytes]:
    """Validate the experiment's exact block, variant, and contrast contract."""

    config, raw = _load_yaml(path)
    if (
        config.get("schema_version") != 1
        or config.get("model_id") != MODEL_ID
        or config.get("stage_id") != STAGE_ID
    ):
        raise ValueError("unexpected existing-block attribution identity")
    required = {
        "model_selection",
        "sources",
        "blocks",
        "evidence_sets",
        "variants",
        "comparisons",
        "replacement_comparisons",
        "evaluation",
        "outputs",
    }
    if required.difference(config):
        raise ValueError("attribution config omits a required section")

    blocks = config["blocks"]
    if not isinstance(blocks, Mapping) or len(blocks) != 8:
        raise ValueError("attribution requires seven atomic blocks and one pair")
    primary = {
        str(block_id): frozenset(str(value) for value in declaration["observation_models"])
        for block_id, declaration in blocks.items()
        if bool(declaration["primary_atomic"])
    }
    if len(primary) != 7 or any(len(models) != 1 for models in primary.values()):
        raise ValueError("primary attribution blocks must be seven single-model sets")
    legacy = frozenset().union(*primary.values())
    pair = blocks.get("inflation_pressure_pair", {})
    if bool(pair.get("primary_atomic", True)) or frozenset(
        str(value) for value in pair.get("observation_models", ())
    ) != {"inflation_expectations", "inflation_input_costs"}:
        raise ValueError("inflation-pressure diagnostic pair is malformed")

    evidence_sets = config["evidence_sets"]
    normalized_sets = {
        str(set_id): frozenset(str(value) for value in declaration["observation_models"])
        for set_id, declaration in evidence_sets.items()
    }
    if normalized_sets.get("none") != frozenset() or normalized_sets.get(
        "all_current"
    ) != legacy:
        raise ValueError("control evidence sets do not match the legacy baseline")
    for block_id, models in primary.items():
        if normalized_sets.get(f"only_{block_id}") != models:
            raise ValueError(f"add-one evidence set is wrong for {block_id}")
        if normalized_sets.get(f"without_{block_id}") != legacy.difference(models):
            raise ValueError(f"leave-one-out evidence set is wrong for {block_id}")

    variants = {str(row["id"]): row for row in config["variants"]}
    if set(CONTROL_IDS).difference(variants):
        raise ValueError("attribution controls are incomplete")
    for variant_id, declaration in variants.items():
        if variant_id in CONTROL_IDS:
            continue
        expected = {
            "non_defining_evidence": True,
            "partial_defining_releases": True,
            "emission": "student_t_7",
            "var": "ols",
            "retail": "baseline_nominal",
            "model_role": "sensitivity",
        }
        if any(declaration.get(key) != value for key, value in expected.items()):
            raise ValueError(f"candidate {variant_id} changes more than evidence membership")

    comparisons = config["comparisons"]
    primary_comparisons = [row for row in comparisons if bool(row["primary_atomic"])]
    if len(primary_comparisons) != 14 or len(comparisons) != 16:
        raise ValueError("expected fourteen atomic and two pair attribution contrasts")
    seen: set[str] = set()
    for declaration in comparisons:
        candidate = str(declaration["candidate_id"])
        reference = str(declaration["reference_id"])
        arm = str(declaration["experiment_arm"])
        block_id = str(declaration["block_id"])
        comparison_id = _comparison_id(candidate, reference)
        if comparison_id in seen or candidate == reference:
            raise ValueError("duplicate or self-referenced attribution contrast")
        seen.add(comparison_id)
        if candidate not in variants or reference not in variants or block_id not in blocks:
            raise ValueError("attribution contrast references an unknown declaration")
        expected_reference = "partial_only" if arm == "add_one" else BASELINE_ID
        if arm not in {"add_one", "leave_one_out"} or reference != expected_reference:
            raise ValueError("attribution contrast uses the wrong causal reference")

    bootstrap = config["evaluation"]["paired_block_bootstrap"]
    if (
        int(bootstrap["replications"]) < 1000
        or int(bootstrap["block_length_months"]) < 2
        or str(bootstrap["resampling"]) != "circular"
    ):
        raise ValueError("attribution bootstrap settings are invalid")
    atomic_ids = {
        _comparison_id(str(row["candidate_id"]), str(row["reference_id"]))
        for row in primary_comparisons
    }
    if set(bootstrap["holm_families"]["primary_atomic"]) != atomic_ids:
        raise ValueError("primary Holm family must contain all fourteen atomic tests")
    if set(config["evaluation"]["metrics"]) != set(METRICS):
        raise ValueError("attribution metrics differ from the declared comparison set")
    return config, raw


def _run_config(
    frozen: Mapping[str, Any], experiment: Mapping[str, Any]
) -> dict[str, Any]:
    """Overlay only variant membership onto the frozen sensitivity contract."""

    result = deepcopy(dict(frozen))
    result["stage_id"] = STAGE_ID
    result["model_selection"] = {
        "baseline": BASELINE_ID,
        "major_benchmarks": ["transition_only", "partial_only"],
        "role_vocabulary": ["baseline", "major_benchmark", "sensitivity"],
        "selection_status": "attribution_experiment_only",
    }
    result["evidence_sets"] = deepcopy(experiment["evidence_sets"])
    result["variants"] = deepcopy(experiment["variants"])
    result["evaluation"] = {
        **deepcopy(dict(frozen["evaluation"])),
        **deepcopy(dict(experiment["evaluation"])),
    }
    return result


def _variant_frame(
    frame: pd.DataFrame, *, column: str, variant_id: str
) -> pd.DataFrame:
    selected = frame.loc[frame[column].astype(str).eq(variant_id)].copy()
    if selected.empty:
        raise ValueError(f"artifact lacks variant {variant_id}")
    return selected


def _invariance_rows(
    current: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    artifact: str,
    variant_column: str,
    variant_id: str,
    key_columns: Sequence[str],
    ignored_columns: frozenset[str] = frozenset(),
    tolerance: float = 1.0e-12,
) -> list[dict[str, object]]:
    """Compare any declared control with its frozen artifact counterpart."""

    left = _variant_frame(current, column=variant_column, variant_id=variant_id)
    right = _variant_frame(frozen, column=variant_column, variant_id=variant_id)
    for frame in (left, right):
        for column in frame.columns:
            if column.endswith(("_date", "_month", "_at", "_cutoff")):
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
    left = left.sort_values(list(key_columns), kind="stable").reset_index(drop=True)
    right = right.sort_values(list(key_columns), kind="stable").reset_index(drop=True)
    ignored = {variant_column, "model_role", *ignored_columns}
    left_columns = set(left).difference(ignored)
    right_columns = set(right).difference(ignored)
    rows: list[dict[str, object]] = [
        {
            "artifact": artifact,
            "variant_id": variant_id,
            "check": "row_count",
            "column": "",
            "current": len(left),
            "frozen": len(right),
            "maximum_absolute_difference": 0.0,
            "passed": len(left) == len(right),
        },
        {
            "artifact": artifact,
            "variant_id": variant_id,
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
        first, second = left[column], right[column]
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
        elif pd.api.types.is_datetime64_any_dtype(
            first
        ) or pd.api.types.is_datetime64_any_dtype(second):
            maximum = 0.0
            first_ns = pd.to_datetime(first, errors="coerce", utc=True).astype(
                "datetime64[ns, UTC]"
            ).astype("int64")
            second_ns = pd.to_datetime(second, errors="coerce", utc=True).astype(
                "datetime64[ns, UTC]"
            ).astype("int64")
            passed = bool(
                np.array_equal(
                    first_ns.to_numpy(dtype=np.int64),
                    second_ns.to_numpy(dtype=np.int64),
                )
            )
        else:
            maximum = 0.0
            first_text = first.astype("string").replace("", pd.NA).fillna("<NA>")
            second_text = second.astype("string").replace("", pd.NA).fillna("<NA>")
            passed = first_text.equals(second_text)
        rows.append(
            {
                "artifact": artifact,
                "variant_id": variant_id,
                "check": "column_values",
                "column": column,
                "current": "",
                "frozen": "",
                "maximum_absolute_difference": maximum,
                "passed": passed,
            }
        )
    return rows


def _eligible_frames(
    frames: Mapping[str, pd.DataFrame], *, initial_date: pd.Timestamp, burn_in_months: int
) -> dict[str, pd.DataFrame]:
    eligible = (initial_date.to_period("M") + int(burn_in_months)).to_timestamp()
    result: dict[str, pd.DataFrame] = {}
    for variant_id, frame in frames.items():
        normalized = frame.copy()
        normalized["reference_month"] = pd.to_datetime(
            normalized["reference_month"], errors="raise"
        )
        result[variant_id] = normalized.loc[
            normalized["reference_month"].ge(eligible)
        ].copy()
    return result


def _paired_delta_frame(
    frames: Mapping[str, pd.DataFrame], declaration: Mapping[str, Any]
) -> pd.DataFrame:
    candidate_id = str(declaration["candidate_id"])
    reference_id = str(declaration["reference_id"])
    keys = ["reference_month", "evaluation_checkpoint"]
    candidate = frames[candidate_id].loc[:, [*keys, *METRICS]]
    reference = frames[reference_id].loc[:, [*keys, *METRICS]]
    paired = candidate.merge(
        reference,
        on=keys,
        how="inner",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    return paired.sort_values(keys, kind="stable").reset_index(drop=True)


def _benefit_multiplier(metric: str, arm: str) -> float:
    candidate_utility_sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    return candidate_utility_sign if arm != "leave_one_out" else -candidate_utility_sign


def _active_reference_months(
    event_audit: pd.DataFrame,
    declaration: Mapping[str, Any],
    blocks: Mapping[str, Any],
) -> set[pd.Timestamp]:
    arm = str(declaration["experiment_arm"])
    source_variant = (
        str(declaration["candidate_id"]) if arm == "add_one" else BASELINE_ID
    )
    models = set(blocks[str(declaration["block_id"])]["observation_models"])
    selected = event_audit.loc[
        event_audit["variant_id"].astype(str).eq(source_variant)
        & event_audit["observation_model_id"].astype(str).isin(models)
        & event_audit["update_status"].astype(str).eq("applied"),
        "reference_month",
    ]
    return set(pd.to_datetime(selected, errors="coerce").dropna())


def _paired_comparisons(
    frames: Mapping[str, pd.DataFrame],
    declarations: Sequence[Mapping[str, Any]],
    *,
    event_audit: pd.DataFrame | None,
    blocks: Mapping[str, Any],
) -> pd.DataFrame:
    """Return full-calendar and descriptive active-month paired deltas."""

    records: list[dict[str, object]] = []
    for declaration in declarations:
        paired = _paired_delta_frame(frames, declaration)
        active = (
            _active_reference_months(event_audit, declaration, blocks)
            if event_audit is not None
            else set()
        )
        scopes = {"full_common_calendar": paired}
        if active:
            scopes["descriptive_active_reference_months"] = paired.loc[
                paired["reference_month"].isin(active)
            ]
        for scope, scoped in scopes.items():
            for checkpoint, group in scoped.groupby("evaluation_checkpoint", sort=True):
                if group.empty:
                    continue
                record: dict[str, object] = {
                    "comparison_id": _comparison_id(
                        str(declaration["candidate_id"]),
                        str(declaration["reference_id"]),
                    ),
                    "candidate_id": str(declaration["candidate_id"]),
                    "reference_id": str(declaration["reference_id"]),
                    "block_id": str(declaration["block_id"]),
                    "experiment_arm": str(declaration["experiment_arm"]),
                    "primary_atomic": bool(declaration["primary_atomic"]),
                    "comparison_scope": scope,
                    "evaluation_checkpoint": str(checkpoint),
                    "paired_months": len(group),
                    "first_paired_month": group["reference_month"].min(),
                    "last_paired_month": group["reference_month"].max(),
                }
                for metric in METRICS:
                    delta = pd.to_numeric(
                        group[f"{metric}_candidate"], errors="raise"
                    ).astype(float) - pd.to_numeric(
                        group[f"{metric}_reference"], errors="raise"
                    ).astype(float)
                    multiplier = _benefit_multiplier(metric, str(declaration["experiment_arm"]))
                    record[f"mean_delta_{metric}"] = float(delta.mean())
                    record[f"median_delta_{metric}"] = float(delta.median())
                    record[f"mean_block_benefit_{metric}"] = float(multiplier * delta.mean())
                records.append(record)
    return pd.DataFrame.from_records(records)


def _holm_adjust(values: pd.Series) -> pd.Series:
    valid = values.dropna().sort_values(kind="stable")
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted.loc[index] = running
    return adjusted


def _segment_aware_circular_means(
    segments: Sequence[np.ndarray],
    *,
    replications: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample each contiguous calendar segment without bridging a gap."""

    total = sum(len(segment) for segment in segments)
    if not total or any(not len(segment) for segment in segments):
        raise ValueError("bootstrap segments must all be nonempty")
    sampled_sum = np.zeros(replications, dtype=float)
    for segment in segments:
        block_count = int(math.ceil(len(segment) / block_length))
        starts = rng.integers(0, len(segment), size=(replications, block_count))
        offsets = np.arange(block_length, dtype=int)
        indices = (starts[:, :, None] + offsets[None, None, :]) % len(segment)
        indices = indices.reshape(replications, -1)[:, : len(segment)]
        sampled_sum += segment[indices].sum(axis=1)
    return sampled_sum / float(total)


def _paired_bootstrap(
    frames: Mapping[str, pd.DataFrame],
    declarations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    family_override: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Bootstrap calendar-ordered paired deltas with explicit references."""

    bootstrap = config["evaluation"]["paired_block_bootstrap"]
    replications = int(bootstrap["replications"])
    block_length = int(bootstrap["block_length_months"])
    confidence = float(bootstrap["confidence_level"])
    seed = int(bootstrap["random_seed"])
    metrics = {str(key): str(value) for key, value in config["evaluation"]["metrics"].items()}
    samples = tuple(str(value) for value in bootstrap["samples"])
    alpha = (1.0 - confidence) / 2.0
    records: list[dict[str, object]] = []
    for declaration in declarations:
        paired = _paired_delta_frame(frames, declaration)
        comparison_id = _comparison_id(
            str(declaration["candidate_id"]), str(declaration["reference_id"])
        )
        for sample_id in samples:
            if sample_id == "full_sample":
                sample_segments = (paired,)
            elif sample_id == "excluding_2020_03_through_2020_05":
                sample_segments = (
                    paired.loc[paired["reference_month"].lt("2020-03-01")],
                    paired.loc[paired["reference_month"].gt("2020-05-01")],
                )
            else:
                raise ValueError(f"unsupported attribution sample {sample_id}")
            for checkpoint in config["evaluation"]["information_stage_checkpoints"]:
                checkpoint_segments = tuple(
                    segment.loc[
                        segment["evaluation_checkpoint"].astype(str).eq(str(checkpoint))
                    ].sort_values("reference_month", kind="stable")
                    for segment in sample_segments
                )
                for metric, direction in metrics.items():
                    delta_segments = tuple(
                        (
                            pd.to_numeric(
                                group[f"{metric}_candidate"], errors="raise"
                            ).astype(float)
                            - pd.to_numeric(
                                group[f"{metric}_reference"], errors="raise"
                            ).astype(float)
                        ).to_numpy(dtype=float)
                        for group in checkpoint_segments
                    )
                    delta = np.concatenate(delta_segments)
                    if not len(delta) or not np.isfinite(delta).all():
                        raise ValueError("paired bootstrap requires a complete monthly metric")
                    identity = (
                        f"{seed}|{sample_id}|{comparison_id}|{checkpoint}|{metric}"
                    ).encode("utf-8")
                    local_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
                    rng = np.random.default_rng(local_seed)
                    boot_means = _segment_aware_circular_means(
                        delta_segments,
                        replications=replications,
                        block_length=block_length,
                        rng=rng,
                    )
                    observed = float(delta.mean())
                    null_rng = np.random.default_rng(local_seed)
                    null_means = _segment_aware_circular_means(
                        tuple(segment - observed for segment in delta_segments),
                        replications=replications,
                        block_length=block_length,
                        rng=null_rng,
                    )
                    p_value = float(
                        (1 + np.count_nonzero(np.abs(null_means) >= abs(observed)))
                        / (replications + 1)
                    )
                    candidate_better = (
                        boot_means < 0.0 if direction == "lower" else boot_means > 0.0
                    )
                    multiplier = _benefit_multiplier(
                        metric, str(declaration["experiment_arm"])
                    )
                    benefit = multiplier * boot_means
                    records.append(
                        {
                            "evaluation_sample": sample_id,
                            "comparison_id": comparison_id,
                            "candidate_id": str(declaration["candidate_id"]),
                            "reference_id": str(declaration["reference_id"]),
                            "block_id": str(declaration["block_id"]),
                            "experiment_arm": str(declaration["experiment_arm"]),
                            "primary_atomic": bool(declaration["primary_atomic"]),
                            "evaluation_checkpoint": str(checkpoint),
                            "metric": metric,
                            "better_direction": direction,
                            "paired_months": len(delta),
                            "mean_candidate_minus_reference": observed,
                            "median_candidate_minus_reference": float(np.median(delta)),
                            "bootstrap_lower": float(np.quantile(boot_means, alpha)),
                            "bootstrap_upper": float(np.quantile(boot_means, 1.0 - alpha)),
                            "bootstrap_probability_candidate_better": float(candidate_better.mean()),
                            "mean_block_benefit": float(multiplier * observed),
                            "block_benefit_lower": float(np.quantile(benefit, alpha)),
                            "block_benefit_upper": float(np.quantile(benefit, 1.0 - alpha)),
                            "bootstrap_probability_block_beneficial": float((benefit > 0.0).mean()),
                            "centered_two_sided_p_value": p_value,
                            "block_length_months": block_length,
                            "calendar_segments": len(delta_segments),
                            "replications": replications,
                            "random_seed": seed,
                        }
                    )
    result = pd.DataFrame.from_records(records)
    families = family_override or bootstrap["holm_families"]
    group_keys = ["evaluation_sample", "evaluation_checkpoint", "metric"]
    for family_id, members in families.items():
        column = f"holm_adjusted_p_value_{family_id}"
        result[column] = np.nan
        member_set = set(str(value) for value in members)
        for _, group in result.loc[result["comparison_id"].isin(member_set)].groupby(
            group_keys, sort=False
        ):
            result.loc[group.index, column] = _holm_adjust(
                group["centered_two_sided_p_value"]
            )
    return result.sort_values(
        ["evaluation_sample", "evaluation_checkpoint", "metric", "comparison_id"],
        kind="stable",
    ).reset_index(drop=True)


def _block_coverage(
    event_audit: pd.DataFrame, blocks: Mapping[str, Any]
) -> pd.DataFrame:
    baseline = event_audit.loc[event_audit["variant_id"].eq(BASELINE_ID)].copy()
    records: list[dict[str, object]] = []
    for block_id, declaration in blocks.items():
        models = set(str(value) for value in declaration["observation_models"])
        selected = baseline.loc[baseline["observation_model_id"].isin(models)].copy()
        status = selected["update_status"].fillna("").astype(str)
        applied = selected.loc[status.eq("applied")]
        records.append(
            {
                "block_id": str(block_id),
                "observation_models": "|".join(sorted(models)),
                "primary_atomic": bool(declaration["primary_atomic"]),
                "audit_rows": len(selected),
                "applied_updates": len(applied),
                "active_reference_months": pd.to_datetime(
                    applied["reference_month"], errors="coerce"
                ).nunique(),
                "fit_error_rows": int(status.str.startswith("fit_error:").sum()),
                "target_exact_predictive_only_rows": int(
                    status.eq("target_exact_predictive_only").sum()
                ),
                "other_non_applied_rows": int(
                    (~status.eq("applied")
                    & ~status.str.startswith("fit_error:")
                    & ~status.eq("target_exact_predictive_only")).sum()
                ),
                "first_applied_release": (
                    pd.to_datetime(applied["release_date"]).min()
                    if not applied.empty
                    else pd.NaT
                ),
                "last_applied_release": (
                    pd.to_datetime(applied["release_date"]).max()
                    if not applied.empty
                    else pd.NaT
                ),
                "median_event_weight": (
                    float(pd.to_numeric(applied["event_weight"], errors="coerce").median())
                    if not applied.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _attribution_summary(
    comparisons: pd.DataFrame,
    bootstrap: pd.DataFrame,
    coverage: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    primary_checkpoint = str(config["evaluation"]["primary_checkpoint"])
    full = comparisons.loc[
        comparisons["evaluation_checkpoint"].eq(primary_checkpoint)
        & comparisons["comparison_scope"].eq("full_common_calendar")
    ]
    inference = bootstrap.loc[
        bootstrap["evaluation_checkpoint"].eq(primary_checkpoint)
        & bootstrap["evaluation_sample"].eq("full_sample")
    ]
    records: list[dict[str, object]] = []
    for block_id, declaration in config["blocks"].items():
        record: dict[str, object] = {
            "block_id": str(block_id),
            "observation_models": "|".join(declaration["observation_models"]),
            "primary_atomic": bool(declaration["primary_atomic"]),
        }
        for arm in ("add_one", "leave_one_out"):
            row = full.loc[
                full["block_id"].eq(block_id) & full["experiment_arm"].eq(arm)
            ]
            if len(row) != 1:
                raise ValueError(f"missing primary attribution row for {block_id}/{arm}")
            comparison_id = str(row.iloc[0]["comparison_id"])
            record[f"{arm}_comparison_id"] = comparison_id
            for metric in METRICS:
                record[f"{arm}_benefit_{metric}"] = float(
                    row.iloc[0][f"mean_block_benefit_{metric}"]
                )
                test = inference.loc[
                    inference["comparison_id"].eq(comparison_id)
                    & inference["metric"].eq(metric)
                ]
                if len(test) != 1:
                    raise ValueError("missing primary bootstrap attribution row")
                record[f"{arm}_probability_beneficial_{metric}"] = float(
                    test.iloc[0]["bootstrap_probability_block_beneficial"]
                )
                holm = test.iloc[0].get("holm_adjusted_p_value_primary_atomic", np.nan)
                record[f"{arm}_holm_p_{metric}"] = (
                    np.nan if pd.isna(holm) else float(holm)
                )
        records.append(record)
    result = pd.DataFrame.from_records(records)
    return result.merge(coverage, on=["block_id", "observation_models", "primary_atomic"])


def build_existing_block_attribution(
    *, project_root: Path, config_path: Path, replay_end: object | None = None
) -> dict[str, Path]:
    """Verify lineage, run both attribution arms, and publish their results."""

    root = project_root.resolve()
    config_path = _project_path(root, config_path)
    experiment, experiment_bytes = _load_attribution_config(config_path)
    sources = experiment["sources"]

    base_path = _project_path(root, sources["base_filter_config"])
    base, base_bytes = _load_yaml(base_path)
    frozen_config_path = _project_path(root, sources["frozen_sensitivity_config"])
    frozen_config, frozen_config_bytes = _load_yaml(frozen_config_path)
    if base.get("model_id") != MODEL_ID or frozen_config.get("model_id") != MODEL_ID:
        raise ValueError("upstream configuration has the wrong model identity")
    legacy_models = frozenset(str(value) for value in base["observation_models"])
    configured_legacy = frozenset(
        str(value)
        for value in experiment["evidence_sets"]["all_current"]["observation_models"]
    )
    if legacy_models != configured_legacy:
        raise ValueError("attribution legacy allowlist differs from the base filter")

    manifest_paths = {
        "score": _project_path(root, sources["score_manifest"]),
        "mapping": _project_path(root, sources["mapping_manifest"]),
        "evidence": _project_path(root, sources["evidence_manifest"]),
        "frozen": _project_path(root, sources["frozen_sensitivity_manifest"]),
        "prior_experiment": _project_path(
            root, sources["prior_evidence_experiment_manifest"]
        ),
    }
    stage_ids = {
        "score": None,
        "mapping": "m02_probability_map",
        "evidence": "m02_release_evidence",
        "frozen": "partial_defining_and_robustness_sensitivities",
        "prior_experiment": "m02_evidence_block_inference_experiments",
    }
    manifests: dict[str, dict[str, Any]] = {}
    manifest_bytes: dict[str, bytes] = {}
    for name, path in manifest_paths.items():
        manifests[name], manifest_bytes[name] = _load_manifest(
            path, stage_id=stage_ids[name]
        )
    frozen_manifest = manifests["frozen"]
    if frozen_manifest.get("base_configuration_sha256") != sha256(base_bytes):
        raise ValueError("base filter differs from the frozen selected model")
    if frozen_manifest.get("configuration_sha256") != sha256(frozen_config_bytes):
        raise ValueError("frozen sensitivity configuration hash mismatch")
    frozen_upstream = {
        str(row["path"]): str(row["sha256"])
        for row in frozen_manifest.get("upstream_manifests", ())
    }
    for name in ("score", "mapping", "evidence"):
        relative = manifest_paths[name].relative_to(root).as_posix()
        if frozen_upstream.get(relative) != sha256(manifest_bytes[name]):
            raise ValueError(f"{name} lineage differs from the frozen baseline")
    prior_manifest = manifests["prior_experiment"]
    if not bool(prior_manifest.get("baseline_invariance_passed")):
        raise ValueError("prior replacement experiment did not preserve its baseline")
    prior_upstream = {
        str(row["path"]): str(row["sha256"])
        for row in prior_manifest.get("upstream_manifests", ())
    }
    frozen_relative = manifest_paths["frozen"].relative_to(root).as_posix()
    if prior_upstream.get(frozen_relative) != sha256(manifest_bytes["frozen"]):
        raise ValueError("prior replacement experiment uses a different frozen model")

    frozen_end = pd.Timestamp(frozen_manifest["coverage"]["replay_end"]).normalize()
    requested_end = pd.Timestamp(
        replay_end if replay_end is not None else base["calendar"]["replay_end"]
    ).normalize()
    if requested_end != frozen_end:
        raise ValueError("attribution replay_end must equal the frozen replay end")

    source_contract = {
        "score_features": "score",
        "defining_components": "score",
        "mapping_history": "mapping",
        "evidence_events": "evidence",
        "frozen_evaluation_rows": "frozen",
        "frozen_event_update_audit": "frozen",
        "frozen_partial_defining_audit": "frozen",
        "frozen_exact_score_audit": "frozen",
        "frozen_latest_marginals": "frozen",
        "prior_evidence_experiment_evaluation_rows": "prior_experiment",
    }
    source_bytes: dict[str, bytes] = {}
    for key, manifest_name in source_contract.items():
        source_bytes[key] = _verified_bytes(
            root, str(sources[key]), manifests[manifest_name]
        )

    scores = pd.read_csv(BytesIO(source_bytes["score_features"]))
    components = pd.read_csv(BytesIO(source_bytes["defining_components"]))
    mapping = pd.read_csv(BytesIO(source_bytes["mapping_history"]))
    evidence = pd.read_csv(BytesIO(source_bytes["evidence_events"]))
    frozen_evaluations = pd.read_csv(
        BytesIO(source_bytes["frozen_evaluation_rows"]), float_precision="round_trip"
    )
    frozen_events = pd.read_csv(
        BytesIO(source_bytes["frozen_event_update_audit"]),
        compression="gzip",
        float_precision="round_trip",
    )
    frozen_partial = pd.read_csv(
        BytesIO(source_bytes["frozen_partial_defining_audit"]),
        compression="gzip",
        float_precision="round_trip",
    )
    frozen_exact = pd.read_csv(
        BytesIO(source_bytes["frozen_exact_score_audit"]), float_precision="round_trip"
    )
    frozen_latest = pd.read_csv(
        BytesIO(source_bytes["frozen_latest_marginals"]), float_precision="round_trip"
    )
    prior_evaluations = pd.read_csv(
        BytesIO(source_bytes["prior_evidence_experiment_evaluation_rows"]),
        float_precision="round_trip",
    )

    run_config = _run_config(frozen_config, experiment)
    registry = variant_registry_from_config(run_config)
    prepared = prepare_observation_data(evidence, scores, base)
    result = run_inference_sensitivities(
        scores,
        mapping,
        prepared,
        components,
        base,
        run_config,
        replay_end=replay_end,
    )
    burn_in = int(experiment["evaluation"]["initialization_burn_in_months"])
    current_frames = {
        str(variant_id): group.copy()
        for variant_id, group in result.evaluations.groupby("filter_variant", sort=False)
    }
    current_frames = _eligible_frames(
        current_frames, initial_date=result.initial_date, burn_in_months=burn_in
    )
    prior_ids = {str(row["candidate_id"]) for row in experiment["replacement_comparisons"]}
    prior_frames = {
        str(variant_id): group.copy()
        for variant_id, group in prior_evaluations.loc[
            prior_evaluations["filter_variant"].isin(prior_ids)
        ].groupby("filter_variant", sort=False)
    }
    prior_frames = _eligible_frames(
        prior_frames, initial_date=result.initial_date, burn_in_months=burn_in
    )
    frames = {**current_frames, **prior_frames}

    summary = _evaluation_summary(
        result.evaluations, initial_date=result.initial_date, burn_in_months=burn_in
    )
    subperiod = _evaluation_subperiod_summary(
        result.evaluations, initial_date=result.initial_date, burn_in_months=burn_in
    )
    comparisons = _paired_comparisons(
        current_frames,
        experiment["comparisons"],
        event_audit=result.event_audit,
        blocks=experiment["blocks"],
    )
    bootstrap = _paired_bootstrap(
        current_frames, experiment["comparisons"], experiment
    )
    replacement_comparisons = _paired_comparisons(
        frames,
        experiment["replacement_comparisons"],
        event_audit=None,
        blocks=experiment["blocks"],
    )
    replacement_ids = [
        _comparison_id(str(row["candidate_id"]), str(row["reference_id"]))
        for row in experiment["replacement_comparisons"]
    ]
    replacement_bootstrap = _paired_bootstrap(
        frames,
        experiment["replacement_comparisons"],
        experiment,
        family_override={"replacement_candidates": replacement_ids},
    )
    profile_coverage = _profile_coverage(
        prepared, result.event_audit, registry, experiment["evidence_sets"]
    )
    block_coverage = _block_coverage(result.event_audit, experiment["blocks"])
    block_summary = _attribution_summary(
        comparisons, bootstrap, block_coverage, experiment
    )

    invariance_records: list[dict[str, object]] = []
    for variant_id in CONTROL_IDS:
        invariance_records.extend(
            _invariance_rows(
                result.evaluations,
                frozen_evaluations,
                artifact="evaluation_rows",
                variant_column="filter_variant",
                variant_id=variant_id,
                key_columns=(
                    "reference_month",
                    "evaluation_checkpoint",
                    "availability_date",
                    "forecast_as_of_date",
                ),
            )
        )
        invariance_records.extend(
            _invariance_rows(
                result.partial_defining_audit,
                frozen_partial,
                artifact="partial_defining_audit",
                variant_column="variant_id",
                variant_id=variant_id,
                key_columns=("release_date", "event_id", "reference_month"),
            )
        )
        invariance_records.extend(
            _invariance_rows(
                result.exact_score_audit,
                frozen_exact,
                artifact="exact_score_audit",
                variant_column="variant_id",
                variant_id=variant_id,
                key_columns=("availability_date", "reference_month"),
            )
        )
        invariance_records.extend(
            _invariance_rows(
                result.latest_marginals,
                frozen_latest,
                artifact="latest_marginals",
                variant_column="variant_id",
                variant_id=variant_id,
                key_columns=("reference_month", "relative_month", "as_of_date"),
            )
        )
    invariance_records.extend(
        _invariance_rows(
            result.event_audit,
            frozen_events,
            artifact="event_update_audit",
            variant_column="variant_id",
            variant_id=BASELINE_ID,
            key_columns=(
                "release_date",
                "observation_model_id",
                "reference_month",
                "event_instance_id",
            ),
            ignored_columns=frozenset({"fit_id"}),
        )
    )
    invariance = pd.DataFrame.from_records(invariance_records)
    if not bool(invariance["passed"].all()):
        failed = invariance.loc[
            ~invariance["passed"], ["artifact", "variant_id", "check", "column"]
        ]
        raise ValueError(
            "attribution changed a frozen control surface: "
            + repr(failed.to_dict(orient="records"))
        )

    outputs = experiment["outputs"]
    processed_dir = _project_path(root, outputs["processed_dir"])
    published_dir = _project_path(root, outputs["published_dir"])
    processed_tables = {
        "evaluation_rows": result.evaluations,
        "evaluation_summary": summary,
        "evaluation_subperiod_summary": subperiod,
        "paired_comparisons": comparisons,
        "paired_block_bootstrap": bootstrap,
        "replacement_comparisons": replacement_comparisons,
        "replacement_block_bootstrap": replacement_bootstrap,
        "block_attribution_summary": block_summary,
        "block_coverage": block_coverage,
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
        "profile_coverage": profile_coverage,
        "baseline_invariance": invariance,
    }
    processed_paths = {
        key: _namespace_path(processed_dir, outputs[key]) for key in processed_tables
    }
    for key, frame in processed_tables.items():
        _write_csv(frame, processed_paths[key])

    public_tables = {
        "public_evaluation_summary": summary,
        "public_evaluation_subperiod_summary": subperiod,
        "public_paired_comparisons": comparisons,
        "public_paired_block_bootstrap": bootstrap,
        "public_replacement_comparisons": replacement_comparisons,
        "public_replacement_block_bootstrap": replacement_bootstrap,
        "public_block_attribution_summary": block_summary,
        "public_block_coverage": block_coverage,
        "public_profile_coverage": profile_coverage,
        "public_baseline_invariance": invariance,
        "public_model_registry": registry,
    }
    public_paths = {
        key: _namespace_path(published_dir, outputs[key]) for key in public_tables
    }
    for key, frame in public_tables.items():
        _write_csv(frame, public_paths[key])

    method_summary = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "replay_start": result.initial_date.date().isoformat(),
        "replay_end": result.replay_end.date().isoformat(),
        "selected_baseline": BASELINE_ID,
        "baseline_changed": False,
        "candidate_status": "attribution_experiment_only_no_automatic_promotion",
        "primary_checkpoint": experiment["evaluation"]["primary_checkpoint"],
        "atomic_blocks": [
            block_id
            for block_id, declaration in experiment["blocks"].items()
            if bool(declaration["primary_atomic"])
        ],
        "contrast_definition": {
            "add_one": "partial_only plus block versus partial_only",
            "leave_one_out": "baseline without block versus student_t_7_combined",
            "raw_delta": "candidate minus reference",
            "block_benefit": "positive means the block improves the declared metric",
        },
        "paired_uncertainty": {
            "method": "calendar-ordered circular moving-block bootstrap",
            "excluded_interval_policy": (
                "resample pre- and post-exclusion calendar segments separately"
            ),
            "replications": int(
                experiment["evaluation"]["paired_block_bootstrap"]["replications"]
            ),
            "block_length_months": int(
                experiment["evaluation"]["paired_block_bootstrap"][
                    "block_length_months"
                ]
            ),
            "primary_multiple_testing": "Holm across all fourteen atomic contrasts",
        },
        "baseline_invariance_checks": len(invariance),
        "baseline_invariance_passed": True,
        "latest_joint_gaussian_states": _latest_joint_state_rows(result),
        "credential_policy": "no network access and no credential read",
    }
    method_summary_path = _namespace_path(
        published_dir, outputs["public_method_summary"]
    )
    _write_json(method_summary, method_summary_path)

    generated = [*processed_paths.values(), *public_paths.values(), method_summary_path]
    implementation_hashes: list[dict[str, object]] = []
    implementation_bytes: list[bytes] = []
    for relative in IMPLEMENTATION_FILES:
        content = _project_path(root, relative).read_bytes()
        implementation_bytes.append(content)
        implementation_hashes.append(
            {"path": relative, "sha256": sha256(content), "bytes": len(content)}
        )
    snapshot = hashlib.sha256()
    for content in (
        experiment_bytes,
        base_bytes,
        frozen_config_bytes,
        *manifest_bytes.values(),
        *source_bytes.values(),
        *implementation_bytes,
    ):
        snapshot.update(sha256(content).encode("ascii"))
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(root).as_posix(),
        "configuration_sha256": sha256(experiment_bytes),
        "base_configuration": base_path.relative_to(root).as_posix(),
        "base_configuration_sha256": sha256(base_bytes),
        "frozen_sensitivity_configuration": frozen_config_path.relative_to(root).as_posix(),
        "frozen_sensitivity_configuration_sha256": sha256(frozen_config_bytes),
        "data_snapshot_sha256": snapshot.hexdigest(),
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
            "controls": list(CONTROL_IDS),
            "candidate_status": "attribution_only_no_candidate_promoted",
            "atomic_comparisons": 14,
            "diagnostic_comparisons": 2,
            "replacement_comparisons": 3,
        },
        "coverage": {
            "replay_start": result.initial_date.date().isoformat(),
            "replay_end": result.replay_end.date().isoformat(),
            "evaluation_rows": len(result.evaluations),
            "event_audit_rows": len(result.event_audit),
            "invariance_checks": len(invariance),
        },
        "generated_file_hashes": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path.read_bytes()),
                "bytes": path.stat().st_size,
            }
            for path in generated
        ],
        "credential_policy": "no network access and no credential read",
    }
    manifest_path = _project_path(root, outputs["manifest"])
    _write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "summary": public_paths["public_block_attribution_summary"],
        "paired": public_paths["public_paired_comparisons"],
        "bootstrap": public_paths["public_paired_block_bootstrap"],
        "replacements": public_paths["public_replacement_comparisons"],
        "invariance": public_paths["public_baseline_invariance"],
        "method": method_summary_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m02_existing_block_attribution.yaml"),
    )
    parser.add_argument("--replay-end", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = build_existing_block_attribution(
        project_root=args.project_root,
        config_path=args.config,
        replay_end=args.replay_end,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
