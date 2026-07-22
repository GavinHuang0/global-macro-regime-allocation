"""Allocation-only Model 02 replay with causal weekly decision snapshots.

The promoted inference replay is immutable because historical manifests hash its
implementation. This module therefore forks only the orchestration function and
reuses the frozen model helpers without changing their source file. Requested
decision dates are inserted into the causal event timeline. Each decision is
read immediately after a possible month roll and before every release on that
calendar date.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from regime_allocation.models.m02_soft_composite import inference_sensitivities as _core


ALL_COMPONENTS = _core.ALL_COMPONENTS
EmissionProfile = _core.EmissionProfile
InferenceSensitivityResult = _core.InferenceSensitivityResult
JointGaussianState = _core.JointGaussianState
PreparedObservationData = _core.PreparedObservationData
REGIME_ORDER = _core.REGIME_ORDER
STATE_NAMES = _core.STATE_NAMES
SensitivityVariant = _core.SensitivityVariant
VarDynamics = _core.VarDynamics

_dynamics = _core._dynamics
_evaluation_record = _core._evaluation_record
_finite_degrees = _core._finite_degrees
_fit_signature = _core._fit_signature
_fit_var = _core._fit_var
_hard_quadrant = _core._hard_quadrant
_initialization_date = _core._initialization_date
_mapping_asof = _core._mapping_asof
_mapping_covariance = _core._mapping_covariance
_mapping_for_exact_score = _core._mapping_for_exact_score
_normalize_mapping_history = _core._normalize_mapping_history
_normalize_scores = _core._normalize_scores
_parameters_for_year = _core._parameters_for_year
_profile_for_event = _core._profile_for_event
_selected_var_pairs = _core._selected_var_pairs
_timestamp = _core._timestamp
_transition_fit_record = _core._transition_fit_record
annual_rolling_origin_tail_scores = _core.annual_rolling_origin_tail_scores
apply_partial_defining_event = _core.apply_partial_defining_event
build_var_pair_audit = _core.build_var_pair_audit
causal_annual_hyperparameter_schedule = _core.causal_annual_hyperparameter_schedule
condition_on_exact_score = _core.condition_on_exact_score
condition_on_exact_score_axes = _core.condition_on_exact_score_axes
fit_causal_component_gaussian = _core.fit_causal_component_gaussian
fit_linear_student_t_emission_fixed = _core.fit_linear_student_t_emission_fixed
initialize_joint_exact = _core.initialize_joint_exact
monthly_quadrant_probabilities = _core.monthly_quadrant_probabilities
prepare_partial_defining_data = _core.prepare_partial_defining_data
robust_var1_weight_audit = _core.robust_var1_weight_audit
roll_joint_gaussian = _core.roll_joint_gaussian
stronger_retail_shrinkage_spec = _core.stronger_retail_shrinkage_spec
update_joint_gaussian = _core.update_joint_gaussian
variants_from_config = _core.variants_from_config


_DECISION_MARGINAL_COLUMNS = (
    "variant_id",
    "model_role",
    "signal_date",
    "as_of_date",
    "same_day_release_evidence_included",
    "target_month",
    "reference_month",
    "relative_month",
    "growth_mean",
    "inflation_mean",
    "growth_variance",
    "growth_inflation_covariance",
    "inflation_variance",
    "exact_score",
    "mapping_reference_month",
    "mapping_available_at",
    "mapping_status",
    "mapping_specification_id",
    "mapping_revision_horizon_months",
    "growth_mapping_variance",
    "growth_inflation_mapping_covariance",
    "inflation_mapping_variance",
    *(f"probability_{regime}" for regime in REGIME_ORDER),
)


@dataclass
class WeeklyDecisionReplayResult(InferenceSensitivityResult):
    """Frozen replay products plus allocation-only decision marginals."""

    decision_marginals: pd.DataFrame


def _normalize_decision_dates(values: Sequence[object]) -> frozenset[pd.Timestamp]:
    """Return distinct UTC-naive calendar dates requested for decision snapshots."""

    if isinstance(values, (str, bytes)):
        raise ValueError("decision_dates must be a sequence of date-like values")
    return frozenset(
        _timestamp(value, label="decision_date", normalize=True) for value in values
    )


def _decision_marginal_records(
    states: Mapping[str, JointGaussianState],
    variants: Sequence[SensitivityVariant],
    mappings: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
) -> list[dict[str, object]]:
    """Read current-month marginals before any releases on signal_date."""

    mapping = _mapping_asof(mappings, signal_date, allow_same_day=False)
    mapping_covariance = None if mapping is None else _mapping_covariance(mapping)
    records: list[dict[str, object]] = []
    for variant in variants:
        state = states[variant.variant_id]
        reference_month = pd.Timestamp(state.reference_months[-1])
        expected_month = signal_date.to_period("M").to_timestamp()
        if reference_month != expected_month:
            raise RuntimeError(
                "weekly decision state is not rolled through its signal month: "
                f"{signal_date.date().isoformat()} -> {reference_month.date().isoformat()}"
            )
        mean, covariance = state.marginal(reference_month)
        revision_horizon = None
        specification_id = None
        if mapping is not None:
            if "revision_horizon_months" in mapping.index and pd.notna(
                mapping["revision_horizon_months"]
            ):
                revision_horizon = int(mapping["revision_horizon_months"])
            if "specification_id" in mapping.index and pd.notna(
                mapping["specification_id"]
            ):
                specification_id = str(mapping["specification_id"])
        record: dict[str, object] = {
            "variant_id": variant.variant_id,
            "model_role": variant.model_role,
            "signal_date": signal_date,
            "as_of_date": signal_date,
            "same_day_release_evidence_included": False,
            "target_month": reference_month,
            "reference_month": reference_month,
            "relative_month": 0,
            "growth_mean": float(mean[0]),
            "inflation_mean": float(mean[1]),
            "growth_variance": float(covariance[0, 0]),
            "growth_inflation_covariance": float(covariance[0, 1]),
            "inflation_variance": float(covariance[1, 1]),
            "exact_score": bool(state.exact_mask[-1]),
            "mapping_reference_month": (
                pd.NaT if mapping is None else pd.Timestamp(mapping["reference_month"])
            ),
            "mapping_available_at": (
                pd.NaT if mapping is None else pd.Timestamp(mapping["score_available_at"])
            ),
            "mapping_status": (
                "unavailable" if mapping is None else str(mapping["mapping_status"])
            ),
            "mapping_specification_id": specification_id,
            "mapping_revision_horizon_months": revision_horizon,
            "growth_mapping_variance": (
                float("nan")
                if mapping_covariance is None
                else float(mapping_covariance[0, 0])
            ),
            "growth_inflation_mapping_covariance": (
                float("nan")
                if mapping_covariance is None
                else float(mapping_covariance[0, 1])
            ),
            "inflation_mapping_variance": (
                float("nan")
                if mapping_covariance is None
                else float(mapping_covariance[1, 1])
            ),
        }
        for regime in REGIME_ORDER:
            record[f"probability_{regime}"] = float("nan")
        if mapping_covariance is not None:
            probabilities = monthly_quadrant_probabilities(
                state, reference_month, mapping_covariance
            )
            for regime in REGIME_ORDER:
                record[f"probability_{regime}"] = float(probabilities[regime])
        records.append(record)
    return records


def run_weekly_decision_replay(
    score_history: pd.DataFrame,
    mapping_history: pd.DataFrame,
    prepared: PreparedObservationData,
    first_release_components: pd.DataFrame,
    base_config: Mapping[str, Any],
    sensitivity_config: Mapping[str, Any],
    *,
    real_retail_prepared: PreparedObservationData | None = None,
    replay_end: object | None = None,
    decision_dates: Sequence[object] = (),
) -> WeeklyDecisionReplayResult:
    """Replay variants causally and capture requested start-of-day decisions."""

    scores = _normalize_scores(score_history)
    mappings = _normalize_mapping_history(
        mapping_history,
        baseline_revision_horizon_months=int(
            base_config["mapping"]["baseline_revision_horizon_months"]
        ),
    )
    variants = variants_from_config(sensitivity_config)
    end = _timestamp(
        replay_end
        if replay_end is not None
        else base_config.get("calendar", {}).get("replay_end"),
        label="replay_end",
        normalize=True,
    )
    requested_decision_dates = _normalize_decision_dates(decision_dates)
    partial = prepare_partial_defining_data(first_release_components, score_history)
    partial_config = sensitivity_config["partial_defining_releases"]
    student_config = sensitivity_config["student_t_emissions"]
    irls_config = student_config["irls"]
    dependence_config = sensitivity_config.get("dependence_diagnostics", {})
    diagnostic_variant_id = str(dependence_config.get("variant_id", "")).strip()
    if diagnostic_variant_id and diagnostic_variant_id not in {
        variant.variant_id for variant in variants
    }:
        raise ValueError("dependence diagnostic names an unknown variant")

    profiles: dict[str, EmissionProfile] = {}
    for model_id, spec in prepared.specifications.items():
        profiles[f"base:{model_id}"] = EmissionProfile(
            profile_id=f"base:{model_id}",
            observation_model_id=model_id,
            spec=spec,
            training=prepared.training_tables[model_id],
        )
    if any(
        variant.retail_method == "stronger_inflation_shrinkage"
        for variant in variants
    ):
        consumer = prepared.specifications["consumer_demand"]
        strong_config = sensitivity_config["retail_sensitivities"][
            "stronger_inflation_shrinkage"
        ]
        profiles["retail_stronger"] = EmissionProfile(
            profile_id="retail_stronger",
            observation_model_id="consumer_demand",
            spec=stronger_retail_shrinkage_spec(
                consumer, strong_config["loading_penalties"]
            ),
            training=prepared.training_tables["consumer_demand"],
        )
    if any(variant.retail_method == "real_decomposition" for variant in variants):
        if real_retail_prepared is None:
            raise ValueError("real-decomposition variant requires prepared retail events")
        real_model_id = "consumer_demand_real_decomposition"
        profiles["retail_real"] = EmissionProfile(
            profile_id="retail_real",
            observation_model_id=real_model_id,
            spec=real_retail_prepared.specifications[real_model_id],
            training=real_retail_prepared.training_tables[real_model_id],
        )

    for variant in variants:
        if variant.observation_model_ids is None:
            continue
        unknown = set(variant.observation_model_ids).difference(
            set(prepared.specifications)
        )
        if unknown:
            raise ValueError(
                f"variant {variant.variant_id} evidence set contains unknown models: "
                + ", ".join(sorted(unknown))
            )

    degrees_grid = _finite_degrees(
        student_config["degrees_of_freedom_sensitivity"]["candidates"]
    )
    families_by_profile: dict[str, set[str]] = {}
    for profile_id, profile in profiles.items():
        for variant in variants:
            if not variant.non_defining_evidence:
                continue
            if _profile_for_event(variant, profile.observation_model_id) == profile_id:
                families_by_profile.setdefault(profile_id, set()).add(
                    variant.emission_family
                )
    used_profiles = {
        profile_id: profile
        for profile_id, profile in profiles.items()
        if profile_id in families_by_profile
    }
    if not used_profiles:
        raise ValueError("no evidence profile is enabled by any runnable variant")

    fold_parts: list[pd.DataFrame] = []
    for profile_id, profile in used_profiles.items():
        families = families_by_profile[profile_id]
        if "selected_student_t" in families:
            profile_degrees = degrees_grid
        else:
            selected: list[float] = []
            if "student_t_7" in families:
                selected.append(7.0)
            if "gaussian" in families:
                selected.append(math.inf)
            profile_degrees = tuple(selected)
        fold_parts.append(
            annual_rolling_origin_tail_scores(
                profile,
                degrees_of_freedom_grid=profile_degrees,
                maximum_iterations=int(irls_config["maximum_iterations"]),
                tolerance=float(irls_config["relative_tolerance"]),
                minimum_weight=float(irls_config["minimum_weight"]),
            )
        )
    fold_scores = pd.concat(fold_parts, ignore_index=True, sort=False)

    first_year = int(scores["score_available_at"].dt.year.min())
    final_year = int(end.year)
    minimum_predictive_events = int(
        student_config["degrees_of_freedom_sensitivity"]["minimum_predictive_events"]
    )
    schedule_parts: list[pd.DataFrame] = []
    family_settings = {
        "gaussian": ((math.inf,), math.inf),
        "student_t_7": ((7.0,), 7.0),
        "selected_student_t": (degrees_grid, 7.0),
    }
    for profile_id, profile in used_profiles.items():
        for family in sorted(families_by_profile[profile_id]):
            allowed, fallback_degrees = family_settings[family]
            schedule = causal_annual_hyperparameter_schedule(
                fold_scores,
                profile_id=profile.profile_id,
                first_year=first_year,
                final_year=final_year,
                allowed_degrees_of_freedom=allowed,
                minimum_predictive_events=minimum_predictive_events,
                fallback_lambda=max(profile.spec.lambda_grid),
                fallback_degrees_of_freedom=fallback_degrees,
            )
            schedule["emission_family"] = family
            schedule_parts.append(schedule)
    schedules = pd.concat(schedule_parts, ignore_index=True, sort=False)

    pair_audit = build_var_pair_audit(scores.set_index("reference_month"))
    minimum_pairs = int(base_config["transition"]["minimum_training_pairs"])
    start, initial_pairs = _initialization_date(
        scores, pair_audit, replay_end=end, minimum_pairs=minimum_pairs
    )
    outside_replay = sorted(
        date for date in requested_decision_dates if date < start or date > end
    )
    if outside_replay:
        formatted = ", ".join(date.date().isoformat() for date in outside_replay)
        raise ValueError(
            "decision_dates must fall within the initialized replay window "
            f"[{start.date().isoformat()}, {end.date().isoformat()}]: {formatted}"
        )
    score_lookup = scores.set_index("reference_month")
    oldest = (start.to_period("M") - 3).to_timestamp()
    oldest_score = score_lookup.loc[oldest, list(STATE_NAMES)].to_numpy(dtype=float)
    initial_by_var: dict[str, JointGaussianState] = {}
    transition_records: list[dict[str, object]] = []
    transition_weight_parts: list[pd.DataFrame] = []
    robust_var_config = sensitivity_config["robust_var"]
    for method in sorted({variant.var_method for variant in variants}):
        fit = _fit_var(initial_pairs, method, robust_var_config)
        edge = _dynamics(fit)
        state = initialize_joint_exact(oldest, oldest_score, [edge, edge, edge])
        for month in state.reference_months[1:]:
            if month in score_lookup.index:
                row = score_lookup.loc[month]
                if pd.Timestamp(row["score_available_at"]) < start:
                    state = condition_on_exact_score(
                        state, month, row.loc[list(STATE_NAMES)].to_numpy(dtype=float)
                    )
        initial_by_var[method] = state
        transition_records.append(
            _transition_fit_record(fit, method=method, fit_date=start, selected=initial_pairs)
        )
        weights = robust_var1_weight_audit(initial_pairs, fit)
        weights.insert(0, "fit_date", start)
        weights.insert(1, "method", method)
        transition_weight_parts.append(weights)
    states = {
        variant.variant_id: initial_by_var[variant.var_method]
        for variant in variants
    }

    evidence_events = prepared.events.copy()
    if real_retail_prepared is not None:
        evidence_events = pd.concat(
            [evidence_events, real_retail_prepared.events], ignore_index=True, sort=False
        )
    if not evidence_events.empty:
        evidence_events = evidence_events.loc[
            evidence_events["release_date"].between(start, end, inclusive="both")
        ].copy()
    partial_events = partial.events.loc[
        partial.events["release_date"].between(start, end, inclusive="both")
    ].copy()
    exact_rows = scores.loc[
        scores["score_available_at"].between(start, end, inclusive="both")
    ].copy()
    timeline = sorted(
        {start, end}
        .union(pd.date_range(start=start, end=end, freq="MS"))
        .union(evidence_events.get("release_date", pd.Series(dtype="datetime64[ns]")))
        .union(partial_events.get("release_date", pd.Series(dtype="datetime64[ns]")))
        .union(exact_rows["score_available_at"])
        .union(requested_decision_dates)
    )

    variant_lookup = {variant.variant_id: variant for variant in variants}
    fit_cache: dict[tuple[object, ...], Any] = {}
    fit_id_cache: dict[tuple[object, ...], str] = {}
    observed_components: dict[pd.Timestamp, dict[str, float]] = {}
    snapshots: dict[tuple[str, pd.Timestamp, str], tuple[pd.Timestamp, JointGaussianState]] = {}
    event_records: list[dict[str, object]] = []
    partial_records: list[dict[str, object]] = []
    exact_records: list[dict[str, object]] = []
    fit_records: list[dict[str, object]] = []
    dependence_records: list[dict[str, object]] = []
    evaluation_records: list[dict[str, object]] = []
    decision_records: list[dict[str, object]] = []

    def snapshot(stage: str, date: pd.Timestamp, month: pd.Timestamp, *, overwrite: bool) -> None:
        for variant_id, state in states.items():
            if month < state.reference_months[0] or month > state.reference_months[-1]:
                continue
            key = (variant_id, month, stage)
            if overwrite or key not in snapshots:
                snapshots[key] = (date, state)

    for current_date in timeline:
        current_date = pd.Timestamp(current_date)
        if current_date > start and current_date.day == 1:
            source_month = next(iter(states.values())).reference_months[-1]
            selected = _selected_var_pairs(
                pair_audit,
                source_month=source_month,
                cutoff=current_date,
                minimum_pairs=minimum_pairs,
            )
            dynamics_by_method: dict[str, VarDynamics] = {}
            for method in sorted({variant.var_method for variant in variants}):
                fit = _fit_var(selected, method, robust_var_config)
                dynamics_by_method[method] = _dynamics(fit)
                transition_records.append(
                    _transition_fit_record(
                        fit, method=method, fit_date=current_date, selected=selected
                    )
                )
                weights = robust_var1_weight_audit(selected, fit)
                weights.insert(0, "fit_date", current_date)
                weights.insert(1, "method", method)
                transition_weight_parts.append(weights)
            for variant in variants:
                states[variant.variant_id] = roll_joint_gaussian(
                    states[variant.variant_id], dynamics_by_method[variant.var_method]
                )

        if current_date in requested_decision_dates:
            decision_records.extend(
                _decision_marginal_records(
                    states,
                    variants,
                    mappings,
                    signal_date=current_date,
                )
            )

        date_partial = partial_events.loc[
            partial_events["release_date"].eq(current_date)
        ].copy()
        for month in sorted(date_partial["reference_month"].unique()):
            reference_month = pd.Timestamp(month)
            if not observed_components.get(reference_month):
                snapshot(
                    "before_any_defining_release",
                    current_date,
                    reference_month,
                    overwrite=False,
                )

        date_exact = exact_rows.loc[exact_rows["score_available_at"].eq(current_date)]
        for row in date_exact.itertuples(index=False):
            reference_month = pd.Timestamp(row.reference_month)
            snapshot(
                "strict_pre_final_score_day",
                current_date,
                reference_month,
                overwrite=True,
            )

        date_evidence = evidence_events.loc[
            evidence_events["release_date"].eq(current_date)
        ].copy()
        # Fit systems once, but compute every robust event weight against each
        # variant's shared pre-release-day state before applying any factor.
        pre_release_states = dict(states)
        pending: list[tuple[str, pd.Timestamp, dict[str, np.ndarray], dict[str, object]]] = []
        for row in date_evidence.sort_values(
            ["observation_model_id", "reference_month", "event_instance_id"],
            kind="mergesort",
        ).itertuples(index=False):
            model_id = str(row.observation_model_id)
            reference_month = pd.Timestamp(row.reference_month)
            for variant in variants:
                if not variant.non_defining_evidence:
                    continue
                profile_id = _profile_for_event(variant, model_id)
                if profile_id is None:
                    continue
                profile = profiles[profile_id]
                state = pre_release_states[variant.variant_id]
                base_audit: dict[str, object] = {
                    "release_date": current_date,
                    "variant_id": variant.variant_id,
                    "profile_id": profile_id,
                    "observation_model_id": model_id,
                    "event_instance_id": row.event_instance_id,
                    "reference_month": reference_month,
                    "emission_family": variant.emission_family,
                }
                if reference_month < state.reference_months[0]:
                    base_audit["update_status"] = "expired_target_outside_path"
                    event_records.append(base_audit)
                    continue
                if reference_month > state.reference_months[-1]:
                    base_audit["update_status"] = "future_target_outside_path"
                    event_records.append(base_audit)
                    continue
                observation = {
                    name: getattr(row, name)
                    for name in profile.spec.response_names
                    if hasattr(row, name) and pd.notna(getattr(row, name))
                }
                controls = {
                    name: getattr(row, name)
                    for name in profile.spec.control_names
                    if hasattr(row, name)
                }
                if not observation:
                    base_audit["update_status"] = "no_observed_response"
                    event_records.append(base_audit)
                    continue
                if len(controls) != len(profile.spec.control_names) or any(
                    pd.isna(value) for value in controls.values()
                ):
                    base_audit["update_status"] = "missing_control"
                    event_records.append(base_audit)
                    continue
                base_lambda, degrees, selection_status = _parameters_for_year(
                    schedules,
                    profile_id,
                    variant.emission_family,
                    current_date.year,
                )
                signature, count, _ = _fit_signature(
                    profile.training,
                    cutoff=current_date,
                    active_controls=profile.spec.control_names,
                )
                cache_key = (
                    profile_id,
                    signature,
                    base_lambda,
                    degrees,
                    profile.spec.control_names,
                )
                fit = fit_cache.get(cache_key)
                fit_status = "cached_fit"
                if fit is None:
                    try:
                        fit = fit_linear_student_t_emission_fixed(
                            profile.training,
                            spec=profile.spec,
                            availability_column="training_available_at",
                            knowledge_cutoff=current_date,
                            base_lambda=base_lambda,
                            degrees_of_freedom=degrees,
                            maximum_iterations=int(irls_config["maximum_iterations"]),
                            tolerance=float(irls_config["relative_tolerance"]),
                            minimum_weight=float(irls_config["minimum_weight"]),
                        )
                    except (ValueError, np.linalg.LinAlgError) as error:
                        base_audit.update(
                            {
                                "update_status": f"fit_error:{error}",
                                "training_count": int(count),
                                "selected_base_lambda": base_lambda,
                                "selected_degrees_of_freedom": degrees,
                                "selection_status": selection_status,
                            }
                        )
                        event_records.append(base_audit)
                        continue
                    fit_cache[cache_key] = fit
                    fit_id = f"robust-emission:{len(fit_records) + 1:06d}"
                    fit_id_cache[cache_key] = fit_id
                    fit_status = "new_fit"
                    fit_records.append(
                        {
                            "fit_id": fit_id,
                            "fit_date": current_date,
                            "profile_id": profile_id,
                            "observation_model_id": model_id,
                            "training_signature": signature,
                            "training_count": int(fit.training_count),
                            "last_training_available_at": (
                                fit.last_training_available_at
                            ),
                            "strict_fit_cutoff": current_date,
                            "selected_base_lambda": base_lambda,
                            "selected_degrees_of_freedom": degrees,
                            "selection_status": selection_status,
                            "irls_iterations": int(fit.irls_iterations),
                            "irls_converged": bool(fit.irls_converged),
                            "ledoit_wolf_shrinkage": float(fit.ledoit_wolf_shrinkage),
                        }
                    )
                fit_id = fit_id_cache[cache_key]
                try:
                    system = fit.observed_system(
                        observation,
                        controls=controls if profile.spec.control_names else None,
                    )
                    prior_mean, prior_covariance = state.marginal(reference_month)
                    approximation = system.approximate_gaussian_system(
                        prior_mean,
                        prior_covariance,
                        minimum_event_weight=float(irls_config["minimum_weight"]),
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    base_audit["update_status"] = f"system_error:{error}"
                    event_records.append(base_audit)
                    continue
                base_audit.update(
                    {
                        "fit_id": fit_id,
                        "fit_status": fit_status,
                        "training_count": int(fit.training_count),
                        "selection_status": selection_status,
                        "selected_base_lambda": base_lambda,
                        "selected_degrees_of_freedom": degrees,
                        "observed_responses": "|".join(system.observed_names),
                        "event_weight": float(approximation.event_weight),
                        "raw_event_weight": float(approximation.raw_event_weight),
                        "mahalanobis_squared": float(approximation.mahalanobis_squared),
                        "approximate_log_predictive_density": float(
                            approximation.approximate_log_predictive_density
                        ),
                        "weight_state_cutoff": "shared_pre_release_day",
                    }
                )
                if variant.variant_id == diagnostic_variant_id:
                    if reference_month in score_lookup.index:
                        target = score_lookup.loc[reference_month]
                        final_score = target.loc[list(STATE_NAMES)].to_numpy(dtype=float)
                        residual = (
                            system.adjusted_observation
                            - system.state_loadings @ final_score
                        )
                        whitened = np.linalg.solve(
                            np.linalg.cholesky(system.residual_scale), residual
                        )
                        marginal_scale = np.sqrt(np.diag(system.residual_scale))
                        last_training = pd.Timestamp(fit.last_training_available_at)
                        if not last_training < current_date:
                            raise RuntimeError(
                                "dependence residual fit is not strictly causal"
                            )
                        for position, response_id in enumerate(system.observed_names):
                            dependence_records.append(
                                {
                                    "variant_id": variant.variant_id,
                                    "block_id": str(row.economic_block),
                                    "model_id": model_id,
                                    "response_id": response_id,
                                    "reference_month": reference_month,
                                    "observation_date": pd.Timestamp(row.reference_date),
                                    "validation_available_at": current_date,
                                    "fit_cutoff": current_date,
                                    "last_training_available_at": last_training,
                                    "target_score_available_at": pd.Timestamp(
                                        target["score_available_at"]
                                    ),
                                    "target_usage": (
                                        "retrospective_completed_first_release_score_only"
                                    ),
                                    "raw_residual": float(residual[position]),
                                    "marginal_standardized_residual": float(
                                        residual[position] / marginal_scale[position]
                                    ),
                                    "standardized_residual": float(whitened[position]),
                                    "event_weight_from_live_prior": float(
                                        approximation.event_weight
                                    ),
                                }
                            )
                pending.append(
                    (
                        variant.variant_id,
                        reference_month,
                        approximation.to_joint_filter_inputs(),
                        base_audit,
                    )
                )
        for variant_id, reference_month, system_inputs, audit in pending:
            try:
                update = update_joint_gaussian(
                    states[variant_id], reference_month, **system_inputs
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                audit["update_status"] = f"update_error:{error}"
            else:
                states[variant_id] = update.state
                audit["update_status"] = (
                    "applied" if update.applied else "target_exact_predictive_only"
                )
                audit["gaussian_moment_log_predictive_density"] = float(
                    update.log_predictive_density
                )
            event_records.append(audit)

        if not date_partial.empty:
            component_fit = None
            try:
                component_fit = fit_causal_component_gaussian(
                    partial.component_history,
                    current_date,
                    minimum_training_samples=int(
                        partial_config["minimum_complete_training_months"]
                    ),
                    covariance_eigenvalue_floor=float(
                        partial_config["covariance_eigenvalue_floor"]
                    ),
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                component_fit_error = str(error)
            else:
                component_fit_error = ""
            for event_id, group in date_partial.groupby("event_id", sort=True):
                reference_month = pd.Timestamp(group["reference_month"].iloc[0])
                new = {
                    str(row.component): float(row.component_z)
                    for row in group.itertuples(index=False)
                }
                previous = observed_components.get(reference_month, {})
                after_names = set(previous).union(new)
                completes_both = set(ALL_COMPONENTS).issubset(after_names)
                if completes_both and bool(
                    (date_exact["reference_month"] == reference_month).any()
                ):
                    for variant in variants:
                        partial_records.append(
                            {
                                "release_date": current_date,
                                "variant_id": variant.variant_id,
                                "event_id": event_id,
                                "reference_month": reference_month,
                                "components": "|".join(new),
                                "status": "skipped_block_completing_both_scores_exact_end_of_day",
                            }
                        )
                    continue
                if component_fit is None:
                    for variant in variants:
                        partial_records.append(
                            {
                                "release_date": current_date,
                                "variant_id": variant.variant_id,
                                "event_id": event_id,
                                "reference_month": reference_month,
                                "components": "|".join(new),
                                "status": f"fit_error:{component_fit_error}",
                            }
                        )
                    continue
                successful_observed: Mapping[str, float] | None = None
                for variant in variants:
                    if not variant.partial_defining_releases:
                        partial_records.append(
                            {
                                "release_date": current_date,
                                "variant_id": variant.variant_id,
                                "event_id": event_id,
                                "reference_month": reference_month,
                                "components": "|".join(new),
                                "status": "partial_defining_suppressed",
                            }
                        )
                        continue
                    state = states[variant.variant_id]
                    if reference_month < state.reference_months[0] or reference_month > state.reference_months[-1]:
                        partial_records.append(
                            {
                                "release_date": current_date,
                                "variant_id": variant.variant_id,
                                "event_id": event_id,
                                "reference_month": reference_month,
                                "components": "|".join(new),
                                "status": "target_outside_path",
                            }
                        )
                        continue
                    partial_update = None
                    try:
                        partial_update = apply_partial_defining_event(
                            state,
                            reference_month,
                            new,
                            previously_observed=previous,
                            component_fit=component_fit,
                            covariance_eigenvalue_floor=float(
                                partial_config["covariance_eigenvalue_floor"]
                            ),
                        )
                    except (ValueError, np.linalg.LinAlgError) as error:
                        status = f"update_error:{error}"
                        exact_axes = ""
                        likelihood_components = ""
                    else:
                        states[variant.variant_id] = partial_update.state
                        successful_observed = partial_update.observed_components
                        status = (
                            "applied" if partial_update.applied else partial_update.reason
                        )
                        exact_axes = "|".join(partial_update.exact_axes)
                        likelihood_components = "|".join(
                            partial_update.likelihood_components
                        )
                    partial_records.append(
                        {
                            "release_date": current_date,
                            "variant_id": variant.variant_id,
                            "event_id": event_id,
                            "reference_month": reference_month,
                            "components": "|".join(new),
                            "previously_observed_components": "|".join(previous),
                            "conditioned_components": (
                                ""
                                if partial_update is None
                                or partial_update.emission is None
                                else "|".join(
                                    partial_update.emission.conditioned_components
                                )
                            ),
                            "likelihood_components": likelihood_components,
                            "exact_axes": exact_axes,
                            "component_training_count": int(component_fit.sample_size),
                            "component_fit_knowledge_cutoff": component_fit.knowledge_cutoff,
                            "latest_component_training_availability": (
                                component_fit.latest_training_availability
                            ),
                            "component_ledoit_wolf_shrinkage": float(
                                component_fit.shrinkage
                            ),
                            "conditional_score_loadings_json": (
                                ""
                                if partial_update is None
                                or partial_update.emission is None
                                else json.dumps(
                                    partial_update.emission.score_loadings.tolist()
                                )
                            ),
                            "conditional_noise_covariance_json": (
                                ""
                                if partial_update is None
                                or partial_update.emission is None
                                else json.dumps(
                                    partial_update.emission.noise_covariance.tolist()
                                )
                            ),
                            "status": status,
                        }
                    )
                if successful_observed is not None:
                    observed_components[reference_month] = dict(successful_observed)
                families = set(group["release_family"].astype(str))
                if "employment_situation" in families:
                    snapshot(
                        "after_employment_situation",
                        current_date,
                        reference_month,
                        overwrite=True,
                    )
                elif "personal_income_and_outlays" not in families:
                    snapshot(
                        "after_midmonth_defining_releases",
                        current_date,
                        reference_month,
                        overwrite=True,
                    )

        if not date_exact.empty:
            pre_mapping = _mapping_asof(mappings, current_date, allow_same_day=False)
            for row in date_exact.itertuples(index=False):
                reference_month = pd.Timestamp(row.reference_month)
                exact_score = np.asarray([row.growth_score, row.inflation_score], dtype=float)
                target_mapping = _mapping_for_exact_score(
                    mappings, reference_month, current_date
                )
                for variant in variants:
                    variant_id = variant.variant_id
                    for stage in sensitivity_config["evaluation"][
                        "information_stage_checkpoints"
                    ]:
                        item = snapshots.get((variant_id, reference_month, str(stage)))
                        if item is None:
                            continue
                        as_of_date, forecast_state = item
                        forecast_mapping = _mapping_asof(
                            mappings, as_of_date, allow_same_day=False
                        )
                        evaluation = _evaluation_record(
                            state=forecast_state,
                            availability_date=current_date,
                            reference_month=reference_month,
                            exact_score=exact_score,
                            filter_variant=variant_id,
                            evaluation_checkpoint=str(stage),
                            same_day_release_evidence_included=False,
                            forecast_mapping=forecast_mapping,
                            target_mapping=target_mapping,
                        )
                        if evaluation is not None:
                            evaluation["forecast_as_of_date"] = as_of_date
                            if forecast_mapping is not None:
                                probabilities = monthly_quadrant_probabilities(
                                    forecast_state,
                                    reference_month,
                                    _mapping_covariance(forecast_mapping),
                                )
                                forecast_hard = max(probabilities, key=probabilities.get)
                                target_hard = _hard_quadrant(exact_score)
                                evaluation["forecast_hard_quadrant"] = forecast_hard
                                evaluation["target_hard_quadrant"] = target_hard
                                evaluation["hard_quadrant_correct"] = bool(
                                    forecast_hard == target_hard
                                )
                            evaluation_records.append(evaluation)
                    state = states[variant_id]
                    audit = {
                        "availability_date": current_date,
                        "reference_month": reference_month,
                        "variant_id": variant_id,
                    }
                    if reference_month < state.reference_months[0] or reference_month > state.reference_months[-1]:
                        audit["status"] = "target_outside_path"
                    else:
                        try:
                            conditioned = condition_on_exact_score_axes(
                                state,
                                reference_month,
                                {
                                    "growth": float(exact_score[0]),
                                    "inflation": float(exact_score[1]),
                                },
                            )
                        except ValueError as error:
                            audit["status"] = f"conditioning_error:{error}"
                        else:
                            states[variant_id] = conditioned
                            audit["status"] = (
                                "identical_exact_no_op"
                                if conditioned is state
                                else "conditioned_exactly"
                            )
                    exact_records.append(audit)

    latest_mapping = _mapping_asof(mappings, end, allow_same_day=True)
    latest_records: list[dict[str, object]] = []
    for variant in variants:
        state = states[variant.variant_id]
        for position, month in enumerate(state.reference_months):
            mean, covariance = state.marginal(month)
            record: dict[str, object] = {
                "variant_id": variant.variant_id,
                "as_of_date": end,
                "reference_month": month,
                "relative_month": position - 3,
                "growth_mean": float(mean[0]),
                "inflation_mean": float(mean[1]),
                "growth_variance": float(covariance[0, 0]),
                "growth_inflation_covariance": float(covariance[0, 1]),
                "inflation_variance": float(covariance[1, 1]),
                "exact_score": bool(state.exact_mask[position]),
            }
            if latest_mapping is not None:
                probabilities = monthly_quadrant_probabilities(
                    state, month, _mapping_covariance(latest_mapping)
                )
                for regime in REGIME_ORDER:
                    record[f"probability_{regime}"] = float(probabilities[regime])
            latest_records.append(record)

    transition_weights = (
        pd.concat(transition_weight_parts, ignore_index=True, sort=False)
        if transition_weight_parts
        else pd.DataFrame()
    )

    role_by_variant = {
        variant.variant_id: variant.model_role for variant in variants
    }

    def with_model_role(
        frame: pd.DataFrame,
        *,
        variant_column: str,
    ) -> pd.DataFrame:
        """Attach the validated publication role beside a variant identifier."""

        if variant_column not in frame.columns:
            return frame
        result = frame.copy()
        roles = result[variant_column].map(role_by_variant)
        if roles.isna().any():
            unknown = sorted(result.loc[roles.isna(), variant_column].astype(str).unique())
            raise ValueError(f"output contains variants without model roles: {unknown}")
        insert_at = result.columns.get_loc(variant_column) + 1
        result.insert(insert_at, "model_role", roles)
        return result

    evaluations = with_model_role(
        pd.DataFrame.from_records(evaluation_records),
        variant_column="filter_variant",
    )
    event_audit = with_model_role(
        pd.DataFrame.from_records(event_records),
        variant_column="variant_id",
    )
    partial_defining_audit = with_model_role(
        pd.DataFrame.from_records(partial_records),
        variant_column="variant_id",
    )
    exact_score_audit = with_model_role(
        pd.DataFrame.from_records(exact_records),
        variant_column="variant_id",
    )
    latest_marginals = with_model_role(
        pd.DataFrame.from_records(latest_records),
        variant_column="variant_id",
    )
    decision_marginals = pd.DataFrame.from_records(
        decision_records,
        columns=_DECISION_MARGINAL_COLUMNS,
    )
    return WeeklyDecisionReplayResult(
        evaluations=evaluations,
        event_audit=event_audit,
        partial_defining_audit=partial_defining_audit,
        exact_score_audit=exact_score_audit,
        transition_fit_audit=pd.DataFrame.from_records(transition_records),
        transition_weight_audit=transition_weights,
        emission_fit_audit=pd.DataFrame.from_records(fit_records),
        tail_fold_scores=fold_scores,
        hyperparameter_schedule=schedules,
        dependence_residuals=pd.DataFrame.from_records(dependence_records),
        latest_marginals=latest_marginals,
        latest_states=dict(states),
        decision_marginals=decision_marginals,
        initial_date=start,
        replay_end=end,
    )

__all__ = [
    "WeeklyDecisionReplayResult",
    "run_weekly_decision_replay",
]
