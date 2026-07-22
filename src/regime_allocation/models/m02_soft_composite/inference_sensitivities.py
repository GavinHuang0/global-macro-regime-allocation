"""Run the selected Model 02 baseline, benchmarks, and sensitivity replays.

This module is intentionally separate from :mod:`walkforward`, which remains
the frozen predecessor Gaussian replay. It selects a robust baseline and adds
four research changes without altering those historical artifacts:

* score-defining components update the four-month state when they are first
  released, while the final two-component PCE event is replaced by the exact
  completed score at the end of that day;
* release blocks may use robust multivariate Student-t IRLS emissions and a
  one-weight Gaussian moment update;
* the monthly score transition may use OLS, multivariate Huber, or Student-t
  VAR(1) estimates while retaining a joint Gaussian state; and
* consumer demand can be estimated with stronger inflation-loading shrinkage
  or with same-vintage real-retail and implicit-price coordinates.

Hyperparameters are selected by calendar-year rolling origins.  A fold fitted
at January 1 uses only rows whose complete training vector was available
strictly before that date and scores rows becoming available during that year.
The parameters used in year ``Y`` are selected only from completed folds with
validation year below ``Y``.  Coefficients are then refitted at each distinct
causal training signature with those preselected parameters.  This slower
annual tuning schedule is both computationally tractable and strictly causal.

Student-t event weights depend on the prior state.  Every same-day weight is
therefore calculated from a shared pre-release-day state and its effective
Gaussian factor is frozen before any factor on that day is applied.  This
prevents stable-but-arbitrary model ordering from changing the weights.

The functions perform no file or network access.  The CLI supplies verified
tables and is responsible for publishing the audit products.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from regime_allocation.models.m02_soft_composite.gaussian_emissions import (
    LinearGaussianEmissionSpec,
    select_causal_emission_rows,
)
from regime_allocation.models.m02_soft_composite.joint_filter import (
    JointGaussianState,
    VarDynamics,
    condition_on_exact_score,
    initialize_joint_exact,
    monthly_quadrant_probabilities,
    roll_joint_gaussian,
    update_joint_gaussian,
)
from regime_allocation.models.m02_soft_composite.partial_defining import (
    ALL_COMPONENTS,
    apply_partial_defining_event,
    condition_on_exact_score_axes,
    fit_causal_component_gaussian,
    prepare_partial_defining_data,
)
from regime_allocation.models.m02_soft_composite.probability_map import (
    REGIME_ORDER,
)
from regime_allocation.models.m02_soft_composite.robust_emissions import (
    fit_linear_student_t_emission_fixed,
    multivariate_student_t_nll,
)
from regime_allocation.models.m02_soft_composite.robust_var import (
    RobustVar1Fit,
    fit_var1_sensitivity,
    robust_var1_weight_audit,
)
from regime_allocation.models.m02_soft_composite.var_transition import (
    build_var_pair_audit,
    select_causal_var_pairs,
)
from regime_allocation.models.m02_soft_composite.walkforward import (
    PreparedObservationData,
    _evaluation_record,
    _fit_signature,
    _mapping_asof,
    _mapping_covariance,
    _mapping_for_exact_score,
    _normalize_mapping_history,
    _normalize_scores,
)


STATE_NAMES = ("growth_score", "inflation_score")
MODEL_ROLES = ("baseline", "major_benchmark", "sensitivity")


@dataclass(frozen=True)
class SensitivityVariant:
    """One fully declared filter variant and its publication role."""

    variant_id: str
    model_role: str
    non_defining_evidence: bool
    partial_defining_releases: bool
    emission_family: str
    var_method: str
    retail_method: str
    evidence_set_id: str = "all_configured"
    observation_model_ids: frozenset[str] | None = None


@dataclass(frozen=True)
class EmissionProfile:
    """Specification and training table shared by one fitted profile."""

    profile_id: str
    observation_model_id: str
    spec: LinearGaussianEmissionSpec
    training: pd.DataFrame


@dataclass
class InferenceSensitivityResult:
    """Compact causal replay and sensitivity audit products."""

    evaluations: pd.DataFrame
    event_audit: pd.DataFrame
    partial_defining_audit: pd.DataFrame
    exact_score_audit: pd.DataFrame
    transition_fit_audit: pd.DataFrame
    transition_weight_audit: pd.DataFrame
    emission_fit_audit: pd.DataFrame
    tail_fold_scores: pd.DataFrame
    hyperparameter_schedule: pd.DataFrame
    dependence_residuals: pd.DataFrame
    latest_marginals: pd.DataFrame
    latest_states: Mapping[str, JointGaussianState]
    initial_date: pd.Timestamp
    replay_end: pd.Timestamp


def _timestamp(value: object, *, label: str, normalize: bool = False) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{label} must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize() if normalize else timestamp


def _finite_degrees(values: Sequence[object]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any((not math.isinf(value) and value <= 2.0) for value in result):
        raise ValueError("degrees-of-freedom candidates must exceed two")
    if len(set(result)) != len(result):
        raise ValueError("degrees-of-freedom candidates must be distinct")
    return result


def variants_from_config(config: Mapping[str, Any]) -> tuple[SensitivityVariant, ...]:
    """Validate and return runnable variants from the inference contract."""

    declarations = _validated_variant_declarations(config)
    evidence_sets = _validated_evidence_sets(config)
    records: list[SensitivityVariant] = []
    for raw in declarations:
        variant_id = str(raw["variant_id"])
        if not bool(raw["enabled"]):
            continue
        emission = str(raw["emission_family"])
        var_method = str(raw["var_method"])
        retail = str(raw["retail_method"])
        if emission not in {"gaussian", "student_t_7", "selected_student_t"}:
            raise ValueError(f"unsupported emission family: {emission}")
        if var_method not in {"ols", "huber", "student_t_7"}:
            raise ValueError(f"unsupported VAR sensitivity: {var_method}")
        if retail not in {
            "baseline_nominal",
            "stronger_inflation_shrinkage",
            "real_decomposition",
        }:
            raise ValueError(f"unsupported retail sensitivity: {retail}")
        records.append(
            SensitivityVariant(
                variant_id=variant_id,
                model_role=str(raw["model_role"]),
                non_defining_evidence=bool(raw["non_defining_evidence"]),
                partial_defining_releases=bool(raw["partial_defining_releases"]),
                emission_family=emission,
                var_method=var_method,
                retail_method=retail,
                evidence_set_id=str(raw["evidence_set_id"]),
                observation_model_ids=evidence_sets[str(raw["evidence_set_id"])],
            )
        )
    if not records:
        raise ValueError("no enabled sensitivity variants")
    return tuple(records)


def _validated_evidence_sets(
    config: Mapping[str, Any],
) -> dict[str, frozenset[str] | None]:
    """Return explicit per-variant observation-model allowlists.

    Historical sensitivity configurations omit ``evidence_sets`` and retain
    their original behavior through the ``all_configured`` sentinel. New
    evidence experiments must declare concrete model IDs so appending candidate
    event rows cannot silently alter the selected baseline.
    """

    raw = config.get("evidence_sets")
    if raw is None:
        return {"all_configured": None}
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("evidence_sets must be a nonempty mapping")
    result: dict[str, frozenset[str] | None] = {}
    for raw_id, declaration in raw.items():
        set_id = str(raw_id).strip()
        if not set_id:
            raise ValueError("evidence-set id cannot be empty")
        if not isinstance(declaration, Mapping):
            raise ValueError(f"evidence set {set_id} must be a mapping")
        models = declaration.get("observation_models")
        if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
            raise ValueError(
                f"evidence set {set_id} observation_models must be a sequence"
            )
        normalized = tuple(str(value).strip() for value in models)
        if any(not value for value in normalized) or len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(f"evidence set {set_id} contains invalid model IDs")
        result[set_id] = frozenset(normalized)
    return result


def _validated_variant_declarations(
    config: Mapping[str, Any],
) -> tuple[dict[str, object], ...]:
    """Normalize and validate the complete enabled/disabled role partition."""

    selection = config.get("model_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("inference config requires model_selection")
    selected_baseline = str(selection.get("baseline", ""))
    major_benchmarks = tuple(
        str(value) for value in selection.get("major_benchmarks", ())
    )
    if not selected_baseline:
        raise ValueError("model_selection requires one baseline")
    if len(major_benchmarks) != 2 or len(set(major_benchmarks)) != 2:
        raise ValueError("model_selection requires two distinct major benchmarks")
    vocabulary = tuple(str(value) for value in selection.get("role_vocabulary", ()))
    if set(vocabulary) != set(MODEL_ROLES):
        raise ValueError("model_selection role_vocabulary is invalid")
    evidence_sets = _validated_evidence_sets(config)

    raw_variants = config.get("variants", ())
    if not isinstance(raw_variants, Sequence) or isinstance(
        raw_variants, (str, bytes)
    ):
        raise ValueError("inference config variants must be a sequence")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_variants):
        if not isinstance(raw, Mapping):
            raise ValueError("each inference variant must be a mapping")
        variant_id = str(raw.get("id", ""))
        if not variant_id:
            raise ValueError("inference variant id cannot be empty")
        if variant_id in seen:
            raise ValueError(f"duplicate sensitivity variant: {variant_id}")
        seen.add(variant_id)
        role = str(raw.get("model_role", ""))
        if role not in MODEL_ROLES:
            raise ValueError(f"unsupported model role for {variant_id}: {role}")
        evidence_set_id = str(raw.get("evidence_set", "all_configured"))
        if evidence_set_id not in evidence_sets:
            raise ValueError(
                f"variant {variant_id} uses unknown evidence set {evidence_set_id}"
            )
        normalized.append(
            {
                "display_order": position,
                "variant_id": variant_id,
                "model_role": role,
                "enabled": bool(raw.get("enabled", True)),
                "non_defining_evidence": bool(raw["non_defining_evidence"]),
                "partial_defining_releases": bool(raw["partial_defining_releases"]),
                "emission_family": str(raw["emission"]),
                "var_method": str(raw["var"]),
                "retail_method": str(raw["retail"]),
                "evidence_set_id": evidence_set_id,
            }
        )

    baseline_ids = {
        str(row["variant_id"])
        for row in normalized
        if row["model_role"] == "baseline"
    }
    if baseline_ids != {selected_baseline}:
        raise ValueError("variant roles do not match model_selection baseline")
    benchmark_ids = {
        str(row["variant_id"])
        for row in normalized
        if row["model_role"] == "major_benchmark"
    }
    if benchmark_ids != set(major_benchmarks):
        raise ValueError("variant roles do not match model_selection benchmarks")
    for row in normalized:
        variant_id = str(row["variant_id"])
        role = str(row["model_role"])
        if role in {"baseline", "major_benchmark"} and not bool(row["enabled"]):
            raise ValueError(f"{role} variant must be enabled: {variant_id}")
        if role == "major_benchmark" and row["var_method"] != "ols":
            raise ValueError(f"major benchmark must use OLS VAR(1): {variant_id}")
        if (
            variant_id not in {selected_baseline, *major_benchmarks}
            and role != "sensitivity"
        ):
            raise ValueError(f"unselected variant must be a sensitivity: {variant_id}")
    return tuple(normalized)


def variant_registry_from_config(config: Mapping[str, Any]) -> pd.DataFrame:
    """Return the complete machine-readable model-role registry."""

    declarations = _validated_variant_declarations(config)
    return pd.DataFrame.from_records(declarations)


def stronger_retail_shrinkage_spec(
    baseline: LinearGaussianEmissionSpec,
    penalties: Mapping[str, Sequence[float]],
) -> LinearGaussianEmissionSpec:
    """Return the nominal consumer specification with declared penalties."""

    matrix: list[tuple[float, float]] = []
    for response in baseline.response_names:
        if response not in penalties:
            raise ValueError(f"retail shrinkage omits {response}")
        row = tuple(float(value) for value in penalties[response])
        if len(row) != 2:
            raise ValueError("each retail loading penalty must have two entries")
        matrix.append(row)
    return replace(
        baseline,
        block_id="consumer_demand_stronger_inflation_shrinkage",
        state_loading_penalties=tuple(matrix),
    )


def real_retail_spec(base_config: Mapping[str, Any]) -> LinearGaussianEmissionSpec:
    """Create the named real-growth/implicit-price consumer specification."""

    emission = base_config["emissions"]
    return LinearGaussianEmissionSpec(
        block_id="consumer_demand_real_decomposition",
        response_names=(
            "real_retail_and_food_services_log_change",
            "implicit_retail_price_log_change",
        ),
        state_loading_penalties=((1.0, 10.0), (10.0, 1.0)),
        exact_zero_mask=((False, False), (False, False)),
        lambda_grid=tuple(float(value) for value in emission["lambda_grid"]),
        minimum_training_samples=60,
        validation_minimum_training_samples=36,
        minimum_validation_observations=24,
        covariance_eigenvalue_floor=float(emission["covariance"]["eigenvalue_floor"]),
    )


def _full_response_prediction(fit: Any, validation: pd.DataFrame) -> np.ndarray:
    return (
        fit.intercept.to_numpy(dtype=float)[None, :]
        + validation.loc[:, fit.spec.state_names].to_numpy(dtype=float)
        @ fit.state_loadings.to_numpy(dtype=float).T
        + validation.loc[:, fit.spec.control_names].to_numpy(dtype=float)
        @ fit.control_loadings.to_numpy(dtype=float).T
    )


def annual_rolling_origin_tail_scores(
    profile: EmissionProfile,
    *,
    degrees_of_freedom_grid: Sequence[float],
    maximum_iterations: int,
    tolerance: float,
    minimum_weight: float,
) -> pd.DataFrame:
    """Score every ridge/tail pair in strictly causal calendar-year folds."""

    degrees = _finite_degrees(degrees_of_freedom_grid)
    selection = select_causal_emission_rows(
        profile.training,
        spec=profile.spec,
        availability_column="training_available_at",
    )
    table = selection.eligible.sort_values("training_available_at", kind="mergesort")
    years = sorted(table["training_available_at"].dt.year.unique())
    records: list[dict[str, object]] = []
    for validation_year in years:
        origin = pd.Timestamp(year=int(validation_year), month=1, day=1)
        next_origin = origin + pd.offsets.YearBegin()
        training = table.loc[table["training_available_at"] < origin]
        validation = table.loc[
            table["training_available_at"].ge(origin)
            & table["training_available_at"].lt(next_origin)
        ]
        if (
            len(training) < profile.spec.validation_minimum_training_samples
            or validation.empty
        ):
            continue
        for degrees_of_freedom in degrees:
            for base_lambda in profile.spec.lambda_grid:
                try:
                    fit = fit_linear_student_t_emission_fixed(
                        training,
                        spec=profile.spec,
                        availability_column="training_available_at",
                        base_lambda=base_lambda,
                        degrees_of_freedom=degrees_of_freedom,
                        maximum_iterations=maximum_iterations,
                        tolerance=tolerance,
                        minimum_weight=minimum_weight,
                    )
                    residuals = (
                        validation.loc[:, profile.spec.response_names].to_numpy(dtype=float)
                        - _full_response_prediction(fit, validation)
                    )
                    losses = np.asarray(
                        multivariate_student_t_nll(
                            residuals,
                            fit.residual_scale.to_numpy(dtype=float),
                            degrees_of_freedom,
                        ),
                        dtype=float,
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    records.append(
                        {
                            "profile_id": profile.profile_id,
                            "observation_model_id": profile.observation_model_id,
                            "validation_year": int(validation_year),
                            "origin": origin,
                            "base_lambda": float(base_lambda),
                            "degrees_of_freedom": float(degrees_of_freedom),
                            "training_observations": int(len(training)),
                            "validation_observations": 0,
                            "total_predictive_nll": math.nan,
                            "mean_predictive_nll": math.nan,
                            "status": f"fit_error:{error}",
                        }
                    )
                    continue
                records.append(
                    {
                        "profile_id": profile.profile_id,
                        "observation_model_id": profile.observation_model_id,
                        "validation_year": int(validation_year),
                        "origin": origin,
                        "base_lambda": float(base_lambda),
                        "degrees_of_freedom": float(degrees_of_freedom),
                        "training_observations": int(len(training)),
                        "last_training_available_at": training[
                            "training_available_at"
                        ].max(),
                        "first_validation_available_at": validation[
                            "training_available_at"
                        ].min(),
                        "validation_observations": int(len(validation)),
                        "total_predictive_nll": float(losses.sum()),
                        "mean_predictive_nll": float(losses.mean()),
                        "status": "scored",
                    }
                )
    return pd.DataFrame.from_records(records)


def causal_annual_hyperparameter_schedule(
    fold_scores: pd.DataFrame,
    *,
    profile_id: str,
    first_year: int,
    final_year: int,
    allowed_degrees_of_freedom: Sequence[float],
    minimum_predictive_events: int,
    fallback_lambda: float,
    fallback_degrees_of_freedom: float,
) -> pd.DataFrame:
    """Select each year's parameters using completed earlier validation folds."""

    allowed = set(_finite_degrees(allowed_degrees_of_freedom))
    records: list[dict[str, object]] = []
    for year in range(int(first_year), int(final_year) + 1):
        eligible = fold_scores.loc[
            fold_scores["profile_id"].eq(profile_id)
            & fold_scores["status"].eq("scored")
            & fold_scores["validation_year"].lt(year)
            & fold_scores["degrees_of_freedom"].isin(allowed)
        ].copy()
        if eligible.empty:
            selected_lambda = float(fallback_lambda)
            selected_degrees = float(fallback_degrees_of_freedom)
            observations = 0
            mean_nll = math.nan
            status = "fallback_no_completed_fold"
        else:
            grouped = (
                eligible.groupby(["base_lambda", "degrees_of_freedom"], as_index=False)
                .agg(
                    total_predictive_nll=("total_predictive_nll", "sum"),
                    validation_observations=("validation_observations", "sum"),
                    validation_folds=("validation_year", "nunique"),
                )
            )
            grouped["mean_predictive_nll"] = (
                grouped["total_predictive_nll"]
                / grouped["validation_observations"].replace(0, np.nan)
            )
            usable = grouped.loc[
                grouped["validation_observations"] >= int(minimum_predictive_events)
            ].copy()
            if usable.empty:
                selected_lambda = float(fallback_lambda)
                selected_degrees = float(fallback_degrees_of_freedom)
                observations = int(grouped["validation_observations"].max())
                mean_nll = math.nan
                status = "fallback_insufficient_predictive_events"
            else:
                minimum = float(usable["mean_predictive_nll"].min())
                tied = usable.loc[
                    np.isclose(
                        usable["mean_predictive_nll"], minimum, rtol=1.0e-12, atol=1.0e-12
                    )
                ]
                chosen = tied.sort_values(
                    ["base_lambda", "degrees_of_freedom"], ascending=[False, False]
                ).iloc[0]
                selected_lambda = float(chosen["base_lambda"])
                selected_degrees = float(chosen["degrees_of_freedom"])
                observations = int(chosen["validation_observations"])
                mean_nll = float(chosen["mean_predictive_nll"])
                status = "selected_from_completed_annual_origins"
        records.append(
            {
                "profile_id": profile_id,
                "selection_year": int(year),
                "selected_base_lambda": selected_lambda,
                "selected_degrees_of_freedom": selected_degrees,
                "prior_validation_observations": observations,
                "prior_mean_predictive_nll": mean_nll,
                "selection_status": status,
            }
        )
    return pd.DataFrame.from_records(records)


def _parameters_for_year(
    schedules: pd.DataFrame,
    profile_id: str,
    family: str,
    year: int,
) -> tuple[float, float, str]:
    rows = schedules.loc[
        schedules["profile_id"].eq(profile_id)
        & schedules["emission_family"].eq(family)
        & schedules["selection_year"].eq(int(year))
    ]
    if len(rows) != 1:
        raise ValueError(f"missing unique hyperparameter schedule for {profile_id}/{family}/{year}")
    row = rows.iloc[0]
    return (
        float(row["selected_base_lambda"]),
        float(row["selected_degrees_of_freedom"]),
        str(row["selection_status"]),
    )


def _selected_var_pairs(
    pair_audit: pd.DataFrame,
    *,
    source_month: pd.Timestamp,
    cutoff: pd.Timestamp,
    minimum_pairs: int,
) -> pd.DataFrame:
    selected = select_causal_var_pairs(
        pair_audit,
        source_reference_month=source_month,
        forecast_available_at=cutoff - pd.Timedelta(nanoseconds=1),
    )
    if len(selected) < minimum_pairs:
        raise ValueError(
            f"VAR(1) needs {minimum_pairs} pairs before {cutoff.date()}, got {len(selected)}"
        )
    return selected


def _fit_var(
    selected: pd.DataFrame,
    method: str,
    robust_config: Mapping[str, Any],
) -> RobustVar1Fit:
    method_config = robust_config["methods"][method]
    if method == "ols":
        estimator = "ols"
    elif method == "huber":
        estimator = "huber"
    elif method == "student_t_7":
        estimator = "student_t"
    else:
        raise ValueError(f"unknown robust VAR method: {method}")
    fit = fit_var1_sensitivity(
        selected,
        estimator=estimator,
        huber_threshold=float(method_config.get("mahalanobis_cutoff", 2.5)),
        degrees_of_freedom=float(method_config.get("degrees_of_freedom", 7.0)),
        maximum_iterations=int(method_config.get("maximum_iterations", 100)),
        tolerance=float(method_config.get("relative_tolerance", 1.0e-8)),
        covariance_eigenvalue_floor=float(
            robust_config.get("covariance_eigenvalue_floor", 1.0e-8)
        ),
    )
    if method != "ols" and not fit.converged:
        raise RuntimeError(
            f"{method} VAR failed to converge; refusing to propagate unaudited weights"
        )
    return fit


def _dynamics(fit: RobustVar1Fit) -> VarDynamics:
    return VarDynamics(
        intercept=np.asarray(fit.intercept, dtype=float),
        transition=np.asarray(fit.transition, dtype=float),
        innovation_covariance=np.asarray(fit.innovation_covariance, dtype=float),
    )


def _initialization_date(
    scores: pd.DataFrame,
    pair_audit: pd.DataFrame,
    *,
    replay_end: pd.Timestamp,
    minimum_pairs: int,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    lookup = scores.set_index("reference_month")
    candidate = (scores["reference_month"].min().to_period("M") + 3).to_timestamp()
    while candidate <= replay_end:
        oldest = (candidate.to_period("M") - 3).to_timestamp()
        if oldest in lookup.index and pd.Timestamp(lookup.loc[oldest, "score_available_at"]) < candidate:
            try:
                selected = _selected_var_pairs(
                    pair_audit,
                    source_month=(candidate.to_period("M") - 1).to_timestamp(),
                    cutoff=candidate,
                    minimum_pairs=minimum_pairs,
                )
            except ValueError:
                pass
            else:
                return candidate, selected
        candidate = (candidate.to_period("M") + 1).to_timestamp()
    raise ValueError("no feasible sensitivity replay initialization")


def _transition_fit_record(
    fit: RobustVar1Fit,
    *,
    method: str,
    fit_date: pd.Timestamp,
    selected: pd.DataFrame,
) -> dict[str, object]:
    return {
        "transition_fit_id": f"{method}:{fit_date:%Y-%m-%d}",
        "fit_date": fit_date,
        "method": method,
        "training_pairs": int(len(selected)),
        "last_pair_available_at": selected["pair_available_at"].max(),
        "intercept_growth": float(fit.intercept[0]),
        "intercept_inflation": float(fit.intercept[1]),
        "a_growth_growth": float(fit.transition[0, 0]),
        "a_growth_inflation": float(fit.transition[0, 1]),
        "a_inflation_growth": float(fit.transition[1, 0]),
        "a_inflation_inflation": float(fit.transition[1, 1]),
        "q_growth": float(fit.innovation_covariance[0, 0]),
        "q_growth_inflation": float(fit.innovation_covariance[0, 1]),
        "q_inflation": float(fit.innovation_covariance[1, 1]),
        "spectral_radius": float(fit.spectral_radius),
        "estimator": fit.estimator,
        "iterations": int(fit.iterations),
        "converged": bool(fit.converged),
        "effective_sample_size": float(fit.effective_sample_size),
        "minimum_weight": float(fit.minimum_weight),
        "downweighted_pairs": int(fit.downweighted_pairs),
    }


def _hard_quadrant(score: Sequence[float]) -> str:
    growth, inflation = (float(value) for value in score)
    if growth >= 0.0 and inflation >= 0.0:
        return REGIME_ORDER[0]
    if growth < 0.0 and inflation >= 0.0:
        return REGIME_ORDER[1]
    if growth >= 0.0 and inflation < 0.0:
        return REGIME_ORDER[2]
    return REGIME_ORDER[3]


def _profile_for_event(
    variant: SensitivityVariant,
    observation_model_id: str,
) -> str | None:
    allowlist_id = (
        "consumer_demand"
        if observation_model_id == "consumer_demand_real_decomposition"
        else observation_model_id
    )
    if (
        variant.observation_model_ids is not None
        and allowlist_id not in variant.observation_model_ids
    ):
        return None
    if observation_model_id == "consumer_demand_real_decomposition":
        return "retail_real" if variant.retail_method == "real_decomposition" else None
    if observation_model_id == "consumer_demand":
        if variant.retail_method == "real_decomposition":
            return None
        if variant.retail_method == "stronger_inflation_shrinkage":
            return "retail_stronger"
    return f"base:{observation_model_id}"


def run_inference_sensitivities(
    score_history: pd.DataFrame,
    mapping_history: pd.DataFrame,
    prepared: PreparedObservationData,
    first_release_components: pd.DataFrame,
    base_config: Mapping[str, Any],
    sensitivity_config: Mapping[str, Any],
    *,
    real_retail_prepared: PreparedObservationData | None = None,
    replay_end: object | None = None,
) -> InferenceSensitivityResult:
    """Replay all enabled partial-release and robustness variants causally."""

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
    return InferenceSensitivityResult(
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
        initial_date=start,
        replay_end=end,
    )


__all__ = [
    "EmissionProfile",
    "InferenceSensitivityResult",
    "MODEL_ROLES",
    "SensitivityVariant",
    "annual_rolling_origin_tail_scores",
    "causal_annual_hyperparameter_schedule",
    "real_retail_spec",
    "run_inference_sensitivities",
    "stronger_retail_shrinkage_spec",
    "variant_registry_from_config",
    "variants_from_config",
]
