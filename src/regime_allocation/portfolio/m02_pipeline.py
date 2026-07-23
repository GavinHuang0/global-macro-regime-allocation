"""Build the promoted Model 02 weekly allocation and causal backtest.

The portfolio uses one-week open-to-open returns, frequency-equivalent weekly
history and shrinkage settings, a 52x covariance annualization factor, and a
one-week expected-return objective.  The promoted reduced-core filter is
sampled at each calendar-week Monday before same-day releases, and targets
trade at the first common adjusted open in that week.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from regime_allocation.backtest import (
    build_weekly_daily_nav,
    build_weekly_open_to_open_holding_returns,
    compute_performance_metrics,
    paired_circular_block_bootstrap,
    simulate_weekly_targets,
)
from regime_allocation.cli.build_m02_current_baseline import (
    BASELINE_ID as PROMOTED_BASELINE_ID,
    MODEL_ID,
    SELECTED_CANDIDATE_MODELS,
    _load_current_config,
    _project_path,
    _run_config,
)
from regime_allocation.cli.build_m02_evidence_experiment_inference import (
    _merge_observation_models,
)
from regime_allocation.cli.build_m02_inference_sensitivities import _load_yaml
from regime_allocation.models.m02_soft_composite.weekly_decision_replay import (
    run_weekly_decision_replay,
)
from regime_allocation.models.m02_soft_composite.walkforward import (
    prepare_observation_data,
)
from regime_allocation.portfolio.baselines import (
    equal_weight_target,
    legacy_sharpe_map,
    static_60_40_target,
)
from regime_allocation.portfolio.estimation import (
    CANONICAL_REGIME_IDS,
    PROBABILITY_COLUMNS,
    PosteriorMixtureMoments,
    RegimeReturnEstimate,
    posterior_mixture_moments,
)
from regime_allocation.portfolio.m02_estimation import (
    fit_causal_weekly_regime_return_model,
)
from regime_allocation.portfolio.m02_weekly import (
    build_weekly_execution_schedule,
    derive_m02_regime_history,
    select_m02_weekly_signals,
)
from regime_allocation.portfolio.optimization import (
    GroupCap,
    OptimizationResult,
    optimize_long_only,
)
from regime_allocation.portfolio.pipeline import (
    AllocationSpecification,
    BASELINE_METHOD,
    BENCHMARK_METHODS,
    POOLED_METHOD,
    _daily_total_returns,
    _group_caps,
    _known_pretrade_weights,
    _pooled_mean_moments,
    _posterior_from_signal,
    _sensitivity_table,
    _write_csv,
    _write_json,
)


STAGE_ID = "weekly_posterior_regime_allocation_backtest"
PERIODS_PER_YEAR = 52
MODEL01_PERIODS_PER_YEAR = 12
MINIMUM_LABELED_WEEKS = 260
BASELINE_PSEUDO_WEEKS = 104.0
ANCHORED_METHOD = "pooled_anchor_posterior_25pct"
ANCHORED_CORE_FRACTION = 0.75
POSTERIOR_SLEEVE_FRACTION = 0.25
M02_PUBLIC_METHODS = (BASELINE_METHOD, ANCHORED_METHOD, *BENCHMARK_METHODS)
POSTERIOR_COMPARATOR_METHODS = (*BENCHMARK_METHODS, ANCHORED_METHOD)
WEEKLY_ASSOCIATION = "calendar_month_containing_reference_week_monday"
WEEKLY_OBJECTIVE = "maximize_one_week_expected_return_net_of_estimated_trading_cost"
LEADING_PARTIAL_WEEK_POLICY = (
    "exclude_monday_anchor_before_common_price_history"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _validate_active_sleeve_policy(config: Mapping[str, Any]) -> None:
    sleeve = config.get("additional_strategies", {}).get(
        "pooled_anchor_posterior_active_sleeve", {}
    )
    expected = {
        "method_id": ANCHORED_METHOD,
        "role": "exploratory_not_promoted",
        "replaces_promoted_baseline": False,
        "construction": "convex_combination_of_weekly_targets",
        "pooled_core_method": POOLED_METHOD,
        "pooled_core_fraction": ANCHORED_CORE_FRACTION,
        "posterior_sleeve_method": BASELINE_METHOD,
        "posterior_sleeve_fraction": POSTERIOR_SLEEVE_FRACTION,
        "realized_transaction_costs": (
            "consolidated_blended_target_at_same_five_basis_points"
        ),
        "paired_comparators": [POOLED_METHOD, BASELINE_METHOD],
    }
    if sleeve != expected:
        raise ValueError("Model 02 anchored active-sleeve strategy differs")


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("Model 02 allocation configuration must be a mapping")
    if (
        int(config.get("schema_version", 0)) != 1
        or config.get("model_id") != MODEL_ID
        or config.get("stage_id") != STAGE_ID
    ):
        raise ValueError("unexpected Model 02 allocation configuration identity")
    assets = config.get("universe", {}).get("strategy_assets")
    if assets != ["SPY", "IEF", "TIP", "HYG", "BIL", "GLD", "LQD"]:
        raise ValueError("configuration strategy universe is not the frozen seven ETFs")
    if config.get("signal", {}).get("variant_id") != PROMOTED_BASELINE_ID:
        raise ValueError("allocation signal is not the promoted Model 02 baseline")
    if config.get("signal", {}).get("regime_order") != list(CANONICAL_REGIME_IDS):
        raise ValueError("configuration regime order is not canonical")
    if config.get("timing", {}).get("target_period") != "calendar_week_monday":
        raise ValueError("Model 02 allocation must use calendar-week Monday anchors")
    if int(config.get("evaluation", {}).get("periods_per_year", 0)) != PERIODS_PER_YEAR:
        raise ValueError("weekly evaluation must use 52 periods per year")
    estimation = config.get("estimation", {})
    if estimation.get("return_frequency") != "weekly_open_to_open":
        raise ValueError("Model 02 return estimation must be weekly open-to-open")
    if estimation.get("leading_partial_week_policy") != LEADING_PARTIAL_WEEK_POLICY:
        raise ValueError("weekly estimation must exclude a truncated leading week")
    if int(estimation.get("minimum_labeled_weeks", 0)) != MINIMUM_LABELED_WEEKS:
        raise ValueError("weekly estimation must require 260 labeled weeks")
    if (
        float(estimation.get("regime_means", {}).get("pseudo_weeks", -1))
        != BASELINE_PSEUDO_WEEKS
    ):
        raise ValueError("weekly regime-mean shrinkage must use 104 pseudo-weeks")
    if (
        estimation.get("regime_labels", {}).get("weekly_association")
        != WEEKLY_ASSOCIATION
    ):
        raise ValueError("weekly returns must use their Monday-month regime label")
    if config.get("optimization", {}).get("objective") != WEEKLY_OBJECTIVE:
        raise ValueError("Model 02 optimization objective must use weekly returns")
    if (
        float(config["optimization"].get("covariance_annualization_factor", -1))
        != PERIODS_PER_YEAR
    ):
        raise ValueError("weekly return covariance must use 52x annualization")
    _validate_active_sleeve_policy(config)
    return config, raw


def _without(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    result = deepcopy(dict(mapping))
    for key in keys:
        result.pop(key, None)
    return result


def _validate_model01_policy_parity(
    config: Mapping[str, Any], template: Mapping[str, Any]
) -> None:
    """Prove that changes from Model 01 are frequency-equivalent translations."""

    if config["universe"] != template["universe"]:
        raise ValueError("Model 02 universe differs from the Model 01 template")
    _validate_active_sleeve_policy(config)
    estimation = config["estimation"]
    template_estimation = template["estimation"]
    if estimation["return_frequency"] != "weekly_open_to_open":
        raise ValueError("Model 02 return frequency is not weekly")
    if estimation["leading_partial_week_policy"] != LEADING_PARTIAL_WEEK_POLICY:
        raise ValueError("Model 02 leading partial-week policy differs")
    expected_weeks = int(
        template_estimation["minimum_labeled_months"]
        * PERIODS_PER_YEAR
        / MODEL01_PERIODS_PER_YEAR
    )
    if int(estimation["minimum_labeled_weeks"]) != expected_weeks:
        raise ValueError("Model 02 minimum history is not frequency-equivalent")
    if (
        float(estimation["model01_history_equivalent_months"])
        != float(template_estimation["minimum_labeled_months"])
    ):
        raise ValueError("Model 02 minimum-history lineage differs from Model 01")
    template_kappa = float(template_estimation["regime_means"]["pseudo_months"])
    expected_kappa = template_kappa * PERIODS_PER_YEAR / MODEL01_PERIODS_PER_YEAR
    if float(estimation["regime_means"]["pseudo_weeks"]) != expected_kappa:
        raise ValueError("Model 02 mean shrinkage is not frequency-equivalent")
    if (
        float(estimation["regime_means"]["model01_equivalent_pseudo_months"])
        != template_kappa
    ):
        raise ValueError("Model 02 mean-shrinkage lineage differs from Model 01")
    expected_regime_means = {
        "estimator": str(template_estimation["regime_means"]["estimator"]).replace(
            "pseudo_month", "pseudo_week"
        ),
        "pseudo_weeks": expected_kappa,
        "model01_equivalent_pseudo_months": template_kappa,
        "formula": template_estimation["regime_means"]["formula"],
        "empty_regime_policy": template_estimation["regime_means"][
            "empty_regime_policy"
        ],
    }
    if estimation["regime_means"] != expected_regime_means:
        raise ValueError("Model 02 regime-mean estimator is not the weekly analogue")
    for key in ("covariance", "posterior_mixture"):
        if config["estimation"][key] != template["estimation"][key]:
            raise ValueError(f"Model 02 estimation policy differs at {key}")
    expected_regime_labels = {
        "source": "model02_completed_composite_scores",
        "rule": "sign_of_exact_growth_and_inflation_scores",
        "availability": "score_available_at",
        "weekly_association": WEEKLY_ASSOCIATION,
    }
    if estimation["regime_labels"] != expected_regime_labels:
        raise ValueError("Model 02 weekly regime-label association differs")
    optimization = config["optimization"]
    template_optimization = template["optimization"]
    if optimization["objective"] != str(template_optimization["objective"]).replace(
        "one_month", "one_week"
    ):
        raise ValueError("Model 02 optimizer objective is not the weekly analogue")
    if float(optimization["covariance_annualization_factor"]) != PERIODS_PER_YEAR:
        raise ValueError("Model 02 covariance annualization is not weekly")
    if _without(
        optimization, "objective", "covariance_annualization_factor"
    ) != _without(
        template_optimization, "objective", "covariance_annualization_factor"
    ):
        raise ValueError("Model 02 optimization policy differs from Model 01")
    for benchmark in (
        "pooled_mean_optimizer",
        "legacy_sharpe_map",
    ):
        if config["benchmarks"][benchmark] != template["benchmarks"][benchmark]:
            raise ValueError(f"Model 02 benchmark differs at {benchmark}")
    for benchmark in ("equal_weight", "static_60_40"):
        if _without(config["benchmarks"][benchmark], "rebalance") != _without(
            template["benchmarks"][benchmark], "rebalance"
        ):
            raise ValueError(f"Model 02 benchmark differs at {benchmark}")
    if (
        config["benchmarks"]["apply_same_realized_transaction_costs"]
        != template["benchmarks"]["apply_same_realized_transaction_costs"]
    ):
        raise ValueError("benchmark transaction-cost policy differs from Model 01")
    sensitivity = config["sensitivity"]
    template_sensitivity = template["sensitivity"]
    if _without(sensitivity, "parameters") != _without(
        template_sensitivity, "parameters"
    ):
        raise ValueError("Model 02 allocation sensitivity design differs")
    weekly_parameters = sensitivity["parameters"]
    monthly_parameters = template_sensitivity["parameters"]
    expected_kappas = [
        float(value) * PERIODS_PER_YEAR / MODEL01_PERIODS_PER_YEAR
        for value in monthly_parameters["regime_mean_pseudo_months"]
    ]
    if list(map(float, weekly_parameters["regime_mean_pseudo_weeks"])) != expected_kappas:
        raise ValueError("Model 02 kappa sensitivities are not frequency-equivalent")
    for key in (
        "annualized_volatility_cap",
        "per_asset_transaction_cost",
        "concentration_cap_multiplier",
    ):
        if weekly_parameters[key] != monthly_parameters[key]:
            raise ValueError(f"Model 02 sensitivity differs at {key}")

    evaluation = config["evaluation"]
    template_evaluation = template["evaluation"]
    for key in ("zero_rate_sharpe", "bil_excess_sharpe", "daily_nav_drawdown"):
        if evaluation[key] != template_evaluation[key]:
            raise ValueError(f"Model 02 evaluation policy differs at {key}")
    translated_metrics = [
        str(metric)
        .replace("worst_month", "worst_week")
        .replace("positive_month_fraction", "positive_week_fraction")
        for metric in template_evaluation["metrics"]
    ]
    if evaluation["metrics"] != translated_metrics:
        raise ValueError("Model 02 evaluation metrics differ from Model 01")
    uncertainty = evaluation["uncertainty"]
    template_uncertainty = template_evaluation["uncertainty"]
    for key in ("baseline_method", "resamples", "confidence_level", "random_seed"):
        if uncertainty[key] != template_uncertainty[key]:
            raise ValueError(f"Model 02 uncertainty policy differs at {key}")
    if uncertainty["method"] != str(template_uncertainty["method"]).replace(
        "monthly", "weekly"
    ):
        raise ValueError("Model 02 uncertainty method is not the weekly analogue")
    if int(uncertainty["block_length_weeks"]) != 26:
        raise ValueError("Model 02 uncertainty block must be 26 weeks")


def _allocation_specifications(
    config: Mapping[str, Any],
) -> tuple[AllocationSpecification, ...]:
    """Return the weekly baseline and one-at-a-time policy sensitivities."""

    baseline_kappa = float(config["estimation"]["regime_means"]["pseudo_weeks"])
    baseline_volatility = float(config["optimization"]["annualized_volatility_cap"])
    first_asset = config["universe"]["strategy_assets"][0]
    baseline_cost = float(
        config["optimization"]["per_asset_transaction_cost"][first_asset]
    )
    specifications = [
        AllocationSpecification(
            method=BASELINE_METHOD,
            specification_type="baseline",
            parameter="baseline",
            parameter_value=1.0,
            kappa=baseline_kappa,
            volatility_cap=baseline_volatility,
            transaction_cost=baseline_cost,
            cap_multiplier=1.0,
        )
    ]
    parameters = config["sensitivity"]["parameters"]
    for raw in parameters["regime_mean_pseudo_weeks"]:
        value = float(raw)
        if value == baseline_kappa:
            continue
        specifications.append(
            AllocationSpecification(
                method=f"sensitivity_kappa_{value:g}",
                specification_type="sensitivity",
                parameter="regime_mean_pseudo_weeks",
                parameter_value=value,
                kappa=value,
                volatility_cap=baseline_volatility,
                transaction_cost=baseline_cost,
                cap_multiplier=1.0,
            )
        )
    for raw in parameters["annualized_volatility_cap"]:
        value = float(raw)
        if value == baseline_volatility:
            continue
        specifications.append(
            AllocationSpecification(
                method=f"sensitivity_volatility_cap_{value:g}",
                specification_type="sensitivity",
                parameter="annualized_volatility_cap",
                parameter_value=value,
                kappa=baseline_kappa,
                volatility_cap=value,
                transaction_cost=baseline_cost,
                cap_multiplier=1.0,
            )
        )
    for raw in parameters["per_asset_transaction_cost"]:
        value = float(raw)
        if value == baseline_cost:
            continue
        specifications.append(
            AllocationSpecification(
                method=f"sensitivity_transaction_cost_{value:g}",
                specification_type="sensitivity",
                parameter="per_asset_transaction_cost",
                parameter_value=value,
                kappa=baseline_kappa,
                volatility_cap=baseline_volatility,
                transaction_cost=value,
                cap_multiplier=1.0,
            )
        )
    for raw in parameters["concentration_cap_multiplier"]:
        value = float(raw)
        if value == 1.0:
            continue
        specifications.append(
            AllocationSpecification(
                method=f"sensitivity_concentration_caps_{value:g}",
                specification_type="sensitivity",
                parameter="concentration_cap_multiplier",
                parameter_value=value,
                kappa=baseline_kappa,
                volatility_cap=baseline_volatility,
                transaction_cost=baseline_cost,
                cap_multiplier=value,
            )
        )
    identifiers = [item.method for item in specifications]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("allocation specification identifiers are not unique")
    return tuple(specifications)


def _manifest_hash(manifest: Mapping[str, Any], relative_path: str) -> str:
    matches = [
        str(row["sha256"])
        for row in manifest.get("source_files", ())
        if isinstance(row, Mapping) and str(row.get("path")) == relative_path
    ]
    if len(set(matches)) != 1:
        raise ValueError(f"current-baseline manifest lacks one hash for {relative_path}")
    return matches[0]


def _read_verified_source(
    root: Path,
    manifest: Mapping[str, Any],
    relative_path: object,
) -> tuple[Path, bytes]:
    path = _project_path(root, relative_path)
    raw = path.read_bytes()
    relative = path.relative_to(root).as_posix()
    if _sha256(raw) != _manifest_hash(manifest, relative):
        raise ValueError(f"current-baseline source hash mismatch: {relative}")
    return path, raw


def _verified_implementation_paths(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Verify every implementation file attested by the promoted manifest."""

    declarations = manifest.get("implementation_files")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("current-baseline manifest has no implementation declarations")
    verified: dict[str, Path] = {}
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise ValueError("current-baseline implementation declaration is invalid")
        relative = str(declaration.get("path", ""))
        expected = str(declaration.get("sha256", ""))
        if not relative or not expected:
            raise ValueError("current-baseline implementation declaration is incomplete")
        path = _project_path(root, relative)
        if _sha256(path.read_bytes()) != expected:
            raise ValueError(
                f"current-baseline implementation hash mismatch: {relative}"
            )
        if relative in verified:
            raise ValueError(
                f"current-baseline implementation is declared twice: {relative}"
            )
        verified[relative] = path
    return tuple(verified.values())


def _promoted_decision_marginals(
    *,
    root: Path,
    current_config_path: Path,
    current_manifest_path: Path,
    decision_dates: Sequence[pd.Timestamp],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[Path, bytes]],
    tuple[Path, ...],
    bytes,
    bytes,
]:
    """Replay the promoted filter and capture pre-release Monday marginals."""

    current, current_bytes = _load_current_config(current_config_path)
    manifest_bytes = current_manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("model_id") != MODEL_ID
        or manifest.get("stage_id") != "current_baseline"
        or not bool(manifest.get("frozen_control_invariance_passed"))
    ):
        raise ValueError("upstream manifest is not the promoted invariant baseline")
    if manifest.get("configuration_sha256") != _sha256(current_bytes):
        raise ValueError("current-baseline configuration hash mismatch")
    if manifest.get("configuration") != current_config_path.relative_to(root).as_posix():
        raise ValueError("current-baseline manifest names a different configuration")
    if str(manifest.get("model_selection", {}).get("baseline")) != PROMOTED_BASELINE_ID:
        raise ValueError("upstream manifest names a different promoted baseline")
    implementation_paths = _verified_implementation_paths(root, manifest)

    sources = current["sources"]
    feature_manifest_path = _project_path(root, sources["feature_revision_manifest"])
    feature_manifest_bytes = feature_manifest_path.read_bytes()
    upstream = manifest.get("upstream_manifest", {})
    if (
        str(upstream.get("path")) != feature_manifest_path.relative_to(root).as_posix()
        or str(upstream.get("sha256")) != _sha256(feature_manifest_bytes)
    ):
        raise ValueError("current-baseline upstream feature manifest mismatch")
    feature_manifest = json.loads(feature_manifest_bytes)

    base_path = _project_path(root, sources["base_filter_config"])
    base, base_bytes = _load_yaml(base_path)
    frozen_path = _project_path(root, sources["frozen_sensitivity_config"])
    frozen, frozen_bytes = _load_yaml(frozen_path)
    feature_path = _project_path(root, sources["feature_revision_config"])
    feature, feature_bytes = _load_yaml(feature_path)
    if _sha256(base_bytes) != str(manifest.get("base_configuration_sha256")):
        raise ValueError("base filter changed after current-baseline promotion")
    if _sha256(frozen_bytes) != str(
        manifest.get("frozen_sensitivity_configuration_sha256")
    ):
        raise ValueError("frozen sensitivity config changed after promotion")
    if _sha256(feature_bytes) != str(
        current["source_integrity"]["feature_revision_config_sha256"]
    ):
        raise ValueError("feature-revision config changed after promotion")
    if feature_manifest.get("configuration_sha256") != _sha256(feature_bytes):
        raise ValueError("feature-revision lineage configuration mismatch")

    needed = (
        "score_features",
        "defining_components",
        "mapping_history",
        "consolidated_event_universe",
    )
    verified = {
        key: _read_verified_source(root, manifest, sources[key]) for key in needed
    }
    selected_specs = feature.get("candidate_observation_models", {})
    missing = set(SELECTED_CANDIDATE_MODELS).difference(selected_specs)
    if missing:
        raise ValueError("feature-revision config omits promoted observation models")
    base_augmented = _merge_observation_models(
        base,
        {model_id: selected_specs[model_id] for model_id in SELECTED_CANDIDATE_MODELS},
    )
    run_config = _run_config(frozen, current)
    replay_end = pd.Timestamp(manifest["coverage"]["replay_end"]).normalize()
    requested = pd.DatetimeIndex(pd.to_datetime(list(decision_dates), errors="raise"))
    if requested.tz is not None:
        requested = requested.tz_localize(None)
    requested = requested.normalize()
    if len(requested) and requested.max() > replay_end:
        raise ValueError("weekly decision date exceeds the promoted replay end")

    scores = pd.read_csv(BytesIO(verified["score_features"][1]))
    components = pd.read_csv(BytesIO(verified["defining_components"][1]))
    mapping = pd.read_csv(BytesIO(verified["mapping_history"][1]))
    events = pd.read_csv(BytesIO(verified["consolidated_event_universe"][1]))
    for column in (
        "release_date",
        "reference_date",
        "reference_month",
        "control_source_reference_date",
        "control_source_reference_month",
    ):
        if column in events:
            events[column] = pd.to_datetime(events[column], format="mixed", errors="raise")
    prepared = prepare_observation_data(events, scores, base_augmented)
    result = run_weekly_decision_replay(
        scores,
        mapping,
        prepared,
        components,
        base_augmented,
        run_config,
        replay_end=replay_end,
        decision_dates=requested,
    )
    lineage: dict[str, tuple[Path, bytes]] = {
        "current_baseline_config": (current_config_path, current_bytes),
        "current_baseline_manifest": (current_manifest_path, manifest_bytes),
        "base_filter_config": (base_path, base_bytes),
        "frozen_sensitivity_config": (frozen_path, frozen_bytes),
        "feature_revision_config": (feature_path, feature_bytes),
        "feature_revision_manifest": (feature_manifest_path, feature_manifest_bytes),
        **verified,
    }
    return (
        result.decision_marginals,
        scores,
        lineage,
        implementation_paths,
        current_bytes,
        manifest_bytes,
    )


def _record_estimate(
    records: list[dict[str, object]],
    *,
    specification: AllocationSpecification,
    signal: pd.Series,
    estimate: RegimeReturnEstimate,
) -> None:
    for regime_id in CANONICAL_REGIME_IDS:
        for asset in estimate.asset_ids:
            sample = estimate.regime_sample_means.loc[regime_id, asset]
            records.append(
                {
                    "method": specification.method,
                    "reference_week": signal["reference_week"],
                    "signal_date": signal["signal_date"],
                    "knowledge_cutoff": estimate.knowledge_cutoff,
                    "pseudo_weeks": estimate.kappa,
                    "training_count": estimate.training_count,
                    "first_training_reference_week": estimate.first_holding_month,
                    "last_training_reference_week": estimate.last_holding_month,
                    "regime_id": regime_id,
                    "regime_count": int(estimate.regime_counts.loc[regime_id]),
                    "ticker": asset,
                    "global_mean": float(estimate.global_mean.loc[asset]),
                    "regime_sample_mean": (
                        float(sample) if pd.notna(sample) else float("nan")
                    ),
                    "regime_shrunk_mean": float(
                        estimate.regime_means.loc[regime_id, asset]
                    ),
                    "within_regime_variance": float(
                        estimate.shared_within_covariance.loc[asset, asset]
                    ),
                    "ledoit_wolf_shrinkage": estimate.ledoit_wolf_shrinkage,
                }
            )


def _record_optimizer(
    *,
    specification: AllocationSpecification,
    signal: pd.Series,
    result: OptimizationResult,
    posterior: pd.Series,
    expected_returns: pd.Series,
    estimated_pretrade_as_of: pd.Timestamp | None,
) -> dict[str, object]:
    feasibility = result.audit.feasibility
    record: dict[str, object] = {
        "method": specification.method,
        "specification_type": specification.specification_type,
        "sensitivity_parameter": specification.parameter,
        "sensitivity_value": specification.parameter_value,
        "reference_week": signal["reference_week"],
        "signal_date": signal["signal_date"],
        "regime_reference_month": signal["regime_reference_month"],
        "estimated_pretrade_as_of": estimated_pretrade_as_of,
        "pseudo_weeks": specification.kappa,
        "volatility_cap": specification.volatility_cap,
        "transaction_cost": specification.transaction_cost,
        "concentration_cap_multiplier": specification.cap_multiplier,
        "current_regime_posterior_used": specification.method != POOLED_METHOD,
        "posterior_entropy": float(
            -(posterior * np.log(posterior.clip(lower=1e-300))).sum()
        ),
        "expected_portfolio_weekly_return": result.expected_monthly_return,
        "estimated_transaction_cost": result.estimated_transaction_cost,
        "net_expected_weekly_return": result.net_expected_monthly_return,
        "estimated_annualized_volatility": result.annualized_volatility,
        "estimated_traded_notional": result.traded_notional,
        "estimated_half_l1_turnover": result.half_l1_turnover,
        "optimization_outcome": result.audit.outcome,
        "primary_solver_success": result.audit.primary_solver_success,
        "primary_solver_status": result.audit.primary_solver_status,
        "primary_solver_iterations": result.audit.primary_solver_iterations,
        "primary_solver_message": result.audit.primary_solver_message,
        "fallback_used": result.audit.fallback_used,
        "fallback_reason": result.audit.fallback_reason,
        "weight_sum": feasibility.weight_sum,
        "maximum_constraint_violation": feasibility.maximum_violation,
        "binding_constraints": "|".join(feasibility.binding_constraints),
    }
    record.update(
        {
            f"expected_weekly_return_{asset}": float(expected_returns.loc[asset])
            for asset in result.assets
        }
    )
    return record


def _weight_records(
    *,
    method: str,
    signal: pd.Series,
    weights: pd.Series,
    simulation_assets: Sequence[str],
    specification_type: str,
    audit: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for ticker in simulation_assets:
        row: dict[str, object] = {
            "method": method,
            "specification_type": specification_type,
            "reference_week": signal["reference_week"],
            "signal_date": signal["signal_date"],
            "execution_date": signal["start_date"],
            "regime_reference_month": signal["regime_reference_month"],
            "ticker": ticker,
            "target_weight": float(weights.get(ticker, 0.0)),
            "is_live_only": not bool(signal["is_complete"]),
        }
        if audit:
            row.update(audit)
        records.append(row)
    return records


def _dynamic_targets(
    *,
    specifications: Sequence[AllocationSpecification],
    signals: pd.DataFrame,
    estimator_returns: pd.DataFrame,
    regime_history: pd.DataFrame,
    prices: pd.DataFrame,
    strategy_assets: Sequence[str],
    simulation_assets: Sequence[str],
    minimum_observations: int,
    covariance_annualization_factor: float,
    asset_caps: Mapping[str, float],
    group_caps: Sequence[GroupCap],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weight_records: list[dict[str, object]] = []
    estimate_records: list[dict[str, object]] = []
    optimizer_records: list[dict[str, object]] = []
    estimate_cache: dict[tuple[pd.Timestamp, float], RegimeReturnEstimate] = {}
    moment_cache: dict[tuple[pd.Timestamp, float, str], PosteriorMixtureMoments] = {}
    price_dates = pd.to_datetime(prices["date"], errors="raise")
    for specification in specifications:
        previous_target: np.ndarray | None = None
        previous_entry_date: pd.Timestamp | None = None
        for _, signal in signals.iterrows():
            signal_date = pd.Timestamp(signal["signal_date"])
            cutoff = signal_date - pd.Timedelta(days=1)
            estimate_key = (cutoff, specification.kappa)
            estimate = estimate_cache.get(estimate_key)
            if estimate is None:
                estimate = fit_causal_weekly_regime_return_model(
                    estimator_returns,
                    regime_history,
                    knowledge_cutoff=cutoff,
                    asset_ids=strategy_assets,
                    kappa=specification.kappa,
                    minimum_observations=minimum_observations,
                )
                estimate_cache[estimate_key] = estimate
            moment_key = (
                signal_date,
                specification.kappa,
                "pooled" if specification.method == POOLED_METHOD else "posterior",
            )
            moments = moment_cache.get(moment_key)
            if moments is None:
                moments = (
                    _pooled_mean_moments(estimate)
                    if specification.method == POOLED_METHOD
                    else posterior_mixture_moments(
                        _posterior_from_signal(signal), estimate
                    )
                )
                moment_cache[moment_key] = moments
            pretrade = _known_pretrade_weights(
                prices,
                assets=strategy_assets,
                previous_target=previous_target,
                previous_entry_date=previous_entry_date,
                signal_date=signal_date,
            )
            pretrade_as_of = None
            if previous_target is not None:
                pretrade_as_of = pd.Timestamp(
                    price_dates.loc[price_dates.lt(signal_date)].max()
                )
            scaled_asset_caps = {
                asset: (
                    1.0
                    if asset == "BIL"
                    else min(
                        1.0,
                        float(asset_caps[asset]) * specification.cap_multiplier,
                    )
                )
                for asset in strategy_assets
            }
            scaled_group_caps = tuple(
                GroupCap(
                    name=group.name,
                    assets=group.assets,
                    maximum=min(
                        1.0, group.maximum * specification.cap_multiplier
                    ),
                )
                for group in group_caps
            )
            result = optimize_long_only(
                assets=strategy_assets,
                # The frozen optimizer API is period-generic despite its legacy
                # monthly argument name.  Model 02 supplies weekly moments here.
                expected_monthly_returns=moments.expected_mean.to_numpy(dtype=float),
                annualized_covariance=(
                    moments.covariance.to_numpy(dtype=float)
                    * covariance_annualization_factor
                ),
                pretrade_weights=pretrade,
                asset_caps=scaled_asset_caps,
                group_caps=scaled_group_caps,
                transaction_costs=specification.transaction_cost,
                volatility_cap=specification.volatility_cap,
                cash_asset="BIL",
            )
            posterior = _posterior_from_signal(signal)
            optimizer_records.append(
                _record_optimizer(
                    specification=specification,
                    signal=signal,
                    result=result,
                    posterior=posterior,
                    expected_returns=moments.expected_mean,
                    estimated_pretrade_as_of=pretrade_as_of,
                )
            )
            _record_estimate(
                estimate_records,
                specification=specification,
                signal=signal,
                estimate=estimate,
            )
            target = pd.Series(result.weight_by_asset, dtype=float)
            weight_records.extend(
                _weight_records(
                    method=specification.method,
                    signal=signal,
                    weights=target,
                    simulation_assets=simulation_assets,
                    specification_type=specification.specification_type,
                    audit={
                        "posterior_map_regime": str(posterior.idxmax()),
                        "optimization_outcome": result.audit.outcome,
                    },
                )
            )
            previous_target = result.as_array()
            previous_entry_date = pd.Timestamp(signal["start_date"])
    return (
        pd.DataFrame.from_records(weight_records),
        pd.DataFrame.from_records(estimate_records),
        pd.DataFrame.from_records(optimizer_records),
    )


def _anchored_active_sleeve_targets(
    dynamic_weights: pd.DataFrame,
    *,
    output_method: str = ANCHORED_METHOD,
    core_method: str = POOLED_METHOD,
    active_method: str = BASELINE_METHOD,
    active_fraction: float = POSTERIOR_SLEEVE_FRACTION,
) -> pd.DataFrame:
    """Blend a pooled core with a limited posterior active sleeve.

    The two source methods remain independently formed virtual sleeves.  Their
    weekly targets are combined before consolidated portfolio accounting, so
    realized trades can net across the pooled core and posterior sleeve.
    """

    if not isinstance(output_method, str) or not output_method:
        raise ValueError("output_method must be a non-empty string")
    if core_method == active_method or output_method in {core_method, active_method}:
        raise ValueError("active-sleeve method identifiers must be distinct")
    if not np.isfinite(active_fraction) or not 0.0 < active_fraction < 1.0:
        raise ValueError("active_fraction must lie strictly between zero and one")
    core_fraction = 1.0 - float(active_fraction)
    required = {
        "method",
        "reference_week",
        "signal_date",
        "execution_date",
        "regime_reference_month",
        "ticker",
        "target_weight",
        "is_live_only",
        "posterior_map_regime",
    }
    missing = required.difference(dynamic_weights.columns)
    if missing:
        raise ValueError(
            "dynamic weights are missing active-sleeve columns: "
            + ", ".join(sorted(missing))
        )
    if output_method in set(dynamic_weights["method"].astype(str)):
        raise ValueError("output active-sleeve method already exists")
    legs = dynamic_weights.loc[
        dynamic_weights["method"].astype(str).isin((core_method, active_method))
    ].copy()
    if set(legs["method"].astype(str)) != {core_method, active_method}:
        raise ValueError("active-sleeve source method is missing")
    keys = ["reference_week", "ticker"]
    if legs.duplicated(["method", *keys]).any():
        raise ValueError("active-sleeve source weights contain duplicate keys")
    metadata = [
        "signal_date",
        "execution_date",
        "regime_reference_month",
        "is_live_only",
    ]
    core = legs.loc[legs["method"].eq(core_method), [*keys, *metadata, "target_weight"]]
    core = core.rename(columns={"target_weight": "pooled_core_target_weight"})
    active = legs.loc[
        legs["method"].eq(active_method),
        [*keys, *metadata, "target_weight", "posterior_map_regime"],
    ].rename(
        columns={
            **{column: f"active_{column}" for column in metadata},
            "target_weight": "posterior_sleeve_target_weight",
        }
    )
    merged = core.merge(active, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("active-sleeve source methods have different week/ticker keys")
    for column in metadata:
        left = merged[column]
        right = merged[f"active_{column}"]
        if not (left.eq(right) | (left.isna() & right.isna())).all():
            raise ValueError(f"active-sleeve source metadata differ at {column}")
    for column in ("pooled_core_target_weight", "posterior_sleeve_target_weight"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        if not np.isfinite(merged[column].to_numpy(dtype=float)).all():
            raise ValueError("active-sleeve source weights must be finite")
        if merged[column].lt(-1.0e-12).any():
            raise ValueError("active-sleeve source weights must be long-only")
    for column in ("pooled_core_target_weight", "posterior_sleeve_target_weight"):
        totals = merged.groupby("reference_week", sort=False)[column].sum()
        if not np.allclose(totals.to_numpy(dtype=float), 1.0, atol=1.0e-10):
            raise ValueError("each active-sleeve source target must sum to one")
    merged["target_weight"] = (
        core_fraction * merged["pooled_core_target_weight"]
        + float(active_fraction) * merged["posterior_sleeve_target_weight"]
    )
    merged["posterior_active_deviation"] = (
        merged["target_weight"] - merged["pooled_core_target_weight"]
    )
    target_totals = merged.groupby("reference_week", sort=False)["target_weight"].sum()
    if not np.allclose(target_totals.to_numpy(dtype=float), 1.0, atol=1.0e-10):
        raise RuntimeError("anchored active-sleeve targets do not sum to one")
    output = pd.DataFrame(
        {
            "method": output_method,
            "specification_type": "anchored_strategy",
            "reference_week": merged["reference_week"],
            "signal_date": merged["signal_date"],
            "execution_date": merged["execution_date"],
            "regime_reference_month": merged["regime_reference_month"],
            "ticker": merged["ticker"],
            "target_weight": merged["target_weight"],
            "is_live_only": merged["is_live_only"],
            "posterior_map_regime": merged["posterior_map_regime"],
            "optimization_outcome": "convex_target_blend",
            "anchor_core_method": core_method,
            "posterior_sleeve_method": active_method,
            "pooled_core_fraction": core_fraction,
            "posterior_sleeve_fraction": float(active_fraction),
            "pooled_core_target_weight": merged["pooled_core_target_weight"],
            "posterior_sleeve_target_weight": merged[
                "posterior_sleeve_target_weight"
            ],
            "posterior_active_deviation": merged["posterior_active_deviation"],
        }
    )
    return output.sort_values(["reference_week", "ticker"], kind="stable").reset_index(
        drop=True
    )


def _benchmark_targets(
    *,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    regime_history: pd.DataFrame,
    strategy_assets: Sequence[str],
    simulation_assets: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_returns = _daily_total_returns(prices, strategy_assets)
    equal = equal_weight_target(strategy_assets)
    static = static_60_40_target(
        simulation_assets,
        risk_asset="SPY",
        defensive_asset="AGG",
        risk_weight=0.60,
    )
    records: list[dict[str, object]] = []
    legacy_records: list[dict[str, object]] = []
    for _, signal in signals.iterrows():
        posterior = _posterior_from_signal(signal)
        legacy = legacy_sharpe_map(
            daily_returns,
            regime_history,
            posterior,
            signal_date=signal["signal_date"],
            asset_ids=strategy_assets,
            minimum_observations=60,
            annualization_factor=252,
            c_shift=0.1,
        )
        for method, target, audit in (
            (
                "legacy_sharpe_map",
                legacy.weights,
                {
                    "posterior_map_regime": legacy.map_regime_id,
                    "optimization_outcome": "legacy_formula",
                },
            ),
            (
                "equal_weight",
                equal,
                {"optimization_outcome": "static_formula"},
            ),
            (
                "static_60_spy_40_agg",
                static,
                {"optimization_outcome": "static_formula"},
            ),
        ):
            records.extend(
                _weight_records(
                    method=method,
                    signal=signal,
                    weights=target,
                    simulation_assets=simulation_assets,
                    specification_type="benchmark",
                    audit=audit,
                )
            )
        for asset in strategy_assets:
            legacy_records.append(
                {
                    "method": "legacy_sharpe_map",
                    "reference_week": signal["reference_week"],
                    "signal_date": signal["signal_date"],
                    "map_regime_id": legacy.map_regime_id,
                    "observation_count": legacy.observation_count,
                    "used_equal_weight_fallback": legacy.used_equal_weight_fallback,
                    "fallback_reason": legacy.fallback_reason,
                    "ticker": asset,
                    "annualized_sharpe_score": float(
                        legacy.annualized_sharpe_scores.loc[asset]
                    ),
                    "target_weight": float(legacy.weights.loc[asset]),
                }
            )
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(legacy_records)


def _metric_view(weekly: pd.DataFrame) -> pd.DataFrame:
    return weekly.rename(columns={"reference_week": "reference_month"})


def _weekly_metric_labels(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "start_reference_month": "start_reference_week",
            "end_reference_month": "end_reference_week",
            "months": "weeks",
            "worst_month": "worst_week",
            "best_month": "best_week",
            "positive_month_fraction": "positive_week_fraction",
        }
    )


def _weekly_estimator_history(
    weekly_returns: pd.DataFrame,
    *,
    strategy_assets: Sequence[str],
) -> pd.DataFrame:
    """Return full-history weekly observations eligible for regime estimation.

    The generic return builder can emit a Monday anchor before the price
    snapshot begins when the first source row falls midweek.  Such a period is
    truncated rather than a verifiable weekly holding return, so estimation
    begins with the first Monday anchor on or after common price history.
    """

    assets = tuple(map(str, strategy_assets))
    required = {"reference_week", "start_date", "end_date", *assets}
    missing = required.difference(weekly_returns.columns)
    if missing:
        raise ValueError(
            "weekly estimator source is missing columns: "
            + ", ".join(sorted(missing))
        )
    history = weekly_returns.copy()
    for column in ("reference_week", "start_date", "end_date"):
        history[column] = pd.to_datetime(history[column], errors="raise").dt.normalize()
    first_common_open = pd.Timestamp(history["start_date"].min())
    first_anchor = first_common_open.to_period("W-SUN").start_time.normalize()
    if first_anchor < first_common_open:
        first_anchor += pd.Timedelta(weeks=1)
    history = history.loc[history["reference_week"].ge(first_anchor)].copy()
    if history.empty:
        raise ValueError("weekly estimator history is empty after source-start guard")
    history = history.rename(columns={"end_date": "return_available_at"})
    return history.loc[
        :, ["reference_week", "return_available_at", *assets]
    ].reset_index(drop=True)


def _latest_payload(
    *,
    config: Mapping[str, Any],
    signals: pd.DataFrame,
    weights: pd.DataFrame,
    optimizer_audit: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    baseline = weights.loc[weights["method"].eq(BASELINE_METHOD)]
    latest_week = pd.Timestamp(baseline["reference_week"].max())
    allocation = baseline.loc[baseline["reference_week"].eq(latest_week)]
    anchored = weights.loc[
        weights["method"].eq(ANCHORED_METHOD)
        & weights["reference_week"].eq(latest_week)
    ]
    if anchored.empty:
        raise ValueError("latest anchored active-sleeve target is missing")
    audit = optimizer_audit.loc[
        optimizer_audit["method"].eq(BASELINE_METHOD)
        & optimizer_audit["reference_week"].eq(latest_week)
    ].iloc[0]
    signal = signals.loc[signals["reference_week"].eq(latest_week)].iloc[0]
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_variant": PROMOTED_BASELINE_ID,
        "stage_id": STAGE_ID,
        "status": "research_output_not_investment_advice",
        "reference_week": latest_week,
        "signal_date": signal["signal_date"],
        "execution_date": signal["start_date"],
        "holding_period_complete": bool(signal["is_complete"]),
        "regime_reference_month": signal["regime_reference_month"],
        "mapping_reference_month": signal.get("mapping_reference_month"),
        "execution_assumption": config["timing"]["execution"],
        "latest_price_date": pd.to_datetime(prices["date"]).max(),
        "posterior": [
            {"regime_id": regime_id, "probability": float(signal[column])}
            for regime_id, column in zip(
                CANONICAL_REGIME_IDS, PROBABILITY_COLUMNS, strict=True
            )
        ],
        "target_weights": [
            {"ticker": row.ticker, "weight": float(row.target_weight)}
            for row in allocation.itertuples(index=False)
            if float(row.target_weight) > 1e-12
        ],
        "exploratory_anchored_strategy": {
            "method": ANCHORED_METHOD,
            "role": "exploratory_not_promoted",
            "replaces_promoted_baseline": False,
            "pooled_core_fraction": ANCHORED_CORE_FRACTION,
            "posterior_sleeve_fraction": POSTERIOR_SLEEVE_FRACTION,
            "target_weights": [
                {"ticker": row.ticker, "weight": float(row.target_weight)}
                for row in anchored.itertuples(index=False)
                if float(row.target_weight) > 1e-12
            ],
        },
        "estimated_weekly_return": float(
            audit["expected_portfolio_weekly_return"]
        ),
        "estimated_annualized_volatility": float(
            audit["estimated_annualized_volatility"]
        ),
        "estimated_rebalance_cost": float(audit["estimated_transaction_cost"]),
        "optimization_outcome": audit["optimization_outcome"],
        "binding_constraints": audit["binding_constraints"],
        "baseline_parameters": {
            "regime_mean_pseudo_weeks": float(audit["pseudo_weeks"]),
            "annualized_volatility_cap": float(audit["volatility_cap"]),
            "per_asset_transaction_cost": float(audit["transaction_cost"]),
            "minimum_labeled_weeks": int(
                config["estimation"]["minimum_labeled_weeks"]
            ),
            "estimation_return_frequency": config["estimation"]["return_frequency"],
            "rebalance_frequency": "weekly",
        },
        "universe_note": (
            "Compact US cross-asset ETF universe; AGG is benchmark-only and "
            "receives zero optimized-strategy weight."
        ),
    }


def _declaration(root: Path, path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def build_m02_allocation_backtest(
    *, project_root: Path, config_path: Path
) -> dict[str, Path]:
    """Build targets, weekly accounting, diagnostics, and public artifacts."""

    root = project_root.resolve()
    config_path = _project_path(root, config_path)
    config, config_bytes = _load_config(config_path)
    sources = config["sources"]
    template_path = _project_path(root, sources["model01_allocation_template"])
    template, template_bytes = _load_yaml(template_path)
    _validate_model01_policy_parity(config, template)
    price_path = _project_path(root, sources["etf_daily_prices"])
    price_bytes = price_path.read_bytes()
    prices = pd.read_csv(BytesIO(price_bytes))
    strategy_assets = tuple(map(str, config["universe"]["strategy_assets"]))
    benchmark_assets = tuple(map(str, config["universe"]["benchmark_only_assets"]))
    simulation_assets = strategy_assets + benchmark_assets

    start_week = pd.Timestamp(config["timing"]["formal_backtest_start"])
    start_week = start_week.to_period("W-SUN").start_time.normalize()
    schedule = build_weekly_execution_schedule(
        prices, assets=simulation_assets
    )
    schedule = schedule.loc[schedule["reference_week"].ge(start_week)].reset_index(
        drop=True
    )
    if schedule.empty or not schedule["reference_week"].eq(start_week).any():
        raise ValueError("formal backtest start has no common weekly execution")

    current_config_path = _project_path(root, sources["current_baseline_config"])
    current_manifest_path = _project_path(root, sources["current_baseline_manifest"])
    (
        decision_marginals,
        scores,
        lineage,
        upstream_implementation_paths,
        _,
        current_manifest_bytes,
    ) = (
        _promoted_decision_marginals(
            root=root,
            current_config_path=current_config_path,
            current_manifest_path=current_manifest_path,
            decision_dates=tuple(schedule["reference_week"]),
        )
    )
    signals = select_m02_weekly_signals(
        decision_marginals,
        schedule["reference_week"],
        baseline_variant_id=PROMOTED_BASELINE_ID,
    )
    baseline_metadata = decision_marginals.loc[
        decision_marginals["variant_id"].astype(str).eq(PROMOTED_BASELINE_ID)
    ].copy()
    baseline_metadata["signal_date"] = pd.to_datetime(
        baseline_metadata["signal_date"], errors="raise"
    ).dt.normalize()
    baseline_metadata["regime_reference_month"] = pd.to_datetime(
        baseline_metadata["reference_month"], errors="raise"
    ).dt.to_period("M").dt.to_timestamp()
    metadata_columns = [
        "variant_id",
        "model_role",
        "signal_date",
        "regime_reference_month",
        "as_of_date",
        "same_day_release_evidence_included",
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
    ]
    signals = signals.merge(
        baseline_metadata.loc[:, metadata_columns],
        on=["signal_date", "regime_reference_month"],
        how="left",
        validate="one_to_one",
    )
    if signals["variant_id"].isna().any():
        raise ValueError("weekly signal audit metadata is incomplete")
    signals = signals.merge(
        schedule,
        on="reference_week",
        how="inner",
        validate="one_to_one",
    ).sort_values("reference_week", kind="stable").reset_index(drop=True)
    if len(signals) != len(schedule):
        raise ValueError("promoted replay did not produce every weekly signal")

    history = derive_m02_regime_history(scores)
    all_weekly_returns = build_weekly_open_to_open_holding_returns(
        prices, assets=simulation_assets
    )
    estimator_returns = _weekly_estimator_history(
        all_weekly_returns,
        strategy_assets=strategy_assets,
    )
    engine_returns = all_weekly_returns.loc[
        all_weekly_returns["reference_week"].ge(start_week)
    ].reset_index(drop=True)
    complete_end_week = pd.Timestamp(engine_returns["reference_week"].max())
    if not signals["reference_week"].eq(complete_end_week).any():
        raise ValueError("complete weekly return range lacks a promoted signal")

    specifications = _allocation_specifications(config)
    baseline_specification = specifications[0]
    pooled_specification = AllocationSpecification(
        method=POOLED_METHOD,
        specification_type="benchmark",
        parameter="pooled_mean_ablation",
        parameter_value=1.0,
        kappa=baseline_specification.kappa,
        volatility_cap=baseline_specification.volatility_cap,
        transaction_cost=baseline_specification.transaction_cost,
        cap_multiplier=1.0,
    )
    dynamic_specifications = (*specifications, pooled_specification)
    dynamic_weights, estimate_audit, optimizer_audit = _dynamic_targets(
        specifications=dynamic_specifications,
        signals=signals,
        estimator_returns=estimator_returns,
        regime_history=history,
        prices=prices,
        strategy_assets=strategy_assets,
        simulation_assets=simulation_assets,
        minimum_observations=int(config["estimation"]["minimum_labeled_weeks"]),
        covariance_annualization_factor=float(
            config["optimization"]["covariance_annualization_factor"]
        ),
        asset_caps={
            str(asset): float(value)
            for asset, value in config["optimization"]["asset_caps"].items()
        },
        group_caps=_group_caps(config),
    )
    sleeve_policy = config["additional_strategies"][
        "pooled_anchor_posterior_active_sleeve"
    ]
    anchored_weights = _anchored_active_sleeve_targets(
        dynamic_weights,
        output_method=str(sleeve_policy["method_id"]),
        core_method=str(sleeve_policy["pooled_core_method"]),
        active_method=str(sleeve_policy["posterior_sleeve_method"]),
        active_fraction=float(sleeve_policy["posterior_sleeve_fraction"]),
    )
    benchmark_weights, legacy_audit = _benchmark_targets(
        signals=signals,
        prices=prices,
        regime_history=history,
        strategy_assets=strategy_assets,
        simulation_assets=simulation_assets,
    )
    weights = pd.concat(
        [dynamic_weights, anchored_weights, benchmark_weights], ignore_index=True
    )
    baseline_cost = float(
        config["optimization"]["per_asset_transaction_cost"][strategy_assets[0]]
    )
    costs = {item.method: item.transaction_cost for item in dynamic_specifications}
    costs.update({method: baseline_cost for method in BENCHMARK_METHODS})
    costs[ANCHORED_METHOD] = baseline_cost
    complete_weights = weights.loc[~weights["is_live_only"]].copy()
    simulations: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    for method, frame in complete_weights.groupby("method", sort=False):
        targets = frame.loc[
            :, ["method", "reference_week", "ticker", "target_weight"]
        ]
        simulations.append(
            simulate_weekly_targets(
                targets,
                engine_returns,
                assets=simulation_assets,
                transaction_costs=costs[method],
            )
        )
        target_frames.append(targets)
    weekly = pd.concat(simulations, ignore_index=True)
    nav = build_weekly_daily_nav(
        pd.concat(target_frames, ignore_index=True),
        weekly,
        prices,
        assets=simulation_assets,
    )
    metric_input = _metric_view(weekly)
    bil = engine_returns.set_index("reference_week")["BIL"]
    bil.index.name = "reference_month"
    metrics = _weekly_metric_labels(
        compute_performance_metrics(
            metric_input,
            nav,
            bil_monthly_returns=bil,
            periods_per_year=PERIODS_PER_YEAR,
        )
    )
    uncertainty_config = config["evaluation"]["uncertainty"]
    posterior_uncertainty = paired_circular_block_bootstrap(
        metric_input,
        baseline_method=str(uncertainty_config["baseline_method"]),
        comparator_methods=POSTERIOR_COMPARATOR_METHODS,
        block_length=int(uncertainty_config["block_length_weeks"]),
        n_resamples=int(uncertainty_config["resamples"]),
        confidence_level=float(uncertainty_config["confidence_level"]),
        seed=int(uncertainty_config["random_seed"]),
        periods_per_year=PERIODS_PER_YEAR,
    )
    anchored_uncertainty = paired_circular_block_bootstrap(
        metric_input,
        baseline_method=ANCHORED_METHOD,
        comparator_methods=(POOLED_METHOD,),
        block_length=int(uncertainty_config["block_length_weeks"]),
        n_resamples=int(uncertainty_config["resamples"]),
        confidence_level=float(uncertainty_config["confidence_level"]),
        seed=int(uncertainty_config["random_seed"]),
        periods_per_year=PERIODS_PER_YEAR,
    )
    uncertainty = _weekly_metric_labels(
        pd.concat([posterior_uncertainty, anchored_uncertainty], ignore_index=True)
    )
    sensitivity = _sensitivity_table(metrics, specifications).rename(
        columns={"kappa": "pseudo_weeks"}
    )

    output = config["outputs"]
    processed_dir = _project_path(root, output["processed_dir"])
    published_dir = _project_path(root, output["published_dir"])
    paths = {
        "signal_table": processed_dir / str(output["signal_table"]),
        "holding_returns": processed_dir
        / str(output["weekly_holding_period_returns"]),
        "estimate_audit": processed_dir / str(output["regime_estimate_audit"]),
        "optimizer_audit": processed_dir / str(output["optimizer_audit"]),
        "weights": processed_dir / str(output["weekly_weights"]),
        "weekly": processed_dir / str(output["weekly_strategy_returns"]),
        "daily_nav": processed_dir / str(output["daily_nav"]),
        "metrics": processed_dir / str(output["performance_metrics"]),
        "uncertainty": processed_dir / str(output["comparison_uncertainty"]),
        "sensitivity": processed_dir / str(output["sensitivity_metrics"]),
        "legacy_audit": processed_dir / str(output["legacy_sharpe_audit"]),
        "manifest": _project_path(root, output["manifest"]),
        "latest": published_dir / str(output["latest_allocation"]),
        "summary": published_dir / str(output["backtest_summary"]),
        "published_performance": published_dir
        / str(output["published_performance"]),
        "published_uncertainty": published_dir
        / str(output["published_uncertainty"]),
        "published_weekly": published_dir
        / str(output["published_weekly_returns"]),
        "published_weights": published_dir
        / str(output["published_weekly_weights"]),
        "published_sensitivity": published_dir
        / str(output["published_sensitivity"]),
    }
    _write_csv(signals, paths["signal_table"])
    _write_csv(engine_returns, paths["holding_returns"])
    _write_csv(estimate_audit, paths["estimate_audit"])
    _write_csv(optimizer_audit, paths["optimizer_audit"])
    _write_csv(weights, paths["weights"])
    _write_csv(weekly, paths["weekly"])
    _write_csv(nav, paths["daily_nav"])
    _write_csv(metrics, paths["metrics"])
    _write_csv(uncertainty, paths["uncertainty"])
    _write_csv(sensitivity, paths["sensitivity"])
    _write_csv(legacy_audit, paths["legacy_audit"])
    public_methods = M02_PUBLIC_METHODS
    published_performance = metrics.loc[
        metrics["method"].isin(public_methods)
    ].copy()
    published_weekly = weekly.loc[weekly["method"].isin(public_methods)].copy()
    published_weights = weights.loc[weights["method"].isin(public_methods)].copy()
    _write_csv(published_performance, paths["published_performance"])
    _write_csv(uncertainty, paths["published_uncertainty"])
    _write_csv(published_weekly, paths["published_weekly"])
    _write_csv(published_weights, paths["published_weights"])
    _write_csv(sensitivity, paths["published_sensitivity"])

    latest = _latest_payload(
        config=config,
        signals=signals,
        weights=weights,
        optimizer_audit=optimizer_audit,
        prices=prices,
    )
    _write_json(latest, paths["latest"])
    latest_signal_week = pd.Timestamp(signals["reference_week"].max())
    summary = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_variant": PROMOTED_BASELINE_ID,
        "stage_id": STAGE_ID,
        "status": "historical_research_backtest_not_investment_advice",
        "backtest_start": start_week,
        "backtest_end": complete_end_week,
        "weeks": int(
            weekly.loc[
                weekly["method"].eq(BASELINE_METHOD), "reference_week"
            ].nunique()
        ),
        "latest_live_target_week": latest_signal_week,
        "estimation_return_frequency": config["estimation"]["return_frequency"],
        "holding_return_frequency": "weekly_open_to_open",
        "forecast_horizon": "one_week",
        "holding_horizon": "one_week",
        "forecast_holding_horizons_aligned": True,
        "minimum_labeled_weeks": int(
            config["estimation"]["minimum_labeled_weeks"]
        ),
        "regime_mean_pseudo_weeks": float(
            config["estimation"]["regime_means"]["pseudo_weeks"]
        ),
        "covariance_annualization_factor": float(
            config["optimization"]["covariance_annualization_factor"]
        ),
        "rebalance_frequency": "weekly",
        "methods": list(public_methods),
        "exploratory_anchored_strategy": {
            "method": ANCHORED_METHOD,
            "role": "exploratory_not_promoted",
            "replaces_promoted_baseline": False,
            "construction": "0.75 * pooled target + 0.25 * posterior target",
            "pooled_core_fraction": ANCHORED_CORE_FRACTION,
            "posterior_sleeve_fraction": POSTERIOR_SLEEVE_FRACTION,
            "transaction_cost_policy": "simulate_consolidated_target_path",
        },
        "performance": published_performance.to_dict(orient="records"),
        "paired_block_bootstrap_comparisons": uncertainty.to_dict(orient="records"),
        "sensitivity_design": "one_parameter_at_a_time_not_used_for_baseline_selection",
        "warnings": [
            "The promoted baseline was selected on the same causal history; "
            "this is not a fresh holdout.",
            "Adjusted ETF history is mutable provider data, not point-in-time market data.",
            "Weekly trading raises turnover and uses a fixed five-basis-point "
            "one-way cost approximation.",
            "The source-truncated 2007-12-31 anchor is excluded from weekly estimation.",
            "The latest target is excluded from performance when its next weekly "
            "execution open is unavailable.",
            "The 25% posterior sleeve was requested after reviewing the same-history "
            "baseline results and is exploratory, not an independently validated choice.",
            "The convex target blend inherits linear allocation caps but is not a "
            "separate optimizer solution under one common covariance estimate.",
        ],
    }
    _write_json(summary, paths["summary"])

    generated_names = (
        "signal_table",
        "holding_returns",
        "estimate_audit",
        "optimizer_audit",
        "weights",
        "weekly",
        "daily_nav",
        "metrics",
        "uncertainty",
        "sensitivity",
        "legacy_audit",
        "latest",
        "summary",
        "published_performance",
        "published_uncertainty",
        "published_weekly",
        "published_weights",
        "published_sensitivity",
    )
    stage_implementation_paths = [
        root / "src/regime_allocation/portfolio/m02_pipeline.py",
        root / "src/regime_allocation/portfolio/m02_estimation.py",
        root / "src/regime_allocation/portfolio/m02_weekly.py",
        root
        / "src/regime_allocation/models/m02_soft_composite/weekly_decision_replay.py",
        root / "src/regime_allocation/models/m02_soft_composite/inference_sensitivities.py",
        root / "src/regime_allocation/models/m02_soft_composite/probability_map.py",
        root / "src/regime_allocation/models/m02_soft_composite/robust_var.py",
        root / "src/regime_allocation/backtest/__init__.py",
        root / "src/regime_allocation/backtest/engine.py",
        root / "src/regime_allocation/backtest/metrics.py",
        root / "src/regime_allocation/portfolio/pipeline.py",
        root / "src/regime_allocation/portfolio/estimation.py",
        root / "src/regime_allocation/portfolio/optimization.py",
        root / "src/regime_allocation/portfolio/baselines.py",
        root / "src/regime_allocation/cli/build_m02_backtest.py",
    ]
    implementation_paths = list(
        dict.fromkeys((*upstream_implementation_paths, *stage_implementation_paths))
    )
    input_rows = [
        _declaration(root, template_path),
        _declaration(root, price_path),
        *(_declaration(root, path) for path, _ in lineage.values()),
    ]
    deduplicated_inputs = {
        str(row["path"]): row for row in input_rows
    }
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_variant": PROMOTED_BASELINE_ID,
        "stage_id": STAGE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(root).as_posix(),
        "configuration_sha256": _sha256(config_bytes),
        "upstream_current_baseline_manifest": {
            "path": current_manifest_path.relative_to(root).as_posix(),
            "sha256": _sha256(current_manifest_bytes),
        },
        "model01_policy_template_sha256": _sha256(template_bytes),
        "inputs": list(deduplicated_inputs.values()),
        "implementation_files": [
            _declaration(root, path) for path in implementation_paths
        ],
        "formal_backtest_start": start_week,
        "formal_backtest_end": complete_end_week,
        "latest_live_target_week": latest_signal_week,
        "weekly_observations": int(
            weekly.loc[
                weekly["method"].eq(BASELINE_METHOD), "reference_week"
            ].nunique()
        ),
        "allocation_specification_count": len(specifications),
        "pooled_mean_ablation_included": True,
        "benchmark_count": len(BENCHMARK_METHODS),
        "alternative_strategy_count": 1,
        "anchored_active_sleeve": {
            "method": ANCHORED_METHOD,
            "role": "exploratory_not_promoted",
            "replaces_promoted_baseline": False,
            "construction": "convex_combination_of_weekly_targets",
            "pooled_core_method": POOLED_METHOD,
            "pooled_core_fraction": ANCHORED_CORE_FRACTION,
            "posterior_sleeve_method": BASELINE_METHOD,
            "posterior_sleeve_fraction": POSTERIOR_SLEEVE_FRACTION,
            "transaction_cost_policy": "simulate_consolidated_target_path",
        },
        "periods_per_year": PERIODS_PER_YEAR,
        "horizon_alignment": {
            "estimation_return_frequency": config["estimation"]["return_frequency"],
            "holding_return_frequency": "weekly_open_to_open",
            "aligned": True,
            "minimum_labeled_weeks": int(
                config["estimation"]["minimum_labeled_weeks"]
            ),
            "regime_mean_pseudo_weeks": float(
                config["estimation"]["regime_means"]["pseudo_weeks"]
            ),
            "covariance_annualization_factor": float(
                config["optimization"]["covariance_annualization_factor"]
            ),
        },
        "output_files": [
            _declaration(root, paths[name]) for name in generated_names
        ],
        "credential_policy": (
            "This stage consumes credential-free processed files and never reads "
            "environment credentials or downloads data."
        ),
    }
    _write_json(manifest, paths["manifest"])
    return {
        "manifest": paths["manifest"],
        "latest_allocation": paths["latest"],
        "performance": paths["published_performance"],
        "weekly_returns": paths["published_weekly"],
        "weekly_weights": paths["published_weights"],
        "uncertainty": paths["published_uncertainty"],
        "sensitivities": paths["published_sensitivity"],
    }


__all__ = ["build_m02_allocation_backtest"]
