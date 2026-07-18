"""End-to-end causal portfolio allocation and backtesting for Model 01."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from regime_allocation.backtest import (
    build_daily_nav,
    build_open_to_open_holding_returns,
    compute_performance_metrics,
    paired_circular_block_bootstrap,
    simulate_monthly_targets,
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
    build_adjusted_open_holding_returns,
    extract_post_month_roll_signals,
    fit_causal_regime_return_model,
    posterior_mixture_moments,
)
from regime_allocation.portfolio.optimization import (
    GroupCap,
    OptimizationResult,
    optimize_long_only,
)


BASELINE_METHOD = "posterior_optimized"
POOLED_METHOD = "pooled_mean_optimizer"
BENCHMARK_METHODS = (
    POOLED_METHOD,
    "legacy_sharpe_map",
    "equal_weight",
    "static_60_spy_40_agg",
)


@dataclass(frozen=True)
class AllocationSpecification:
    """One baseline or one-at-a-time allocation sensitivity."""

    method: str
    specification_type: str
    parameter: str
    parameter_value: float
    kappa: float
    volatility_cap: float
    transaction_cost: float
    cap_multiplier: float


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


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d", compression=compression)
    temporary.replace(path)


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("allocation configuration must be a mapping")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported allocation configuration schema")
    if config.get("model_id") != "m01_deterministic_composite":
        raise ValueError("allocation configuration has the wrong model_id")
    if config.get("stage_id") != "posterior_regime_allocation_backtest":
        raise ValueError("allocation configuration has the wrong stage_id")
    assets = config.get("universe", {}).get("strategy_assets")
    if assets != ["SPY", "IEF", "TIP", "HYG", "BIL", "GLD", "LQD"]:
        raise ValueError("configuration strategy universe is not the frozen seven ETFs")
    if config.get("signal", {}).get("regime_order") != list(CANONICAL_REGIME_IDS):
        raise ValueError("configuration regime order is not canonical")
    estimation = config.get("estimation", {})
    if float(estimation.get("regime_means", {}).get("pseudo_months", -1)) != 24.0:
        raise ValueError("baseline regime-mean shrinkage must use 24 pseudo-months")
    if estimation.get("covariance", {}).get("estimator") != "ledoit_wolf":
        raise ValueError("baseline covariance must use Ledoit-Wolf")
    optimization = config.get("optimization", {})
    if float(optimization.get("annualized_volatility_cap", -1)) != 0.10:
        raise ValueError("baseline annualized volatility cap must equal 10%")
    costs = optimization.get("per_asset_transaction_cost", {})
    if set(costs) != set(assets) or any(float(costs[item]) != 0.0005 for item in assets):
        raise ValueError("all baseline strategy-asset transaction costs must equal 5 bp")
    return config, raw


def _allocation_specifications(config: Mapping[str, Any]) -> tuple[AllocationSpecification, ...]:
    baseline_kappa = float(config["estimation"]["regime_means"]["pseudo_months"])
    baseline_volatility = float(config["optimization"]["annualized_volatility_cap"])
    first_asset = config["universe"]["strategy_assets"][0]
    baseline_cost = float(config["optimization"]["per_asset_transaction_cost"][first_asset])
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
    for raw in parameters["regime_mean_pseudo_months"]:
        value = float(raw)
        if value == baseline_kappa:
            continue
        specifications.append(
            AllocationSpecification(
                method=f"sensitivity_kappa_{value:g}",
                specification_type="sensitivity",
                parameter="regime_mean_pseudo_months",
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


def _group_caps(config: Mapping[str, Any]) -> tuple[GroupCap, ...]:
    return tuple(
        GroupCap(
            name=str(name),
            assets=tuple(map(str, value["assets"])),
            maximum=float(value["maximum"]),
        )
        for name, value in config["optimization"]["group_caps"].items()
    )


def _daily_total_returns(prices: pd.DataFrame, assets: Sequence[str]) -> pd.DataFrame:
    selected = prices.loc[
        prices["ticker"].astype(str).isin(assets),
        ["date", "ticker", "adjusted_close"],
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"], errors="raise")
    selected["adjusted_close"] = pd.to_numeric(selected["adjusted_close"], errors="raise")
    closes = selected.pivot(index="date", columns="ticker", values="adjusted_close")
    closes = closes.reindex(columns=list(assets)).sort_index()
    if closes.isna().any().any() or (closes <= 0).any().any():
        raise ValueError("daily adjusted closes must form a complete positive panel")
    returns = closes.pct_change(fill_method=None).dropna(how="any")
    output = returns.reset_index().rename_axis(columns=None)
    return output


def _known_pretrade_weights(
    prices: pd.DataFrame,
    *,
    assets: Sequence[str],
    previous_target: np.ndarray | None,
    previous_entry_date: pd.Timestamp | None,
    signal_date: pd.Timestamp,
) -> np.ndarray:
    if previous_target is None:
        return np.zeros(len(assets), dtype=float)
    if previous_entry_date is None:
        raise RuntimeError("previous entry date is required after initial formation")
    selected = prices.loc[
        prices["ticker"].astype(str).isin(assets),
        ["date", "ticker", "adjusted_open", "adjusted_close"],
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"], errors="raise")
    entry = selected.loc[selected["date"].eq(previous_entry_date)].set_index("ticker")
    known = selected.loc[selected["date"].lt(signal_date)]
    if known.empty:
        raise ValueError("no close is available before the allocation signal")
    last_date = pd.Timestamp(known["date"].max())
    last = known.loc[known["date"].eq(last_date)].set_index("ticker")
    if set(assets).difference(entry.index) or set(assets).difference(last.index):
        raise ValueError("pretrade drift data are incomplete")
    entry_open = entry.loc[list(assets), "adjusted_open"].to_numpy(dtype=float)
    last_close = last.loc[list(assets), "adjusted_close"].to_numpy(dtype=float)
    gross_values = previous_target * last_close / entry_open
    total = float(gross_values.sum())
    if not np.isfinite(gross_values).all() or total <= 0:
        raise RuntimeError("known pretrade weight calculation failed")
    return gross_values / total


def _posterior_from_signal(signal: pd.Series) -> pd.Series:
    values = signal.loc[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    return pd.Series(values, index=CANONICAL_REGIME_IDS, dtype=float)


def _pooled_mean_moments(estimate: RegimeReturnEstimate) -> PosteriorMixtureMoments:
    """Return a causal no-current-regime ablation using the same risk estimator."""

    frequencies = estimate.regime_counts.astype(float) / float(estimate.training_count)
    assets = list(estimate.asset_ids)
    sample_means = estimate.regime_sample_means.loc[
        list(CANONICAL_REGIME_IDS), assets
    ].copy()
    for regime_id in CANONICAL_REGIME_IDS:
        if int(estimate.regime_counts.loc[regime_id]) == 0:
            sample_means.loc[regime_id] = estimate.global_mean.loc[assets]
    global_mean = estimate.global_mean.loc[assets].to_numpy(dtype=float)
    centered = sample_means.to_numpy(dtype=float) - global_mean
    between = np.einsum(
        "r,ri,rj->ij",
        frequencies.loc[list(CANONICAL_REGIME_IDS)].to_numpy(dtype=float),
        centered,
        centered,
    )
    within = estimate.shared_within_covariance.loc[assets, assets].to_numpy(dtype=float)
    total = (within + between + (within + between).T) / 2.0
    index = pd.Index(assets, name="asset_id")
    return PosteriorMixtureMoments(
        posterior=frequencies.rename("historical_regime_frequency"),
        expected_mean=estimate.global_mean.loc[assets].rename("expected_return"),
        within_regime_covariance=pd.DataFrame(within, index=index, columns=assets),
        between_regime_covariance=pd.DataFrame(between, index=index, columns=assets),
        covariance=pd.DataFrame(total, index=index, columns=assets),
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
            records.append(
                {
                    "method": specification.method,
                    "holding_month": signal["holding_month"],
                    "signal_date": signal["signal_date"],
                    "knowledge_cutoff": estimate.knowledge_cutoff,
                    "kappa": estimate.kappa,
                    "training_count": estimate.training_count,
                    "first_training_holding_month": estimate.first_holding_month,
                    "last_training_holding_month": estimate.last_holding_month,
                    "regime_id": regime_id,
                    "regime_count": int(estimate.regime_counts.loc[regime_id]),
                    "ticker": asset,
                    "global_mean": float(estimate.global_mean.loc[asset]),
                    "regime_sample_mean": float(
                        estimate.regime_sample_means.loc[regime_id, asset]
                    )
                    if pd.notna(estimate.regime_sample_means.loc[regime_id, asset])
                    else float("nan"),
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
        "holding_month": signal["holding_month"],
        "signal_date": signal["signal_date"],
        "checkpoint_id": signal["checkpoint_id"],
        "estimated_pretrade_as_of": estimated_pretrade_as_of,
        "kappa": specification.kappa,
        "volatility_cap": specification.volatility_cap,
        "transaction_cost": specification.transaction_cost,
        "concentration_cap_multiplier": specification.cap_multiplier,
        "current_regime_posterior_used": specification.method != POOLED_METHOD,
        "posterior_entropy": float(-(posterior * np.log(posterior.clip(lower=1e-300))).sum()),
        "expected_portfolio_monthly_return": result.expected_monthly_return,
        "estimated_transaction_cost": result.estimated_transaction_cost,
        "net_expected_monthly_return": result.net_expected_monthly_return,
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
            f"expected_return_{asset}": float(expected_returns.loc[asset])
            for asset in result.assets
        }
    )
    return record


def _expanded_weight_records(
    *,
    method: str,
    reference_month: pd.Timestamp,
    signal_date: pd.Timestamp,
    weights: pd.Series,
    simulation_assets: Sequence[str],
    specification_type: str,
    is_live_only: bool,
    audit: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    records = []
    for ticker in simulation_assets:
        row: dict[str, object] = {
            "method": method,
            "specification_type": specification_type,
            "reference_month": reference_month,
            "signal_date": signal_date,
            "ticker": ticker,
            "target_weight": float(weights.get(ticker, 0.0)),
            "is_live_only": bool(is_live_only),
        }
        if audit:
            row.update(audit)
        records.append(row)
    return records


def _build_dynamic_targets(
    *,
    specifications: Sequence[AllocationSpecification],
    signals: pd.DataFrame,
    estimator_returns: pd.DataFrame,
    regime_history: pd.DataFrame,
    prices: pd.DataFrame,
    strategy_assets: Sequence[str],
    simulation_assets: Sequence[str],
    complete_end_month: pd.Timestamp,
    minimum_observations: int,
    asset_caps: Mapping[str, float],
    group_caps: Sequence[GroupCap],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weight_records: list[dict[str, object]] = []
    estimate_records: list[dict[str, object]] = []
    optimizer_records: list[dict[str, object]] = []
    estimate_cache: dict[tuple[pd.Timestamp, float], RegimeReturnEstimate] = {}
    moment_cache: dict[tuple[pd.Timestamp, float, str], Any] = {}
    entry_dates = estimator_returns.set_index("holding_month")["entry_date"]
    price_dates = pd.to_datetime(prices["date"], errors="raise")
    for specification in specifications:
        previous_target: np.ndarray | None = None
        previous_entry_date: pd.Timestamp | None = None
        for _, signal in signals.iterrows():
            holding_month = pd.Timestamp(signal["holding_month"])
            cache_key = (holding_month, specification.kappa)
            estimate = estimate_cache.get(cache_key)
            if estimate is None:
                estimate = fit_causal_regime_return_model(
                    estimator_returns,
                    regime_history,
                    knowledge_cutoff=signal["transition_training_cutoff"],
                    asset_ids=strategy_assets,
                    kappa=specification.kappa,
                    minimum_observations=minimum_observations,
                )
                estimate_cache[cache_key] = estimate
            moment_key = (
                holding_month,
                specification.kappa,
                "pooled" if specification.method == POOLED_METHOD else "posterior",
            )
            moments = moment_cache.get(moment_key)
            if moments is None:
                if specification.method == POOLED_METHOD:
                    moments = _pooled_mean_moments(estimate)
                else:
                    moments = posterior_mixture_moments(
                        _posterior_from_signal(signal), estimate
                    )
                moment_cache[moment_key] = moments
            pretrade = _known_pretrade_weights(
                prices,
                assets=strategy_assets,
                previous_target=previous_target,
                previous_entry_date=previous_entry_date,
                signal_date=pd.Timestamp(signal["signal_date"]),
            )
            estimated_pretrade_as_of = None
            if previous_target is not None:
                estimated_pretrade_as_of = pd.Timestamp(
                    price_dates.loc[price_dates.lt(signal["signal_date"])].max()
                )
            scaled_asset_caps = {
                asset: (
                    1.0
                    if asset == "BIL"
                    else min(1.0, float(asset_caps[asset]) * specification.cap_multiplier)
                )
                for asset in strategy_assets
            }
            scaled_group_caps = tuple(
                GroupCap(
                    name=group.name,
                    assets=group.assets,
                    maximum=min(1.0, group.maximum * specification.cap_multiplier),
                )
                for group in group_caps
            )
            result = optimize_long_only(
                assets=strategy_assets,
                expected_monthly_returns=moments.expected_mean.to_numpy(dtype=float),
                annualized_covariance=moments.covariance.to_numpy(dtype=float) * 12.0,
                pretrade_weights=pretrade,
                asset_caps=scaled_asset_caps,
                group_caps=scaled_group_caps,
                transaction_costs=specification.transaction_cost,
                volatility_cap=specification.volatility_cap,
                cash_asset="BIL",
            )
            posterior = _posterior_from_signal(signal)
            audit = _record_optimizer(
                specification=specification,
                signal=signal,
                result=result,
                posterior=posterior,
                expected_returns=moments.expected_mean,
                estimated_pretrade_as_of=estimated_pretrade_as_of,
            )
            optimizer_records.append(audit)
            _record_estimate(
                estimate_records,
                specification=specification,
                signal=signal,
                estimate=estimate,
            )
            target = pd.Series(result.weight_by_asset, dtype=float)
            weight_records.extend(
                _expanded_weight_records(
                    method=specification.method,
                    reference_month=holding_month,
                    signal_date=pd.Timestamp(signal["signal_date"]),
                    weights=target,
                    simulation_assets=simulation_assets,
                    specification_type=specification.specification_type,
                    is_live_only=holding_month > complete_end_month,
                    audit={
                        "posterior_map_regime": str(posterior.idxmax()),
                        "optimization_outcome": result.audit.outcome,
                    },
                )
            )
            previous_target = result.as_array()
            if holding_month in entry_dates.index:
                previous_entry_date = pd.Timestamp(entry_dates.loc[holding_month])
    return (
        pd.DataFrame.from_records(weight_records),
        pd.DataFrame.from_records(estimate_records),
        pd.DataFrame.from_records(optimizer_records),
    )


def _build_benchmark_targets(
    *,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    regime_history: pd.DataFrame,
    strategy_assets: Sequence[str],
    simulation_assets: Sequence[str],
    complete_end_month: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_returns = _daily_total_returns(prices, strategy_assets)
    records: list[dict[str, object]] = []
    legacy_audit: list[dict[str, object]] = []
    equal = equal_weight_target(strategy_assets)
    static = static_60_40_target(
        simulation_assets,
        risk_asset="SPY",
        defensive_asset="AGG",
        risk_weight=0.60,
    )
    for _, signal in signals.iterrows():
        month = pd.Timestamp(signal["holding_month"])
        signal_date = pd.Timestamp(signal["signal_date"])
        live = month > complete_end_month
        posterior = _posterior_from_signal(signal)
        legacy = legacy_sharpe_map(
            daily_returns,
            regime_history,
            posterior,
            signal_date=signal_date,
            asset_ids=strategy_assets,
            minimum_observations=60,
            annualization_factor=252,
            c_shift=0.1,
        )
        records.extend(
            _expanded_weight_records(
                method="legacy_sharpe_map",
                reference_month=month,
                signal_date=signal_date,
                weights=legacy.weights,
                simulation_assets=simulation_assets,
                specification_type="benchmark",
                is_live_only=live,
                audit={
                    "posterior_map_regime": legacy.map_regime_id,
                    "optimization_outcome": "legacy_formula",
                },
            )
        )
        records.extend(
            _expanded_weight_records(
                method="equal_weight",
                reference_month=month,
                signal_date=signal_date,
                weights=equal,
                simulation_assets=simulation_assets,
                specification_type="benchmark",
                is_live_only=live,
                audit={"optimization_outcome": "static_formula"},
            )
        )
        records.extend(
            _expanded_weight_records(
                method="static_60_spy_40_agg",
                reference_month=month,
                signal_date=signal_date,
                weights=static,
                simulation_assets=simulation_assets,
                specification_type="benchmark",
                is_live_only=live,
                audit={"optimization_outcome": "static_formula"},
            )
        )
        for asset in strategy_assets:
            legacy_audit.append(
                {
                    "method": "legacy_sharpe_map",
                    "holding_month": month,
                    "signal_date": signal_date,
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
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(legacy_audit)


def _simulate_methods(
    *,
    weights: pd.DataFrame,
    holding_returns: pd.DataFrame,
    prices: pd.DataFrame,
    simulation_assets: Sequence[str],
    specifications: Sequence[AllocationSpecification],
    baseline_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost_by_method = {item.method: item.transaction_cost for item in specifications}
    cost_by_method.update({method: baseline_cost for method in BENCHMARK_METHODS})
    simulations: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    complete = weights.loc[~weights["is_live_only"]].copy()
    for method, frame in complete.groupby("method", sort=False):
        targets = frame.loc[
            :, ["method", "reference_month", "ticker", "target_weight"]
        ]
        simulation = simulate_monthly_targets(
            targets,
            holding_returns,
            assets=simulation_assets,
            transaction_costs=cost_by_method[method],
        )
        simulations.append(simulation)
        target_frames.append(targets)
    monthly = pd.concat(simulations, ignore_index=True)
    targets = pd.concat(target_frames, ignore_index=True)
    nav = build_daily_nav(
        targets,
        monthly,
        prices,
        assets=simulation_assets,
    )
    return monthly, nav


def _sensitivity_table(
    metrics: pd.DataFrame,
    specifications: Sequence[AllocationSpecification],
) -> pd.DataFrame:
    baseline = metrics.loc[metrics["method"].eq(BASELINE_METHOD)].iloc[0]
    records = []
    metric_names = [
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_zero_rate",
        "sharpe_excess_bil",
        "maximum_drawdown",
        "annualized_one_way_turnover",
        "annualized_cost_drag",
    ]
    by_method = metrics.set_index("method")
    for specification in specifications:
        row: dict[str, object] = {
            "method": specification.method,
            "specification_type": specification.specification_type,
            "parameter": specification.parameter,
            "parameter_value": specification.parameter_value,
            "kappa": specification.kappa,
            "annualized_volatility_cap": specification.volatility_cap,
            "per_asset_transaction_cost": specification.transaction_cost,
            "concentration_cap_multiplier": specification.cap_multiplier,
        }
        values = by_method.loc[specification.method]
        for metric in metric_names:
            row[metric] = float(values[metric])
            row[f"delta_{metric}_vs_baseline"] = float(values[metric] - baseline[metric])
        records.append(row)
    return pd.DataFrame.from_records(records)


def _latest_allocation_payload(
    *,
    weights: pd.DataFrame,
    optimizer_audit: pd.DataFrame,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = weights.loc[weights["method"].eq(BASELINE_METHOD)]
    latest_month = pd.Timestamp(baseline["reference_month"].max())
    allocation = baseline.loc[baseline["reference_month"].eq(latest_month)]
    audit = optimizer_audit.loc[
        optimizer_audit["method"].eq(BASELINE_METHOD)
        & optimizer_audit["holding_month"].eq(latest_month)
    ].iloc[0]
    signal = signals.loc[signals["holding_month"].eq(latest_month)].iloc[0]
    return {
        "schema_version": 1,
        "model_id": config["model_id"],
        "stage_id": config["stage_id"],
        "status": "research_output_not_investment_advice",
        "reference_month": latest_month,
        "signal_date": signal["signal_date"],
        "signal_checkpoint_id": signal["checkpoint_id"],
        "execution_assumption": config["timing"]["execution"],
        "latest_price_date": pd.to_datetime(prices["date"]).max(),
        "posterior": [
            {
                "regime_id": regime_id,
                "probability": float(signal[column]),
            }
            for regime_id, column in zip(
                CANONICAL_REGIME_IDS, PROBABILITY_COLUMNS, strict=True
            )
        ],
        "target_weights": [
            {
                "ticker": row.ticker,
                "weight": float(row.target_weight),
            }
            for row in allocation.itertuples(index=False)
            if float(row.target_weight) > 1e-12
        ],
        "estimated_monthly_return": float(audit["expected_portfolio_monthly_return"]),
        "estimated_annualized_volatility": float(
            audit["estimated_annualized_volatility"]
        ),
        "estimated_rebalance_cost": float(audit["estimated_transaction_cost"]),
        "optimization_outcome": audit["optimization_outcome"],
        "binding_constraints": audit["binding_constraints"],
        "baseline_parameters": {
            "regime_mean_pseudo_months": float(audit["kappa"]),
            "annualized_volatility_cap": float(audit["volatility_cap"]),
            "per_asset_transaction_cost": float(audit["transaction_cost"]),
            "minimum_labeled_months": int(
                config["estimation"]["minimum_labeled_months"]
            ),
        },
        "universe_note": (
            "Compact US cross-asset ETF universe; not a complete global FICC universe. "
            "AGG is benchmark-only and receives zero strategy weight."
        ),
    }


def build_allocation_backtest(
    *,
    project_root: Path,
    config_path: Path,
) -> dict[str, Path]:
    """Build all Model 01 targets, simulations, diagnostics, and public artifacts."""

    config, config_bytes = _load_config(config_path)
    sources = config["sources"]
    source_paths = {
        name: project_root / str(path)
        for name, path in sources.items()
    }
    source_bytes = {name: path.read_bytes() for name, path in source_paths.items()}
    checkpoints = pd.read_csv(source_paths["checkpoint_index"])
    marginals = pd.read_csv(source_paths["marginal_checkpoints"])
    history = pd.read_csv(source_paths["regime_history"])
    prices = pd.read_csv(source_paths["etf_daily_prices"])
    strategy_assets = tuple(map(str, config["universe"]["strategy_assets"]))
    benchmark_assets = tuple(map(str, config["universe"]["benchmark_only_assets"]))
    simulation_assets = strategy_assets + benchmark_assets
    signals = extract_post_month_roll_signals(
        checkpoints,
        marginals,
        specification_id=str(config["signal"]["specification_id"]),
    )
    start_month = pd.Timestamp(config["timing"]["formal_backtest_start"]).to_period(
        "M"
    ).to_timestamp()
    signals = signals.loc[signals["holding_month"].ge(start_month)].reset_index(drop=True)
    engine_returns = build_open_to_open_holding_returns(
        prices,
        assets=simulation_assets,
    )
    estimator_returns = build_adjusted_open_holding_returns(
        prices,
        asset_ids=strategy_assets,
    )
    complete_end_month = pd.Timestamp(engine_returns["reference_month"].max())
    signals = signals.loc[
        signals["holding_month"].le(
            (complete_end_month.to_period("M") + 1).to_timestamp()
        )
    ].reset_index(drop=True)
    if signals.empty or not signals["holding_month"].eq(start_month).any():
        raise ValueError("formal backtest start has no causal monthly signal")
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
    dynamic_weights, estimate_audit, optimizer_audit = _build_dynamic_targets(
        specifications=dynamic_specifications,
        signals=signals,
        estimator_returns=estimator_returns,
        regime_history=history,
        prices=prices,
        strategy_assets=strategy_assets,
        simulation_assets=simulation_assets,
        complete_end_month=complete_end_month,
        minimum_observations=int(config["estimation"]["minimum_labeled_months"]),
        asset_caps={
            str(asset): float(value)
            for asset, value in config["optimization"]["asset_caps"].items()
        },
        group_caps=_group_caps(config),
    )
    benchmark_weights, legacy_audit = _build_benchmark_targets(
        signals=signals,
        prices=prices,
        regime_history=history,
        strategy_assets=strategy_assets,
        simulation_assets=simulation_assets,
        complete_end_month=complete_end_month,
    )
    weights = pd.concat([dynamic_weights, benchmark_weights], ignore_index=True)
    baseline_cost = float(
        config["optimization"]["per_asset_transaction_cost"][strategy_assets[0]]
    )
    monthly, daily_nav = _simulate_methods(
        weights=weights,
        holding_returns=engine_returns,
        prices=prices,
        simulation_assets=simulation_assets,
        specifications=dynamic_specifications,
        baseline_cost=baseline_cost,
    )
    bil_returns = engine_returns.set_index("reference_month")["BIL"]
    metrics = compute_performance_metrics(
        monthly,
        daily_nav,
        bil_monthly_returns=bil_returns,
        periods_per_year=12,
    )
    uncertainty_config = config["evaluation"]["uncertainty"]
    uncertainty = paired_circular_block_bootstrap(
        monthly,
        baseline_method=str(uncertainty_config["baseline_method"]),
        comparator_methods=BENCHMARK_METHODS,
        block_length=int(uncertainty_config["block_length_months"]),
        n_resamples=int(uncertainty_config["resamples"]),
        confidence_level=float(uncertainty_config["confidence_level"]),
        seed=int(uncertainty_config["random_seed"]),
    )
    sensitivity = _sensitivity_table(metrics, specifications)
    outputs = config["outputs"]
    processed_dir = project_root / str(outputs["processed_dir"])
    published_dir = project_root / str(outputs["published_dir"])
    paths = {
        "signal_table": processed_dir / str(outputs["signal_table"]),
        "holding_returns": processed_dir / str(outputs["holding_period_returns"]),
        "estimate_audit": processed_dir / str(outputs["regime_estimate_audit"]),
        "optimizer_audit": processed_dir / str(outputs["optimizer_audit"]),
        "weights": processed_dir / str(outputs["monthly_weights"]),
        "monthly": processed_dir / str(outputs["monthly_strategy_returns"]),
        "daily_nav": processed_dir / str(outputs["daily_nav"]),
        "metrics": processed_dir / str(outputs["performance_metrics"]),
        "uncertainty": processed_dir / str(outputs["comparison_uncertainty"]),
        "sensitivity": processed_dir / str(outputs["sensitivity_metrics"]),
        "legacy_audit": processed_dir / "legacy_sharpe_audit.csv",
        "manifest": project_root / str(outputs["manifest"]),
        "latest": published_dir / str(outputs["latest_allocation"]),
        "summary": published_dir / str(outputs["backtest_summary"]),
        "published_performance": published_dir / str(outputs["published_performance"]),
        "published_uncertainty": published_dir
        / str(outputs["published_uncertainty"]),
        "published_monthly": published_dir / str(outputs["published_monthly_returns"]),
        "published_weights": published_dir / str(outputs["published_monthly_weights"]),
        "published_sensitivity": published_dir / str(outputs["published_sensitivity"]),
    }
    engine_returns = engine_returns.loc[
        engine_returns["reference_month"].ge(start_month)
    ].reset_index(drop=True)
    _write_csv(signals, paths["signal_table"])
    _write_csv(engine_returns, paths["holding_returns"])
    _write_csv(estimate_audit, paths["estimate_audit"])
    _write_csv(optimizer_audit, paths["optimizer_audit"])
    _write_csv(weights, paths["weights"])
    _write_csv(monthly, paths["monthly"])
    _write_csv(daily_nav, paths["daily_nav"])
    _write_csv(metrics, paths["metrics"])
    _write_csv(uncertainty, paths["uncertainty"])
    _write_csv(sensitivity, paths["sensitivity"])
    _write_csv(legacy_audit, paths["legacy_audit"])
    public_methods = (BASELINE_METHOD, *BENCHMARK_METHODS)
    published_performance = metrics.loc[metrics["method"].isin(public_methods)].copy()
    published_monthly = monthly.loc[monthly["method"].isin(public_methods)].copy()
    published_weights = weights.loc[weights["method"].isin(public_methods)].copy()
    _write_csv(published_performance, paths["published_performance"])
    _write_csv(uncertainty, paths["published_uncertainty"])
    _write_csv(published_monthly, paths["published_monthly"])
    _write_csv(published_weights, paths["published_weights"])
    _write_csv(sensitivity, paths["published_sensitivity"])
    latest = _latest_allocation_payload(
        weights=weights,
        optimizer_audit=optimizer_audit,
        signals=signals,
        prices=prices,
        config=config,
    )
    _write_json(latest, paths["latest"])
    summary = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "stage_id": config["stage_id"],
        "status": "historical_research_backtest_not_investment_advice",
        "backtest_start": start_month,
        "backtest_end": complete_end_month,
        "months": int(
            monthly.loc[monthly["method"].eq(BASELINE_METHOD), "reference_month"].nunique()
        ),
        "latest_live_target_month": pd.Timestamp(signals["holding_month"].max()),
        "universe_assessment": (
            "Appropriate as a compact, liquid US cross-asset test universe, with "
            "material overlap across IEF/TIP/LQD/HYG and without global currency, "
            "commodity-sector, or non-US bond breadth."
        ),
        "methods": list(public_methods),
        "performance": published_performance.to_dict(orient="records"),
        "paired_block_bootstrap_comparisons": uncertainty.to_dict(orient="records"),
        "sensitivity_design": "one_parameter_at_a_time_not_used_for_baseline_selection",
        "warnings": [
            "Adjusted ETF history is used; this is not a survivorship-bias-free index study.",
            "Expected returns are difficult to estimate and results are sample-dependent.",
            "Transaction costs are a fixed one-way approximation and exclude taxes/market impact.",
            "The July 2026 target is published but excluded from performance "
            "because its holding month is incomplete.",
        ],
    }
    _write_json(summary, paths["summary"])
    generated_paths = [
        paths[name]
        for name in (
            "signal_table",
            "holding_returns",
            "estimate_audit",
            "optimizer_audit",
            "weights",
            "monthly",
            "daily_nav",
            "metrics",
            "uncertainty",
            "sensitivity",
            "legacy_audit",
            "latest",
            "summary",
            "published_performance",
            "published_uncertainty",
            "published_monthly",
            "published_weights",
            "published_sensitivity",
        )
    ]
    implementation_paths = [
        project_root / "src/regime_allocation/portfolio/pipeline.py",
        project_root / "src/regime_allocation/portfolio/estimation.py",
        project_root / "src/regime_allocation/portfolio/optimization.py",
        project_root / "src/regime_allocation/portfolio/baselines.py",
        project_root / "src/regime_allocation/backtest/engine.py",
        project_root / "src/regime_allocation/backtest/metrics.py",
        project_root / "src/regime_allocation/cli/build_m01_backtest.py",
    ]
    manifest = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "stage_id": config["stage_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(project_root).as_posix(),
        "configuration_sha256": _sha256(config_bytes),
        "inputs": {
            name: {
                "path": source_paths[name].relative_to(project_root).as_posix(),
                "bytes": len(source_bytes[name]),
                "sha256": _sha256(source_bytes[name]),
            }
            for name in source_paths
        },
        "implementation_files": [
            {
                "path": path.relative_to(project_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path.read_bytes()),
            }
            for path in implementation_paths
        ],
        "formal_backtest_start": start_month,
        "formal_backtest_end": complete_end_month,
        "latest_live_target_month": pd.Timestamp(signals["holding_month"].max()),
        "allocation_specification_count": len(specifications),
        "pooled_mean_ablation_included": True,
        "benchmark_count": len(BENCHMARK_METHODS),
        "output_files": [
            {
                "path": path.relative_to(project_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path.read_bytes()),
            }
            for path in generated_paths
        ],
        "credential_policy": (
            "This stage consumes credential-free processed files and never reads, "
            "logs, or serializes FRED_API_KEY or brokerage credentials."
        ),
    }
    _write_json(manifest, paths["manifest"])
    return paths
