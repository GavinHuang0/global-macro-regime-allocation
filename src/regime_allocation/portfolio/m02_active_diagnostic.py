"""Pinned-posterior active optimizer and conditional oracle diagnostic.

The stage consumes the already-published Model 02 weekly backtest under pinned
hashes.  "Pinned" describes this experiment's input snapshot, not Model 02's
governance status.  The stage never reruns or alters the Bayesian filter.  The
weekly posterior is translated through the same causal return estimator, and the
recomputed posterior and pooled expected returns must match the pinned
optimizer audit before any new target is accepted.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from regime_allocation.portfolio.estimation import (
    CANONICAL_REGIME_IDS,
    PROBABILITY_COLUMNS,
    RegimeReturnEstimate,
    posterior_mixture_moments,
)
from regime_allocation.portfolio.m02_active_optimization import (
    ActiveOptimizationResult,
    optimize_benchmark_relative_active,
)
from regime_allocation.portfolio.m02_estimation import (
    fit_causal_weekly_regime_return_model,
)
from regime_allocation.portfolio.m02_pipeline import (
    PERIODS_PER_YEAR,
    _declaration,
    _metric_view,
    _weekly_estimator_history,
    _weekly_metric_labels,
)
from regime_allocation.portfolio.m02_weekly import derive_m02_regime_history
from regime_allocation.portfolio.optimization import GroupCap
from regime_allocation.portfolio.pipeline import (
    BASELINE_METHOD,
    POOLED_METHOD,
    _known_pretrade_weights,
    _pooled_mean_moments,
    _posterior_from_signal,
    _write_csv,
    _write_json,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "benchmark_relative_active_optimizer_diagnostic"
ACTIVE_METHOD = "pooled_relative_posterior_active_te100bp"
ORACLE_METHOD = "oracle_regime_pooled_relative_active_te100bp"
STATIC_METHOD = "static_60_spy_40_agg"
TRACKING_ERROR_CAP = 0.01
ACTIVE_WEIGHT_CAP = 0.05
ONE_WAY_ACTIVE_CAP = 0.10
TOTAL_VOLATILITY_CAP = 0.10
TRANSACTION_COST = 0.0005
FROZEN_MANIFEST_SHA256 = (
    "5821824dcfe6a82347d1df176467b74d0199cb7724bb08f5e07daf08801fbc4a"
)
FROZEN_SIGNAL_SHA256 = (
    "8cb4d27056ec5b024ad5e2cbf4549429da98def25b8602e91275731e8046111f"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _project_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("active diagnostic configuration must be a mapping")
    if (
        int(config.get("schema_version", 0)) != 1
        or config.get("model_id") != MODEL_ID
        or config.get("stage_id") != STAGE_ID
    ):
        raise ValueError("unexpected active diagnostic configuration identity")
    frozen = config.get("frozen_upstream", {})
    if frozen.get("manifest_sha256") != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen Model 02 backtest manifest hash differs")
    if frozen.get("signal_table_sha256") != FROZEN_SIGNAL_SHA256:
        raise ValueError("frozen posterior signal-table hash differs")
    expected_active = {
        "method_id": ACTIVE_METHOD,
        "role": "exploratory_not_promoted",
        "replaces_promoted_baseline": False,
        "pooled_benchmark_method": POOLED_METHOD,
        "posterior_source_method": BASELINE_METHOD,
        "incremental_expected_return": "posterior_minus_pooled",
        "covariance": "causal_pooled_annualized_covariance",
        "covariance_annualization_factor": 52.0,
        "annualized_tracking_error_cap": TRACKING_ERROR_CAP,
        "maximum_absolute_active_weight": ACTIVE_WEIGHT_CAP,
        "maximum_one_way_active_exposure": ONE_WAY_ACTIVE_CAP,
        "annualized_total_volatility_cap": TOTAL_VOLATILITY_CAP,
        "transaction_cost": TRANSACTION_COST,
        "asset_caps": {
            "SPY": 0.35,
            "IEF": 0.50,
            "TIP": 0.40,
            "HYG": 0.25,
            "BIL": 1.00,
            "GLD": 0.25,
            "LQD": 0.40,
        },
        "group_caps": {
            "equity_and_high_yield": {
                "assets": ["SPY", "HYG"],
                "maximum": 0.50,
            },
            "corporate_credit": {
                "assets": ["HYG", "LQD"],
                "maximum": 0.50,
            },
            "rate_sensitive": {
                "assets": ["IEF", "TIP", "LQD"],
                "maximum": 0.75,
            },
        },
        "solver": {
            "method": "SLSQP",
            "tolerance": 1.0e-12,
            "maximum_iterations": 2000,
            "feasibility_tolerance": 1.0e-7,
            "fallback": "pooled_target",
        },
    }
    if config.get("active_optimizer") != expected_active:
        raise ValueError("locked benchmark-relative active specification differs")
    gate = config.get("significance_gate", {})
    expected_gate = {
        "required_comparators": [POOLED_METHOD, STATIC_METHOD],
        "return": "net_weekly_return",
        "statistic": "annualized_arithmetic_mean_difference",
        "bootstrap": "paired_circular_block",
        "block_length_weeks": 26,
        "resamples": 10000,
        "confidence_level": 0.95,
        "random_seed": 20260718,
        "pass_rule": (
            "lower_two_sided_confidence_bound_strictly_above_zero_for_all_comparators"
        ),
        "on_fail": "run_realized_current_month_regime_oracle",
    }
    if gate != expected_gate:
        raise ValueError("locked active significance gate differs")
    oracle = config.get("oracle_diagnostic", {})
    if (
        oracle.get("method_id") != ORACLE_METHOD
        or oracle.get("role") != "infeasible_lookahead_diagnostic"
        or oracle.get("trigger") != "significance_gate_failure"
    ):
        raise ValueError("locked oracle diagnostic identity differs")
    return config, raw


def _read_verified(root: Path, relative_path: object, expected_hash: object) -> tuple[Path, bytes]:
    path = _project_path(root, relative_path)
    raw = path.read_bytes()
    if _sha256(raw) != str(expected_hash):
        raise ValueError(f"frozen source hash mismatch: {path.relative_to(root)}")
    return path, raw


def _manifest_declaration(
    manifest: Mapping[str, Any], *, path: Path, root: Path
) -> Mapping[str, Any]:
    relative = path.relative_to(root).as_posix()
    matches = [
        row
        for section in ("inputs", "output_files")
        for row in manifest.get(section, ())
        if isinstance(row, Mapping) and str(row.get("path")) == relative
    ]
    if len(matches) != 1:
        raise ValueError(f"upstream manifest does not uniquely declare {relative}")
    return matches[0]


def _group_caps(config: Mapping[str, Any]) -> tuple[GroupCap, ...]:
    return tuple(
        GroupCap(
            name=str(name),
            assets=tuple(map(str, declaration["assets"])),
            maximum=float(declaration["maximum"]),
        )
        for name, declaration in config["active_optimizer"]["group_caps"].items()
    )


def _posterior_series(
    signal: pd.Series, *, oracle_regime_id: str | None
) -> pd.Series:
    if oracle_regime_id is None:
        return _posterior_from_signal(signal)
    if oracle_regime_id not in CANONICAL_REGIME_IDS:
        raise ValueError(f"unknown oracle regime {oracle_regime_id!r}")
    return pd.Series(
        {
            regime_id: float(regime_id == oracle_regime_id)
            for regime_id in CANONICAL_REGIME_IDS
        },
        dtype=float,
    )


def _frozen_expected_returns(
    audit: pd.DataFrame,
    *,
    method: str,
    reference_week: pd.Timestamp,
    assets: Sequence[str],
) -> pd.Series:
    rows = audit.loc[
        audit["method"].astype(str).eq(method)
        & audit["reference_week"].eq(reference_week)
    ]
    if len(rows) != 1:
        raise ValueError(f"frozen optimizer audit lacks one {method} row at {reference_week}")
    row = rows.iloc[0]
    return pd.Series(
        {
            asset: float(row[f"expected_weekly_return_{asset}"])
            for asset in assets
        },
        dtype=float,
    )


def _active_audit_record(
    *,
    method: str,
    signal: pd.Series,
    result: ActiveOptimizationResult,
    posterior: pd.Series,
    posterior_expected: pd.Series,
    pooled_expected: pd.Series,
    benchmark: pd.Series,
    estimated_pretrade_as_of: pd.Timestamp | None,
    oracle_regime_id: str | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "method": method,
        "specification_type": (
            "oracle_infeasible_diagnostic"
            if oracle_regime_id is not None
            else "benchmark_relative_active"
        ),
        "reference_week": signal["reference_week"],
        "signal_date": signal["signal_date"],
        "execution_date": signal["start_date"],
        "regime_reference_month": signal["regime_reference_month"],
        "estimated_pretrade_as_of": estimated_pretrade_as_of,
        "oracle_regime_id": oracle_regime_id,
        "future_regime_information_used": oracle_regime_id is not None,
        "posterior_entropy": float(
            -(posterior * np.log(posterior.clip(lower=1.0e-300))).sum()
        ),
        "expected_active_weekly_return": result.expected_active_weekly_return,
        "estimated_strategy_rebalance_cost": result.estimated_transaction_cost,
        "objective_net_of_strategy_rebalance_cost": (
            result.objective_net_of_strategy_rebalance_cost
        ),
        "estimated_traded_notional": result.traded_notional,
        "estimated_half_l1_turnover": result.half_l1_turnover,
        "annualized_tracking_error": result.annualized_tracking_error,
        "annualized_total_volatility": result.annualized_total_volatility,
        "maximum_absolute_active_weight": (
            result.feasibility.maximum_absolute_active_weight
        ),
        "one_way_active_exposure": result.feasibility.one_way_active_exposure,
        "active_weight_sum": result.feasibility.active_weight_sum,
        "weight_sum": result.feasibility.weight_sum,
        "maximum_constraint_violation": (
            result.feasibility.maximum_constraint_violation
        ),
        "binding_constraints": "|".join(
            result.feasibility.binding_constraints
        ),
        "optimization_outcome": result.outcome,
        "primary_solver_success": result.primary_solver_success,
        "primary_solver_status": result.primary_solver_status,
        "primary_solver_iterations": result.primary_solver_iterations,
        "primary_solver_message": result.primary_solver_message,
        "fallback_used": result.fallback_used,
        "fallback_reason": result.fallback_reason,
    }
    for asset in result.assets:
        record[f"posterior_expected_weekly_return_{asset}"] = float(
            posterior_expected.loc[asset]
        )
        record[f"pooled_expected_weekly_return_{asset}"] = float(
            pooled_expected.loc[asset]
        )
        record[f"incremental_expected_weekly_return_{asset}"] = float(
            posterior_expected.loc[asset] - pooled_expected.loc[asset]
        )
        record[f"pooled_benchmark_weight_{asset}"] = float(benchmark.loc[asset])
        record[f"target_weight_{asset}"] = float(
            result.weight_by_asset[asset]
        )
        record[f"active_weight_{asset}"] = float(
            result.active_weight_by_asset[asset]
        )
    return record


def _build_active_targets(
    *,
    method: str,
    signals: pd.DataFrame,
    frozen_pooled_weights: pd.DataFrame,
    frozen_optimizer_audit: pd.DataFrame,
    estimator_returns: pd.DataFrame,
    regime_history: pd.DataFrame,
    prices: pd.DataFrame,
    strategy_assets: Sequence[str],
    simulation_assets: Sequence[str],
    config: Mapping[str, Any],
    estimate_cache: dict[pd.Timestamp, RegimeReturnEstimate],
    oracle_regime_by_week: Mapping[pd.Timestamp, str] | None = None,
    validate_frozen_posterior: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build active targets around pinned pooled weights on an independent path.

    Return estimates use information available through the day before each
    signal, while the active strategy maintains its own pretrade-weight path.
    ``oracle_regime_by_week`` replaces the posterior with realized one-hot
    regimes only for the explicitly infeasible oracle diagnostic.
    """

    active_config = config["active_optimizer"]
    pooled = frozen_pooled_weights.loc[
        frozen_pooled_weights["method"].astype(str).eq(POOLED_METHOD)
    ].copy()
    pooled_pivot = pooled.pivot(
        index="reference_week", columns="ticker", values="target_weight"
    )
    missing_assets = set(strategy_assets).difference(pooled_pivot.columns)
    if missing_assets:
        raise ValueError(f"frozen pooled weights omit assets: {sorted(missing_assets)}")
    previous_target: np.ndarray | None = None
    previous_entry_date: pd.Timestamp | None = None
    price_dates = pd.to_datetime(prices["date"], errors="raise")
    weight_records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    maximum_frozen_mean_mismatch = 0.0
    for _, signal in signals.sort_values("reference_week", kind="stable").iterrows():
        reference_week = pd.Timestamp(signal["reference_week"])
        signal_date = pd.Timestamp(signal["signal_date"])
        cutoff = signal_date - pd.Timedelta(days=1)
        estimate = estimate_cache.get(cutoff)
        if estimate is None:
            estimate = fit_causal_weekly_regime_return_model(
                estimator_returns,
                regime_history,
                knowledge_cutoff=cutoff,
                asset_ids=strategy_assets,
                kappa=104.0,
                minimum_observations=260,
            )
            estimate_cache[cutoff] = estimate
        pooled_moments = _pooled_mean_moments(estimate)
        oracle_regime_id = (
            None
            if oracle_regime_by_week is None
            else str(oracle_regime_by_week[reference_week])
        )
        posterior = _posterior_series(
            signal, oracle_regime_id=oracle_regime_id
        )
        posterior_moments = posterior_mixture_moments(posterior, estimate)
        if validate_frozen_posterior:
            for frozen_method, expected in (
                (BASELINE_METHOD, posterior_moments.expected_mean),
                (POOLED_METHOD, pooled_moments.expected_mean),
            ):
                frozen = _frozen_expected_returns(
                    frozen_optimizer_audit,
                    method=frozen_method,
                    reference_week=reference_week,
                    assets=strategy_assets,
                )
                mismatch = float(
                    np.max(
                        np.abs(
                            expected.loc[list(strategy_assets)].to_numpy(dtype=float)
                            - frozen.loc[list(strategy_assets)].to_numpy(dtype=float)
                        )
                    )
                )
                maximum_frozen_mean_mismatch = max(
                    maximum_frozen_mean_mismatch, mismatch
                )
                if mismatch > 1.0e-12:
                    raise ValueError(
                        f"recomputed {frozen_method} mean differs from frozen audit "
                        f"at {reference_week}: {mismatch:.6g}"
                    )
        if reference_week not in pooled_pivot.index:
            raise ValueError(f"frozen pooled target missing at {reference_week}")
        benchmark = pooled_pivot.loc[
            reference_week, list(strategy_assets)
        ].astype(float)
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
        incremental = (
            posterior_moments.expected_mean - pooled_moments.expected_mean
        ).loc[list(strategy_assets)]
        result = optimize_benchmark_relative_active(
            assets=strategy_assets,
            incremental_expected_returns=incremental.to_numpy(dtype=float),
            annualized_covariance=(
                pooled_moments.covariance.loc[
                    list(strategy_assets), list(strategy_assets)
                ].to_numpy(dtype=float)
                * float(active_config["covariance_annualization_factor"])
            ),
            benchmark_weights=benchmark.to_numpy(dtype=float),
            pretrade_weights=pretrade,
            asset_caps={
                str(asset): float(value)
                for asset, value in active_config["asset_caps"].items()
            },
            group_caps=_group_caps(config),
            transaction_costs=float(active_config["transaction_cost"]),
            tracking_error_cap=float(
                active_config["annualized_tracking_error_cap"]
            ),
            active_weight_cap=float(
                active_config["maximum_absolute_active_weight"]
            ),
            one_way_active_cap=float(
                active_config["maximum_one_way_active_exposure"]
            ),
            total_volatility_cap=float(
                active_config["annualized_total_volatility_cap"]
            ),
            feasibility_tolerance=float(
                active_config["solver"]["feasibility_tolerance"]
            ),
            maximum_iterations=int(
                active_config["solver"]["maximum_iterations"]
            ),
            solver_tolerance=float(active_config["solver"]["tolerance"]),
        )
        audit_records.append(
            _active_audit_record(
                method=method,
                signal=signal,
                result=result,
                posterior=posterior,
                posterior_expected=posterior_moments.expected_mean,
                pooled_expected=pooled_moments.expected_mean,
                benchmark=benchmark,
                estimated_pretrade_as_of=pretrade_as_of,
                oracle_regime_id=oracle_regime_id,
            )
        )
        for asset in simulation_assets:
            benchmark_weight = float(benchmark.get(asset, 0.0))
            target_weight = float(result.weight_by_asset.get(asset, 0.0))
            weight_records.append(
                {
                    "method": method,
                    "specification_type": (
                        "oracle_infeasible_diagnostic"
                        if oracle_regime_id is not None
                        else "benchmark_relative_active"
                    ),
                    "reference_week": reference_week,
                    "signal_date": signal_date,
                    "execution_date": signal["start_date"],
                    "regime_reference_month": signal["regime_reference_month"],
                    "ticker": asset,
                    "target_weight": target_weight,
                    "pooled_benchmark_weight": benchmark_weight,
                    "active_weight": target_weight - benchmark_weight,
                    "is_live_only": not bool(signal["is_complete"]),
                    "oracle_regime_id": oracle_regime_id,
                    "future_regime_information_used": oracle_regime_id is not None,
                    "optimization_outcome": result.outcome,
                }
            )
        previous_target = result.as_array()
        previous_entry_date = pd.Timestamp(signal["start_date"])
    return (
        pd.DataFrame.from_records(weight_records),
        pd.DataFrame.from_records(audit_records),
        maximum_frozen_mean_mismatch,
    )


def _simulate_methods(
    *,
    method_weights: pd.DataFrame,
    holding_returns: pd.DataFrame,
    prices: pd.DataFrame,
    simulation_assets: Sequence[str],
    cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    simulations: list[pd.DataFrame] = []
    targets: list[pd.DataFrame] = []
    for method, frame in method_weights.groupby("method", sort=False):
        target = frame.loc[
            :, ["method", "reference_week", "ticker", "target_weight"]
        ].copy()
        simulations.append(
            simulate_weekly_targets(
                target,
                holding_returns,
                assets=simulation_assets,
                transaction_costs=cost,
            )
        )
        targets.append(target)
    weekly = pd.concat(simulations, ignore_index=True)
    nav = build_weekly_daily_nav(
        pd.concat(targets, ignore_index=True),
        weekly,
        prices,
        assets=simulation_assets,
    )
    return weekly, nav


def _performance(
    weekly: pd.DataFrame,
    nav: pd.DataFrame,
    holding_returns: pd.DataFrame,
) -> pd.DataFrame:
    bil = holding_returns.set_index("reference_week")["BIL"]
    bil.index.name = "reference_month"
    return _weekly_metric_labels(
        compute_performance_metrics(
            _metric_view(weekly),
            nav,
            bil_monthly_returns=bil,
            periods_per_year=PERIODS_PER_YEAR,
        )
    )


def _uncertainty(
    weekly: pd.DataFrame,
    *,
    baseline_method: str,
    comparator_methods: Sequence[str],
    gate: Mapping[str, Any],
) -> pd.DataFrame:
    return _weekly_metric_labels(
        paired_circular_block_bootstrap(
            _metric_view(weekly),
            baseline_method=baseline_method,
            comparator_methods=tuple(comparator_methods),
            block_length=int(gate["block_length_weeks"]),
            n_resamples=int(gate["resamples"]),
            confidence_level=float(gate["confidence_level"]),
            seed=int(gate["random_seed"]),
            periods_per_year=PERIODS_PER_YEAR,
        )
    )


def _gate_result(
    uncertainty: pd.DataFrame, *, required_comparators: Sequence[str]
) -> tuple[bool, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    passed = True
    for comparator in required_comparators:
        rows = uncertainty.loc[
            uncertainty["baseline_method"].astype(str).eq(ACTIVE_METHOD)
            & uncertainty["comparator_method"].astype(str).eq(comparator)
        ]
        if len(rows) != 1:
            lower = float("nan")
            comparison_passed = False
        else:
            lower = float(
                rows.iloc[0]["annualized_mean_difference_ci_lower"]
            )
            comparison_passed = bool(np.isfinite(lower) and lower > 0.0)
        records.append(
            {
                "comparator_method": comparator,
                "annualized_mean_difference_ci_lower": lower,
                "passed": comparison_passed,
            }
        )
        passed = passed and comparison_passed
    return passed, records


def _oracle_primary_window(
    signals: pd.DataFrame, regime_history: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = signals.loc[signals["is_complete"].astype(bool)].copy()
    truth = regime_history.rename(
        columns={
            "reference_month": "regime_reference_month",
            "regime_id": "oracle_regime_id",
            "label_available_at": "oracle_label_available_at",
        }
    )
    merged = complete.merge(
        truth,
        on="regime_reference_month",
        how="left",
        validate="many_to_one",
    ).sort_values("reference_week", kind="stable").reset_index(drop=True)
    missing_positions = np.flatnonzero(merged["oracle_regime_id"].isna().to_numpy())
    stop = int(missing_positions[0]) if len(missing_positions) else len(merged)
    primary = merged.iloc[:stop].copy()
    if primary.empty:
        raise ValueError("oracle has no contiguous exact-truth window")
    gaps = primary["reference_week"].diff().dropna()
    if not gaps.eq(pd.Timedelta(days=7)).all():
        raise ValueError("oracle primary window is not weekly contiguous")
    truth_audit = primary.loc[
        :,
        [
            "reference_week",
            "signal_date",
            "regime_reference_month",
            "oracle_regime_id",
            "oracle_label_available_at",
        ],
    ].copy()
    truth_audit["future_information_used"] = True
    truth_audit["information_scope"] = (
        "eventual_exact_current_month_hard_quadrant_only"
    )
    truth_audit["available_at_signal"] = (
        pd.to_datetime(truth_audit["oracle_label_available_at"])
        <= pd.to_datetime(truth_audit["signal_date"])
    )
    return primary, truth_audit


def _oracle_regime_attribution(
    weekly: pd.DataFrame, truth_audit: pd.DataFrame
) -> pd.DataFrame:
    """Summarize oracle-minus-pooled net returns by known realized regime."""

    net = weekly.pivot(
        index="reference_week", columns="method", values="net_return"
    )
    gross = weekly.pivot(
        index="reference_week", columns="method", values="gross_return"
    )
    truth = truth_audit.set_index("reference_week")[["oracle_regime_id"]]
    frame = pd.DataFrame(
        {
            "net_difference": net[ORACLE_METHOD] - net[POOLED_METHOD],
            "gross_difference": gross[ORACLE_METHOD] - gross[POOLED_METHOD],
        }
    ).join(truth, how="inner")
    records: list[dict[str, object]] = []
    for regime_id in CANONICAL_REGIME_IDS:
        group = frame.loc[frame["oracle_regime_id"].eq(regime_id)]
        annualized_net = float(group["net_difference"].mean() * PERIODS_PER_YEAR)
        annualized_gross = float(
            group["gross_difference"].mean() * PERIODS_PER_YEAR
        )
        tracking_error = float(
            group["net_difference"].std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
        )
        records.append(
            {
                "oracle_regime_id": regime_id,
                "weeks": int(len(group)),
                "annualized_mean_net_difference_vs_pooled": annualized_net,
                "annualized_mean_gross_difference_vs_pooled": annualized_gross,
                "annualized_tracking_error_vs_pooled": tracking_error,
                "information_ratio_vs_pooled": (
                    annualized_net / tracking_error
                    if tracking_error > 0.0
                    else float("nan")
                ),
                "positive_net_week_fraction": float(
                    group["net_difference"].gt(0.0).mean()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _method_weights(
    *,
    active_weights: pd.DataFrame,
    frozen_weights: pd.DataFrame,
    methods: Sequence[str],
    end_week: pd.Timestamp,
) -> pd.DataFrame:
    frozen = frozen_weights.loc[
        frozen_weights["method"].astype(str).isin(methods)
        & frozen_weights["reference_week"].le(end_week)
        & ~frozen_weights["is_live_only"].astype(bool)
    ].copy()
    active = active_weights.loc[
        active_weights["reference_week"].le(end_week)
        & ~active_weights["is_live_only"].astype(bool)
    ].copy()
    return pd.concat([active, frozen], ignore_index=True, sort=False)


def build_m02_active_optimizer_diagnostic(
    *, project_root: Path, config_path: Path
) -> dict[str, Path]:
    """Build the pinned-posterior active test and its conditional oracle.

    The infeasible realized-regime oracle runs only if the active strategy
    fails the configured paired-bootstrap significance gate.
    """

    root = project_root.resolve()
    config_path = _project_path(root, config_path)
    config, config_bytes = _load_config(config_path)
    frozen = config["frozen_upstream"]
    manifest_path, manifest_bytes = _read_verified(
        root, frozen["manifest"], frozen["manifest_sha256"]
    )
    manifest = json.loads(manifest_bytes)
    verified: dict[str, tuple[Path, bytes]] = {}
    for key in (
        "signal_table",
        "weekly_weights",
        "optimizer_audit",
        "price_history",
        "composite_scores",
    ):
        path, raw = _read_verified(root, frozen[key], frozen[f"{key}_sha256"])
        declaration = _manifest_declaration(manifest, path=path, root=root)
        if (
            str(declaration["sha256"]) != _sha256(raw)
            or int(declaration["bytes"]) != len(raw)
        ):
            raise ValueError(f"upstream declaration mismatch for {path.relative_to(root)}")
        verified[key] = (path, raw)

    signals = pd.read_csv(verified["signal_table"][0])
    for column in (
        "reference_week",
        "signal_date",
        "regime_reference_month",
        "start_date",
        "end_date",
    ):
        signals[column] = pd.to_datetime(signals[column], errors="coerce")
    frozen_weights = pd.read_csv(verified["weekly_weights"][0], low_memory=False)
    for column in ("reference_week", "signal_date", "execution_date"):
        frozen_weights[column] = pd.to_datetime(
            frozen_weights[column], errors="raise"
        )
    frozen_audit = pd.read_csv(verified["optimizer_audit"][0])
    for column in ("reference_week", "signal_date"):
        frozen_audit[column] = pd.to_datetime(frozen_audit[column], errors="raise")
    prices = pd.read_csv(verified["price_history"][0])
    scores = pd.read_csv(verified["composite_scores"][0])
    strategy_assets = tuple(map(str, config["universe"]["strategy_assets"]))
    simulation_assets = strategy_assets + tuple(
        map(str, config["universe"]["benchmark_only_assets"])
    )
    all_holding_returns = build_weekly_open_to_open_holding_returns(
        prices, assets=simulation_assets
    )
    estimator_returns = _weekly_estimator_history(
        all_holding_returns, strategy_assets=strategy_assets
    )
    start_week = pd.Timestamp(manifest["formal_backtest_start"])
    end_week = pd.Timestamp(manifest["formal_backtest_end"])
    holding_returns = all_holding_returns.loc[
        all_holding_returns["reference_week"].between(start_week, end_week)
    ].copy()
    if len(holding_returns) != int(manifest["weekly_observations"]):
        raise ValueError("reconstructed holding-return coverage differs from frozen stage")
    regime_history = derive_m02_regime_history(scores)
    estimate_cache: dict[pd.Timestamp, RegimeReturnEstimate] = {}
    pooled_source = frozen_weights.loc[
        frozen_weights["method"].astype(str).eq(POOLED_METHOD)
    ]
    active_weights, active_audit, maximum_mean_mismatch = _build_active_targets(
        method=ACTIVE_METHOD,
        signals=signals,
        frozen_pooled_weights=pooled_source,
        frozen_optimizer_audit=frozen_audit,
        estimator_returns=estimator_returns,
        regime_history=regime_history,
        prices=prices,
        strategy_assets=strategy_assets,
        simulation_assets=simulation_assets,
        config=config,
        estimate_cache=estimate_cache,
        validate_frozen_posterior=True,
    )
    active_method_weights = _method_weights(
        active_weights=active_weights,
        frozen_weights=frozen_weights,
        methods=(POOLED_METHOD, STATIC_METHOD),
        end_week=end_week,
    )
    active_weekly, active_nav = _simulate_methods(
        method_weights=active_method_weights,
        holding_returns=holding_returns,
        prices=prices,
        simulation_assets=simulation_assets,
        cost=TRANSACTION_COST,
    )
    active_performance = _performance(
        active_weekly, active_nav, holding_returns
    )
    gate = config["significance_gate"]
    active_uncertainty = _uncertainty(
        active_weekly,
        baseline_method=ACTIVE_METHOD,
        comparator_methods=tuple(gate["required_comparators"]),
        gate=gate,
    )
    gate_passed, gate_comparisons = _gate_result(
        active_uncertainty,
        required_comparators=tuple(gate["required_comparators"]),
    )
    oracle_triggered = not gate_passed
    oracle_truth = pd.DataFrame()
    oracle_weights = pd.DataFrame()
    oracle_audit = pd.DataFrame()
    oracle_weekly = pd.DataFrame()
    oracle_nav = pd.DataFrame()
    oracle_performance = pd.DataFrame()
    oracle_uncertainty = pd.DataFrame()
    oracle_attribution = pd.DataFrame()
    oracle_primary: pd.DataFrame | None = None
    if oracle_triggered:
        oracle_primary, oracle_truth = _oracle_primary_window(
            signals, regime_history
        )
        oracle_map = {
            pd.Timestamp(row.reference_week): str(row.oracle_regime_id)
            for row in oracle_truth.itertuples(index=False)
        }
        oracle_weights, oracle_audit, _ = _build_active_targets(
            method=ORACLE_METHOD,
            signals=oracle_primary,
            frozen_pooled_weights=pooled_source,
            frozen_optimizer_audit=frozen_audit,
            estimator_returns=estimator_returns,
            regime_history=regime_history,
            prices=prices,
            strategy_assets=strategy_assets,
            simulation_assets=simulation_assets,
            config=config,
            estimate_cache=estimate_cache,
            oracle_regime_by_week=oracle_map,
            validate_frozen_posterior=False,
        )
        oracle_end = pd.Timestamp(oracle_primary["reference_week"].max())
        active_prefix = active_weights.loc[
            active_weights["reference_week"].le(oracle_end)
        ].copy()
        oracle_method_weights = pd.concat(
            [
                oracle_weights,
                _method_weights(
                    active_weights=active_prefix,
                    frozen_weights=frozen_weights,
                    methods=(POOLED_METHOD, STATIC_METHOD),
                    end_week=oracle_end,
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        oracle_holding = holding_returns.loc[
            holding_returns["reference_week"].le(oracle_end)
        ].copy()
        oracle_weekly, oracle_nav = _simulate_methods(
            method_weights=oracle_method_weights,
            holding_returns=oracle_holding,
            prices=prices,
            simulation_assets=simulation_assets,
            cost=TRANSACTION_COST,
        )
        oracle_performance = _performance(
            oracle_weekly, oracle_nav, oracle_holding
        )
        oracle_primary_uncertainty = _uncertainty(
            oracle_weekly,
            baseline_method=ORACLE_METHOD,
            comparator_methods=(
                POOLED_METHOD,
                ACTIVE_METHOD,
                STATIC_METHOD,
            ),
            gate=gate,
        )
        active_prefix_uncertainty = _uncertainty(
            oracle_weekly,
            baseline_method=ACTIVE_METHOD,
            comparator_methods=(POOLED_METHOD,),
            gate=gate,
        )
        oracle_uncertainty = pd.concat(
            [oracle_primary_uncertainty, active_prefix_uncertainty],
            ignore_index=True,
        )
        oracle_attribution = _oracle_regime_attribution(
            oracle_weekly, oracle_truth
        )

    output = config["outputs"]
    processed = _project_path(root, output["processed_dir"])
    published = _project_path(root, output["published_dir"])
    processed_paths = {
        name: processed / str(output[name])
        for name in (
            "active_weights",
            "active_returns",
            "active_daily_nav",
            "active_optimizer_audit",
            "active_performance",
            "active_uncertainty",
        )
    }
    published_paths = {
        name: published / str(output[name])
        for name in (
            "active_weights",
            "active_returns",
            "active_performance",
            "active_uncertainty",
        )
    }
    _write_csv(active_weights, processed_paths["active_weights"])
    _write_csv(active_weekly, processed_paths["active_returns"])
    _write_csv(active_nav, processed_paths["active_daily_nav"])
    _write_csv(active_audit, processed_paths["active_optimizer_audit"])
    _write_csv(active_performance, processed_paths["active_performance"])
    _write_csv(active_uncertainty, processed_paths["active_uncertainty"])
    for name in published_paths:
        source = {
            "active_weights": active_weights,
            "active_returns": active_weekly,
            "active_performance": active_performance,
            "active_uncertainty": active_uncertainty,
        }[name]
        _write_csv(source, published_paths[name])

    oracle_processed_paths: dict[str, Path] = {}
    oracle_published_paths: dict[str, Path] = {}
    if oracle_triggered:
        oracle_processed_paths = {
            name: processed / str(output[name])
            for name in (
                "oracle_truth_audit",
                "oracle_weights",
                "oracle_returns",
                "oracle_daily_nav",
                "oracle_optimizer_audit",
                "oracle_performance",
                "oracle_uncertainty",
                "oracle_regime_attribution",
            )
        }
        oracle_published_paths = {
            name: published / str(output[name])
            for name in (
                "oracle_truth_audit",
                "oracle_weights",
                "oracle_returns",
                "oracle_performance",
                "oracle_uncertainty",
                "oracle_regime_attribution",
            )
        }
        oracle_frames = {
            "oracle_truth_audit": oracle_truth,
            "oracle_weights": oracle_weights,
            "oracle_returns": oracle_weekly,
            "oracle_daily_nav": oracle_nav,
            "oracle_optimizer_audit": oracle_audit,
            "oracle_performance": oracle_performance,
            "oracle_uncertainty": oracle_uncertainty,
            "oracle_regime_attribution": oracle_attribution,
        }
        for name, path in oracle_processed_paths.items():
            _write_csv(oracle_frames[name], path)
        for name, path in oracle_published_paths.items():
            _write_csv(oracle_frames[name], path)

    summary_path = published / str(output["summary"])
    latest_active_week = pd.Timestamp(active_weights["reference_week"].max())
    latest_active = active_weights.loc[
        active_weights["reference_week"].eq(latest_active_week)
    ]
    summary = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "status": "historical_exploratory_diagnostic_not_investment_advice",
        "frozen_posterior": {
            "upstream_manifest": frozen["manifest"],
            "upstream_manifest_sha256": frozen["manifest_sha256"],
            "signal_table": frozen["signal_table"],
            "signal_table_sha256": frozen["signal_table_sha256"],
            "posterior_refit": False,
            "maximum_recomputed_expected_mean_mismatch": maximum_mean_mismatch,
        },
        "active_optimizer": config["active_optimizer"],
        "latest_active_target": {
            "reference_week": latest_active_week,
            "role": "exploratory_not_promoted",
            "target_weights": [
                {
                    "ticker": str(row.ticker),
                    "weight": float(row.target_weight),
                    "pooled_benchmark_weight": float(
                        row.pooled_benchmark_weight
                    ),
                    "active_weight": float(row.active_weight),
                }
                for row in latest_active.itertuples(index=False)
                if str(row.ticker) != "AGG"
            ],
        },
        "active_performance": active_performance.to_dict(orient="records"),
        "active_uncertainty": active_uncertainty.to_dict(orient="records"),
        "significance_gate": {
            **gate,
            "passed": gate_passed,
            "comparison_results": gate_comparisons,
            "oracle_triggered": oracle_triggered,
        },
        "oracle_diagnostic": {
            **config["oracle_diagnostic"],
            "triggered": oracle_triggered,
            "primary_start_week": (
                pd.Timestamp(oracle_primary["reference_week"].min())
                if oracle_primary is not None
                else None
            ),
            "primary_end_week": (
                pd.Timestamp(oracle_primary["reference_week"].max())
                if oracle_primary is not None
                else None
            ),
            "primary_weeks": (
                int(len(oracle_primary)) if oracle_primary is not None else 0
            ),
            "truth_months": (
                int(oracle_truth["regime_reference_month"].nunique())
                if oracle_triggered
                else 0
            ),
            "regime_runs": (
                int(
                    oracle_truth["oracle_regime_id"]
                    .ne(oracle_truth["oracle_regime_id"].shift())
                    .sum()
                )
                if oracle_triggered
                else 0
            ),
            "performance": oracle_performance.to_dict(orient="records"),
            "uncertainty": oracle_uncertainty.to_dict(orient="records"),
            "regime_attribution": oracle_attribution.to_dict(orient="records"),
        },
        "warnings": [
            "The posterior and upstream allocation artifacts are hash-frozen; "
            "this stage does not refit the Bayesian model.",
            "The active specification and gate were selected after reviewing the "
            "same historical backtest and are exploratory, not confirmatory.",
            "The oracle deliberately uses the eventually finalized current-month "
            "hard regime and is infeasible in real time.",
            "The oracle changes only the regime probability vector; return-model "
            "training, covariance, optimizer, constraints, and costs remain unchanged.",
            "The oracle primary window stops before the first missing exact regime "
            "truth and does not bridge gaps or forward-fill labels.",
        ],
    }
    _write_json(summary, summary_path)

    implementation_paths = [
        root / "src/regime_allocation/portfolio/m02_active_diagnostic.py",
        root / "src/regime_allocation/portfolio/m02_active_optimization.py",
        root / "src/regime_allocation/cli/build_m02_active_diagnostic.py",
        root / "src/regime_allocation/portfolio/m02_estimation.py",
        root / "src/regime_allocation/portfolio/m02_weekly.py",
        root / "src/regime_allocation/portfolio/m02_pipeline.py",
        root / "src/regime_allocation/portfolio/estimation.py",
        root / "src/regime_allocation/portfolio/pipeline.py",
        root / "src/regime_allocation/backtest/engine.py",
        root / "src/regime_allocation/backtest/metrics.py",
    ]
    input_paths = [
        _declaration(root, config_path),
        _declaration(root, manifest_path),
        *(_declaration(root, path) for path, _ in verified.values()),
    ]
    generated_paths = [
        *processed_paths.values(),
        *published_paths.values(),
        *oracle_processed_paths.values(),
        *oracle_published_paths.values(),
        summary_path,
    ]
    manifest_output_path = _project_path(root, output["manifest"])
    diagnostic_manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(root).as_posix(),
        "configuration_sha256": _sha256(config_bytes),
        "frozen_upstream_manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": _sha256(manifest_bytes),
        },
        "frozen_posterior_signal": {
            "path": verified["signal_table"][0].relative_to(root).as_posix(),
            "sha256": _sha256(verified["signal_table"][1]),
            "refit": False,
        },
        "inputs": list(
            {
                str(row["path"]): row
                for row in input_paths
            }.values()
        ),
        "implementation_files": [
            _declaration(root, path) for path in implementation_paths
        ],
        "active_method": ACTIVE_METHOD,
        "active_weeks": int(
            active_weekly.loc[
                active_weekly["method"].astype(str).eq(ACTIVE_METHOD),
                "reference_week",
            ].nunique()
        ),
        "significance_gate_passed": gate_passed,
        "oracle_triggered": oracle_triggered,
        "oracle_method": ORACLE_METHOD if oracle_triggered else None,
        "oracle_primary_weeks": (
            int(len(oracle_primary)) if oracle_primary is not None else 0
        ),
        "output_files": [_declaration(root, path) for path in generated_paths],
        "credential_policy": (
            "This offline diagnostic reads only hash-verified local artifacts and "
            "never reads credentials or downloads data."
        ),
    }
    _write_json(diagnostic_manifest, manifest_output_path)
    return {
        "manifest": manifest_output_path,
        "summary": summary_path,
        "active_performance": published_paths["active_performance"],
        "active_uncertainty": published_paths["active_uncertainty"],
        **(
            {
                "oracle_performance": oracle_published_paths[
                    "oracle_performance"
                ],
                "oracle_uncertainty": oracle_published_paths[
                    "oracle_uncertainty"
                ],
            }
            if oracle_triggered
            else {}
        ),
    }


__all__ = [
    "ACTIVE_METHOD",
    "ORACLE_METHOD",
    "build_m02_active_optimizer_diagnostic",
]
