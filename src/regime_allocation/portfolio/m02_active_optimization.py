"""Benchmark-relative active optimization for Model 02 diagnostics.

This module is intentionally Model-02-specific.  The shared production
optimizer is part of Model 01's frozen implementation lineage and is therefore
left byte-identical.

For a pooled benchmark target ``b`` and posterior incremental expected return
``alpha = mu_posterior - mu_pooled``, the solver chooses ``w = b + a``:

    maximize alpha' a - sum_i c_i |w_i - w_i_pretrade|

subject to the existing total-portfolio constraints plus explicit active
weight, one-way active exposure, and tracking-error limits.  The pooled
annualized covariance is used for both total risk and tracking error so the
posterior changes only the expected-return view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from regime_allocation.portfolio.optimization import GroupCap


@dataclass(frozen=True)
class ActiveFeasibilityReport:
    """Independent audit of total and benchmark-relative constraints."""

    feasible: bool
    weight_sum: float
    active_weight_sum: float
    maximum_absolute_active_weight: float
    one_way_active_exposure: float
    annualized_tracking_error: float
    annualized_total_volatility: float
    maximum_constraint_violation: float
    binding_constraints: tuple[str, ...]


@dataclass(frozen=True)
class ActiveOptimizationResult:
    """Target and audit quantities for one benchmark-relative decision."""

    assets: tuple[str, ...]
    weights: tuple[float, ...]
    benchmark_weights: tuple[float, ...]
    active_weights: tuple[float, ...]
    expected_active_weekly_return: float
    estimated_transaction_cost: float
    objective_net_of_strategy_rebalance_cost: float
    traded_notional: float
    half_l1_turnover: float
    annualized_tracking_error: float
    annualized_total_volatility: float
    outcome: str
    primary_solver_success: bool
    primary_solver_status: int | None
    primary_solver_iterations: int | None
    primary_solver_message: str
    fallback_used: bool
    fallback_reason: str | None
    feasibility: ActiveFeasibilityReport

    @property
    def weight_by_asset(self) -> dict[str, float]:
        """Return total target weights keyed by asset."""

        return dict(zip(self.assets, self.weights, strict=True))

    @property
    def active_weight_by_asset(self) -> dict[str, float]:
        """Return active target weights keyed by asset."""

        return dict(zip(self.assets, self.active_weights, strict=True))

    def as_array(self) -> np.ndarray:
        """Return a new NumPy copy of the total target."""

        return np.asarray(self.weights, dtype=float)


@dataclass(frozen=True)
class _Inputs:
    assets: tuple[str, ...]
    alpha: np.ndarray
    covariance: np.ndarray
    benchmark: np.ndarray
    pretrade: np.ndarray
    pretrade_is_invested: bool
    caps: np.ndarray
    costs: np.ndarray
    group_caps: tuple[GroupCap, ...]
    group_indices: tuple[np.ndarray, ...]
    tracking_error_cap: float
    active_weight_cap: float
    one_way_active_cap: float
    total_volatility_cap: float
    tolerance: float
    binding_tolerance: float
    maximum_iterations: int
    solver_tolerance: float


def _as_vector(
    value: float | Sequence[float] | Mapping[str, float],
    *,
    assets: tuple[str, ...],
    name: str,
) -> np.ndarray:
    if isinstance(value, Mapping):
        missing = set(assets).difference(value)
        extra = set(value).difference(assets)
        if missing or extra:
            raise ValueError(
                f"{name} keys must exactly match assets; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        result = np.asarray([value[asset] for asset in assets], dtype=float)
    elif np.isscalar(value):
        result = np.full(len(assets), float(value), dtype=float)
    else:
        result = np.asarray(value, dtype=float)
    if result.shape != (len(assets),) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of shape ({len(assets)},)")
    return result


def _validate_inputs(
    *,
    assets: Sequence[str],
    incremental_expected_returns: Sequence[float],
    annualized_covariance: Sequence[Sequence[float]],
    benchmark_weights: Sequence[float],
    pretrade_weights: Sequence[float],
    asset_caps: Sequence[float] | Mapping[str, float],
    group_caps: Sequence[GroupCap],
    transaction_costs: float | Sequence[float] | Mapping[str, float],
    tracking_error_cap: float,
    active_weight_cap: float,
    one_way_active_cap: float,
    total_volatility_cap: float,
    feasibility_tolerance: float,
    binding_tolerance: float,
    maximum_iterations: int,
    solver_tolerance: float,
) -> _Inputs:
    asset_tuple = tuple(map(str, assets))
    if not asset_tuple or len(set(asset_tuple)) != len(asset_tuple):
        raise ValueError("assets must be non-empty and unique")
    n_assets = len(asset_tuple)
    alpha = np.asarray(incremental_expected_returns, dtype=float)
    if alpha.shape != (n_assets,) or not np.isfinite(alpha).all():
        raise ValueError(
            f"incremental_expected_returns must have shape ({n_assets},)"
        )
    covariance = np.asarray(annualized_covariance, dtype=float)
    if covariance.shape != (n_assets, n_assets) or not np.isfinite(covariance).all():
        raise ValueError(
            f"annualized_covariance must have shape ({n_assets}, {n_assets})"
        )
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(covariance, covariance.T, atol=1e-10 * scale, rtol=1e-10):
        raise ValueError("annualized_covariance must be symmetric")
    covariance = (covariance + covariance.T) / 2.0
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    if minimum_eigenvalue < -1e-10 * scale:
        raise ValueError("annualized_covariance must be positive semidefinite")
    if minimum_eigenvalue < 0.0:
        covariance = covariance + np.eye(n_assets) * (-minimum_eigenvalue)

    benchmark = np.asarray(benchmark_weights, dtype=float)
    pretrade = np.asarray(pretrade_weights, dtype=float)
    for name, vector in (("benchmark_weights", benchmark), ("pretrade_weights", pretrade)):
        if vector.shape != (n_assets,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must have shape ({n_assets},)")
        if vector.min() < -feasibility_tolerance:
            raise ValueError(f"{name} cannot contain negative values")
    if abs(float(benchmark.sum()) - 1.0) > feasibility_tolerance:
        raise ValueError("benchmark_weights must sum to one")
    benchmark = benchmark / benchmark.sum()
    pretrade_is_zero = bool(np.all(np.abs(pretrade) <= feasibility_tolerance))
    pretrade_is_invested = bool(
        abs(float(pretrade.sum()) - 1.0) <= feasibility_tolerance
    )
    if not (pretrade_is_zero or pretrade_is_invested):
        raise ValueError("pretrade_weights must sum to one or be all zero")
    if pretrade_is_invested:
        pretrade = pretrade / pretrade.sum()

    caps = _as_vector(asset_caps, assets=asset_tuple, name="asset_caps")
    costs = _as_vector(
        transaction_costs, assets=asset_tuple, name="transaction_costs"
    )
    if np.any(caps < 0.0) or np.any(caps > 1.0 + feasibility_tolerance):
        raise ValueError("asset_caps must lie in [0, 1]")
    if np.any(costs < 0.0):
        raise ValueError("transaction_costs cannot be negative")
    caps = np.minimum(caps, 1.0)

    index = {asset: number for number, asset in enumerate(asset_tuple)}
    validated_groups: list[GroupCap] = []
    group_indices: list[np.ndarray] = []
    seen_names: set[str] = set()
    for group in group_caps:
        if not isinstance(group, GroupCap):
            raise TypeError("group_caps must contain GroupCap values")
        if not group.name or group.name in seen_names:
            raise ValueError("group cap names must be non-empty and unique")
        if not group.assets or len(set(group.assets)) != len(group.assets):
            raise ValueError(f"group {group.name!r} must contain unique assets")
        unknown = set(group.assets).difference(asset_tuple)
        if unknown:
            raise ValueError(f"group {group.name!r} has unknown assets: {sorted(unknown)}")
        if not np.isfinite(group.maximum) or not 0.0 <= group.maximum <= 1.0:
            raise ValueError(f"group {group.name!r} maximum must lie in [0, 1]")
        seen_names.add(group.name)
        validated_groups.append(group)
        group_indices.append(np.asarray([index[a] for a in group.assets], dtype=int))

    positive_values = {
        "tracking_error_cap": tracking_error_cap,
        "active_weight_cap": active_weight_cap,
        "one_way_active_cap": one_way_active_cap,
        "total_volatility_cap": total_volatility_cap,
        "feasibility_tolerance": feasibility_tolerance,
        "binding_tolerance": binding_tolerance,
        "solver_tolerance": solver_tolerance,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if active_weight_cap > 1.0 or one_way_active_cap > 1.0:
        raise ValueError("active exposure caps cannot exceed one")
    if int(maximum_iterations) <= 0:
        raise ValueError("maximum_iterations must be positive")

    inputs = _Inputs(
        assets=asset_tuple,
        alpha=alpha,
        covariance=covariance,
        benchmark=benchmark,
        pretrade=pretrade,
        pretrade_is_invested=pretrade_is_invested,
        caps=caps,
        costs=costs,
        group_caps=tuple(validated_groups),
        group_indices=tuple(group_indices),
        tracking_error_cap=float(tracking_error_cap),
        active_weight_cap=float(active_weight_cap),
        one_way_active_cap=float(one_way_active_cap),
        total_volatility_cap=float(total_volatility_cap),
        tolerance=float(feasibility_tolerance),
        binding_tolerance=float(binding_tolerance),
        maximum_iterations=int(maximum_iterations),
        solver_tolerance=float(solver_tolerance),
    )
    benchmark_report = _feasibility(inputs.benchmark, inputs)
    if not benchmark_report.feasible:
        raise ValueError(
            "benchmark target is infeasible under the locked active policy; "
            f"maximum violation={benchmark_report.maximum_constraint_violation:.6g}"
        )
    return inputs


def _volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    variance = float(weights @ covariance @ weights)
    if variance < 0.0 and abs(variance) <= 1e-12:
        variance = 0.0
    return float(np.sqrt(variance)) if variance >= 0.0 else float("nan")


def _feasibility(weights: np.ndarray, inputs: _Inputs) -> ActiveFeasibilityReport:
    if weights.shape != inputs.benchmark.shape or not np.isfinite(weights).all():
        return ActiveFeasibilityReport(
            feasible=False,
            weight_sum=float("nan"),
            active_weight_sum=float("nan"),
            maximum_absolute_active_weight=float("inf"),
            one_way_active_exposure=float("inf"),
            annualized_tracking_error=float("inf"),
            annualized_total_volatility=float("inf"),
            maximum_constraint_violation=float("inf"),
            binding_constraints=(),
        )
    active = weights - inputs.benchmark
    weight_sum = float(weights.sum())
    active_sum = float(active.sum())
    maximum_active = float(np.max(np.abs(active)))
    one_way_active = float(0.5 * np.abs(active).sum())
    tracking_error = _volatility(active, inputs.covariance)
    total_volatility = _volatility(weights, inputs.covariance)
    group_violations: list[float] = []
    binding: list[str] = []
    for group, indices in zip(inputs.group_caps, inputs.group_indices, strict=True):
        total = float(weights[indices].sum())
        group_violations.append(max(0.0, total - group.maximum))
        if group.maximum - total <= inputs.binding_tolerance:
            binding.append(f"group:{group.name}")
    violations = [
        abs(weight_sum - 1.0),
        abs(active_sum),
        max(0.0, float(-weights.min())),
        max(0.0, float(np.max(weights - inputs.caps))),
        max(0.0, maximum_active - inputs.active_weight_cap),
        max(0.0, one_way_active - inputs.one_way_active_cap),
        (
            float("inf")
            if not np.isfinite(tracking_error)
            else max(0.0, tracking_error - inputs.tracking_error_cap)
        ),
        (
            float("inf")
            if not np.isfinite(total_volatility)
            else max(0.0, total_volatility - inputs.total_volatility_cap)
        ),
        *group_violations,
    ]
    for asset, weight, cap, active_weight in zip(
        inputs.assets, weights, inputs.caps, active, strict=True
    ):
        if weight <= inputs.binding_tolerance:
            binding.append(f"lower:{asset}")
        if cap - weight <= inputs.binding_tolerance:
            binding.append(f"cap:{asset}")
        if inputs.active_weight_cap - active_weight <= inputs.binding_tolerance:
            binding.append(f"active_upper:{asset}")
        if inputs.active_weight_cap + active_weight <= inputs.binding_tolerance:
            binding.append(f"active_lower:{asset}")
    if inputs.one_way_active_cap - one_way_active <= inputs.binding_tolerance:
        binding.append("one_way_active")
    if inputs.tracking_error_cap - tracking_error <= inputs.binding_tolerance:
        binding.append("tracking_error")
    if inputs.total_volatility_cap - total_volatility <= inputs.binding_tolerance:
        binding.append("total_volatility")
    maximum_violation = max(violations)
    return ActiveFeasibilityReport(
        feasible=bool(maximum_violation <= inputs.tolerance),
        weight_sum=weight_sum,
        active_weight_sum=active_sum,
        maximum_absolute_active_weight=maximum_active,
        one_way_active_exposure=one_way_active,
        annualized_tracking_error=tracking_error,
        annualized_total_volatility=total_volatility,
        maximum_constraint_violation=maximum_violation,
        binding_constraints=tuple(binding),
    )


def _primary_solve(inputs: _Inputs) -> OptimizeResult:
    n_assets = len(inputs.assets)
    initial_weights = (
        inputs.pretrade.copy()
        if inputs.pretrade_is_invested
        and _feasibility(inputs.pretrade, inputs).feasible
        else inputs.benchmark.copy()
    )
    initial = np.concatenate(
        [
            initial_weights,
            np.abs(initial_weights - inputs.pretrade),
        ]
    )
    lower = np.maximum(0.0, inputs.benchmark - inputs.active_weight_cap)
    upper = np.minimum(inputs.caps, inputs.benchmark + inputs.active_weight_cap)
    bounds = [
        *((float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True)),
        *((0.0, 1.0) for _ in range(n_assets)),
    ]

    def weights(values: np.ndarray) -> np.ndarray:
        return values[:n_assets]

    constraints: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda values: float(weights(values).sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda values: float(
                inputs.tracking_error_cap**2
                - (weights(values) - inputs.benchmark)
                @ inputs.covariance
                @ (weights(values) - inputs.benchmark)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda values: float(
                inputs.total_volatility_cap**2
                - weights(values) @ inputs.covariance @ weights(values)
            ),
        },
        {
            "type": "ineq",
            "fun": lambda values: float(
                2.0 * inputs.one_way_active_cap
                - np.abs(weights(values) - inputs.benchmark).sum()
            ),
        },
        {
            "type": "ineq",
            "fun": lambda values: values[n_assets : 2 * n_assets]
            - (weights(values) - inputs.pretrade),
        },
        {
            "type": "ineq",
            "fun": lambda values: values[n_assets : 2 * n_assets]
            + (weights(values) - inputs.pretrade),
        },
    ]
    for group, indices in zip(inputs.group_caps, inputs.group_indices, strict=True):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values, idx=indices, cap=group.maximum: float(
                    cap - weights(values)[idx].sum()
                ),
            }
        )

    def objective(values: np.ndarray) -> float:
        active = weights(values) - inputs.benchmark
        trades = values[n_assets : 2 * n_assets]
        return float(-inputs.alpha @ active + inputs.costs @ trades)

    return minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": inputs.solver_tolerance,
            "maxiter": inputs.maximum_iterations,
            "disp": False,
        },
    )


def _metadata(attempt: OptimizeResult) -> tuple[bool, int | None, int | None, str]:
    status = getattr(attempt, "status", None)
    iterations = getattr(attempt, "nit", None)
    return (
        bool(getattr(attempt, "success", False)),
        int(status) if status is not None else None,
        int(iterations) if iterations is not None else None,
        str(getattr(attempt, "message", "")),
    )


def _result(
    *,
    inputs: _Inputs,
    weights: np.ndarray,
    attempt: OptimizeResult,
    outcome: str,
    fallback_reason: str | None,
) -> ActiveOptimizationResult:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    report = _feasibility(weights, inputs)
    if not report.feasible:
        raise RuntimeError(
            f"selected active target is infeasible: "
            f"{report.maximum_constraint_violation:.6g}"
        )
    active = weights - inputs.benchmark
    trades = np.abs(weights - inputs.pretrade)
    expected_active = float(inputs.alpha @ active)
    estimated_cost = float(inputs.costs @ trades)
    success, status, iterations, message = _metadata(attempt)
    return ActiveOptimizationResult(
        assets=inputs.assets,
        weights=tuple(map(float, weights)),
        benchmark_weights=tuple(map(float, inputs.benchmark)),
        active_weights=tuple(map(float, active)),
        expected_active_weekly_return=expected_active,
        estimated_transaction_cost=estimated_cost,
        objective_net_of_strategy_rebalance_cost=expected_active - estimated_cost,
        traded_notional=float(trades.sum()),
        half_l1_turnover=float(0.5 * trades.sum()),
        annualized_tracking_error=report.annualized_tracking_error,
        annualized_total_volatility=report.annualized_total_volatility,
        outcome=outcome,
        primary_solver_success=success,
        primary_solver_status=status,
        primary_solver_iterations=iterations,
        primary_solver_message=message,
        fallback_used=outcome != "optimal",
        fallback_reason=fallback_reason,
        feasibility=report,
    )


def optimize_benchmark_relative_active(
    *,
    assets: Sequence[str],
    incremental_expected_returns: Sequence[float],
    annualized_covariance: Sequence[Sequence[float]],
    benchmark_weights: Sequence[float],
    pretrade_weights: Sequence[float],
    asset_caps: Sequence[float] | Mapping[str, float],
    group_caps: Sequence[GroupCap] = (),
    transaction_costs: float | Sequence[float] | Mapping[str, float] = 0.0005,
    tracking_error_cap: float = 0.01,
    active_weight_cap: float = 0.05,
    one_way_active_cap: float = 0.10,
    total_volatility_cap: float = 0.10,
    feasibility_tolerance: float = 1.0e-7,
    binding_tolerance: float = 1.0e-5,
    maximum_iterations: int = 2_000,
    solver_tolerance: float = 1.0e-12,
) -> ActiveOptimizationResult:
    """Solve the locked benchmark-relative Model 02 active allocation.

    Incremental returns are one-week simple returns; covariance, tracking
    error, and volatility limits are annualized. Costs apply per unit of
    one-way traded notional. If SLSQP fails or returns an infeasible solution,
    the independently validated pooled benchmark is returned without relaxing
    any constraint.
    """

    inputs = _validate_inputs(
        assets=assets,
        incremental_expected_returns=incremental_expected_returns,
        annualized_covariance=annualized_covariance,
        benchmark_weights=benchmark_weights,
        pretrade_weights=pretrade_weights,
        asset_caps=asset_caps,
        group_caps=group_caps,
        transaction_costs=transaction_costs,
        tracking_error_cap=tracking_error_cap,
        active_weight_cap=active_weight_cap,
        one_way_active_cap=one_way_active_cap,
        total_volatility_cap=total_volatility_cap,
        feasibility_tolerance=feasibility_tolerance,
        binding_tolerance=binding_tolerance,
        maximum_iterations=maximum_iterations,
        solver_tolerance=solver_tolerance,
    )
    attempt = _primary_solve(inputs)
    values = np.asarray(getattr(attempt, "x", ()), dtype=float)
    n_assets = len(inputs.assets)
    weights = values[:n_assets] if values.shape == (2 * n_assets,) else None
    auxiliary_feasible = False
    if weights is not None and np.isfinite(values).all():
        trades = values[n_assets : 2 * n_assets]
        auxiliary_feasible = bool(
            np.min(trades) >= -inputs.tolerance
            and np.max(np.abs(weights - inputs.pretrade) - trades)
            <= inputs.tolerance
        )
    if (
        bool(getattr(attempt, "success", False))
        and weights is not None
        and _feasibility(weights, inputs).feasible
        and auxiliary_feasible
    ):
        return _result(
            inputs=inputs,
            weights=weights,
            attempt=attempt,
            outcome="optimal",
            fallback_reason=None,
        )
    reason = str(getattr(attempt, "message", "primary solver failed"))
    if not auxiliary_feasible:
        reason = f"{reason}; auxiliary absolute-value constraints infeasible"
    return _result(
        inputs=inputs,
        weights=inputs.benchmark,
        attempt=attempt,
        outcome="fallback_pooled_target",
        fallback_reason=reason,
    )


__all__ = [
    "ActiveFeasibilityReport",
    "ActiveOptimizationResult",
    "optimize_benchmark_relative_active",
]
