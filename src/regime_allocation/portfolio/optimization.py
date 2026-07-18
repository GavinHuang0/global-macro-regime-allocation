"""Convex long-only portfolio optimization for Model 01.

The production objective is deliberately simple and auditable.  Expected
returns are supplied in one-month simple-return units, while the covariance
matrix and volatility ceiling are annualized::

    maximize  mu' w - sum_i c_i u_i

    subject to 1'w = 1, 0 <= w_i <= cap_i,
               u_i >= |w_i - w_i_pretrade|,
               w' Sigma_annual w <= volatility_cap**2,
               and configured group caps.

``c_i`` is a one-way cost per dollar bought or sold.  Consequently a complete
rotation from one fully invested asset to another has ``sum(abs(delta_w))=2``
and pays both a sale and a purchase cost.

SciPy's SLSQP solver is used with explicit auxiliary absolute-trade variables.
Every solver output is independently checked against the constraints.  A
failed or infeasible primary solve follows a deterministic fallback hierarchy:

1. hold the current fully invested portfolio if it is feasible;
2. solve for the feasible minimum-variance portfolio;
3. hold 100% of the configured cash proxy if that portfolio is feasible.

No fallback relaxes a cap or the volatility limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize


DEFAULT_TRANSACTION_COST = 0.0005
DEFAULT_VOLATILITY_CAP = 0.10


@dataclass(frozen=True)
class GroupCap:
    """Maximum aggregate weight for a named collection of assets."""

    name: str
    assets: tuple[str, ...]
    maximum: float


BASELINE_ASSET_CAPS: dict[str, float] = {
    "SPY": 0.35,
    "IEF": 0.50,
    "TIP": 0.40,
    "HYG": 0.25,
    "BIL": 1.00,
    "GLD": 0.25,
    "LQD": 0.40,
}

BASELINE_GROUP_CAPS: tuple[GroupCap, ...] = (
    GroupCap("growth_and_high_yield", ("SPY", "HYG"), 0.50),
    GroupCap("corporate_credit", ("HYG", "LQD"), 0.50),
    GroupCap("rate_sensitive", ("IEF", "TIP", "LQD"), 0.75),
)


class OptimizationError(RuntimeError):
    """Raised when inputs are invalid or no feasible portfolio can be produced."""


@dataclass(frozen=True)
class FeasibilityReport:
    """Independent constraint audit for a candidate weight vector."""

    feasible: bool
    weight_sum: float
    full_investment_error: float
    lower_bound_violation: float
    individual_cap_violation: float
    group_cap_violations: tuple[tuple[str, float], ...]
    annualized_volatility: float
    volatility_cap_violation: float
    maximum_violation: float
    binding_constraints: tuple[str, ...]


@dataclass(frozen=True)
class OptimizationAudit:
    """Solver and fallback diagnostics for one rebalance decision."""

    outcome: str
    primary_solver_success: bool
    primary_solver_status: int | None
    primary_solver_message: str
    primary_solver_iterations: int | None
    fallback_solver_success: bool | None
    fallback_solver_status: int | None
    fallback_solver_message: str | None
    fallback_solver_iterations: int | None
    fallback_used: bool
    fallback_reason: str | None
    feasibility: FeasibilityReport


@dataclass(frozen=True)
class OptimizationResult:
    """Feasible target portfolio and quantities needed by the backtest audit."""

    assets: tuple[str, ...]
    weights: tuple[float, ...]
    expected_monthly_return: float
    estimated_transaction_cost: float
    net_expected_monthly_return: float
    traded_notional: float
    half_l1_turnover: float
    annualized_volatility: float
    audit: OptimizationAudit

    @property
    def weight_by_asset(self) -> dict[str, float]:
        """Return target weights keyed by asset without exposing mutable state."""

        return dict(zip(self.assets, self.weights, strict=True))

    def as_array(self) -> np.ndarray:
        """Return a new NumPy copy of the target weights."""

        return np.asarray(self.weights, dtype=float)


@dataclass(frozen=True)
class _ValidatedInputs:
    assets: tuple[str, ...]
    expected_returns: np.ndarray
    covariance: np.ndarray
    pretrade: np.ndarray
    pretrade_is_fully_invested: bool
    caps: np.ndarray
    costs: np.ndarray
    group_caps: tuple[GroupCap, ...]
    group_indices: tuple[np.ndarray, ...]
    volatility_cap: float
    tolerance: float
    binding_tolerance: float
    cash_index: int


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
        if result.shape != (len(assets),):
            raise ValueError(f"{name} must have shape ({len(assets)},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_inputs(
    *,
    assets: Sequence[str],
    expected_monthly_returns: Sequence[float],
    annualized_covariance: Sequence[Sequence[float]],
    pretrade_weights: Sequence[float],
    asset_caps: Sequence[float] | Mapping[str, float],
    group_caps: Sequence[GroupCap],
    transaction_costs: float | Sequence[float] | Mapping[str, float],
    volatility_cap: float,
    cash_asset: str,
    feasibility_tolerance: float,
    binding_tolerance: float,
) -> _ValidatedInputs:
    asset_tuple = tuple(assets)
    if not asset_tuple:
        raise ValueError("assets cannot be empty")
    if any(not isinstance(asset, str) or not asset for asset in asset_tuple):
        raise ValueError("every asset must be a non-empty string")
    if len(set(asset_tuple)) != len(asset_tuple):
        raise ValueError("assets must be unique")
    if cash_asset not in asset_tuple:
        raise ValueError(f"cash_asset {cash_asset!r} is not present in assets")

    n_assets = len(asset_tuple)
    expected = np.asarray(expected_monthly_returns, dtype=float)
    if expected.shape != (n_assets,) or not np.isfinite(expected).all():
        raise ValueError(
            f"expected_monthly_returns must be a finite vector of shape ({n_assets},)"
        )

    covariance = np.asarray(annualized_covariance, dtype=float)
    if covariance.shape != (n_assets, n_assets):
        raise ValueError(
            f"annualized_covariance must have shape ({n_assets}, {n_assets})"
        )
    if not np.isfinite(covariance).all():
        raise ValueError("annualized_covariance must contain only finite values")
    scale = max(1.0, float(np.max(np.abs(covariance))))
    symmetry_tolerance = 1e-10 * scale
    if not np.allclose(covariance, covariance.T, atol=symmetry_tolerance, rtol=1e-10):
        raise ValueError("annualized_covariance must be symmetric")
    covariance = (covariance + covariance.T) / 2.0
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    eigenvalue_tolerance = 1e-10 * scale
    if minimum_eigenvalue < -eigenvalue_tolerance:
        raise ValueError(
            "annualized_covariance must be positive semidefinite; "
            f"minimum eigenvalue={minimum_eigenvalue:.6g}"
        )
    if minimum_eigenvalue < 0.0:
        covariance = covariance + np.eye(n_assets) * (-minimum_eigenvalue)

    pretrade = np.asarray(pretrade_weights, dtype=float)
    if pretrade.shape != (n_assets,) or not np.isfinite(pretrade).all():
        raise ValueError(f"pretrade_weights must be a finite vector of shape ({n_assets},)")
    if np.min(pretrade) < -feasibility_tolerance:
        raise ValueError("pretrade_weights cannot contain negative values")
    pretrade = np.where(np.abs(pretrade) <= feasibility_tolerance, 0.0, pretrade)
    pretrade_sum = float(pretrade.sum())
    pretrade_is_zero = bool(np.all(np.abs(pretrade) <= feasibility_tolerance))
    pretrade_is_invested = bool(abs(pretrade_sum - 1.0) <= feasibility_tolerance)
    if not (pretrade_is_zero or pretrade_is_invested):
        raise ValueError(
            "pretrade_weights must either sum to one or be the all-zero initial-formation vector"
        )
    if pretrade_is_invested:
        pretrade = pretrade / pretrade_sum

    caps = _as_vector(asset_caps, assets=asset_tuple, name="asset_caps")
    if np.any(caps < 0.0) or np.any(caps > 1.0 + feasibility_tolerance):
        raise ValueError("asset_caps must lie in [0, 1]")
    caps = np.minimum(caps, 1.0)

    costs = _as_vector(transaction_costs, assets=asset_tuple, name="transaction_costs")
    if np.any(costs < 0.0):
        raise ValueError("transaction_costs cannot be negative")

    validated_groups: list[GroupCap] = []
    group_indices: list[np.ndarray] = []
    seen_group_names: set[str] = set()
    index_by_asset = {asset: index for index, asset in enumerate(asset_tuple)}
    for group in group_caps:
        if not isinstance(group, GroupCap):
            raise TypeError("group_caps must contain GroupCap values")
        if not group.name or group.name in seen_group_names:
            raise ValueError("group cap names must be non-empty and unique")
        if not group.assets or len(set(group.assets)) != len(group.assets):
            raise ValueError(f"group {group.name!r} must contain unique assets")
        unknown = set(group.assets).difference(asset_tuple)
        if unknown:
            raise ValueError(f"group {group.name!r} contains unknown assets: {sorted(unknown)}")
        if not np.isfinite(group.maximum) or not 0.0 <= group.maximum <= 1.0:
            raise ValueError(f"group {group.name!r} maximum must lie in [0, 1]")
        seen_group_names.add(group.name)
        validated_groups.append(group)
        group_indices.append(
            np.asarray([index_by_asset[asset] for asset in group.assets], dtype=int)
        )

    if not np.isfinite(volatility_cap) or volatility_cap <= 0.0:
        raise ValueError("volatility_cap must be finite and positive")
    if not np.isfinite(feasibility_tolerance) or feasibility_tolerance <= 0.0:
        raise ValueError("feasibility_tolerance must be finite and positive")
    if not np.isfinite(binding_tolerance) or binding_tolerance <= 0.0:
        raise ValueError("binding_tolerance must be finite and positive")

    return _ValidatedInputs(
        assets=asset_tuple,
        expected_returns=expected,
        covariance=covariance,
        pretrade=pretrade,
        pretrade_is_fully_invested=pretrade_is_invested,
        caps=caps,
        costs=costs,
        group_caps=tuple(validated_groups),
        group_indices=tuple(group_indices),
        volatility_cap=float(volatility_cap),
        tolerance=float(feasibility_tolerance),
        binding_tolerance=float(binding_tolerance),
        cash_index=index_by_asset[cash_asset],
    )


def _annualized_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    variance = float(weights @ covariance @ weights)
    if variance < 0.0 and abs(variance) <= 1e-12:
        variance = 0.0
    if variance < 0.0:
        return float("nan")
    return float(np.sqrt(variance))


def _feasibility_report(
    weights: np.ndarray,
    inputs: _ValidatedInputs,
) -> FeasibilityReport:
    if weights.shape != (len(inputs.assets),) or not np.isfinite(weights).all():
        return FeasibilityReport(
            feasible=False,
            weight_sum=float("nan"),
            full_investment_error=float("inf"),
            lower_bound_violation=float("inf"),
            individual_cap_violation=float("inf"),
            group_cap_violations=tuple(
                (group.name, float("inf")) for group in inputs.group_caps
            ),
            annualized_volatility=float("nan"),
            volatility_cap_violation=float("inf"),
            maximum_violation=float("inf"),
            binding_constraints=(),
        )

    weight_sum = float(weights.sum())
    full_investment_error = abs(weight_sum - 1.0)
    lower_violation = max(0.0, float(-weights.min()))
    individual_violation = max(0.0, float(np.max(weights - inputs.caps)))
    group_violations: list[tuple[str, float]] = []
    binding: list[str] = []
    for group, indices in zip(inputs.group_caps, inputs.group_indices, strict=True):
        total = float(weights[indices].sum())
        violation = max(0.0, total - group.maximum)
        group_violations.append((group.name, violation))
        if group.maximum - total <= inputs.binding_tolerance:
            binding.append(f"group:{group.name}")

    volatility = _annualized_volatility(weights, inputs.covariance)
    volatility_violation = (
        float("inf")
        if not np.isfinite(volatility)
        else max(0.0, volatility - inputs.volatility_cap)
    )
    violations = [
        full_investment_error,
        lower_violation,
        individual_violation,
        volatility_violation,
        *(violation for _, violation in group_violations),
    ]
    maximum_violation = max(violations)

    for asset, weight, cap in zip(inputs.assets, weights, inputs.caps, strict=True):
        if weight <= inputs.binding_tolerance:
            binding.append(f"lower:{asset}")
        if cap - weight <= inputs.binding_tolerance:
            binding.append(f"cap:{asset}")
    if inputs.volatility_cap - volatility <= inputs.binding_tolerance:
        binding.append("volatility")

    return FeasibilityReport(
        feasible=bool(maximum_violation <= inputs.tolerance),
        weight_sum=weight_sum,
        full_investment_error=full_investment_error,
        lower_bound_violation=lower_violation,
        individual_cap_violation=individual_violation,
        group_cap_violations=tuple(group_violations),
        annualized_volatility=volatility,
        volatility_cap_violation=volatility_violation,
        maximum_violation=maximum_violation,
        binding_constraints=tuple(binding),
    )


def _cash_weights(inputs: _ValidatedInputs) -> np.ndarray:
    weights = np.zeros(len(inputs.assets), dtype=float)
    weights[inputs.cash_index] = 1.0
    return weights


def _capped_start(inputs: _ValidatedInputs) -> np.ndarray:
    """Create a deterministic bounded start even when it is not fully feasible."""

    n_assets = len(inputs.assets)
    weights = np.minimum(inputs.caps, 1.0 / n_assets)
    remaining = 1.0 - float(weights.sum())
    for index in range(n_assets):
        if remaining <= inputs.tolerance:
            break
        addition = min(remaining, inputs.caps[index] - weights[index])
        if addition > 0.0:
            weights[index] += addition
            remaining -= addition
    if weights.sum() > 0.0 and remaining > inputs.tolerance:
        weights /= weights.sum()
    return weights


def _initial_weights(inputs: _ValidatedInputs) -> np.ndarray:
    if inputs.pretrade_is_fully_invested:
        report = _feasibility_report(inputs.pretrade, inputs)
        if report.feasible:
            return inputs.pretrade.copy()
    cash = _cash_weights(inputs)
    if _feasibility_report(cash, inputs).feasible:
        return cash
    return _capped_start(inputs)


def _constraints(inputs: _ValidatedInputs, *, with_trade_variables: bool):
    n_assets = len(inputs.assets)

    def weights_of(values: np.ndarray) -> np.ndarray:
        """Extract portfolio weights from a solver decision vector."""
        return values[:n_assets] if with_trade_variables else values

    constraints: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda values: float(weights_of(values).sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda values: float(
                inputs.volatility_cap**2
                - weights_of(values) @ inputs.covariance @ weights_of(values)
            ),
        },
    ]
    for indices, group in zip(inputs.group_indices, inputs.group_caps, strict=True):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values, idx=indices, cap=group.maximum: float(
                    cap - weights_of(values)[idx].sum()
                ),
            }
        )
    if with_trade_variables:
        constraints.extend(
            [
                {
                    "type": "ineq",
                    "fun": lambda values: values[n_assets:]
                    - (values[:n_assets] - inputs.pretrade),
                },
                {
                    "type": "ineq",
                    "fun": lambda values: values[n_assets:]
                    + (values[:n_assets] - inputs.pretrade),
                },
            ]
        )
    return constraints


def _solve_primary(inputs: _ValidatedInputs) -> OptimizeResult:
    n_assets = len(inputs.assets)
    initial_weights = _initial_weights(inputs)
    initial_trades = np.abs(initial_weights - inputs.pretrade)
    initial = np.concatenate([initial_weights, initial_trades])
    bounds = [
        *((0.0, float(cap)) for cap in inputs.caps),
        *((0.0, 1.0) for _ in range(n_assets)),
    ]

    def objective(values: np.ndarray) -> float:
        """Return negative net expected return for SciPy minimization."""
        weights = values[:n_assets]
        trades = values[n_assets:]
        return float(-inputs.expected_returns @ weights + inputs.costs @ trades)

    return minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=_constraints(inputs, with_trade_variables=True),
        options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
    )


def _solve_minimum_variance(inputs: _ValidatedInputs) -> OptimizeResult:
    initial = _initial_weights(inputs)
    bounds = [(0.0, float(cap)) for cap in inputs.caps]

    def objective(weights: np.ndarray) -> float:
        """Return annualized portfolio variance for fallback minimization."""
        return float(weights @ inputs.covariance @ weights)

    return minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=_constraints(inputs, with_trade_variables=False),
        options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
    )


def _solver_weights(
    attempt: OptimizeResult,
    *,
    n_assets: int,
    with_trade_variables: bool,
) -> np.ndarray | None:
    values = np.asarray(getattr(attempt, "x", ()), dtype=float)
    expected_length = 2 * n_assets if with_trade_variables else n_assets
    if values.shape != (expected_length,) or not np.isfinite(values).all():
        return None
    return values[:n_assets].copy()


def _primary_auxiliaries_feasible(
    attempt: OptimizeResult,
    *,
    inputs: _ValidatedInputs,
) -> bool:
    n_assets = len(inputs.assets)
    values = np.asarray(getattr(attempt, "x", ()), dtype=float)
    if values.shape != (2 * n_assets,) or not np.isfinite(values).all():
        return False
    trades = values[n_assets:]
    required = np.abs(values[:n_assets] - inputs.pretrade)
    return bool(
        np.min(trades) >= -inputs.tolerance
        and np.max(required - trades) <= inputs.tolerance
    )


def _attempt_metadata(attempt: OptimizeResult) -> tuple[bool, int | None, str, int | None]:
    success = bool(getattr(attempt, "success", False))
    raw_status = getattr(attempt, "status", None)
    status = int(raw_status) if raw_status is not None else None
    message = str(getattr(attempt, "message", ""))
    raw_iterations = getattr(attempt, "nit", None)
    iterations = int(raw_iterations) if raw_iterations is not None else None
    return success, status, message, iterations


def _build_result(
    *,
    inputs: _ValidatedInputs,
    weights: np.ndarray,
    outcome: str,
    primary_attempt: OptimizeResult,
    fallback_reason: str | None,
    fallback_attempt: OptimizeResult | None = None,
) -> OptimizationResult:
    report = _feasibility_report(weights, inputs)
    if not report.feasible:
        raise OptimizationError(
            f"internal error: selected {outcome!r} portfolio is infeasible; "
            f"maximum violation={report.maximum_violation:.6g}"
        )
    weights = np.where(np.abs(weights) <= inputs.tolerance, 0.0, weights)
    weights = weights / weights.sum()
    normalized_report = _feasibility_report(weights, inputs)
    if not normalized_report.feasible:
        raise OptimizationError(
            f"internal error: normalized {outcome!r} portfolio is infeasible; "
            f"maximum violation={normalized_report.maximum_violation:.6g}"
        )
    trades = np.abs(weights - inputs.pretrade)
    expected = float(inputs.expected_returns @ weights)
    estimated_cost = float(inputs.costs @ trades)
    primary_success, primary_status, primary_message, primary_iterations = (
        _attempt_metadata(primary_attempt)
    )
    if fallback_attempt is None:
        fallback_success = None
        fallback_status = None
        fallback_message = None
        fallback_iterations = None
    else:
        (
            fallback_success,
            fallback_status,
            fallback_message,
            fallback_iterations,
        ) = _attempt_metadata(fallback_attempt)
    audit = OptimizationAudit(
        outcome=outcome,
        primary_solver_success=primary_success,
        primary_solver_status=primary_status,
        primary_solver_message=primary_message,
        primary_solver_iterations=primary_iterations,
        fallback_solver_success=fallback_success,
        fallback_solver_status=fallback_status,
        fallback_solver_message=fallback_message,
        fallback_solver_iterations=fallback_iterations,
        fallback_used=outcome != "optimal",
        fallback_reason=fallback_reason,
        feasibility=_feasibility_report(weights, inputs),
    )
    return OptimizationResult(
        assets=inputs.assets,
        weights=tuple(float(value) for value in weights),
        expected_monthly_return=expected,
        estimated_transaction_cost=estimated_cost,
        net_expected_monthly_return=expected - estimated_cost,
        traded_notional=float(trades.sum()),
        half_l1_turnover=float(0.5 * trades.sum()),
        annualized_volatility=audit.feasibility.annualized_volatility,
        audit=audit,
    )


def optimize_long_only(
    *,
    assets: Sequence[str],
    expected_monthly_returns: Sequence[float],
    annualized_covariance: Sequence[Sequence[float]],
    pretrade_weights: Sequence[float],
    asset_caps: Sequence[float] | Mapping[str, float],
    group_caps: Sequence[GroupCap] = (),
    transaction_costs: float | Sequence[float] | Mapping[str, float] = (
        DEFAULT_TRANSACTION_COST
    ),
    volatility_cap: float = DEFAULT_VOLATILITY_CAP,
    cash_asset: str = "BIL",
    feasibility_tolerance: float = 1e-7,
    binding_tolerance: float = 1e-5,
) -> OptimizationResult:
    """Solve the turnover-aware long-only allocation problem.

    ``pretrade_weights`` may be a fully invested portfolio or the all-zero
    vector used for initial formation.  The zero vector makes the initial
    purchase cost explicit: with uniform five-basis-point costs, every fully
    invested target pays exactly five basis points.

    The returned portfolio is always independently verified.  Invalid model
    inputs raise ``ValueError``; exhaustion of all feasible fallbacks raises
    :class:`OptimizationError`.
    """

    inputs = _validate_inputs(
        assets=assets,
        expected_monthly_returns=expected_monthly_returns,
        annualized_covariance=annualized_covariance,
        pretrade_weights=pretrade_weights,
        asset_caps=asset_caps,
        group_caps=group_caps,
        transaction_costs=transaction_costs,
        volatility_cap=volatility_cap,
        cash_asset=cash_asset,
        feasibility_tolerance=feasibility_tolerance,
        binding_tolerance=binding_tolerance,
    )

    primary = _solve_primary(inputs)
    primary_weights = _solver_weights(
        primary, n_assets=len(inputs.assets), with_trade_variables=True
    )
    primary_report = (
        _feasibility_report(primary_weights, inputs)
        if primary_weights is not None
        else None
    )
    primary_success = bool(getattr(primary, "success", False))
    auxiliaries_feasible = _primary_auxiliaries_feasible(primary, inputs=inputs)
    if (
        primary_success
        and primary_weights is not None
        and primary_report is not None
        and primary_report.feasible
        and auxiliaries_feasible
    ):
        return _build_result(
            inputs=inputs,
            weights=primary_weights,
            outcome="optimal",
            primary_attempt=primary,
            fallback_reason=None,
        )

    failure_parts = [str(getattr(primary, "message", "primary solver failed"))]
    if primary_report is not None and not primary_report.feasible:
        failure_parts.append(
            f"maximum constraint violation={primary_report.maximum_violation:.6g}"
        )
    if not auxiliaries_feasible:
        failure_parts.append("absolute-trade auxiliaries were infeasible")
    fallback_reason = "; ".join(part for part in failure_parts if part)

    if inputs.pretrade_is_fully_invested:
        hold_report = _feasibility_report(inputs.pretrade, inputs)
        if hold_report.feasible:
            return _build_result(
                inputs=inputs,
                weights=inputs.pretrade.copy(),
                outcome="fallback_hold_current",
                primary_attempt=primary,
                fallback_reason=fallback_reason,
            )

    minimum_variance = _solve_minimum_variance(inputs)
    minimum_variance_weights = _solver_weights(
        minimum_variance,
        n_assets=len(inputs.assets),
        with_trade_variables=False,
    )
    if bool(getattr(minimum_variance, "success", False)) and minimum_variance_weights is not None:
        minimum_variance_report = _feasibility_report(minimum_variance_weights, inputs)
        if minimum_variance_report.feasible:
            return _build_result(
                inputs=inputs,
                weights=minimum_variance_weights,
                outcome="fallback_minimum_variance",
                primary_attempt=primary,
                fallback_reason=fallback_reason,
                fallback_attempt=minimum_variance,
            )

    cash = _cash_weights(inputs)
    if _feasibility_report(cash, inputs).feasible:
        return _build_result(
            inputs=inputs,
            weights=cash,
            outcome="fallback_cash",
            primary_attempt=primary,
            fallback_reason=fallback_reason,
            fallback_attempt=minimum_variance,
        )

    minimum_variance_message = str(getattr(minimum_variance, "message", ""))
    raise OptimizationError(
        "primary optimization failed and no fallback was feasible; "
        f"primary={fallback_reason!r}; minimum_variance={minimum_variance_message!r}; "
        f"100% {cash_asset} violates one or more configured constraints"
    )
