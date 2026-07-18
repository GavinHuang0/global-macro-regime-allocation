"""Test constrained long-only optimization and its ordered fallback policy.

Small expected-return, covariance, cost, and constraint inputs verify full
investment, per-asset and group caps, volatility caps, turnover penalties, and
initial formation costs. Mocked solver failures establish the hold-current,
minimum-variance, cash, and terminal-error sequence without relaxing constraints.
The tests produce no files and protect feasibility and execution semantics rather
than any particular historical result.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from regime_allocation.portfolio import optimization
from regime_allocation.portfolio.optimization import (
    GroupCap,
    OptimizationError,
    optimize_long_only,
)


def _failed_attempt(n_values: int, message: str = "forced failure") -> OptimizeResult:
    """Create a deterministic unsuccessful SciPy result for fallback tests."""
    return OptimizeResult(
        x=np.zeros(n_values),
        success=False,
        status=9,
        message=message,
        nit=1,
    )


def test_optimizer_respects_asset_group_and_volatility_caps() -> None:
    assets = ("SPY", "IEF", "BIL")
    covariance = np.diag([0.20**2, 0.08**2, 0.01**2])
    result = optimize_long_only(
        assets=assets,
        expected_monthly_returns=(0.020, 0.008, 0.001),
        annualized_covariance=covariance,
        pretrade_weights=(0.0, 0.0, 1.0),
        asset_caps={"SPY": 0.35, "IEF": 0.50, "BIL": 1.0},
        group_caps=(GroupCap("risk", ("SPY", "IEF"), 0.70),),
        transaction_costs=0.0005,
        volatility_cap=0.08,
    )

    weights = result.weight_by_asset
    assert result.audit.outcome == "optimal"
    assert result.audit.feasibility.feasible
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["SPY"] <= 0.35 + 1e-7
    assert weights["IEF"] <= 0.50 + 1e-7
    assert weights["SPY"] + weights["IEF"] <= 0.70 + 1e-7
    assert result.annualized_volatility <= 0.08 + 1e-7
    assert result.estimated_transaction_cost == pytest.approx(
        0.0005 * result.traded_notional
    )
    assert result.net_expected_monthly_return == pytest.approx(
        result.expected_monthly_return - result.estimated_transaction_cost
    )


def test_initial_formation_allows_zero_weights_and_charges_one_purchase_cost() -> None:
    result = optimize_long_only(
        assets=("BIL",),
        expected_monthly_returns=(0.002,),
        annualized_covariance=((0.01**2,),),
        pretrade_weights=(0.0,),
        asset_caps={"BIL": 1.0},
        transaction_costs=0.0005,
        volatility_cap=0.10,
    )

    assert result.weights == pytest.approx((1.0,))
    assert result.traded_notional == pytest.approx(1.0)
    assert result.half_l1_turnover == pytest.approx(0.5)
    assert result.estimated_transaction_cost == pytest.approx(0.0005)
    assert result.net_expected_monthly_return == pytest.approx(0.0015)


def test_turnover_penalty_prefers_holding_when_expected_returns_are_equal() -> None:
    result = optimize_long_only(
        assets=("SPY", "BIL"),
        expected_monthly_returns=(0.002, 0.002),
        annualized_covariance=((0.02**2, 0.0), (0.0, 0.01**2)),
        pretrade_weights=(0.4, 0.6),
        asset_caps={"SPY": 1.0, "BIL": 1.0},
        transaction_costs=0.0005,
        volatility_cap=0.10,
    )

    assert result.weights == pytest.approx((0.4, 0.6), abs=1e-7)
    assert result.traded_notional == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("pretrade_weights", (0.2, 0.2), "all-zero initial-formation"),
        (
            "annualized_covariance",
            ((0.01, 0.02), (0.02, 0.01)),
            "positive semidefinite",
        ),
        ("asset_caps", {"SPY": 1.0}, "exactly match"),
        ("transaction_costs", -0.001, "cannot be negative"),
    ],
)
def test_invalid_inputs_are_rejected(field: str, replacement: object, match: str) -> None:
    arguments: dict[str, object] = {
        "assets": ("SPY", "BIL"),
        "expected_monthly_returns": (0.01, 0.001),
        "annualized_covariance": ((0.04, 0.0), (0.0, 0.0001)),
        "pretrade_weights": (0.5, 0.5),
        "asset_caps": {"SPY": 1.0, "BIL": 1.0},
        "transaction_costs": 0.0005,
        "volatility_cap": 0.10,
    }
    arguments[field] = replacement

    with pytest.raises(ValueError, match=match):
        optimize_long_only(**arguments)  # type: ignore[arg-type]


def test_unknown_group_asset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown assets"):
        optimize_long_only(
            assets=("SPY", "BIL"),
            expected_monthly_returns=(0.01, 0.001),
            annualized_covariance=((0.04, 0.0), (0.0, 0.0001)),
            pretrade_weights=(0.5, 0.5),
            asset_caps={"SPY": 1.0, "BIL": 1.0},
            group_caps=(GroupCap("bad", ("HYG",), 0.5),),
        )


def test_primary_failure_holds_current_portfolio_before_other_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        optimization,
        "_solve_primary",
        lambda inputs: _failed_attempt(2 * len(inputs.assets)),
    )

    def should_not_run(_inputs: object) -> OptimizeResult:
        raise AssertionError("minimum-variance fallback should not run")

    monkeypatch.setattr(optimization, "_solve_minimum_variance", should_not_run)
    result = optimize_long_only(
        assets=("SPY", "BIL"),
        expected_monthly_returns=(0.01, 0.001),
        annualized_covariance=((0.04, 0.0), (0.0, 0.0001)),
        pretrade_weights=(0.2, 0.8),
        asset_caps={"SPY": 0.4, "BIL": 1.0},
        volatility_cap=0.10,
    )

    assert result.weights == pytest.approx((0.2, 0.8))
    assert result.audit.outcome == "fallback_hold_current"
    assert result.audit.fallback_used
    assert "forced failure" in str(result.audit.fallback_reason)


def test_primary_failure_uses_minimum_variance_when_current_is_infeasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        optimization,
        "_solve_primary",
        lambda inputs: _failed_attempt(2 * len(inputs.assets)),
    )
    result = optimize_long_only(
        assets=("SPY", "BIL"),
        expected_monthly_returns=(0.01, 0.001),
        annualized_covariance=((0.20**2, 0.0), (0.0, 0.01**2)),
        pretrade_weights=(1.0, 0.0),
        asset_caps={"SPY": 0.4, "BIL": 1.0},
        volatility_cap=0.10,
    )

    assert result.audit.outcome == "fallback_minimum_variance"
    assert result.audit.feasibility.feasible
    assert result.weight_by_asset["SPY"] <= 0.4 + 1e-7
    assert result.weight_by_asset["BIL"] > 0.5


def test_cash_fallback_is_used_after_minimum_variance_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        optimization,
        "_solve_primary",
        lambda inputs: _failed_attempt(2 * len(inputs.assets), "primary failed"),
    )
    monkeypatch.setattr(
        optimization,
        "_solve_minimum_variance",
        lambda inputs: _failed_attempt(len(inputs.assets), "minimum variance failed"),
    )
    result = optimize_long_only(
        assets=("SPY", "BIL"),
        expected_monthly_returns=(0.01, 0.001),
        annualized_covariance=((0.20**2, 0.0), (0.0, 0.01**2)),
        pretrade_weights=(1.0, 0.0),
        asset_caps={"SPY": 0.4, "BIL": 1.0},
        volatility_cap=0.10,
    )

    assert result.audit.outcome == "fallback_cash"
    assert result.weights == pytest.approx((0.0, 1.0))
    assert result.audit.feasibility.feasible


def test_exhausted_fallbacks_raise_instead_of_relaxing_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        optimization,
        "_solve_primary",
        lambda inputs: _failed_attempt(2 * len(inputs.assets), "primary failed"),
    )
    monkeypatch.setattr(
        optimization,
        "_solve_minimum_variance",
        lambda inputs: _failed_attempt(len(inputs.assets), "minimum variance failed"),
    )

    with pytest.raises(OptimizationError, match="no fallback was feasible"):
        optimize_long_only(
            assets=("SPY", "BIL"),
            expected_monthly_returns=(0.01, 0.001),
            annualized_covariance=((0.20**2, 0.0), (0.0, 0.20**2)),
            pretrade_weights=(1.0, 0.0),
            asset_caps={"SPY": 0.6, "BIL": 0.6},
            group_caps=(GroupCap("everything", ("SPY", "BIL"), 0.8),),
            volatility_cap=0.10,
        )


def test_result_weight_mapping_and_array_are_independent() -> None:
    result = optimize_long_only(
        assets=("SPY", "BIL"),
        expected_monthly_returns=(0.01, 0.001),
        annualized_covariance=((0.04, 0.0), (0.0, 0.0001)),
        pretrade_weights=(0.0, 1.0),
        asset_caps={"SPY": 0.3, "BIL": 1.0},
        volatility_cap=0.10,
    )

    array = result.as_array()
    array[0] = 99.0
    mapping = result.weight_by_asset
    mapping["SPY"] = 99.0
    assert result.weights[0] != 99.0
