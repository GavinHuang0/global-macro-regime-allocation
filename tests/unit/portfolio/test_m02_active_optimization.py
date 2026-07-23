"""Contracts for Model 02's benchmark-relative active optimizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

import regime_allocation.portfolio.m02_active_optimization as active_module
from regime_allocation.portfolio.m02_active_diagnostic import (
    ACTIVE_METHOD,
    FROZEN_MANIFEST_SHA256,
    FROZEN_SIGNAL_SHA256,
    ORACLE_METHOD,
    _load_config,
    _oracle_primary_window,
)
from regime_allocation.portfolio.m02_active_optimization import (
    optimize_benchmark_relative_active,
)
from regime_allocation.portfolio.optimization import GroupCap


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/models/m02_active_optimizer_diagnostic.yaml"


def _solve(
    *,
    alpha: tuple[float, float, float] = (0.002, -0.001, -0.001),
    costs: float = 0.0,
):
    return optimize_benchmark_relative_active(
        assets=("A", "B", "C"),
        incremental_expected_returns=alpha,
        annualized_covariance=np.diag([0.04, 0.02, 0.01]),
        benchmark_weights=(0.4, 0.3, 0.3),
        pretrade_weights=(0.4, 0.3, 0.3),
        asset_caps={"A": 0.7, "B": 0.7, "C": 0.7},
        group_caps=(GroupCap("ab", ("A", "B"), 0.75),),
        transaction_costs=costs,
        tracking_error_cap=0.02,
        active_weight_cap=0.05,
        one_way_active_cap=0.10,
        total_volatility_cap=0.20,
    )


def test_config_locks_frozen_sources_active_budget_and_gate() -> None:
    config, raw = _load_config(CONFIG)

    assert raw
    assert config["frozen_upstream"]["manifest_sha256"] == FROZEN_MANIFEST_SHA256
    assert config["frozen_upstream"]["signal_table_sha256"] == FROZEN_SIGNAL_SHA256
    active = config["active_optimizer"]
    assert active["method_id"] == ACTIVE_METHOD
    assert active["annualized_tracking_error_cap"] == 0.01
    assert active["maximum_absolute_active_weight"] == 0.05
    assert active["maximum_one_way_active_exposure"] == 0.10
    assert active["annualized_total_volatility_cap"] == 0.10
    assert config["oracle_diagnostic"]["method_id"] == ORACLE_METHOD
    assert config["significance_gate"]["required_comparators"] == [
        "pooled_mean_optimizer",
        "static_60_spy_40_agg",
    ]


def test_active_solver_tilts_toward_incremental_alpha_with_all_limits_audited() -> None:
    result = _solve()

    assert result.outcome == "optimal"
    assert result.weight_by_asset["A"] > 0.4
    assert sum(result.weights) == pytest.approx(1.0)
    assert sum(result.active_weights) == pytest.approx(0.0)
    assert max(map(abs, result.active_weights)) <= 0.05 + 1.0e-10
    assert 0.5 * sum(map(abs, result.active_weights)) <= 0.10 + 1.0e-10
    assert result.annualized_tracking_error <= 0.02 + 1.0e-10
    assert result.annualized_total_volatility <= 0.20 + 1.0e-10
    assert result.feasibility.feasible
    assert result.feasibility.maximum_constraint_violation <= 1.0e-7
    assert result.expected_active_weekly_return > 0.0


def test_two_leg_cost_hurdle_can_leave_the_pooled_target_unchanged() -> None:
    result = _solve(alpha=(0.0002, -0.0001, -0.0001), costs=0.0005)

    assert result.weights == pytest.approx((0.4, 0.3, 0.3), abs=1.0e-8)
    assert result.active_weights == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-8)
    assert result.estimated_transaction_cost == pytest.approx(0.0, abs=1.0e-10)


def test_failed_primary_solver_falls_back_exactly_to_pooled_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = OptimizeResult(
        success=False,
        status=9,
        message="forced failure",
        nit=1,
        x=np.full(9, np.nan),
    )
    monkeypatch.setattr(active_module, "minimize", lambda *args, **kwargs: failed)

    result = _solve()

    assert result.outcome == "fallback_pooled_target"
    assert result.fallback_used
    assert result.weights == pytest.approx((0.4, 0.3, 0.3))
    assert result.active_weights == pytest.approx((0.0, 0.0, 0.0))


def test_oracle_primary_window_stops_before_first_missing_truth() -> None:
    weeks = pd.date_range("2020-01-06", periods=5, freq="7D")
    signals = pd.DataFrame(
        {
            "reference_week": weeks,
            "signal_date": weeks,
            "regime_reference_month": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"]
            ),
            "is_complete": True,
        }
    )
    history = pd.DataFrame(
        {
            "reference_month": [pd.Timestamp("2020-01-01")],
            "regime_id": ["growth_up_inflation_down"],
            "label_available_at": [pd.Timestamp("2020-03-01")],
        }
    )

    primary, truth = _oracle_primary_window(signals, history)

    assert primary["reference_week"].tolist() == list(weeks[:3])
    assert truth["oracle_regime_id"].eq("growth_up_inflation_down").all()
    assert truth["future_information_used"].all()
    assert not truth["available_at_signal"].any()
