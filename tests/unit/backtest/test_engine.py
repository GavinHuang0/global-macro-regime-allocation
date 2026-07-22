"""Test portfolio execution timing, drift, costs, and daily NAV reconstruction.

Small synthetic adjusted-open price panels and target weights are passed to the
production backtest engine. Assertions verify next-session holding returns,
pre-trade drift, one-way turnover, transaction-cost accounting, and the explicit
pre-trade NAV point that prevents the initial cost from disappearing from
drawdown. The module performs no I/O and exists to guard executable backtest
semantics rather than investment performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.backtest.engine import (
    build_daily_nav,
    build_open_to_open_holding_returns,
    build_weekly_daily_nav,
    build_weekly_open_to_open_holding_returns,
    drift_weights,
    simulate_monthly_targets,
    simulate_weekly_targets,
)


def test_holding_returns_use_first_session_adjusted_open() -> None:
    rows = []
    for date, values in (
        ("2020-01-02", {"A": 100.0, "B": 50.0}),
        ("2020-01-31", {"A": 105.0, "B": 51.0}),
        ("2020-02-03", {"A": 110.0, "B": 55.0}),
        ("2020-03-02", {"A": 121.0, "B": 55.0}),
    ):
        for ticker, adjusted_open in values.items():
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_open": adjusted_open,
                    "adjusted_close": adjusted_open,
                }
            )
    result = build_open_to_open_holding_returns(pd.DataFrame(rows), assets=("A", "B"))
    assert list(result["reference_month"].dt.strftime("%Y-%m")) == ["2020-01", "2020-02"]
    assert np.isclose(result.loc[0, "A"], 0.10)
    assert np.isclose(result.loc[1, "A"], 0.10)
    assert np.isclose(result.loc[0, "B"], 0.10)
    assert np.isclose(result.loc[1, "B"], 0.0)


def test_weekly_holding_returns_use_holiday_open_and_omit_incomplete_week() -> None:
    rows = []
    for date, values in (
        ("2020-01-21", {"A": 100.0, "B": 50.0}),
        ("2020-01-24", {"A": 105.0, "B": 51.0}),
        ("2020-01-27", {"A": 110.0, "B": 55.0}),
        ("2020-01-31", {"A": 115.0, "B": 54.0}),
        ("2020-02-03", {"A": 121.0, "B": 55.0}),
        ("2020-02-07", {"A": 125.0, "B": 56.0}),
    ):
        for ticker, adjusted_open in values.items():
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_open": adjusted_open,
                    "adjusted_close": adjusted_open,
                }
            )

    result = build_weekly_open_to_open_holding_returns(
        pd.DataFrame(rows), assets=("A", "B")
    )

    assert result["reference_week"].tolist() == [
        pd.Timestamp("2020-01-20"),
        pd.Timestamp("2020-01-27"),
    ]
    assert result["start_date"].tolist() == [
        pd.Timestamp("2020-01-21"),
        pd.Timestamp("2020-01-27"),
    ]
    assert result["end_date"].tolist() == [
        pd.Timestamp("2020-01-27"),
        pd.Timestamp("2020-02-03"),
    ]
    assert result["trading_sessions"].tolist() == [2, 2]
    assert np.isclose(result.loc[0, "A"], 0.10)
    assert np.isclose(result.loc[1, "A"], 0.10)
    assert np.isclose(result.loc[0, "B"], 0.10)
    assert np.isclose(result.loc[1, "B"], 0.0)


def test_drift_weights_and_transaction_cost_accounting() -> None:
    drifted = drift_weights(np.array([0.5, 0.5]), np.array([0.10, 0.0]))
    np.testing.assert_allclose(drifted, np.array([0.55, 0.50]) / 1.05)

    holdings = pd.DataFrame(
        {
            "reference_month": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "start_date": pd.to_datetime(["2020-01-02", "2020-02-03"]),
            "end_date": pd.to_datetime(["2020-02-03", "2020-03-02"]),
            "A": [0.10, 0.0],
            "B": [0.0, 0.0],
        }
    )
    targets = pd.DataFrame(
        {
            "method": ["x"] * 4,
            "reference_month": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"]
            ),
            "ticker": ["A", "B", "A", "B"],
            "target_weight": [0.5, 0.5, 0.5, 0.5],
        }
    )
    result = simulate_monthly_targets(
        targets,
        holdings,
        assets=("A", "B"),
        transaction_costs=0.0005,
    )
    # Initial purchases sum to 100%, so initial cost is exactly five basis points.
    assert np.isclose(result.loc[0, "transaction_cost_rate"], 0.0005)
    expected_first = (1 - 0.0005) * 1.05 - 1
    assert np.isclose(result.loc[0, "net_return"], expected_first)
    expected_second_cost = 0.0005 * np.abs(np.array([0.5, 0.5]) - drifted).sum()
    assert np.isclose(result.loc[1, "transaction_cost_rate"], expected_second_cost)


def test_daily_nav_records_pre_trade_capital_before_initial_cost() -> None:
    prices = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "adjusted_open": price,
                "adjusted_close": price,
            }
            for date, price in (("2020-01-02", 100.0), ("2020-02-03", 110.0))
            for ticker in ("A",)
        ]
    )
    targets = pd.DataFrame(
        {
            "method": ["x"],
            "reference_month": pd.to_datetime(["2020-01-01"]),
            "ticker": ["A"],
            "target_weight": [1.0],
        }
    )
    monthly = pd.DataFrame(
        {
            "method": ["x"],
            "reference_month": pd.to_datetime(["2020-01-01"]),
            "start_date": pd.to_datetime(["2020-01-02"]),
            "end_date": pd.to_datetime(["2020-02-03"]),
            "transaction_cost_rate": [0.0005],
        }
    )

    result = build_daily_nav(targets, monthly, prices, assets=("A",))
    start = result.loc[result["date"].eq(pd.Timestamp("2020-01-02"))]
    assert list(start["phase"]) == ["pre_trade_open", "post_trade_open", "close"]
    np.testing.assert_allclose(start["nav"], [1.0, 0.9995, 0.9995])


def test_weekly_targets_and_daily_nav_include_rebalance_costs() -> None:
    prices = pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "adjusted_open": value,
                "adjusted_close": value,
            }
            for date, values in (
                ("2020-01-06", {"A": 100.0, "B": 100.0}),
                ("2020-01-10", {"A": 108.0, "B": 100.0}),
                ("2020-01-13", {"A": 110.0, "B": 100.0}),
                ("2020-01-17", {"A": 110.0, "B": 100.0}),
                ("2020-01-21", {"A": 110.0, "B": 100.0}),
            )
            for ticker, value in values.items()
        ]
    )
    holding_returns = build_weekly_open_to_open_holding_returns(
        prices, assets=("A", "B")
    )
    targets = pd.DataFrame(
        {
            "method": ["x"] * 4,
            "reference_week": pd.to_datetime(
                ["2020-01-06", "2020-01-06", "2020-01-13", "2020-01-13"]
            ),
            "ticker": ["A", "B", "A", "B"],
            "target_weight": [0.5, 0.5, 0.5, 0.5],
        }
    )

    simulation = simulate_weekly_targets(
        targets,
        holding_returns,
        assets=("A", "B"),
        transaction_costs=0.0005,
    )

    assert simulation["reference_week"].tolist() == [
        pd.Timestamp("2020-01-06"),
        pd.Timestamp("2020-01-13"),
    ]
    assert np.isclose(simulation.loc[0, "transaction_cost_rate"], 0.0005)
    drifted = np.array([0.55, 0.50]) / 1.05
    expected_second_cost = 0.0005 * np.abs(np.array([0.5, 0.5]) - drifted).sum()
    assert np.isclose(simulation.loc[1, "transaction_cost_rate"], expected_second_cost)

    nav = build_weekly_daily_nav(targets, simulation, prices, assets=("A", "B"))
    start = nav.loc[nav["date"].eq(pd.Timestamp("2020-01-06"))]
    assert list(start["phase"]) == ["pre_trade_open", "post_trade_open", "close"]
    np.testing.assert_allclose(start["nav"], [1.0, 0.9995, 0.9995])
    terminal = nav.loc[nav["phase"].eq("terminal_open")].iloc[0]
    assert terminal["date"] == pd.Timestamp("2020-01-21")
    assert terminal["reference_week"] == pd.Timestamp("2020-01-20")
    assert np.isclose(terminal["nav"], simulation.iloc[-1]["nav_after_period"])


def test_weekly_simulation_rejects_skipped_target_weeks() -> None:
    holding_returns = pd.DataFrame(
        {
            "reference_week": pd.to_datetime(
                ["2020-01-06", "2020-01-13", "2020-01-20"]
            ),
            "start_date": pd.to_datetime(
                ["2020-01-06", "2020-01-13", "2020-01-20"]
            ),
            "end_date": pd.to_datetime(
                ["2020-01-13", "2020-01-20", "2020-01-27"]
            ),
            "A": [0.01, 0.02, 0.03],
        }
    )
    targets = pd.DataFrame(
        {
            "method": ["x", "x"],
            "reference_week": pd.to_datetime(["2020-01-06", "2020-01-20"]),
            "ticker": ["A", "A"],
            "target_weight": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="consecutive calendar weeks"):
        simulate_weekly_targets(
            targets,
            holding_returns,
            assets=("A",),
            transaction_costs=0.0005,
        )
