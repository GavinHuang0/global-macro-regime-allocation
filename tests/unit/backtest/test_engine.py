from __future__ import annotations

import numpy as np
import pandas as pd

from regime_allocation.backtest.engine import (
    build_open_to_open_holding_returns,
    drift_weights,
    simulate_monthly_targets,
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
