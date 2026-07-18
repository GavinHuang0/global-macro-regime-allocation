"""Test frozen allocation configuration and causal portfolio-pipeline helpers.

The repository YAML and small synthetic price/return tables are used to verify
the baseline plus one-at-a-time sensitivity grid, selection of only prices known
before each signal, the all-cash initial pre-trade state, and the pooled-mean
ablation. No full backtest or file output is produced; this module guards the
configuration and timing assumptions that connect inference to optimization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_allocation.portfolio.pipeline import (
    BASELINE_METHOD,
    _allocation_specifications,
    _known_pretrade_weights,
    _load_config,
    _pooled_mean_moments,
)
from regime_allocation.portfolio.estimation import fit_regime_return_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "models" / "m01_regime_allocation_backtest.yaml"


def test_frozen_config_builds_baseline_and_ten_oat_sensitivities() -> None:
    config, raw = _load_config(CONFIG_PATH)
    specifications = _allocation_specifications(config)

    assert raw
    assert specifications[0].method == BASELINE_METHOD
    assert specifications[0].kappa == 24.0
    assert specifications[0].volatility_cap == 0.10
    assert specifications[0].transaction_cost == 0.0005
    assert len(specifications) == 11
    assert len({item.method for item in specifications}) == 11
    kappa_sensitivities = {
        item.kappa
        for item in specifications
        if item.parameter == "regime_mean_pseudo_months"
    }
    assert kappa_sensitivities == {
        0.0,
        12.0,
        48.0,
        96.0,
    }
    assert {
        item.cap_multiplier
        for item in specifications
        if item.parameter == "concentration_cap_multiplier"
    } == {0.8, 1.2}


def test_known_pretrade_weights_use_only_close_before_signal() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-31",
                    "2020-01-31",
                    "2020-02-03",
                    "2020-02-03",
                ]
            ),
            "ticker": ["SPY", "BIL"] * 3,
            "adjusted_open": [100.0, 100.0, 110.0, 100.0, 500.0, 500.0],
            "adjusted_close": [100.0, 100.0, 120.0, 100.0, 500.0, 500.0],
        }
    )
    actual = _known_pretrade_weights(
        prices,
        assets=("SPY", "BIL"),
        previous_target=np.array([0.5, 0.5]),
        previous_entry_date=pd.Timestamp("2020-01-02"),
        signal_date=pd.Timestamp("2020-02-01"),
    )

    assert actual == pytest.approx(np.array([0.6 / 1.1, 0.5 / 1.1]))


def test_initial_pretrade_vector_is_all_zero() -> None:
    actual = _known_pretrade_weights(
        pd.DataFrame(),
        assets=("SPY", "BIL"),
        previous_target=None,
        previous_entry_date=None,
        signal_date=pd.Timestamp("2020-01-01"),
    )

    assert actual == pytest.approx(np.zeros(2))


def test_pooled_mean_ablation_removes_current_regime_expected_return() -> None:
    regimes = [
        "growth_up_inflation_up",
        "growth_down_inflation_up",
        "growth_up_inflation_down",
        "growth_down_inflation_down",
    ]
    rows = []
    for index, regime in enumerate(regimes * 2):
        rows.append(
            {
                "holding_month": pd.Timestamp("2010-01-01")
                + pd.offsets.MonthBegin(index),
                "regime_id": regime,
                "SPY": 0.01 * (index + 1),
                "BIL": 0.001,
            }
        )
    estimate = fit_regime_return_model(
        pd.DataFrame(rows),
        asset_ids=("SPY", "BIL"),
        kappa=24.0,
        minimum_observations=2,
    )
    moments = _pooled_mean_moments(estimate)

    assert moments.expected_mean.to_numpy() == pytest.approx(
        estimate.global_mean.to_numpy()
    )
    assert moments.posterior.sum() == pytest.approx(1.0)
    assert np.linalg.eigvalsh(moments.covariance.to_numpy()).min() >= -1e-12
