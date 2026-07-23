"""Test Model 02's causal weekly regime-return estimator."""

from __future__ import annotations

import pandas as pd
import pytest

from regime_allocation.portfolio.estimation import CANONICAL_REGIME_IDS
from regime_allocation.portfolio.m02_estimation import (
    fit_causal_weekly_regime_return_model,
)


A, B, C, D = CANONICAL_REGIME_IDS


def _weekly_returns() -> pd.DataFrame:
    """Return Monday-anchored observations crossing a calendar-month boundary."""
    return pd.DataFrame(
        {
            "reference_week": pd.to_datetime(
                ["2020-01-20", "2020-01-27", "2020-02-03", "2020-02-10"]
            ),
            "return_available_at": pd.to_datetime(
                ["2020-01-27", "2020-02-03", "2020-02-10", "2020-02-17"]
            ),
            "SPY": [0.01, 0.02, -0.01, 0.03],
            "IEF": [0.00, 0.01, 0.02, -0.01],
        }
    )


def _regime_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference_month": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "regime_id": [A, B],
            "label_available_at": pd.to_datetime(["2020-02-01", "2020-02-12"]),
        }
    )


def test_weekly_fit_maps_each_monday_to_its_calendar_month() -> None:
    estimate = fit_causal_weekly_regime_return_model(
        _weekly_returns(),
        _regime_history(),
        knowledge_cutoff="2020-02-17",
        asset_ids=["SPY", "IEF"],
        kappa=2.0,
        minimum_observations=2,
    )

    assert estimate.training_count == 4
    assert estimate.regime_counts.to_dict() == {A: 2, B: 2, C: 0, D: 0}
    assert estimate.first_holding_month == pd.Timestamp("2020-01-20")
    assert estimate.last_holding_month == pd.Timestamp("2020-02-10")
    january = _weekly_returns().loc[:1, ["SPY", "IEF"]].mean()
    february = _weekly_returns().loc[2:, ["SPY", "IEF"]].mean()
    pooled = _weekly_returns()[["SPY", "IEF"]].mean()
    pd.testing.assert_series_equal(
        estimate.regime_means.loc[A],
        (2.0 * january + 2.0 * pooled) / 4.0,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        estimate.regime_means.loc[B],
        (2.0 * february + 2.0 * pooled) / 4.0,
        check_names=False,
    )


def test_weekly_fit_enforces_label_and_return_cutoffs() -> None:
    label_limited = fit_causal_weekly_regime_return_model(
        _weekly_returns(),
        _regime_history(),
        knowledge_cutoff="2020-02-10",
        asset_ids=["SPY", "IEF"],
        minimum_observations=2,
    )

    assert label_limited.training_count == 2
    assert label_limited.regime_counts.to_dict() == {A: 2, B: 0, C: 0, D: 0}
    assert label_limited.last_holding_month == pd.Timestamp("2020-01-27")

    early_labels = _regime_history()
    early_labels.loc[1, "label_available_at"] = "2020-02-08"
    return_limited = fit_causal_weekly_regime_return_model(
        _weekly_returns(),
        early_labels,
        knowledge_cutoff="2020-02-10",
        asset_ids=["SPY", "IEF"],
        minimum_observations=2,
    )

    assert return_limited.training_count == 3
    assert return_limited.regime_counts.to_dict() == {A: 2, B: 1, C: 0, D: 0}
    assert return_limited.last_holding_month == pd.Timestamp("2020-02-03")


def test_weekly_fit_rejects_duplicate_or_non_monday_periods() -> None:
    duplicate = _weekly_returns()
    duplicate.loc[1, "reference_week"] = duplicate.loc[0, "reference_week"]
    with pytest.raises(ValueError, match="duplicate reference weeks"):
        fit_causal_weekly_regime_return_model(
            duplicate,
            _regime_history(),
            knowledge_cutoff="2020-02-17",
            asset_ids=["SPY", "IEF"],
            minimum_observations=2,
        )

    non_monday = _weekly_returns()
    non_monday.loc[0, "reference_week"] = "2020-01-21"
    with pytest.raises(ValueError, match="must be Mondays"):
        fit_causal_weekly_regime_return_model(
            non_monday,
            _regime_history(),
            knowledge_cutoff="2020-02-17",
            asset_ids=["SPY", "IEF"],
            minimum_observations=2,
        )


@pytest.mark.parametrize("invalid_date", ["2020-01-19", "2020-02-03"])
def test_weekly_fit_rejects_return_availability_outside_the_next_week(
    invalid_date: str,
) -> None:
    malformed = _weekly_returns()
    malformed.loc[0, "return_available_at"] = invalid_date

    with pytest.raises(ValueError, match="immediately following"):
        fit_causal_weekly_regime_return_model(
            malformed,
            _regime_history(),
            knowledge_cutoff="2020-02-17",
            asset_ids=["SPY", "IEF"],
            minimum_observations=2,
        )


def test_weekly_fit_counts_minimum_in_weekly_observations() -> None:
    with pytest.raises(ValueError, match="5 common labeled weekly observations"):
        fit_causal_weekly_regime_return_model(
            _weekly_returns(),
            _regime_history(),
            knowledge_cutoff="2020-02-17",
            asset_ids=["SPY", "IEF"],
            minimum_observations=5,
        )
