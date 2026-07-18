"""Unit tests for transparent portfolio baseline targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.portfolio.baselines import (
    equal_weight_target,
    legacy_sharpe_map,
    legacy_sharpe_map_weights,
    static_60_40_target,
)


ASSETS = ("SPY", "IEF", "BIL")


def test_equal_weight_target_is_exact_and_preserves_asset_order() -> None:
    weights = equal_weight_target(("GLD", "SPY", "IEF", "BIL"))

    assert weights.index.tolist() == ["GLD", "SPY", "IEF", "BIL"]
    np.testing.assert_allclose(weights.to_numpy(), np.full(4, 0.25))
    assert weights.sum() == 1.0


@pytest.mark.parametrize("asset_ids", [(), ("SPY", "SPY"), ("SPY", "")])
def test_equal_weight_target_rejects_invalid_universes(asset_ids: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        equal_weight_target(asset_ids)


def test_static_60_40_target_names_both_legs_and_zeros_other_assets() -> None:
    weights = static_60_40_target(
        ("SPY", "IEF", "TIP", "GLD"),
        risk_asset="SPY",
        defensive_asset="IEF",
    )

    assert weights.to_dict() == {"SPY": 0.6, "IEF": 0.4, "TIP": 0.0, "GLD": 0.0}
    assert weights.sum() == 1.0


def test_static_60_40_target_is_configurable() -> None:
    weights = static_60_40_target(
        ("SPY", "AGG", "BIL"),
        risk_asset="SPY",
        defensive_asset="AGG",
        risk_weight=0.55,
    )

    assert weights.loc["SPY"] == 0.55
    assert weights.loc["AGG"] == pytest.approx(0.45)
    assert weights.loc["BIL"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"risk_asset": "SPY", "defensive_asset": "SPY"},
        {"risk_asset": "SPY", "defensive_asset": "AGG"},
        {"risk_weight": -0.01},
        {"risk_weight": 1.01},
        {"risk_weight": np.nan},
    ],
)
def test_static_60_40_target_rejects_invalid_specifications(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        static_60_40_target(("SPY", "IEF", "BIL"), **kwargs)


def _returns(dates: pd.DatetimeIndex) -> pd.DataFrame:
    position = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "SPY": 0.001 + 0.0002 * ((position % 5.0) - 2.0),
            "IEF": 0.0004 + 0.0001 * ((position % 3.0) - 1.0),
            "BIL": 0.0001 + 0.00002 * ((position % 2.0) - 0.5),
        }
    )


def _one_regime_history(
    months: pd.DatetimeIndex,
    *,
    regime_id: str = "regime_a",
    availability_lag_days: int = 35,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference_month": months,
            "regime_id": regime_id,
            "label_available_at": months + pd.to_timedelta(availability_lag_days, unit="D"),
        }
    )


def test_legacy_sharpe_map_uses_exact_score_shift_and_map_regime() -> None:
    dates = pd.bdate_range("2020-01-01", periods=70)
    daily = _returns(dates)
    months = pd.date_range("2020-01-01", "2020-04-01", freq="MS")
    history = _one_regime_history(months, availability_lag_days=10)
    signal_date = pd.Timestamp("2020-05-01")

    result = legacy_sharpe_map(
        daily,
        history,
        {"regime_b": 0.2, "regime_a": 0.8},
        signal_date=signal_date,
        asset_ids=ASSETS,
    )

    eligible = daily.loc[daily["date"] < signal_date, list(ASSETS)]
    scores = np.sqrt(252.0) * eligible.mean() / eligible.std(ddof=1)
    shifted = scores + abs(scores.min()) + 0.1
    expected = shifted / shifted.sum()

    assert result.map_regime_id == "regime_a"
    assert result.observation_count == 70
    assert not result.used_equal_weight_fallback
    assert result.fallback_reason is None
    pd.testing.assert_series_equal(
        result.annualized_sharpe_scores,
        scores.rename("annualized_sharpe"),
        check_index_type=False,
    )
    pd.testing.assert_series_equal(
        result.weights,
        expected.rename("target_weight"),
        check_index_type=False,
    )
    assert (result.weights > 0.0).all()
    assert result.weights.sum() == pytest.approx(1.0)


def test_legacy_sharpe_map_requires_sixty_common_observations() -> None:
    dates = pd.bdate_range("2020-01-01", periods=59)
    daily = _returns(dates)
    history = _one_regime_history(
        pd.date_range("2020-01-01", "2020-03-01", freq="MS"),
        availability_lag_days=5,
    )

    result = legacy_sharpe_map(
        daily,
        history,
        {"regime_a": 1.0},
        signal_date="2020-05-01",
        asset_ids=ASSETS,
    )

    assert result.observation_count == 59
    assert result.used_equal_weight_fallback
    assert result.fallback_reason == "insufficient_same_regime_observations"
    pd.testing.assert_series_equal(result.weights, equal_weight_target(ASSETS))
    assert result.annualized_sharpe_scores.isna().all()


def test_legacy_sharpe_map_accepts_exactly_sixty_observations() -> None:
    dates = pd.bdate_range("2020-01-01", periods=60)
    result = legacy_sharpe_map(
        _returns(dates),
        _one_regime_history(
            pd.date_range("2020-01-01", "2020-03-01", freq="MS"),
            availability_lag_days=5,
        ),
        {"regime_a": 1.0},
        signal_date="2020-05-01",
        asset_ids=ASSETS,
    )

    assert result.observation_count == 60
    assert not result.used_equal_weight_fallback


def test_legacy_sharpe_map_enforces_return_and_label_cutoffs_strictly() -> None:
    dates = pd.DatetimeIndex(
        ["2020-01-02", "2020-01-03", "2020-02-03", "2020-05-01"]
    )
    daily = _returns(dates)
    # January is eligible; February becomes known on (not before) the signal;
    # the May signal-date return is also strictly excluded.
    history = pd.DataFrame(
        {
            "reference_month": ["2020-01-01", "2020-02-01", "2020-05-01"],
            "regime_id": ["regime_a", "regime_a", "regime_a"],
            "label_available_at": ["2020-02-10", "2020-05-01", "2020-04-01"],
        }
    )

    result = legacy_sharpe_map(
        daily,
        history,
        {"regime_a": 1.0},
        signal_date="2020-05-01",
        asset_ids=ASSETS,
        minimum_observations=2,
    )

    assert result.observation_count == 2
    assert not result.used_equal_weight_fallback


def test_legacy_sharpe_map_uses_only_the_map_regime() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-04-30")
    daily = _returns(dates)
    history = pd.DataFrame(
        {
            "reference_month": pd.date_range("2020-01-01", "2020-04-01", freq="MS"),
            "regime_id": ["regime_a", "regime_b", "regime_a", "regime_b"],
            "label_available_at": [
                "2020-02-05",
                "2020-03-05",
                "2020-04-05",
                "2020-04-20",
            ],
        }
    )

    result = legacy_sharpe_map(
        daily,
        history,
        {"regime_a": 0.4, "regime_b": 0.6},
        signal_date="2020-05-01",
        asset_ids=ASSETS,
        minimum_observations=2,
    )

    expected_count = int(((dates.month == 2) | (dates.month == 4)).sum())
    assert result.map_regime_id == "regime_b"
    assert result.observation_count == expected_count


def test_legacy_sharpe_map_neutralizes_zero_and_nonfinite_scores() -> None:
    dates = pd.bdate_range("2020-01-01", periods=65)
    daily = _returns(dates)
    daily["BIL"] = 0.0001  # Positive mean and exactly zero volatility.
    history = _one_regime_history(
        pd.date_range("2020-01-01", "2020-04-01", freq="MS"),
        availability_lag_days=5,
    )

    result = legacy_sharpe_map(
        daily,
        history,
        {"regime_a": 1.0},
        signal_date="2020-05-01",
        asset_ids=ASSETS,
    )

    assert result.annualized_sharpe_scores.loc["BIL"] == 0.0
    assert np.isfinite(result.weights.to_numpy()).all()
    assert (result.weights > 0.0).all()
    assert result.weights.sum() == pytest.approx(1.0)


def test_legacy_sharpe_map_drops_nonfinite_rows_on_a_common_sample() -> None:
    dates = pd.bdate_range("2020-01-01", periods=62)
    daily = _returns(dates)
    daily.loc[0, "SPY"] = np.inf
    daily.loc[1, "IEF"] = np.nan
    history = _one_regime_history(
        pd.date_range("2020-01-01", "2020-04-01", freq="MS"),
        availability_lag_days=5,
    )

    result = legacy_sharpe_map(
        daily,
        history,
        {"regime_a": 1.0},
        signal_date="2020-05-01",
        asset_ids=ASSETS,
    )

    assert result.observation_count == 60
    assert not result.used_equal_weight_fallback


def test_legacy_sharpe_weights_wrapper_returns_only_weights() -> None:
    dates = pd.bdate_range("2020-01-01", periods=60)
    daily = _returns(dates)
    history = _one_regime_history(
        pd.date_range("2020-01-01", "2020-03-01", freq="MS"),
        availability_lag_days=5,
    )

    weights = legacy_sharpe_map_weights(
        daily,
        history,
        {"regime_a": 1.0},
        signal_date="2020-05-01",
        asset_ids=ASSETS,
    )

    assert isinstance(weights, pd.Series)
    assert weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "posterior",
    [{}, {"regime_a": -0.1, "regime_b": 1.1}, {"regime_a": np.nan}, {"regime_a": 0.0}],
)
def test_legacy_sharpe_map_rejects_invalid_posteriors(
    posterior: dict[str, float],
) -> None:
    dates = pd.bdate_range("2020-01-01", periods=60)
    with pytest.raises(ValueError):
        legacy_sharpe_map(
            _returns(dates),
            _one_regime_history(pd.DatetimeIndex(["2020-01-01"])),
            posterior,
            signal_date="2020-05-01",
            asset_ids=ASSETS,
        )
