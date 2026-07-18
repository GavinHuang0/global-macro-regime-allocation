"""Test the deterministic growth and inflation composite specification.

Synthetic monthly component levels exercise the frozen transformation registry,
strictly lagged expanding z-scores, equal weighting, 60-observation warm-up, and
three-month full-window smoothing. Tests also establish exact tie and missing-data
behavior and verify that future observations cannot change earlier features.
There is no I/O; the module locks the mathematical target definition used by all
later Model 01 stages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.features.composites import (
    COMPONENT_TRANSFORMS,
    build_composite_features,
    lagged_expanding_zscore,
    transform_release_levels,
)


GROWTH_COMPONENTS = (
    "payrolls",
    "industrial_production",
    "consumer_activity",
    "unemployment_rate",
)
INFLATION_COMPONENTS = (
    "core_cpi",
    "core_pce",
    "producer_prices",
    "average_hourly_earnings",
)
ALL_COMPONENTS = GROWTH_COMPONENTS + INFLATION_COMPONENTS
LOG_DIFFERENCE_COMPONENTS = (
    "industrial_production",
    "consumer_activity",
    "core_cpi",
    "core_pce",
    "producer_prices",
    "average_hourly_earnings",
)


def _synthetic_release_levels(periods: int = 80) -> pd.DataFrame:
    """Build positive monthly levels whose intended changes vary over time."""
    dates = pd.date_range("1990-01-01", periods=periods, freq="MS")
    step_index = np.arange(periods - 1, dtype=float)

    changes = {
        component: (
            0.15 * np.sin(step_index / (2.0 + position / 3.0))
            + 0.07 * np.cos(step_index / (3.0 + position / 4.0))
            + 0.003 * step_index
        )
        for position, component in enumerate(ALL_COMPONENTS, start=1)
    }

    levels = pd.DataFrame(index=dates)
    levels["payrolls"] = np.r_[
        100_000.0,
        100_000.0 + np.cumsum(100.0 * changes["payrolls"]),
    ]
    levels["unemployment_rate"] = np.r_[
        8.0,
        8.0 - np.cumsum(0.02 * changes["unemployment_rate"]),
    ]

    for position, component in enumerate(LOG_DIFFERENCE_COMPONENTS, start=1):
        initial_level = 80.0 + 5.0 * position
        levels[component] = np.r_[
            initial_level,
            initial_level * np.exp(np.cumsum(changes[component]) / 100.0),
        ]

    return levels.loc[:, list(ALL_COMPONENTS)]


def test_transform_registry_covers_exactly_the_required_components() -> None:
    assert set(COMPONENT_TRANSFORMS) == set(ALL_COMPONENTS)


def test_component_transforms_use_documented_first_differences_and_log_changes() -> None:
    dates = pd.date_range("2000-01-01", periods=3, freq="MS")
    levels = pd.DataFrame(
        {
            "payrolls": [100_000.0, 100_125.0, 100_200.0],
            "industrial_production": [100.0, 102.0, 101.0],
            "consumer_activity": [200.0, 206.0, 203.0],
            "unemployment_rate": [5.0, 4.8, 4.9],
            "core_cpi": [250.0, 251.0, 252.5],
            "core_pce": [110.0, 110.4, 110.9],
            "producer_prices": [180.0, 181.8, 181.0],
            "average_hourly_earnings": [25.0, 25.2, 25.3],
        },
        index=dates,
    )

    transformed = transform_release_levels(levels)

    assert transformed.loc[dates[1], "payrolls"] == pytest.approx(125.0)
    assert transformed.loc[dates[2], "payrolls"] == pytest.approx(75.0)
    assert transformed.loc[dates[1], "unemployment_rate"] == pytest.approx(0.2)
    assert transformed.loc[dates[2], "unemployment_rate"] == pytest.approx(-0.1)

    for component in LOG_DIFFERENCE_COMPONENTS:
        expected = 100.0 * np.log(levels[component].iloc[1] / levels[component].iloc[0])
        assert transformed[component].iloc[1] == pytest.approx(expected)

    assert transformed.iloc[0].isna().all()


def test_lagged_expanding_zscore_uses_only_prior_60_values_and_sample_std() -> None:
    values = pd.Series(np.arange(61, dtype=float), name="component")

    actual = lagged_expanding_zscore(values)

    assert actual.iloc[:60].isna().all()
    expected = (values.iloc[60] - values.iloc[:60].mean()) / values.iloc[:60].std(ddof=1)
    assert actual.iloc[60] == pytest.approx(expected)


def test_lagged_expanding_zscore_is_invariant_to_future_values() -> None:
    values = pd.Series(np.sin(np.arange(100, dtype=float) / 4.0))
    shocked_future = values.copy()
    shocked_future.iloc[75:] = shocked_future.iloc[75:] + 1_000_000.0

    baseline = lagged_expanding_zscore(values, min_periods=10, ddof=1)
    shocked = lagged_expanding_zscore(shocked_future, min_periods=10, ddof=1)

    pd.testing.assert_series_equal(baseline.iloc[:75], shocked.iloc[:75])


def test_default_minimum_history_delays_raw_and_smoothed_composites() -> None:
    levels = _synthetic_release_levels()

    features = build_composite_features(levels)

    # The first level difference is missing. At index 61, exactly 60 prior
    # transformed observations are available; three raw months exist at index 63.
    assert features["growth_raw"].first_valid_index() == levels.index[61]
    assert features["inflation_raw"].first_valid_index() == levels.index[61]
    assert features["growth_smoothed"].first_valid_index() == levels.index[63]
    assert features["inflation_smoothed"].first_valid_index() == levels.index[63]


def test_raw_composites_are_equal_weight_means_of_all_four_zscores() -> None:
    features = build_composite_features(
        _synthetic_release_levels(),
        min_history=5,
        smoothing_window=3,
    )

    for axis, components in (
        ("growth", GROWTH_COMPONENTS),
        ("inflation", INFLATION_COMPONENTS),
    ):
        z_columns = [f"{component}_z" for component in components]
        complete = features[z_columns].notna().all(axis=1)
        assert complete.any()
        expected = features.loc[complete, z_columns].sum(axis=1) / 4.0
        pd.testing.assert_series_equal(
            features.loc[complete, f"{axis}_raw"],
            expected,
            check_names=False,
        )


def test_smoothed_composites_are_trailing_three_month_full_window_means() -> None:
    features = build_composite_features(
        _synthetic_release_levels(),
        min_history=5,
        smoothing_window=3,
    )

    for axis in ("growth", "inflation"):
        expected = features[f"{axis}_raw"].rolling(window=3, min_periods=3).mean()
        pd.testing.assert_series_equal(
            features[f"{axis}_smoothed"],
            expected,
            check_names=False,
        )


def test_missing_component_value_does_not_trigger_dynamic_reweighting() -> None:
    levels = _synthetic_release_levels(periods=30)
    missing_month = levels.index[15]
    levels.loc[missing_month, "payrolls"] = np.nan

    features = build_composite_features(levels, min_history=5, smoothing_window=3)

    assert pd.isna(features.loc[missing_month, "payrolls_z"])
    assert pd.isna(features.loc[missing_month, "growth_raw"])
    assert pd.isna(features.loc[missing_month, "growth_smoothed"])
    assert pd.notna(features.loc[missing_month, "inflation_raw"])


def test_missing_required_component_column_is_rejected() -> None:
    levels = _synthetic_release_levels().drop(columns="average_hourly_earnings")

    with pytest.raises((KeyError, ValueError), match="average_hourly_earnings"):
        build_composite_features(levels)
