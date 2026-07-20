"""Test Model 02's unsmoothed deterministic composite-score contract.

Synthetic transformed features verify strictly lagged standardization, fixed
equal weights, missing-value behavior, immediate score availability after the
warm-up, and invariance of past scores to future data.  No quadrant label or
probability calculation belongs to this stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.scores import (
    ALL_COMPONENTS,
    GROWTH_COMPONENTS,
    INFLATION_COMPONENTS,
    build_composite_scores,
    lagged_expanding_zscore,
)


def _transformed_panel(periods: int = 90) -> pd.DataFrame:
    months = pd.date_range("1995-01-01", periods=periods, freq="MS")
    index = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            component: (
                np.sin(index / (2.0 + position / 5.0))
                + 0.2 * np.cos(index / (4.0 + position / 7.0))
                + 0.002 * position * index
            )
            for position, component in enumerate(ALL_COMPONENTS, start=1)
        },
        index=months,
    )


def test_lagged_zscore_uses_only_strictly_prior_history() -> None:
    values = pd.Series(np.arange(61, dtype=float))

    actual = lagged_expanding_zscore(values, min_periods=60, ddof=1)

    assert actual.iloc[:60].isna().all()
    expected = (values.iloc[60] - values.iloc[:60].mean()) / values.iloc[:60].std(
        ddof=1
    )
    assert actual.iloc[60] == pytest.approx(expected)


def test_scores_are_fixed_equal_weight_means_without_smoothing() -> None:
    transformed = _transformed_panel()
    scores = build_composite_scores(transformed, min_history=12)

    assert "growth_smoothed" not in scores
    assert "inflation_smoothed" not in scores
    for axis, components in (
        ("growth", GROWTH_COMPONENTS),
        ("inflation", INFLATION_COMPONENTS),
    ):
        columns = [f"{component}_z" for component in components]
        complete = scores[columns].notna().all(axis=1)
        expected = scores.loc[complete, columns].sum(axis=1) / 4.0
        pd.testing.assert_series_equal(
            scores.loc[complete, f"{axis}_score"], expected, check_names=False
        )


def test_first_score_is_not_delayed_by_a_trailing_window() -> None:
    transformed = _transformed_panel(periods=30)

    scores = build_composite_scores(transformed, min_history=10)

    assert scores["growth_score"].first_valid_index() == transformed.index[10]
    assert scores["inflation_score"].first_valid_index() == transformed.index[10]


def test_missing_component_never_triggers_dynamic_reweighting() -> None:
    transformed = _transformed_panel(periods=30)
    missing_month = transformed.index[18]
    transformed.loc[missing_month, "payrolls"] = np.nan

    scores = build_composite_scores(transformed, min_history=10)

    assert pd.isna(scores.loc[missing_month, "payrolls_z"])
    assert pd.isna(scores.loc[missing_month, "growth_score"])
    assert pd.notna(scores.loc[missing_month, "inflation_score"])


def test_future_values_cannot_change_past_scores() -> None:
    transformed = _transformed_panel(periods=90)
    shocked = transformed.copy()
    shocked.iloc[70:] = shocked.iloc[70:] + 1_000_000.0

    baseline_scores = build_composite_scores(transformed, min_history=12)
    shocked_scores = build_composite_scores(shocked, min_history=12)

    pd.testing.assert_frame_equal(
        baseline_scores.iloc[:70],
        shocked_scores.iloc[:70],
    )


def test_missing_component_column_is_rejected() -> None:
    transformed = _transformed_panel().drop(columns="core_pce")

    with pytest.raises(ValueError, match="core_pce"):
        build_composite_scores(transformed)
