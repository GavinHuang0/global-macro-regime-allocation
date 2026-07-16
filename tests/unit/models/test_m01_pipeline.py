"""Unit tests for deterministic four-quadrant regime classification."""

from __future__ import annotations

import numpy as np
import pytest

from regime_allocation.models.m01_deterministic_composite.pipeline import (
    Regime,
    classify_regime,
)


@pytest.mark.parametrize(
    ("growth_score", "inflation_score", "expected"),
    [
        (1.0, 1.0, Regime.GROWTH_UP_INFLATION_UP),
        (-1.0, 1.0, Regime.GROWTH_DOWN_INFLATION_UP),
        (1.0, -1.0, Regime.GROWTH_UP_INFLATION_DOWN),
        (-1.0, -1.0, Regime.GROWTH_DOWN_INFLATION_DOWN),
    ],
)
def test_classify_regime_maps_all_four_quadrants(
    growth_score: float,
    inflation_score: float,
    expected: Regime,
) -> None:
    assert classify_regime(growth_score, inflation_score) is expected


@pytest.mark.parametrize(
    ("growth_score", "inflation_score", "expected"),
    [
        (0.0, 1.0, Regime.GROWTH_UP_INFLATION_UP),
        (-0.0, -1.0, Regime.GROWTH_UP_INFLATION_DOWN),
        (-1.0, 0.0, Regime.GROWTH_DOWN_INFLATION_UP),
        (0.0, 0.0, Regime.GROWTH_UP_INFLATION_UP),
    ],
)
def test_exact_zero_ties_are_assigned_to_the_up_side(
    growth_score: float,
    inflation_score: float,
    expected: Regime,
) -> None:
    assert classify_regime(growth_score, inflation_score) is expected


@pytest.mark.parametrize(
    ("growth_score", "inflation_score"),
    [(np.nan, 0.0), (0.0, np.nan)],
)
def test_missing_axis_score_cannot_be_classified(
    growth_score: float,
    inflation_score: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        classify_regime(growth_score, inflation_score)


def test_regime_values_are_stable_public_identifiers() -> None:
    assert {regime.value for regime in Regime} == {
        "growth_up_inflation_up",
        "growth_down_inflation_up",
        "growth_up_inflation_down",
        "growth_down_inflation_down",
    }
