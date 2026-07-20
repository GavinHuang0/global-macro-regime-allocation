"""Test Model 02's causal Gaussian quadrant probability mapping.

The suite verifies the delete-one-component estimator, correlated bivariate
quadrant integration, covariance decomposition, causal training cutoffs, and
future-data invariance.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.probability_map import (
    REGIME_ORDER,
    causal_quadrant_mapping_history,
    delete_one_jackknife_variance,
    gaussian_quadrant_weights,
)


def test_delete_one_jackknife_equals_component_sample_variance_over_four() -> None:
    values = np.asarray([-1.0, 0.0, 2.0, 5.0])

    actual = delete_one_jackknife_variance(values)

    assert actual == pytest.approx(float(np.var(values, ddof=1) / 4.0))


def test_independent_zero_mean_map_assigns_one_quarter_to_each_quadrant() -> None:
    weights, probabilities = gaussian_quadrant_weights(
        [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]
    )

    assert list(weights) == list(REGIME_ORDER)
    assert list(probabilities) == list(REGIME_ORDER)
    assert all(value == pytest.approx(0.25) for value in weights.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_positive_correlation_increases_same_sign_zero_mean_quadrants() -> None:
    correlation = 0.6
    _, probabilities = gaussian_quadrant_weights(
        [0.0, 0.0], [[1.0, correlation], [correlation, 1.0]]
    )
    expected_same_sign = 0.25 + math.asin(correlation) / (2.0 * math.pi)

    assert probabilities["growth_up_inflation_up"] == pytest.approx(
        expected_same_sign, abs=1.0e-8
    )
    assert probabilities["growth_down_inflation_down"] == pytest.approx(
        expected_same_sign, abs=1.0e-8
    )
    assert probabilities["growth_down_inflation_up"] < 0.25
    assert probabilities["growth_up_inflation_down"] < 0.25


def _causal_inputs(periods: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = pd.date_range("2010-01-01", periods=periods, freq="MS")
    available = months + pd.offsets.MonthEnd(1)
    scores = pd.DataFrame(
        {
            "growth_score": np.linspace(-0.5, 0.5, periods),
            "inflation_score": np.linspace(0.3, -0.3, periods),
            "score_available_at": available,
        },
        index=months,
    )
    disagreement = pd.DataFrame(
        {
            "score_available_at": available,
            "growth_jackknife_variance": np.linspace(0.1, 0.2, periods),
            "inflation_jackknife_variance": np.linspace(0.2, 0.3, periods),
            "disagreement_status": "available",
        },
        index=months,
    )
    revisions: list[dict[str, object]] = []
    for position, month in enumerate(months):
        for horizon in (3, 12):
            revisions.append(
                {
                    "reference_month": month,
                    "horizon_months": horizon,
                    "revision_available_at": available[position],
                    "growth_revision_error": (-1.0) ** position * 0.02 * (position + 1),
                    "inflation_revision_error": 0.01 * (position - 3),
                    "revision_status": "available",
                }
            )
    return scores, disagreement, pd.DataFrame(revisions)


def test_causal_history_uses_only_strictly_earlier_available_inputs() -> None:
    scores, disagreement, revisions = _causal_inputs()

    history = causal_quadrant_mapping_history(
        scores,
        disagreement,
        revisions,
        horizons=(3, 12),
        baseline_horizon=12,
        minimum_disagreement_months=3,
        minimum_revision_months=3,
    )
    row = history[
        (history["reference_month"] == scores.index[4])
        & (history["revision_horizon_months"] == 12)
    ].iloc[0]

    assert row["mapping_status"] == "available"
    assert row["disagreement_training_months"] == 4
    assert row["revision_training_months"] == 4
    assert row["growth_disagreement_variance"] == pytest.approx(
        disagreement.iloc[:4]["growth_jackknife_variance"].mean()
    )
    assert row["probability_sum"] == pytest.approx(1.0)
    assert row["growth_inflation_map_covariance"] == pytest.approx(
        row["growth_inflation_revision_covariance"]
    )


def test_future_uncertainty_observations_cannot_change_past_probabilities() -> None:
    scores, disagreement, revisions = _causal_inputs()
    baseline = causal_quadrant_mapping_history(
        scores,
        disagreement,
        revisions,
        horizons=(3, 12),
        baseline_horizon=12,
        minimum_disagreement_months=3,
        minimum_revision_months=3,
    )
    shocked_disagreement = disagreement.copy()
    shocked_disagreement.iloc[8:, 1:3] = 1_000_000.0
    shocked_revisions = revisions.copy()
    future = shocked_revisions["reference_month"] >= scores.index[8]
    shocked_revisions.loc[
        future, ["growth_revision_error", "inflation_revision_error"]
    ] = 1_000_000.0
    shocked = causal_quadrant_mapping_history(
        scores,
        shocked_disagreement,
        shocked_revisions,
        horizons=(3, 12),
        baseline_horizon=12,
        minimum_disagreement_months=3,
        minimum_revision_months=3,
    )

    columns = [
        "growth_map_variance",
        "inflation_map_variance",
        *(f"probability_{regime}" for regime in REGIME_ORDER),
    ]
    past = baseline["reference_month"] < scores.index[8]
    pd.testing.assert_frame_equal(
        baseline.loc[past, columns].reset_index(drop=True),
        shocked.loc[past, columns].reset_index(drop=True),
    )

