"""Tests for Model 01's probabilistic evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from regime_allocation.models.m01_deterministic_composite.evaluation import (
    STATE_IDS,
    axis_brier_scores,
    balanced_accuracy,
    confidence_calibration_bins,
    classwise_calibration_bins,
    classwise_expected_calibration_error,
    encode_regime_targets,
    evaluate_regime_probabilities,
    expected_calibration_error,
    macro_f1_score,
    macro_recall,
    map_accuracy,
    multiclass_brier_score,
    negative_log_likelihood,
)
from regime_allocation.models.m01_deterministic_composite.pipeline import REGIME_ORDER


def _one_hot() -> np.ndarray:
    return np.eye(4)


def test_perfect_forecasts_have_perfect_scores() -> None:
    forecasts = _one_hot()
    metrics = evaluate_regime_probabilities(
        forecasts, STATE_IDS, calibration_bin_count=5
    )

    assert metrics.observation_count == 4
    assert metrics.negative_log_likelihood == pytest.approx(0.0)
    assert metrics.multiclass_brier_score == pytest.approx(0.0)
    assert metrics.map_accuracy == pytest.approx(1.0)
    assert metrics.balanced_accuracy == pytest.approx(1.0)
    assert metrics.macro_recall == pytest.approx(1.0)
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.growth_axis_brier_score == pytest.approx(0.0)
    assert metrics.inflation_axis_brier_score == pytest.approx(0.0)
    assert metrics.expected_calibration_error == pytest.approx(0.0)
    assert sum(item.count for item in metrics.calibration_bins) == 4
    assert metrics.calibration_bins[-1].upper_bound_inclusive


def test_uniform_forecasts_use_unscaled_multiclass_brier_convention() -> None:
    forecasts = np.full((4, 4), 0.25)

    assert negative_log_likelihood(forecasts, STATE_IDS) == pytest.approx(math.log(4))
    assert multiclass_brier_score(forecasts, STATE_IDS) == pytest.approx(0.75)
    # Canonical tie-breaking predicts state zero for every row.
    assert map_accuracy(forecasts, STATE_IDS) == pytest.approx(0.25)
    assert balanced_accuracy(forecasts, STATE_IDS) == pytest.approx(0.25)
    assert macro_recall(forecasts, STATE_IDS) == pytest.approx(0.25)
    assert macro_f1_score(forecasts, STATE_IDS) == pytest.approx(0.1)
    assert expected_calibration_error(forecasts, STATE_IDS, bin_count=4) == pytest.approx(
        0.0
    )


def test_axis_brier_scores_respect_quadrant_mapping() -> None:
    # The forecast is certain that growth is up, while splitting inflation.
    forecasts = np.array([[0.60, 0.00, 0.40, 0.00]])

    growth, inflation = axis_brier_scores(forecasts, [STATE_IDS[2]])

    assert growth == pytest.approx(0.0)
    # Truth is inflation down while forecast inflation-up probability is 0.6.
    assert inflation == pytest.approx(0.36)


def test_balanced_accuracy_averages_only_supported_target_classes() -> None:
    forecasts = np.array(
        [
            [0.8, 0.1, 0.1, 0.0],
            [0.6, 0.3, 0.1, 0.0],
            [0.2, 0.7, 0.1, 0.0],
            [0.1, 0.6, 0.2, 0.1],
        ]
    )
    targets = [STATE_IDS[0], STATE_IDS[1], STATE_IDS[1], STATE_IDS[1]]

    # Recall is 1/1 for state zero and 2/3 for state one.
    assert balanced_accuracy(forecasts, targets) == pytest.approx(5.0 / 6.0)


def test_confidence_calibration_bins_have_frozen_boundaries_and_empty_markers() -> None:
    forecasts = np.array(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.60, 0.20, 0.10, 0.10],
            [0.90, 0.05, 0.03, 0.02],
        ]
    )
    targets = [STATE_IDS[0], STATE_IDS[1], STATE_IDS[0]]

    bins = confidence_calibration_bins(forecasts, targets, bin_count=5)

    assert len(bins) == 5
    assert [item.count for item in bins] == [0, 0, 1, 1, 1]
    assert bins[0].mean_confidence is None
    assert bins[2].mean_confidence == pytest.approx(0.4)
    assert bins[2].empirical_accuracy == pytest.approx(1.0)
    assert bins[3].mean_confidence == pytest.approx(0.6)
    assert bins[3].empirical_accuracy == pytest.approx(0.0)
    assert bins[4].mean_confidence == pytest.approx(0.9)
    assert bins[4].empirical_accuracy == pytest.approx(1.0)
    assert expected_calibration_error(forecasts, targets, bin_count=5) == pytest.approx(
        (0.6 + 0.6 + 0.1) / 3
    )


def test_classwise_calibration_reports_every_regime_and_probability_bin() -> None:
    forecasts = np.full((4, 4), 0.25)

    bins = classwise_calibration_bins(forecasts, STATE_IDS, bin_count=4)

    assert len(bins) == 16
    occupied = [item for item in bins if item.count]
    assert len(occupied) == 4
    assert {item.regime_id for item in occupied} == set(STATE_IDS)
    assert all(item.mean_probability == pytest.approx(0.25) for item in occupied)
    assert all(item.empirical_frequency == pytest.approx(0.25) for item in occupied)
    assert classwise_expected_calibration_error(
        forecasts, STATE_IDS, bin_count=4
    ) == pytest.approx(0.0)


def test_zero_probability_for_observed_state_produces_infinite_log_loss() -> None:
    forecasts = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert negative_log_likelihood(forecasts, [STATE_IDS[1]]) == math.inf
    assert (
        evaluate_regime_probabilities(forecasts, [STATE_IDS[1]])
        .negative_log_likelihood
        == math.inf
    )


def test_regime_target_encoding_preserves_canonical_order_and_enum_support() -> None:
    encoded = encode_regime_targets([REGIME_ORDER[2], STATE_IDS[0]])
    np.testing.assert_array_equal(encoded, [2, 0])


@pytest.mark.parametrize(
    ("forecasts", "targets", "error", "match"),
    [
        (np.full((2, 3), 1 / 3), STATE_IDS[:2], ValueError, "shape"),
        (np.empty((0, 4)), [], ValueError, "at least one"),
        (np.full((2, 4), 0.20), STATE_IDS[:2], ValueError, "sum"),
        (
            np.array([[0.5, 0.5, 0.0, np.nan]]),
            [STATE_IDS[0]],
            ValueError,
            "finite",
        ),
        (
            np.array([[0.5, 0.6, -0.1, 0.0]]),
            [STATE_IDS[0]],
            ValueError,
            "negative",
        ),
        (np.full((2, 4), 0.25), [STATE_IDS[0]], ValueError, "count"),
        (np.full((1, 4), 0.25), ["unknown"], ValueError, "unknown"),
        (np.full((1, 4), 0.25), [0], TypeError, "Regime"),
    ],
)
def test_metric_inputs_are_strictly_validated(
    forecasts: np.ndarray,
    targets: list[object] | tuple[str, ...],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        evaluate_regime_probabilities(forecasts, targets)  # type: ignore[arg-type]


@pytest.mark.parametrize("bin_count", [1, 0, -1])
def test_calibration_rejects_too_few_bins(bin_count: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        confidence_calibration_bins(_one_hot(), STATE_IDS, bin_count=bin_count)


def test_calibration_rejects_boolean_bin_count() -> None:
    with pytest.raises(TypeError, match="integer"):
        confidence_calibration_bins(_one_hot(), STATE_IDS, bin_count=True)
