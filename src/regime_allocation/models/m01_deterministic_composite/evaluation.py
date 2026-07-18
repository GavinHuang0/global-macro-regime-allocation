"""Evaluate Model 01's four-state probability forecasts.

Inputs are normalized probability rows in the canonical state order and their
realized deterministic regime IDs. Outputs cover proper probability scores,
classification accuracy and class balance, separate growth/inflation axis
scores, entropy, and top-label and one-versus-rest calibration bins. This module
does not choose which checkpoints are eligible; callers must first exclude any
forecast made on or after its target label became available.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import numpy as np

from regime_allocation.models.m01_deterministic_composite.pipeline import (
    REGIME_ORDER,
    Regime,
)


STATE_IDS = tuple(regime.value for regime in REGIME_ORDER)
STATE_COUNT = len(STATE_IDS)
_GROWTH_UP_POSITIONS = (0, 2)
_INFLATION_UP_POSITIONS = (0, 1)
_NORMALIZATION_ATOL = 1.0e-10


@dataclass(frozen=True)
class CalibrationBin:
    """One equal-width bin for top-label confidence calibration."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    upper_bound_inclusive: bool
    count: int
    mean_confidence: float | None
    empirical_accuracy: float | None
    absolute_gap: float | None


@dataclass(frozen=True)
class ClasswiseCalibrationBin:
    """One one-vs-rest reliability bin for one canonical regime."""

    regime_id: str
    bin_index: int
    lower_bound: float
    upper_bound: float
    upper_bound_inclusive: bool
    count: int
    mean_probability: float | None
    empirical_frequency: float | None
    absolute_gap: float | None


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate scores under explicitly frozen metric conventions."""

    observation_count: int
    negative_log_likelihood: float
    multiclass_brier_score: float
    map_accuracy: float
    balanced_accuracy: float
    macro_recall: float
    macro_f1: float
    growth_axis_brier_score: float
    inflation_axis_brier_score: float
    expected_calibration_error: float
    classwise_expected_calibration_error: float
    mean_posterior_entropy: float
    calibration_bins: tuple[CalibrationBin, ...]
    classwise_calibration_bins: tuple[ClasswiseCalibrationBin, ...]


def _as_probabilities(probabilities: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1:] != (STATE_COUNT,):
        raise ValueError(
            f"probabilities must have shape (observations, {STATE_COUNT})"
        )
    if values.shape[0] == 0:
        raise ValueError("at least one forecast observation is required")
    if not np.isfinite(values).all():
        raise ValueError("forecast probabilities must contain only finite values")
    if (values < 0.0).any():
        raise ValueError("forecast probabilities cannot be negative")
    if not np.allclose(
        values.sum(axis=1),
        1.0,
        atol=_NORMALIZATION_ATOL,
        rtol=_NORMALIZATION_ATOL,
    ):
        raise ValueError("every forecast probability row must sum to one")
    return values.copy()


def encode_regime_targets(
    targets: Iterable[Regime | str],
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    """Encode canonical regime IDs as integer positions without reordering."""

    if isinstance(targets, (str, bytes)):
        raise TypeError("targets must be an iterable of regime identifiers")
    encoded: list[int] = []
    for target in targets:
        if isinstance(target, Regime):
            state_id = target.value
        elif isinstance(target, str):
            state_id = target
        else:
            raise TypeError("targets must contain Regime values or canonical state IDs")
        try:
            encoded.append(STATE_IDS.index(state_id))
        except ValueError as error:
            raise ValueError(f"unknown regime identifier: {state_id!r}") from error
    values = np.asarray(encoded, dtype=np.int64)
    if expected_length is not None and len(values) != expected_length:
        raise ValueError("target count must match forecast observation count")
    if len(values) == 0:
        raise ValueError("at least one target is required")
    return values


def _validated_inputs(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
) -> tuple[np.ndarray, np.ndarray]:
    forecasts = _as_probabilities(probabilities)
    encoded = encode_regime_targets(targets, expected_length=len(forecasts))
    return forecasts, encoded


def negative_log_likelihood(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
) -> float:
    """Return mean multiclass log loss in nats; lower is better.

    A zero probability assigned to an observed regime produces infinite loss,
    rather than being silently clipped.
    """

    forecasts, encoded = _validated_inputs(probabilities, targets)
    assigned = forecasts[np.arange(len(forecasts)), encoded]
    if np.any(assigned == 0.0):
        return math.inf
    return float(-np.log(assigned).mean())


def multiclass_brier_score(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
) -> float:
    """Return the mean unscaled multiclass Brier score in ``[0, 2]``."""

    forecasts, encoded = _validated_inputs(probabilities, targets)
    observed = np.zeros_like(forecasts)
    observed[np.arange(len(forecasts)), encoded] = 1.0
    return float(np.square(forecasts - observed).sum(axis=1).mean())


def map_accuracy(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
) -> float:
    """Return maximum-a-posteriori classification accuracy.

    Exact probability ties are resolved by the permanent canonical state order.
    """

    forecasts, encoded = _validated_inputs(probabilities, targets)
    predicted = forecasts.argmax(axis=1)
    return float(np.mean(predicted == encoded))


def _classification_metrics(
    forecasts: np.ndarray,
    encoded: np.ndarray,
) -> tuple[float, float]:
    predicted = forecasts.argmax(axis=1)

    recalls: list[float] = []
    f1_scores: list[float] = []
    for state in range(STATE_COUNT):
        true_state = encoded == state
        predicted_state = predicted == state
        true_positives = int(np.sum(true_state & predicted_state))
        false_positives = int(np.sum(~true_state & predicted_state))
        false_negatives = int(np.sum(true_state & ~predicted_state))
        support = true_positives + false_negatives
        if support:
            recalls.append(true_positives / support)
        denominator = 2 * true_positives + false_positives + false_negatives
        if denominator:
            f1_scores.append(2 * true_positives / denominator)

    # At least one class has support because empty inputs are rejected. The F1
    # active-class union is also necessarily non-empty.
    return float(np.mean(recalls)), float(np.mean(f1_scores))


def macro_recall(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
) -> float:
    """Return recall averaged equally over regimes present in the targets."""

    forecasts, encoded = _validated_inputs(probabilities, targets)
    recall, _ = _classification_metrics(forecasts, encoded)
    return recall


def balanced_accuracy(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
) -> float:
    """Return balanced accuracy, equal to supported-class macro recall."""

    return macro_recall(probabilities, targets)


def macro_f1_score(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
) -> float:
    """Return F1 averaged over the union of observed and predicted regimes."""

    forecasts, encoded = _validated_inputs(probabilities, targets)
    _, score = _classification_metrics(forecasts, encoded)
    return score


def axis_brier_scores(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
) -> tuple[float, float]:
    """Return binary Brier scores for the growth-up and inflation-up axes."""

    forecasts, encoded = _validated_inputs(probabilities, targets)
    growth_probability = forecasts[:, _GROWTH_UP_POSITIONS].sum(axis=1)
    inflation_probability = forecasts[:, _INFLATION_UP_POSITIONS].sum(axis=1)
    growth_observed = np.isin(encoded, _GROWTH_UP_POSITIONS).astype(float)
    inflation_observed = np.isin(encoded, _INFLATION_UP_POSITIONS).astype(float)
    return (
        float(np.square(growth_probability - growth_observed).mean()),
        float(np.square(inflation_probability - inflation_observed).mean()),
    )


def confidence_calibration_bins(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
    *,
    bin_count: int = 10,
) -> tuple[CalibrationBin, ...]:
    """Build equal-width top-label confidence calibration bins.

    The first nine bins for the default configuration are left-closed and
    right-open. The final bin includes confidence one. Empty-bin statistics are
    represented by ``None`` rather than fabricated zeros.
    """

    if isinstance(bin_count, bool) or not isinstance(bin_count, (int, np.integer)):
        raise TypeError("bin_count must be an integer")
    if int(bin_count) < 2:
        raise ValueError("bin_count must be at least two")
    bins = int(bin_count)
    forecasts, encoded = _validated_inputs(probabilities, targets)
    predicted = forecasts.argmax(axis=1)
    confidence = forecasts.max(axis=1)
    correct = (predicted == encoded).astype(float)
    membership = np.minimum(np.floor(confidence * bins).astype(int), bins - 1)

    output: list[CalibrationBin] = []
    for index in range(bins):
        selected = membership == index
        count = int(selected.sum())
        if count:
            mean_confidence = float(confidence[selected].mean())
            empirical_accuracy = float(correct[selected].mean())
            absolute_gap = abs(mean_confidence - empirical_accuracy)
        else:
            mean_confidence = None
            empirical_accuracy = None
            absolute_gap = None
        output.append(
            CalibrationBin(
                bin_index=index,
                lower_bound=index / bins,
                upper_bound=(index + 1) / bins,
                upper_bound_inclusive=index == bins - 1,
                count=count,
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                absolute_gap=absolute_gap,
            )
        )
    return tuple(output)


def expected_calibration_error(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
    *,
    bin_count: int = 10,
) -> float:
    """Return count-weighted absolute top-label calibration error."""

    forecasts, encoded = _validated_inputs(probabilities, targets)
    target_ids = [STATE_IDS[position] for position in encoded]
    bins = confidence_calibration_bins(
        forecasts, target_ids, bin_count=bin_count
    )
    total = len(forecasts)
    return float(
        sum(
            item.count * item.absolute_gap
            for item in bins
            if item.absolute_gap is not None
        )
        / total
    )


def classwise_calibration_bins(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
    *,
    bin_count: int = 10,
) -> tuple[ClasswiseCalibrationBin, ...]:
    """Build equal-width one-vs-rest reliability bins for every regime."""

    if isinstance(bin_count, bool) or not isinstance(bin_count, (int, np.integer)):
        raise TypeError("bin_count must be an integer")
    if int(bin_count) < 2:
        raise ValueError("bin_count must be at least two")
    bins = int(bin_count)
    forecasts, encoded = _validated_inputs(probabilities, targets)
    records: list[ClasswiseCalibrationBin] = []
    for state_position, state_id in enumerate(STATE_IDS):
        state_probabilities = forecasts[:, state_position]
        observed = (encoded == state_position).astype(float)
        membership = np.minimum(
            np.floor(state_probabilities * bins).astype(int), bins - 1
        )
        for index in range(bins):
            selected = membership == index
            count = int(selected.sum())
            if count:
                mean_probability = float(state_probabilities[selected].mean())
                empirical_frequency = float(observed[selected].mean())
                absolute_gap = abs(mean_probability - empirical_frequency)
            else:
                mean_probability = None
                empirical_frequency = None
                absolute_gap = None
            records.append(
                ClasswiseCalibrationBin(
                    regime_id=state_id,
                    bin_index=index,
                    lower_bound=index / bins,
                    upper_bound=(index + 1) / bins,
                    upper_bound_inclusive=index == bins - 1,
                    count=count,
                    mean_probability=mean_probability,
                    empirical_frequency=empirical_frequency,
                    absolute_gap=absolute_gap,
                )
            )
    return tuple(records)


def classwise_expected_calibration_error(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
    *,
    bin_count: int = 10,
) -> float:
    """Return macro one-vs-rest ECE across the four canonical regimes."""

    forecasts, encoded = _validated_inputs(probabilities, targets)
    target_ids = [STATE_IDS[position] for position in encoded]
    bins = classwise_calibration_bins(
        forecasts, target_ids, bin_count=bin_count
    )
    class_errors: list[float] = []
    for state_id in STATE_IDS:
        state_bins = [item for item in bins if item.regime_id == state_id]
        class_errors.append(
            sum(
                item.count * item.absolute_gap
                for item in state_bins
                if item.absolute_gap is not None
            )
            / len(forecasts)
        )
    return float(np.mean(class_errors))


def evaluate_regime_probabilities(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    targets: Iterable[Regime | str],
    *,
    calibration_bin_count: int = 10,
) -> EvaluationMetrics:
    """Calculate the frozen Model 01 metric suite in one validated pass."""

    forecasts, encoded = _validated_inputs(probabilities, targets)
    target_ids = [STATE_IDS[position] for position in encoded]
    assigned = forecasts[np.arange(len(forecasts)), encoded]
    nll = math.inf if np.any(assigned == 0.0) else float(-np.log(assigned).mean())

    observed = np.zeros_like(forecasts)
    observed[np.arange(len(forecasts)), encoded] = 1.0
    brier = float(np.square(forecasts - observed).sum(axis=1).mean())
    predicted = forecasts.argmax(axis=1)
    accuracy = float(np.mean(predicted == encoded))
    recall, f1 = _classification_metrics(forecasts, encoded)
    growth_brier, inflation_brier = axis_brier_scores(forecasts, target_ids)
    bins = confidence_calibration_bins(
        forecasts, target_ids, bin_count=calibration_bin_count
    )
    ece = float(
        sum(
            item.count * item.absolute_gap
            for item in bins
            if item.absolute_gap is not None
        )
        / len(forecasts)
    )
    classwise_bins = classwise_calibration_bins(
        forecasts, target_ids, bin_count=calibration_bin_count
    )
    classwise_ece = classwise_expected_calibration_error(
        forecasts, target_ids, bin_count=calibration_bin_count
    )
    positive = forecasts > 0.0
    entropy_terms = np.zeros_like(forecasts)
    entropy_terms[positive] = -forecasts[positive] * np.log(forecasts[positive])
    mean_entropy = float(entropy_terms.sum(axis=1).mean())
    return EvaluationMetrics(
        observation_count=len(forecasts),
        negative_log_likelihood=nll,
        multiclass_brier_score=brier,
        map_accuracy=accuracy,
        balanced_accuracy=recall,
        macro_recall=recall,
        macro_f1=f1,
        growth_axis_brier_score=growth_brier,
        inflation_axis_brier_score=inflation_brier,
        expected_calibration_error=ece,
        classwise_expected_calibration_error=classwise_ece,
        mean_posterior_entropy=mean_entropy,
        calibration_bins=bins,
        classwise_calibration_bins=classwise_bins,
    )
