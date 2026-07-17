"""Generic joint-path filtering primitives for Model 01.

The functions in this module know nothing about a particular release family or
likelihood estimator.  They operate on a four-month probability tensor in the
canonical Model 01 state order, which keeps the event orchestration layer
separate from the underlying probability arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from regime_allocation.models.m01_deterministic_composite.pipeline import (
    REGIME_ORDER,
    Regime,
)


STATE_IDS = tuple(regime.value for regime in REGIME_ORDER)
STATE_COUNT = len(STATE_IDS)
PATH_MONTHS = 4
PATH_SHAPE = (STATE_COUNT,) * PATH_MONTHS
_NORMALIZATION_ATOL = 1.0e-10


@dataclass(frozen=True)
class AxisLogLikelihood:
    """One log-likelihood vector attached to one month in the path.

    ``axis`` is zero for the oldest retained month and three for the newest.
    More than one item may target the same axis; their log-likelihoods are then
    combined atomically with all other items in the update.
    """

    axis: int
    values: np.ndarray


@dataclass(frozen=True)
class PathUpdateResult:
    """Posterior and information diagnostics from one atomic update."""

    posterior: np.ndarray
    log_normalizer: float
    prior_entropy: float
    posterior_entropy: float
    kl_divergence: float


def _as_probability_vector(
    values: np.ndarray | Iterable[float],
    *,
    name: str,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (STATE_COUNT,):
        raise ValueError(f"{name} must have shape ({STATE_COUNT},)")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} must contain only finite values")
    if (probabilities < 0.0).any():
        raise ValueError(f"{name} cannot contain negative probabilities")
    if not np.isclose(
        probabilities.sum(), 1.0, atol=_NORMALIZATION_ATOL, rtol=_NORMALIZATION_ATOL
    ):
        raise ValueError(f"{name} must sum to one")
    return probabilities.copy()


def _as_transition_array(
    transition_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    if isinstance(transition_matrix, pd.DataFrame):
        labels = tuple(map(str, transition_matrix.index))
        columns = tuple(map(str, transition_matrix.columns))
        if labels != STATE_IDS or columns != STATE_IDS:
            raise ValueError("transition DataFrame does not use canonical state order")
        matrix = transition_matrix.to_numpy(dtype=float)
    else:
        matrix = np.asarray(transition_matrix, dtype=float)
    expected_shape = (STATE_COUNT, STATE_COUNT)
    if matrix.shape != expected_shape:
        raise ValueError(f"transition matrix must have shape {expected_shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("transition matrix must contain only finite values")
    if (matrix < 0.0).any():
        raise ValueError("transition matrix cannot contain negative probabilities")
    if not np.allclose(
        matrix.sum(axis=1),
        1.0,
        atol=_NORMALIZATION_ATOL,
        rtol=_NORMALIZATION_ATOL,
    ):
        raise ValueError("transition matrix rows must sum to one")
    return matrix.copy()


def _as_path_probabilities(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != PATH_SHAPE:
        raise ValueError(f"{name} must have shape {PATH_SHAPE}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} must contain only finite values")
    if (probabilities < 0.0).any():
        raise ValueError(f"{name} cannot contain negative probabilities")
    if not np.isclose(
        probabilities.sum(), 1.0, atol=_NORMALIZATION_ATOL, rtol=_NORMALIZATION_ATOL
    ):
        raise ValueError(f"{name} must sum to one")
    return probabilities.copy()


def _validate_axis(axis: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise TypeError("path axis must be an integer")
    normalized = int(axis)
    if not 0 <= normalized < PATH_MONTHS:
        raise ValueError(f"path axis must be between 0 and {PATH_MONTHS - 1}")
    return normalized


def _state_position(state: Regime | str) -> int:
    if isinstance(state, Regime):
        state_id = state.value
    elif isinstance(state, str):
        state_id = state
    else:
        raise TypeError("confirmed regime must be a Regime or canonical state ID")
    try:
        return STATE_IDS.index(state_id)
    except ValueError as error:
        raise ValueError(f"unknown regime identifier: {state_id!r}") from error


def initialize_markov_path(
    initial_probabilities: np.ndarray | Iterable[float],
    transition_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Construct ``P(R[m-3:m])`` from a marginal and first-order dynamics.

    For states ``a, b, c, d``, the returned tensor is

    ``pi[a] * A[a,b] * A[b,c] * A[c,d]``.
    """

    initial = _as_probability_vector(
        initial_probabilities, name="initial probabilities"
    )
    transition = _as_transition_array(transition_matrix)
    path = np.einsum(
        "a,ab,bc,cd->abcd",
        initial,
        transition,
        transition,
        transition,
    )
    if not np.isclose(path.sum(), 1.0, atol=1.0e-12, rtol=1.0e-12):
        raise RuntimeError("Markov path initialization did not preserve probability mass")
    return path


def path_marginal(path_probabilities: np.ndarray, axis: int) -> np.ndarray:
    """Return the canonical four-state marginal for one path month."""

    probabilities = _as_path_probabilities(
        path_probabilities, name="path probabilities"
    )
    target_axis = _validate_axis(axis)
    summed_axes = tuple(item for item in range(PATH_MONTHS) if item != target_axis)
    return probabilities.sum(axis=summed_axes)


def path_marginals(path_probabilities: np.ndarray) -> np.ndarray:
    """Return a ``(4 months, 4 states)`` matrix of path marginals."""

    probabilities = _as_path_probabilities(
        path_probabilities, name="path probabilities"
    )
    return np.vstack(
        [
            probabilities.sum(
                axis=tuple(item for item in range(PATH_MONTHS) if item != axis)
            )
            for axis in range(PATH_MONTHS)
        ]
    )


def probability_entropy(probabilities: np.ndarray) -> float:
    """Return Shannon entropy in nats for a normalized probability array."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("probabilities must be a non-empty array")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(
        values.sum(), 1.0, atol=_NORMALIZATION_ATOL, rtol=_NORMALIZATION_ATOL
    ):
        raise ValueError("probabilities must sum to one")
    positive = values > 0.0
    return float(-np.sum(values[positive] * np.log(values[positive])))


def relative_entropy(posterior: np.ndarray, prior: np.ndarray) -> float:
    """Return ``KL(posterior || prior)`` in nats.

    The result is infinite if the posterior assigns positive mass where the
    prior assigns zero mass.
    """

    posterior_values = np.asarray(posterior, dtype=float)
    prior_values = np.asarray(prior, dtype=float)
    if posterior_values.shape != prior_values.shape:
        raise ValueError("posterior and prior must have the same shape")
    probability_entropy(posterior_values)
    probability_entropy(prior_values)
    positive = posterior_values > 0.0
    if np.any(prior_values[positive] == 0.0):
        return math.inf
    return float(
        np.sum(
            posterior_values[positive]
            * (np.log(posterior_values[positive]) - np.log(prior_values[positive]))
        )
    )


def _result(
    prior: np.ndarray,
    posterior: np.ndarray,
    *,
    log_normalizer: float,
) -> PathUpdateResult:
    return PathUpdateResult(
        posterior=posterior,
        log_normalizer=float(log_normalizer),
        prior_entropy=probability_entropy(prior),
        posterior_entropy=probability_entropy(posterior),
        kl_divergence=relative_entropy(posterior, prior),
    )


def update_path_log_likelihoods(
    path_prior: np.ndarray,
    updates: Iterable[AxisLogLikelihood | tuple[int, np.ndarray]],
) -> PathUpdateResult:
    """Apply one or more state likelihood vectors as one atomic update.

    Values are log likelihoods, not probabilities, so finite positive log
    densities are valid. ``-inf`` is allowed to represent an impossible state;
    ``NaN`` and positive infinity are rejected. At least one supplied update is
    required and the combined likelihood must leave positive prior mass.
    """

    prior = _as_path_probabilities(path_prior, name="path prior")
    normalized_updates: list[tuple[int, np.ndarray]] = []
    for item in updates:
        if isinstance(item, AxisLogLikelihood):
            raw_axis = item.axis
            raw_values = item.values
        else:
            try:
                raw_axis, raw_values = item
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "each update must be AxisLogLikelihood or an (axis, values) pair"
                ) from error
        axis = _validate_axis(raw_axis)
        values = np.asarray(raw_values, dtype=float)
        if values.shape != (STATE_COUNT,):
            raise ValueError(
                f"axis log likelihoods must have shape ({STATE_COUNT},)"
            )
        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("axis log likelihoods cannot contain NaN or +inf")
        if not np.isfinite(values).any():
            raise ValueError("each axis likelihood must allow at least one state")
        normalized_updates.append((axis, values.copy()))
    if not normalized_updates:
        raise ValueError("at least one axis log likelihood update is required")

    log_weights = np.full(PATH_SHAPE, -np.inf, dtype=float)
    positive_prior = prior > 0.0
    log_weights[positive_prior] = np.log(prior[positive_prior])
    for axis, values in normalized_updates:
        broadcast_shape = [1] * PATH_MONTHS
        broadcast_shape[axis] = STATE_COUNT
        log_weights += values.reshape(broadcast_shape)

    log_normalizer = float(logsumexp(log_weights))
    if not math.isfinite(log_normalizer):
        raise ValueError("combined likelihood eliminates all prior probability mass")
    posterior = np.exp(log_weights - log_normalizer)
    posterior /= posterior.sum()
    return _result(prior, posterior, log_normalizer=log_normalizer)


def condition_path_on_regimes(
    path_prior: np.ndarray,
    confirmations: Mapping[int, Regime | str],
) -> PathUpdateResult:
    """Hard-condition one atomic path update on known monthly regimes."""

    prior = _as_path_probabilities(path_prior, name="path prior")
    if not confirmations:
        raise ValueError("at least one regime confirmation is required")

    normalized: list[tuple[int, int]] = []
    for raw_axis, state in confirmations.items():
        normalized.append((_validate_axis(raw_axis), _state_position(state)))

    mask = np.ones(PATH_SHAPE, dtype=bool)
    for axis, state_position in normalized:
        axis_mask = np.zeros(STATE_COUNT, dtype=bool)
        axis_mask[state_position] = True
        broadcast_shape = [1] * PATH_MONTHS
        broadcast_shape[axis] = STATE_COUNT
        mask &= axis_mask.reshape(broadcast_shape)

    retained_mass = float(prior[mask].sum())
    if not math.isfinite(retained_mass) or retained_mass <= 0.0:
        raise ValueError("confirmed regimes have zero probability under the path prior")
    posterior = np.where(mask, prior / retained_mass, 0.0)
    return _result(prior, posterior, log_normalizer=math.log(retained_mass))
