"""Tests for Model 01's generic joint-path filtering primitives."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m01_deterministic_composite.inference import (
    PATH_SHAPE,
    STATE_IDS,
    AxisLogLikelihood,
    condition_path_on_regimes,
    initialize_markov_path,
    path_marginal,
    path_marginals,
    probability_entropy,
    relative_entropy,
    update_path_log_likelihoods,
)


def _transition() -> np.ndarray:
    return np.array(
        [
            [0.70, 0.10, 0.10, 0.10],
            [0.20, 0.50, 0.20, 0.10],
            [0.10, 0.20, 0.60, 0.10],
            [0.10, 0.20, 0.30, 0.40],
        ]
    )


def _uniform_path() -> np.ndarray:
    return np.full(PATH_SHAPE, 1.0 / np.prod(PATH_SHAPE))


def test_markov_path_initialization_matches_chain_factorization() -> None:
    initial = np.array([0.1, 0.2, 0.3, 0.4])
    transition = _transition()

    path = initialize_markov_path(initial, transition)

    assert path.shape == PATH_SHAPE
    assert path.sum() == pytest.approx(1.0)
    assert path[1, 2, 2, 0] == pytest.approx(
        initial[1] * transition[1, 2] * transition[2, 2] * transition[2, 0]
    )
    np.testing.assert_allclose(path_marginal(path, 0), initial)
    np.testing.assert_allclose(path_marginal(path, 1), initial @ transition)


def test_markov_initialization_accepts_only_canonically_labeled_frame() -> None:
    frame = pd.DataFrame(_transition(), index=STATE_IDS, columns=STATE_IDS)
    initialized = initialize_markov_path(np.full(4, 0.25), frame)
    assert initialized.sum() == pytest.approx(1.0)

    reordered = frame.iloc[::-1]
    with pytest.raises(ValueError, match="canonical"):
        initialize_markov_path(np.full(4, 0.25), reordered)


@pytest.mark.parametrize(
    ("initial", "transition", "match"),
    [
        (np.full(3, 1 / 3), _transition(), "shape"),
        (np.full(4, 0.20), _transition(), "sum"),
        (np.array([0.4, 0.3, 0.3, -0.0]), np.full((4, 4), 0.20), "rows"),
        (np.array([0.4, 0.3, 0.3, -0.1]), _transition(), "negative|sum"),
    ],
)
def test_markov_initialization_rejects_invalid_inputs(
    initial: np.ndarray,
    transition: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        initialize_markov_path(initial, transition)


def test_atomic_log_likelihood_update_across_axes_matches_manual_product() -> None:
    prior = initialize_markov_path(np.full(4, 0.25), _transition())
    prior_copy = prior.copy()
    oldest = np.log([0.1, 0.2, 0.3, 0.4])
    newest = np.log([0.5, 0.3, 0.1, 0.1])

    result = update_path_log_likelihoods(
        prior,
        [AxisLogLikelihood(0, oldest), (3, newest)],
    )

    expected = prior * np.exp(oldest)[:, None, None, None]
    expected *= np.exp(newest)[None, None, None, :]
    evidence = expected.sum()
    expected /= evidence
    np.testing.assert_allclose(result.posterior, expected)
    np.testing.assert_array_equal(prior, prior_copy)
    assert result.log_normalizer == pytest.approx(math.log(evidence))
    assert result.kl_divergence >= 0.0


def test_atomic_update_combines_repeated_same_axis_likelihoods() -> None:
    prior = _uniform_path()
    first = np.log([0.1, 0.2, 0.3, 0.4])
    second = np.log([0.4, 0.3, 0.2, 0.1])

    result = update_path_log_likelihoods(prior, [(2, first), (2, second)])

    expected_marginal = np.exp(first + second)
    expected_marginal /= expected_marginal.sum()
    np.testing.assert_allclose(path_marginal(result.posterior, 2), expected_marginal)


def test_log_likelihood_update_is_stable_under_common_log_shift() -> None:
    prior = _uniform_path()
    likelihood = np.array([-1002.0, -1001.0, -1000.0, -999.0])
    shifted = likelihood + 998.0

    first = update_path_log_likelihoods(prior, [(1, likelihood)])
    second = update_path_log_likelihoods(prior, [(1, shifted)])

    np.testing.assert_allclose(first.posterior, second.posterior)
    assert second.log_normalizer - first.log_normalizer == pytest.approx(998.0)


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    [
        ([], ValueError, "at least one"),
        ([(4, np.zeros(4))], ValueError, "axis"),
        ([(True, np.zeros(4))], TypeError, "integer"),
        ([(0, np.zeros(3))], ValueError, "shape"),
        ([(0, np.array([0.0, np.nan, 0.0, 0.0]))], ValueError, "NaN"),
        ([(0, np.array([0.0, np.inf, 0.0, 0.0]))], ValueError, "inf"),
        ([(0, np.full(4, -np.inf))], ValueError, "at least one state"),
    ],
)
def test_log_likelihood_update_rejects_invalid_updates(
    updates: list[tuple[object, np.ndarray]],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        update_path_log_likelihoods(_uniform_path(), updates)


def test_log_likelihood_update_rejects_elimination_of_prior_support() -> None:
    prior = np.zeros(PATH_SHAPE)
    prior[0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="eliminates"):
        update_path_log_likelihoods(
            prior, [(0, np.array([-np.inf, 0.0, 0.0, 0.0]))]
        )


def test_hard_confirmation_conditions_multiple_axes_atomically() -> None:
    prior = _uniform_path()

    result = condition_path_on_regimes(prior, {0: STATE_IDS[1], 3: STATE_IDS[2]})

    expected = np.zeros(PATH_SHAPE)
    expected[1, :, :, 2] = 1.0 / 16.0
    np.testing.assert_allclose(result.posterior, expected)
    assert result.log_normalizer == pytest.approx(math.log(1.0 / 16.0))
    assert result.posterior_entropy == pytest.approx(math.log(16.0))
    assert result.kl_divergence == pytest.approx(math.log(16.0))


def test_hard_confirmation_rejects_invalid_or_impossible_states() -> None:
    prior = np.zeros(PATH_SHAPE)
    prior[0, 0, 0, 0] = 1.0

    with pytest.raises(ValueError, match="at least one"):
        condition_path_on_regimes(prior, {})
    with pytest.raises(ValueError, match="unknown"):
        condition_path_on_regimes(prior, {0: "not_a_regime"})
    with pytest.raises(ValueError, match="zero probability"):
        condition_path_on_regimes(prior, {0: STATE_IDS[1]})


def test_marginals_entropy_and_relative_entropy_are_exact() -> None:
    prior = _uniform_path()
    marginals = path_marginals(prior)
    np.testing.assert_allclose(marginals, np.full((4, 4), 0.25))
    assert probability_entropy(prior) == pytest.approx(math.log(256.0))

    posterior = np.zeros(PATH_SHAPE)
    posterior[0, :, :, :] = 1.0 / 64.0
    assert relative_entropy(posterior, prior) == pytest.approx(math.log(4.0))


def test_relative_entropy_is_infinite_outside_prior_support() -> None:
    prior = np.zeros(4)
    prior[0] = 1.0
    posterior = np.zeros(4)
    posterior[1] = 1.0
    assert relative_entropy(posterior, prior) == math.inf
