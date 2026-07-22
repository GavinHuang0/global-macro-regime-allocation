"""Test Model 02's rolling four-month joint Gaussian filter primitives.

The tests cover sequential initialization, the exact block roll algebra,
cross-month information flow, Joseph release updates, exact-score conditioning,
mapping-only quadrant softness, event-order invariance, target-date guards, and
the optional deterministic 256-path Sobol readout.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.joint_filter import (
    JointGaussianState,
    VarDynamics,
    condition_on_exact_score,
    initialize_joint_exact,
    initialize_joint_gaussian,
    joint_quadrant_path_probabilities,
    monthly_quadrant_probabilities,
    roll_joint_gaussian,
    update_joint_gaussian,
)
from regime_allocation.models.m02_soft_composite.probability_map import REGIME_ORDER


def _edge(
    *,
    transition: np.ndarray | None = None,
    innovation: np.ndarray | None = None,
) -> VarDynamics:
    return VarDynamics(
        intercept=np.array([0.1, -0.2]),
        transition=(
            np.array([[0.7, 0.2], [-0.1, 0.6]])
            if transition is None
            else transition
        ),
        innovation_covariance=(
            np.array([[0.3, 0.04], [0.04, 0.2]])
            if innovation is None
            else innovation
        ),
    )


def _state() -> JointGaussianState:
    return initialize_joint_gaussian(
        "2024-01-01",
        np.array([0.3, -0.4]),
        np.array([[0.6, 0.1], [0.1, 0.5]]),
        [_edge(), _edge(), _edge()],
    )


def test_initialization_matches_sequential_var_moments_and_cross_covariance() -> None:
    edge = _edge()
    mean0 = np.array([0.3, -0.4])
    covariance0 = np.array([[0.6, 0.1], [0.1, 0.5]])

    state = initialize_joint_gaussian(
        "2024-01-01", mean0, covariance0, [edge, edge, edge]
    )

    means = [mean0]
    covariances = [covariance0]
    for _ in range(3):
        means.append(edge.intercept + edge.transition @ means[-1])
        covariances.append(
            edge.transition @ covariances[-1] @ edge.transition.T
            + edge.innovation_covariance
        )
    np.testing.assert_allclose(state.mean.reshape(4, 2), means)
    for position, covariance in enumerate(covariances):
        block = slice(2 * position, 2 * position + 2)
        np.testing.assert_allclose(state.covariance[block, block], covariance)
    np.testing.assert_allclose(
        state.covariance[0:2, 2:4], covariance0 @ edge.transition.T
    )


def test_roll_is_block_shift_and_adds_q_only_to_new_month() -> None:
    state = _state()
    edge = _edge()
    rolled = roll_joint_gaussian(state, edge)

    np.testing.assert_allclose(rolled.mean[:6], state.mean[2:])
    np.testing.assert_allclose(rolled.covariance[:6, :6], state.covariance[2:, 2:])
    expected_cross = state.covariance[2:, 6:8] @ edge.transition.T
    np.testing.assert_allclose(rolled.covariance[:6, 6:8], expected_cross)
    expected_new = (
        edge.transition @ state.covariance[6:8, 6:8] @ edge.transition.T
        + edge.innovation_covariance
    )
    np.testing.assert_allclose(rolled.covariance[6:8, 6:8], expected_new)


def test_exact_newest_score_then_roll_gives_new_month_variance_q() -> None:
    state = condition_on_exact_score(_state(), "2024-04-01", [0.5, -0.1])
    edge = _edge()

    rolled = roll_joint_gaussian(state, edge)

    np.testing.assert_allclose(
        rolled.covariance[6:8, 6:8], edge.innovation_covariance
    )
    np.testing.assert_allclose(rolled.covariance[:6, 6:8], 0.0, atol=1.0e-12)
    assert rolled.exact_mask == (*state.exact_mask[1:], False)


def test_linear_update_matches_direct_gaussian_conditioning_formula() -> None:
    state = _state()
    target = "2024-03-01"
    loading = np.array([[1.0, 0.0], [0.3, 0.8]])
    noise = np.array([[0.2, 0.03], [0.03, 0.4]])
    observation = np.array([0.7, -0.2])
    offset = np.array([0.05, -0.1])
    global_loading = np.zeros((2, 8))
    global_loading[:, 4:6] = loading
    innovation = observation - offset - global_loading @ state.mean
    innovation_covariance = (
        global_loading @ state.covariance @ global_loading.T + noise
    )
    gain = state.covariance @ global_loading.T @ np.linalg.inv(
        innovation_covariance
    )
    expected_mean = state.mean + gain @ innovation
    expected_covariance = state.covariance - gain @ global_loading @ state.covariance

    result = update_joint_gaussian(
        state,
        target,
        observation,
        loading,
        noise,
        intercept=offset,
    )

    assert result.applied
    np.testing.assert_allclose(result.state.mean, expected_mean)
    np.testing.assert_allclose(result.state.covariance, expected_covariance, atol=1e-12)
    assert np.linalg.eigvalsh(result.state.covariance).min() >= -1.0e-10


def test_partial_growth_evidence_updates_other_months_through_cross_covariance() -> None:
    state = _state()

    result = update_joint_gaussian(
        state,
        "2024-02-01",
        [2.0],
        [[1.0, 0.0]],
        [[0.1]],
    )

    assert not np.allclose(result.state.mean[0:2], state.mean[0:2])
    assert not np.allclose(result.state.mean[6:8], state.mean[6:8])


def test_independent_same_day_updates_are_order_invariant() -> None:
    state = _state()

    first_ab = update_joint_gaussian(
        state, "2024-04-01", [0.8], [[1.0, 0.0]], 0.25
    ).state
    final_ab = update_joint_gaussian(
        first_ab, "2024-04-01", [-0.3], [[0.0, 1.0]], 0.4
    ).state
    first_ba = update_joint_gaussian(
        state, "2024-04-01", [-0.3], [[0.0, 1.0]], 0.4
    ).state
    final_ba = update_joint_gaussian(
        first_ba, "2024-04-01", [0.8], [[1.0, 0.0]], 0.25
    ).state

    np.testing.assert_allclose(final_ab.mean, final_ba.mean, atol=1.0e-12)
    np.testing.assert_allclose(final_ab.covariance, final_ba.covariance, atol=1.0e-12)


def test_exact_conditioning_matches_schur_complement_and_is_idempotent() -> None:
    state = _state()
    block = slice(2, 4)
    score = np.array([-0.7, 0.9])
    gain = state.covariance[:, block] @ np.linalg.inv(
        state.covariance[block, block]
    )
    expected_mean = state.mean + gain @ (score - state.mean[block])
    expected_covariance = state.covariance - gain @ state.covariance[block, :]

    conditioned = condition_on_exact_score(state, "2024-02-01", score)

    expected_mean[block] = score
    expected_covariance[block, :] = 0.0
    expected_covariance[:, block] = 0.0
    np.testing.assert_allclose(conditioned.mean, expected_mean)
    np.testing.assert_allclose(conditioned.covariance, expected_covariance, atol=1e-12)
    assert conditioned.exact_mask[1]
    assert condition_on_exact_score(conditioned, "2024-02-01", score) is conditioned
    with pytest.raises(ValueError, match="conflicting exact score"):
        condition_on_exact_score(conditioned, "2024-02-01", [0.0, 0.0])


def test_release_for_exact_month_is_diagnostic_no_op() -> None:
    exact = condition_on_exact_score(_state(), "2024-03-01", [0.2, -0.1])

    result = update_joint_gaussian(
        exact, "2024-03-01", [100.0], [[1.0, 0.0]], [[0.5]]
    )

    assert not result.applied
    assert result.reason == "target_score_exact"
    assert result.state is exact
    assert np.isfinite(result.log_predictive_density)


def test_exact_score_still_has_soft_quadrant_probabilities_at_readout() -> None:
    exact = condition_on_exact_score(_state(), "2024-04-01", [0.0, 0.0])

    probabilities = monthly_quadrant_probabilities(
        exact, "2024-04-01", [[1.0, 0.0], [0.0, 1.0]]
    )

    assert all(probability == pytest.approx(0.25) for probability in probabilities.values())


def test_future_and_expired_targets_are_rejected() -> None:
    state = _state()

    with pytest.raises(ValueError, match="future target month"):
        update_joint_gaussian(
            state, "2024-05-01", [0.0], [[1.0, 0.0]], [[1.0]]
        )
    with pytest.raises(ValueError, match="expired target month"):
        condition_on_exact_score(state, "2023-12-01", [0.0, 0.0])


def test_initialize_from_exact_oldest_has_zero_oldest_cross_covariance() -> None:
    state = initialize_joint_exact(
        "2024-01-01", [0.4, -0.2], [_edge(), _edge(), _edge()]
    )

    assert state.exact_mask[0]
    np.testing.assert_allclose(state.covariance[0:2, :], 0.0)
    np.testing.assert_allclose(
        state.covariance[2:4, 2:4], _edge().innovation_covariance
    )


def test_sobol_path_readout_is_deterministic_complete_and_normalized() -> None:
    state = _state()
    omega = np.array([[0.2, 0.02], [0.02, 0.15]])

    first = joint_quadrant_path_probabilities(
        state, omega, sobol_samples=1024, seed=9
    )
    second = joint_quadrant_path_probabilities(
        state, omega, sobol_samples=1024, seed=9
    )

    assert first == second
    assert len(first) == 256
    assert set(first) == set(itertools.product(REGIME_ORDER, repeat=4))
    assert sum(first.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "covariance",
    [
        np.diag([1.0] * 7 + [-0.1]),
        np.full((8, 8), np.nan),
        np.eye(8) + np.triu(np.ones((8, 8)), 1),
    ],
)
def test_state_rejects_invalid_covariance(covariance: np.ndarray) -> None:
    months = tuple(pd.date_range("2024-01-01", periods=4, freq="MS"))
    with pytest.raises(ValueError, match="joint covariance"):
        JointGaussianState(months, np.zeros(8), covariance)
