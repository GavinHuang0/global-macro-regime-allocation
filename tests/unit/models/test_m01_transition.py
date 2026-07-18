"""Test estimation and propagation of Model 01's fixed Markov transition model.

Controlled monthly histories verify canonical state order, consecutive-calendar
counting, inclusive knowledge cutoffs, both-endpoint label availability, Jeffreys
Dirichlet smoothing, raw MLE reporting, credible intervals, and invariance to
future labels. Exact 4-by-4 tensors then test the four-month path-shift operator.
The suite performs no I/O and locks both causal estimation and probability-mass
conservation.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m01_deterministic_composite.pipeline import (
    REGIME_ORDER,
    Regime,
)
from regime_allocation.models.m01_deterministic_composite import transition
from regime_allocation.models.m01_deterministic_composite.transition import (
    estimate_transition_matrix,
    propagate_joint_path,
)


A, B, C, D = REGIME_ORDER
EXPECTED_ORDER = (
    Regime.GROWTH_UP_INFLATION_UP,
    Regime.GROWTH_DOWN_INFLATION_UP,
    Regime.GROWTH_UP_INFLATION_DOWN,
    Regime.GROWTH_DOWN_INFLATION_DOWN,
)


def _matrix_values(matrix: object) -> np.ndarray:
    """Return a numeric view while allowing labeled pandas outputs."""
    if isinstance(matrix, pd.DataFrame):
        expected_labels = [regime.value for regime in REGIME_ORDER]
        assert list(matrix.index) == expected_labels
        assert list(matrix.columns) == expected_labels
        return matrix.to_numpy(dtype=float)
    return np.asarray(matrix, dtype=float)


def _history(
    rows: Iterable[tuple[str, Regime | str | None, str | None]],
) -> pd.DataFrame:
    """Convert concise month, regime, and availability tuples into history rows."""
    records = []
    for reference_month, regime, available_at in rows:
        regime_id = regime.value if isinstance(regime, Regime) else regime
        records.append(
            {
                "reference_month": pd.Timestamp(reference_month),
                "regime_id": regime_id,
                "label_available_at": (
                    pd.Timestamp(available_at) if available_at is not None else pd.NaT
                ),
                "data_status": "classified" if regime_id is not None else "unavailable",
            }
        )
    return pd.DataFrame.from_records(records)


def _history_for_isolated_transitions(
    pairs: Iterable[tuple[Regime, Regime]],
) -> pd.DataFrame:
    """Place each requested transition in its own two-month calendar block."""
    rows: list[tuple[str, Regime, str]] = []
    start = pd.Period("1980-01", freq="M")
    for position, (source, destination) in enumerate(pairs):
        source_month = start + 3 * position
        destination_month = source_month + 1
        rows.extend(
            [
                (
                    source_month.to_timestamp().date().isoformat(),
                    source,
                    (source_month + 2).to_timestamp().date().isoformat(),
                ),
                (
                    destination_month.to_timestamp().date().isoformat(),
                    destination,
                    (destination_month + 2).to_timestamp().date().isoformat(),
                ),
            ]
        )
    return _history(rows)


def _estimate(history: pd.DataFrame, **kwargs: object) -> object:
    """Estimate a transition model with a deliberately distant knowledge cutoff."""
    return estimate_transition_matrix(
        history,
        knowledge_cutoff=pd.Timestamp("2100-01-01"),
        **kwargs,
    )


def test_regime_order_is_explicit_and_stable() -> None:
    assert tuple(REGIME_ORDER) == EXPECTED_ORDER
    assert [regime.value for regime in REGIME_ORDER] == [
        "growth_up_inflation_up",
        "growth_down_inflation_up",
        "growth_up_inflation_down",
        "growth_down_inflation_down",
    ]


def test_counts_only_valid_consecutive_reference_months() -> None:
    history = _history(
        [
            ("2020-01-01", A, "2020-02-10"),
            ("2020-02-01", B, "2020-03-10"),
            ("2020-03-01", B, "2020-04-10"),
            ("2020-04-01", None, None),
            ("2020-05-01", C, "2020-06-10"),
            ("2020-06-01", D, "2020-07-10"),
            ("2020-07-01", A, "2020-08-10"),
        ]
    )

    estimate = _estimate(history)

    expected = np.zeros((4, 4), dtype=float)
    expected[0, 1] = 1
    expected[1, 1] = 1
    expected[2, 3] = 1
    expected[3, 0] = 1
    np.testing.assert_array_equal(_matrix_values(estimate.counts), expected)


def test_physical_calendar_gap_is_not_compressed_into_a_transition() -> None:
    history = _history(
        [
            ("2020-01-01", A, "2020-02-10"),
            ("2020-02-01", B, "2020-03-10"),
            ("2020-04-01", C, "2020-05-10"),
        ]
    )

    counts = _matrix_values(_estimate(history).counts)

    assert counts.sum() == 1
    assert counts[0, 1] == 1
    assert counts[1, 2] == 0


def test_knowledge_cutoff_uses_label_availability_and_is_inclusive() -> None:
    history = _history(
        [
            ("2020-01-01", A, "2020-02-15"),
            ("2020-02-01", B, "2020-03-15"),
            ("2020-03-01", C, "2020-05-01"),
        ]
    )

    before = estimate_transition_matrix(
        history,
        knowledge_cutoff=pd.Timestamp("2020-03-14"),
    )
    on_first_destination = estimate_transition_matrix(
        history,
        knowledge_cutoff=pd.Timestamp("2020-03-15"),
    )
    before_second_destination = estimate_transition_matrix(
        history,
        knowledge_cutoff=pd.Timestamp("2020-04-30"),
    )
    on_second_destination = estimate_transition_matrix(
        history,
        knowledge_cutoff=pd.Timestamp("2020-05-01"),
    )

    assert _matrix_values(before.counts).sum() == 0
    assert _matrix_values(on_first_destination.counts)[0, 1] == 1
    np.testing.assert_array_equal(
        _matrix_values(before_second_destination.counts),
        _matrix_values(on_first_destination.counts),
    )
    assert _matrix_values(on_second_destination.counts)[1, 2] == 1


def test_both_endpoint_labels_must_be_known_by_the_cutoff() -> None:
    history = _history(
        [
            ("2020-01-01", A, "2020-06-01"),
            ("2020-02-01", B, "2020-03-01"),
        ]
    )

    too_early = estimate_transition_matrix(
        history,
        knowledge_cutoff=pd.Timestamp("2020-05-31"),
    )
    both_known = estimate_transition_matrix(
        history,
        knowledge_cutoff=pd.Timestamp("2020-06-01"),
    )

    assert _matrix_values(too_early.counts).sum() == 0
    assert _matrix_values(both_known.counts)[0, 1] == 1


def test_estimate_at_cutoff_is_invariant_to_future_rows_and_labels() -> None:
    initial = _history(
        [
            ("2020-01-01", A, "2020-02-15"),
            ("2020-02-01", B, "2020-03-15"),
            ("2020-03-01", C, "2020-05-01"),
        ]
    )
    changed_future = pd.concat(
        [
            initial.assign(
                regime_id=[A.value, B.value, D.value],
                label_available_at=pd.to_datetime(
                    ["2020-02-15", "2020-03-15", "2020-05-01"]
                ),
            ),
            _history([("2020-04-01", A, "2020-06-01")]),
        ],
        ignore_index=True,
    )

    baseline = estimate_transition_matrix(
        initial,
        knowledge_cutoff=pd.Timestamp("2020-03-31"),
    )
    shocked = estimate_transition_matrix(
        changed_future,
        knowledge_cutoff=pd.Timestamp("2020-03-31"),
    )

    for field in (
        "counts",
        "mle_probabilities",
        "posterior_predictive",
        "ci_lower",
        "ci_upper",
    ):
        np.testing.assert_allclose(
            _matrix_values(getattr(baseline, field)),
            _matrix_values(getattr(shocked, field)),
            equal_nan=True,
        )


def test_default_jeffreys_smoothing_matches_exact_posterior_predictive() -> None:
    pairs = [
        (A, A),
        (A, A),
        (A, B),
        (B, C),
        (D, A),
        (D, B),
        (D, C),
        (D, D),
    ]
    estimate = _estimate(_history_for_isolated_transitions(pairs))

    expected_counts = np.array(
        [
            [2, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=float,
    )
    expected_predictive = np.array(
        [
            [0.5, 0.3, 0.1, 0.1],
            [1 / 6, 1 / 6, 1 / 2, 1 / 6],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    np.testing.assert_array_equal(_matrix_values(estimate.counts), expected_counts)
    np.testing.assert_allclose(
        _matrix_values(estimate.posterior_predictive),
        expected_predictive,
    )
    assert np.all(_matrix_values(estimate.posterior_predictive) > 0)
    np.testing.assert_allclose(
        _matrix_values(estimate.posterior_predictive).sum(axis=1),
        np.ones(4),
    )


def test_raw_mle_is_reported_separately_from_smoothed_probabilities() -> None:
    estimate = _estimate(_history_for_isolated_transitions([(A, A), (A, B)]))

    mle = _matrix_values(estimate.mle_probabilities)
    predictive = _matrix_values(estimate.posterior_predictive)

    np.testing.assert_allclose(mle[0], [0.5, 0.5, 0.0, 0.0])
    assert np.isnan(mle[1:]).all()
    np.testing.assert_allclose(predictive[1:], np.full((3, 4), 0.25))


def test_no_transition_history_produces_uniform_predictive_rows() -> None:
    history = _history([("2020-01-01", A, "2020-02-01")])

    estimate = _estimate(history)

    assert _matrix_values(estimate.counts).sum() == 0
    np.testing.assert_allclose(
        _matrix_values(estimate.posterior_predictive),
        np.full((4, 4), 0.25),
    )
    assert np.isnan(_matrix_values(estimate.mle_probabilities)).all()


@pytest.mark.parametrize("alpha", [0.0, -0.5, np.nan, np.inf])
def test_invalid_dirichlet_alpha_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        _estimate(_history([("2020-01-01", A, "2020-02-01")]), alpha=alpha)


def test_marginal_credible_interval_matches_beta_quantiles() -> None:
    estimate = _estimate(
        _history_for_isolated_transitions([(A, A), (A, A), (A, B)]),
        credible_interval_level=0.95,
    )

    # Dirichlet(2.5, 1.5, 0.5, 0.5) has a Beta(2.5, 2.5)
    # marginal for the A -> A probability.
    lower = _matrix_values(estimate.ci_lower)
    upper = _matrix_values(estimate.ci_upper)
    predictive = _matrix_values(estimate.posterior_predictive)

    # Symmetric Beta(a, a) quantiles mirror around one half.
    assert lower[0, 0] == pytest.approx(1.0 - upper[0, 0])
    assert lower[0, 0] < 0.5 < upper[0, 0]
    assert np.all((0.0 <= lower) & (lower <= predictive))
    assert np.all((predictive <= upper) & (upper <= 1.0))


@pytest.mark.parametrize(
    ("probability", "a", "b", "expected"),
    [
        (0.25, 1.0, 1.0, 0.25),
        (0.25, 2.0, 1.0, 0.5),
        (0.75, 1.0, 2.0, 0.5),
    ],
)
def test_dependency_free_beta_quantile_matches_closed_forms(
    monkeypatch: pytest.MonkeyPatch,
    probability: float,
    a: float,
    b: float,
    expected: float,
) -> None:
    monkeypatch.setattr(transition, "_scipy_beta", None)

    assert transition._beta_ppf(probability, a, b) == pytest.approx(
        expected, abs=1e-12
    )


def test_credible_interval_narrows_with_more_same_proportion_data() -> None:
    low_pairs = [(A, A)] * 8 + [(A, B)] + [(A, D)]
    high_pairs = [(A, A)] * 80 + [(A, B)] * 10 + [(A, D)] * 10

    low = _estimate(_history_for_isolated_transitions(low_pairs))
    high = _estimate(_history_for_isolated_transitions(high_pairs))
    low_width = _matrix_values(low.ci_upper)[0, 0] - _matrix_values(low.ci_lower)[0, 0]
    high_width = (
        _matrix_values(high.ci_upper)[0, 0]
        - _matrix_values(high.ci_lower)[0, 0]
    )

    assert high_width < low_width


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_credible_interval_level_is_rejected(level: float) -> None:
    with pytest.raises(ValueError, match="credible"):
        _estimate(
            _history([("2020-01-01", A, "2020-02-01")]),
            credible_interval_level=level,
        )


def test_unsorted_history_is_sorted_by_reference_month() -> None:
    sorted_history = _history(
        [
            ("2020-01-01", A, "2020-02-01"),
            ("2020-02-01", B, "2020-03-01"),
            ("2020-03-01", C, "2020-04-01"),
        ]
    )
    unsorted_history = sorted_history.iloc[[2, 0, 1]].reset_index(drop=True)

    expected = _matrix_values(_estimate(sorted_history).counts)
    actual = _matrix_values(_estimate(unsorted_history).counts)

    np.testing.assert_array_equal(actual, expected)


def test_duplicate_reference_month_is_rejected() -> None:
    history = _history(
        [
            ("2020-01-01", A, "2020-02-01"),
            ("2020-01-01", B, "2020-02-02"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate"):
        _estimate(history)


def test_unknown_regime_identifier_is_rejected() -> None:
    history = _history(
        [
            ("2020-01-01", A, "2020-02-01"),
            ("2020-02-01", "not_a_regime", "2020-03-01"),
        ]
    )

    with pytest.raises(ValueError, match="regime"):
        _estimate(history)


def test_missing_required_history_column_is_rejected() -> None:
    history = _history([("2020-01-01", A, "2020-02-01")]).drop(
        columns="label_available_at"
    )

    with pytest.raises(ValueError, match="label_available_at"):
        _estimate(history)


def _example_transition_matrix() -> np.ndarray:
    """Return an asymmetric stochastic matrix for joint-path propagation tests."""
    return np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.2, 0.5, 0.2, 0.1],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )


def test_joint_path_shift_appends_transition_from_newest_state() -> None:
    posterior = np.zeros((4, 4, 4, 4))
    posterior[0, 1, 2, 3] = 1.0

    shifted = propagate_joint_path(posterior, _example_transition_matrix())

    expected = np.zeros_like(posterior)
    expected[1, 2, 3, :] = [0.1, 0.2, 0.3, 0.4]
    np.testing.assert_allclose(shifted, expected)


def test_joint_path_shift_marginalizes_the_oldest_state() -> None:
    posterior = np.zeros((4, 4, 4, 4))
    posterior[0, 1, 2, 3] = 0.25
    posterior[2, 1, 2, 3] = 0.75

    shifted = propagate_joint_path(posterior, _example_transition_matrix())

    expected = np.zeros_like(posterior)
    expected[1, 2, 3, :] = [0.1, 0.2, 0.3, 0.4]
    np.testing.assert_allclose(shifted, expected)


def test_joint_path_shift_propagates_full_current_state_uncertainty() -> None:
    posterior = np.zeros((4, 4, 4, 4))
    posterior[0, 0, 0, 0] = 0.6
    posterior[1, 1, 1, 2] = 0.4
    transition = _example_transition_matrix()

    shifted = propagate_joint_path(posterior, transition)
    next_month_marginal = shifted.sum(axis=(0, 1, 2))

    np.testing.assert_allclose(
        next_month_marginal,
        0.6 * transition[0] + 0.4 * transition[2],
    )


def test_joint_path_shift_preserves_overlap_and_probability_mass() -> None:
    rng = np.random.default_rng(20260716)
    posterior = rng.dirichlet(np.ones(4**4)).reshape((4, 4, 4, 4))
    transition = rng.dirichlet(np.ones(4), size=4)

    shifted = propagate_joint_path(posterior, transition)

    assert np.isfinite(shifted).all()
    assert np.all(shifted >= 0.0)
    assert shifted.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(shifted.sum(axis=3), posterior.sum(axis=0))


@pytest.mark.parametrize(
    ("posterior", "transition", "match"),
    [
        (np.full((4, 4, 4), 1 / 64), _example_transition_matrix(), "shape"),
        (
            np.full((4, 4, 4, 4), 1 / 512),
            _example_transition_matrix(),
            "sum|mass|normalize",
        ),
        (
            np.full((4, 4, 4, 4), 1 / 256),
            np.full((3, 3), 1 / 3),
            "shape",
        ),
        (
            np.full((4, 4, 4, 4), 1 / 256),
            np.full((4, 4), 0.20),
            "row|sum|stochastic",
        ),
    ],
)
def test_joint_path_shift_rejects_invalid_shapes_or_normalization(
    posterior: np.ndarray,
    transition: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        propagate_joint_path(posterior, transition)


@pytest.mark.parametrize("target", ["posterior", "transition"])
@pytest.mark.parametrize("invalid_value", [-0.1, np.nan, np.inf])
def test_joint_path_shift_rejects_negative_or_nonfinite_probabilities(
    target: str,
    invalid_value: float,
) -> None:
    posterior = np.full((4, 4, 4, 4), 1 / 256)
    transition = _example_transition_matrix()
    if target == "posterior":
        posterior.flat[0] = invalid_value
    else:
        transition[0, 0] = invalid_value

    with pytest.raises(ValueError, match="negative|finite|probab"):
        propagate_joint_path(posterior, transition)
