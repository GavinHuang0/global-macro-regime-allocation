"""Test causal estimation and Gaussian propagation for Model 02's VAR(1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.probability_map import REGIME_ORDER
from regime_allocation.models.m02_soft_composite.var_transition import (
    Var1Fit,
    build_var_pair_audit,
    causal_var1_prior_history,
    fit_var1_ols,
    propagate_var1_prior,
    select_causal_var_pairs,
)


def _simulated_pair_frame(observations: int = 400) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260721)
    intercept = np.array([0.15, -0.08])
    transition = np.array([[0.55, 0.12], [-0.06, 0.70]])
    noise = rng.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[0.16, 0.03], [0.03, 0.09]],
        size=observations,
    )
    states = np.zeros((observations + 1, 2))
    for position in range(observations):
        states[position + 1] = intercept + transition @ states[position] + noise[position]
    frame = pd.DataFrame(
        {
            "source_growth_score": states[:-1, 0],
            "source_inflation_score": states[:-1, 1],
            "destination_growth_score": states[1:, 0],
            "destination_inflation_score": states[1:, 1],
        }
    )
    return frame, intercept, transition


def _score_history(periods: int = 12) -> pd.DataFrame:
    months = pd.date_range("2020-01-01", periods=periods, freq="MS")
    position = np.arange(periods, dtype=float)
    frame = pd.DataFrame(
        {
            "growth_score": np.sin(position / 2.0) + 0.1 * position,
            "inflation_score": np.cos(position / 3.0) - 0.05 * position,
            "score_available_at": months + pd.offsets.MonthEnd(1) + pd.Timedelta(days=10),
        },
        index=months,
    )
    frame.index.name = "reference_month"
    return frame


def _mapping_history(scores: pd.DataFrame) -> pd.DataFrame:
    result = scores.reset_index().copy()
    result["revision_horizon_months"] = 12
    result["mapping_status"] = "available"
    result["growth_map_variance"] = 0.4
    result["growth_inflation_map_covariance"] = 0.05
    result["inflation_map_variance"] = 0.3
    return result


def test_var_ols_recovers_known_dynamics_in_a_long_sample() -> None:
    pairs, expected_intercept, expected_transition = _simulated_pair_frame()

    fit = fit_var1_ols(pairs)

    np.testing.assert_allclose(fit.intercept, expected_intercept, atol=0.05)
    np.testing.assert_allclose(fit.transition, expected_transition, atol=0.06)
    assert fit.training_pairs == 400
    assert fit.residual_degrees_of_freedom == 397
    assert fit.design_rank == 3
    np.testing.assert_allclose(fit.residual_mean, 0.0, atol=1.0e-12)


def test_innovation_covariance_uses_residual_degrees_of_freedom() -> None:
    pairs, _, _ = _simulated_pair_frame(observations=40)
    fit = fit_var1_ols(pairs)
    source = pairs[["source_growth_score", "source_inflation_score"]].to_numpy()
    target = pairs[
        ["destination_growth_score", "destination_inflation_score"]
    ].to_numpy()
    residuals = target - (
        fit.intercept + source @ fit.transition.T
    )
    expected = residuals.T @ residuals / (len(pairs) - 3)

    np.testing.assert_allclose(fit.innovation_covariance, expected)


def test_var_ols_rejects_only_four_pairs() -> None:
    pairs, _, _ = _simulated_pair_frame(observations=4)

    with pytest.raises(ValueError, match="at least five"):
        fit_var1_ols(pairs)


def test_prior_uses_exact_source_and_adds_mapping_noise_at_readout() -> None:
    fit = Var1Fit(
        intercept=np.array([0.2, -0.1]),
        transition=np.array([[0.7, 0.2], [-0.1, 0.5]]),
        innovation_covariance=np.array([[0.3, 0.04], [0.04, 0.2]]),
        training_pairs=100,
        residual_degrees_of_freedom=97,
        design_rank=3,
        design_condition_number=2.0,
        spectral_radius=0.6,
        growth_r_squared=0.4,
        inflation_r_squared=0.5,
        residual_mean=np.zeros(2),
    )
    exact_score = np.array([0.4, -0.3])
    mapping_proxy = np.array([[0.5, 0.08], [0.08, 0.25]])

    prior_mean, latent_covariance, reporting_covariance = propagate_var1_prior(
        exact_score, mapping_proxy, fit
    )

    np.testing.assert_allclose(
        prior_mean, fit.intercept + fit.transition @ exact_score
    )
    np.testing.assert_allclose(latent_covariance, fit.innovation_covariance)
    np.testing.assert_allclose(
        reporting_covariance, fit.innovation_covariance + mapping_proxy
    )
    incorrectly_propagated = (
        fit.innovation_covariance
        + fit.transition @ mapping_proxy @ fit.transition.T
    )
    assert not np.allclose(reporting_covariance, incorrectly_propagated)


def test_pair_selection_does_not_bridge_gaps_or_admit_future_responses() -> None:
    scores = _score_history(8).drop(pd.Timestamp("2020-04-01"))
    audit = build_var_pair_audit(scores)

    assert "nonconsecutive_reference_months" in set(audit["pair_status"])
    selected = select_causal_var_pairs(
        audit,
        source_reference_month=pd.Timestamp("2020-06-01"),
        forecast_available_at=pd.Timestamp("2020-07-31"),
    )

    assert selected["destination_reference_month"].max() == pd.Timestamp("2020-06-01")
    month_gaps = (
        selected["destination_reference_month"].dt.to_period("M")
        - selected["source_reference_month"].dt.to_period("M")
    )
    assert all(item.n == 1 for item in month_gaps)


def test_score_history_rejects_timezone_aware_availability() -> None:
    scores = _score_history(8)
    scores["score_available_at"] = scores["score_available_at"].dt.tz_localize(
        "UTC"
    )

    with pytest.raises(ValueError, match="timezone-naive"):
        build_var_pair_audit(scores)


def test_future_rows_do_not_change_an_earlier_prior() -> None:
    scores = _score_history(15)
    mapping = _mapping_history(scores)
    audit = build_var_pair_audit(scores)
    baseline = causal_var1_prior_history(
        scores,
        mapping,
        audit,
        minimum_training_pairs=5,
    )
    cutoff_month = pd.Timestamp("2020-09-01")
    baseline_row = baseline[baseline["source_reference_month"] == cutoff_month].iloc[0]

    changed = scores.copy()
    future = changed.index > cutoff_month
    rng = np.random.default_rng(7)
    changed.loc[future, ["growth_score", "inflation_score"]] = rng.normal(
        loc=0.0,
        scale=100.0,
        size=(int(future.sum()), 2),
    )
    changed_mapping = _mapping_history(changed)
    changed_history = causal_var1_prior_history(
        changed,
        changed_mapping,
        build_var_pair_audit(changed),
        minimum_training_pairs=5,
    )
    changed_row = changed_history[
        changed_history["source_reference_month"] == cutoff_month
    ].iloc[0]

    columns = [
        "growth_prior_mean",
        "inflation_prior_mean",
        "growth_prior_variance",
        "growth_inflation_prior_covariance",
        "inflation_prior_variance",
        *(f"probability_{regime}" for regime in REGIME_ORDER),
    ]
    np.testing.assert_allclose(
        baseline_row[columns].to_numpy(dtype=float),
        changed_row[columns].to_numpy(dtype=float),
    )


def test_walkforward_history_emits_normalized_priors_after_warmup() -> None:
    scores = _score_history(30)
    mapping = _mapping_history(scores)
    history = causal_var1_prior_history(
        scores,
        mapping,
        build_var_pair_audit(scores),
        minimum_training_pairs=12,
    )
    available = history[history["prior_status"] == "available"]
    probabilities = [f"probability_{regime}" for regime in REGIME_ORDER]

    assert not available.empty
    np.testing.assert_allclose(available[probabilities].sum(axis=1), 1.0)
    assert (available[probabilities] >= 0.0).all().all()
    assert (
        available["fit_latest_destination_month"]
        <= available["source_reference_month"]
    ).all()
    assert (available["training_pairs"] >= 12).all()
    np.testing.assert_allclose(
        available["growth_latent_prior_variance"],
        available["growth_innovation_variance"],
    )
    np.testing.assert_allclose(
        available["growth_inflation_latent_prior_covariance"],
        available["growth_inflation_innovation_covariance"],
    )
    np.testing.assert_allclose(
        available["inflation_latent_prior_variance"],
        available["inflation_innovation_variance"],
    )
    np.testing.assert_allclose(
        available["growth_prior_variance"],
        available["growth_innovation_variance"]
        + available["growth_mapping_proxy_variance"],
    )
    np.testing.assert_allclose(
        available["growth_inflation_prior_covariance"],
        available["growth_inflation_innovation_covariance"]
        + available["growth_inflation_mapping_proxy_covariance"],
    )
    np.testing.assert_allclose(
        available["inflation_prior_variance"],
        available["inflation_innovation_variance"]
        + available["inflation_mapping_proxy_variance"],
    )


@pytest.mark.parametrize(
    "covariance",
    [
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
    ],
)
def test_prior_propagation_rejects_invalid_mapping_proxy_covariance(
    covariance: np.ndarray,
) -> None:
    fit = fit_var1_ols(_simulated_pair_frame(observations=20)[0])
    with pytest.raises(ValueError, match="covariance"):
        propagate_var1_prior(np.zeros(2), covariance, fit)
