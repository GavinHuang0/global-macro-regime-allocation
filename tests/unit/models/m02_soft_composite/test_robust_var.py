"""Test robust Model 02 VAR sensitivities and their audit trail."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.robust_var import (
    RobustVar1Fit,
    fit_var1_huber,
    fit_var1_sensitivity,
    fit_var1_student_t,
    robust_var1_weight_audit,
)
from regime_allocation.models.m02_soft_composite.var_transition import (
    Var1Fit,
    fit_var1_ols,
    propagate_var1_prior,
)


def _simulated_pairs(observations: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(20260722)
    intercept = np.array([0.12, -0.05])
    transition = np.array([[0.62, 0.10], [-0.08, 0.68]])
    innovations = rng.multivariate_normal(
        [0.0, 0.0],
        [[0.14, 0.025], [0.025, 0.08]],
        size=observations,
    )
    states = np.zeros((observations + 1, 2), dtype=float)
    for position in range(observations):
        states[position + 1] = (
            intercept + transition @ states[position] + innovations[position]
        )
    months = pd.date_range("2000-02-01", periods=observations, freq="MS")
    return pd.DataFrame(
        {
            "source_reference_month": months - pd.offsets.MonthBegin(1),
            "destination_reference_month": months,
            "source_growth_score": states[:-1, 0],
            "source_inflation_score": states[:-1, 1],
            "destination_growth_score": states[1:, 0],
            "destination_inflation_score": states[1:, 1],
        }
    )


def _parameter_error(fit: Var1Fit, reference: Var1Fit) -> float:
    return float(
        np.linalg.norm(fit.intercept - reference.intercept)
        + np.linalg.norm(fit.transition - reference.transition)
    )


def _contaminated_pairs() -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    clean = _simulated_pairs()
    contaminated = clean.copy()
    outliers = [55, 143, 241]
    contaminated.loc[outliers, "destination_growth_score"] += [14.0, -18.0, 16.0]
    contaminated.loc[outliers, "destination_inflation_score"] += [-11.0, 13.0, 15.0]
    return clean, contaminated, outliers


def test_infinite_student_t_is_the_unchanged_ols_baseline() -> None:
    pairs = _simulated_pairs(80)
    expected = fit_var1_ols(pairs)

    fit = fit_var1_student_t(pairs, degrees_of_freedom=math.inf)

    assert isinstance(fit, RobustVar1Fit)
    assert fit.estimator == "gaussian_ols"
    assert fit.converged
    np.testing.assert_allclose(fit.intercept, expected.intercept)
    np.testing.assert_allclose(fit.transition, expected.transition)
    np.testing.assert_allclose(
        fit.innovation_covariance,
        expected.innovation_covariance,
    )
    np.testing.assert_array_equal(fit.event_weights, 1.0)
    assert fit.effective_sample_size == pytest.approx(len(pairs))
    assert fit.downweighted_pairs == 0


def test_huber_irls_downweights_joint_outliers_and_limits_parameter_damage() -> None:
    clean, contaminated, outliers = _contaminated_pairs()
    reference = fit_var1_ols(clean)
    contaminated_ols = fit_var1_ols(contaminated)

    fit = fit_var1_huber(
        contaminated,
        threshold=2.5,
        maximum_iterations=300,
        tolerance=1.0e-10,
    )

    assert fit.converged
    assert fit.estimator == "huber_irls"
    assert fit.downweighted_pairs > 0
    assert max(fit.event_weights[outliers]) < 0.1
    assert _parameter_error(fit, reference) < _parameter_error(
        fit_var1_sensitivity(contaminated, estimator="ols"),
        reference,
    )
    assert _parameter_error(fit, reference) < float(
        np.linalg.norm(contaminated_ols.intercept - reference.intercept)
        + np.linalg.norm(contaminated_ols.transition - reference.transition)
    )
    assert np.linalg.eigvalsh(fit.innovation_covariance).min() > 0.0


def test_student_t_irls_exposes_scale_and_gaussian_plugin_covariance() -> None:
    clean, contaminated, outliers = _contaminated_pairs()
    reference = fit_var1_ols(clean)

    fit = fit_var1_student_t(
        contaminated,
        degrees_of_freedom=7.0,
        maximum_iterations=300,
        tolerance=1.0e-10,
    )

    assert fit.converged
    assert fit.estimator == "student_t_irls"
    assert fit.gaussian_covariance_multiplier == pytest.approx(7.0 / 5.0)
    np.testing.assert_allclose(
        fit.innovation_covariance,
        fit.working_scale_matrix * (7.0 / 5.0),
    )
    assert max(fit.event_weights[outliers]) < 0.02
    assert _parameter_error(fit, reference) < _parameter_error(
        fit_var1_sensitivity(contaminated, estimator="ols"),
        reference,
    )
    assert fit.minimum_weight == pytest.approx(float(fit.event_weights.min()))
    assert fit.maximum_weight == pytest.approx(float(fit.event_weights.max()))
    assert 0.0 < fit.effective_sample_size <= len(contaminated)


def test_robust_fit_uses_existing_gaussian_prior_contract() -> None:
    fit = fit_var1_student_t(_simulated_pairs(100), degrees_of_freedom=5.0)
    score = np.array([0.2, -0.1])
    mapping = np.array([[0.3, 0.04], [0.04, 0.2]])

    mean, latent_covariance, reporting_covariance = propagate_var1_prior(
        score,
        mapping,
        fit,
    )

    np.testing.assert_allclose(mean, fit.intercept + fit.transition @ score)
    np.testing.assert_allclose(latent_covariance, fit.innovation_covariance)
    np.testing.assert_allclose(
        reporting_covariance,
        fit.innovation_covariance + mapping,
    )


def test_weight_audit_retains_pair_lineage_and_reconstructs_residuals() -> None:
    pairs = _simulated_pairs(60)
    fit = fit_var1_huber(pairs)

    audit = robust_var1_weight_audit(pairs, fit)

    assert audit["destination_reference_month"].equals(
        pairs["destination_reference_month"]
    )
    np.testing.assert_allclose(audit["event_weight"], fit.event_weights)
    np.testing.assert_allclose(
        audit["squared_mahalanobis_distance"],
        fit.squared_mahalanobis_distances,
    )
    np.testing.assert_allclose(
        audit["destination_growth_score"],
        audit["fitted_destination_growth_score"] + audit["growth_innovation"],
    )
    np.testing.assert_allclose(
        audit["destination_inflation_score"],
        audit["fitted_destination_inflation_score"]
        + audit["inflation_innovation"],
    )
    assert set(audit["var_estimator"]) == {"huber_irls"}


@pytest.mark.parametrize("degrees_of_freedom", [2.0, 1.0, np.nan, -np.inf])
def test_student_t_rejects_nonfinite_covariance_degrees_of_freedom(
    degrees_of_freedom: float,
) -> None:
    with pytest.raises(ValueError, match="exceed two"):
        fit_var1_student_t(
            _simulated_pairs(20),
            degrees_of_freedom=degrees_of_freedom,
        )


def test_dispatch_rejects_unknown_estimator() -> None:
    with pytest.raises(ValueError, match="ols, huber, student_t"):
        fit_var1_sensitivity(
            _simulated_pairs(20),
            estimator="unsupported",  # type: ignore[arg-type]
        )


def test_iteration_limit_must_be_an_integer() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        fit_var1_huber(
            _simulated_pairs(20),
            maximum_iterations=2.5,  # type: ignore[arg-type]
        )
