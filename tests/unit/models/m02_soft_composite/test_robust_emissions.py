"""Test opt-in robust Student-t emissions and Gaussian moment updates."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, multivariate_t

from regime_allocation.models.m02_soft_composite.gaussian_emissions import (
    LinearGaussianEmissionSpec,
    fit_linear_gaussian_emission,
)
from regime_allocation.models.m02_soft_composite.joint_filter import (
    JointGaussianState,
    update_joint_gaussian,
)
from regime_allocation.models.m02_soft_composite.robust_emissions import (
    DEFAULT_DEGREES_OF_FREEDOM_GRID,
    fit_linear_student_t_emission,
    fit_linear_student_t_emission_fixed,
    multivariate_student_t_nll,
    student_t_event_weight,
)


def _table(observations: int = 52, *, outlier: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(901)
    states = rng.normal(size=(observations, 2))
    control = rng.normal(size=observations)
    noise = rng.normal(scale=0.22, size=(observations, 2))
    response = (
        np.asarray([0.2, -0.1])
        + states @ np.asarray([[1.1, -0.3], [0.4, 0.8]])
        + control[:, None] @ np.asarray([[0.25, -0.2]])
        + noise
    )
    if outlier:
        response[35] += np.asarray([25.0, -18.0])
    return pd.DataFrame(
        {
            "training_available_at": pd.date_range(
                "2012-01-01", periods=observations, freq="MS"
            ),
            "growth_score": states[:, 0],
            "inflation_score": states[:, 1],
            "control": control,
            "activity": response[:, 0],
            "prices": response[:, 1],
        }
    )


def _spec(
    *,
    lambda_grid: tuple[float, ...] = (0.1,),
    restrictions: tuple[tuple[bool, bool], ...] = (
        (False, True),
        (False, False),
    ),
) -> LinearGaussianEmissionSpec:
    return LinearGaussianEmissionSpec(
        block_id="robust_test",
        response_names=("activity", "prices"),
        control_names=("control",),
        state_loading_penalties=((1.0, 20.0), (1.0, 1.0)),
        exact_zero_mask=restrictions,
        lambda_grid=lambda_grid,
        minimum_training_samples=28,
        validation_minimum_training_samples=18,
        minimum_validation_observations=8,
        covariance_eigenvalue_floor=1.0e-8,
    )


def test_student_t_density_matches_scipy_and_gaussian_limit() -> None:
    residual = np.asarray([0.8, -1.2])
    scale = np.asarray([[1.4, 0.25], [0.25, 0.9]])

    actual = -float(multivariate_student_t_nll(residual, scale, 7.0))
    expected = float(multivariate_t.logpdf(residual, shape=scale, df=7.0))
    gaussian = -float(multivariate_student_t_nll(residual, scale, math.inf))

    assert actual == pytest.approx(expected, abs=1.0e-12)
    assert gaussian == pytest.approx(
        float(multivariate_normal.logpdf(residual, cov=scale)), abs=1.0e-12
    )


def test_irls_downweights_outlier_and_preserves_exact_zero_mask() -> None:
    table = _table(outlier=True)
    fit = fit_linear_student_t_emission(
        table,
        spec=_spec(),
        availability_column="training_available_at",
        degrees_of_freedom_grid=(7.0,),
        maximum_iterations=80,
    )

    assert fit.selected_degrees_of_freedom == 7.0
    assert fit.state_loadings.loc["activity", "inflation_score"] == 0.0
    assert fit.irls_converged
    assert fit.training_weights.iloc[35] < 0.05
    assert fit.training_weights.median() > fit.training_weights.iloc[35]
    assert np.linalg.eigvalsh(fit.residual_scale.to_numpy()).min() > 0.0


def test_robust_fit_is_unchanged_by_rows_not_known_at_cutoff() -> None:
    table = _table(observations=58, outlier=True)
    cutoff = pd.Timestamp(table.loc[48, "training_available_at"])
    baseline = fit_linear_student_t_emission(
        table,
        spec=_spec(),
        availability_column="training_available_at",
        knowledge_cutoff=cutoff,
        degrees_of_freedom_grid=(7.0,),
    )
    changed = table.copy()
    future = changed["training_available_at"] >= cutoff
    changed.loc[future, ["activity", "prices"]] = 1.0e9
    changed.loc[future, ["growth_score", "inflation_score"]] = -1.0e9
    refit = fit_linear_student_t_emission(
        changed,
        spec=_spec(),
        availability_column="training_available_at",
        knowledge_cutoff=cutoff,
        degrees_of_freedom_grid=(7.0,),
    )

    pd.testing.assert_frame_equal(refit.state_loadings, baseline.state_loadings)
    pd.testing.assert_frame_equal(refit.residual_scale, baseline.residual_scale)
    pd.testing.assert_frame_equal(
        refit.hyperparameter_selection, baseline.hyperparameter_selection
    )
    assert refit.last_training_available_at < cutoff


def test_heavier_tail_grid_is_scored_causally_and_audit_is_strict_json() -> None:
    fit = fit_linear_student_t_emission(
        _table(observations=40, outlier=True),
        spec=_spec(restrictions=((True, True), (True, True))),
        availability_column="training_available_at",
        degrees_of_freedom_grid=DEFAULT_DEGREES_OF_FREEDOM_GRID,
        maximum_iterations=50,
    )

    selection = fit.hyperparameter_selection
    assert len(selection) == len(DEFAULT_DEGREES_OF_FREEDOM_GRID)
    assert int(selection["selected"].sum()) == 1
    assert set(selection["degrees_of_freedom"]) == set(
        DEFAULT_DEGREES_OF_FREEDOM_GRID
    )
    assert (selection["validation_observations"] >= 8).all()
    json.dumps(fit.to_audit_dict(), allow_nan=False)


def test_event_weight_and_robust_update_limit_an_extreme_release() -> None:
    fit = fit_linear_student_t_emission(
        _table(),
        spec=_spec(),
        availability_column="training_available_at",
        degrees_of_freedom_grid=(7.0,),
    )
    prior_mean = np.zeros(2)
    prior_covariance = np.eye(2)
    controls = {"control": 0.0}
    moderate = fit.update_gaussian_state(
        prior_mean,
        prior_covariance,
        {"activity": 0.2, "prices": -0.1},
        controls=controls,
    )
    extreme = fit.update_gaussian_state(
        prior_mean,
        prior_covariance,
        {"activity": 30.0, "prices": -25.0},
        controls=controls,
    )

    assert extreme.event_weight < 0.01
    assert extreme.event_weight < moderate.event_weight
    assert np.linalg.norm(extreme.kalman_gain) < np.linalg.norm(moderate.kalman_gain)
    metadata = extreme.to_metadata_dict()
    assert metadata["event_weight"] == pytest.approx(extreme.event_weight)
    json.dumps(metadata, allow_nan=False)


def test_adaptive_system_plugs_into_existing_joint_gaussian_filter() -> None:
    fit = fit_linear_student_t_emission(
        _table(),
        spec=_spec(),
        availability_column="training_available_at",
        degrees_of_freedom_grid=(7.0,),
    )
    observation = {"activity": 0.7, "prices": np.nan}
    controls = {"control": 0.2}
    prior_mean = np.asarray([0.1, -0.1])
    prior_covariance = np.asarray([[0.8, 0.1], [0.1, 0.6]])
    local = fit.update_gaussian_state(
        prior_mean, prior_covariance, observation, controls=controls
    )
    system = fit.observed_system(observation, controls=controls)
    approximation = system.approximate_gaussian_system(
        prior_mean, prior_covariance
    )
    joint_mean = np.zeros(8)
    joint_mean[4:6] = prior_mean
    joint_covariance = np.eye(8)
    joint_covariance[4:6, 4:6] = prior_covariance
    state = JointGaussianState(
        tuple(pd.date_range("2024-01-01", periods=4, freq="MS")),
        joint_mean,
        joint_covariance,
    )
    joint = update_joint_gaussian(
        state,
        "2024-03-01",
        **approximation.to_joint_filter_inputs(),
    )
    joint_mean_after, joint_covariance_after = joint.state.marginal("2024-03-01")

    np.testing.assert_allclose(joint_mean_after, local.posterior_mean)
    np.testing.assert_allclose(joint_covariance_after, local.posterior_covariance)
    assert approximation.event_weight == pytest.approx(local.event_weight)


def test_infinite_nu_nested_model_matches_gaussian_fit_and_update() -> None:
    table = _table()
    spec = _spec()
    gaussian = fit_linear_gaussian_emission(
        table, spec=spec, availability_column="training_available_at"
    )
    robust = fit_linear_student_t_emission(
        table,
        spec=spec,
        availability_column="training_available_at",
        degrees_of_freedom_grid=(math.inf,),
    )

    np.testing.assert_allclose(robust.intercept, gaussian.intercept)
    np.testing.assert_allclose(robust.state_loadings, gaussian.state_loadings)
    np.testing.assert_allclose(
        robust.residual_scale, gaussian.residual_covariance
    )
    observation = {"activity": 0.8, "prices": -0.4}
    controls = {"control": 0.2}
    gaussian_update = gaussian.update_gaussian_state(
        [0.1, -0.2], [[0.9, 0.1], [0.1, 0.7]], observation, controls=controls
    )
    robust_update = robust.update_gaussian_state(
        [0.1, -0.2], [[0.9, 0.1], [0.1, 0.7]], observation, controls=controls
    )
    assert robust_update.event_weight == 1.0
    np.testing.assert_allclose(
        robust_update.posterior_mean, gaussian_update.posterior_mean
    )
    np.testing.assert_allclose(
        robust_update.posterior_covariance, gaussian_update.posterior_covariance
    )


def test_robust_update_accepts_fully_exact_zero_covariance_state() -> None:
    fit = fit_linear_student_t_emission_fixed(
        _table(),
        spec=_spec(),
        availability_column="training_available_at",
        base_lambda=0.1,
        degrees_of_freedom=7.0,
    )
    prior_mean = np.asarray([0.3, -0.2])
    prior_covariance = np.zeros((2, 2))
    update = fit.update_gaussian_state(
        prior_mean,
        prior_covariance,
        {"activity": 25.0, "prices": -20.0},
        controls={"control": 0.0},
    )

    np.testing.assert_allclose(update.posterior_mean, prior_mean)
    np.testing.assert_allclose(update.posterior_covariance, prior_covariance)
    np.testing.assert_allclose(update.kalman_gain, 0.0)
    assert 0.0 < update.event_weight < 1.0
    assert np.isfinite(update.approximate_log_predictive_density)


def test_robust_update_preserves_one_exact_axis_in_psd_state() -> None:
    fit = fit_linear_student_t_emission_fixed(
        _table(),
        spec=_spec(),
        availability_column="training_available_at",
        base_lambda=0.1,
        degrees_of_freedom=7.0,
    )
    prior_covariance = np.asarray([[0.0, 0.0], [0.0, 0.6]])
    update = fit.update_gaussian_state(
        [0.3, -0.2],
        prior_covariance,
        {"activity": 0.8, "prices": -0.4},
        controls={"control": 0.1},
    )

    assert update.posterior_mean[0] == pytest.approx(0.3)
    np.testing.assert_allclose(update.posterior_covariance[0, :], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(update.posterior_covariance[:, 0], 0.0, atol=1.0e-14)
    assert 0.0 <= update.posterior_covariance[1, 1] < 0.6
    assert np.linalg.eigvalsh(update.posterior_covariance).min() >= -1.0e-12


def test_event_weight_formula_and_invalid_degrees_of_freedom() -> None:
    residual = np.asarray([3.0, 4.0])
    scale = np.eye(2)

    assert student_t_event_weight(residual, scale, 7.0) == pytest.approx(9.0 / 32.0)
    assert student_t_event_weight(residual, scale, math.inf) == 1.0
    with pytest.raises(ValueError, match="exceed two"):
        student_t_event_weight(residual, scale, 2.0)
    with pytest.raises(ValueError, match="duplicates"):
        fit_linear_student_t_emission(
            _table(),
            spec=_spec(),
            availability_column="training_available_at",
            degrees_of_freedom_grid=(7.0, 7.0),
        )


def test_fixed_refit_matches_single_candidate_tuned_fit_without_retuning() -> None:
    table = _table(outlier=True)
    spec = _spec()
    tuned = fit_linear_student_t_emission(
        table,
        spec=spec,
        availability_column="training_available_at",
        degrees_of_freedom_grid=(7.0,),
    )
    fixed = fit_linear_student_t_emission_fixed(
        table,
        spec=spec,
        availability_column="training_available_at",
        base_lambda=0.1,
        degrees_of_freedom=7.0,
    )

    pd.testing.assert_series_equal(fixed.intercept, tuned.intercept)
    pd.testing.assert_frame_equal(fixed.state_loadings, tuned.state_loadings)
    pd.testing.assert_frame_equal(fixed.residual_scale, tuned.residual_scale)
    assert fixed.hyperparameter_selection_method == (
        "fixed_without_rolling_origin_retuning"
    )
    assert fixed.hyperparameter_selection.to_dict(orient="records") == [
        {
            "base_lambda": 0.1,
            "degrees_of_freedom": 7.0,
            "selected": True,
            "selection_method": "fixed_without_rolling_origin_retuning",
            "validation_performed": False,
        }
    ]
    audit = fixed.to_audit_dict()
    assert audit["rolling_origin_rule"] == (
        "not_applicable_hyperparameters_fixed_before_refit"
    )
    json.dumps(audit, allow_nan=False)


def test_fixed_refit_does_not_require_rolling_validation_sample() -> None:
    table = _table(observations=30)
    spec = LinearGaussianEmissionSpec(
        block_id="fixed_without_validation",
        response_names=("activity", "prices"),
        control_names=("control",),
        state_loading_penalties=((1.0, 1.0), (1.0, 1.0)),
        lambda_grid=(1.0,),
        minimum_training_samples=25,
        validation_minimum_training_samples=29,
        minimum_validation_observations=10,
    )

    fixed = fit_linear_student_t_emission_fixed(
        table,
        spec=spec,
        availability_column="training_available_at",
        base_lambda=1.0,
        degrees_of_freedom=7.0,
    )

    assert fixed.training_count == 30
    assert fixed.hyperparameter_selection["validation_performed"].eq(False).all()
    with pytest.raises(ValueError, match="too few rolling-origin"):
        fit_linear_student_t_emission(
            table,
            spec=spec,
            availability_column="training_available_at",
            degrees_of_freedom_grid=(7.0,),
        )


def test_fixed_refit_applies_strict_cutoff_and_rejects_unregistered_lambda() -> None:
    table = _table(observations=45)
    cutoff = pd.Timestamp(table.loc[38, "training_available_at"])
    fit = fit_linear_student_t_emission_fixed(
        table,
        spec=_spec(),
        availability_column="training_available_at",
        knowledge_cutoff=cutoff,
        base_lambda=0.1,
        degrees_of_freedom=7.0,
    )

    assert fit.training_count == 38
    assert fit.last_training_available_at < cutoff
    assert fit.selection_diagnostics[
        "excluded_not_available_strictly_before_cutoff"
    ] == 7
    with pytest.raises(ValueError, match="frozen lambda_grid"):
        fit_linear_student_t_emission_fixed(
            table,
            spec=_spec(),
            availability_column="training_available_at",
            base_lambda=3.0,
            degrees_of_freedom=7.0,
        )
