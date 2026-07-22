"""Test causal linear-Gaussian emission estimation for Model 02."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.gaussian_emissions import (
    LinearGaussianEmissionSpec,
    fit_linear_gaussian_emission,
    rolling_origin_residuals,
    select_causal_emission_rows,
)
from regime_allocation.models.m02_soft_composite.joint_filter import (
    JointGaussianState,
    update_joint_gaussian,
)


def _emission_table(observations: int = 72, *, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(observations, 2))
    rate_control = rng.normal(size=observations)
    noise = rng.multivariate_normal(
        np.zeros(2), [[0.12, 0.05], [0.05, 0.20]], size=observations
    )
    responses = (
        np.asarray([0.3, -0.2])
        + states @ np.asarray([[0.9, -0.4], [0.5, 0.7]])
        + rate_control[:, None] @ np.asarray([[0.25, -0.35]])
        + noise
    )
    return pd.DataFrame(
        {
            "training_available_at": pd.date_range(
                "2010-01-15", periods=observations, freq="MS"
            ),
            "growth_score": states[:, 0],
            "inflation_score": states[:, 1],
            "rate_control": rate_control,
            "activity": responses[:, 0],
            "prices": responses[:, 1],
        }
    )


def _spec(
    *,
    penalties: tuple[tuple[float, float], ...] = ((1.0, 8.0), (1.0, 1.0)),
    restrictions: tuple[tuple[bool, bool], ...] = ((False, True), (False, False)),
    lambda_grid: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0),
) -> LinearGaussianEmissionSpec:
    return LinearGaussianEmissionSpec(
        block_id="test_block",
        response_names=("activity", "prices"),
        control_names=("rate_control",),
        state_loading_penalties=penalties,
        exact_zero_mask=restrictions,
        lambda_grid=lambda_grid,
        minimum_training_samples=30,
        validation_minimum_training_samples=15,
        minimum_validation_observations=10,
        covariance_eigenvalue_floor=1.0e-7,
    )


def test_exact_zero_restriction_and_fit_audit_are_preserved() -> None:
    fit = fit_linear_gaussian_emission(
        _emission_table(),
        spec=_spec(),
        availability_column="training_available_at",
    )

    assert fit.state_loadings.loc["activity", "inflation_score"] == 0.0
    assert fit.state_loadings.loc["prices", "inflation_score"] != 0.0
    assert fit.control_loadings.shape == (2, 1)
    assert fit.selected_base_lambda in fit.spec.lambda_grid
    audit = fit.to_audit_dict()
    json.dumps(audit)
    assert audit["rolling_origin_rule"] == (
        "equal availability timestamps held out together"
    )
    assert audit["specification"]["exact_zero_mask"][0][1] is True


def test_larger_structured_ridge_penalty_shrinks_state_loadings() -> None:
    table = _emission_table(observations=80)
    weak = fit_linear_gaussian_emission(
        table,
        spec=_spec(
            penalties=((1.0, 1.0), (1.0, 1.0)),
            restrictions=((False, False), (False, False)),
            lambda_grid=(100.0,),
        ),
        availability_column="training_available_at",
    )
    strong = fit_linear_gaussian_emission(
        table,
        spec=_spec(
            penalties=((1.0, 100.0), (1.0, 1.0)),
            restrictions=((False, False), (False, False)),
            lambda_grid=(100.0,),
        ),
        availability_column="training_available_at",
    )

    assert abs(strong.state_loadings.loc["activity", "inflation_score"]) < abs(
        weak.state_loadings.loc["activity", "inflation_score"]
    )
    assert abs(strong.state_loadings.loc["activity", "inflation_score"]) < 0.02


def test_future_rows_cannot_change_lambda_or_fit_at_an_earlier_cutoff() -> None:
    table = _emission_table(observations=80)
    cutoff = pd.Timestamp(table.loc[60, "training_available_at"])
    baseline = fit_linear_gaussian_emission(
        table,
        spec=_spec(),
        availability_column="training_available_at",
        knowledge_cutoff=cutoff,
    )
    changed = table.copy()
    future = changed["training_available_at"] >= cutoff
    changed.loc[future, ["activity", "prices"]] = 1_000_000.0
    changed.loc[future, ["growth_score", "inflation_score"]] = -1_000_000.0
    refit = fit_linear_gaussian_emission(
        changed,
        spec=_spec(),
        availability_column="training_available_at",
        knowledge_cutoff=cutoff,
    )

    assert refit.selected_base_lambda == baseline.selected_base_lambda
    pd.testing.assert_frame_equal(refit.state_loadings, baseline.state_loadings)
    pd.testing.assert_frame_equal(
        refit.residual_covariance, baseline.residual_covariance
    )
    pd.testing.assert_frame_equal(
        refit.lambda_selection, baseline.lambda_selection
    )
    assert refit.last_training_available_at < cutoff


def test_ledoit_wolf_covariance_is_positive_definite_after_floor() -> None:
    table = _emission_table()
    table["prices"] = table["activity"]
    fit = fit_linear_gaussian_emission(
        table,
        spec=_spec(restrictions=((False, False), (False, False))),
        availability_column="training_available_at",
    )
    eigenvalues = np.linalg.eigvalsh(fit.residual_covariance.to_numpy())

    assert eigenvalues.min() >= fit.spec.covariance_eigenvalue_floor * (1.0 - 1e-8)
    assert fit.covariance_minimum_eigenvalue_after_floor > 0.0
    assert 0.0 <= fit.ledoit_wolf_shrinkage <= 1.0


def test_partial_observation_updates_only_the_observed_subvector() -> None:
    fit = fit_linear_gaussian_emission(
        _emission_table(),
        spec=_spec(),
        availability_column="training_available_at",
    )
    update = fit.update_gaussian_state(
        [0.1, -0.1],
        [[0.8, 0.1], [0.1, 0.6]],
        {"activity": 0.7, "prices": np.nan},
        controls={"rate_control": 0.2},
    )

    assert update.observed_names == ("activity",)
    assert update.omitted_names == ("prices",)
    assert update.predictive_covariance.shape == (1, 1)
    assert update.kalman_gain.shape == (2, 1)
    assert np.linalg.det(update.posterior_covariance) < np.linalg.det(
        update.prior_covariance
    )
    assert np.isfinite(update.log_predictive_density)
    json.dumps(update.to_metadata_dict())


def test_observed_system_plugs_directly_into_joint_filter() -> None:
    fit = fit_linear_gaussian_emission(
        _emission_table(),
        spec=_spec(),
        availability_column="training_available_at",
    )
    observation = {"activity": 0.7, "prices": np.nan}
    controls = {"rate_control": 0.2}
    prior_mean = np.asarray([0.1, -0.1])
    prior_covariance = np.asarray([[0.8, 0.1], [0.1, 0.6]])
    local = fit.update_gaussian_state(
        prior_mean,
        prior_covariance,
        observation,
        controls=controls,
    )
    joint_mean = np.zeros(8)
    joint_mean[4:6] = prior_mean
    joint_covariance = np.eye(8)
    joint_covariance[4:6, 4:6] = prior_covariance
    joint = JointGaussianState(
        tuple(pd.date_range("2024-01-01", periods=4, freq="MS")),
        joint_mean,
        joint_covariance,
    )
    system = fit.observed_system(observation, controls=controls)
    joint_update = update_joint_gaussian(
        joint,
        "2024-03-01",
        **system.to_joint_filter_inputs(),
    )
    joint_posterior_mean, joint_posterior_covariance = joint_update.state.marginal(
        "2024-03-01"
    )

    assert system.state_loadings.shape == (1, 2)
    assert system.residual_covariance.shape == (1, 1)
    np.testing.assert_allclose(joint_posterior_mean, local.posterior_mean)
    np.testing.assert_allclose(joint_posterior_covariance, local.posterior_covariance)
    assert joint_update.log_predictive_density == pytest.approx(
        local.log_predictive_density
    )


def test_lambda_ties_are_resolved_toward_stronger_regularization() -> None:
    table = _emission_table(observations=55)
    spec = LinearGaussianEmissionSpec(
        block_id="all_restricted",
        response_names=("activity",),
        exact_zero_mask=((True, True),),
        lambda_grid=(0.0, 0.5, 5.0, 50.0),
        minimum_training_samples=25,
        validation_minimum_training_samples=12,
        minimum_validation_observations=8,
    )
    fit = fit_linear_gaussian_emission(
        table,
        spec=spec,
        availability_column="training_available_at",
    )

    assert fit.selected_base_lambda == 50.0
    assert int(fit.lambda_selection["selected"].sum()) == 1


def test_rolling_residuals_hold_equal_timestamps_out_together() -> None:
    table = _emission_table(observations=45)
    duplicate = table.iloc[[30]].copy()
    duplicate[["activity", "prices"]] += 0.1
    table = pd.concat([table, duplicate], ignore_index=True)
    origin = pd.Timestamp(duplicate.iloc[0]["training_available_at"])
    residuals = rolling_origin_residuals(
        table,
        spec=_spec(lambda_grid=(1.0,)),
        availability_column="training_available_at",
        base_lambda=1.0,
    )
    held_out = residuals["validation_available_at"] == origin

    assert int(held_out.sum()) == 2
    assert set(residuals.loc[held_out, "training_rows"]) == {30}


def test_rolling_residuals_use_fold_specific_standardization_and_response_order() -> None:
    table = _emission_table(observations=52)
    residuals = rolling_origin_residuals(
        table,
        spec=_spec(lambda_grid=(1.0,)),
        availability_column="training_available_at",
        base_lambda=1.0,
    )

    expected_suffixes = [
        "actual_activity",
        "residual_activity",
        "predictive_marginal_sd_activity",
        "standardized_residual_activity",
        "cholesky_whitened_residual_activity",
        "actual_prices",
        "residual_prices",
        "predictive_marginal_sd_prices",
        "standardized_residual_prices",
        "cholesky_whitened_residual_prices",
    ]
    assert [column for column in residuals if column in expected_suffixes] == expected_suffixes
    for response in ("activity", "prices"):
        np.testing.assert_allclose(
            residuals[f"standardized_residual_{response}"],
            residuals[f"residual_{response}"]
            / residuals[f"predictive_marginal_sd_{response}"],
        )
        assert (residuals[f"predictive_marginal_sd_{response}"] > 0.0).all()
    # The first Cholesky coordinate is exactly its marginal standardization;
    # later coordinates additionally remove covariance with earlier responses.
    np.testing.assert_allclose(
        residuals["cholesky_whitened_residual_activity"],
        residuals["standardized_residual_activity"],
    )


def test_constant_control_is_dropped_per_fold_then_activates_causally() -> None:
    table = _emission_table(observations=60)
    table["rate_control"] = 0.0
    table.loc[40:, "rate_control"] = 1.0
    residuals = rolling_origin_residuals(
        table,
        spec=_spec(lambda_grid=(1.0,)),
        availability_column="training_available_at",
        base_lambda=1.0,
    )

    first_active_row = pd.Timestamp(table.loc[40, "training_available_at"])
    next_row = pd.Timestamp(table.loc[41, "training_available_at"])
    before_variation = residuals.loc[
        residuals["validation_available_at"] == first_active_row
    ].iloc[0]
    after_variation = residuals.loc[
        residuals["validation_available_at"] == next_row
    ].iloc[0]
    assert before_variation["dropped_constant_controls"] == "rate_control"
    assert after_variation["dropped_constant_controls"] == ""


def test_causal_selection_audits_incomplete_and_cutoff_rows() -> None:
    table = _emission_table(observations=40)
    table.loc[3, "prices"] = np.nan
    cutoff = pd.Timestamp(table.loc[35, "training_available_at"])
    selection = select_causal_emission_rows(
        table,
        spec=_spec(),
        availability_column="training_available_at",
        knowledge_cutoff=cutoff,
    )

    assert selection.diagnostics["excluded_incomplete_model_vector"] == 1
    assert (
        selection.diagnostics["excluded_not_available_strictly_before_cutoff"]
        == 5
    )
    assert selection.eligible["training_available_at"].max() < cutoff


def test_spec_rejects_nonincreasing_lambda_grid_and_bad_restriction_shape() -> None:
    with pytest.raises(ValueError, match="lambda_grid"):
        _spec(lambda_grid=(0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="exact_zero_mask"):
        LinearGaussianEmissionSpec(
            block_id="bad",
            response_names=("activity", "prices"),
            exact_zero_mask=((False, True),),
        )


def test_constant_control_is_dropped_in_final_fit_and_audited() -> None:
    table = _emission_table()
    table["rate_control"] = 1.0

    fit = fit_linear_gaussian_emission(
        table,
        spec=_spec(),
        availability_column="training_available_at",
    )

    assert fit.dropped_constant_controls == ("rate_control",)
    np.testing.assert_allclose(fit.control_loadings["rate_control"], 0.0)
    assert fit.to_audit_dict()["dropped_constant_controls"] == ["rate_control"]
