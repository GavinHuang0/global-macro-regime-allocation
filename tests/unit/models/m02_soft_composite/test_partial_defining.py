"""Test causal partial score-defining releases for Model 02.

The tests verify artifact preparation, strict fit cutoffs, the exact
equal-weight score identity, sequential conditional-Gaussian factorization,
duplicate-information rejection, and exact one- or two-axis conditioning in
the rolling joint Gaussian filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.joint_filter import (
    JointGaussianState,
    VarDynamics,
    condition_on_exact_score,
    initialize_joint_gaussian,
)
from regime_allocation.models.m02_soft_composite.partial_defining import (
    COMPONENT_WEIGHT,
    ConditionalComponentEmission,
    apply_partial_defining_event,
    condition_on_exact_score_axes,
    conditional_component_emission,
    fit_causal_component_gaussian,
    prepare_partial_defining_data,
    score_weight_matrix,
)
from regime_allocation.models.m02_soft_composite.scores import (
    ALL_COMPONENTS,
    GROWTH_COMPONENTS,
    INFLATION_COMPONENTS,
)


def _artifact_frames(months: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_months = pd.date_range("2018-01-01", periods=months, freq="MS")
    rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for month_number, reference_month in enumerate(reference_months):
        component_values = {
            component: (
                np.sin((month_number + 1) * (component_number + 1) / 7.0)
                + 0.08 * month_number
                + 0.03 * component_number
            )
            for component_number, component in enumerate(ALL_COMPONENTS)
        }
        release_offsets = {
            "payrolls": 35,
            "unemployment_rate": 35,
            "average_hourly_earnings": 35,
            "core_cpi": 42,
            "producer_prices": 43,
            "industrial_production": 48,
            "consumer_activity": 58,
            "core_pce": 58,
        }
        for component, value in component_values.items():
            rows.append(
                {
                    "reference_month": reference_month,
                    "component": component,
                    "release_date": reference_month + pd.Timedelta(
                        days=release_offsets[component]
                    ),
                    "transformed_value": value * 0.2,
                }
            )
        score_row: dict[str, object] = {
            "reference_month": reference_month,
            "growth_score": COMPONENT_WEIGHT
            * sum(component_values[name] for name in GROWTH_COMPONENTS),
            "inflation_score": COMPONENT_WEIGHT
            * sum(component_values[name] for name in INFLATION_COMPONENTS),
            "score_available_at": reference_month + pd.Timedelta(days=58),
        }
        for component, value in component_values.items():
            score_row[f"{component}_z"] = value
            score_row[f"{component}_transformed"] = value * 0.2
        score_rows.append(score_row)
    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def _joint_state() -> JointGaussianState:
    edge = VarDynamics(
        intercept=np.array([0.0, 0.0]),
        transition=np.array([[0.75, 0.1], [0.05, 0.7]]),
        innovation_covariance=np.array([[0.5, 0.08], [0.08, 0.4]]),
    )
    return initialize_joint_gaussian(
        "2021-01-01",
        np.array([0.0, 0.0]),
        np.array([[0.8, 0.1], [0.1, 0.7]]),
        [edge, edge, edge],
    )


def _fit() -> object:
    releases, scores = _artifact_frames(60)
    prepared = prepare_partial_defining_data(releases, scores)
    return fit_causal_component_gaussian(
        prepared.component_history,
        "2023-02-01",
        minimum_training_samples=9,
    )


def test_weight_matrix_encodes_four_fixed_quarters_per_axis() -> None:
    weights = score_weight_matrix()

    assert weights.shape == (2, 8)
    np.testing.assert_allclose(weights.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(weights[0, :4], COMPONENT_WEIGHT)
    np.testing.assert_allclose(weights[0, 4:], 0.0)
    np.testing.assert_allclose(weights[1, :4], 0.0)
    np.testing.assert_allclose(weights[1, 4:], COMPONENT_WEIGHT)


def test_preparation_builds_atomic_events_and_exact_contributions() -> None:
    releases, scores = _artifact_frames(12)

    prepared = prepare_partial_defining_data(releases, scores)

    first_month = pd.Timestamp("2018-01-01")
    employment = prepared.events.loc[
        prepared.events["reference_month"].eq(first_month)
        & prepared.events["release_family"].eq("employment_situation")
    ]
    assert employment["event_id"].nunique() == 1
    assert set(employment["event_component_count"]) == {3}
    assert set(employment["component"]) == {
        "payrolls",
        "unemployment_rate",
        "average_hourly_earnings",
    }
    np.testing.assert_allclose(
        prepared.events["weighted_contribution"],
        COMPONENT_WEIGHT * prepared.events["component_z"],
    )
    history = prepared.component_history.iloc[0]
    assert history["growth_score"] == pytest.approx(
        COMPONENT_WEIGHT * sum(history[name] for name in GROWTH_COMPONENTS)
    )


def test_preparation_rejects_artifact_disagreement() -> None:
    releases, scores = _artifact_frames(12)
    releases.loc[0, "transformed_value"] += 1.0

    with pytest.raises(ValueError, match="artifacts disagree"):
        prepare_partial_defining_data(releases, scores)


def test_component_fit_uses_strict_score_availability_cutoff() -> None:
    releases, scores = _artifact_frames(20)
    prepared = prepare_partial_defining_data(releases, scores)
    cutoff = prepared.component_history.iloc[12]["training_available_at"]

    fit = fit_causal_component_gaussian(
        prepared.component_history,
        cutoff,
        minimum_training_samples=9,
    )

    assert fit.sample_size == 12
    assert fit.latest_training_availability < pd.Timestamp(cutoff)
    assert fit.last_reference_month == pd.Timestamp("2018-12-01")
    assert np.linalg.eigvalsh(fit.covariance).min() > 0.0


def test_conditional_emission_dimensions_and_offsets_are_explicit() -> None:
    fit = _fit()

    emission = conditional_component_emission(
        fit,
        ("core_cpi", "producer_prices"),
        ("payrolls", "unemployment_rate"),
    )

    assert isinstance(emission, ConditionalComponentEmission)
    assert emission.score_loadings.shape == (2, 2)
    assert emission.conditioned_loadings.shape == (2, 2)
    assert emission.noise_covariance.shape == (2, 2)
    assert np.linalg.eigvalsh(emission.noise_covariance).min() > 0.0
    offset = emission.offset({"payrolls": 0.3, "unemployment_rate": -0.2})
    assert offset.shape == (2,)
    with pytest.raises(ValueError, match="do not match"):
        emission.offset({"payrolls": 0.3})


def test_sequential_conditional_updates_are_order_invariant() -> None:
    fit = _fit()
    state = _joint_state()
    target = "2021-04-01"

    first_a = apply_partial_defining_event(
        state,
        target,
        {"payrolls": 0.8},
        previously_observed={},
        component_fit=fit,
    )
    final_ab = apply_partial_defining_event(
        first_a.state,
        target,
        {"industrial_production": -0.25},
        previously_observed=first_a.observed_components,
        component_fit=fit,
    )
    first_b = apply_partial_defining_event(
        state,
        target,
        {"industrial_production": -0.25},
        previously_observed={},
        component_fit=fit,
    )
    final_ba = apply_partial_defining_event(
        first_b.state,
        target,
        {"payrolls": 0.8},
        previously_observed=first_b.observed_components,
        component_fit=fit,
    )

    np.testing.assert_allclose(final_ab.state.mean, final_ba.state.mean, atol=1.0e-9)
    np.testing.assert_allclose(
        final_ab.state.covariance, final_ba.state.covariance, atol=1.0e-9
    )
    assert set(final_ab.observed_components) == {
        "payrolls",
        "industrial_production",
    }


def test_duplicate_component_is_rejected_to_prevent_double_counting() -> None:
    fit = _fit()

    with pytest.raises(ValueError, match="cannot be processed twice"):
        apply_partial_defining_event(
            _joint_state(),
            "2021-04-01",
            {"payrolls": 0.2},
            previously_observed={"payrolls": 0.2},
            component_fit=fit,
        )


def test_four_released_growth_components_make_growth_axis_exact() -> None:
    fit = _fit()
    observations = {
        "payrolls": 1.0,
        "industrial_production": 0.5,
        "consumer_activity": -0.25,
        "unemployment_rate": 0.75,
    }

    result = apply_partial_defining_event(
        _joint_state(),
        "2021-04-01",
        observations,
        previously_observed={},
        component_fit=fit,
    )

    target_position = result.state.position("2021-04-01")
    growth_index = 2 * target_position
    assert result.exact_axes == ("growth",)
    assert result.likelihood_components == ()
    assert result.state.mean[growth_index] == pytest.approx(0.5)
    np.testing.assert_allclose(result.state.covariance[growth_index, :], 0.0)
    assert not result.state.exact_mask[target_position]


def test_partial_axis_then_second_axis_produces_exact_score_pair() -> None:
    fit = _fit()
    state = _joint_state()
    target = "2021-04-01"
    growth = {component: value for component, value in zip(GROWTH_COMPONENTS, [1, 2, 3, 4])}
    inflation = {
        component: value
        for component, value in zip(INFLATION_COMPONENTS, [-1, -2, -3, -4])
    }

    first = apply_partial_defining_event(
        state,
        target,
        growth,
        previously_observed={},
        component_fit=fit,
    )
    final = apply_partial_defining_event(
        first.state,
        target,
        inflation,
        previously_observed=first.observed_components,
        component_fit=fit,
    )

    position = final.state.position(target)
    assert final.state.exact_mask[position]
    np.testing.assert_allclose(final.state.marginal(target)[0], [2.5, -2.5])
    np.testing.assert_allclose(final.state.marginal(target)[1], 0.0)
    assert condition_on_exact_score(final.state, target, [2.5, -2.5]) is final.state


def test_single_axis_conditioning_is_repeatable_and_conflicts_are_rejected() -> None:
    state = _joint_state()
    target = "2021-03-01"

    first = condition_on_exact_score_axes(state, target, {"growth": 0.75})
    second = condition_on_exact_score_axes(first, target, {"growth": 0.75})

    np.testing.assert_allclose(first.mean, second.mean)
    with pytest.raises(ValueError, match="conflicting repeated"):
        condition_on_exact_score_axes(first, target, {"growth": 0.8})


def test_small_but_positive_axis_variance_is_not_mistaken_for_exact() -> None:
    state = _joint_state()
    covariance = state.covariance.copy()
    position = state.position("2021-03-01")
    inflation_index = 2 * position + 1
    covariance[inflation_index, :] *= 1.0e-5
    covariance[:, inflation_index] *= 1.0e-5
    narrow = JointGaussianState(
        reference_months=state.reference_months,
        mean=state.mean,
        covariance=covariance,
    )

    conditioned = condition_on_exact_score_axes(
        narrow, "2021-03-01", {"inflation": 0.3}
    )

    assert conditioned.mean[inflation_index] == pytest.approx(0.3)
    np.testing.assert_allclose(conditioned.covariance[inflation_index, :], 0.0)
