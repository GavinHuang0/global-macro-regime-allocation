"""Focused tests for Model 02's integrated inference sensitivities."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regime_allocation.models.m02_soft_composite.gaussian_emissions import (
    LinearGaussianEmissionSpec,
)
from regime_allocation.models.m02_soft_composite.inference_sensitivities import (
    causal_annual_hyperparameter_schedule,
    stronger_retail_shrinkage_spec,
    variants_from_config,
)
from regime_allocation.models.m02_soft_composite.joint_filter import (
    VarDynamics,
    initialize_joint_exact,
    update_joint_gaussian,
)
from regime_allocation.models.m02_soft_composite.robust_emissions import (
    StudentTObservedEmissionSystem,
)


ROOT = Path(__file__).resolve().parents[4]


def test_disabled_unidentified_stress_variant_is_not_run() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/models/m02_inference_sensitivities.yaml").read_text(
            encoding="utf-8"
        )
    )
    variants = variants_from_config(config)
    identifiers = {variant.variant_id for variant in variants}
    assert "retail_stress_interaction_combined" not in identifiers
    assert "student_t_7_combined" in identifiers
    assert "partial_only" in identifiers


def test_annual_schedule_never_uses_current_or_future_fold() -> None:
    folds = pd.DataFrame(
        {
            "profile_id": ["p"] * 6,
            "status": ["scored"] * 6,
            "validation_year": [2018, 2018, 2019, 2019, 2020, 2020],
            "base_lambda": [1.0, 10.0] * 3,
            "degrees_of_freedom": [7.0] * 6,
            "validation_observations": [30] * 6,
            "total_predictive_nll": [30.0, 60.0, 30.0, 60.0, 3000.0, 0.0],
        }
    )
    schedule = causal_annual_hyperparameter_schedule(
        folds,
        profile_id="p",
        first_year=2019,
        final_year=2021,
        allowed_degrees_of_freedom=(7.0,),
        minimum_predictive_events=24,
        fallback_lambda=10.0,
        fallback_degrees_of_freedom=7.0,
    ).set_index("selection_year")
    # The dramatic 2020 fold cannot alter the parameters used during 2020.
    assert schedule.loc[2019, "selected_base_lambda"] == 1.0
    assert schedule.loc[2020, "selected_base_lambda"] == 1.0
    assert schedule.loc[2021, "selected_base_lambda"] == 10.0


def test_annual_schedule_enforces_minimum_predictive_events() -> None:
    folds = pd.DataFrame(
        {
            "profile_id": ["p"],
            "status": ["scored"],
            "validation_year": [2020],
            "base_lambda": [1.0],
            "degrees_of_freedom": [7.0],
            "validation_observations": [23],
            "total_predictive_nll": [1.0],
        }
    )
    row = causal_annual_hyperparameter_schedule(
        folds,
        profile_id="p",
        first_year=2021,
        final_year=2021,
        allowed_degrees_of_freedom=(7.0,),
        minimum_predictive_events=24,
        fallback_lambda=100.0,
        fallback_degrees_of_freedom=7.0,
    ).iloc[0]
    assert row["selected_base_lambda"] == 100.0
    assert row["selection_status"] == "fallback_insufficient_predictive_events"


def test_stronger_retail_spec_changes_only_declared_penalties() -> None:
    baseline = LinearGaussianEmissionSpec(
        block_id="consumer_demand",
        response_names=("a", "b"),
        state_loading_penalties=((1.0, 1.0), (1.0, 1.0)),
        lambda_grid=(0.1, 1.0),
    )
    changed = stronger_retail_shrinkage_spec(
        baseline, {"a": (1.0, 10.0), "b": (1.0, 10.0)}
    )
    assert changed.state_loading_penalties == ((1.0, 10.0), (1.0, 10.0))
    assert changed.response_names == baseline.response_names
    assert changed.lambda_grid == baseline.lambda_grid


def test_frozen_same_day_student_t_factors_are_order_invariant() -> None:
    dynamics = VarDynamics(
        intercept=np.zeros(2),
        transition=np.eye(2) * 0.8,
        innovation_covariance=np.eye(2) * 0.2,
    )
    state = initialize_joint_exact(
        "2020-01-01", np.zeros(2), [dynamics, dynamics, dynamics]
    )
    month = pd.Timestamp("2020-04-01")
    prior_mean, prior_covariance = state.marginal(month)
    systems = [
        StudentTObservedEmissionSystem(
            observed_names=(name,),
            omitted_names=(),
            observation=np.asarray([value]),
            intercept=np.zeros(1),
            state_loadings=np.asarray([loading]),
            control_contribution=np.zeros(1),
            residual_scale=np.asarray([[0.5]]),
            degrees_of_freedom=7.0,
        )
        for name, value, loading in (
            ("growth_release", 1.5, [1.0, 0.2]),
            ("inflation_release", -0.8, [0.1, 1.0]),
        )
    ]
    frozen = [
        system.approximate_gaussian_system(prior_mean, prior_covariance)
        for system in systems
    ]

    def apply(order: tuple[int, int]):
        current = state
        for position in order:
            current = update_joint_gaussian(
                current, month, **frozen[position].to_joint_filter_inputs()
            ).state
        return current

    first = apply((0, 1))
    second = apply((1, 0))
    np.testing.assert_allclose(first.mean, second.mean, atol=1.0e-12)
    np.testing.assert_allclose(first.covariance, second.covariance, atol=1.0e-12)
    assert all(math.isfinite(item.event_weight) for item in frozen)
