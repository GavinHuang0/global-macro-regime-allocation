"""Test Model 02's causal event preparation and walk-forward orchestration."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.walkforward import (
    prepare_observation_data,
    run_event_driven_filter,
)


def _histories(periods: int = 52) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    months = pd.date_range("2000-01-01", periods=periods, freq="MS")
    rng = np.random.default_rng(321)
    scores = np.zeros((periods, 2))
    transition = np.asarray([[0.62, 0.08], [-0.07, 0.53]])
    for position in range(1, periods):
        scores[position] = (
            transition @ scores[position - 1]
            + rng.multivariate_normal([0.0, 0.0], [[0.05, 0.01], [0.01, 0.04]])
        )
    available = months + pd.offsets.MonthBegin(2) + pd.Timedelta(days=4)
    score_history = pd.DataFrame(
        {
            "reference_month": months,
            "growth_score": scores[:, 0],
            "inflation_score": scores[:, 1],
            "score_available_at": available,
        }
    )
    mapping_history = pd.DataFrame(
        {
            "reference_month": months,
            "score_available_at": available,
            "mapping_status": "available",
            "is_baseline": True,
            "revision_horizon_months": 12,
            "growth_map_variance": 0.20 + np.arange(periods) * 0.0001,
            "growth_inflation_map_covariance": 0.01,
            "inflation_map_variance": 0.16 + np.arange(periods) * 0.0001,
        }
    )
    return score_history, mapping_history, scores


def _config(*, controls: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "transition": {"minimum_training_pairs": 12},
        "mapping": {
            "baseline_revision_horizon_months": 12,
            "joint_path_samples": 256,
        },
        "calendar": {"replay_end": "2003-06-30"},
        "emissions": {
            "lambda_grid": [0.1],
            "covariance": {"eigenvalue_floor": 1.0e-8},
        },
        "observation_models": {
            "signal_model": {
                "economic_block": "test_signal",
                "event_block": "shared_event_block",
                "responses": ["signal"],
                "controls": list(controls),
                "loading_penalties": [[1.0, 8.0]],
                "exact_zero_mask": [[False, False]],
                "minimum_training_samples": 8,
                "validation_minimum_training_samples": 5,
                "minimum_validation_observations": 3,
            }
        },
    }


def _event_row(
    *,
    event_id: str,
    release_date: pd.Timestamp,
    reference_month: pd.Timestamp,
    feature_name: str,
    value: float,
    series_id: str = "TEST",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_group_id": f"group:{release_date:%Y-%m-%d}",
        "release_block": "shared_event_block",
        "release_date": release_date,
        "reference_month": reference_month,
        "feature_name": feature_name,
        "feature_value": value,
        "feature_status": "available",
        "series_id": series_id,
    }


def _events(
    score_history: pd.DataFrame,
    score_values: np.ndarray,
    *,
    controls: tuple[str, ...] = (),
) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    records: list[dict[str, object]] = []
    for position, row in enumerate(score_history.itertuples(index=False)):
        release = pd.Timestamp(row.reference_month) + pd.Timedelta(days=15)
        records.append(
            _event_row(
                event_id=f"signal-{position}",
                release_date=release,
                reference_month=pd.Timestamp(row.reference_month),
                feature_name="signal",
                value=float(0.9 * score_values[position, 0] + rng.normal(0.0, 0.08)),
            )
        )
        if "rate" in controls:
            records.append(
                _event_row(
                    event_id=f"signal-{position}",
                    release_date=release,
                    reference_month=pd.Timestamp(row.reference_month),
                    feature_name="rate",
                    value=float(np.sin(position / 4.0)),
                    series_id="RATE",
                )
            )
        if "methodology_dummy" in controls:
            records.append(
                _event_row(
                    event_id=f"signal-{position}",
                    release_date=release,
                    reference_month=pd.Timestamp(row.reference_month),
                    feature_name="methodology_dummy",
                    value=float(position >= 30),
                    series_id="DUMMY",
                )
            )
    return pd.DataFrame.from_records(records)


def test_prepare_vectors_uses_maximum_availability_and_separates_shared_block() -> None:
    scores, _, values = _histories(periods=20)
    events = _events(scores, values)
    # A second asynchronous model shares the event block but uses another name.
    extra = _event_row(
        event_id="input-cost",
        release_date=pd.Timestamp("2000-06-20"),
        reference_month=pd.Timestamp("2000-05-01"),
        feature_name="input_cost",
        value=1.5,
        series_id="PPI",
    )
    events = pd.concat([events, pd.DataFrame([extra])], ignore_index=True)
    config = _config()
    config["observation_models"]["input_cost_model"] = {
        "economic_block": "inflation_pressure",
        "event_block": "shared_event_block",
        "responses": ["input_cost"],
        "controls": [],
        "loading_penalties": [[10.0, 1.0]],
        "exact_zero_mask": [[False, False]],
        "minimum_training_samples": 8,
        "validation_minimum_training_samples": 5,
        "minimum_validation_observations": 3,
    }

    prepared = prepare_observation_data(events, scores, config)

    signal = prepared.training_tables["signal_model"].iloc[0]
    assert signal["training_available_at"] == signal["target_score_available_at"]
    assert set(prepared.events["observation_model_id"]) == {
        "signal_model",
        "input_cost_model",
    }
    cost = prepared.events.loc[
        prepared.events["observation_model_id"] == "input_cost_model"
    ]
    assert len(cost) == 1
    assert cost.iloc[0]["input_cost"] == pytest.approx(1.5)
    assert "signal" not in cost.columns or pd.isna(cost.iloc[0].get("signal"))


def test_monthly_training_vector_joins_async_responses_at_latest_availability() -> None:
    scores, _, _ = _histories(periods=20)
    reference_month = pd.Timestamp("2000-05-01")
    first_release = pd.Timestamp("2000-06-10")
    second_release = pd.Timestamp("2000-06-24")
    events = pd.DataFrame(
        [
            _event_row(
                event_id="orders",
                release_date=first_release,
                reference_month=reference_month,
                feature_name="orders",
                value=0.4,
            ),
            _event_row(
                event_id="shipments",
                release_date=second_release,
                reference_month=reference_month,
                feature_name="shipments",
                value=-0.2,
            ),
        ]
    )
    config = _config()
    config["observation_models"] = {
        "investment": {
            "economic_block": "investment",
            "event_block": "shared_event_block",
            "responses": ["orders", "shipments"],
            "controls": [],
            "loading_penalties": [[1.0, 8.0], [1.0, 8.0]],
            "exact_zero_mask": [[False, False], [False, False]],
            "minimum_training_samples": 8,
            "validation_minimum_training_samples": 5,
            "minimum_validation_observations": 3,
        }
    }

    prepared = prepare_observation_data(events, scores, config)

    assert len(prepared.events) == 2
    assert prepared.events["complete_response_vector"].eq(False).all()
    training = prepared.training_tables["investment"].iloc[0]
    target_available = scores.set_index("reference_month").loc[
        reference_month, "score_available_at"
    ]
    assert training["orders"] == pytest.approx(0.4)
    assert training["shipments"] == pytest.approx(-0.2)
    assert training["response_available_at"] == second_release
    assert training["training_available_at"] == max(second_release, target_available)


def test_replay_orders_evidence_before_exact_score_and_uses_mapping_causally() -> None:
    scores, mappings, values = _histories()
    events = _events(scores, values)
    target = scores.iloc[28]
    exact_date = pd.Timestamp(target["score_available_at"])
    # Same-day evidence must be processed before the score becomes exact.
    events = pd.concat(
        [
            events,
            pd.DataFrame(
                [
                    _event_row(
                        event_id="same-day-target-evidence",
                        release_date=exact_date,
                        reference_month=pd.Timestamp(target["reference_month"]),
                        feature_name="signal",
                        value=4.0,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    config = _config()
    config["calendar"]["replay_end"] = str((exact_date + pd.Timedelta(days=3)).date())
    prepared = prepare_observation_data(events, scores, config)

    result = run_event_driven_filter(
        scores, mappings, prepared, config, store_joint_paths=False
    )

    same_day = result.event_audit.loc[
        result.event_audit["event_ids"].str.contains("same-day-target-evidence")
    ].iloc[0]
    assert same_day["update_status"] == "applied"
    assert bool(same_day["residual_recorded"])
    assert same_day["diagnostic_state_cutoff"] == "shared_pre_release_group"
    exact_audit = result.exact_score_audit.loc[
        (result.exact_score_audit["reference_month"] == target["reference_month"])
        & (result.exact_score_audit["filter_variant"] == "evidence_filter")
    ].iloc[0]
    assert exact_audit["status"] == "conditioned_exactly"

    checkpoints = result.checkpoints.loc[result.checkpoints["as_of_date"] == exact_date]
    pre_day = checkpoints.loc[
        checkpoints["checkpoint_type"] == "pre_exact_score_day"
    ].iloc[0]
    before = checkpoints.loc[checkpoints["checkpoint_type"] == "pre_exact_score"].iloc[0]
    after = checkpoints.loc[checkpoints["checkpoint_type"] == "post_exact_score"].iloc[0]
    assert pre_day["mapping_available_at"] < exact_date
    assert before["mapping_available_at"] < exact_date
    assert after["mapping_available_at"] == exact_date
    evaluated = result.evaluation_rows.loc[
        result.evaluation_rows["reference_month"] == target["reference_month"]
    ]
    assert set(evaluated["filter_variant"]) == {
        "evidence_filter",
        "transition_only",
    }
    assert set(evaluated["evaluation_checkpoint"]) == {
        "strict_pre_day",
        "post_release_pre_exact_sensitivity",
    }
    strict = evaluated.loc[
        evaluated["evaluation_checkpoint"] == "strict_pre_day"
    ]
    sensitivity = evaluated.loc[
        evaluated["evaluation_checkpoint"]
        == "post_release_pre_exact_sensitivity"
    ]
    assert not strict["same_day_release_evidence_included"].any()
    assert bool(
        sensitivity.loc[
            sensitivity["filter_variant"] == "evidence_filter",
            "same_day_release_evidence_included",
        ].iloc[0]
    )
    assert not bool(
        sensitivity.loc[
            sensitivity["filter_variant"] == "transition_only",
            "same_day_release_evidence_included",
        ].iloc[0]
    )


def test_exact_target_release_is_no_op_but_keeps_causal_predictive_residual() -> None:
    scores, mappings, values = _histories()
    events = _events(scores, values)
    target = scores.iloc[27]
    exact_date = pd.Timestamp(target["score_available_at"])
    delayed_date = exact_date + pd.Timedelta(days=5)
    events = pd.concat(
        [
            events,
            pd.DataFrame(
                [
                    _event_row(
                        event_id="delayed-evidence",
                        release_date=delayed_date,
                        reference_month=pd.Timestamp(target["reference_month"]),
                        feature_name="signal",
                        value=-3.0,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    config = _config()
    config["calendar"]["replay_end"] = str((delayed_date + pd.Timedelta(days=1)).date())
    prepared = prepare_observation_data(events, scores, config)

    result = run_event_driven_filter(
        scores, mappings, prepared, config, store_joint_paths=False
    )

    audit = result.event_audit.loc[
        result.event_audit["event_ids"].str.contains("delayed-evidence")
    ].iloc[0]
    assert audit["update_status"] == "target_exact_predictive_only"
    assert bool(audit["residual_recorded"])
    residual = result.predictive_residuals.loc[
        result.predictive_residuals["event_instance_id"] == audit["event_instance_id"]
    ].iloc[0]
    assert residual["response_id"] == "signal"
    assert not bool(residual["state_update_applied"])
    assert residual["diagnostic_state_cutoff"] == "shared_pre_release_group"
    assert residual["whitened_innovation"] == pytest.approx(
        residual["standardized_innovation"]
    )


def test_same_day_exact_score_batch_is_atomic_for_sensitivity_evaluation() -> None:
    scores, mappings, values = _histories()
    first_index, second_index = 30, 31
    shared_date = pd.Timestamp(scores.iloc[second_index]["score_available_at"])
    scores.loc[[first_index, second_index], "score_available_at"] = shared_date
    mappings.loc[[first_index, second_index], "score_available_at"] = shared_date
    events = _events(scores, values)
    config = _config()
    config["calendar"]["replay_end"] = str((shared_date + pd.Timedelta(days=1)).date())
    prepared = prepare_observation_data(events, scores, config)

    result = run_event_driven_filter(
        scores, mappings, prepared, config, store_joint_paths=False
    )

    target_month = pd.Timestamp(scores.iloc[second_index]["reference_month"])
    rows = result.evaluation_rows.loc[
        result.evaluation_rows["reference_month"].eq(target_month)
    ]
    for variant in ("evidence_filter", "transition_only"):
        variant_rows = rows.loc[rows["filter_variant"].eq(variant)].set_index(
            "evaluation_checkpoint"
        )
        primary = variant_rows.loc["strict_pre_day"]
        sensitivity = variant_rows.loc["post_release_pre_exact_sensitivity"]
        # No release occurs on the shared score date.  Therefore both metrics
        # must use the same frozen state; conditioning the first score in the
        # batch must not leak into the second score's sensitivity.
        assert sensitivity["growth_error"] == pytest.approx(primary["growth_error"])
        assert sensitivity["inflation_error"] == pytest.approx(
            primary["inflation_error"]
        )


def test_future_data_cannot_change_an_earlier_replay() -> None:
    scores, mappings, values = _histories()
    events = _events(scores, values)
    config = _config()
    cutoff = pd.Timestamp("2002-12-31")
    config["calendar"]["replay_end"] = str(cutoff.date())
    baseline_prepared = prepare_observation_data(events, scores, config)
    baseline = run_event_driven_filter(
        scores, mappings, baseline_prepared, config, store_joint_paths=False
    )

    changed_scores = scores.copy()
    future_scores = changed_scores["score_available_at"] > cutoff
    changed_scores.loc[future_scores, ["growth_score", "inflation_score"]] = 1_000_000.0
    changed_events = events.copy()
    future_events = pd.to_datetime(changed_events["release_date"]) > cutoff
    changed_events.loc[future_events, "feature_value"] = -1_000_000.0
    changed_mappings = mappings.copy()
    future_mappings = changed_mappings["score_available_at"] > cutoff
    changed_mappings.loc[
        future_mappings,
        ["growth_map_variance", "inflation_map_variance"],
    ] = 1_000_000.0
    changed_prepared = prepare_observation_data(changed_events, changed_scores, config)
    changed = run_event_driven_filter(
        changed_scores,
        changed_mappings,
        changed_prepared,
        config,
        store_joint_paths=False,
    )

    for variant in ("evidence_filter", "transition_only"):
        np.testing.assert_allclose(
            changed.latest_states[variant].mean,
            baseline.latest_states[variant].mean,
        )
        np.testing.assert_allclose(
            changed.latest_states[variant].covariance,
            baseline.latest_states[variant].covariance,
        )


def test_constant_control_is_dropped_without_suppressing_early_evidence() -> None:
    scores, mappings, values = _histories()
    controls = ("rate", "methodology_dummy")
    events = _events(scores, values, controls=controls)
    config = _config(controls=controls)
    config["calendar"]["replay_end"] = "2002-01-31"  # before the dummy changes
    prepared = prepare_observation_data(events, scores, config)

    result = run_event_driven_filter(
        scores, mappings, prepared, config, store_joint_paths=False
    )

    assert not result.emission_fit_audit.empty
    first = result.emission_fit_audit.iloc[0]
    assert first["active_controls"] == "rate"
    assert first["dropped_unidentified_controls"] == "methodology_dummy"
    assert (result.event_audit["update_status"] == "applied").any()


def test_distinct_target_month_guard_is_applied_before_emission_fit() -> None:
    scores, mappings, values = _histories()
    events = _events(scores, values)
    config = _config()
    config["observation_models"]["signal_model"][
        "minimum_distinct_target_months"
    ] = 100
    config["calendar"]["replay_end"] = "2002-01-31"
    prepared = prepare_observation_data(events, scores, config)

    result = run_event_driven_filter(
        scores, mappings, prepared, config, store_joint_paths=False
    )

    assert result.emission_fit_audit.empty
    assert set(result.event_audit["update_status"]) == {
        "insufficient_distinct_target_months"
    }
    assert (
        result.event_audit["distinct_target_months"]
        < result.event_audit["required_distinct_target_months"]
    ).all()
