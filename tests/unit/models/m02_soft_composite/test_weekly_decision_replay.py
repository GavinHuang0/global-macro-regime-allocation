"""Focused tests for allocation-only Model 02 decision snapshots."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.inference_sensitivities import (
    SensitivityVariant,
    run_inference_sensitivities,
)
from regime_allocation.models.m02_soft_composite.joint_filter import (
    VarDynamics,
    initialize_joint_exact,
    monthly_quadrant_probabilities,
    roll_joint_gaussian,
)
from regime_allocation.models.m02_soft_composite.probability_map import REGIME_ORDER
from regime_allocation.models.m02_soft_composite.weekly_decision_replay import (
    _decision_marginal_records,
    _normalize_decision_dates,
    run_weekly_decision_replay,
)


def _replace_once(source: str, before: str, after: str = "") -> str:
    assert source.count(before) == 1
    return source.replace(before, after, 1)


def test_allocation_fork_preserves_the_frozen_replay_body() -> None:
    allocation_source = inspect.getsource(run_weekly_decision_replay)
    allocation_source = _replace_once(
        allocation_source,
        "def run_weekly_decision_replay(",
        "def run_inference_sensitivities(",
    )
    allocation_source = _replace_once(
        allocation_source,
        "    decision_dates: Sequence[object] = (),\n",
    )
    allocation_source = _replace_once(
        allocation_source,
        ") -> WeeklyDecisionReplayResult:",
        ") -> InferenceSensitivityResult:",
    )
    allocation_source = _replace_once(
        allocation_source,
        '    """Replay variants causally and capture requested start-of-day decisions."""',
        '    """Replay all enabled partial-release and robustness variants causally."""',
    )
    allocation_source = _replace_once(
        allocation_source,
        "    requested_decision_dates = _normalize_decision_dates(decision_dates)\n",
    )
    allocation_source = _replace_once(
        allocation_source,
        "    outside_replay = sorted(\n"
        "        date for date in requested_decision_dates if date < start or date > end\n"
        "    )\n"
        "    if outside_replay:\n"
        '        formatted = ", ".join(date.date().isoformat() for date in outside_replay)\n'
        "        raise ValueError(\n"
        '            "decision_dates must fall within the initialized replay window "\n'
        '            f"[{start.date().isoformat()}, {end.date().isoformat()}]: {formatted}"\n'
        "        )\n",
    )
    allocation_source = _replace_once(
        allocation_source,
        "        .union(requested_decision_dates)\n",
    )
    allocation_source = _replace_once(
        allocation_source,
        "    decision_records: list[dict[str, object]] = []\n",
    )
    allocation_source = _replace_once(
        allocation_source,
        "        if current_date in requested_decision_dates:\n"
        "            decision_records.extend(\n"
        "                _decision_marginal_records(\n"
        "                    states,\n"
        "                    variants,\n"
        "                    mappings,\n"
        "                    signal_date=current_date,\n"
        "                )\n"
        "            )\n\n",
    )
    allocation_source = _replace_once(
        allocation_source,
        "    decision_marginals = pd.DataFrame.from_records(\n"
        "        decision_records,\n"
        "        columns=_DECISION_MARGINAL_COLUMNS,\n"
        "    )\n",
    )
    allocation_source = _replace_once(
        allocation_source,
        "    return WeeklyDecisionReplayResult(",
        "    return InferenceSensitivityResult(",
    )
    allocation_source = _replace_once(
        allocation_source,
        "        decision_marginals=decision_marginals,\n",
    )

    assert allocation_source == inspect.getsource(run_inference_sensitivities)


def test_decision_snapshot_is_after_month_roll_and_before_same_day_events() -> None:
    source = inspect.getsource(run_weekly_decision_replay)

    roll_position = source.index("states[variant.variant_id] = roll_joint_gaussian(")
    decision_position = source.index("if current_date in requested_decision_dates:")
    first_same_day_event_position = source.index("date_partial = partial_events.loc[")

    assert roll_position < decision_position < first_same_day_event_position


def test_decision_dates_are_normalized_and_deduplicated() -> None:
    dates = _normalize_decision_dates(
        (
            "2020-05-04 01:00:00-04:00",
            pd.Timestamp("2020-05-04 23:00:00", tz="UTC"),
            "2020-05-11",
        )
    )

    assert dates == frozenset(
        {pd.Timestamp("2020-05-04"), pd.Timestamp("2020-05-11")}
    )
    assert _normalize_decision_dates(()) == frozenset()
    with pytest.raises(ValueError, match="sequence"):
        _normalize_decision_dates("2020-05-04")


def test_decision_marginal_uses_post_roll_state_and_strict_prior_mapping() -> None:
    dynamics = VarDynamics(
        intercept=np.asarray([0.1, -0.2]),
        transition=np.eye(2) * 0.8,
        innovation_covariance=np.asarray([[0.4, 0.05], [0.05, 0.3]]),
    )
    initial = initialize_joint_exact(
        "2020-01-01", np.asarray([0.5, -0.25]), [dynamics, dynamics, dynamics]
    )
    rolled = roll_joint_gaussian(initial, dynamics)
    signal_date = pd.Timestamp("2020-05-01")
    mappings = pd.DataFrame(
        {
            "reference_month": pd.to_datetime(["2020-02-01", "2020-03-01"]),
            "score_available_at": pd.to_datetime(["2020-04-30", "2020-05-01"]),
            "mapping_status": ["available", "available"],
            "specification_id": ["prior", "same_day"],
            "revision_horizon_months": [12, 12],
            "growth_map_variance": [0.2, 20.0],
            "growth_inflation_map_covariance": [0.01, 0.0],
            "inflation_map_variance": [0.1, 20.0],
        }
    )
    variant = SensitivityVariant(
        variant_id="selected",
        model_role="baseline",
        non_defining_evidence=True,
        partial_defining_releases=True,
        emission_family="student_t_7",
        var_method="ols",
        retail_method="baseline_nominal",
    )

    row = _decision_marginal_records(
        {variant.variant_id: rolled},
        (variant,),
        mappings,
        signal_date=signal_date,
    )[0]

    target_month = pd.Timestamp("2020-05-01")
    expected = monthly_quadrant_probabilities(
        rolled,
        target_month,
        np.asarray([[0.2, 0.01], [0.01, 0.1]]),
    )
    mean, covariance = rolled.marginal(target_month)
    assert row["variant_id"] == "selected"
    assert row["model_role"] == "baseline"
    assert row["signal_date"] == signal_date
    assert row["as_of_date"] == signal_date
    assert row["same_day_release_evidence_included"] is False
    assert row["target_month"] == target_month
    assert row["reference_month"] == target_month
    assert row["relative_month"] == 0
    assert row["mapping_reference_month"] == pd.Timestamp("2020-02-01")
    assert row["mapping_available_at"] == pd.Timestamp("2020-04-30")
    assert row["mapping_specification_id"] == "prior"
    assert row["mapping_revision_horizon_months"] == 12
    assert row["growth_mean"] == pytest.approx(mean[0])
    assert row["inflation_mean"] == pytest.approx(mean[1])
    assert row["growth_variance"] == pytest.approx(covariance[0, 0])
    assert row["growth_inflation_covariance"] == pytest.approx(covariance[0, 1])
    assert row["inflation_variance"] == pytest.approx(covariance[1, 1])
    assert row["exact_score"] is False
    for regime in REGIME_ORDER:
        assert row[f"probability_{regime}"] == pytest.approx(expected[regime])
