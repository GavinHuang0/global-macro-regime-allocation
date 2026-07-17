"""Tests for Model 01's causal walk-forward orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_allocation.models.m01_deterministic_composite.inference import STATE_IDS
from regime_allocation.models.m01_deterministic_composite.walkforward import (
    LikelihoodSpecification,
    attach_forecast_targets,
    evaluate_forecast_table,
    prepare_block_vectors,
    run_walk_forward_filter,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference_month": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
            ),
            "regime_id": [STATE_IDS[0], STATE_IDS[1], STATE_IDS[2], STATE_IDS[3]],
            "label_available_at": pd.to_datetime(
                ["2020-02-10", "2020-03-10", "2020-04-10", "2020-05-10"]
            ),
        }
    )


def _events() -> pd.DataFrame:
    records = []
    values = [0.0, 1.0, 2.0, -1.0]
    releases = ["2020-02-05", "2020-03-05", "2020-04-05", "2020-04-20"]
    months = ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
    for position, (release, month, value) in enumerate(
        zip(releases, months, values, strict=True)
    ):
        records.append(
            {
                "event_id": f"weekly_claims:{release}:{month}",
                "event_group_id": f"weekly_claims:{release}",
                "release_block": "weekly_claims",
                "release_date": release,
                "reference_month": month,
                "feature_name": "initial_claims_innovation",
                "series_id": "ICSA",
                "feature_value": value,
                "feature_status": "available",
            }
        )
        records.append(
            {
                "event_id": f"continued:{position}",
                "event_group_id": f"weekly_claims:{release}",
                "release_block": "weekly_claims",
                "release_date": release,
                "reference_month": month,
                "feature_name": "continued_claims_innovation",
                "series_id": "CCSA",
                "feature_value": 99.0,
                "feature_status": "available",
            }
        )
    return pd.DataFrame.from_records(records)


def _vectors() -> dict[str, pd.DataFrame]:
    return prepare_block_vectors(
        _events(),
        _history(),
        block_configuration={
            "weekly_claims": {
                "feature_names": ("initial_claims_innovation",),
                "source_series": ("ICSA",),
            }
        },
    )


def test_prepare_block_vectors_keeps_icsa_only_and_joins_target_timing() -> None:
    vectors = _vectors()["weekly_claims"]

    assert len(vectors) == 4
    assert "initial_claims_innovation" in vectors
    assert "continued_claims_innovation" not in vectors
    assert vectors["complete_vector"].all()
    assert vectors["training_available_at"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-02-10",
        "2020-03-10",
        "2020-04-10",
        "2020-05-10",
    ]


def test_walk_forward_applies_strict_fit_then_saves_hard_confirmation() -> None:
    specification = LikelihoodSpecification(
        specification_id="baseline",
        minimum_complete_vectors=2,
    )
    result = run_walk_forward_filter(
        _history(),
        _vectors(),
        specification=specification,
        filter_start="2020-04-01",
        filter_end="2020-04-30",
    )

    applied = result.event_audit.loc[
        result.event_audit["update_status"] == "applied"
    ]
    assert len(applied) == 2
    assert applied["training_count"].tolist() == [2, 3]
    confirmation = result.checkpoints.loc[
        result.checkpoints["checkpoint_type"] == "post_confirmation_group"
    ].iloc[0]
    march = result.marginals.loc[
        (result.marginals["checkpoint_id"] == confirmation["checkpoint_id"])
        & (result.marginals["reference_month"] == pd.Timestamp("2020-03-01"))
        & (result.marginals["marginal_type"] == "path")
    ]
    realized = march.set_index("regime_id")["probability"]
    assert realized.loc[STATE_IDS[2]] == 1.0
    assert np.isclose(realized.drop(STATE_IDS[2]).sum(), 0.0)


def test_same_day_leading_release_precedes_pre_confirmation_checkpoint() -> None:
    events = _events()
    same_day = pd.DataFrame.from_records(
        [
            {
                "event_id": "weekly_claims:2020-04-10:2020-03",
                "event_group_id": "weekly_claims:2020-04-10",
                "release_block": "weekly_claims",
                "release_date": "2020-04-10",
                "reference_month": "2020-03-01",
                "feature_name": "initial_claims_innovation",
                "series_id": "ICSA",
                "feature_value": 3.0,
                "feature_status": "available",
            }
        ]
    )
    vectors = prepare_block_vectors(
        pd.concat([events, same_day], ignore_index=True),
        _history(),
        block_configuration={
            "weekly_claims": {
                "feature_names": ("initial_claims_innovation",),
                "source_series": ("ICSA",),
            }
        },
    )
    result = run_walk_forward_filter(
        _history(),
        vectors,
        specification=LikelihoodSpecification(
            specification_id="baseline",
            minimum_complete_vectors=2,
        ),
        filter_start="2020-04-01",
        filter_end="2020-04-30",
    )

    same_date = result.checkpoints.loc[
        result.checkpoints["as_of_date"].eq(pd.Timestamp("2020-04-10"))
    ]
    post_release = same_date.loc[
        same_date["checkpoint_type"].eq("post_release_group")
    ].iloc[0]
    pre_confirmation = same_date.loc[
        same_date["checkpoint_type"].eq("pre_confirmation_group")
    ].iloc[0]

    assert pre_confirmation["parent_checkpoint_id"] == post_release["checkpoint_id"]
    assert pre_confirmation["phase_order"] > post_release["phase_order"]
    same_day_audit = result.event_audit.loc[
        result.event_audit["event_id"].eq(
            "weekly_claims:2020-04-10:2020-03"
        )
    ].iloc[0]
    assert same_day_audit["update_status"] == "applied"


def test_forecast_evaluation_uses_transition_only_on_identical_rows() -> None:
    baseline = run_walk_forward_filter(
        _history(),
        _vectors(),
        specification=LikelihoodSpecification(
            specification_id="baseline",
            minimum_complete_vectors=2,
        ),
        filter_start="2020-04-01",
        filter_end="2020-04-30",
        store_detailed_paths=False,
    )
    transition = run_walk_forward_filter(
        _history(),
        _vectors(),
        specification=LikelihoodSpecification(
            specification_id="transition_only",
            minimum_complete_vectors=2,
        ),
        filter_start="2020-04-01",
        filter_end="2020-04-30",
        apply_evidence=False,
        store_detailed_paths=False,
        store_checkpoint_artifacts=False,
        store_audits=False,
    )
    forecasts = attach_forecast_targets(
        pd.concat([baseline.forecasts, transition.forecasts], ignore_index=True),
        _history(),
        evaluation_start="2020-04-01",
    )
    evaluated = evaluate_forecast_table(forecasts, calibration_bin_count=4)

    fixed = evaluated.metrics.loc[
        evaluated.metrics["checkpoint_type"].isin(["month_start", "month_end"])
    ]
    assert set(fixed["specification_id"]) == {"baseline", "transition_only"}
    assert fixed["observation_count"].eq(1).all()
    assert fixed["brier_skill_vs_transition_only"].notna().all()
