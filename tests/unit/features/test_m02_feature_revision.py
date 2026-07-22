"""Unit contracts for dependence-aware Model 02 feature revision events."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.features.m02_feature_revision import (
    BUSINESS_ACTIVITY_RESPONSE,
    BUSINESS_PIPELINE_RESPONSE,
    BUSINESS_REVISION_BLOCK,
    CONDITIONAL_CLAIMS_BLOCK,
    CONDITIONAL_CLAIMS_CONTROL,
    build_business_investment_revision_events,
    build_conditional_continued_claims_events,
)


def _row(
    *,
    feature_name: str,
    release_date: str,
    reference_date: str,
    value: float,
) -> dict[str, object]:
    reference = pd.Timestamp(reference_date)
    return {
        "event_id": f"old:{feature_name}:{reference_date}",
        "event_group_id": f"old:{feature_name}:{release_date}",
        "release_block": "weekly",
        "release_date": release_date,
        "reference_date": reference_date,
        "reference_month": reference.to_period("M").to_timestamp(),
        "frequency": "weekly",
        "feature_name": feature_name,
        "series_id": "ICSA" if feature_name.startswith("initial") else "CCSA",
        "provider_id": "test",
        "source_url": "https://example.invalid",
        "current_value": 1.0,
        "previous_value_as_of_release": np.nan,
        "transform": "expanding_log_ar1_innovation",
        "transformed_value": value,
        "feature_value": value,
        "feature_status": "available",
        "release_lag_days": 0,
        "is_target_defining": False,
    }


def test_conditional_claims_preserves_ccsa_month_across_boundary() -> None:
    events = pd.DataFrame(
        [
            _row(
                feature_name="continued_claims_innovation",
                release_date="2025-02-06",
                reference_date="2025-01-25",
                value=0.4,
            ),
            _row(
                feature_name="initial_claims_innovation",
                release_date="2025-02-06",
                reference_date="2025-02-01",
                value=-0.2,
            ),
        ]
    )
    result, audit = build_conditional_continued_claims_events(events)
    assert len(result) == 2
    assert set(result["release_block"]) == {CONDITIONAL_CLAIMS_BLOCK}
    assert set(result["reference_month"]) == {pd.Timestamp("2025-01-01")}
    assert set(result["reference_date"]) == {pd.Timestamp("2025-01-25")}
    assert set(result["control_source_reference_date"]) == {
        pd.Timestamp("2025-02-01")
    }
    assert CONDITIONAL_CLAIMS_CONTROL in set(result["feature_name"])
    assert audit.paired_rows == 1
    assert audit.cross_month_pairs == 1
    assert audit.unmatched_rows == 0


def test_conditional_claims_keeps_unmatched_response_for_missing_control_audit() -> None:
    events = pd.DataFrame(
        [
            _row(
                feature_name="continued_claims_innovation",
                release_date="2025-02-06",
                reference_date="2025-01-25",
                value=0.4,
            ),
            _row(
                feature_name="initial_claims_innovation",
                release_date="2025-02-13",
                reference_date="2025-02-08",
                value=-0.2,
            ),
        ]
    )
    result, audit = build_conditional_continued_claims_events(events)
    assert len(result) == 1
    assert result.iloc[0]["feature_name"] == "continued_claims_innovation"
    assert audit.paired_rows == 0
    assert audit.unmatched_rows == 1


def test_business_pipeline_uses_strictly_prior_publication_groups() -> None:
    rows: list[dict[str, object]] = []
    raw_gaps: list[float] = []
    for index, reference in enumerate(pd.date_range("2020-01-01", periods=30, freq="MS")):
        release = reference + pd.offsets.MonthBegin(1) + pd.Timedelta(days=19)
        shipment_change = 0.1 * index
        raw_gap = float(np.sin(index / 3.0))
        raw_gaps.append(raw_gap)
        for feature_name, transformed, current, previous in (
            (
                "core_capital_goods_orders_log_change",
                shipment_change + raw_gap,
                100.0 + index,
                99.0 + index,
            ),
            (
                BUSINESS_ACTIVITY_RESPONSE,
                shipment_change,
                90.0 + index,
                89.0 + index,
            ),
        ):
            rows.append(
                {
                    "event_id": f"legacy:{release.date()}:{reference.date()}",
                    "event_group_id": f"legacy:{release.date()}",
                    "release_block": "business_investment",
                    "release_date": release,
                    "reference_date": reference,
                    "reference_month": reference,
                    "frequency": "monthly",
                    "feature_name": feature_name,
                    "series_id": "NEWORDER" if "orders" in feature_name else "ANXAVS",
                    "provider_id": "test",
                    "source_url": "https://example.invalid",
                    "current_value": current,
                    "previous_value_as_of_release": previous,
                    "transform": "same_vintage_log_change",
                    "transformed_value": transformed,
                    "feature_value": transformed / 2.0,
                    "feature_status": "available",
                    "release_lag_days": 50,
                    "is_target_defining": False,
                }
            )
    result, audit = build_business_investment_revision_events(pd.DataFrame(rows))
    assert set(result["release_block"]) == {BUSINESS_REVISION_BLOCK}
    pipeline = result.loc[
        result["feature_name"].eq(BUSINESS_PIPELINE_RESPONSE)
    ].reset_index(drop=True)
    assert (pipeline.loc[:23, "feature_status"] == "standardization_warmup").all()
    expected = (raw_gaps[24] - np.mean(raw_gaps[:24])) / np.std(
        raw_gaps[:24], ddof=1
    )
    assert pipeline.loc[24, "feature_status"] == "available"
    assert pipeline.loc[24, "feature_value"] == pytest.approx(expected)
    assert pipeline.loc[24, "standardization_prior_count"] == 24
    assert audit.source_event_groups == 30
    assert audit.matched_event_groups == 30
    assert audit.available_joint_events == 6
