"""Tests for provider-neutral, causal release-evidence event records."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from regime_allocation.data.providers.vintage_matrix import FirstReleaseObservation
from regime_allocation.features.claims import build_claims_ar1_features
from regime_allocation.features.release_evidence import (
    EVENT_TABLE_COLUMNS,
    build_claims_release_events,
    build_monthly_release_events,
    validate_event_table,
)


def _month_end(reference_month: pd.Timestamp) -> pd.Timestamp:
    return reference_month.to_period("M").end_time.normalize()


def _monthly_records(
    transformed: list[float],
    *,
    series_id: str = "RSAFS",
    start: str = "2020-01-01",
) -> pd.DataFrame:
    reference_months = pd.date_range(start, periods=len(transformed), freq="MS")
    release_dates = [
        _month_end(month) + pd.Timedelta(days=12) for month in reference_months
    ]
    current = np.exp(np.cumsum(np.asarray(transformed, dtype=float) / 100.0)) * 100.0
    previous = current / np.exp(np.asarray(transformed, dtype=float) / 100.0)
    return pd.DataFrame(
        {
            "reference_month": reference_months,
            "series_id": series_id,
            "release_date": release_dates,
            "current_value": current,
            "previous_value_as_of_release": previous,
            "transform": "log_difference",
            "transformed_value": transformed,
            "release_lag_days": [
                (release - _month_end(reference)).days
                for release, reference in zip(release_dates, reference_months)
            ],
        }
    )


def _build_monthly(
    records: pd.DataFrame,
    *,
    block: str = "retail_sales",
    feature: str = "retail_sales_log_change",
    min_history: int = 2,
) -> pd.DataFrame:
    return build_monthly_release_events(
        records,
        release_block=block,
        feature_name=feature,
        frequency="monthly",
        provider_id="test-provider",
        source_url="https://example.test/series",
        min_standardization_history=min_history,
        ddof=1,
    )


def _claims_observations(periods: int = 14) -> list[FirstReleaseObservation]:
    references = [date(2020, 1, 4) + timedelta(days=7 * index) for index in range(periods)]
    time = np.arange(periods, dtype=float)
    levels = np.exp(11.8 + 0.001 * time + 0.06 * np.sin(time / 2.7))
    return [
        FirstReleaseObservation(
            reference_date=reference,
            release_date=reference + timedelta(days=5),
            value=float(value),
        )
        for reference, value in zip(references, levels)
    ]


def _build_claims(
    observations: list[FirstReleaseObservation],
    *,
    series_id: str = "ICSA",
    feature_name: str = "initial_claims_innovation",
) -> pd.DataFrame:
    return build_claims_release_events(
        observations,
        release_block="claims",
        feature_name=feature_name,
        series_id=series_id,
        provider_id="test-provider",
        source_url="https://example.test/series",
        min_ar_history=5,
        min_standardization_history=3,
        ddof=1,
        max_release_lag_days=92,
    )


def test_monthly_output_has_frozen_contract_provenance_and_reference_mapping() -> None:
    records = _monthly_records([1.0, 3.0, 5.0, 7.0])
    # The builder normalizes any date within a reference month to month-start.
    records.loc[0, "reference_month"] = pd.Timestamp("2020-01-19")

    events = _build_monthly(records)

    assert tuple(events.columns) == EVENT_TABLE_COLUMNS
    assert events["reference_date"].tolist() == list(
        pd.date_range("2020-01-01", periods=4, freq="MS")
    )
    pd.testing.assert_series_equal(
        events["reference_date"], events["reference_month"], check_names=False
    )
    assert events["release_block"].eq("retail_sales").all()
    assert events["frequency"].eq("monthly").all()
    assert events["provider_id"].eq("test-provider").all()
    assert events["source_url"].eq("https://example.test/series").all()
    assert not events["is_target_defining"].any()
    assert events.loc[0, "event_id"] == "retail_sales:2020-02-12:2020-01-01"
    assert events.loc[0, "event_group_id"] == "retail_sales:2020-02-12"
    validate_event_table(events)


def test_monthly_standardization_uses_strictly_prior_releases() -> None:
    events = _build_monthly(_monthly_records([1.0, 3.0, 5.0, 1_000.0]))

    expected_mean = np.mean([1.0, 3.0])
    expected_std = np.std([1.0, 3.0], ddof=1)
    assert events.loc[2, "standardization_prior_count"] == 2
    assert events.loc[2, "standardization_prior_mean"] == pytest.approx(expected_mean)
    assert events.loc[2, "standardization_prior_std"] == pytest.approx(expected_std)
    assert events.loc[2, "feature_value"] == pytest.approx(
        (5.0 - expected_mean) / expected_std
    )

    changed = _monthly_records([1.0, 3.0, -5_000.0, -8_000.0])
    changed_events = _build_monthly(changed)
    pd.testing.assert_frame_equal(
        events.iloc[:2],
        changed_events.iloc[:2],
    )
    assert changed_events.loc[2, "standardization_prior_mean"] == pytest.approx(
        expected_mean
    )
    assert changed_events.loc[2, "standardization_prior_std"] == pytest.approx(
        expected_std
    )


def test_same_date_monthly_catch_up_rows_share_one_strict_prior_information_set() -> None:
    records = _monthly_records([1.0, 3.0, 5.0, 9.0])
    catch_up_date = pd.Timestamp("2020-05-15")
    records.loc[2:3, "release_date"] = catch_up_date
    records.loc[2:3, "release_lag_days"] = [
        (catch_up_date - _month_end(reference)).days
        for reference in records.loc[2:3, "reference_month"]
    ]

    events = _build_monthly(records)
    catch_up = events.loc[events["release_date"] == catch_up_date]

    assert len(catch_up) == 2
    assert catch_up["event_group_id"].nunique() == 1
    assert catch_up["event_id"].nunique() == 2
    assert catch_up["standardization_prior_count"].tolist() == [2, 2]
    np.testing.assert_allclose(catch_up["standardization_prior_mean"], 2.0)
    np.testing.assert_allclose(
        catch_up["standardization_prior_std"], np.std([1.0, 3.0], ddof=1)
    )
    expected = (np.asarray([5.0, 9.0]) - 2.0) / np.std([1.0, 3.0], ddof=1)
    np.testing.assert_allclose(catch_up["feature_value"], expected)


def test_distinct_features_in_one_release_block_share_event_identifiers() -> None:
    headline = _build_monthly(_monthly_records([1.0, 3.0, 5.0], series_id="RSAFS"))
    ex_auto = _build_monthly(
        _monthly_records([2.0, 4.0, 8.0], series_id="RSFSXMV"),
        feature="retail_sales_ex_motor_vehicles_log_change",
    )
    combined = pd.concat([headline, ex_auto]).sort_values(
        ["release_date", "feature_name"]
    ).reset_index(drop=True)

    for _, same_release in combined.groupby("release_date"):
        assert same_release["event_group_id"].nunique() == 1
        assert same_release["event_id"].nunique() == 1
    validate_event_table(combined)


def test_monthly_warmup_and_zero_variance_are_distinguished() -> None:
    events = _build_monthly(_monthly_records([1.0, 1.0, 1.0, 2.0, 4.0]))

    assert events["feature_status"].tolist() == [
        "standardization_warmup",
        "standardization_warmup",
        "standardization_unidentified",
        "standardization_unidentified",
        "available",
    ]
    assert events.loc[:3, "feature_value"].isna().all()
    assert pd.notna(events.loc[4, "feature_value"])
    validate_event_table(events)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.drop(columns="transformed_value"),
            "missing columns.*transformed_value",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate reference months",
        ),
        (
            lambda frame: frame.assign(series_id=["RSAFS", "OTHER", "RSAFS"]),
            "exactly one series",
        ),
        (
            lambda frame: frame.assign(release_lag_days=0),
            "release lags do not match",
        ),
        (
            lambda frame: frame.assign(transformed_value=[1.0, np.inf, 3.0]),
            "transformed values must be finite",
        ),
    ],
)
def test_invalid_monthly_records_are_rejected(mutator: object, message: str) -> None:
    records = _monthly_records([1.0, 2.0, 3.0])
    invalid = mutator(records)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        _build_monthly(invalid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"release_block": "bad:block"}, "reserved character"),
        ({"feature_name": "  "}, "cannot be empty"),
        ({"min_standardization_history": 1}, "at least 2"),
        ({"min_standardization_history": 2, "ddof": 2}, "ddof.*smaller"),
    ],
)
def test_invalid_monthly_configuration_is_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "release_block": "retail_sales",
        "feature_name": "retail_sales_log_change",
        "frequency": "monthly",
        "provider_id": "test-provider",
        "source_url": "https://example.test/series",
        "min_standardization_history": 2,
        "ddof": 1,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_monthly_release_events(
            _monthly_records([1.0, 2.0, 3.0]),
            **arguments,  # type: ignore[arg-type]
        )


def test_claims_events_match_direct_expanding_ar_features_and_audit() -> None:
    observations = _claims_observations()
    events = _build_claims(observations)
    levels = pd.Series(
        [item.value for item in observations],
        index=pd.DatetimeIndex(item.reference_date for item in observations),
    )
    expected = build_claims_ar1_features(
        levels,
        release_dates=[item.release_date for item in observations],
        min_history=5,
        min_standardization_history=3,
        ddof=1,
    )

    assert tuple(events.columns) == EVENT_TABLE_COLUMNS
    assert events["feature_status"].iloc[:5].eq("ar_warmup").all()
    assert events["feature_status"].iloc[5:8].eq("standardization_warmup").all()
    assert events["feature_status"].iloc[8:].eq("available").all()
    np.testing.assert_allclose(
        events["feature_value"], expected["standardized_innovation"], equal_nan=True
    )
    np.testing.assert_allclose(
        events["transformed_value"], expected["innovation_log"], equal_nan=True
    )
    np.testing.assert_allclose(
        events["standardization_prior_mean"],
        expected["innovation_prior_mean"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        events["standardization_prior_std"],
        expected["innovation_prior_std"],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        events["ar_prior_observation_count"],
        expected["ar_prior_observation_count"],
    )
    assert events["transform"].eq("expanding_log_ar1_innovation").all()
    assert events["release_lag_days"].eq(5).all()
    assert events["reference_date"].dt.dayofweek.eq(5).all()
    validate_event_table(events)


def test_claims_builder_is_causal_under_current_and_future_shocks() -> None:
    observations = _claims_observations(periods=16)
    changed = list(observations)
    for position in range(10, len(changed)):
        item = changed[position]
        changed[position] = FirstReleaseObservation(
            reference_date=item.reference_date,
            release_date=item.release_date,
            value=item.value * 50.0,
        )

    baseline = _build_claims(observations)
    shocked = _build_claims(changed)
    causal_audit = [
        "ar_intercept",
        "ar_lag1_coefficient",
        "forecast_log_level",
        "forecast_level",
        "standardization_prior_count",
        "standardization_prior_mean",
        "standardization_prior_std",
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[:9, causal_audit],
        shocked.loc[:9, causal_audit],
    )
    assert baseline.loc[10, "innovation_log"] != pytest.approx(
        shocked.loc[10, "innovation_log"]
    )
    # The shocked current observation cannot alter its own forecast or priors.
    pd.testing.assert_series_equal(
        baseline.loc[10, causal_audit],
        shocked.loc[10, causal_audit],
    )


def test_claims_same_publication_date_groups_components_but_not_reference_weeks() -> None:
    initial_observations = _claims_observations(periods=12)
    continued_observations = [
        FirstReleaseObservation(
            reference_date=item.reference_date - timedelta(days=7),
            release_date=item.release_date,
            value=item.value * 8.0,
        )
        for item in initial_observations
    ]
    initial = _build_claims(initial_observations)
    continued = _build_claims(
        continued_observations,
        series_id="CCSA",
        feature_name="continued_claims_innovation",
    )
    combined = pd.concat([initial, continued]).sort_values(
        ["release_date", "feature_name"]
    ).reset_index(drop=True)

    for _, same_release in combined.groupby("release_date"):
        assert same_release["event_group_id"].nunique() == 1
        assert same_release["event_id"].nunique() == 2
    validate_event_table(combined)


def test_claims_catch_up_batch_freezes_fit_and_prior_across_reference_weeks() -> None:
    observations = _claims_observations(periods=16)
    catch_up_date = observations[12].release_date
    for position in range(10, 13):
        item = observations[position]
        observations[position] = FirstReleaseObservation(
            reference_date=item.reference_date,
            release_date=catch_up_date,
            value=item.value,
        )

    events = _build_claims(observations)
    batch = events.loc[events["release_date"] == pd.Timestamp(catch_up_date)]

    assert len(batch) == 3
    assert batch["reference_date"].is_monotonic_increasing
    assert batch["event_group_id"].nunique() == 1
    assert batch["event_id"].nunique() == 3
    assert batch["ar_prior_observation_count"].eq(10).all()
    assert batch["ar_transition_count"].eq(9).all()
    assert batch["ar_intercept"].nunique() == 1
    assert batch["ar_lag1_coefficient"].nunique() == 1
    assert batch["standardization_prior_count"].nunique() == 1
    assert batch["standardization_prior_mean"].nunique() == 1
    assert batch["standardization_prior_std"].nunique() == 1
    validate_event_table(events)


def test_claims_archive_start_latest_only_drops_bootstrap_history() -> None:
    observations = _claims_observations(periods=8)
    bootstrap_release = observations[2].release_date
    for position in range(3):
        item = observations[position]
        observations[position] = FirstReleaseObservation(
            reference_date=item.reference_date,
            release_date=bootstrap_release,
            value=item.value,
        )

    filtered = _build_claims(observations)
    unfiltered = build_claims_release_events(
        observations,
        release_block="claims",
        feature_name="initial_claims_innovation",
        series_id="ICSA",
        provider_id="test-provider",
        source_url="https://example.test/series",
        min_ar_history=5,
        min_standardization_history=3,
        ddof=1,
        max_release_lag_days=92,
        archive_start_latest_only=False,
    )

    assert filtered["reference_date"].tolist() == [
        pd.Timestamp(item.reference_date) for item in observations[2:]
    ]
    assert unfiltered["reference_date"].tolist() == [
        pd.Timestamp(item.reference_date) for item in observations
    ]
    assert filtered["ar_prior_observation_count"].tolist() == list(
        range(len(filtered))
    )


def test_claims_filters_ineligible_release_lags_before_feature_construction() -> None:
    observations = _claims_observations(periods=10)
    too_early = observations[0]
    too_late = observations[1]
    observations[0] = FirstReleaseObservation(
        reference_date=too_early.reference_date,
        release_date=too_early.reference_date - timedelta(days=1),
        value=too_early.value,
    )
    observations[1] = FirstReleaseObservation(
        reference_date=too_late.reference_date,
        release_date=too_late.reference_date + timedelta(days=100),
        value=too_late.value,
    )

    events = _build_claims(observations)

    assert len(events) == 8
    assert events["ar_prior_observation_count"].tolist() == list(range(8))
    assert events["release_lag_days"].between(0, 92).all()


def test_constant_claims_history_is_reported_as_unidentified_after_ar_warmup() -> None:
    observations = _claims_observations(periods=10)
    observations = [
        FirstReleaseObservation(item.reference_date, item.release_date, 200_000.0)
        for item in observations
    ]

    events = _build_claims(observations)

    assert events["feature_status"].iloc[:5].eq("ar_warmup").all()
    assert events["feature_status"].iloc[5:].eq("ar_unidentified").all()
    assert events["innovation_log"].isna().all()
    assert events["feature_value"].isna().all()
    validate_event_table(events)


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        (
            [
                FirstReleaseObservation(date(2020, 1, 4), date(2020, 1, 9), 1.0),
                FirstReleaseObservation(date(2020, 1, 4), date(2020, 1, 16), 2.0),
            ],
            "duplicate reference dates",
        ),
        (
            [
                FirstReleaseObservation(date(2020, 1, 11), date(2020, 1, 16), 1.0),
                FirstReleaseObservation(date(2020, 1, 4), date(2020, 1, 17), 2.0),
            ],
            "increase with publication order",
        ),
        (
            [
                FirstReleaseObservation(date(2020, 1, 4), date(2020, 1, 9), 1.0),
                FirstReleaseObservation(date(2020, 1, 18), date(2020, 1, 23), 2.0),
            ],
            "unbroken weekly sequence",
        ),
    ],
)
def test_invalid_claims_sequences_are_rejected(
    observations: list[FirstReleaseObservation], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_claims(observations)


def test_validate_event_table_rejects_each_contract_violation() -> None:
    valid = _build_monthly(_monthly_records([1.0, 3.0, 5.0]))
    mutations: list[tuple[pd.DataFrame, str]] = []
    mutations.append((valid.drop(columns="source_url"), "frozen contract"))
    mutations.append((valid.iloc[0:0], "cannot be empty"))
    missing_id = valid.copy()
    missing_id.loc[0, "event_id"] = np.nan
    mutations.append((missing_id, "identifiers cannot be missing"))
    duplicate = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
    mutations.append((duplicate, "duplicate feature rows"))
    defining = valid.copy()
    defining.loc[0, "is_target_defining"] = True
    mutations.append((defining, "target-defining row"))
    available_missing = valid.copy()
    available_missing.loc[2, "feature_value"] = np.nan
    mutations.append((available_missing, "available event rows"))
    unavailable_value = valid.copy()
    unavailable_value.loc[0, "feature_value"] = 1.0
    mutations.append((unavailable_value, "unavailable event rows"))
    impossible_date = valid.copy()
    impossible_date.loc[0, "release_date"] = pd.Timestamp("2019-12-01")
    mutations.append((impossible_date, "cannot precede"))

    for invalid, message in mutations:
        with pytest.raises(ValueError, match=message):
            validate_event_table(invalid)
