"""Prove M03 source audits depend only on visible, same-vintage information."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from regime_allocation.data.source_availability import (
    extract_source_features,
    extract_source_observations,
)


def _matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TEST_20200110": [90.0, 100.0, None, None, None, None],
            "TEST_20200207": [91.0, 102.0, 112.0, None, None, None],
            "TEST_20200306": [92.0, 103.0, 113.0, 120.0, None, None],
        },
        index=pd.to_datetime(
            ["2019-11-01", "2019-12-01", "2020-01-01", "2020-02-01", "2020-03-01", "2020-05-01"]
        ),
    )


def _features(matrix: pd.DataFrame, **kwargs) -> pd.DataFrame:
    options = {
        "series_id": "TEST",
        "component": "test_component",
        "transform": "difference",
        "reference_start": date(2019, 11, 1),
        "reference_end": date(2020, 5, 1),
        "as_of": datetime(2020, 6, 1, tzinfo=UTC),
        "snapshot_id": "sha256:fixture",
    }
    options.update(kwargs)
    return extract_source_features(matrix, **options)


def test_complete_month_ledger_and_same_vintage_prior() -> None:
    ledger = _features(_matrix()).set_index("reference_month")

    assert list(ledger.index) == list(pd.date_range("2019-11-01", "2020-05-01", freq="MS"))
    assert ledger["reason"].tolist() == [
        "archive_bootstrap",
        "retained",
        "retained",
        "retained",
        "unavailable_by_asof",
        "absent_reference",
        "unavailable_by_asof",
    ]
    january = ledger.loc["2020-01-01"]
    assert january["current_value"] == 112
    assert january["previous_value_as_of_release"] == 102
    assert january["transformed_value"] == 10
    assert january["archive_vintage_date"] == pd.Timestamp("2020-02-07")
    assert january["archive_available_at"] == pd.Timestamp("2020-02-08T05:00:00Z")
    assert ledger["publication_timestamp"].isna().all()
    assert ledger["timestamp_precision"].eq("date").all()
    assert ledger["snapshot_id"].eq("sha256:fixture").all()
    assert ledger.loc[ledger["status"].eq("excluded"), "transformed_value"].isna().all()


def test_future_vintage_perturbations_cannot_change_features_or_exclusions() -> None:
    prefix = _matrix()
    full = prefix.copy()
    full["TEST_20200910"] = [999, "bad", -1, np.inf, 42, 88]
    full["TEST_20200810"] = [None, 10, 11, 12, None, None]
    expected = _features(prefix)

    assert_frame_equal(_features(full), expected)
    full["TEST_20200810"] = [np.inf, 1, None, "broken", None, 3]
    assert_frame_equal(_features(full), expected)


def test_prefix_of_visible_columns_matches_full_archive_at_each_cutoff() -> None:
    matrix = _matrix()
    for count, cutoff in enumerate(
        [
            "2020-01-11T05:00:00+00:00",
            "2020-02-08T05:00:00+00:00",
            "2020-03-07T05:00:00+00:00",
        ],
        1,
    ):
        now = datetime.fromisoformat(cutoff)
        assert_frame_equal(
            _features(matrix, as_of=now), _features(matrix.iloc[:, :count], as_of=now)
        )


def test_output_range_does_not_change_archive_bootstrap_policy() -> None:
    matrix = _matrix()
    full = _features(matrix)
    subset = _features(matrix, reference_start=date(2019, 11, 1), reference_end=date(2019, 11, 30))

    assert_frame_equal(subset, full.iloc[:1].reset_index(drop=True))
    assert subset.iloc[0]["reason"] == "archive_bootstrap"


def test_impossible_future_reference_cannot_affect_bootstrap_selection() -> None:
    matrix = _matrix()
    expected = _features(matrix)
    matrix.loc[pd.Timestamp("2030-01-01")] = [7, 8, 9]

    assert_frame_equal(_features(matrix), expected)


@pytest.mark.parametrize(
    "vintage,eligible",
    [
        ("20200307", "2020-03-08T05:00:00+00:00"),
        ("20200308", "2020-03-09T04:00:00+00:00"),
        ("20201031", "2020-11-01T04:00:00+00:00"),
        ("20201101", "2020-11-02T05:00:00+00:00"),
    ],
)
def test_date_only_eligibility_is_next_local_midnight_across_dst(vintage, eligible) -> None:
    now = datetime.fromisoformat(eligible)
    reference = pd.Timestamp(vintage[:4] + "-" + vintage[4:6] + "-01") - pd.offsets.MonthBegin(1)
    prior = reference - pd.offsets.MonthBegin(1)
    matrix = pd.DataFrame({f"TEST_{vintage}": [100.0, 110.0]}, index=[prior, reference])
    options = {"reference_start": reference.date(), "reference_end": reference.date()}
    before = _features(
        matrix, **options, as_of=(pd.Timestamp(now) - pd.Timedelta(seconds=1)).to_pydatetime()
    )
    at = _features(matrix, **options, as_of=now)

    assert before.iloc[0]["reason"] == "unavailable_by_asof"
    assert before["archive_vintage_date"].isna().all()
    assert at.iloc[0]["status"] == "retained"
    assert at.iloc[0]["archive_available_at"] == pd.Timestamp(now)


def test_no_visible_vintage_returns_explicit_no_feature_ledger() -> None:
    matrix = pd.DataFrame(
        {"TEST_20200810": [10, 11]}, index=pd.to_datetime(["2020-01-01", "2020-02-01"])
    )
    ledger = _features(matrix, reference_start=date(2019, 12, 1), reference_end=date(2020, 2, 1))

    assert len(ledger) == 3
    assert ledger["reason"].eq("unavailable_by_asof").all()
    assert ledger["current_value"].isna().all()
    assert ledger["archive_available_at"].isna().all()
    assert str(ledger["archive_available_at"].dtype) == "datetime64[ns, UTC]"


def test_active_support_keeps_every_month_and_permits_same_vintage_warmup() -> None:
    ledger = _features(_matrix(), active_start=date(2019, 12, 1), active_end=date(2020, 1, 31))

    assert ledger["reason"].tolist() == [
        "outside_active_support",
        "retained",
        "retained",
        "outside_active_support",
        "outside_active_support",
        "outside_active_support",
        "outside_active_support",
    ]
    assert ledger.iloc[1]["previous_value_as_of_release"] == 90.0


@pytest.mark.parametrize(
    "current,prior,reason",
    [
        (np.inf, 100.0, "invalid_current_value"),
        ("invalid", 100.0, "invalid_current_value"),
        (True, 100.0, "invalid_current_value"),
        (110.0, np.inf, "invalid_prior_value"),
        (110.0, "invalid", "invalid_prior_value"),
        (110.0, np.nan, "missing_prior_value"),
        (0.0, 100.0, "nonpositive_log_levels"),
        (110.0, -1.0, "nonpositive_log_levels"),
    ],
)
def test_invalid_first_appearance_cannot_be_repaired_by_later_vintage(
    current, prior, reason
) -> None:
    matrix = pd.DataFrame(
        {"TEST_20200110": [prior, current], "TEST_20200207": [100.0, 115.0]},
        index=pd.to_datetime(["2019-11-01", "2019-12-01"]),
    )
    ledger = _features(
        matrix,
        reference_start=date(2019, 12, 1),
        reference_end=date(2019, 12, 1),
        transform="log_difference",
    )

    row = ledger.iloc[0]
    assert row["status"] == "excluded"
    assert row["reason"] == reason
    assert row["archive_vintage_date"] == pd.Timestamp("2020-01-10")
    assert pd.isna(row["transformed_value"])


def test_missing_prior_reference_is_distinct_from_missing_prior_value() -> None:
    matrix = pd.DataFrame({"TEST_20200110": [110.0]}, index=pd.to_datetime(["2019-12-01"]))
    ledger = _features(matrix, reference_start=date(2019, 12, 1), reference_end=date(2019, 12, 1))

    assert ledger.iloc[0]["reason"] == "missing_prior_reference"


@pytest.mark.parametrize(
    "transform,expected",
    [
        ("difference", 10.0),
        ("negative_difference", -10.0),
        ("log_difference", 100 * math.log(1.1)),
    ],
)
def test_supported_transforms(transform, expected) -> None:
    matrix = pd.DataFrame(
        {"TEST_20200110": [100.0, 110.0]}, index=pd.to_datetime(["2019-11-01", "2019-12-01"])
    )
    ledger = _features(
        matrix,
        reference_start=date(2019, 12, 1),
        reference_end=date(2019, 12, 1),
        transform=transform,
    )

    assert ledger.iloc[0]["transformed_value"] == pytest.approx(expected)


def test_numeric_overflow_is_audited_but_log_ratio_avoids_unnecessary_overflow() -> None:
    index = pd.to_datetime(["2019-11-01", "2019-12-01"])
    options = {"reference_start": date(2019, 12, 1), "reference_end": date(2019, 12, 1)}
    overflow = _features(pd.DataFrame({"TEST_20200110": [-1e308, 1e308]}, index=index), **options)
    stable_log = _features(
        pd.DataFrame({"TEST_20200110": [1e-308, 1e308]}, index=index),
        **options,
        transform="log_difference",
    )

    assert overflow.iloc[0]["reason"] == "invalid_transformed_value"
    assert stable_log.iloc[0]["status"] == "retained"
    assert math.isfinite(stable_log.iloc[0]["transformed_value"])


def test_negative_lag_and_late_archive_backfill_have_distinct_reasons() -> None:
    matrix = pd.DataFrame(
        {"TEST_20200306": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2019-11-01", "2019-12-01", "2020-03-01"]),
    )
    ledger = _features(matrix, archive_start_latest_only=False, max_release_lag_days=30)

    assert (
        ledger.set_index("reference_month").loc["2019-12-01", "reason"] == "excessive_release_lag"
    )
    assert ledger.set_index("reference_month").loc["2020-03-01", "reason"] == "negative_release_lag"


def test_future_requested_month_remains_an_excluded_ledger_row() -> None:
    ledger = _features(_matrix(), as_of=datetime(2020, 3, 7, 5, tzinfo=UTC))

    assert (
        ledger.loc[ledger["reference_month"].ge("2020-04-01"), "reason"]
        .eq("future_reference")
        .all()
    )


@pytest.mark.parametrize("kind", ["reference", "vintage"])
def test_duplicate_dates_are_rejected(kind) -> None:
    matrix = _matrix()
    if kind == "reference":
        matrix = pd.concat([matrix, matrix.iloc[:1]])
    else:
        matrix["OTHER_20200110"] = matrix["TEST_20200110"]

    with pytest.raises(ValueError, match=f"duplicate {kind} dates"):
        _features(matrix)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"as_of": datetime(2020, 3, 7)}, "timezone-aware"),  # noqa: DTZ001
        ({"max_release_lag_days": -1}, "nonnegative"),
        ({"max_release_lag_days": 1.5}, "integer"),
        ({"archive_start_latest_only": "true"}, "boolean"),
        ({"reference_start": date(2021, 1, 1)}, "reference_start"),
        ({"active_start": date(2021, 1, 1), "active_end": date(2020, 1, 1)}, "active_start"),
        ({"transform": "percent_change"}, "unsupported transform"),
    ],
)
def test_invalid_extraction_configuration(options, match) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        _features(_matrix(), **options)


def test_observation_ledger_has_native_weekly_calendar_without_feature_policies() -> None:
    matrix = pd.DataFrame(
        {"ICNSA_20200110": [10.0, 11.0, None], "ICNSA_20200117": [99.0, 12.0, 13.0]},
        index=pd.to_datetime(["2019-12-28", "2020-01-04", "2020-01-11"]),
    )
    ledger = extract_source_observations(
        matrix,
        series_id="ICNSA",
        frequency="weekly_saturday",
        snapshot_id="fixture",
        reference_start=date(2019, 12, 27),
        reference_end=date(2020, 1, 18),
        as_of=datetime(2020, 1, 18, 5, tzinfo=UTC),
    )

    assert ledger["reference_date"].tolist() == list(
        pd.date_range("2019-12-28", "2020-01-18", freq="W-SAT")
    )
    assert ledger["current_value"].iloc[:3].tolist() == [10, 11, 13]
    assert ledger["reason"].tolist() == ["archive_first_observed"] * 3 + ["absent_reference"]
    assert ledger["transformed_value"].isna().all()
    assert ledger["previous_value_as_of_release"].isna().all()
    assert ledger["publication_timestamp"].isna().all()
    assert ledger["observation_semantics"].eq("archive_first_observed").all()


def test_observation_audit_is_prefix_invariant_and_does_not_repair_invalid_level() -> None:
    matrix = pd.DataFrame(
        {"TEST_20200201": [float("inf")], "TEST_20200301": [100.0]},
        index=pd.to_datetime(["2020-01-01"]),
    )
    options = {
        "series_id": "TEST",
        "reference_start": date(2020, 1, 1),
        "reference_end": date(2020, 1, 31),
        "as_of": datetime(2020, 2, 2, 5, tzinfo=UTC),
    }
    actual = extract_source_observations(matrix, **options)
    assert actual.iloc[0]["reason"] == "invalid_current_value"
    assert_frame_equal(actual, extract_source_observations(matrix.iloc[:, :1], **options))
    options["as_of"] = datetime(2020, 4, 1, tzinfo=UTC)
    assert (
        extract_source_observations(matrix, **options).iloc[0]["reason"] == "invalid_current_value"
    )


def test_observation_ledger_rejects_mismatched_weekly_calendar() -> None:
    matrix = pd.DataFrame({"TEST_20200110": [1.0]}, index=pd.to_datetime(["2020-01-03"]))
    with pytest.raises(ValueError, match="Saturdays"):
        extract_source_observations(
            matrix,
            series_id="TEST",
            frequency="weekly_saturday",
            reference_start=date(2020, 1, 1),
            reference_end=date(2020, 1, 31),
            as_of=datetime(2020, 2, 1, tzinfo=UTC),
        )


def test_empty_archive_preserves_requested_calendar_and_no_observations() -> None:
    ledger = _features(pd.DataFrame())

    assert len(ledger) == 7
    assert ledger["reason"].eq("unavailable_by_asof").all()
    assert ledger["transformed_value"].isna().all()


def test_weekly_observation_support_and_future_rows_are_explicit() -> None:
    matrix = pd.DataFrame(
        {"ICNSA_20200110": [10.0, 11.0, 12.0]},
        index=pd.to_datetime(["2019-12-28", "2020-01-04", "2020-01-11"]),
    )
    ledger = extract_source_observations(
        matrix,
        series_id="ICNSA",
        frequency="weekly_saturday",
        reference_start=date(2019, 12, 27),
        reference_end=date(2020, 1, 18),
        active_start=date(2020, 1, 1),
        as_of=datetime(2020, 1, 11, 5, tzinfo=UTC),
    )

    assert ledger["reason"].tolist() == [
        "outside_active_support",
        "archive_first_observed",
        "reference_after_vintage",
        "future_reference",
    ]


def test_custom_availability_timezone_is_used_without_inventing_publication_time() -> None:
    ledger = _features(
        _matrix(),
        reference_start=date(2019, 12, 1),
        reference_end=date(2019, 12, 31),
        as_of=datetime(2020, 1, 11, tzinfo=UTC),
        availability_timezone="UTC",
    )

    assert ledger.iloc[0]["status"] == "retained"
    assert ledger.iloc[0]["archive_available_at"] == pd.Timestamp("2020-01-11T00:00:00Z")
    assert pd.isna(ledger.iloc[0]["publication_timestamp"])


def test_empty_weekly_range_still_has_a_stable_ledger_schema() -> None:
    ledger = extract_source_observations(
        pd.DataFrame(),
        series_id="ICNSA",
        frequency="weekly_saturday",
        reference_start=date(2020, 1, 6),
        reference_end=date(2020, 1, 10),
        as_of=datetime(2020, 1, 11, 5, tzinfo=UTC),
    )

    assert ledger.empty
    assert str(ledger["reference_date"].dtype) == "datetime64[ns]"
    assert str(ledger["archive_available_at"].dtype) == "datetime64[ns, UTC]"
