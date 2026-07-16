"""Tests for release-coherent first-release feature extraction."""

from __future__ import annotations

import pandas as pd
import pytest

from regime_allocation.data.first_release import (
    apply_transform,
    extract_first_release_features,
)


def _revision_matrix() -> pd.DataFrame:
    """Return a matrix where each new month appears in a later vintage."""
    return pd.DataFrame(
        {
            "TEST_20200110": [90.0, 100.0, None, None],
            "TEST_20200207": [91.0, 102.0, 112.0, None],
            "TEST_20200306": [92.0, 103.0, 113.0, 120.0],
        },
        index=pd.to_datetime(
            ["2019-11-01", "2019-12-01", "2020-01-01", "2020-02-01"]
        ),
    )


def test_extract_selects_each_months_earliest_available_vintage() -> None:
    actual = extract_first_release_features(
        _revision_matrix(),
        series_id="TEST",
        component="test_component",
        transform="difference",
        max_release_lag_days=60,
    ).set_index("reference_month")

    assert list(actual.index) == [
        pd.Timestamp("2019-12-01"),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-01"),
    ]
    assert actual.loc[pd.Timestamp("2019-12-01"), "release_date"] == pd.Timestamp(
        "2020-01-10"
    )
    assert actual.loc[pd.Timestamp("2020-01-01"), "release_date"] == pd.Timestamp(
        "2020-02-07"
    )
    assert actual.loc[pd.Timestamp("2020-02-01"), "release_date"] == pd.Timestamp(
        "2020-03-06"
    )
    assert actual.loc[pd.Timestamp("2020-01-01"), "current_value"] == 112.0


def test_transform_uses_current_and_prior_month_from_the_same_release_vintage() -> None:
    actual = extract_first_release_features(
        _revision_matrix(),
        series_id="TEST",
        component="test_component",
        transform="difference",
        max_release_lag_days=60,
    ).set_index("reference_month")

    january = actual.loc[pd.Timestamp("2020-01-01")]

    # January first appears in the 2020-02-07 vintage. The corresponding
    # December value is 102, not December's initial-release value of 100 and
    # not its later revision of 103.
    assert january["previous_value_as_of_release"] == 102.0
    assert january["transformed_value"] == 10.0


def test_bulk_backfill_outside_release_lag_limit_is_rejected() -> None:
    matrix = pd.DataFrame(
        {"TEST_20200110": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"]),
    )

    actual = extract_first_release_features(
        matrix,
        series_id="TEST",
        component="test_component",
        transform="difference",
        max_release_lag_days=92,
    )

    assert actual.empty
    assert list(actual.columns) == [
        "reference_month",
        "component",
        "series_id",
        "release_date",
        "current_value",
        "previous_value_as_of_release",
        "transform",
        "transformed_value",
        "release_lag_days",
    ]


@pytest.mark.parametrize(
    ("current", "previous"),
    [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -1.0)],
)
def test_log_difference_rejects_nonpositive_levels(
    current: float,
    previous: float,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        apply_transform(current, previous, "log_difference")
