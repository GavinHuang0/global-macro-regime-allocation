"""Unit tests for the additive Model 02 evidence-experiment transforms."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from regime_allocation.features.m02_evidence_experiments import (
    extract_first_release_levels,
    extract_first_release_log_change,
    same_vintage_log_ratio_matrix,
)


def _matrix(values: dict[str, list[float]]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(next(iter(values.values()))), freq="MS")
    return pd.DataFrame(values, index=index)


def test_contemporaneous_level_excludes_archive_backfill() -> None:
    matrix = _matrix(
        {
            "SURVEY_20200315": [1.0, 2.0, 3.0],
            "SURVEY_20200415": [1.0, 2.0, 3.0],
        }
    )
    result = extract_first_release_levels(
        matrix,
        series_id="SURVEY",
        component="survey",
        contemporaneous=True,
        archive_start_latest_only=True,
    )
    assert list(result["reference_month"]) == [pd.Timestamp("2020-03-01")]
    assert result.iloc[0]["transformed_value"] == 3.0
    assert result.iloc[0]["release_lag_days"] == -16
    assert result.attrs["extraction_diagnostics"][
        "rows_excluded_archive_bootstrap"
    ] == 2


def test_retrospective_level_rejects_negative_lag() -> None:
    matrix = _matrix(
        {
            "LEVEL_20200215": [10.0, 11.0],
            "LEVEL_20200315": [10.0, 11.0],
        }
    )
    result = extract_first_release_levels(
        matrix,
        series_id="LEVEL",
        component="level",
        contemporaneous=False,
        archive_start_latest_only=False,
    )
    assert list(result["reference_month"]) == [pd.Timestamp("2020-01-01")]
    assert result.iloc[0]["release_date"] == pd.Timestamp("2020-02-15")


def test_lagged_log_change_uses_lag_from_same_vintage() -> None:
    index = pd.date_range("2019-01-01", periods=14, freq="MS")
    first = np.arange(100.0, 114.0)
    second = first + 1000.0
    # January 2020 first appears in the second snapshot.  Its lagged January
    # 2019 value must therefore be 1100, not the earlier snapshot's 100.
    first[12:] = np.nan
    matrix = pd.DataFrame(
        {"PRICE_20200115": first, "PRICE_20200215": second}, index=index
    )
    result = extract_first_release_log_change(
        matrix,
        series_id="PRICE",
        component="price",
        lag_months=12,
        transform_name="same_vintage_log_change_12m",
        archive_start_latest_only=False,
    )
    january_2020 = result.loc[
        result["reference_month"].eq(pd.Timestamp("2020-01-01"))
    ].iloc[0]
    assert january_2020["current_value"] == 1112.0
    assert january_2020["previous_value_as_of_release"] == 1100.0
    assert january_2020["transformed_value"] == pytest.approx(
        100.0 * math.log(1112.0 / 1100.0)
    )


def test_same_vintage_log_ratio_requires_exact_common_snapshot() -> None:
    numerator = _matrix(
        {
            "NUM_20200215": [200.0, 220.0],
            "NUM_20200315": [210.0, 231.0],
        }
    )
    denominator = _matrix(
        {
            "DEN_20200215": [100.0, 110.0],
            "DEN_20200415": [100.0, 110.0],
        }
    )
    result = same_vintage_log_ratio_matrix(
        numerator, denominator, derived_series_id="RATIO"
    )
    assert list(result.columns) == ["RATIO_20200215"]
    np.testing.assert_allclose(result.iloc[:, 0], 100.0 * math.log(2.0))


def test_log_ratio_masks_nonpositive_cells() -> None:
    numerator = _matrix({"NUM_20200215": [100.0, -1.0]})
    denominator = _matrix({"DEN_20200215": [50.0, 50.0]})
    result = same_vintage_log_ratio_matrix(
        numerator, denominator, derived_series_id="RATIO"
    )
    assert np.isfinite(result.iloc[0, 0])
    assert pd.isna(result.iloc[1, 0])
