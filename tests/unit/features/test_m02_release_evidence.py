"""Protect Model 02's series-specific contemporaneous-release semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_allocation.features.m02_release_evidence import (
    extract_contemporaneous_monthly_features,
)


def test_contemporaneous_extractor_keeps_same_month_releases_and_bootstrap_policy() -> None:
    months = pd.date_range("2021-08-01", "2022-02-01", freq="MS")
    levels = pd.Series(
        [1.8, 1.9, 2.0, 2.1, 2.05, 2.2, 2.3],
        index=months,
    )
    matrix = pd.DataFrame(index=months)
    # The selected archive begins in November and exposes older history. Only
    # November is a genuine real-time row at that first snapshot.
    first_release = pd.Timestamp("2021-11-10")
    first_values = np.full(len(months), np.nan)
    first_values[:4] = levels.iloc[:4]
    matrix[f"EXPINF1YR_{first_release:%Y%m%d}"] = first_values
    for position in range(4, len(months)):
        release = months[position] + pd.Timedelta(days=9)
        values = np.full(len(months), np.nan)
        values[: position + 1] = levels.iloc[: position + 1]
        matrix[f"EXPINF1YR_{release:%Y%m%d}"] = values

    records = extract_contemporaneous_monthly_features(
        matrix,
        series_id="EXPINF1YR",
        component="inflation_pressure",
        transform="difference",
        archive_start_latest_only=True,
    )

    assert records["reference_month"].tolist() == list(months[3:])
    assert (
        records["release_date"]
        <= records["reference_month"].dt.to_period("M").dt.end_time
    ).all()
    assert (records["release_lag_days"] < 0).all()
    assert records.iloc[0]["transformed_value"] == levels.iloc[3] - levels.iloc[2]
    assert records.attrs["extraction_diagnostics"][
        "rows_excluded_archive_bootstrap"
    ] == 3


def test_contemporaneous_extractor_rejects_next_month_first_appearance() -> None:
    months = pd.date_range("2022-01-01", periods=3, freq="MS")
    matrix = pd.DataFrame(index=months)
    for position, month in enumerate(months):
        release = (month.to_period("M") + 1).to_timestamp() + pd.Timedelta(days=5)
        values = np.full(len(months), np.nan)
        values[: position + 1] = [2.0 + 0.1 * index for index in range(position + 1)]
        matrix[f"EXPINF1YR_{release:%Y%m%d}"] = values

    records = extract_contemporaneous_monthly_features(
        matrix,
        series_id="EXPINF1YR",
        component="inflation_pressure",
        transform="difference",
        archive_start_latest_only=False,
    )

    assert records.empty
    assert records.attrs["extraction_diagnostics"][
        "rows_excluded_outside_reference_month"
    ] == len(months)
