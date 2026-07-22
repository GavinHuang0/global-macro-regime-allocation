"""Construct Model 02 release-evidence features that Model 01 did not contain.

The functions in this module are deterministic, provider-neutral transforms.
They never retrieve data and never estimate a Bayesian observation model.  The
retail helper forms a motor-vehicle level from two exact same-vintage matrices;
the housing helper aligns first-release weekly mortgage rates to the later
housing publication and computes a reference-month average-rate change; and the
binary-control helper adds the documented Freddie Mac methodology-break
indicator without standardizing a zero-one control.

All release-dependent calculations use only observations whose publication
date is no later than the event they control.  Causal expanding
standardization remains delegated to :func:`build_monthly_release_events`, the
same tested implementation used by Model 01.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
import math

import numpy as np
import pandas as pd

from regime_allocation.data.providers.vintage_matrix import (
    FirstReleaseObservation,
    vintage_date_from_column,
)
from regime_allocation.data.first_release import apply_transform
from regime_allocation.features.release_evidence import EVENT_TABLE_COLUMNS


def extract_contemporaneous_monthly_features(
    matrix: pd.DataFrame,
    *,
    series_id: str,
    component: str,
    transform: str,
    archive_start_latest_only: bool = True,
) -> pd.DataFrame:
    """Extract same-vintage changes published within their reference month.

    This narrowly scoped rule supports contemporaneous monthly estimates such
    as ``EXPINF1YR``.  A reference month is eligible only when its first
    appearance is on or after that month's first calendar day and no later than
    its final calendar day.  ``release_lag_days`` remains measured from
    month-end for schema compatibility, so valid contemporaneous rows normally
    carry a negative value.  Retrospective releases must continue to use the
    generic extractor in :mod:`regime_allocation.data.first_release`.
    """

    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("contemporaneous feature input must be a pandas DataFrame")
    if matrix.index.has_duplicates:
        raise ValueError("contemporaneous matrix repeats a reference date")
    if not isinstance(archive_start_latest_only, bool):
        raise TypeError("archive_start_latest_only must be a boolean")
    ordered_columns = sorted(matrix.columns, key=vintage_date_from_column)
    ordered = matrix.reindex(columns=ordered_columns).sort_index()
    first_column = str(ordered_columns[0]) if ordered_columns else None
    archive_latest_reference = None
    if archive_start_latest_only and first_column is not None:
        first_snapshot = pd.to_numeric(ordered[first_column], errors="coerce")
        available = first_snapshot.index[first_snapshot.notna()]
        if len(available):
            archive_latest_reference = pd.Timestamp(available.max()).to_period("M")

    records: list[dict[str, object]] = []
    diagnostics = {
        "matrix_rows": len(ordered),
        "rows_without_any_vintage": 0,
        "rows_excluded_archive_bootstrap": 0,
        "rows_excluded_outside_reference_month": 0,
        "rows_excluded_missing_prior": 0,
    }
    for raw_reference, row in ordered.iterrows():
        available = pd.to_numeric(row, errors="coerce").dropna()
        if available.empty:
            diagnostics["rows_without_any_vintage"] += 1
            continue
        release_column = str(available.index[0])
        release_date = vintage_date_from_column(release_column)
        month = pd.Timestamp(raw_reference).to_period("M")
        if (
            archive_start_latest_only
            and release_column == first_column
            and archive_latest_reference is not None
            and month != archive_latest_reference
        ):
            diagnostics["rows_excluded_archive_bootstrap"] += 1
            continue
        month_start = month.start_time.date()
        month_end = month.end_time.normalize().date()
        if not month_start <= release_date <= month_end:
            diagnostics["rows_excluded_outside_reference_month"] += 1
            continue
        previous_month = (month - 1).to_timestamp()
        if previous_month not in ordered.index:
            diagnostics["rows_excluded_missing_prior"] += 1
            continue
        previous_value = ordered.at[previous_month, release_column]
        if pd.isna(previous_value):
            diagnostics["rows_excluded_missing_prior"] += 1
            continue
        current_value = float(available.iloc[0])
        previous_value = float(previous_value)
        records.append(
            {
                "reference_month": month.to_timestamp(),
                "component": component,
                "series_id": series_id,
                "release_date": pd.Timestamp(release_date),
                "current_value": current_value,
                "previous_value_as_of_release": previous_value,
                "transform": transform,
                "transformed_value": apply_transform(
                    current_value, previous_value, transform
                ),
                "release_lag_days": (release_date - month_end).days,
            }
        )
    columns = [
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
    output = pd.DataFrame.from_records(records, columns=columns)
    if not output.empty:
        output = output.sort_values("reference_month").reset_index(drop=True)
    diagnostics["rows_retained"] = len(output)
    output.attrs["extraction_diagnostics"] = diagnostics
    return output


def same_vintage_level_difference(
    total: pd.DataFrame,
    excluded_component: pd.DataFrame,
    *,
    derived_series_id: str,
) -> pd.DataFrame:
    """Subtract two level matrices only at vintages present in both inputs.

    No value is carried between vintage columns.  A derived cell therefore
    exists only when both source levels were observable in the exact same
    provider snapshot.  This is required for the disjoint retail coordinates
    ``sales excluding motor vehicles`` and ``motor-vehicle sales``.
    """

    if not isinstance(total, pd.DataFrame) or not isinstance(
        excluded_component, pd.DataFrame
    ):
        raise TypeError("same-vintage inputs must be pandas DataFrames")
    if not derived_series_id.strip():
        raise ValueError("derived_series_id cannot be empty")
    if total.index.has_duplicates or excluded_component.index.has_duplicates:
        raise ValueError("same-vintage inputs cannot have duplicate reference dates")

    def by_vintage(frame: pd.DataFrame) -> dict[date, str]:
        result: dict[date, str] = {}
        for raw_column in frame.columns:
            column = str(raw_column)
            vintage = vintage_date_from_column(column)
            if vintage in result:
                raise ValueError("same-vintage input repeats a vintage date")
            result[vintage] = column
        return result

    total_columns = by_vintage(total)
    excluded_columns = by_vintage(excluded_component)
    common_vintages = sorted(set(total_columns).intersection(excluded_columns))
    if not common_vintages:
        raise ValueError("retail source matrices have no common vintage dates")

    common_index = total.index.intersection(excluded_component.index).sort_values()
    if common_index.empty:
        raise ValueError("retail source matrices have no common reference dates")
    output = pd.DataFrame(index=common_index)
    for vintage in common_vintages:
        total_values = pd.to_numeric(
            total.loc[common_index, total_columns[vintage]], errors="coerce"
        )
        excluded_values = pd.to_numeric(
            excluded_component.loc[common_index, excluded_columns[vintage]],
            errors="coerce",
        )
        derived = total_values - excluded_values
        # Non-positive motor-vehicle levels cannot support a log change.  Keep
        # them missing so the standard first-release extractor excludes them.
        output[f"{derived_series_id}_{vintage:%Y%m%d}"] = derived.where(
            derived > 0.0
        )
    output.index.name = "reference_month"
    return output


def disjoint_motor_sales_records_from_m01_events(
    events: pd.DataFrame,
    *,
    total_series_id: str = "RSAFS",
    ex_motor_series_id: str = "RSFSXMV",
    derived_series_id: str = "RSAFS_MINUS_RSFSXMV",
) -> pd.DataFrame:
    """Derive same-vintage motor-sales log changes from Model 01 audit rows.

    Model 01 retained the current and previous level used for each
    first-release log change.  Requiring equal release and reference dates for
    the two source rows proves that both subtractions use one shared vintage.
    """

    required = {
        "series_id",
        "reference_month",
        "release_date",
        "current_value",
        "previous_value_as_of_release",
        "release_lag_days",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(
            "Model 01 retail events are missing columns: "
            + ", ".join(sorted(missing))
        )

    source = events.loc[
        events["series_id"].isin([total_series_id, ex_motor_series_id]),
        list(required),
    ].copy()
    source["reference_month"] = (
        pd.to_datetime(source["reference_month"], errors="raise")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    source["release_date"] = pd.to_datetime(source["release_date"], errors="raise")
    if source.duplicated(["series_id", "reference_month"]).any():
        raise ValueError("Model 01 retail events repeat a series reference month")
    indexed = source.set_index(["reference_month", "series_id"])
    total = indexed.xs(total_series_id, level="series_id")
    ex_motor = indexed.xs(ex_motor_series_id, level="series_id")
    if not total.index.equals(ex_motor.index):
        missing_total = ex_motor.index.difference(total.index)
        missing_ex_motor = total.index.difference(ex_motor.index)
        raise ValueError(
            "retail source coverage differs; "
            f"missing_total={list(missing_total[:3])}, "
            f"missing_ex_motor={list(missing_ex_motor[:3])}"
        )
    if not total["release_date"].equals(ex_motor["release_date"]):
        raise ValueError("retail source rows do not share exact vintage dates")
    if not total["release_lag_days"].equals(ex_motor["release_lag_days"]):
        raise ValueError("retail source rows disagree on release lag")

    current = pd.to_numeric(total["current_value"], errors="raise") - pd.to_numeric(
        ex_motor["current_value"], errors="raise"
    )
    previous = pd.to_numeric(
        total["previous_value_as_of_release"], errors="raise"
    ) - pd.to_numeric(ex_motor["previous_value_as_of_release"], errors="raise")
    if (current <= 0.0).any() or (previous <= 0.0).any():
        raise ValueError("derived motor-sales levels must be positive")
    transformed = 100.0 * np.log(current / previous)
    records = pd.DataFrame(
        {
            "reference_month": total.index,
            "component": "motor_vehicle_sales",
            "series_id": derived_series_id,
            "release_date": total["release_date"].to_numpy(),
            "current_value": current.to_numpy(dtype=float),
            "previous_value_as_of_release": previous.to_numpy(dtype=float),
            "transform": "log_difference",
            "transformed_value": transformed.to_numpy(dtype=float),
            "release_lag_days": total["release_lag_days"].to_numpy(dtype=int),
        }
    )
    return records.sort_values("reference_month").reset_index(drop=True)


def build_monthly_rate_control_records(
    housing_schedule: pd.DataFrame,
    mortgage_observations: Iterable[FirstReleaseObservation],
    *,
    series_id: str = "MORTGAGE30US",
    methodology_change_date: date = date(2022, 11, 17),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align monthly mortgage-rate controls to housing release events.

    The current and prior values are arithmetic averages of the weekly
    first-release observations belonging to their respective reference months.
    A row is emitted only if every weekly observation used was public no later
    than the housing event.  The transformed control is the current average
    minus the prior-month average.  The companion binary control is one when at
    least one current-month mortgage observation was released under the
    methodology introduced on ``methodology_change_date``.
    """

    required = {"reference_month", "release_date"}
    missing = required.difference(housing_schedule.columns)
    if missing:
        raise ValueError(
            "housing schedule is missing columns: " + ", ".join(sorted(missing))
        )
    schedule = housing_schedule.loc[:, ["reference_month", "release_date"]].copy()
    schedule["reference_month"] = (
        pd.to_datetime(schedule["reference_month"], errors="raise")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    schedule["release_date"] = pd.to_datetime(
        schedule["release_date"], errors="raise"
    )
    schedule = schedule.drop_duplicates().sort_values(
        ["release_date", "reference_month"]
    )
    if schedule.duplicated("reference_month").any():
        raise ValueError("housing schedule has multiple release dates for a month")
    if (schedule["release_date"] < schedule["reference_month"]).any():
        raise ValueError("housing releases cannot precede their reference month")

    observations = pd.DataFrame.from_records(
        [
            {
                "reference_date": pd.Timestamp(item.reference_date),
                "release_date": pd.Timestamp(item.release_date),
                "value": float(item.value),
            }
            for item in mortgage_observations
        ]
    )
    columns = [
        "reference_month",
        "series_id",
        "release_date",
        "current_value",
        "previous_value_as_of_release",
        "transform",
        "transformed_value",
        "release_lag_days",
    ]
    dummy_columns = [
        "reference_month",
        "release_date",
        "methodology_dummy",
        "release_lag_days",
    ]
    if observations.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=dummy_columns)
    if not np.isfinite(observations["value"]).all():
        raise ValueError("mortgage observations must be finite")
    if (observations["value"] <= 0.0).any():
        raise ValueError("mortgage observations must be positive")
    if observations.duplicated("reference_date").any():
        raise ValueError("mortgage observations repeat a reference date")
    observations["reference_month"] = (
        observations["reference_date"].dt.to_period("M").dt.to_timestamp()
    )

    controls: list[dict[str, object]] = []
    dummies: list[dict[str, object]] = []
    for row in schedule.itertuples(index=False):
        reference_month = pd.Timestamp(row.reference_month)
        release_date = pd.Timestamp(row.release_date)
        previous_month = (reference_month.to_period("M") - 1).to_timestamp()
        current = observations.loc[
            (observations["reference_month"] == reference_month)
            & (observations["release_date"] <= release_date)
        ]
        previous = observations.loc[
            (observations["reference_month"] == previous_month)
            & (observations["release_date"] <= release_date)
        ]
        if current.empty or previous.empty:
            continue
        current_average = float(current["value"].mean())
        previous_average = float(previous["value"].mean())
        if not math.isfinite(current_average) or not math.isfinite(previous_average):
            raise ValueError("monthly mortgage averages must be finite")
        month_end = reference_month.to_period("M").end_time.normalize()
        release_lag_days = int((release_date - month_end).days)
        if release_lag_days < 0:
            raise ValueError("housing release precedes its reference-month end")
        controls.append(
            {
                "reference_month": reference_month,
                "series_id": series_id,
                "release_date": release_date,
                "current_value": current_average,
                "previous_value_as_of_release": previous_average,
                "transform": "reference_month_average_difference",
                "transformed_value": current_average - previous_average,
                "release_lag_days": release_lag_days,
            }
        )
        dummies.append(
            {
                "reference_month": reference_month,
                "release_date": release_date,
                "methodology_dummy": int(
                    (current["release_date"] >= pd.Timestamp(methodology_change_date))
                    .any()
                ),
                "release_lag_days": release_lag_days,
            }
        )
    return (
        pd.DataFrame.from_records(controls, columns=columns),
        pd.DataFrame.from_records(dummies, columns=dummy_columns),
    )


def build_binary_control_events(
    controls: pd.DataFrame,
    *,
    release_block: str,
    feature_name: str,
    series_id: str,
    provider_id: str,
    source_url: str,
) -> pd.DataFrame:
    """Encode an already-defined zero-one control in the canonical event schema."""

    required = {
        "reference_month",
        "release_date",
        "methodology_dummy",
        "release_lag_days",
    }
    missing = required.difference(controls.columns)
    if missing:
        raise ValueError(
            "binary controls are missing columns: " + ", ".join(sorted(missing))
        )
    records: list[dict[str, object]] = []
    for row in controls.itertuples(index=False):
        reference_month = pd.Timestamp(row.reference_month).to_period("M").to_timestamp()
        release_date = pd.Timestamp(row.release_date)
        value = int(row.methodology_dummy)
        if value not in (0, 1):
            raise ValueError("methodology control must contain only zero or one")
        group_id = f"{release_block}:{release_date.date().isoformat()}"
        event_id = f"{group_id}:{reference_month.date().isoformat()}"
        records.append(
            {
                "event_id": event_id,
                "event_group_id": group_id,
                "release_block": release_block,
                "release_date": release_date,
                "reference_date": reference_month,
                "reference_month": reference_month,
                "frequency": "monthly",
                "feature_name": feature_name,
                "series_id": series_id,
                "provider_id": provider_id,
                "source_url": source_url,
                "current_value": value,
                "previous_value_as_of_release": np.nan,
                "transform": "post_methodology_indicator",
                "transformed_value": value,
                "feature_value": value,
                "feature_status": "available",
                "release_lag_days": int(row.release_lag_days),
                "is_target_defining": False,
            }
        )
    output = pd.DataFrame.from_records(records).reindex(columns=EVENT_TABLE_COLUMNS)
    if output.empty:
        return output
    for column in ("release_date", "reference_date", "reference_month"):
        output[column] = pd.to_datetime(output[column])
    output["is_target_defining"] = output["is_target_defining"].astype(bool)
    return output.sort_values(
        ["release_date", "event_group_id", "reference_date", "feature_name"]
    ).reset_index(drop=True)


__all__ = [
    "build_binary_control_events",
    "build_monthly_rate_control_records",
    "disjoint_motor_sales_records_from_m01_events",
    "extract_contemporaneous_monthly_features",
    "same_vintage_level_difference",
]
