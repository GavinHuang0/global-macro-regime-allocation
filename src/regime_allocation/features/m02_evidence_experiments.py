"""Build point-in-time feature records for Model 02 evidence experiments.

This module contains only deterministic, in-memory transformations used by
the additive evidence-block experiment stage.  It deliberately does not read
files, call data providers, fit likelihoods, or update a posterior.  Every
monthly transformation selects the first eligible publication vintage for the
reference month and, when a lagged value is needed, takes that lagged value
from the exact same vintage.

The helpers cover the transformations that the frozen Model 02 evidence stage
does not provide: contemporaneous diffusion-index levels, retrospective level
features, twelve-month log changes for non-seasonally-adjusted import prices,
and a same-vintage log backlog-to-shipments ratio.  Archive bootstrap rows are
excluded rather than treated as historical real-time releases.
"""

from __future__ import annotations

from datetime import date
import math

import numpy as np
import pandas as pd

from regime_allocation.data.providers.vintage_matrix import vintage_date_from_column


_FEATURE_RECORD_COLUMNS = (
    "reference_month",
    "component",
    "series_id",
    "release_date",
    "current_value",
    "previous_value_as_of_release",
    "transform",
    "transformed_value",
    "release_lag_days",
)


def _ordered_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("vintage input must be a pandas DataFrame")
    if matrix.index.has_duplicates:
        raise ValueError("vintage input repeats a reference date")
    columns = sorted((str(value) for value in matrix.columns), key=vintage_date_from_column)
    ordered = matrix.copy()
    ordered.columns = [str(value) for value in ordered.columns]
    ordered = ordered.reindex(columns=columns).sort_index()
    ordered.index = (
        pd.to_datetime(ordered.index, errors="raise").to_period("M").to_timestamp()
    )
    if ordered.index.has_duplicates:
        raise ValueError("vintage input repeats a reference month")
    return ordered, columns


def _archive_latest_reference(
    ordered: pd.DataFrame,
    columns: list[str],
    *,
    archive_start_latest_only: bool,
) -> pd.Timestamp | None:
    if not archive_start_latest_only or not columns:
        return None
    first = pd.to_numeric(ordered[columns[0]], errors="coerce")
    available = first.index[first.notna()]
    return None if not len(available) else pd.Timestamp(available.max())


def _empty_records(diagnostics: dict[str, int]) -> pd.DataFrame:
    result = pd.DataFrame(columns=_FEATURE_RECORD_COLUMNS)
    result.attrs["extraction_diagnostics"] = diagnostics
    return result


def extract_first_release_levels(
    matrix: pd.DataFrame,
    *,
    series_id: str,
    component: str,
    transform_name: str = "level",
    contemporaneous: bool = False,
    max_release_lag_days: int = 92,
    archive_start_latest_only: bool = True,
) -> pd.DataFrame:
    """Extract causally published monthly levels without differencing them.

    ``contemporaneous=True`` requires publication within the labeled reference
    month, which is the appropriate contract for current-month manufacturing
    diffusion indexes.  Otherwise publication must occur from month-end
    through ``max_release_lag_days`` after month-end.
    """

    if not str(series_id).strip() or not str(component).strip():
        raise ValueError("series_id and component cannot be empty")
    if not str(transform_name).strip():
        raise ValueError("transform_name cannot be empty")
    if max_release_lag_days < 0:
        raise ValueError("max_release_lag_days cannot be negative")
    ordered, columns = _ordered_matrix(matrix)
    archive_latest = _archive_latest_reference(
        ordered,
        columns,
        archive_start_latest_only=archive_start_latest_only,
    )
    first_column = columns[0] if columns else None
    diagnostics = {
        "matrix_rows": len(ordered),
        "rows_without_any_vintage": 0,
        "rows_excluded_archive_bootstrap": 0,
        "rows_excluded_release_timing": 0,
        "rows_retained": 0,
    }
    records: list[dict[str, object]] = []
    for reference_month, row in ordered.iterrows():
        available = pd.to_numeric(row, errors="coerce").dropna()
        if available.empty:
            diagnostics["rows_without_any_vintage"] += 1
            continue
        release_column = str(available.index[0])
        release_date = vintage_date_from_column(release_column)
        reference = pd.Timestamp(reference_month).to_period("M")
        if (
            archive_start_latest_only
            and release_column == first_column
            and archive_latest is not None
            and reference.to_timestamp() != archive_latest
        ):
            diagnostics["rows_excluded_archive_bootstrap"] += 1
            continue
        month_start = reference.start_time.date()
        month_end = reference.end_time.normalize().date()
        lag = (release_date - month_end).days
        timing_ok = (
            month_start <= release_date <= month_end
            if contemporaneous
            else 0 <= lag <= max_release_lag_days
        )
        if not timing_ok:
            diagnostics["rows_excluded_release_timing"] += 1
            continue
        value = float(available.iloc[0])
        if not math.isfinite(value):
            raise ValueError("first-release level must be finite")
        records.append(
            {
                "reference_month": reference.to_timestamp(),
                "component": component,
                "series_id": series_id,
                "release_date": pd.Timestamp(release_date),
                "current_value": value,
                "previous_value_as_of_release": np.nan,
                "transform": transform_name,
                "transformed_value": value,
                "release_lag_days": lag,
            }
        )
    diagnostics["rows_retained"] = len(records)
    if not records:
        return _empty_records(diagnostics)
    result = pd.DataFrame.from_records(records, columns=_FEATURE_RECORD_COLUMNS)
    result = result.sort_values("reference_month").reset_index(drop=True)
    result.attrs["extraction_diagnostics"] = diagnostics
    return result


def extract_first_release_log_change(
    matrix: pd.DataFrame,
    *,
    series_id: str,
    component: str,
    lag_months: int,
    transform_name: str,
    max_release_lag_days: int = 92,
    archive_start_latest_only: bool = True,
) -> pd.DataFrame:
    """Return ``100 log(x[m] / x[m-lag])`` from one publication vintage."""

    if isinstance(lag_months, bool) or int(lag_months) < 1:
        raise ValueError("lag_months must be a positive integer")
    lag_months = int(lag_months)
    ordered, columns = _ordered_matrix(matrix)
    archive_latest = _archive_latest_reference(
        ordered,
        columns,
        archive_start_latest_only=archive_start_latest_only,
    )
    first_column = columns[0] if columns else None
    diagnostics = {
        "matrix_rows": len(ordered),
        "rows_without_any_vintage": 0,
        "rows_excluded_archive_bootstrap": 0,
        "rows_excluded_release_timing": 0,
        "rows_excluded_missing_lag": 0,
        "rows_retained": 0,
    }
    records: list[dict[str, object]] = []
    for reference_month, row in ordered.iterrows():
        available = pd.to_numeric(row, errors="coerce").dropna()
        if available.empty:
            diagnostics["rows_without_any_vintage"] += 1
            continue
        release_column = str(available.index[0])
        release_date = vintage_date_from_column(release_column)
        reference = pd.Timestamp(reference_month).to_period("M")
        if (
            archive_start_latest_only
            and release_column == first_column
            and archive_latest is not None
            and reference.to_timestamp() != archive_latest
        ):
            diagnostics["rows_excluded_archive_bootstrap"] += 1
            continue
        month_end = reference.end_time.normalize().date()
        release_lag_days = (release_date - month_end).days
        if release_lag_days < 0 or release_lag_days > max_release_lag_days:
            diagnostics["rows_excluded_release_timing"] += 1
            continue
        lagged_month = (reference - lag_months).to_timestamp()
        if lagged_month not in ordered.index:
            diagnostics["rows_excluded_missing_lag"] += 1
            continue
        lagged = ordered.at[lagged_month, release_column]
        if pd.isna(lagged):
            diagnostics["rows_excluded_missing_lag"] += 1
            continue
        current = float(available.iloc[0])
        previous = float(lagged)
        if current <= 0.0 or previous <= 0.0:
            raise ValueError("log-change levels must be strictly positive")
        records.append(
            {
                "reference_month": reference.to_timestamp(),
                "component": component,
                "series_id": series_id,
                "release_date": pd.Timestamp(release_date),
                "current_value": current,
                "previous_value_as_of_release": previous,
                "transform": transform_name,
                "transformed_value": 100.0 * math.log(current / previous),
                "release_lag_days": release_lag_days,
            }
        )
    diagnostics["rows_retained"] = len(records)
    if not records:
        return _empty_records(diagnostics)
    result = pd.DataFrame.from_records(records, columns=_FEATURE_RECORD_COLUMNS)
    result = result.sort_values("reference_month").reset_index(drop=True)
    result.attrs["extraction_diagnostics"] = diagnostics
    return result


def same_vintage_log_ratio_matrix(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
    *,
    derived_series_id: str,
) -> pd.DataFrame:
    """Construct ``100 log(numerator / denominator)`` at common vintages.

    A cell is emitted only when both inputs exist for the same reference month
    in the exact same provider snapshot.  This prevents a later revision of
    either series from leaking into an earlier backlog-to-shipments feature.
    """

    if not str(derived_series_id).strip():
        raise ValueError("derived_series_id cannot be empty")
    left, left_columns = _ordered_matrix(numerator)
    right, right_columns = _ordered_matrix(denominator)
    left_by_vintage = {vintage_date_from_column(name): name for name in left_columns}
    right_by_vintage = {vintage_date_from_column(name): name for name in right_columns}
    if len(left_by_vintage) != len(left_columns) or len(right_by_vintage) != len(
        right_columns
    ):
        raise ValueError("an input repeats a vintage date")
    common_vintages = sorted(set(left_by_vintage).intersection(right_by_vintage))
    common_index = left.index.intersection(right.index).sort_values()
    if not common_vintages or common_index.empty:
        raise ValueError("ratio inputs have no common reference/vintage support")
    derived_columns: dict[str, pd.Series] = {}
    for vintage in common_vintages:
        numerator_values = pd.to_numeric(
            left.loc[common_index, left_by_vintage[vintage]], errors="coerce"
        )
        denominator_values = pd.to_numeric(
            right.loc[common_index, right_by_vintage[vintage]], errors="coerce"
        )
        valid = (numerator_values > 0.0) & (denominator_values > 0.0)
        ratio = numerator_values.where(valid) / denominator_values.where(valid)
        derived_columns[f"{derived_series_id}_{vintage:%Y%m%d}"] = 100.0 * np.log(
            ratio
        )
    # Construct once rather than inserting hundreds of vintage columns one by
    # one; this keeps the frame contiguous without changing any values.
    output = pd.DataFrame(derived_columns, index=common_index)
    output.index.name = "reference_month"
    return output


__all__ = [
    "extract_first_release_levels",
    "extract_first_release_log_change",
    "same_vintage_log_ratio_matrix",
]
