"""Extract release-coherent monthly features from vintage matrices.

Inputs are provider-neutral level matrices whose columns represent historical
publication vintages. For each reference month, this module identifies the
first vintage in which the level appeared, verifies that appearance against the
configured release-lag and archive-start policies, and transforms current and
prior levels from the same vintage. Outputs include both usable features and a
row-level exclusion audit; no later revision may repair an ineligible first
appearance.
"""

from __future__ import annotations

from datetime import date
import math

import pandas as pd

from regime_allocation.data.providers.vintage_matrix import vintage_date_from_column


VALID_TRANSFORMS = {"difference", "negative_difference", "log_difference"}


def apply_transform(current: float, previous: float, transform: str) -> float:
    """Transform two same-vintage levels into one monthly component value.

    ``difference`` returns current minus previous, ``negative_difference``
    reverses that sign, and ``log_difference`` returns 100 times the log ratio.
    The log form requires positive levels. Keeping both levels in one vintage
    prevents a revision published later from entering the earlier feature.
    """
    if transform == "difference":
        return current - previous
    if transform == "negative_difference":
        return -(current - previous)
    if transform == "log_difference":
        if current <= 0 or previous <= 0:
            raise ValueError("log-difference levels must be positive")
        return 100.0 * math.log(current / previous)
    raise ValueError(f"unsupported transform: {transform}")


def extract_first_release_features(
    matrix: pd.DataFrame,
    *,
    series_id: str,
    component: str,
    transform: str,
    max_release_lag_days: int = 92,
    archive_start_latest_only: bool = False,
) -> pd.DataFrame:
    """Freeze each monthly transformation at the month’s first release vintage.

    For reference month ``m``, the current and previous levels are both selected
    from the earliest vintage containing ``m``. This prevents revisions and
    index rebasing from contaminating month-over-month changes.
    """

    if transform not in VALID_TRANSFORMS:
        raise ValueError(f"unsupported transform: {transform}")
    if max_release_lag_days < 0:
        raise ValueError("max_release_lag_days must be non-negative")
    if not isinstance(archive_start_latest_only, bool):
        raise TypeError("archive_start_latest_only must be a boolean")

    ordered_columns = sorted(matrix.columns, key=vintage_date_from_column)
    ordered = matrix.reindex(columns=ordered_columns).sort_index()
    first_column = str(ordered_columns[0]) if ordered_columns else None
    archive_latest_reference = None
    if archive_start_latest_only and first_column is not None:
        first_snapshot = pd.to_numeric(ordered[first_column], errors="coerce")
        available_references = first_snapshot.index[first_snapshot.notna()]
        if len(available_references):
            archive_latest_reference = pd.Timestamp(available_references.max())
    records: list[dict[str, object]] = []
    diagnostics = {
        "matrix_rows": len(ordered),
        "rows_without_any_vintage": 0,
        "rows_excluded_archive_bootstrap": 0,
        "rows_excluded_negative_lag": 0,
        "rows_excluded_backfill_lag": 0,
        "rows_excluded_missing_prior": 0,
    }
    for reference_month, row in ordered.iterrows():
        available = row.dropna()
        if available.empty:
            diagnostics["rows_without_any_vintage"] += 1
            continue
        release_column = str(available.index[0])
        release_date = vintage_date_from_column(release_column)
        month = pd.Timestamp(reference_month).to_period("M")
        if (
            archive_start_latest_only
            and release_column == first_column
            and archive_latest_reference is not None
            and month.to_timestamp() != archive_latest_reference
        ):
            diagnostics["rows_excluded_archive_bootstrap"] += 1
            continue
        month_end = month.end_time.normalize().date()
        release_lag_days = (release_date - month_end).days
        if release_lag_days < 0:
            diagnostics["rows_excluded_negative_lag"] += 1
            continue
        if release_lag_days > max_release_lag_days:
            # The first selected vintage may bulk-backfill a long pre-archive
            # history. Such rows are not contemporaneous first releases.
            diagnostics["rows_excluded_backfill_lag"] += 1
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
                "release_lag_days": release_lag_days,
            }
        )

    if not records:
        empty = pd.DataFrame(
            columns=[
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
        )
        empty.attrs["extraction_diagnostics"] = diagnostics
        return empty
    output = pd.DataFrame.from_records(records).sort_values("reference_month")
    diagnostics["rows_retained"] = len(output)
    output.attrs["extraction_diagnostics"] = diagnostics
    return output
