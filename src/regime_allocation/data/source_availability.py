"""Causal archive-availability ledgers for M03 source-data research.

A dated vintage is eligible only at midnight on the following calendar day in
the declared availability timezone. This is a conservative archive clock, not
an assertion about an economic publication's unverified intraday timestamp.
Future vintage cells are never read. Every requested reference period receives
one retained or excluded ledger row, including periods absent from the matrix.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta
from numbers import Integral
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from regime_allocation.data.providers.vintage_matrix import vintage_date_from_column

_TRANSFORMS = {"difference", "negative_difference", "log_difference"}
_FREQUENCIES = {"monthly", "weekly_saturday"}
_COLUMNS = (
    "reference_date",
    "reference_month",
    "frequency",
    "series_id",
    "component",
    "snapshot_id",
    "status",
    "reason",
    "observation_semantics",
    "archive_vintage_date",
    "archive_available_at",
    "publication_timestamp",
    "timestamp_precision",
    "availability_timezone",
    "transform",
    "current_value",
    "previous_value_as_of_release",
    "transformed_value",
    "release_lag_days",
)


def _date(value: date, name: str) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise TypeError(f"{name} must be a date, not a timestamp")
    return value


def _calendar(reference_start: date, reference_end: date, frequency: str) -> pd.DatetimeIndex:
    start = _date(reference_start, "reference_start")
    end = _date(reference_end, "reference_end")
    if start > end:
        raise ValueError("reference_start must not follow reference_end")
    if frequency not in _FREQUENCIES:
        raise ValueError(f"unsupported frequency: {frequency}")
    if frequency == "monthly":
        start, end = start.replace(day=1), end.replace(day=1)
    return pd.date_range(start, end, freq="MS" if frequency == "monthly" else "W-SAT")


def _prepare_matrix(
    matrix: pd.DataFrame, *, as_of: datetime, availability_timezone: str, frequency: str
) -> tuple[pd.DataFrame, list[tuple[Any, date, pd.Timestamp]], date]:
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a DataFrame")
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be a timezone-aware datetime")
    if pd.isna(as_of):
        raise ValueError("as_of must be a valid datetime")
    timezone = ZoneInfo(availability_timezone)
    cutoff = as_of.astimezone(UTC)
    references: list[pd.Timestamp] = []
    for value in matrix.index:
        if isinstance(value, (int, float, bool, np.number)):
            raise TypeError("reference dates must be explicit calendar dates")
        try:
            reference = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid matrix reference date") from error
        if pd.isna(reference) or reference.tzinfo is not None or reference != reference.normalize():
            raise ValueError("reference dates must be valid timezone-naive midnights")
        if frequency == "monthly" and reference.day != 1:
            raise ValueError("monthly matrix references must be month starts")
        if frequency == "weekly_saturday" and reference.dayofweek != 5:
            raise ValueError("weekly_saturday matrix references must be Saturdays")
        references.append(reference)
    index = pd.DatetimeIndex(references)
    if index.has_duplicates:
        raise ValueError("matrix contains duplicate reference dates")

    # Validate dates using labels only; do not coerce or inspect future values.
    metadata: list[tuple[int, Any, date, pd.Timestamp]] = []
    seen: set[date] = set()
    for position, column in enumerate(matrix.columns):
        vintage = vintage_date_from_column(str(column))
        if vintage in seen:
            raise ValueError("matrix contains duplicate vintage dates")
        seen.add(vintage)
        available = datetime.combine(vintage + timedelta(days=1), time.min, timezone)
        if available.astimezone(UTC) <= cutoff:
            metadata.append((position, column, vintage, pd.Timestamp(available).tz_convert("UTC")))
    metadata.sort(key=lambda item: item[2])
    visible = matrix.iloc[:, [item[0] for item in metadata]].copy()
    visible.index = index
    visible = visible.sort_index()
    return (
        visible,
        [(column, vintage, available) for _, column, vintage, available in metadata],
        (as_of.astimezone(timezone).date()),
    )


def _missing(value: object) -> bool:
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _finite(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _support(
    active_start: date | None, active_end: date | None, frequency: str
) -> tuple[date | None, date | None]:
    start = None if active_start is None else _date(active_start, "active_start")
    end = None if active_end is None else _date(active_end, "active_end")
    if start is not None and end is not None and start > end:
        raise ValueError("active_start must not follow active_end")
    if frequency == "monthly":
        start = None if start is None else start.replace(day=1)
        end = None if end is None else end.replace(day=1)
    return start, end


def _level_row(
    reference: pd.Timestamp,
    *,
    visible: pd.DataFrame,
    vintages: list[tuple[Any, date, pd.Timestamp]],
    local_cutoff: date,
    series_id: str,
    component: str,
    snapshot_id: str,
    frequency: str,
    availability_timezone: str,
    active_start: date | None,
    active_end: date | None,
) -> tuple[dict[str, object], Any | None]:
    row: dict[str, object] = {
        "reference_date": reference,
        "reference_month": reference.to_period("M").to_timestamp(),
        "frequency": frequency,
        "series_id": series_id,
        "component": component,
        "snapshot_id": snapshot_id,
        "status": "excluded",
        "reason": "unavailable_by_asof",
        "observation_semantics": "archive_first_observed",
        "archive_vintage_date": pd.NaT,
        "archive_available_at": pd.NaT,
        "publication_timestamp": pd.NaT,
        "timestamp_precision": "date",
        "availability_timezone": availability_timezone,
        "transform": "level",
        "current_value": float("nan"),
        "previous_value_as_of_release": float("nan"),
        "transformed_value": float("nan"),
        "release_lag_days": None,
    }
    day = reference.date()
    if (active_start is not None and day < active_start) or (
        active_end is not None and day > active_end
    ):
        row["reason"] = "outside_active_support"
    elif day > local_cutoff:
        row["reason"] = "future_reference"
    elif not vintages:
        pass
    elif reference not in visible.index:
        row["reason"] = "absent_reference"
    else:
        for column, vintage, available in vintages:
            value = visible.at[reference, column]
            if _missing(value):
                continue
            row.update(archive_vintage_date=pd.Timestamp(vintage), archive_available_at=available)
            numeric = _finite(value)
            if numeric is None:
                row["reason"] = "invalid_current_value"
            elif day > vintage:
                row["reason"] = "reference_after_vintage"
                row["current_value"] = numeric
            else:
                row.update(
                    status="retained", reason="archive_first_observed", current_value=numeric
                )
            return row, column
    return row, None


def _ledger(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=_COLUMNS)
    for column in ("reference_date", "reference_month", "archive_vintage_date"):
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    for column in ("archive_available_at", "publication_timestamp"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[ns, UTC]")
    for column in ("current_value", "previous_value_as_of_release", "transformed_value"):
        frame[column] = frame[column].astype(float)
    frame["release_lag_days"] = frame["release_lag_days"].astype("Int64")
    return frame


def extract_source_observations(
    matrix: pd.DataFrame,
    *,
    series_id: str,
    reference_start: date,
    reference_end: date,
    as_of: datetime,
    frequency: str = "monthly",
    active_start: date | None = None,
    active_end: date | None = None,
    snapshot_id: str = "",
    component: str = "",
    availability_timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Audit first observed archive levels without asserting genuine first publication.

    Monthly endpoints and active-support bounds designate their containing
    months. Weekly endpoints enumerate Saturdays within the inclusive range.
    No lag, bootstrap, or feature-transformation policy is applied. An absent
    index row is distinguished from a present row with no visible observation;
    when no vintage is visible, availability is unknown for all eligible rows.
    """
    references = _calendar(reference_start, reference_end, frequency)
    start, end = _support(active_start, active_end, frequency)
    visible, vintages, local_cutoff = _prepare_matrix(
        matrix, as_of=as_of, availability_timezone=availability_timezone, frequency=frequency
    )
    return _ledger(
        [
            _level_row(
                reference,
                visible=visible,
                vintages=vintages,
                local_cutoff=local_cutoff,
                series_id=series_id,
                component=component,
                snapshot_id=snapshot_id,
                frequency=frequency,
                availability_timezone=availability_timezone,
                active_start=start,
                active_end=end,
            )[0]
            for reference in references
        ]
    )


def extract_source_features(
    matrix: pd.DataFrame,
    *,
    series_id: str,
    component: str,
    transform: str,
    reference_start: date,
    reference_end: date,
    as_of: datetime,
    max_release_lag_days: int = 92,
    archive_start_latest_only: bool = True,
    active_start: date | None = None,
    active_end: date | None = None,
    snapshot_id: str = "",
    availability_timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Audit every monthly feature at its first visible nonmissing appearance.

    Both levels must be finite and come from the same vintage. An invalid
    first appearance is excluded permanently; a later revision cannot repair
    it. The initial visible archive's latest completed reference month is the
    only bootstrap candidate when ``archive_start_latest_only`` is enabled.
    This selection uses the entire visible snapshot, not the requested output
    range, and ignores impossible post-vintage references. Prior-month levels
    may be outside active output support, but must exist in that same vintage.
    """
    if transform not in _TRANSFORMS:
        raise ValueError(f"unsupported transform: {transform}")
    if isinstance(max_release_lag_days, bool) or not isinstance(max_release_lag_days, Integral):
        raise TypeError("max_release_lag_days must be an integer")
    if max_release_lag_days < 0:
        raise ValueError("max_release_lag_days must be nonnegative")
    if not isinstance(archive_start_latest_only, bool):
        raise TypeError("archive_start_latest_only must be a boolean")
    references = _calendar(reference_start, reference_end, "monthly")
    start, end = _support(active_start, active_end, "monthly")
    visible, vintages, local_cutoff = _prepare_matrix(
        matrix, as_of=as_of, availability_timezone=availability_timezone, frequency="monthly"
    )
    bootstrap_latest = None
    first_column = vintages[0][0] if vintages else None
    if archive_start_latest_only and vintages:
        first_vintage = vintages[0][1]
        candidates = [
            reference
            for reference in visible.index
            if reference.to_period("M").end_time.date() <= first_vintage
            and not _missing(visible.at[reference, first_column])
        ]
        bootstrap_latest = max(candidates) if candidates else None
    records = []
    for reference in references:
        row, column = _level_row(
            reference,
            visible=visible,
            vintages=vintages,
            local_cutoff=local_cutoff,
            series_id=series_id,
            component=component,
            snapshot_id=snapshot_id,
            frequency="monthly",
            availability_timezone=availability_timezone,
            active_start=start,
            active_end=end,
        )
        row["transform"] = transform
        records.append(row)
        if row["reason"] == "reference_after_vintage":
            row["reason"] = "negative_release_lag"
            row["release_lag_days"] = (
                pd.Timestamp(row["archive_vintage_date"]).date()
                - reference.to_period("M").end_time.date()
            ).days
        if row["status"] != "retained":
            continue
        row["status"] = "excluded"
        lag = (
            pd.Timestamp(row["archive_vintage_date"]).date()
            - reference.to_period("M").end_time.date()
        ).days
        row["release_lag_days"] = lag
        if lag < 0:
            row["reason"] = "negative_release_lag"
            continue
        if archive_start_latest_only and column == first_column and reference != bootstrap_latest:
            row["reason"] = "archive_bootstrap"
            continue
        if lag > max_release_lag_days:
            row["reason"] = "excessive_release_lag"
            continue
        previous = (reference.to_period("M") - 1).to_timestamp()
        if previous not in visible.index:
            row["reason"] = "missing_prior_reference"
            continue
        prior = visible.at[previous, column]
        if _missing(prior):
            row["reason"] = "missing_prior_value"
            continue
        prior_numeric = _finite(prior)
        if prior_numeric is None:
            row["reason"] = "invalid_prior_value"
            continue
        row["previous_value_as_of_release"] = prior_numeric
        current = float(row["current_value"])
        if transform == "log_difference" and (current <= 0 or prior_numeric <= 0):
            row["reason"] = "nonpositive_log_levels"
            continue
        if transform == "log_difference":
            result = 100.0 * (math.log(current) - math.log(prior_numeric))
        else:
            result = (current - prior_numeric) * (-1 if transform == "negative_difference" else 1)
        if not math.isfinite(result):
            row["reason"] = "invalid_transformed_value"
            continue
        row.update(status="retained", reason="retained", transformed_value=result)
    return _ledger(records)
