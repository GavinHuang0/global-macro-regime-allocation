"""Causal feature and event-table construction for non-defining evidence.

This module contains no Bayesian inference.  It converts point-in-time release
records into a provider-neutral, long event table that a later filtering layer
can consume without reinterpreting release dates or recomputing features.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
import math

import numpy as np
import pandas as pd

from regime_allocation.data.providers.vintage_matrix import FirstReleaseObservation
from regime_allocation.features.claims import build_claims_ar1_features


EVENT_TABLE_COLUMNS = (
    "event_id",
    "event_group_id",
    "release_block",
    "release_date",
    "reference_date",
    "reference_month",
    "frequency",
    "feature_name",
    "series_id",
    "provider_id",
    "source_url",
    "current_value",
    "previous_value_as_of_release",
    "transform",
    "transformed_value",
    "feature_value",
    "feature_status",
    "release_lag_days",
    "is_target_defining",
    "ar_prior_observation_count",
    "ar_transition_count",
    "ar_intercept",
    "ar_lag1_coefficient",
    "forecast_log_level",
    "forecast_level",
    "innovation_log",
    "standardization_prior_count",
    "standardization_prior_mean",
    "standardization_prior_std",
)


def _event_identifiers(
    release_block: str,
    release_date: pd.Timestamp,
    reference_date: pd.Timestamp,
) -> tuple[str, str]:
    release_text = release_date.date().isoformat()
    reference_text = reference_date.date().isoformat()
    group_id = f"{release_block}:{release_text}"
    return f"{group_id}:{reference_text}", group_id


def _validate_identifier(value: str, *, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    if any(character in normalized for character in ":,\n\r"):
        raise ValueError(f"{name} contains a reserved character")
    return normalized


def _finalize_event_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.reindex(columns=EVENT_TABLE_COLUMNS).copy()
    for column in ("release_date", "reference_date", "reference_month"):
        output[column] = pd.to_datetime(output[column])
    output["is_target_defining"] = output["is_target_defining"].astype(bool)
    output = output.sort_values(
        ["release_date", "event_group_id", "reference_date", "feature_name"]
    ).reset_index(drop=True)
    return output


def build_monthly_release_events(
    first_release_features: pd.DataFrame,
    *,
    release_block: str,
    feature_name: str,
    frequency: str,
    provider_id: str,
    source_url: str,
    min_standardization_history: int = 60,
    ddof: int = 1,
) -> pd.DataFrame:
    """Convert same-vintage monthly transforms into causal event features."""

    release_block = _validate_identifier(release_block, name="release_block")
    feature_name = _validate_identifier(feature_name, name="feature_name")
    frequency = _validate_identifier(frequency, name="frequency")
    provider_id = _validate_identifier(provider_id, name="provider_id")
    if min_standardization_history < 2:
        raise ValueError("min_standardization_history must be at least 2")
    if ddof < 0 or ddof >= min_standardization_history:
        raise ValueError(
            "ddof must be non-negative and smaller than standardization history"
        )

    required = {
        "reference_month",
        "series_id",
        "release_date",
        "current_value",
        "previous_value_as_of_release",
        "transform",
        "transformed_value",
        "release_lag_days",
    }
    missing = required.difference(first_release_features.columns)
    if missing:
        raise ValueError(
            f"monthly release records are missing columns: {', '.join(sorted(missing))}"
        )
    if first_release_features.empty:
        return _finalize_event_table(pd.DataFrame(columns=EVENT_TABLE_COLUMNS))

    records = first_release_features.copy()
    records["reference_month"] = (
        pd.to_datetime(records["reference_month"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    records["release_date"] = pd.to_datetime(records["release_date"])
    if records[["reference_month", "release_date"]].isna().any().any():
        raise ValueError("monthly release dates cannot be missing")
    if records.duplicated(["series_id", "reference_month"]).any():
        raise ValueError("monthly release records contain duplicate reference months")
    series_ids = records["series_id"].astype(str).unique()
    if len(series_ids) != 1:
        raise ValueError("monthly release events must contain exactly one series")
    expected_lags = (
        records["release_date"]
        - records["reference_month"].dt.to_period("M").dt.end_time.dt.normalize()
    ).dt.days
    supplied_lags = pd.to_numeric(records["release_lag_days"], errors="raise")
    if not supplied_lags.equals(expected_lags):
        raise ValueError("monthly release lags do not match reference-period ends")
    records = records.sort_values(["release_date", "reference_month"]).reset_index(
        drop=True
    )
    transformed = pd.to_numeric(
        records["transformed_value"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(transformed).all():
        raise ValueError("monthly transformed values must be finite")

    # A catch-up publication can expose more than one reference month on the
    # same date.  Every row in that publication must use the same information
    # set: transformed values released on that date are appended to history
    # only after the whole group has been standardized.
    standardized = np.full(len(records), np.nan, dtype=float)
    prior_counts = np.zeros(len(records), dtype=np.int64)
    prior_means = np.full(len(records), np.nan, dtype=float)
    prior_stds = np.full(len(records), np.nan, dtype=float)
    prior_values: list[float] = []
    for _, positions in records.groupby("release_date", sort=True).indices.items():
        locations = np.asarray(positions, dtype=np.int64)
        prior_count = len(prior_values)
        prior_mean = np.nan
        prior_std = np.nan
        if prior_count:
            prior_array = np.asarray(prior_values, dtype=float)
            prior_mean = float(prior_array.mean())
            if prior_count > ddof:
                prior_std = float(prior_array.std(ddof=ddof))
        prior_counts[locations] = prior_count
        prior_means[locations] = prior_mean
        prior_stds[locations] = prior_std
        if (
            prior_count >= min_standardization_history
            and math.isfinite(prior_std)
            and prior_std > 0.0
        ):
            standardized[locations] = (
                transformed[locations] - prior_mean
            ) / prior_std
        prior_values.extend(transformed[locations].tolist())

    output_records: list[dict[str, object]] = []
    for position, row in enumerate(records.itertuples(index=False)):
        release_date = pd.Timestamp(row.release_date)
        reference_month = pd.Timestamp(row.reference_month).to_period("M").to_timestamp()
        event_id, event_group_id = _event_identifiers(
            release_block, release_date, reference_month
        )
        feature_value = float(standardized[position])
        available = math.isfinite(feature_value)
        if available:
            status = "available"
        elif prior_counts[position] < min_standardization_history:
            status = "standardization_warmup"
        else:
            status = "standardization_unidentified"
        output_records.append(
            {
                "event_id": event_id,
                "event_group_id": event_group_id,
                "release_block": release_block,
                "release_date": release_date,
                "reference_date": reference_month,
                "reference_month": reference_month,
                "frequency": frequency,
                "feature_name": feature_name,
                "series_id": str(row.series_id),
                "provider_id": provider_id,
                "source_url": source_url,
                "current_value": float(row.current_value),
                "previous_value_as_of_release": float(
                    row.previous_value_as_of_release
                ),
                "transform": str(row.transform),
                "transformed_value": float(row.transformed_value),
                "feature_value": feature_value if available else np.nan,
                "feature_status": status,
                "release_lag_days": int(row.release_lag_days),
                "is_target_defining": False,
                "standardization_prior_count": int(prior_counts[position]),
                "standardization_prior_mean": prior_means[position],
                "standardization_prior_std": prior_stds[position],
            }
        )
    return _finalize_event_table(pd.DataFrame.from_records(output_records))


def build_claims_release_events(
    observations: Iterable[FirstReleaseObservation],
    *,
    release_block: str,
    feature_name: str,
    series_id: str,
    provider_id: str,
    source_url: str,
    min_ar_history: int = 52,
    min_standardization_history: int = 26,
    ddof: int = 1,
    max_release_lag_days: int = 92,
    archive_start_latest_only: bool = True,
) -> pd.DataFrame:
    """Create publication-group-causal log-AR(1) claims innovations.

    ALFRED can expose an archive bootstrap as many old reference weeks sharing
    the series' earliest observed release date.  When
    ``archive_start_latest_only`` is true, only that first snapshot's latest
    reference week is eligible; later delayed-release batches are retained in
    full when they satisfy ``max_release_lag_days``.
    """

    release_block = _validate_identifier(release_block, name="release_block")
    feature_name = _validate_identifier(feature_name, name="feature_name")
    series_id = _validate_identifier(series_id, name="series_id")
    provider_id = _validate_identifier(provider_id, name="provider_id")
    if max_release_lag_days < 0:
        raise ValueError("max_release_lag_days must be non-negative")
    if not isinstance(archive_start_latest_only, bool):
        raise TypeError("archive_start_latest_only must be a boolean")

    records = sorted(observations, key=lambda item: (item.release_date, item.reference_date))
    reference_dates = [item.reference_date for item in records]
    if len(reference_dates) != len(set(reference_dates)):
        raise ValueError("claims observations contain duplicate reference dates")
    retained: list[FirstReleaseObservation] = []
    earliest_release_date = records[0].release_date if records else None
    earliest_latest_reference = None
    if archive_start_latest_only and earliest_release_date is not None:
        earliest_latest_reference = max(
            item.reference_date
            for item in records
            if item.release_date == earliest_release_date
        )
    for observation in records:
        if (
            archive_start_latest_only
            and observation.release_date == earliest_release_date
            and observation.reference_date != earliest_latest_reference
        ):
            continue
        lag = (observation.release_date - observation.reference_date).days
        if 0 <= lag <= max_release_lag_days:
            retained.append(observation)
    if not retained:
        return _finalize_event_table(pd.DataFrame(columns=EVENT_TABLE_COLUMNS))
    retained_references = [item.reference_date for item in retained]
    if retained_references != sorted(retained_references):
        raise ValueError(
            "claims reference dates must increase with publication order"
        )
    reference_index = pd.DatetimeIndex(retained_references)
    if len(reference_index) > 1 and not (
        reference_index.to_series().diff().dropna().dt.days == 7
    ).all():
        raise ValueError("claims observations must form an unbroken weekly sequence")
    levels = pd.Series(
        [item.value for item in retained],
        index=reference_index,
        name=series_id,
        dtype=float,
    )
    ar = build_claims_ar1_features(
        levels,
        release_dates=[item.release_date for item in retained],
        min_history=min_ar_history,
        min_standardization_history=min_standardization_history,
        ddof=ddof,
    )

    output_records: list[dict[str, object]] = []
    for position, observation in enumerate(retained):
        audit = ar.iloc[position]
        release_date = pd.Timestamp(observation.release_date)
        reference_date = pd.Timestamp(observation.reference_date)
        reference_month = reference_date.to_period("M").to_timestamp()
        event_id, event_group_id = _event_identifiers(
            release_block, release_date, reference_date
        )
        forecast = audit["forecast_log_level"]
        innovation = audit["innovation_log"]
        feature_value = audit["standardized_innovation"]
        if pd.notna(feature_value):
            status = "available"
        elif (
            pd.notna(innovation)
            and int(audit["standardization_prior_count"])
            < min_standardization_history
        ):
            status = "standardization_warmup"
        elif pd.notna(innovation):
            status = "standardization_unidentified"
        elif int(audit["ar_prior_observation_count"]) < min_ar_history:
            status = "ar_warmup"
        else:
            status = "ar_unidentified"
        output_records.append(
            {
                "event_id": event_id,
                "event_group_id": event_group_id,
                "release_block": release_block,
                "release_date": release_date,
                "reference_date": reference_date,
                "reference_month": reference_month,
                "frequency": "weekly",
                "feature_name": feature_name,
                "series_id": series_id,
                "provider_id": provider_id,
                "source_url": source_url,
                "current_value": observation.value,
                "previous_value_as_of_release": np.nan,
                "transform": "expanding_log_ar1_innovation",
                "transformed_value": innovation,
                "feature_value": feature_value,
                "feature_status": status,
                "release_lag_days": (
                    observation.release_date - observation.reference_date
                ).days,
                "is_target_defining": False,
                "ar_prior_observation_count": int(
                    audit["ar_prior_observation_count"]
                ),
                "ar_transition_count": int(audit["ar_transition_count"]),
                "ar_intercept": audit["ar_intercept"],
                "ar_lag1_coefficient": audit["ar_lag1_coefficient"],
                "forecast_log_level": forecast,
                "forecast_level": audit["forecast_level"],
                "innovation_log": innovation,
                "standardization_prior_count": int(
                    audit["standardization_prior_count"]
                ),
                "standardization_prior_mean": audit["innovation_prior_mean"],
                "standardization_prior_std": audit["innovation_prior_std"],
            }
        )
    return _finalize_event_table(pd.DataFrame.from_records(output_records))


def validate_event_table(events: pd.DataFrame) -> None:
    """Validate the canonical event-table contract before publication."""

    if tuple(events.columns) != EVENT_TABLE_COLUMNS:
        raise ValueError("event table columns do not match the frozen contract")
    if events.empty:
        raise ValueError("event table cannot be empty")
    if events["event_id"].isna().any() or events["event_group_id"].isna().any():
        raise ValueError("event identifiers cannot be missing")
    duplicate_keys = events.duplicated(["event_id", "feature_name", "series_id"])
    if duplicate_keys.any():
        raise ValueError("event table contains duplicate feature rows")
    if events["is_target_defining"].any():
        raise ValueError("non-defining event table contains a target-defining row")
    available = events["feature_status"] == "available"
    if events.loc[available, "feature_value"].isna().any():
        raise ValueError("available event rows must contain a feature value")
    if events.loc[~available, "feature_value"].notna().any():
        raise ValueError("unavailable event rows cannot contain a feature value")
    if (events["release_date"] < events["reference_date"]).any():
        raise ValueError("event release date cannot precede its reference date")
