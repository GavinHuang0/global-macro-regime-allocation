"""Measure fixed-horizon source revisions in Model 02 score units.

For each component and reference month, both the current and previous levels
are read from one exact three- or twelve-month as-of vintage.  The transformed
revision is divided by the original first-release expanding standard
deviation, so the resulting growth/inflation errors live in the same units as
the deterministic composite scores.  No revised history is used to refit the
score standardizer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from regime_allocation.data.first_release import apply_transform
from regime_allocation.models.m02_soft_composite.scores import (
    ALL_COMPONENTS,
    GROWTH_COMPONENTS,
    INFLATION_COMPONENTS,
    lagged_expanding_moments,
)


def first_release_standardization_scales(
    first_release_long: pd.DataFrame,
    *,
    min_history: int,
    ddof: int,
) -> pd.DataFrame:
    """Reconstruct the exact causal component scales used by the score stage."""

    required = {"reference_month", "component", "transformed_value"}
    missing = required.difference(first_release_long.columns)
    if missing:
        raise ValueError(
            "first-release table is missing columns: " + ", ".join(sorted(missing))
        )
    duplicates = first_release_long.duplicated(
        ["reference_month", "component"], keep=False
    )
    if duplicates.any():
        raise ValueError("first-release table contains duplicate component months")

    transformed = first_release_long.pivot(
        index="reference_month", columns="component", values="transformed_value"
    ).sort_index()
    scales = pd.DataFrame(index=transformed.index)
    for component in ALL_COMPONENTS:
        if component not in transformed:
            raise ValueError(f"first-release table omits component {component}")
        _, prior_std = lagged_expanding_moments(
            transformed[component], min_periods=min_history, ddof=ddof
        )
        scales[component] = prior_std
    scales.index.name = "reference_month"
    return scales


def required_exact_vintages(
    first_release_long: pd.DataFrame,
    *,
    reference_months: Sequence[pd.Timestamp],
    horizons: Sequence[int],
    knowledge_cutoff: pd.Timestamp,
) -> dict[str, tuple[pd.Timestamp, ...]]:
    """Return exact mature as-of dates required for each source series."""

    if not horizons or any(int(value) < 1 for value in horizons):
        raise ValueError("revision horizons must contain positive month counts")
    eligible_months = {pd.Timestamp(value) for value in reference_months}
    subset = first_release_long[
        pd.to_datetime(first_release_long["reference_month"]).isin(eligible_months)
    ]
    requested: dict[str, set[pd.Timestamp]] = {}
    cutoff = pd.Timestamp(knowledge_cutoff).normalize()
    for row in subset.itertuples(index=False):
        release_date = pd.Timestamp(row.release_date).normalize()
        series_dates = requested.setdefault(str(row.series_id), set())
        for horizon in horizons:
            maturity = release_date + pd.DateOffset(months=int(horizon))
            if maturity <= cutoff:
                series_dates.add(maturity)
    return {
        series_id: tuple(sorted(dates))
        for series_id, dates in requested.items()
        if dates
    }


def component_revision_errors(
    first_release_long: pd.DataFrame,
    *,
    scales: pd.DataFrame,
    exact_vintage_matrices: Mapping[str, pd.DataFrame],
    reference_months: Sequence[pd.Timestamp],
    horizons: Sequence[int],
    knowledge_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Compute component-level later-minus-first revision errors."""

    required = {
        "reference_month",
        "component",
        "series_id",
        "release_date",
        "transform",
        "transformed_value",
    }
    missing = required.difference(first_release_long.columns)
    if missing:
        raise ValueError(
            "first-release table is missing columns: " + ", ".join(sorted(missing))
        )
    eligible_months = {pd.Timestamp(value) for value in reference_months}
    subset = first_release_long[
        pd.to_datetime(first_release_long["reference_month"]).isin(eligible_months)
    ].copy()
    if subset.duplicated(["reference_month", "component"]).any():
        raise ValueError("first-release table contains duplicate component months")

    cutoff = pd.Timestamp(knowledge_cutoff).normalize()
    records: list[dict[str, object]] = []
    for row in subset.sort_values(["reference_month", "component"]).itertuples(
        index=False
    ):
        reference_month = pd.Timestamp(row.reference_month)
        component = str(row.component)
        series_id = str(row.series_id)
        previous_month = (reference_month.to_period("M") - 1).to_timestamp()
        first_release_date = pd.Timestamp(row.release_date).normalize()
        scale = (
            float(scales.at[reference_month, component])
            if reference_month in scales.index and component in scales
            else float("nan")
        )
        matrix = exact_vintage_matrices.get(series_id)

        for raw_horizon in horizons:
            horizon = int(raw_horizon)
            maturity = first_release_date + pd.DateOffset(months=horizon)
            record: dict[str, object] = {
                "reference_month": reference_month,
                "component": component,
                "series_id": series_id,
                "horizon_months": horizon,
                "first_release_date": first_release_date,
                "revision_available_at": maturity,
                "revision_vintage_date": maturity,
                "first_transformed_value": float(row.transformed_value),
                "first_release_scale": scale,
                "revised_current_value": np.nan,
                "revised_previous_value": np.nan,
                "revised_transformed_value": np.nan,
                "standardized_revision_error": np.nan,
                "revision_status": "available",
            }
            if maturity > cutoff:
                record["revision_status"] = "horizon_not_mature"
                records.append(record)
                continue
            if not np.isfinite(scale) or scale <= 0.0:
                record["revision_status"] = "missing_first_release_scale"
                records.append(record)
                continue
            if matrix is None:
                record["revision_status"] = "missing_revision_matrix"
                records.append(record)
                continue
            vintage_column = f"{series_id}_{maturity:%Y%m%d}"
            if vintage_column not in matrix.columns:
                record["revision_status"] = "missing_exact_vintage"
                records.append(record)
                continue
            if reference_month not in matrix.index or previous_month not in matrix.index:
                record["revision_status"] = "missing_reference_level"
                records.append(record)
                continue
            current = matrix.at[reference_month, vintage_column]
            previous = matrix.at[previous_month, vintage_column]
            if pd.isna(current) or pd.isna(previous):
                record["revision_status"] = "missing_reference_level"
                records.append(record)
                continue
            current_value = float(current)
            previous_value = float(previous)
            try:
                revised = apply_transform(
                    current_value, previous_value, str(row.transform)
                )
            except ValueError:
                record["revision_status"] = "invalid_revised_transform"
                records.append(record)
                continue
            record.update(
                {
                    "revised_current_value": current_value,
                    "revised_previous_value": previous_value,
                    "revised_transformed_value": revised,
                    "standardized_revision_error": (
                        revised - float(row.transformed_value)
                    )
                    / scale,
                }
            )
            records.append(record)

    return pd.DataFrame.from_records(records).sort_values(
        ["reference_month", "horizon_months", "component"]
    )


def aggregate_axis_revision_errors(
    component_errors: pd.DataFrame,
    *,
    reference_months: Sequence[pd.Timestamp],
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Aggregate complete component revisions into bivariate score errors."""

    records: list[dict[str, object]] = []
    for reference_month in sorted(pd.Timestamp(value) for value in reference_months):
        for raw_horizon in horizons:
            horizon = int(raw_horizon)
            group = component_errors[
                (component_errors["reference_month"] == reference_month)
                & (component_errors["horizon_months"] == horizon)
            ]
            status_counts = group["revision_status"].value_counts().to_dict()
            record: dict[str, object] = {
                "reference_month": reference_month,
                "horizon_months": horizon,
                "revision_available_at": pd.NaT,
                "growth_revision_error": np.nan,
                "inflation_revision_error": np.nan,
                "revision_status": "incomplete_component_revision",
                "component_statuses": ";".join(
                    f"{key}:{value}" for key, value in sorted(status_counts.items())
                ),
            }
            if len(group) != len(ALL_COMPONENTS):
                record["revision_status"] = "missing_component_row"
                records.append(record)
                continue
            if not (group["revision_status"] == "available").all():
                if (group["revision_status"] == "horizon_not_mature").any():
                    record["revision_status"] = "horizon_not_mature"
                records.append(record)
                continue
            indexed = group.set_index("component")
            if set(indexed.index) != set(ALL_COMPONENTS):
                record["revision_status"] = "component_set_mismatch"
                records.append(record)
                continue
            errors = indexed["standardized_revision_error"].astype(float)
            record.update(
                {
                    "revision_available_at": pd.to_datetime(
                        group["revision_available_at"]
                    ).max(),
                    "growth_revision_error": float(
                        errors.reindex(GROWTH_COMPONENTS).mean()
                    ),
                    "inflation_revision_error": float(
                        errors.reindex(INFLATION_COMPONENTS).mean()
                    ),
                    "revision_status": "available",
                }
            )
            records.append(record)
    return pd.DataFrame.from_records(records).sort_values(
        ["reference_month", "horizon_months"]
    )
