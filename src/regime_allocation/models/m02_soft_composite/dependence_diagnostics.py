"""Diagnose residual dependence across Model 02 release blocks.

The Bayesian filter's baseline event likelihood factorizes across release
blocks.  That approximation is credible only when genuinely out-of-sample
emission residuals do not retain large cross-block or serial dependence.  This
module therefore consumes *causal rolling-origin* residual rows and reports:

* monthly cross-model Pearson and Spearman correlations at leads/lags
  ``-1, 0, +1``;
* Benjamini-Hochberg adjusted q-values across the complete family of reported
  cross-block tests; and
* Ljung-Box portmanteau diagnostics, with their own Benjamini-Hochberg family,
  for each residual series.

Weekly ICSA residuals are averaged within their reference month only for the
cross-block comparison.  Their serial tests always use the original weekly
sequence.  A single publication can legitimately contain a catch-up batch of
several weekly observations.  Those rows are retained separately and, when an
``observation_date`` column is supplied, ordered by that original observation
date.  Monthly response series are never silently averaged: duplicate
response/month rows raise an error because they would make the diagnostic's
sampling unit ambiguous.

Responses estimated jointly in one observation model are omitted from the
pairwise cross-model table because their dependence is already represented by
that model's residual covariance.  Distinct observation models remain eligible
even when they share an economic ``block_id``; this is essential for
asynchronous releases such as two inflation-pressure submodels.

These are diagnostics, not proofs of conditional independence.  In
particular, a large p-value can reflect low power, short histories, or noisy
emission estimates; it must not be interpreted as evidence that two blocks
are independent.  Correlation p- and q-values are explicitly labeled
exploratory when either series retains detected serial dependence.  The module
performs no file or network access and does not depend on ``statsmodels``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats


INDEPENDENCE_CAVEAT = (
    "Failure to reject a dependence test does not establish conditional "
    "independence; inspect effect sizes, sample coverage, serial dependence, "
    "and statistical power."
)


@dataclass(frozen=True)
class ResidualColumnSpec:
    """Column names required by the dependence diagnostics."""

    block_id: str = "block_id"
    model_id: str = "model_id"
    response_id: str = "response_id"
    reference_month: str = "reference_month"
    validation_available_at: str = "validation_available_at"
    residual: str = "standardized_residual"
    observation_date: str | None = None

    def __post_init__(self) -> None:
        required_fields = (
            "block_id",
            "model_id",
            "response_id",
            "reference_month",
            "validation_available_at",
            "residual",
        )
        normalized = tuple(
            str(getattr(self, field)).strip() for field in required_fields
        )
        if any(not value for value in normalized):
            raise ValueError("residual diagnostic column names cannot be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("residual diagnostic column names must be distinct")
        for field, value in zip(required_fields, normalized):
            object.__setattr__(self, field, value)

        if self.observation_date is not None:
            observation_date = str(self.observation_date).strip()
            if not observation_date:
                raise ValueError("observation_date column name cannot be empty")
            if observation_date in normalized:
                raise ValueError("residual diagnostic column names must be distinct")
            object.__setattr__(self, "observation_date", observation_date)


@dataclass(frozen=True)
class ReleaseBlockDependenceReport:
    """Tables and audit metadata produced by one dependence analysis."""

    monthly_residuals: pd.DataFrame
    cross_block_tests: pd.DataFrame
    serial_tests: pd.DataFrame
    metadata: dict[str, Any]

    def to_audit_dict(self) -> dict[str, Any]:
        """Return JSON-safe report metadata and table row counts."""

        return {
            **self.metadata,
            "monthly_residual_rows": int(len(self.monthly_residuals)),
            "cross_block_test_rows": int(len(self.cross_block_tests)),
            "serial_test_rows": int(len(self.serial_tests)),
        }


def _identifier_set(values: Sequence[str], *, label: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of identifiers")
    normalized = tuple(str(value).strip().casefold() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{label} must contain at least one nonempty identifier")
    return frozenset(normalized)


def normalize_oos_residuals(
    residuals: pd.DataFrame,
    *,
    columns: ResidualColumnSpec = ResidualColumnSpec(),
) -> pd.DataFrame:
    """Validate and normalize causal rolling-origin residual rows.

    The function cannot independently prove that a residual was generated
    causally.  The caller must supply a validation-availability timestamp from
    the rolling-origin fit.  Requiring and preserving that timestamp makes the
    point-in-time lineage auditable downstream.
    """

    if not isinstance(residuals, pd.DataFrame):
        raise TypeError("residuals must be a pandas DataFrame")
    required = [
        columns.block_id,
        columns.model_id,
        columns.response_id,
        columns.reference_month,
        columns.validation_available_at,
        columns.residual,
    ]
    if columns.observation_date is not None:
        required.append(columns.observation_date)
    missing = set(required).difference(residuals.columns)
    if missing:
        raise ValueError(
            "residual table omits columns: " + ", ".join(sorted(missing))
        )

    table = residuals.loc[:, required].copy()
    # Stable source order is an explicit final tie-breaker when an old caller
    # cannot supply the original observation date for a same-publication batch.
    table["source_row_order"] = np.arange(len(table), dtype=np.int64)
    rename = {
        columns.block_id: "block_id",
        columns.model_id: "model_id",
        columns.response_id: "response_id",
        columns.reference_month: "reference_month",
        columns.validation_available_at: "validation_available_at",
        columns.residual: "standardized_residual",
    }
    if columns.observation_date is not None:
        rename[columns.observation_date] = "observation_date"
    table = table.rename(columns=rename)

    for identifier in ("block_id", "model_id", "response_id"):
        table[identifier] = table[identifier].astype("string").str.strip()
        invalid = table[identifier].isna() | table[identifier].eq("")
        if bool(invalid.any()):
            raise ValueError(f"{identifier} contains missing or empty values")

    raw_month = table["reference_month"]
    parsed_month = pd.to_datetime(raw_month, errors="coerce", utc=True)
    if bool(parsed_month.isna().any()):
        raise ValueError("reference_month contains invalid timestamps")
    table["reference_month"] = (
        parsed_month.dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    )

    parsed_available = pd.to_datetime(
        table["validation_available_at"], errors="coerce", utc=True
    )
    if bool(parsed_available.isna().any()):
        raise ValueError("validation_available_at contains invalid timestamps")
    table["validation_available_at"] = parsed_available.dt.tz_convert(None)

    if columns.observation_date is not None:
        parsed_observation = pd.to_datetime(
            table["observation_date"], errors="coerce", utc=True
        )
        if bool(parsed_observation.isna().any()):
            raise ValueError("observation_date contains invalid timestamps")
        table["observation_date"] = parsed_observation.dt.tz_convert(None)

    values = pd.to_numeric(table["standardized_residual"], errors="coerce")
    if bool(values.isna().any()) or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("standardized_residual must contain finite numeric values")
    table["standardized_residual"] = values.astype(float)

    # Do not declare rows duplicates merely because they share a publication
    # timestamp.  FRED/ALFRED catch-up releases can publish several distinct
    # weekly observations together.  Downstream monthly-series validation still
    # rejects ambiguous duplicate response/month rows.  Weekly serial ordering
    # uses observation_date when present and otherwise discloses its fallback.
    sort_columns = ["validation_available_at"]
    if columns.observation_date is not None:
        sort_columns.append("observation_date")
    sort_columns.extend(
        ["reference_month", "block_id", "model_id", "response_id", "source_row_order"]
    )
    return table.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def aggregate_residuals_to_monthly(
    residuals: pd.DataFrame,
    *,
    weekly_response_ids: Sequence[str] = ("ICSA",),
    columns: ResidualColumnSpec = ResidualColumnSpec(),
) -> pd.DataFrame:
    """Create the monthly residual panel used for cross-block tests.

    Weekly response identifiers are averaged by reference month.  Every other
    series must contribute at most one residual per reference month.
    ``validation_available_at`` for an aggregate is the latest timestamp among
    its constituent residuals, preserving a conservative availability marker.
    """

    table = normalize_oos_residuals(residuals, columns=columns)
    weekly_ids = _identifier_set(
        weekly_response_ids, label="weekly_response_ids"
    )
    weekly_mask = table["response_id"].str.casefold().isin(weekly_ids)
    identifiers = ["block_id", "model_id", "response_id", "reference_month"]

    weekly = table.loc[weekly_mask]
    if weekly.empty:
        weekly_monthly = pd.DataFrame()
    else:
        weekly_monthly = (
            weekly.groupby(identifiers, as_index=False, sort=True, observed=True)
            .agg(
                standardized_residual=("standardized_residual", "mean"),
                validation_available_at=("validation_available_at", "max"),
                source_observation_count=("standardized_residual", "size"),
            )
            .assign(source_frequency="weekly_aggregated")
        )

    monthly = table.loc[~weekly_mask].copy()
    if bool(monthly.duplicated(identifiers, keep=False).any()):
        duplicates = monthly.loc[
            monthly.duplicated(identifiers, keep=False), identifiers
        ].drop_duplicates()
        example = duplicates.iloc[0].to_dict()
        raise ValueError(
            "nonweekly residual series has duplicate reference-month rows: "
            f"{example}"
        )
    if not monthly.empty:
        monthly["source_observation_count"] = 1
        monthly["source_frequency"] = "monthly"

    pieces = [piece for piece in (weekly_monthly, monthly) if not piece.empty]
    output_columns = [
        "block_id",
        "model_id",
        "response_id",
        "reference_month",
        "validation_available_at",
        "standardized_residual",
        "source_observation_count",
        "source_frequency",
    ]
    if not pieces:
        return pd.DataFrame(columns=output_columns)
    output = pd.concat(pieces, ignore_index=True, sort=False)
    return output.loc[:, output_columns].sort_values(
        ["reference_month", "block_id", "model_id", "response_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg q-values, preserving missing entries."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    invalid = np.isfinite(values) & ((values < 0.0) | (values > 1.0))
    if bool(invalid.any()):
        raise ValueError("finite p-values must lie in [0, 1]")
    finite_positions = np.flatnonzero(np.isfinite(values))
    adjusted = np.full(values.shape, np.nan, dtype=float)
    if not finite_positions.size:
        return adjusted

    finite_values = values[finite_positions]
    order = np.argsort(finite_values, kind="mergesort")
    ordered = finite_values[order]
    count = len(ordered)
    raw_adjusted = ordered * count / np.arange(1, count + 1, dtype=float)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    restored = np.empty(count, dtype=float)
    restored[order] = monotone
    adjusted[finite_positions] = restored
    return adjusted


def _series_key(row: pd.Series) -> tuple[str, str, str]:
    return (str(row["block_id"]), str(row["model_id"]), str(row["response_id"]))


def _correlation_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    statistic: str,
    minimum_observations: int,
) -> tuple[float, float, str]:
    if len(left) < minimum_observations:
        return math.nan, math.nan, "insufficient_observations"
    if np.var(left) <= np.finfo(float).eps or np.var(right) <= np.finfo(float).eps:
        return math.nan, math.nan, "constant_residual"
    if statistic == "pearson":
        result = stats.pearsonr(left, right)
    elif statistic == "spearman":
        result = stats.spearmanr(left, right)
    else:  # pragma: no cover - protected by the internal caller
        raise ValueError(f"unsupported correlation statistic: {statistic}")
    return float(result.statistic), float(result.pvalue), "ok"


def cross_block_correlation_tests(
    monthly_residuals: pd.DataFrame,
    *,
    lags: Sequence[int] = (-1, 0, 1),
    minimum_pair_observations: int = 12,
) -> pd.DataFrame:
    """Test monthly residual correlation across distinct observation models.

    Each unordered series pair is assigned a deterministic left/right order.
    Lag ``h`` correlates the left residual for month ``m`` with the right
    residual for month ``m + h``.  Benjamini-Hochberg q-values are computed
    jointly across every valid Pearson and Spearman test returned here.
    Responses from the same ``(block_id, model_id)`` are skipped because their
    joint covariance belongs inside that observation model.  Different models
    within the same economic block are retained.
    """

    required = {
        "block_id",
        "model_id",
        "response_id",
        "reference_month",
        "standardized_residual",
    }
    missing = required.difference(monthly_residuals.columns)
    if missing:
        raise ValueError(
            "monthly residual table omits columns: " + ", ".join(sorted(missing))
        )
    if minimum_pair_observations < 3:
        raise ValueError("minimum_pair_observations must be at least three")
    normalized_lags = tuple(int(lag) for lag in lags)
    if len(normalized_lags) != len(set(normalized_lags)):
        raise ValueError("lags must not contain duplicates")
    if any(int(lag) != lag for lag in lags):
        raise ValueError("lags must be integer month offsets")

    table = monthly_residuals.copy()
    months = pd.to_datetime(table["reference_month"], errors="coerce")
    values = pd.to_numeric(table["standardized_residual"], errors="coerce")
    if bool(months.isna().any()) or bool(values.isna().any()):
        raise ValueError("monthly residual months and values must be valid")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("monthly residual values must be finite")
    table["reference_period"] = months.dt.to_period("M")
    table["standardized_residual"] = values.astype(float)
    keys = ["block_id", "model_id", "response_id", "reference_period"]
    if bool(table.duplicated(keys, keep=False).any()):
        raise ValueError("monthly residual table has duplicate series/month rows")

    series_columns = ["block_id", "model_id", "response_id"]
    series = sorted(
        {
            tuple(str(value) for value in row)
            for row in table.loc[:, series_columns].itertuples(index=False, name=None)
        }
    )
    rows: list[dict[str, Any]] = []
    indexed = {
        key: group.loc[:, ["reference_period", "standardized_residual"]]
        .sort_values("reference_period")
        .reset_index(drop=True)
        for key, group in table.groupby(series_columns, sort=False, observed=True)
    }
    for left_key, right_key in combinations(series, 2):
        if left_key[:2] == right_key[:2]:
            continue
        left = indexed[left_key].rename(
            columns={"standardized_residual": "left_residual"}
        )
        right_base = indexed[right_key].rename(
            columns={"standardized_residual": "right_residual"}
        )
        for lag in normalized_lags:
            right = right_base.copy()
            right["reference_period"] = right["reference_period"].map(
                lambda period, offset=lag: period - offset
            )
            aligned = left.merge(right, on="reference_period", how="inner")
            left_values = aligned["left_residual"].to_numpy(dtype=float)
            right_values = aligned["right_residual"].to_numpy(dtype=float)
            for statistic in ("pearson", "spearman"):
                coefficient, p_value, status = _correlation_test(
                    left_values,
                    right_values,
                    statistic=statistic,
                    minimum_observations=minimum_pair_observations,
                )
                rows.append(
                    {
                        "left_block_id": left_key[0],
                        "left_model_id": left_key[1],
                        "left_response_id": left_key[2],
                        "right_block_id": right_key[0],
                        "right_model_id": right_key[1],
                        "right_response_id": right_key[2],
                        "same_economic_block": left_key[0] == right_key[0],
                        "lag_months": int(lag),
                        "statistic": statistic,
                        "correlation": coefficient,
                        "sample_count": int(len(aligned)),
                        "p_value": p_value,
                        "status": status,
                    }
                )

    output_columns = [
        "left_block_id",
        "left_model_id",
        "left_response_id",
        "right_block_id",
        "right_model_id",
        "right_response_id",
        "same_economic_block",
        "lag_months",
        "statistic",
        "correlation",
        "sample_count",
        "p_value",
        "q_value",
        "status",
        "bh_family",
        "lag_definition",
        "left_serial_dependence_detected",
        "right_serial_dependence_detected",
        "serial_dependence_detected_either",
        "correlation_p_q_interpretation",
    ]
    if not rows:
        return pd.DataFrame(columns=output_columns)
    output = pd.DataFrame(rows)
    output["q_value"] = benjamini_hochberg(output["p_value"].to_numpy())
    output["bh_family"] = "all_cross_model_pairs_statistics_and_lags"
    output["lag_definition"] = (
        "right_reference_month = left_reference_month + lag_months"
    )
    output["left_serial_dependence_detected"] = pd.NA
    output["right_serial_dependence_detected"] = pd.NA
    output["serial_dependence_detected_either"] = pd.NA
    output["correlation_p_q_interpretation"] = (
        "exploratory_serial_diagnostic_not_attached"
    )
    return output.loc[:, output_columns].sort_values(
        [
            "left_block_id",
            "left_model_id",
            "left_response_id",
            "right_block_id",
            "right_model_id",
            "right_response_id",
            "lag_months",
            "statistic",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _ljung_box(
    values: np.ndarray,
    *,
    lag: int,
    minimum_observations: int,
) -> tuple[float, float, str]:
    count = len(values)
    if count < minimum_observations:
        return math.nan, math.nan, "insufficient_observations"
    centered = values - values.mean()
    denominator = float(centered @ centered)
    if denominator <= np.finfo(float).eps:
        return math.nan, math.nan, "constant_residual"
    autocorrelations = np.asarray(
        [
            float(centered[offset:] @ centered[:-offset]) / denominator
            for offset in range(1, lag + 1)
        ]
    )
    offsets = np.arange(1, lag + 1, dtype=float)
    statistic = count * (count + 2.0) * np.sum(
        autocorrelations**2 / (count - offsets)
    )
    p_value = stats.chi2.sf(statistic, df=lag)
    return float(statistic), float(p_value), "ok"


def serial_ljung_box_tests(
    residuals: pd.DataFrame,
    *,
    weekly_response_ids: Sequence[str] = ("ICSA",),
    weekly_lags: Sequence[int] = (1, 4, 8),
    monthly_lags: Sequence[int] = (1, 3, 6),
    minimum_observations: int = 12,
    minimum_cycles_per_lag: int = 3,
    columns: ResidualColumnSpec = ResidualColumnSpec(),
) -> pd.DataFrame:
    """Compute per-series Ljung-Box diagnostics at weekly/monthly lags.

    ICSA is tested on its original weekly rolling-origin residual sequence,
    before the monthly mean used in cross-block analysis.  When the optional
    ``observation_date`` column is configured, weekly rows are ordered by that
    original date and calendar-week gaps are audited.  Otherwise, rows are
    ordered by publication timestamp, reference month, and stable source order;
    any same-publication batches are explicitly disclosed.  Other models are
    ordered by reference month.  Their lags are calendar-month lags only when
    the observed reference months are contiguous, and otherwise are labeled
    ordered-observation lags with missing calendar periods.  The chi-square
    reference uses ``degrees_of_freedom = lag`` because the input table does not
    encode emission-model parameter counts.
    """

    table = normalize_oos_residuals(residuals, columns=columns)
    weekly_ids = _identifier_set(
        weekly_response_ids, label="weekly_response_ids"
    )
    if minimum_observations < 3:
        raise ValueError("minimum_observations must be at least three")
    if minimum_cycles_per_lag < 1:
        raise ValueError("minimum_cycles_per_lag must be positive")

    def valid_lags(values: Sequence[int], label: str) -> tuple[int, ...]:
        normalized = tuple(int(value) for value in values)
        if (
            not normalized
            or len(normalized) != len(set(normalized))
            or any(value < 1 for value in normalized)
            or any(int(original) != original for original in values)
        ):
            raise ValueError(f"{label} must contain distinct positive integers")
        return normalized

    weekly_lag_values = valid_lags(weekly_lags, "weekly_lags")
    monthly_lag_values = valid_lags(monthly_lags, "monthly_lags")
    rows: list[dict[str, Any]] = []
    group_columns = ["block_id", "model_id", "response_id"]
    for key, group in table.groupby(group_columns, sort=True, observed=True):
        publication_batch_sizes = group.groupby(
            "validation_available_at", sort=False, observed=True
        ).size()
        repeated_publication_batches = publication_batch_sizes.loc[
            publication_batch_sizes.gt(1)
        ]
        same_publication_date_batch_count = int(len(repeated_publication_batches))
        maximum_same_publication_date_batch_size = (
            int(repeated_publication_batches.max())
            if not repeated_publication_batches.empty
            else 1
        )
        is_weekly = str(key[2]).casefold() in weekly_ids
        if is_weekly:
            lag_values = weekly_lag_values
            frequency = "weekly"
            if "observation_date" in group:
                ordered = group.sort_values(
                    [
                        "observation_date",
                        "validation_available_at",
                        "reference_month",
                        "source_row_order",
                    ],
                    kind="mergesort",
                )
                observation_dates = ordered["observation_date"].to_numpy(
                    dtype="datetime64[D]"
                )
                day_gaps = np.diff(observation_dates).astype("timedelta64[D]").astype(int)
                calendar_gap_count = int(np.sum(day_gaps != 7))
                if calendar_gap_count:
                    lag_semantics = (
                        "ordered_weekly_observations_by_observation_date_with_"
                        "calendar_gaps"
                    )
                    calendar_spacing_status = "missing_or_irregular_observation_weeks"
                else:
                    lag_semantics = "calendar_weeks_by_observation_date"
                    calendar_spacing_status = "calendar_week_contiguous"
                within_publication_date_order = (
                    "observation_date_then_stable_source_order_for_ties"
                )
            else:
                ordered = group.sort_values(
                    [
                        "validation_available_at",
                        "reference_month",
                        "source_row_order",
                    ],
                    kind="mergesort",
                )
                calendar_gap_count = math.nan
                if same_publication_date_batch_count:
                    lag_semantics = (
                        "ordered_weekly_observations_same_publication_batches_"
                        "calendar_spacing_not_asserted"
                    )
                    calendar_spacing_status = (
                        "same_publication_batches_without_observation_date"
                    )
                    within_publication_date_order = (
                        "reference_month_then_stable_source_order"
                    )
                else:
                    lag_semantics = (
                        "ordered_weekly_observations_calendar_spacing_not_asserted"
                    )
                    calendar_spacing_status = (
                        "not_assessed_from_training_availability"
                    )
                    within_publication_date_order = "not_applicable_no_shared_timestamp"
        else:
            duplicate_month = group.duplicated("reference_month", keep=False)
            if bool(duplicate_month.any()):
                raise ValueError(
                    "nonweekly residual series has duplicate reference-month rows: "
                    f"{key}"
                )
            ordered = group.sort_values("reference_month", kind="mergesort")
            lag_values = monthly_lag_values
            frequency = "monthly"
            periods = ordered["reference_month"].dt.to_period("M")
            ordinal = periods.astype("int64").to_numpy()
            calendar_gap_count = int(np.sum(np.diff(ordinal) != 1))
            if calendar_gap_count:
                lag_semantics = "ordered_monthly_observations_with_calendar_gaps"
                calendar_spacing_status = "missing_reference_months"
            else:
                lag_semantics = "calendar_months"
                calendar_spacing_status = "calendar_contiguous"
            within_publication_date_order = (
                "reference_month_then_stable_source_order"
                if same_publication_date_batch_count
                else "not_applicable_no_shared_timestamp"
            )
        values = ordered["standardized_residual"].to_numpy(dtype=float)
        for lag in lag_values:
            required = max(
                minimum_observations,
                minimum_cycles_per_lag * lag,
                lag + 2,
            )
            statistic, p_value, status = _ljung_box(
                values,
                lag=lag,
                minimum_observations=required,
            )
            rows.append(
                {
                    "block_id": key[0],
                    "model_id": key[1],
                    "response_id": key[2],
                    "frequency": frequency,
                    "lag": int(lag),
                    "sample_count": int(len(values)),
                    "minimum_required": int(required),
                    "q_statistic": statistic,
                    "degrees_of_freedom": int(lag),
                    "p_value": p_value,
                    "status": status,
                    "lag_semantics": lag_semantics,
                    "calendar_gap_count": calendar_gap_count,
                    "calendar_spacing_status": calendar_spacing_status,
                    "same_publication_date_batch_count": (
                        same_publication_date_batch_count
                    ),
                    "maximum_same_publication_date_batch_size": (
                        maximum_same_publication_date_batch_size
                    ),
                    "within_publication_date_order": within_publication_date_order,
                }
            )
    output_columns = [
        "block_id",
        "model_id",
        "response_id",
        "frequency",
        "lag",
        "sample_count",
        "minimum_required",
        "q_statistic",
        "degrees_of_freedom",
        "p_value",
        "q_value",
        "status",
        "bh_family",
        "serial_dependence_detected_bh_5pct",
        "inference_label",
        "lag_semantics",
        "calendar_gap_count",
        "calendar_spacing_status",
        "same_publication_date_batch_count",
        "maximum_same_publication_date_batch_size",
        "within_publication_date_order",
    ]
    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=output_columns)
    output["q_value"] = benjamini_hochberg(output["p_value"].to_numpy())
    output["bh_family"] = "all_valid_serial_series_and_lags"
    output["serial_dependence_detected_bh_5pct"] = (
        output["q_value"].notna() & output["q_value"].lt(0.05)
    )
    output["inference_label"] = "exploratory_portmanteau_diagnostic"
    return output.loc[:, output_columns]


def build_release_block_dependence_report(
    residuals: pd.DataFrame,
    *,
    weekly_response_ids: Sequence[str] = ("ICSA",),
    cross_block_lags: Sequence[int] = (-1, 0, 1),
    minimum_pair_observations: int = 12,
    serial_significance_level: float = 0.05,
    columns: ResidualColumnSpec = ResidualColumnSpec(),
) -> ReleaseBlockDependenceReport:
    """Build all Model 02 residual-dependence diagnostics."""

    if (
        not math.isfinite(float(serial_significance_level))
        or not 0.0 < float(serial_significance_level) < 1.0
    ):
        raise ValueError("serial_significance_level must lie strictly between zero and one")
    monthly = aggregate_residuals_to_monthly(
        residuals,
        weekly_response_ids=weekly_response_ids,
        columns=columns,
    )
    cross_block = cross_block_correlation_tests(
        monthly,
        lags=cross_block_lags,
        minimum_pair_observations=minimum_pair_observations,
    )
    serial = serial_ljung_box_tests(
        residuals,
        weekly_response_ids=weekly_response_ids,
        columns=columns,
    )
    serial_flags = (
        serial.assign(
            _detected=serial["q_value"].notna()
            & serial["q_value"].lt(float(serial_significance_level))
        )
        .groupby(["block_id", "model_id", "response_id"], observed=True)[
            "_detected"
        ]
        .any()
        .to_dict()
    )
    if not cross_block.empty:
        left_keys = zip(
            cross_block["left_block_id"],
            cross_block["left_model_id"],
            cross_block["left_response_id"],
        )
        right_keys = zip(
            cross_block["right_block_id"],
            cross_block["right_model_id"],
            cross_block["right_response_id"],
        )
        left_detected = [bool(serial_flags.get(tuple(key), False)) for key in left_keys]
        right_detected = [
            bool(serial_flags.get(tuple(key), False)) for key in right_keys
        ]
        either = np.asarray(left_detected) | np.asarray(right_detected)
        cross_block["left_serial_dependence_detected"] = left_detected
        cross_block["right_serial_dependence_detected"] = right_detected
        cross_block["serial_dependence_detected_either"] = either
        cross_block["correlation_p_q_interpretation"] = np.where(
            either,
            "exploratory_serial_dependence_detected",
            "exploratory_no_serial_dependence_detected",
        )
    metadata = {
        "method": "causal_oos_release_block_residual_dependence_diagnostics",
        "weekly_aggregation": (
            "ICSA residual mean by reference month for cross-block correlations only; "
            "serial tests retain every weekly residual"
        ),
        "same_publication_batch_policy": (
            "retain every row; order weekly serial residuals by configured "
            "observation_date, otherwise by publication time, reference month, "
            "and stable source order with an explicit caveat"
        ),
        "weekly_serial_lag_caveat": (
            "calendar-week semantics require an observation_date column and "
            "contiguous seven-day observation spacing"
        ),
        "cross_block_statistics": ["pearson", "spearman"],
        "cross_block_lags": [int(lag) for lag in cross_block_lags],
        "multiple_testing": (
            "Benjamini-Hochberg across all valid cross-model pairs, "
            "statistics, and lags"
        ),
        "serial_diagnostic": "Ljung-Box with df equal to reported lag",
        "serial_multiple_testing": (
            "Benjamini-Hochberg across all valid residual-series/lag tests"
        ),
        "serial_significance_level_for_correlation_label": float(
            serial_significance_level
        ),
        "correlation_p_q_values": "exploratory",
        "interpretation_caveat": INDEPENDENCE_CAVEAT,
    }
    return ReleaseBlockDependenceReport(
        monthly_residuals=monthly,
        cross_block_tests=cross_block,
        serial_tests=serial,
        metadata=metadata,
    )
