"""Construct causal AR(1) innovations for weekly initial claims.

The functions in this module are deliberately independent of data retrieval and
release-calendar logic. They expect a series ordered by reference week and an
optional parallel release-date sequence. Each AR(1) is estimated strictly from
observations published before the release-date group being forecast, and each
innovation is standardized strictly from innovations published on earlier dates.
The output records levels, fitted parameters, forecasts, residuals, expanding
scales, standardized innovations, and audit counts for every eligible release.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from numbers import Integral

import numpy as np
import pandas as pd


CLAIMS_AR1_COLUMNS = (
    "level",
    "log_level",
    "ar_prior_observation_count",
    "ar_transition_count",
    "ar_intercept",
    "ar_lag1_coefficient",
    "forecast_log_level",
    "forecast_level",
    "innovation_log",
    "standardization_prior_count",
    "innovation_prior_mean",
    "innovation_prior_std",
    "standardized_innovation",
)


def _validate_integer_parameter(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_levels(levels: pd.Series) -> pd.Series:
    if not isinstance(levels, pd.Series):
        raise TypeError("levels must be a pandas Series")
    if not levels.index.is_unique:
        raise ValueError("levels index must be unique")
    if levels.index.hasnans:
        raise ValueError("levels index cannot contain missing values")
    if not levels.index.is_monotonic_increasing:
        raise ValueError("levels must be ordered by increasing reference sequence")

    try:
        numeric = pd.to_numeric(levels, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("levels must contain only numeric values") from error

    values = numeric.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("levels must contain only finite, non-missing values")
    if (values <= 0.0).any():
        bad_positions = np.flatnonzero(values <= 0.0)[:3]
        labels = ", ".join(str(levels.index[position]) for position in bad_positions)
        raise ValueError(f"claims levels must be strictly positive; invalid at {labels}")
    return numeric


def _validate_release_dates(
    release_dates: Iterable[object] | None,
    *,
    levels_index: pd.Index,
) -> pd.Index:
    if release_dates is None:
        return levels_index.copy()
    if isinstance(release_dates, (str, bytes)):
        raise TypeError("release_dates must be an iterable aligned with levels")
    if isinstance(release_dates, pd.Series) and not release_dates.index.equals(
        levels_index
    ):
        raise ValueError("release_dates Series index must match levels index")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(list(release_dates), errors="raise"))
    except (TypeError, ValueError) as error:
        raise ValueError("release_dates must contain valid dates") from error
    if len(dates) != len(levels_index):
        raise ValueError("release_dates length must match levels")
    if dates.hasnans:
        raise ValueError("release_dates cannot contain missing values")
    if not dates.is_monotonic_increasing:
        raise ValueError("release_dates must be monotonically non-decreasing")
    return dates


def _fit_ar1_with_intercept(prior_log_levels: np.ndarray) -> tuple[float, float] | None:
    """Fit ``y_t = intercept + coefficient * y_(t-1)`` by OLS."""

    lagged = prior_log_levels[:-1]
    current = prior_log_levels[1:]
    lagged_mean = float(lagged.mean())
    current_mean = float(current.mean())
    centered_lagged = lagged - lagged_mean
    denominator = float(centered_lagged @ centered_lagged)

    # A constant prior history does not identify an intercept and a slope. Use
    # a scale-aware tolerance so numerical noise is not mistaken for variation.
    scale = max(1.0, float(lagged @ lagged))
    tolerance = np.finfo(float).eps * scale
    if denominator <= tolerance:
        return None

    coefficient = float(centered_lagged @ (current - current_mean) / denominator)
    intercept = current_mean - coefficient * lagged_mean
    if not (math.isfinite(intercept) and math.isfinite(coefficient)):
        return None
    return intercept, coefficient


def build_claims_ar1_features(
    levels: pd.Series,
    *,
    release_dates: Iterable[object] | None = None,
    min_history: int = 52,
    min_standardization_history: int | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """Build publication-group-causal AR(1) innovations from claims levels.

    Parameters
    ----------
    levels:
        Positive claims levels in reference-week order. The index must be unique
        and monotonically increasing. Missing observations must be resolved by
        the caller because a gap changes which observation is the lag-one level.
    release_dates:
        Publication dates aligned row-for-row with ``levels``. Duplicate dates
        represent catch-up batches. If omitted, the levels index is used and
        therefore every row is its own release group.
    min_history:
        Minimum number of level observations from *earlier publication dates*
        required to estimate the AR(1). At a singleton first eligible row this
        supplies ``min_history - 1`` lag/current transition pairs.
    min_standardization_history:
        Minimum number of valid innovations from earlier publication dates
        needed to standardize a batch. Defaults to ``min_history``.
    ddof:
        Delta degrees of freedom for the prior-innovation standard deviation.

    Returns
    -------
    pandas.DataFrame
        A frame indexed like ``levels``. Forecasts and ``innovation_log`` are
        in natural-log units; ``forecast_level`` converts the forecast back to
        original units. Counts, fitted parameters, and prior moments provide a
        row-level audit trail.

    Notes
    -----
    The AR parameters are frozen across a publication-date group and use only
    levels from earlier release dates. Within a catch-up group, rows are
    processed in reference order and each one-step forecast conditions on the
    preceding actual level, including an earlier level from the same group. The
    standardization moments are also frozen across the group and use only valid
    innovations from strictly earlier release dates. Levels and innovations
    from a group enter their respective histories only after the complete group.
    """

    min_history = _validate_integer_parameter(
        min_history,
        name="min_history",
        minimum=3,
    )
    if min_standardization_history is None:
        min_standardization_history = min_history
    min_standardization_history = _validate_integer_parameter(
        min_standardization_history,
        name="min_standardization_history",
        minimum=2,
    )
    ddof = _validate_integer_parameter(ddof, name="ddof", minimum=0)
    if ddof >= min_standardization_history:
        raise ValueError("ddof must be smaller than min_standardization_history")

    numeric = _validate_levels(levels)
    publication_dates = _validate_release_dates(
        release_dates,
        levels_index=numeric.index,
    )
    log_levels = np.log(numeric.to_numpy())
    row_count = len(numeric)

    output = pd.DataFrame(index=numeric.index, columns=CLAIMS_AR1_COLUMNS, dtype=float)
    output["level"] = numeric
    output["log_level"] = log_levels

    prior_innovations: list[float] = []
    group_start = 0
    while group_start < row_count:
        group_end = group_start + 1
        while (
            group_end < row_count
            and publication_dates[group_end] == publication_dates[group_start]
        ):
            group_end += 1

        positions = range(group_start, group_end)
        ar_prior_count = group_start
        ar_transition_count = max(ar_prior_count - 1, 0)
        standardization_prior_count = len(prior_innovations)
        prior_mean = np.nan
        prior_std = np.nan
        if standardization_prior_count:
            prior_values = np.asarray(prior_innovations, dtype=float)
            prior_mean = float(prior_values.mean())
            if standardization_prior_count > ddof:
                prior_std = float(prior_values.std(ddof=ddof))

        for position in positions:
            output.iat[
                position, output.columns.get_loc("ar_prior_observation_count")
            ] = ar_prior_count
            output.iat[
                position, output.columns.get_loc("ar_transition_count")
            ] = ar_transition_count
            output.iat[
                position, output.columns.get_loc("standardization_prior_count")
            ] = standardization_prior_count
            output.iat[
                position, output.columns.get_loc("innovation_prior_mean")
            ] = prior_mean
            output.iat[
                position, output.columns.get_loc("innovation_prior_std")
            ] = prior_std

        fitted = None
        if ar_prior_count >= min_history:
            fitted = _fit_ar1_with_intercept(log_levels[:group_start])

        group_innovations: list[float] = []
        if fitted is not None:
            intercept, coefficient = fitted
            for position in positions:
                # AR eligibility guarantees at least three earlier levels, so a
                # preceding actual reference-week value exists for every row.
                forecast_log = intercept + coefficient * log_levels[position - 1]
                innovation = log_levels[position] - forecast_log
                if not (math.isfinite(forecast_log) and math.isfinite(innovation)):
                    continue

                output.iat[
                    position, output.columns.get_loc("ar_intercept")
                ] = intercept
                output.iat[
                    position, output.columns.get_loc("ar_lag1_coefficient")
                ] = coefficient
                output.iat[
                    position, output.columns.get_loc("forecast_log_level")
                ] = forecast_log
                with np.errstate(over="ignore", invalid="ignore"):
                    forecast_level = float(np.exp(forecast_log))
                if math.isfinite(forecast_level):
                    output.iat[
                        position, output.columns.get_loc("forecast_level")
                    ] = forecast_level
                output.iat[
                    position, output.columns.get_loc("innovation_log")
                ] = innovation

                if (
                    standardization_prior_count >= min_standardization_history
                    and math.isfinite(prior_std)
                    and prior_std > 0.0
                ):
                    output.iat[
                        position,
                        output.columns.get_loc("standardized_innovation"),
                    ] = (innovation - prior_mean) / prior_std
                group_innovations.append(innovation)

        # This is intentionally after the complete release-date group. It
        # prevents an earlier catch-up row from entering the fitted AR or the
        # standardization distribution used by another row in the same batch.
        prior_innovations.extend(group_innovations)
        group_start = group_end

    for count_column in (
        "ar_prior_observation_count",
        "ar_transition_count",
        "standardization_prior_count",
    ):
        output[count_column] = output[count_column].astype(np.int64)

    return output
