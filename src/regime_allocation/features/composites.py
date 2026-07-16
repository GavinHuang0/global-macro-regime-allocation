"""Deterministic growth and inflation composite construction.

The helpers in this module are intentionally independent of any data provider.
Point-in-time vintage selection happens in :mod:`regime_allocation.data` before
these functions standardize and combine the monthly component features.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


GROWTH_COMPONENTS = (
    "payrolls",
    "industrial_production",
    "consumer_activity",
    "unemployment_rate",
)

INFLATION_COMPONENTS = (
    "core_cpi",
    "core_pce",
    "producer_prices",
    "average_hourly_earnings",
)

ALL_COMPONENTS = GROWTH_COMPONENTS + INFLATION_COMPONENTS


def _difference(values: pd.Series) -> pd.Series:
    return values.diff()


def _negative_difference(values: pd.Series) -> pd.Series:
    return -values.diff()


def _log_difference(values: pd.Series) -> pd.Series:
    invalid = values.notna() & (values <= 0)
    if invalid.any():
        dates = ", ".join(str(item) for item in values.index[invalid][:3])
        raise ValueError(f"log-difference inputs must be positive; invalid at {dates}")
    return 100.0 * np.log(values / values.shift(1))


COMPONENT_TRANSFORMS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "payrolls": _difference,
    "industrial_production": _log_difference,
    "consumer_activity": _log_difference,
    "unemployment_rate": _negative_difference,
    "core_cpi": _log_difference,
    "core_pce": _log_difference,
    "producer_prices": _log_difference,
    "average_hourly_earnings": _log_difference,
}


def _validate_component_columns(frame: pd.DataFrame) -> None:
    missing = [component for component in ALL_COMPONENTS if component not in frame]
    if missing:
        raise ValueError(f"missing required component columns: {', '.join(missing)}")


def transform_release_levels(levels: pd.DataFrame) -> pd.DataFrame:
    """Transform a level matrix whose rows share a coherent vintage.

    This helper is appropriate for synthetic tests or a single-vintage history.
    Production point-in-time data must instead transform the current and prior
    month from the *same first-release vintage*; see ``data.first_release``.
    """

    _validate_component_columns(levels)
    transformed = pd.DataFrame(index=levels.index)
    for component, transform in COMPONENT_TRANSFORMS.items():
        transformed[component] = transform(
            pd.to_numeric(levels[component], errors="coerce")
        )
    return transformed


def lagged_expanding_zscore(
    values: pd.Series,
    *,
    min_periods: int = 60,
    ddof: int = 1,
) -> pd.Series:
    """Standardize each value using expanding statistics ending one month earlier."""

    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    prior_mean = numeric.expanding(min_periods=min_periods).mean().shift(1)
    prior_std = numeric.expanding(min_periods=min_periods).std(ddof=ddof).shift(1)
    result = (numeric - prior_mean) / prior_std
    result = result.where(prior_std > 0)
    result.name = values.name
    return result


def build_composites_from_transformed(
    transformed: pd.DataFrame,
    *,
    min_history: int = 60,
    smoothing_window: int = 3,
    ddof: int = 1,
) -> pd.DataFrame:
    """Create lagged z-scores and equal-weight, trailing-smoothed composites."""

    _validate_component_columns(transformed)
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be positive")

    output = pd.DataFrame(index=transformed.index)
    for component in ALL_COMPONENTS:
        output[f"{component}_transformed"] = pd.to_numeric(
            transformed[component], errors="coerce"
        )
        output[f"{component}_z"] = lagged_expanding_zscore(
            transformed[component], min_periods=min_history, ddof=ddof
        )

    for axis, components in (
        ("growth", GROWTH_COMPONENTS),
        ("inflation", INFLATION_COMPONENTS),
    ):
        z_columns = [f"{component}_z" for component in components]
        # skipna=False enforces the fixed 25% weights. A missing component makes
        # the entire axis unavailable instead of silently reweighting it.
        output[f"{axis}_raw"] = output[z_columns].mean(axis=1, skipna=False)
        output[f"{axis}_smoothed"] = (
            output[f"{axis}_raw"]
            .rolling(window=smoothing_window, min_periods=smoothing_window)
            .mean()
        )

    return output


def build_composite_features(
    levels: pd.DataFrame,
    *,
    min_history: int = 60,
    smoothing_window: int = 3,
    ddof: int = 1,
) -> pd.DataFrame:
    """Convenience wrapper for a coherent-vintage level history."""

    transformed = transform_release_levels(levels)
    return build_composites_from_transformed(
        transformed,
        min_history=min_history,
        smoothing_window=smoothing_window,
        ddof=ddof,
    )

