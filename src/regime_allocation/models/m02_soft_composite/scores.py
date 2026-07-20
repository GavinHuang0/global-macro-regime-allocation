"""Construct Model 02's deterministic growth and inflation scores.

Inputs are release-coherent, point-in-time component transformations created
before this module is called.  Every component is standardized against only
strictly earlier available observations.  Each axis is the fixed equal-weight
mean of four component z-scores.  Unlike Model 01, the scores are not averaged
over a trailing three-month window and are never converted to quadrant labels
in this module.
"""

from __future__ import annotations

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


def _validate_component_columns(frame: pd.DataFrame) -> None:
    missing = [component for component in ALL_COMPONENTS if component not in frame]
    if missing:
        raise ValueError(f"missing required component columns: {', '.join(missing)}")


def lagged_expanding_zscore(
    values: pd.Series,
    *,
    min_periods: int = 60,
    ddof: int = 1,
) -> pd.Series:
    """Standardize each value using observations from prior months only."""

    prior_mean, prior_std = lagged_expanding_moments(
        values, min_periods=min_periods, ddof=ddof
    )
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = (numeric - prior_mean) / prior_std
    result = result.where(prior_std > 0)
    result.name = values.name
    return result


def lagged_expanding_moments(
    values: pd.Series,
    *,
    min_periods: int = 60,
    ddof: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """Return causal expanding means and scales for one component series."""

    if min_periods < 2:
        raise ValueError("min_periods must be at least 2")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    prior_mean = numeric.expanding(min_periods=min_periods).mean().shift(1)
    prior_std = numeric.expanding(min_periods=min_periods).std(ddof=ddof).shift(1)
    prior_mean.name = values.name
    prior_std.name = values.name
    return prior_mean, prior_std


def build_composite_scores(
    transformed: pd.DataFrame,
    *,
    min_history: int = 60,
    ddof: int = 1,
) -> pd.DataFrame:
    """Return component audit columns and unsmoothed equal-weight scores.

    Missing values are never dynamically reweighted: all four component
    z-scores must be available for an axis score to exist.
    """

    _validate_component_columns(transformed)
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
        output[f"{axis}_score"] = output[z_columns].mean(axis=1, skipna=False)

    return output
