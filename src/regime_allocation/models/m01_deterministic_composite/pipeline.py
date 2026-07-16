"""Four-quadrant regime classification for model 01."""

from __future__ import annotations

from enum import StrEnum
import math

import pandas as pd


class Regime(StrEnum):
    """Stable machine identifiers for the four composite quadrants."""

    GROWTH_UP_INFLATION_UP = "growth_up_inflation_up"
    GROWTH_DOWN_INFLATION_UP = "growth_down_inflation_up"
    GROWTH_UP_INFLATION_DOWN = "growth_up_inflation_down"
    GROWTH_DOWN_INFLATION_DOWN = "growth_down_inflation_down"


REGIME_ORDER = (
    Regime.GROWTH_UP_INFLATION_UP,
    Regime.GROWTH_DOWN_INFLATION_UP,
    Regime.GROWTH_UP_INFLATION_DOWN,
    Regime.GROWTH_DOWN_INFLATION_DOWN,
)


REGIME_LABELS = {
    Regime.GROWTH_UP_INFLATION_UP: "Growth composite up / inflation composite up",
    Regime.GROWTH_DOWN_INFLATION_UP: "Growth composite down / inflation composite up",
    Regime.GROWTH_UP_INFLATION_DOWN: "Growth composite up / inflation composite down",
    Regime.GROWTH_DOWN_INFLATION_DOWN: "Growth composite down / inflation composite down",
}


def classify_regime(growth_score: float, inflation_score: float) -> Regime:
    """Map finite scores to a quadrant; exact zero belongs to the up side."""

    if not math.isfinite(growth_score) or not math.isfinite(inflation_score):
        raise ValueError("growth and inflation scores must be finite")

    growth_up = growth_score >= 0.0
    inflation_up = inflation_score >= 0.0
    if growth_up and inflation_up:
        return Regime.GROWTH_UP_INFLATION_UP
    if not growth_up and inflation_up:
        return Regime.GROWTH_DOWN_INFLATION_UP
    if growth_up and not inflation_up:
        return Regime.GROWTH_UP_INFLATION_DOWN
    return Regime.GROWTH_DOWN_INFLATION_DOWN


def classify_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Append regime identifiers and human labels to a composite feature frame."""

    required = {"growth_smoothed", "inflation_smoothed"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"missing score columns: {', '.join(sorted(missing))}")

    output = features.copy()
    regimes: list[str | None] = []
    labels: list[str | None] = []
    for growth, inflation in output[
        ["growth_smoothed", "inflation_smoothed"]
    ].itertuples(index=False, name=None):
        if pd.isna(growth) or pd.isna(inflation):
            regimes.append(None)
            labels.append(None)
            continue
        regime = classify_regime(float(growth), float(inflation))
        regimes.append(regime.value)
        labels.append(REGIME_LABELS[regime])

    output["regime_id"] = pd.Series(regimes, index=output.index, dtype="string")
    output["regime_label"] = pd.Series(labels, index=output.index, dtype="string")
    return output
