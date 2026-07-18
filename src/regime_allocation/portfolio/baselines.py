"""Transparent portfolio targets used as Model 01 backtest comparators.

The legacy Sharpe allocator deliberately preserves the course project's score
transformation while enforcing the rebuilt project's point-in-time contract.
It selects the maximum-a-posteriori (MAP) regime from a causal posterior and
uses only earlier daily returns whose calendar-month label was already known
strictly before the signal date. The module also supplies deterministic equal
weight and static 60/40 targets. Outputs are normalized target vectors plus,
for the legacy method, selection and fallback diagnostics suitable for audit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


__all__ = [
    "LegacySharpeMapResult",
    "equal_weight_target",
    "legacy_sharpe_map",
    "legacy_sharpe_map_weights",
    "static_60_40_target",
]


@dataclass(frozen=True)
class LegacySharpeMapResult:
    """Weights and audit fields from the causal legacy Sharpe rule."""

    weights: pd.Series
    map_regime_id: str
    observation_count: int
    annualized_sharpe_scores: pd.Series
    used_equal_weight_fallback: bool
    fallback_reason: str | None


def _asset_index(asset_ids: Sequence[str]) -> pd.Index:
    assets = pd.Index([str(asset_id) for asset_id in asset_ids], dtype="object")
    if assets.empty:
        raise ValueError("asset_ids must contain at least one asset")
    if assets.has_duplicates:
        duplicates = assets[assets.duplicated()].unique().tolist()
        raise ValueError(f"asset_ids contains duplicates: {duplicates}")
    if (assets.str.len() == 0).any():
        raise ValueError("asset_ids cannot contain an empty identifier")
    return assets


def equal_weight_target(asset_ids: Sequence[str]) -> pd.Series:
    """Return an exactly equal-weight target in the supplied asset order."""

    assets = _asset_index(asset_ids)
    weights = np.full(len(assets), 1.0 / len(assets), dtype=float)
    return pd.Series(weights, index=assets, name="target_weight")


def static_60_40_target(
    asset_ids: Sequence[str],
    *,
    risk_asset: str = "SPY",
    defensive_asset: str = "IEF",
    risk_weight: float = 0.60,
) -> pd.Series:
    """Return a configurable static two-asset target over a larger universe.

    Assets other than ``risk_asset`` and ``defensive_asset`` receive zero.
    The conventional 60/40 target is obtained with the default
    ``risk_weight=0.60``; exposing the parameter keeps the function useful for
    explicitly named sensitivity variants without changing its semantics.
    """

    assets = _asset_index(asset_ids)
    risk_asset = str(risk_asset)
    defensive_asset = str(defensive_asset)
    if risk_asset == defensive_asset:
        raise ValueError("risk_asset and defensive_asset must be different")
    missing = [asset for asset in (risk_asset, defensive_asset) if asset not in assets]
    if missing:
        raise ValueError(f"static 60/40 assets are absent from asset_ids: {missing}")
    if not np.isfinite(risk_weight) or not 0.0 <= risk_weight <= 1.0:
        raise ValueError("risk_weight must be finite and between zero and one")

    weights = pd.Series(0.0, index=assets, name="target_weight")
    weights.loc[risk_asset] = float(risk_weight)
    weights.loc[defensive_asset] = 1.0 - float(risk_weight)
    return weights


def _as_naive_normalized_timestamp(value: object, *, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid timestamp") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _posterior_map_regime(
    posterior_probabilities: Mapping[str, float] | pd.Series,
) -> str:
    probabilities = pd.Series(posterior_probabilities, dtype=float)
    probabilities.index = probabilities.index.map(str)
    if probabilities.empty:
        raise ValueError("posterior_probabilities cannot be empty")
    if probabilities.index.has_duplicates:
        raise ValueError("posterior_probabilities contains duplicate regime IDs")
    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("posterior probabilities must be finite and nonnegative")
    if float(values.sum()) <= 0.0:
        raise ValueError("posterior probabilities must have positive total mass")
    # idxmax returns the first supplied regime on an exact tie. Model callers
    # pass the repository's canonical regime order, making tie handling stable.
    return str(probabilities.idxmax())


def _prepare_daily_returns(
    daily_returns: pd.DataFrame,
    assets: pd.Index,
) -> pd.DataFrame:
    missing_assets = [asset for asset in assets if asset not in daily_returns.columns]
    if missing_assets:
        raise ValueError(f"daily_returns is missing assets: {missing_assets}")

    if "date" in daily_returns.columns:
        dates = pd.to_datetime(daily_returns["date"], errors="raise")
    elif isinstance(daily_returns.index, pd.DatetimeIndex):
        dates = daily_returns.index
    else:
        raise ValueError("daily_returns must have a date column or DatetimeIndex")
    dates = pd.DatetimeIndex(dates)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()
    if dates.has_duplicates:
        raise ValueError("daily_returns contains duplicate dates")

    prepared = daily_returns.loc[:, assets].apply(pd.to_numeric, errors="coerce")
    prepared.index = dates
    prepared = prepared.replace([np.inf, -np.inf], np.nan).sort_index()
    return prepared


def _prepare_regime_history(regime_history: pd.DataFrame) -> pd.DataFrame:
    required = {"reference_month", "regime_id", "label_available_at"}
    missing = sorted(required.difference(regime_history.columns))
    if missing:
        raise ValueError(f"regime_history is missing columns: {missing}")

    history = regime_history.loc[:, sorted(required)].copy()
    history["reference_month"] = pd.to_datetime(
        history["reference_month"], errors="raise"
    )
    history["label_available_at"] = pd.to_datetime(
        history["label_available_at"], errors="coerce"
    )
    for column in ("reference_month", "label_available_at"):
        series = history[column]
        if series.dt.tz is not None:
            series = series.dt.tz_localize(None)
        history[column] = series.dt.normalize()
    history["reference_month"] = (
        history["reference_month"].dt.to_period("M").dt.to_timestamp()
    )
    history["regime_id"] = history["regime_id"].where(
        history["regime_id"].notna(), pd.NA
    )
    history.loc[history["regime_id"].notna(), "regime_id"] = history.loc[
        history["regime_id"].notna(), "regime_id"
    ].astype(str)
    return history


def legacy_sharpe_map(
    daily_returns: pd.DataFrame,
    regime_history: pd.DataFrame,
    posterior_probabilities: Mapping[str, float] | pd.Series,
    *,
    signal_date: object,
    asset_ids: Sequence[str] | None = None,
    minimum_observations: int = 60,
    annualization_factor: int = 252,
    c_shift: float = 0.1,
) -> LegacySharpeMapResult:
    """Apply the legacy Sharpe score rule with causal regime selection.

    The MAP regime is chosen from ``posterior_probabilities``. A daily return
    is eligible only when it occurred strictly before ``signal_date`` and its
    calendar month's deterministic label became available strictly before the
    same signal date. All assets use one common complete-case sample.

    If fewer than ``minimum_observations`` eligible days remain, the function
    returns equal weights. Otherwise the legacy score and positivity shift are

    ``score_i = sqrt(annualization_factor) * mean_i / std_i``

    ``shift = abs(min_i(score_i)) + c_shift``.

    A zero-variance or otherwise nonfinite asset score is replaced with zero,
    a neutral finite score. This retains usable information from the other
    assets and guarantees finite positive weights. If the transformed weights
    nevertheless cannot be normalized, the allocator safely falls back to
    equal weight.
    """

    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least two")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    if not np.isfinite(c_shift) or c_shift <= 0.0:
        raise ValueError("c_shift must be finite and strictly positive")

    if asset_ids is None:
        inferred = [column for column in daily_returns.columns if column != "date"]
        assets = _asset_index(inferred)
    else:
        assets = _asset_index(asset_ids)
    equal_weights = equal_weight_target(assets.tolist())
    signal = _as_naive_normalized_timestamp(signal_date, field_name="signal_date")
    map_regime_id = _posterior_map_regime(posterior_probabilities)
    returns = _prepare_daily_returns(daily_returns, assets)
    history = _prepare_regime_history(regime_history)

    eligible_history = history.loc[
        history["regime_id"].notna()
        & history["label_available_at"].notna()
        & history["label_available_at"].lt(signal)
    ].copy()
    if eligible_history["reference_month"].duplicated().any():
        duplicates = (
            eligible_history.loc[
                eligible_history["reference_month"].duplicated(keep=False),
                "reference_month",
            ]
            .dt.strftime("%Y-%m")
            .unique()
            .tolist()
        )
        raise ValueError(f"eligible regime_history contains duplicate months: {duplicates}")
    regime_by_month = eligible_history.set_index("reference_month")["regime_id"]

    return_months = returns.index.to_period("M").to_timestamp()
    return_regimes = pd.Series(return_months, index=returns.index).map(regime_by_month)
    eligible = returns.loc[
        (returns.index < signal) & return_regimes.eq(map_regime_id)
    ].dropna(how="any")
    observation_count = int(len(eligible))
    empty_scores = pd.Series(np.nan, index=assets, name="annualized_sharpe")
    if observation_count < minimum_observations:
        return LegacySharpeMapResult(
            weights=equal_weights,
            map_regime_id=map_regime_id,
            observation_count=observation_count,
            annualized_sharpe_scores=empty_scores,
            used_equal_weight_fallback=True,
            fallback_reason="insufficient_same_regime_observations",
        )

    means = eligible.mean(axis=0)
    standard_deviations = eligible.std(axis=0, ddof=1)
    valid_score = (
        np.isfinite(means)
        & np.isfinite(standard_deviations)
        & standard_deviations.gt(np.finfo(float).eps)
    )
    scores = pd.Series(0.0, index=assets, dtype=float)
    scores.loc[valid_score] = (
        np.sqrt(float(annualization_factor))
        * means.loc[valid_score]
        / standard_deviations.loc[valid_score]
    )
    scores = scores.where(np.isfinite(scores), 0.0)
    scores.name = "annualized_sharpe"
    shift = abs(float(scores.min())) + float(c_shift)
    shifted_scores = scores + shift
    denominator = float(shifted_scores.sum())
    if (
        not np.isfinite(shifted_scores.to_numpy(dtype=float)).all()
        or (shifted_scores <= 0.0).any()
        or not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        return LegacySharpeMapResult(
            weights=equal_weights,
            map_regime_id=map_regime_id,
            observation_count=observation_count,
            annualized_sharpe_scores=scores,
            used_equal_weight_fallback=True,
            fallback_reason="unnormalizable_shifted_scores",
        )

    weights = (shifted_scores / denominator).rename("target_weight")
    return LegacySharpeMapResult(
        weights=weights,
        map_regime_id=map_regime_id,
        observation_count=observation_count,
        annualized_sharpe_scores=scores,
        used_equal_weight_fallback=False,
        fallback_reason=None,
    )


def legacy_sharpe_map_weights(
    daily_returns: pd.DataFrame,
    regime_history: pd.DataFrame,
    posterior_probabilities: Mapping[str, float] | pd.Series,
    *,
    signal_date: object,
    asset_ids: Sequence[str] | None = None,
    minimum_observations: int = 60,
    annualization_factor: int = 252,
    c_shift: float = 0.1,
) -> pd.Series:
    """Return only the target weights from :func:`legacy_sharpe_map`."""

    return legacy_sharpe_map(
        daily_returns,
        regime_history,
        posterior_probabilities,
        signal_date=signal_date,
        asset_ids=asset_ids,
        minimum_observations=minimum_observations,
        annualization_factor=annualization_factor,
        c_shift=c_shift,
    ).weights
