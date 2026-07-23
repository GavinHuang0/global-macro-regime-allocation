"""Causal weekly return estimation for Model 02 allocation.

Model 01's monthly estimator is historically attested, so Model 02 keeps its
weekly horizon alignment in this isolated module.  Weekly observations inherit
the deterministic regime label for the calendar month containing the reference
Monday.  Mean shrinkage counts weeks (not months), while covariance remains in
weekly units for the caller to annualize.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from regime_allocation.portfolio.estimation import (
    CANONICAL_REGIME_IDS,
    RegimeReturnEstimate,
    _normalize_dates,
    _normalize_months,
    _require_columns,
    _validate_covariance,
)


def _fit_labeled_weekly_regime_returns(
    labeled_returns: pd.DataFrame,
    *,
    asset_ids: Sequence[str],
    kappa: float,
    minimum_observations: int,
    knowledge_cutoff: pd.Timestamp,
) -> RegimeReturnEstimate:
    """Fit shrunk regime moments to an already causal weekly sample."""

    assets = tuple(map(str, asset_ids))
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("asset_ids must be non-empty and unique")
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and non-negative")
    if (
        isinstance(minimum_observations, bool)
        or not isinstance(minimum_observations, (int, np.integer))
        or minimum_observations < 2
    ):
        raise ValueError("minimum_observations must be an integer of at least two")
    _require_columns(
        labeled_returns,
        ["reference_week", "regime_id", *assets],
        "weekly labeled return table",
    )
    training = labeled_returns.copy()
    training["reference_week"] = _normalize_dates(
        training["reference_week"], name="training reference_week"
    )
    if training["reference_week"].dt.dayofweek.ne(0).any():
        raise ValueError("training reference_week values must be Mondays")
    if training["reference_week"].duplicated().any():
        raise ValueError("training reference weeks must be unique")
    training["regime_id"] = training["regime_id"].astype(str)
    unknown = sorted(set(training["regime_id"]).difference(CANONICAL_REGIME_IDS))
    if unknown:
        raise ValueError(f"unknown regime identifiers: {unknown}")
    returns = training.loc[:, list(assets)].apply(pd.to_numeric, errors="coerce")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("training returns must be finite and common across assets")
    if len(training) < minimum_observations:
        raise ValueError(
            f"at least {minimum_observations} common labeled weekly observations "
            "are required"
        )

    global_mean = returns.mean(axis=0)
    regime_counts = (
        training["regime_id"]
        .value_counts()
        .reindex(CANONICAL_REGIME_IDS, fill_value=0)
        .astype(int)
    )
    sample_means = pd.DataFrame(
        np.nan,
        index=pd.Index(CANONICAL_REGIME_IDS, name="regime_id"),
        columns=list(assets),
        dtype=float,
    )
    production_means = sample_means.copy()
    residual_parts: list[np.ndarray] = []
    pooled_values = global_mean.to_numpy(dtype=float)
    for regime_id in CANONICAL_REGIME_IDS:
        mask = training["regime_id"].eq(regime_id).to_numpy()
        count = int(mask.sum())
        if count:
            regime_values = values[mask]
            sample = regime_values.mean(axis=0)
            sample_means.loc[regime_id] = sample
            residual_parts.append(regime_values - sample)
            production_means.loc[regime_id] = (
                count * sample + kappa * pooled_values
            ) / (count + kappa)
        else:
            production_means.loc[regime_id] = pooled_values

    residuals = np.vstack(residual_parts)
    if residuals.shape != values.shape or not np.isfinite(residuals).all():
        raise RuntimeError("within-regime residual construction failed")
    covariance_estimator = LedoitWolf(assume_centered=True).fit(residuals)
    covariance_array = np.asarray(covariance_estimator.covariance_, dtype=float)
    covariance_array = (covariance_array + covariance_array.T) / 2.0
    _validate_covariance(
        covariance_array,
        dimension=len(assets),
        name="Ledoit-Wolf within-regime covariance",
    )
    if not np.isfinite(pooled_values).all():
        raise RuntimeError("global mean estimation produced a non-finite value")
    if not np.isfinite(production_means.to_numpy(dtype=float)).all():
        raise RuntimeError("regime mean estimation produced a non-finite value")
    covariance = pd.DataFrame(
        covariance_array,
        index=pd.Index(assets, name="asset_id"),
        columns=list(assets),
    )
    return RegimeReturnEstimate(
        asset_ids=assets,
        regime_ids=CANONICAL_REGIME_IDS,
        kappa=float(kappa),
        minimum_observations=int(minimum_observations),
        training_count=len(training),
        regime_counts=regime_counts,
        global_mean=global_mean.reindex(list(assets)),
        regime_sample_means=sample_means,
        regime_means=production_means,
        shared_within_covariance=covariance,
        ledoit_wolf_shrinkage=float(covariance_estimator.shrinkage_),
        knowledge_cutoff=knowledge_cutoff,
        first_holding_month=training["reference_week"].min(),
        last_holding_month=training["reference_week"].max(),
    )


def fit_causal_weekly_regime_return_model(
    weekly_holding_returns: pd.DataFrame,
    regime_history: pd.DataFrame,
    *,
    knowledge_cutoff: str | pd.Timestamp,
    asset_ids: Sequence[str],
    kappa: float = 104.0,
    minimum_observations: int = 260,
) -> RegimeReturnEstimate:
    """Fit weekly regime moments using only information available by a cutoff.

    Each ``reference_week`` must be a normalized Monday.  Its return receives
    the label for the calendar month containing that Monday, even when the
    open-to-open holding interval exits in the following month.  Both the
    weekly return and monthly label must be available on or before the cutoff.
    ``kappa`` and ``minimum_observations`` are counts of weekly observations.
    """

    assets = tuple(map(str, asset_ids))
    _require_columns(
        weekly_holding_returns,
        ["reference_week", "return_available_at", *assets],
        "weekly holding return table",
    )
    _require_columns(
        regime_history,
        ["reference_month", "regime_id", "label_available_at"],
        "regime history",
    )
    cutoff = pd.Timestamp(knowledge_cutoff)
    if pd.isna(cutoff):
        raise ValueError("knowledge_cutoff must be a valid date")
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(None)
    cutoff = cutoff.normalize()

    returns = weekly_holding_returns.copy()
    returns["reference_week"] = _normalize_dates(
        returns["reference_week"], name="return reference_week"
    )
    if returns["reference_week"].dt.dayofweek.ne(0).any():
        raise ValueError("return reference_week values must be Mondays")
    if returns["reference_week"].duplicated().any():
        raise ValueError("weekly holding returns contain duplicate reference weeks")
    returns["return_available_at"] = _normalize_dates(
        returns["return_available_at"], name="return_available_at"
    )
    availability_week = (
        returns["return_available_at"].dt.to_period("W-SUN").dt.start_time
    )
    expected_availability_week = returns["reference_week"] + pd.Timedelta(weeks=1)
    if availability_week.ne(expected_availability_week).any():
        raise ValueError(
            "return_available_at must fall in the immediately following "
            "Monday-anchored week"
        )
    returns["_regime_reference_month"] = (
        returns["reference_week"].dt.to_period("M").dt.to_timestamp()
    )

    history = regime_history.loc[
        :, ["reference_month", "regime_id", "label_available_at"]
    ].copy()
    history["reference_month"] = _normalize_months(
        history["reference_month"], name="regime reference_month"
    )
    history["label_available_at"] = pd.to_datetime(
        history["label_available_at"], errors="coerce"
    ).dt.normalize()
    if history["reference_month"].duplicated().any():
        raise ValueError("regime history contains duplicate reference months")
    history = history.rename(
        columns={"reference_month": "_regime_reference_month"}
    )
    merged = returns.merge(
        history,
        on="_regime_reference_month",
        how="left",
        validate="many_to_one",
    )
    eligible = merged.loc[
        merged["return_available_at"].le(cutoff)
        & merged["label_available_at"].le(cutoff)
        & merged["regime_id"].notna()
    ].copy()
    return _fit_labeled_weekly_regime_returns(
        eligible,
        asset_ids=assets,
        kappa=kappa,
        minimum_observations=minimum_observations,
        knowledge_cutoff=cutoff,
    )


__all__ = ["fit_causal_weekly_regime_return_model"]
