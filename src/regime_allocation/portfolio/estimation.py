"""Causal return estimation for posterior-weighted regime allocation.

The helpers in this module deliberately separate three ideas:

* an archived, causal probability signal for the month being traded;
* monthly ETF holding returns measured from one executable adjusted open to
  the next;
* expanding regime-conditioned return estimates available at a decision
  cutoff.

The production regime means use an exact pseudo-month shrinkage rule.  The
shared within-regime covariance is fit to residuals around *unshrunk* sample
means, so mean shrinkage cannot mechanically inflate estimated risk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from regime_allocation.models.m01_deterministic_composite.pipeline import (
    REGIME_ORDER,
)


CANONICAL_REGIME_IDS = tuple(regime.value for regime in REGIME_ORDER)
PROBABILITY_COLUMNS = tuple(
    f"probability_{regime_id}" for regime_id in CANONICAL_REGIME_IDS
)
_NORMALIZATION_TOLERANCE = 1.0e-10
_PSD_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class RegimeReturnEstimate:
    """Causal regime-return moments available at one knowledge cutoff."""

    asset_ids: tuple[str, ...]
    regime_ids: tuple[str, ...]
    kappa: float
    minimum_observations: int
    training_count: int
    regime_counts: pd.Series
    global_mean: pd.Series
    regime_sample_means: pd.DataFrame
    regime_means: pd.DataFrame
    shared_within_covariance: pd.DataFrame
    ledoit_wolf_shrinkage: float
    knowledge_cutoff: pd.Timestamp | None
    first_holding_month: pd.Timestamp | None
    last_holding_month: pd.Timestamp | None


@dataclass(frozen=True)
class PosteriorMixtureMoments:
    """Expected moments after integrating over the regime posterior."""

    posterior: pd.Series
    expected_mean: pd.Series
    within_regime_covariance: pd.DataFrame
    between_regime_covariance: pd.DataFrame
    covariance: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _normalize_dates(values: pd.Series, *, name: str) -> pd.Series:
    converted = pd.to_datetime(values, errors="coerce")
    invalid = values.notna() & converted.isna()
    if invalid.any() or converted.isna().any():
        raise ValueError(f"{name} contains an invalid or missing date")
    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_convert(None)
    return converted.dt.normalize()


def _normalize_months(values: pd.Series, *, name: str) -> pd.Series:
    converted = _normalize_dates(values, name=name)
    if (converted.dt.day != 1).any():
        raise ValueError(f"{name} must contain normalized month starts")
    return converted


def _validate_probability_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (len(CANONICAL_REGIME_IDS),):
        raise ValueError(
            f"{name} must have shape ({len(CANONICAL_REGIME_IDS)},)"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} must contain only finite values")
    if (probabilities < 0.0).any():
        raise ValueError(f"{name} cannot contain negative values")
    if not np.isclose(
        probabilities.sum(),
        1.0,
        atol=_NORMALIZATION_TOLERANCE,
        rtol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(f"{name} must sum to one")
    return probabilities


def _validate_covariance(
    values: np.ndarray,
    *,
    dimension: int,
    name: str,
) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    if covariance.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.isfinite(covariance).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(covariance, covariance.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -_PSD_TOLERANCE * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return covariance


def extract_post_month_roll_signals(
    checkpoints: pd.DataFrame,
    marginals: pd.DataFrame,
    *,
    specification_id: str = "baseline",
) -> pd.DataFrame:
    """Extract the causal current-month marginal after each monthly path roll.

    The selected checkpoint is phase one on the first calendar day of the
    holding month: the previous joint posterior has already been propagated
    with a transition matrix trained through the prior day, but same-day
    releases and confirmations have not yet been applied.
    """

    _require_columns(
        checkpoints,
        [
            "checkpoint_id",
            "specification_id",
            "as_of_date",
            "phase_order",
            "checkpoint_type",
            "anchor_month",
            "transition_training_cutoff",
        ],
        "checkpoint table",
    )
    _require_columns(
        marginals,
        [
            "checkpoint_id",
            "specification_id",
            "reference_month",
            "relative_month",
            "marginal_type",
            "regime_id",
            "probability",
        ],
        "marginal table",
    )
    selected = checkpoints.loc[
        checkpoints["specification_id"].astype(str).eq(specification_id)
        & checkpoints["checkpoint_type"].astype(str).eq("post_month_roll")
    ].copy()
    if selected.empty:
        raise ValueError(
            f"no post_month_roll checkpoints exist for {specification_id!r}"
        )
    if selected["checkpoint_id"].duplicated().any():
        raise ValueError("post_month_roll checkpoint IDs must be unique")
    selected["signal_date"] = _normalize_dates(
        selected["as_of_date"], name="checkpoint as_of_date"
    )
    selected["holding_month"] = _normalize_months(
        selected["anchor_month"], name="checkpoint anchor_month"
    )
    selected["transition_training_cutoff"] = _normalize_dates(
        selected["transition_training_cutoff"],
        name="transition_training_cutoff",
    )
    if not selected["phase_order"].eq(1).all():
        raise ValueError("post_month_roll checkpoints must have phase_order=1")
    if not selected["signal_date"].eq(selected["holding_month"]).all():
        raise ValueError("post_month_roll signals must occur at holding-month start")
    if not (
        selected["transition_training_cutoff"] < selected["signal_date"]
    ).all():
        raise ValueError("transition training must end before the signal date")
    if selected["holding_month"].duplicated().any():
        raise ValueError("each holding month must have exactly one signal")

    selected_marginals = marginals.loc[
        marginals["checkpoint_id"].isin(selected["checkpoint_id"])
        & marginals["specification_id"].astype(str).eq(specification_id)
        & marginals["marginal_type"].astype(str).eq("path")
        & pd.to_numeric(marginals["relative_month"], errors="coerce").eq(0)
    ].copy()
    selected_marginals["reference_month"] = _normalize_months(
        selected_marginals["reference_month"], name="marginal reference_month"
    )
    selected_marginals["regime_id"] = selected_marginals["regime_id"].astype(str)
    selected_marginals["probability"] = pd.to_numeric(
        selected_marginals["probability"], errors="coerce"
    )
    metadata = selected.set_index("checkpoint_id")
    expected_month = selected_marginals["checkpoint_id"].map(
        metadata["holding_month"]
    )
    if not selected_marginals["reference_month"].eq(expected_month).all():
        raise ValueError("current-month marginals must reference the anchor month")

    records: list[dict[str, object]] = []
    for checkpoint_id, group in selected_marginals.groupby(
        "checkpoint_id", sort=False
    ):
        if group["regime_id"].duplicated().any():
            raise ValueError(f"checkpoint {checkpoint_id!r} has duplicate regimes")
        supplied = set(group["regime_id"])
        if supplied != set(CANONICAL_REGIME_IDS):
            raise ValueError(
                f"checkpoint {checkpoint_id!r} does not contain canonical regimes"
            )
        ordered = group.set_index("regime_id").loc[list(CANONICAL_REGIME_IDS)]
        probabilities = _validate_probability_vector(
            ordered["probability"].to_numpy(dtype=float),
            name=f"checkpoint {checkpoint_id!r} probabilities",
        )
        checkpoint = metadata.loc[checkpoint_id]
        row: dict[str, object] = {
            "holding_month": checkpoint["holding_month"],
            "signal_date": checkpoint["signal_date"],
            "checkpoint_id": checkpoint_id,
            "transition_training_cutoff": checkpoint[
                "transition_training_cutoff"
            ],
        }
        row.update(
            {
                column: float(probabilities[position])
                for position, column in enumerate(PROBABILITY_COLUMNS)
            }
        )
        records.append(row)
    output = pd.DataFrame.from_records(records).sort_values("holding_month")
    if len(output) != len(selected):
        raise ValueError("every post_month_roll checkpoint must have one marginal")
    return output.reset_index(drop=True)


def build_adjusted_open_holding_returns(
    daily_prices: pd.DataFrame,
    *,
    asset_ids: Sequence[str],
) -> pd.DataFrame:
    """Build complete first-adjusted-open to next-first-open monthly returns.

    The first common US trading session in month ``m`` is the entry and the
    first common session in ``m+1`` is the exit.  The final incomplete month is
    intentionally omitted.  Missing assets, mismatched execution dates, and
    skipped calendar months fail rather than being forward-filled.
    """

    assets = tuple(map(str, asset_ids))
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("asset_ids must be non-empty and unique")
    _require_columns(
        daily_prices,
        ["date", "ticker", "adjusted_open"],
        "daily price table",
    )
    prices = daily_prices.loc[
        daily_prices["ticker"].astype(str).isin(assets),
        ["date", "ticker", "adjusted_open"],
    ].copy()
    prices["date"] = _normalize_dates(prices["date"], name="price date")
    prices["ticker"] = prices["ticker"].astype(str)
    prices["adjusted_open"] = pd.to_numeric(
        prices["adjusted_open"], errors="coerce"
    )
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("daily price table contains duplicate ticker-date rows")
    supplied_assets = set(prices["ticker"])
    if supplied_assets != set(assets):
        missing = sorted(set(assets).difference(supplied_assets))
        raise ValueError(f"daily price table is missing assets: {missing}")
    values = prices["adjusted_open"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("adjusted opens must be finite and strictly positive")
    prices["holding_month"] = prices["date"].dt.to_period("M").dt.to_timestamp()
    first_rows = (
        prices.sort_values(["holding_month", "ticker", "date"])
        .groupby(["holding_month", "ticker"], as_index=False, sort=True)
        .first()
    )
    date_matrix = first_rows.pivot(
        index="holding_month", columns="ticker", values="date"
    ).reindex(columns=list(assets))
    price_matrix = first_rows.pivot(
        index="holding_month", columns="ticker", values="adjusted_open"
    ).reindex(columns=list(assets))
    if date_matrix.isna().any().any() or price_matrix.isna().any().any():
        raise ValueError("every included month must contain every requested asset")
    mismatched_dates = date_matrix.nunique(axis=1).ne(1)
    if mismatched_dates.any():
        month = date_matrix.index[mismatched_dates][0]
        raise ValueError(f"assets do not share an execution date in {month:%Y-%m}")
    months = list(price_matrix.index.sort_values())
    records: list[dict[str, object]] = []
    for position in range(len(months) - 1):
        month = pd.Timestamp(months[position])
        following = pd.Timestamp(months[position + 1])
        expected = (month.to_period("M") + 1).to_timestamp()
        if following != expected:
            raise ValueError(f"missing trading month after {month:%Y-%m}")
        entry = price_matrix.loc[month].to_numpy(dtype=float)
        exit_values = price_matrix.loc[following].to_numpy(dtype=float)
        returns = exit_values / entry - 1.0
        if not np.isfinite(returns).all() or (returns <= -1.0).any():
            raise ValueError(f"invalid holding return for {month:%Y-%m}")
        row: dict[str, object] = {
            "holding_month": month,
            "entry_date": pd.Timestamp(date_matrix.loc[month].iloc[0]),
            "exit_date": pd.Timestamp(date_matrix.loc[following].iloc[0]),
            "return_available_at": pd.Timestamp(
                date_matrix.loc[following].iloc[0]
            ),
        }
        row.update(
            {
                asset: float(returns[asset_position])
                for asset_position, asset in enumerate(assets)
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def _fit_labeled_regime_returns(
    labeled_returns: pd.DataFrame,
    *,
    asset_ids: Sequence[str],
    kappa: float,
    minimum_observations: int,
    knowledge_cutoff: pd.Timestamp | None,
) -> RegimeReturnEstimate:
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
        ["holding_month", "regime_id", *assets],
        "labeled return table",
    )
    training = labeled_returns.copy()
    training["holding_month"] = _normalize_months(
        training["holding_month"], name="training holding_month"
    )
    if training["holding_month"].duplicated().any():
        raise ValueError("training holding months must be unique")
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
            f"at least {minimum_observations} common labeled months are required"
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
    for regime_id in CANONICAL_REGIME_IDS:
        mask = training["regime_id"].eq(regime_id).to_numpy()
        count = int(mask.sum())
        if count:
            regime_values = values[mask]
            sample = regime_values.mean(axis=0)
            sample_means.loc[regime_id] = sample
            residual_parts.append(regime_values - sample)
        else:
            sample = global_mean.to_numpy(dtype=float)
        if count == 0:
            # A missing state has no raw conditional mean.  The pooled mean is
            # the documented fallback even for the kappa=0 sensitivity case.
            production_means.loc[regime_id] = global_mean.to_numpy(dtype=float)
        else:
            production_means.loc[regime_id] = (
                count * sample + kappa * global_mean.to_numpy(dtype=float)
            ) / (count + kappa)
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
    means_array = production_means.to_numpy(dtype=float)
    if not np.isfinite(global_mean.to_numpy(dtype=float)).all():
        raise RuntimeError("global mean estimation produced a non-finite value")
    if not np.isfinite(means_array).all():
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
        first_holding_month=training["holding_month"].min(),
        last_holding_month=training["holding_month"].max(),
    )


def fit_regime_return_model(
    labeled_returns: pd.DataFrame,
    *,
    asset_ids: Sequence[str],
    kappa: float = 24.0,
    minimum_observations: int = 60,
) -> RegimeReturnEstimate:
    """Fit moments to an already selected common labeled return sample."""

    return _fit_labeled_regime_returns(
        labeled_returns,
        asset_ids=asset_ids,
        kappa=kappa,
        minimum_observations=minimum_observations,
        knowledge_cutoff=None,
    )


def fit_causal_regime_return_model(
    monthly_holding_returns: pd.DataFrame,
    regime_history: pd.DataFrame,
    *,
    knowledge_cutoff: str | pd.Timestamp,
    asset_ids: Sequence[str],
    kappa: float = 24.0,
    minimum_observations: int = 60,
) -> RegimeReturnEstimate:
    """Fit moments using only returns and labels available by ``knowledge_cutoff``.

    A holding month is eligible only if its next-open return and deterministic
    regime label were both available on or before the cutoff.  Callers using a
    post-month-roll signal should pass its recorded prior-day transition
    training cutoff, which excludes the just-opening month's return.
    """

    assets = tuple(map(str, asset_ids))
    _require_columns(
        monthly_holding_returns,
        ["holding_month", "return_available_at", *assets],
        "monthly holding return table",
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
    returns = monthly_holding_returns.copy()
    returns["holding_month"] = _normalize_months(
        returns["holding_month"], name="return holding_month"
    )
    returns["return_available_at"] = _normalize_dates(
        returns["return_available_at"], name="return_available_at"
    )
    if returns["holding_month"].duplicated().any():
        raise ValueError("monthly holding returns contain duplicate months")
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
    history = history.rename(columns={"reference_month": "holding_month"})
    merged = returns.merge(
        history,
        on="holding_month",
        how="left",
        validate="one_to_one",
    )
    eligible = merged.loc[
        merged["return_available_at"].le(cutoff)
        & merged["label_available_at"].le(cutoff)
        & merged["regime_id"].notna()
    ].copy()
    return _fit_labeled_regime_returns(
        eligible,
        asset_ids=assets,
        kappa=kappa,
        minimum_observations=minimum_observations,
        knowledge_cutoff=cutoff,
    )


def posterior_mixture_moments(
    posterior: Mapping[str, float] | Sequence[float] | pd.Series,
    estimate: RegimeReturnEstimate,
) -> PosteriorMixtureMoments:
    """Integrate shrunk regime means and shared risk over a posterior.

    With shared within-regime covariance ``C`` and regime means ``mu_r``, the
    total predictive covariance is

    ``C + sum_r p_r (mu_r - mu) (mu_r - mu)'``.
    """

    if estimate.regime_ids != CANONICAL_REGIME_IDS:
        raise ValueError("estimate does not use the canonical regime order")
    if isinstance(posterior, Mapping):
        supplied = set(map(str, posterior.keys()))
        if supplied != set(CANONICAL_REGIME_IDS):
            raise ValueError("posterior mapping must contain canonical regimes")
        probabilities = np.array(
            [float(posterior[regime_id]) for regime_id in CANONICAL_REGIME_IDS]
        )
    else:
        probabilities = np.asarray(posterior, dtype=float)
    probabilities = _validate_probability_vector(
        probabilities, name="regime posterior"
    )
    assets = estimate.asset_ids
    means = estimate.regime_means.loc[
        list(CANONICAL_REGIME_IDS), list(assets)
    ].to_numpy(dtype=float)
    within = estimate.shared_within_covariance.loc[
        list(assets), list(assets)
    ].to_numpy(dtype=float)
    if not np.isfinite(means).all():
        raise ValueError("regime means must be finite")
    _validate_covariance(
        within,
        dimension=len(assets),
        name="shared within-regime covariance",
    )
    expected_mean = probabilities @ means
    centered = means - expected_mean
    between = np.einsum("r,ri,rj->ij", probabilities, centered, centered)
    between = (between + between.T) / 2.0
    total = within + between
    total = (total + total.T) / 2.0
    _validate_covariance(
        between,
        dimension=len(assets),
        name="between-regime covariance",
    )
    _validate_covariance(
        total,
        dimension=len(assets),
        name="posterior mixture covariance",
    )
    index = pd.Index(assets, name="asset_id")
    return PosteriorMixtureMoments(
        posterior=pd.Series(
            probabilities,
            index=pd.Index(CANONICAL_REGIME_IDS, name="regime_id"),
            name="probability",
        ),
        expected_mean=pd.Series(expected_mean, index=index, name="expected_return"),
        within_regime_covariance=pd.DataFrame(
            within, index=index, columns=list(assets)
        ),
        between_regime_covariance=pd.DataFrame(
            between, index=index, columns=list(assets)
        ),
        covariance=pd.DataFrame(total, index=index, columns=list(assets)),
    )
