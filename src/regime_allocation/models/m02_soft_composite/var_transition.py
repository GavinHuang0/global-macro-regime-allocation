"""Estimate Model 02's causal two-score VAR(1) and reporting prior.

The estimator fits an expanding multivariate OLS regression with an intercept
to exact consecutive pairs of first-release growth and inflation scores. For a
source month ``m``, only response months no later than ``m`` whose scores were
available by that source score's cutoff may enter the fit.

The state evolved by the VAR is the released composite-score center ``Z_m``.
Once released, ``Z_m`` is exact under this model, so the one-step latent
forecast covariance is the fitted innovation covariance ``Q``. Quadrant
membership is a separate reporting map ``U_m = Z_m + epsilon_map``. Because a
future mapping covariance is not observable at the source cutoff, this
standalone stage uses the latest causally available mapping covariance as an
explicit proxy and adds it at readout. It is never propagated through the VAR
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from regime_allocation.models.m02_soft_composite.probability_map import (
    REGIME_ORDER,
    gaussian_quadrant_weights,
)


PAIR_COLUMNS = (
    "source_reference_month",
    "destination_reference_month",
    "source_score_available_at",
    "destination_score_available_at",
    "pair_available_at",
    "source_growth_score",
    "source_inflation_score",
    "destination_growth_score",
    "destination_inflation_score",
    "pair_status",
)


@dataclass(frozen=True)
class Var1Fit:
    """One plug-in VAR(1) estimate and its numerical diagnostics."""

    intercept: np.ndarray
    transition: np.ndarray
    innovation_covariance: np.ndarray
    training_pairs: int
    residual_degrees_of_freedom: int
    design_rank: int
    design_condition_number: float
    spectral_radius: float
    growth_r_squared: float
    inflation_r_squared: float
    residual_mean: np.ndarray


def _validate_score_history(score_features: pd.DataFrame) -> pd.DataFrame:
    required = {"growth_score", "inflation_score", "score_available_at"}
    missing = required.difference(score_features.columns)
    if missing:
        raise ValueError("score history omits columns: " + ", ".join(sorted(missing)))
    history = score_features.copy()
    try:
        history.index = pd.to_datetime(history.index, errors="coerce")
    except ValueError:
        raise ValueError("score reference months must be timezone-naive") from None
    if getattr(history.index, "tz", None) is not None:
        raise ValueError("score reference months must be timezone-naive")
    if history.index.isna().any() or history.index.has_duplicates:
        raise ValueError("score history requires unique valid reference months")
    if any(timestamp.day != 1 for timestamp in history.index):
        raise ValueError("score reference months must be normalized to month start")
    try:
        history["score_available_at"] = pd.to_datetime(
            history["score_available_at"], errors="coerce"
        )
    except ValueError:
        raise ValueError("score availability dates must be timezone-naive") from None
    if getattr(history["score_available_at"].dt, "tz", None) is not None:
        raise ValueError("score availability dates must be timezone-naive")
    return history.sort_index()


def build_var_pair_audit(score_features: pd.DataFrame) -> pd.DataFrame:
    """Return every adjacent score-row pair and its base eligibility status."""

    history = _validate_score_history(score_features)
    records: list[dict[str, object]] = []
    for position in range(max(len(history) - 1, 0)):
        source = history.iloc[position]
        destination = history.iloc[position + 1]
        source_month = pd.Timestamp(history.index[position])
        destination_month = pd.Timestamp(history.index[position + 1])
        expected = (source_month.to_period("M") + 1).to_timestamp()
        source_available = source["score_available_at"]
        destination_available = destination["score_available_at"]
        pair_available = (
            max(pd.Timestamp(source_available), pd.Timestamp(destination_available))
            if pd.notna(source_available) and pd.notna(destination_available)
            else pd.NaT
        )
        score_values = source[
            ["growth_score", "inflation_score"]
        ].tolist() + destination[["growth_score", "inflation_score"]].tolist()
        if destination_month != expected:
            status = "nonconsecutive_reference_months"
        elif not all(pd.notna(value) and np.isfinite(float(value)) for value in score_values):
            status = "missing_score"
        elif pd.isna(pair_available):
            status = "missing_availability_date"
        else:
            status = "eligible"
        records.append(
            {
                "source_reference_month": source_month,
                "destination_reference_month": destination_month,
                "source_score_available_at": source_available,
                "destination_score_available_at": destination_available,
                "pair_available_at": pair_available,
                "source_growth_score": source["growth_score"],
                "source_inflation_score": source["inflation_score"],
                "destination_growth_score": destination["growth_score"],
                "destination_inflation_score": destination["inflation_score"],
                "pair_status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=PAIR_COLUMNS)


def select_causal_var_pairs(
    pair_audit: pd.DataFrame,
    *,
    source_reference_month: pd.Timestamp,
    forecast_available_at: pd.Timestamp,
) -> pd.DataFrame:
    """Select pairs known by a source cutoff without admitting future months."""

    missing = set(PAIR_COLUMNS).difference(pair_audit.columns)
    if missing:
        raise ValueError("pair audit omits required columns")
    source_month = pd.Timestamp(source_reference_month)
    cutoff = pd.Timestamp(forecast_available_at)
    if source_month.tzinfo is not None or cutoff.tzinfo is not None:
        raise ValueError("VAR pair-selection cutoffs must be timezone-naive")
    selected = pair_audit[
        (pair_audit["pair_status"] == "eligible")
        & (pd.to_datetime(pair_audit["destination_reference_month"]) <= source_month)
        & (pd.to_datetime(pair_audit["pair_available_at"]) <= cutoff)
    ].copy()
    return selected.sort_values("destination_reference_month").reset_index(drop=True)


def _validated_covariance(
    covariance: np.ndarray,
    *,
    positive_definite: bool,
) -> np.ndarray:
    values = np.asarray(covariance, dtype=float)
    if values.shape != (2, 2) or not np.isfinite(values).all():
        raise ValueError("covariance must be a finite 2x2 matrix")
    if not np.allclose(values, values.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("covariance must be symmetric")
    values = (values + values.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(values)
    scale = max(1.0, float(np.abs(eigenvalues).max()))
    if float(eigenvalues.min()) < -1.0e-12 * scale:
        raise ValueError("covariance must be positive semidefinite")
    if positive_definite and float(eigenvalues.min()) <= 0.0:
        raise ValueError("covariance must be positive definite")
    return values


def fit_var1_ols(training_pairs: pd.DataFrame) -> Var1Fit:
    """Fit ``s[t] = intercept + A s[t-1] + innovation`` by multivariate OLS."""

    required = {
        "source_growth_score",
        "source_inflation_score",
        "destination_growth_score",
        "destination_inflation_score",
    }
    if required.difference(training_pairs.columns):
        raise ValueError("training pairs omit required score columns")
    source = training_pairs[
        ["source_growth_score", "source_inflation_score"]
    ].to_numpy(dtype=float)
    destination = training_pairs[
        ["destination_growth_score", "destination_inflation_score"]
    ].to_numpy(dtype=float)
    if source.shape[0] < 5 or source.shape != destination.shape:
        raise ValueError("VAR(1) OLS requires at least five bivariate pairs")
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("VAR(1) training scores must be finite")

    design = np.column_stack([np.ones(len(source)), source])
    coefficients, _, rank, _ = np.linalg.lstsq(design, destination, rcond=None)
    if int(rank) != design.shape[1]:
        raise ValueError("VAR(1) design matrix is rank deficient")
    residuals = destination - design @ coefficients
    residual_dof = len(source) - design.shape[1]
    innovation = residuals.T @ residuals / residual_dof
    innovation = _validated_covariance(innovation, positive_definite=True)
    intercept = np.asarray(coefficients[0], dtype=float)
    transition = np.asarray(coefficients[1:].T, dtype=float)
    eigenvalues = np.linalg.eigvals(transition)

    total_sums = np.square(destination - destination.mean(axis=0)).sum(axis=0)
    residual_sums = np.square(residuals).sum(axis=0)
    r_squared = np.where(
        total_sums > 0.0,
        1.0 - residual_sums / total_sums,
        np.nan,
    )
    return Var1Fit(
        intercept=intercept,
        transition=transition,
        innovation_covariance=innovation,
        training_pairs=len(source),
        residual_degrees_of_freedom=residual_dof,
        design_rank=int(rank),
        design_condition_number=float(np.linalg.cond(design)),
        spectral_radius=float(np.max(np.abs(eigenvalues))),
        growth_r_squared=float(r_squared[0]),
        inflation_r_squared=float(r_squared[1]),
        residual_mean=np.asarray(residuals.mean(axis=0), dtype=float),
    )


def propagate_var1_prior(
    exact_current_score: np.ndarray,
    target_mapping_covariance_proxy: np.ndarray,
    fit: Var1Fit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forecast an exact score center and add mapping noise only at readout.

    Returns the forecast mean, the latent score-center covariance, and the
    reporting covariance used for Gaussian quadrant integration. The source
    score has no covariance because it is observed exactly. The target mapping
    covariance is a causal proxy for the target month's as-yet-unavailable
    reporting perturbation and is assumed independent of the VAR innovation.
    """

    score = np.asarray(exact_current_score, dtype=float)
    if score.shape != (2,) or not np.isfinite(score).all():
        raise ValueError("exact current score must be a finite pair")
    mapping_proxy = _validated_covariance(
        np.asarray(target_mapping_covariance_proxy, dtype=float),
        positive_definite=False,
    )
    transition = np.asarray(fit.transition, dtype=float)
    intercept = np.asarray(fit.intercept, dtype=float)
    if transition.shape != (2, 2) or intercept.shape != (2,):
        raise ValueError("VAR(1) parameters have incompatible shapes")
    innovation = _validated_covariance(
        fit.innovation_covariance, positive_definite=True
    )
    forecast_mean = intercept + transition @ score
    latent_forecast_covariance = innovation.copy()
    reporting_forecast_covariance = innovation + mapping_proxy
    reporting_forecast_covariance = _validated_covariance(
        reporting_forecast_covariance, positive_definite=True
    )
    return (
        forecast_mean,
        latent_forecast_covariance,
        reporting_forecast_covariance,
    )


def _empty_prior_record(
    row: pd.Series,
    *,
    source_month: pd.Timestamp,
    status: str,
) -> dict[str, object]:
    target_month = (source_month.to_period("M") + 1).to_timestamp()
    target_month_end = target_month + pd.offsets.MonthEnd(1)
    raw_available_at = row.get("score_available_at", pd.NaT)
    available_at = (
        pd.Timestamp(raw_available_at) if pd.notna(raw_available_at) else pd.NaT
    )
    record: dict[str, object] = {
        "source_reference_month": source_month,
        "target_reference_month": target_month,
        "target_month_end": target_month_end,
        "prior_available_at": available_at,
        "prior_delay_from_target_start_days": (
            int((available_at - target_month).days)
            if pd.notna(available_at)
            else np.nan
        ),
        "available_by_target_start": (
            bool(available_at <= target_month) if pd.notna(available_at) else np.nan
        ),
        "available_by_target_end": (
            bool(available_at <= target_month_end)
            if pd.notna(available_at)
            else np.nan
        ),
        "prior_kind": "as_of_exact_source_score_release",
        "source_mapping_status": row.get("mapping_status", "unknown"),
        "source_revision_horizon_months": row.get(
            "revision_horizon_months", np.nan
        ),
        "source_growth_score": row.get("growth_score", np.nan),
        "source_inflation_score": row.get("inflation_score", np.nan),
        "training_pairs": 0,
        "fit_latest_destination_month": pd.NaT,
        "residual_degrees_of_freedom": np.nan,
        "design_rank": np.nan,
        "design_condition_number": np.nan,
        "spectral_radius": np.nan,
        "is_stable": np.nan,
        "growth_r_squared": np.nan,
        "inflation_r_squared": np.nan,
        "growth_residual_mean": np.nan,
        "inflation_residual_mean": np.nan,
        "growth_intercept": np.nan,
        "inflation_intercept": np.nan,
        "a_growth_from_growth": np.nan,
        "a_growth_from_inflation": np.nan,
        "a_inflation_from_growth": np.nan,
        "a_inflation_from_inflation": np.nan,
        "growth_innovation_variance": np.nan,
        "growth_inflation_innovation_covariance": np.nan,
        "inflation_innovation_variance": np.nan,
        "mapping_proxy_reference_month": pd.NaT,
        "mapping_proxy_available_at": pd.NaT,
        "growth_mapping_proxy_variance": np.nan,
        "growth_inflation_mapping_proxy_covariance": np.nan,
        "inflation_mapping_proxy_variance": np.nan,
        "growth_prior_mean": np.nan,
        "inflation_prior_mean": np.nan,
        "growth_latent_prior_variance": np.nan,
        "growth_inflation_latent_prior_covariance": np.nan,
        "inflation_latent_prior_variance": np.nan,
        "growth_prior_variance": np.nan,
        "growth_inflation_prior_covariance": np.nan,
        "inflation_prior_variance": np.nan,
        "prior_correlation": np.nan,
        "quadrant_weight_sum": np.nan,
        "probability_sum": np.nan,
        "entropy": np.nan,
        "prior_status": status,
    }
    for regime in REGIME_ORDER:
        record[f"weight_{regime}"] = np.nan
        record[f"probability_{regime}"] = np.nan
    return record


def causal_var1_prior_history(
    score_features: pd.DataFrame,
    mapping_history: pd.DataFrame,
    pair_audit: pd.DataFrame,
    *,
    minimum_training_pairs: int,
) -> pd.DataFrame:
    """Fit expanding VAR models and produce causal next-month quadrant priors."""

    if minimum_training_pairs < 5:
        raise ValueError("minimum_training_pairs must be at least five")
    scores = _validate_score_history(score_features)
    mappings = mapping_history.copy()
    required_mapping_columns = {
        "reference_month",
        "score_available_at",
        "growth_score",
        "inflation_score",
        "mapping_status",
        "revision_horizon_months",
        "growth_map_variance",
        "growth_inflation_map_covariance",
        "inflation_map_variance",
    }
    missing_mapping_columns = required_mapping_columns.difference(mappings.columns)
    if missing_mapping_columns:
        raise ValueError(
            "mapping history omits columns: "
            + ", ".join(sorted(missing_mapping_columns))
        )
    try:
        mappings["reference_month"] = pd.to_datetime(
            mappings["reference_month"], errors="coerce"
        )
        mappings["score_available_at"] = pd.to_datetime(
            mappings["score_available_at"], errors="coerce"
        )
    except ValueError:
        raise ValueError("mapping dates must be timezone-naive") from None
    if (
        getattr(mappings["reference_month"].dt, "tz", None) is not None
        or getattr(mappings["score_available_at"].dt, "tz", None) is not None
    ):
        raise ValueError("mapping dates must be timezone-naive")
    if mappings["reference_month"].isna().any():
        raise ValueError("mapping history contains an invalid reference month")
    if mappings["reference_month"].duplicated().any():
        raise ValueError("mapping history contains duplicate reference months")
    mappings = mappings.set_index("reference_month").sort_index()
    if not mappings.index.equals(scores.index):
        raise ValueError("mapping and score histories must use identical months")
    for column in ("growth_score", "inflation_score"):
        mapping_values = mappings[column].to_numpy(dtype=float)
        score_values = scores[column].to_numpy(dtype=float)
        if not np.array_equal(mapping_values, score_values, equal_nan=True):
            raise ValueError(
                f"mapping and score histories disagree on {column}"
            )
    if not mappings["score_available_at"].equals(scores["score_available_at"]):
        raise ValueError(
            "mapping and score histories disagree on score_available_at"
        )

    records: list[dict[str, object]] = []
    for source_month, row in mappings.iterrows():
        month = pd.Timestamp(source_month)
        if str(row.get("mapping_status")) != "available":
            records.append(
                _empty_prior_record(row, source_month=month, status="source_map_unavailable")
            )
            continue
        cutoff = row.get("score_available_at")
        if pd.isna(cutoff):
            records.append(
                _empty_prior_record(row, source_month=month, status="source_map_unavailable")
            )
            continue
        training = select_causal_var_pairs(
            pair_audit,
            source_reference_month=month,
            forecast_available_at=pd.Timestamp(cutoff),
        )
        record = _empty_prior_record(
            row, source_month=month, status="insufficient_transition_history"
        )
        record["training_pairs"] = len(training)
        if not training.empty:
            record["fit_latest_destination_month"] = training[
                "destination_reference_month"
            ].max()
        if len(training) < minimum_training_pairs:
            records.append(record)
            continue

        fit = fit_var1_ols(training)
        source_mean = np.asarray(
            [row["growth_score"], row["inflation_score"]], dtype=float
        )
        causal_mapping_rows = mappings[
            (mappings["mapping_status"] == "available")
            & (mappings["score_available_at"] <= pd.Timestamp(cutoff))
            & (mappings.index <= month)
        ].sort_values(["score_available_at"], kind="stable")
        if causal_mapping_rows.empty:
            record["prior_status"] = "mapping_proxy_unavailable"
            records.append(record)
            continue
        mapping_proxy_month = pd.Timestamp(causal_mapping_rows.index[-1])
        mapping_proxy_row = causal_mapping_rows.iloc[-1]
        mapping_proxy_covariance = np.asarray(
            [
                [
                    mapping_proxy_row["growth_map_variance"],
                    mapping_proxy_row["growth_inflation_map_covariance"],
                ],
                [
                    mapping_proxy_row["growth_inflation_map_covariance"],
                    mapping_proxy_row["inflation_map_variance"],
                ],
            ],
            dtype=float,
        )
        prior_mean, latent_prior_covariance, prior_covariance = propagate_var1_prior(
            source_mean, mapping_proxy_covariance, fit
        )
        weights, probabilities = gaussian_quadrant_weights(
            prior_mean, prior_covariance
        )
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities.values()
            if probability > 0.0
        )
        prior_correlation = float(
            prior_covariance[0, 1]
            / math.sqrt(prior_covariance[0, 0] * prior_covariance[1, 1])
        )
        record.update(
            {
                "residual_degrees_of_freedom": fit.residual_degrees_of_freedom,
                "design_rank": fit.design_rank,
                "design_condition_number": fit.design_condition_number,
                "spectral_radius": fit.spectral_radius,
                "is_stable": fit.spectral_radius < 1.0,
                "growth_r_squared": fit.growth_r_squared,
                "inflation_r_squared": fit.inflation_r_squared,
                "growth_residual_mean": float(fit.residual_mean[0]),
                "inflation_residual_mean": float(fit.residual_mean[1]),
                "growth_intercept": float(fit.intercept[0]),
                "inflation_intercept": float(fit.intercept[1]),
                "a_growth_from_growth": float(fit.transition[0, 0]),
                "a_growth_from_inflation": float(fit.transition[0, 1]),
                "a_inflation_from_growth": float(fit.transition[1, 0]),
                "a_inflation_from_inflation": float(fit.transition[1, 1]),
                "growth_innovation_variance": float(
                    fit.innovation_covariance[0, 0]
                ),
                "growth_inflation_innovation_covariance": float(
                    fit.innovation_covariance[0, 1]
                ),
                "inflation_innovation_variance": float(
                    fit.innovation_covariance[1, 1]
                ),
                "mapping_proxy_reference_month": mapping_proxy_month,
                "mapping_proxy_available_at": pd.Timestamp(
                    mapping_proxy_row["score_available_at"]
                ),
                "growth_mapping_proxy_variance": float(
                    mapping_proxy_covariance[0, 0]
                ),
                "growth_inflation_mapping_proxy_covariance": float(
                    mapping_proxy_covariance[0, 1]
                ),
                "inflation_mapping_proxy_variance": float(
                    mapping_proxy_covariance[1, 1]
                ),
                "growth_prior_mean": float(prior_mean[0]),
                "inflation_prior_mean": float(prior_mean[1]),
                "growth_latent_prior_variance": float(
                    latent_prior_covariance[0, 0]
                ),
                "growth_inflation_latent_prior_covariance": float(
                    latent_prior_covariance[0, 1]
                ),
                "inflation_latent_prior_variance": float(
                    latent_prior_covariance[1, 1]
                ),
                "growth_prior_variance": float(prior_covariance[0, 0]),
                "growth_inflation_prior_covariance": float(
                    prior_covariance[0, 1]
                ),
                "inflation_prior_variance": float(prior_covariance[1, 1]),
                "prior_correlation": prior_correlation,
                "quadrant_weight_sum": float(sum(weights.values())),
                "probability_sum": float(sum(probabilities.values())),
                "entropy": float(entropy),
                "prior_status": "available",
            }
        )
        for regime in REGIME_ORDER:
            record[f"weight_{regime}"] = weights[regime]
            record[f"probability_{regime}"] = probabilities[regime]
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values("source_reference_month")
