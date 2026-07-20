"""Map Model 02 composite scores into Gaussian quadrant probabilities.

The observed growth/inflation score is perturbed by zero-mean Gaussian mapping
uncertainty.  Mapping covariance is the sum of a diagonal, pooled jackknife
disagreement estimate and one fixed-horizon bivariate revision covariance.
Every historical estimate is expanding and uses only observations that were
fully available strictly before that month's score cutoff.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from regime_allocation.models.m02_soft_composite.scores import (
    GROWTH_COMPONENTS,
    INFLATION_COMPONENTS,
)


REGIME_ORDER = (
    "growth_up_inflation_up",
    "growth_down_inflation_up",
    "growth_up_inflation_down",
    "growth_down_inflation_down",
)

REGIME_LABELS = {
    "growth_up_inflation_up": "Growth composite up / inflation composite up",
    "growth_down_inflation_up": "Growth composite down / inflation composite up",
    "growth_up_inflation_down": "Growth composite up / inflation composite down",
    "growth_down_inflation_down": "Growth composite down / inflation composite down",
}


def delete_one_jackknife_variance(values: Sequence[float]) -> float:
    """Return the delete-one variance of an equal-weight component mean."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("jackknife input must be a one-dimensional sample")
    if not np.isfinite(array).all():
        raise ValueError("jackknife input must contain only finite values")
    leave_one_out = (array.sum() - array) / (array.size - 1)
    center = float(leave_one_out.mean())
    return float(
        ((array.size - 1) / array.size)
        * np.square(leave_one_out - center).sum()
    )


def component_disagreement_history(score_features: pd.DataFrame) -> pd.DataFrame:
    """Return month-specific growth and inflation jackknife variances."""

    required = {
        "score_available_at",
        *(f"{component}_z" for component in GROWTH_COMPONENTS),
        *(f"{component}_z" for component in INFLATION_COMPONENTS),
    }
    missing = required.difference(score_features.columns)
    if missing:
        raise ValueError("score features omit columns: " + ", ".join(sorted(missing)))

    records: list[dict[str, object]] = []
    for reference_month, row in score_features.sort_index().iterrows():
        growth = row[[f"{item}_z" for item in GROWTH_COMPONENTS]].to_numpy(
            dtype=float
        )
        inflation = row[[f"{item}_z" for item in INFLATION_COMPONENTS]].to_numpy(
            dtype=float
        )
        available = np.isfinite(growth).all() and np.isfinite(inflation).all()
        records.append(
            {
                "reference_month": pd.Timestamp(reference_month),
                "score_available_at": row["score_available_at"],
                "growth_jackknife_variance": (
                    delete_one_jackknife_variance(growth) if available else np.nan
                ),
                "inflation_jackknife_variance": (
                    delete_one_jackknife_variance(inflation) if available else np.nan
                ),
                "disagreement_status": "available" if available else "score_unavailable",
            }
        )
    return pd.DataFrame.from_records(records).set_index("reference_month")


def empirical_revision_covariance(errors: pd.DataFrame) -> np.ndarray:
    """Return the centered, sample covariance of bivariate revision errors."""

    required = ["growth_revision_error", "inflation_revision_error"]
    if any(column not in errors for column in required):
        raise ValueError("revision errors omit a required axis")
    values = errors[required].to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2:
        raise ValueError("at least two complete bivariate errors are required")
    if not np.isfinite(values).all():
        raise ValueError("revision errors must contain only finite values")
    covariance = np.asarray(np.cov(values, rowvar=False, ddof=1), dtype=float)
    return _validate_covariance(covariance, positive_definite=False)


def _validate_covariance(
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    positive_definite: bool,
) -> np.ndarray:
    values = np.asarray(covariance, dtype=float)
    if values.shape != (2, 2) or not np.isfinite(values).all():
        raise ValueError("mapping covariance must be a finite 2x2 matrix")
    if not np.allclose(values, values.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("mapping covariance must be symmetric")
    values = (values + values.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(values)
    scale = max(1.0, float(np.abs(eigenvalues).max()))
    if float(eigenvalues.min()) < -1.0e-12 * scale:
        raise ValueError("mapping covariance must be positive semidefinite")
    if positive_definite and (
        float(eigenvalues.min()) <= 0.0 or np.any(np.diag(values) <= 0.0)
    ):
        raise ValueError("mapping covariance must be positive definite")
    return values


def gaussian_quadrant_weights(
    score: Sequence[float],
    covariance: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return raw Gaussian quadrant masses and normalized probabilities."""

    mean = np.asarray(score, dtype=float)
    if mean.shape != (2,) or not np.isfinite(mean).all():
        raise ValueError("score must be a finite growth/inflation pair")
    omega = _validate_covariance(covariance, positive_definite=True)
    sigma_growth = math.sqrt(float(omega[0, 0]))
    sigma_inflation = math.sqrt(float(omega[1, 1]))
    correlation = float(
        omega[0, 1] / (sigma_growth * sigma_inflation)
    )
    if not -1.0 < correlation < 1.0:
        raise ValueError("mapping covariance implies a degenerate correlation")

    growth_threshold = -float(mean[0]) / sigma_growth
    inflation_threshold = -float(mean[1]) / sigma_inflation
    growth_down = float(norm.cdf(growth_threshold))
    inflation_down = float(norm.cdf(inflation_threshold))
    both_down = float(
        multivariate_normal.cdf(
            [growth_threshold, inflation_threshold],
            mean=[0.0, 0.0],
            cov=[[1.0, correlation], [correlation, 1.0]],
            maxpts=100_000,
            abseps=1.0e-10,
            releps=1.0e-10,
            rng=np.random.default_rng(0),
        )
    )
    raw = np.asarray(
        [
            1.0 - growth_down - inflation_down + both_down,
            growth_down - both_down,
            inflation_down - both_down,
            both_down,
        ],
        dtype=float,
    )
    tolerance = 1.0e-9
    if not np.isfinite(raw).all() or float(raw.min()) < -tolerance:
        raise ValueError("Gaussian quadrant integration returned invalid mass")
    raw = np.clip(raw, 0.0, 1.0)
    total = float(raw.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Gaussian quadrant masses have an invalid total")
    probabilities = raw / total
    return (
        {regime: float(value) for regime, value in zip(REGIME_ORDER, raw)},
        {
            regime: float(value)
            for regime, value in zip(REGIME_ORDER, probabilities)
        },
    )


def _empty_mapping_record(
    *,
    reference_month: pd.Timestamp,
    score_available_at: object,
    growth_score: object,
    inflation_score: object,
    horizon: int,
    baseline_horizon: int,
    status: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "reference_month": reference_month,
        "score_available_at": score_available_at,
        "revision_horizon_months": horizon,
        "specification_id": (
            f"baseline_revision_{horizon}m"
            if horizon == baseline_horizon
            else f"sensitivity_revision_{horizon}m"
        ),
        "is_baseline": horizon == baseline_horizon,
        "growth_score": growth_score,
        "inflation_score": inflation_score,
        "disagreement_training_months": 0,
        "revision_training_months": 0,
        "growth_disagreement_variance": np.nan,
        "inflation_disagreement_variance": np.nan,
        "growth_revision_mean": np.nan,
        "inflation_revision_mean": np.nan,
        "growth_revision_variance": np.nan,
        "growth_inflation_revision_covariance": np.nan,
        "inflation_revision_variance": np.nan,
        "growth_map_variance": np.nan,
        "growth_inflation_map_covariance": np.nan,
        "inflation_map_variance": np.nan,
        "map_correlation": np.nan,
        "quadrant_weight_sum": np.nan,
        "probability_sum": np.nan,
        "entropy": np.nan,
        "mapping_status": status,
    }
    for regime in REGIME_ORDER:
        record[f"weight_{regime}"] = np.nan
        record[f"probability_{regime}"] = np.nan
    return record


def causal_quadrant_mapping_history(
    score_features: pd.DataFrame,
    disagreement: pd.DataFrame,
    revision_errors: pd.DataFrame,
    *,
    horizons: Sequence[int],
    baseline_horizon: int,
    minimum_disagreement_months: int,
    minimum_revision_months: int,
) -> pd.DataFrame:
    """Estimate walk-forward mapping covariances and quadrant probabilities."""

    if baseline_horizon not in horizons:
        raise ValueError("baseline revision horizon must be one of horizons")
    if minimum_disagreement_months < 1 or minimum_revision_months < 2:
        raise ValueError("uncertainty minimum-history settings are invalid")
    required_scores = {"growth_score", "inflation_score", "score_available_at"}
    missing_scores = required_scores.difference(score_features.columns)
    if missing_scores:
        raise ValueError("score features omit required mapping inputs")
    required_revisions = {
        "reference_month",
        "horizon_months",
        "revision_available_at",
        "growth_revision_error",
        "inflation_revision_error",
        "revision_status",
    }
    if required_revisions.difference(revision_errors.columns):
        raise ValueError("revision-error table omits required columns")

    records: list[dict[str, object]] = []
    ordered_scores = score_features.sort_index()
    for reference_month, score_row in ordered_scores.iterrows():
        month = pd.Timestamp(reference_month)
        for raw_horizon in horizons:
            horizon = int(raw_horizon)
            available_at = score_row["score_available_at"]
            growth_score = score_row["growth_score"]
            inflation_score = score_row["inflation_score"]
            if (
                pd.isna(available_at)
                or pd.isna(growth_score)
                or pd.isna(inflation_score)
            ):
                records.append(
                    _empty_mapping_record(
                        reference_month=month,
                        score_available_at=available_at,
                        growth_score=growth_score,
                        inflation_score=inflation_score,
                        horizon=horizon,
                        baseline_horizon=baseline_horizon,
                        status="score_unavailable",
                    )
                )
                continue

            cutoff = pd.Timestamp(available_at)
            record = _empty_mapping_record(
                reference_month=month,
                score_available_at=cutoff,
                growth_score=float(growth_score),
                inflation_score=float(inflation_score),
                horizon=horizon,
                baseline_horizon=baseline_horizon,
                status="insufficient_uncertainty_history",
            )
            disagreement_training = disagreement[
                (disagreement.index < month)
                & (pd.to_datetime(disagreement["score_available_at"]) < cutoff)
                & (disagreement["disagreement_status"] == "available")
            ].dropna(
                subset=[
                    "growth_jackknife_variance",
                    "inflation_jackknife_variance",
                ]
            )
            revision_training = revision_errors[
                (pd.to_datetime(revision_errors["reference_month"]) < month)
                & (revision_errors["horizon_months"] == horizon)
                & (pd.to_datetime(revision_errors["revision_available_at"]) < cutoff)
                & (revision_errors["revision_status"] == "available")
            ].dropna(
                subset=["growth_revision_error", "inflation_revision_error"]
            )
            record["disagreement_training_months"] = len(disagreement_training)
            record["revision_training_months"] = len(revision_training)
            if (
                len(disagreement_training) < minimum_disagreement_months
                or len(revision_training) < minimum_revision_months
            ):
                records.append(record)
                continue

            growth_disagreement = float(
                disagreement_training["growth_jackknife_variance"].mean()
            )
            inflation_disagreement = float(
                disagreement_training["inflation_jackknife_variance"].mean()
            )
            omega_disagreement = np.diag(
                [growth_disagreement, inflation_disagreement]
            )
            omega_revision = empirical_revision_covariance(revision_training)
            omega_map = _validate_covariance(
                omega_disagreement + omega_revision,
                positive_definite=True,
            )
            revision_means = revision_training[
                ["growth_revision_error", "inflation_revision_error"]
            ].mean()
            weights, probabilities = gaussian_quadrant_weights(
                [float(growth_score), float(inflation_score)], omega_map
            )
            entropy = -sum(
                probability * math.log(probability)
                for probability in probabilities.values()
                if probability > 0.0
            )
            map_correlation = float(
                omega_map[0, 1]
                / math.sqrt(float(omega_map[0, 0] * omega_map[1, 1]))
            )
            record.update(
                {
                    "growth_disagreement_variance": growth_disagreement,
                    "inflation_disagreement_variance": inflation_disagreement,
                    "growth_revision_mean": float(revision_means.iloc[0]),
                    "inflation_revision_mean": float(revision_means.iloc[1]),
                    "growth_revision_variance": float(omega_revision[0, 0]),
                    "growth_inflation_revision_covariance": float(
                        omega_revision[0, 1]
                    ),
                    "inflation_revision_variance": float(omega_revision[1, 1]),
                    "growth_map_variance": float(omega_map[0, 0]),
                    "growth_inflation_map_covariance": float(omega_map[0, 1]),
                    "inflation_map_variance": float(omega_map[1, 1]),
                    "map_correlation": map_correlation,
                    "quadrant_weight_sum": float(sum(weights.values())),
                    "probability_sum": float(sum(probabilities.values())),
                    "entropy": float(entropy),
                    "mapping_status": "available",
                }
            )
            for regime in REGIME_ORDER:
                record[f"weight_{regime}"] = weights[regime]
                record[f"probability_{regime}"] = probabilities[regime]
            records.append(record)

    return pd.DataFrame.from_records(records).sort_values(
        ["reference_month", "revision_horizon_months"]
    )
