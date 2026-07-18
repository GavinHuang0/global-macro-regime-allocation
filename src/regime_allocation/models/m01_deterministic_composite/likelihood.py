"""Causal release-block likelihoods for Model 01.

The module estimates one Student-t (or Gaussian sensitivity) emission per
regime. Means differ by regime and shrink toward the block-wide mean, while a
single residual covariance is shared by every regime in a block.

No filtering or posterior update lives here. The causal-selection helper makes
the information-set boundary explicit so that a later filter cannot train on a
release whose deterministic target label was not yet available. Inputs are
complete vectors from one release block and their causal labels; outputs are
regime means, a shared positive-definite scale matrix, log densities, and a
fully serializable fit audit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.covariance import LedoitWolf

from regime_allocation.models.m01_deterministic_composite.pipeline import (
    REGIME_ORDER,
)


CANONICAL_REGIME_IDS = tuple(regime.value for regime in REGIME_ORDER)
CAUSAL_VECTOR_COLUMNS = (
    "event_id",
    "release_date",
    "reference_month",
    "label_available_at",
    "regime_id",
)
SUPPORTED_COVARIANCE_METHODS = (
    "ledoit_wolf",
    "empirical",
    "fixed_spherical",
)


def _normalized_timestamp(
    value: date | datetime | str | pd.Timestamp,
    *,
    name: str,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _validated_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(feature_names, (str, bytes)):
        raise TypeError("feature_names must be a sequence of column names")
    names = tuple(str(name).strip() for name in feature_names)
    if not names or any(not name for name in names):
        raise ValueError("feature_names must contain at least one non-empty name")
    if len(names) != len(set(names)):
        raise ValueError("feature_names cannot contain duplicates")
    return names


def _parse_date_column(
    frame: pd.DataFrame,
    column: str,
    *,
    allow_missing: bool,
) -> pd.Series:
    raw = frame[column]
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    invalid = raw.notna() & parsed.isna()
    if invalid.any():
        raise ValueError(f"{column} contains an invalid date")
    if not allow_missing and parsed.isna().any():
        raise ValueError(f"{column} cannot contain missing dates")
    return parsed.dt.tz_convert(None).dt.normalize()


def _as_numeric_features(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    try:
        numeric = frame.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("feature vectors must contain only numeric values") from error
    return numeric.astype(float)


@dataclass(frozen=True)
class CausalTrainingSelection:
    """Eligible vectors and the row-level reasons for every exclusion."""

    knowledge_cutoff: pd.Timestamp
    feature_names: tuple[str, ...]
    eligible: pd.DataFrame
    audit: pd.DataFrame
    diagnostics: dict[str, int]

    def to_audit_dict(self) -> dict[str, Any]:
        """Return credential-free, JSON-serializable selection metadata."""

        return {
            "knowledge_cutoff": self.knowledge_cutoff.date().isoformat(),
            "strict_cutoff_rule": "training_available_at < knowledge_cutoff",
            "feature_names": list(self.feature_names),
            "diagnostics": {key: int(value) for key, value in self.diagnostics.items()},
        }


def select_causal_training_vectors(
    vectors: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    knowledge_cutoff: date | datetime | str | pd.Timestamp,
) -> CausalTrainingSelection:
    """Select complete labeled vectors knowable strictly before a cutoff.

    A vector becomes training-ready only when both its release and its target
    regime label are available. Equality with ``knowledge_cutoff`` is excluded
    deliberately because the repository has a daily, not intraday, timestamp
    contract.
    """

    if not isinstance(vectors, pd.DataFrame):
        raise TypeError("vectors must be a pandas DataFrame")
    names = _validated_feature_names(feature_names)
    required = set(CAUSAL_VECTOR_COLUMNS).union(names)
    missing = required.difference(vectors.columns)
    if missing:
        raise ValueError(
            f"training vectors are missing columns: {', '.join(sorted(missing))}"
        )
    cutoff = _normalized_timestamp(knowledge_cutoff, name="knowledge_cutoff")
    audit = vectors.copy()
    if audit["event_id"].isna().any():
        raise ValueError("event_id cannot be missing")
    if audit["event_id"].astype(str).duplicated().any():
        raise ValueError("training vectors contain duplicate event_id values")

    audit["release_date"] = _parse_date_column(
        audit, "release_date", allow_missing=False
    )
    audit["reference_month"] = _parse_date_column(
        audit, "reference_month", allow_missing=False
    )
    if (audit["reference_month"].dt.day != 1).any():
        raise ValueError("reference_month values must be normalized to month start")
    audit["label_available_at"] = _parse_date_column(
        audit, "label_available_at", allow_missing=True
    )

    supplied_regimes = set(
        audit.loc[audit["regime_id"].notna(), "regime_id"].astype(str)
    )
    unknown = sorted(supplied_regimes.difference(CANONICAL_REGIME_IDS))
    if unknown:
        raise ValueError(f"unknown regime identifiers: {unknown}")
    audit["regime_id"] = audit["regime_id"].astype("string")
    numeric = _as_numeric_features(audit, names)
    audit.loc[:, names] = numeric

    complete = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    has_regime = audit["regime_id"].notna().to_numpy()
    has_label_date = audit["label_available_at"].notna().to_numpy()
    training_available_at = pd.concat(
        [audit["release_date"], audit["label_available_at"]], axis=1
    ).max(axis=1)
    training_available_at = training_available_at.where(
        audit["label_available_at"].notna()
    )

    reasons = np.full(len(audit), "", dtype=object)
    reasons[~has_regime] = "missing_regime"
    reasons[has_regime & ~has_label_date] = "missing_label_availability"
    reasons[has_regime & has_label_date & ~complete] = "incomplete_feature_vector"
    otherwise_ready = has_regime & has_label_date & complete
    not_before = otherwise_ready & (
        training_available_at.to_numpy(dtype="datetime64[ns]")
        >= cutoff.to_datetime64()
    )
    reasons[not_before] = "not_available_strictly_before_cutoff"
    eligible_mask = reasons == ""

    audit["training_available_at"] = training_available_at
    audit["training_eligible"] = eligible_mask
    audit["training_exclusion_reason"] = reasons
    audit = audit.sort_values(
        ["release_date", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    eligible = audit.loc[audit["training_eligible"]].copy().reset_index(drop=True)
    reason_counts = audit["training_exclusion_reason"].value_counts()
    diagnostics = {
        "input_vectors": int(len(audit)),
        "eligible_vectors": int(len(eligible)),
        "excluded_missing_regime": int(reason_counts.get("missing_regime", 0)),
        "excluded_missing_label_availability": int(
            reason_counts.get("missing_label_availability", 0)
        ),
        "excluded_incomplete_feature_vector": int(
            reason_counts.get("incomplete_feature_vector", 0)
        ),
        "excluded_not_available_strictly_before_cutoff": int(
            reason_counts.get("not_available_strictly_before_cutoff", 0)
        ),
    }
    return CausalTrainingSelection(
        knowledge_cutoff=cutoff,
        feature_names=names,
        eligible=eligible,
        audit=audit,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class BlockLikelihoodFit:
    """Fitted regime-conditional likelihood for one complete release block."""

    block_id: str
    feature_names: tuple[str, ...]
    state_order: tuple[str, ...]
    training_count: int
    regime_counts: pd.Series
    pooled_mean: pd.Series
    regime_means: pd.DataFrame
    covariance: pd.DataFrame
    distribution_scale: pd.DataFrame
    kappa: float
    degrees_of_freedom: float | None
    scale_multiplier: float
    covariance_method: str
    applied_shrinkage: float
    learned_shrinkage: float | None
    knowledge_cutoff: pd.Timestamp | None
    first_training_available_at: pd.Timestamp | None
    last_training_available_at: pd.Timestamp | None
    training_diagnostics: dict[str, int]

    @property
    def distribution(self) -> str:
        """Return the fitted density family name used in audit artifacts."""
        return "gaussian" if self.degrees_of_freedom is None else "student_t"

    def log_likelihoods(
        self,
        observation: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray,
    ) -> pd.Series:
        """Return one log density in canonical regime order."""

        values = _observation_array(observation, self.feature_names)
        scale = self.distribution_scale.to_numpy(dtype=float)
        sign, log_determinant = np.linalg.slogdet(scale)
        if sign <= 0 or not math.isfinite(float(log_determinant)):
            raise RuntimeError("stored distribution scale is not positive definite")
        dimension = len(self.feature_names)
        results: list[float] = []
        for regime_id in self.state_order:
            difference = values - self.regime_means.loc[regime_id].to_numpy(dtype=float)
            mahalanobis = float(difference @ np.linalg.solve(scale, difference))
            if self.degrees_of_freedom is None:
                log_density = -0.5 * (
                    dimension * math.log(2.0 * math.pi)
                    + float(log_determinant)
                    + mahalanobis
                )
            else:
                nu = self.degrees_of_freedom
                log_density = (
                    float(gammaln((nu + dimension) / 2.0))
                    - float(gammaln(nu / 2.0))
                    - 0.5
                    * (
                        dimension * math.log(nu * math.pi)
                        + float(log_determinant)
                    )
                    - 0.5
                    * (nu + dimension)
                    * math.log1p(mahalanobis / nu)
                )
            results.append(log_density)
        return pd.Series(results, index=self.state_order, name="log_likelihood")

    def to_audit_dict(self) -> dict[str, Any]:
        """Return a complete JSON-serializable fit record."""

        def timestamp_text(value: pd.Timestamp | None) -> str | None:
            """Render an optional normalized timestamp as an ISO date."""
            return value.date().isoformat() if value is not None else None

        means = {
            regime_id: {
                feature: float(self.regime_means.loc[regime_id, feature])
                for feature in self.feature_names
            }
            for regime_id in self.state_order
        }
        covariance = {
            row: {
                column: float(self.covariance.loc[row, column])
                for column in self.feature_names
            }
            for row in self.feature_names
        }
        distribution_scale = {
            row: {
                column: float(self.distribution_scale.loc[row, column])
                for column in self.feature_names
            }
            for row in self.feature_names
        }
        return {
            "block_id": self.block_id,
            "feature_names": list(self.feature_names),
            "state_order": list(self.state_order),
            "training_count": int(self.training_count),
            "regime_counts": {
                regime_id: int(self.regime_counts.loc[regime_id])
                for regime_id in self.state_order
            },
            "pooled_mean": {
                feature: float(self.pooled_mean.loc[feature])
                for feature in self.feature_names
            },
            "regime_means": means,
            "covariance": covariance,
            "distribution_scale": distribution_scale,
            "distribution": self.distribution,
            "degrees_of_freedom": self.degrees_of_freedom,
            "scale_standard_deviation_multiplier": float(self.scale_multiplier),
            "student_t_scale_factor": (
                None
                if self.degrees_of_freedom is None
                else (self.degrees_of_freedom - 2.0) / self.degrees_of_freedom
            ),
            "kappa": float(self.kappa),
            "covariance_method": self.covariance_method,
            "applied_shrinkage": float(self.applied_shrinkage),
            "learned_shrinkage": self.learned_shrinkage,
            "knowledge_cutoff": timestamp_text(self.knowledge_cutoff),
            "first_training_available_at": timestamp_text(
                self.first_training_available_at
            ),
            "last_training_available_at": timestamp_text(
                self.last_training_available_at
            ),
            "covariance_minimum_eigenvalue": float(
                np.linalg.eigvalsh(self.covariance.to_numpy(dtype=float)).min()
            ),
            "training_diagnostics": {
                key: int(value) for key, value in self.training_diagnostics.items()
            },
        }


def _observation_array(
    observation: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    if isinstance(observation, pd.Series):
        missing = set(feature_names).difference(observation.index.astype(str))
        if missing:
            raise ValueError(f"observation is missing features: {sorted(missing)}")
        values = observation.loc[list(feature_names)].to_numpy(dtype=float)
    elif isinstance(observation, Mapping):
        missing = set(feature_names).difference(observation)
        if missing:
            raise ValueError(f"observation is missing features: {sorted(missing)}")
        values = np.asarray([observation[name] for name in feature_names], dtype=float)
    else:
        values = np.asarray(observation, dtype=float)
    if values.shape != (len(feature_names),):
        raise ValueError(
            f"observation must have shape ({len(feature_names)},), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("observation must contain only finite values")
    return values


def _validated_degrees_of_freedom(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("degrees_of_freedom must be numeric or None")
    result = float(value)
    if not math.isfinite(result) or result <= 2.0:
        raise ValueError("finite Student-t degrees_of_freedom must exceed 2")
    return result


def _positive_definite_matrix(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains a nonfinite value")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    largest = float(eigenvalues[-1])
    numerical_floor = np.finfo(float).eps * max(len(matrix), 1) * largest
    if largest <= 0.0 or float(eigenvalues[0]) <= numerical_floor:
        raise ValueError(f"{name} must be positive definite")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return matrix


def fit_block_likelihood(
    vectors: pd.DataFrame,
    *,
    block_id: str,
    feature_names: Sequence[str],
    kappa: float = 5.0,
    degrees_of_freedom: float | None = 7.0,
    scale_multiplier: float = 1.0,
    covariance_method: str = "ledoit_wolf",
    fixed_shrinkage: float | None = None,
    knowledge_cutoff: date | datetime | str | pd.Timestamp | None = None,
    training_diagnostics: Mapping[str, int] | None = None,
) -> BlockLikelihoodFit:
    """Fit shrunken regime means and one shared block covariance.

    ``vectors`` must already contain complete, causally eligible rows. Use
    :func:`select_causal_training_vectors` or
    :func:`fit_causal_block_likelihood` when operating on an unfiltered table.
    """

    if not isinstance(vectors, pd.DataFrame):
        raise TypeError("vectors must be a pandas DataFrame")
    normalized_block_id = str(block_id).strip()
    if not normalized_block_id:
        raise ValueError("block_id cannot be empty")
    names = _validated_feature_names(feature_names)
    missing = {"regime_id", *names}.difference(vectors.columns)
    if missing:
        raise ValueError(f"vectors are missing columns: {', '.join(sorted(missing))}")
    if len(vectors) < 2:
        raise ValueError("at least two complete training vectors are required")
    if isinstance(kappa, bool):
        raise TypeError("kappa must be numeric")
    kappa_value = float(kappa)
    if not math.isfinite(kappa_value) or kappa_value < 0.0:
        raise ValueError("kappa must be finite and non-negative")
    nu = _validated_degrees_of_freedom(degrees_of_freedom)
    if isinstance(scale_multiplier, bool):
        raise TypeError("scale_multiplier must be numeric")
    scale_multiplier_value = float(scale_multiplier)
    if not math.isfinite(scale_multiplier_value) or scale_multiplier_value <= 0.0:
        raise ValueError("scale_multiplier must be finite and strictly positive")
    if covariance_method not in SUPPORTED_COVARIANCE_METHODS:
        raise ValueError(
            f"covariance_method must be one of {SUPPORTED_COVARIANCE_METHODS}"
        )
    if covariance_method == "fixed_spherical":
        if fixed_shrinkage is None or isinstance(fixed_shrinkage, bool):
            raise ValueError(
                "fixed_spherical covariance requires numeric fixed_shrinkage"
            )
        shrinkage_value = float(fixed_shrinkage)
        if not math.isfinite(shrinkage_value) or not 0.0 <= shrinkage_value <= 1.0:
            raise ValueError("fixed_shrinkage must be between zero and one")
    elif fixed_shrinkage is not None:
        raise ValueError(
            "fixed_shrinkage is valid only with fixed_spherical covariance"
        )
    else:
        shrinkage_value = 0.0

    regime_values = vectors["regime_id"]
    if regime_values.isna().any():
        raise ValueError("fit vectors cannot contain a missing regime_id")
    supplied_regimes = set(regime_values.astype(str))
    unknown = sorted(supplied_regimes.difference(CANONICAL_REGIME_IDS))
    if unknown:
        raise ValueError(f"unknown regime identifiers: {unknown}")
    numeric = _as_numeric_features(vectors, names)
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("fit vectors must be complete and finite")

    cutoff = (
        _normalized_timestamp(knowledge_cutoff, name="knowledge_cutoff")
        if knowledge_cutoff is not None
        else None
    )
    first_available: pd.Timestamp | None = None
    last_available: pd.Timestamp | None = None
    if "training_available_at" in vectors.columns:
        available = pd.to_datetime(
            vectors["training_available_at"], errors="coerce", utc=True
        )
        if available.isna().any():
            raise ValueError("training_available_at cannot be missing or invalid")
        available = available.dt.tz_convert(None).dt.normalize()
        if cutoff is not None and (available >= cutoff).any():
            raise ValueError(
                "training vectors must be available strictly before knowledge_cutoff"
            )
        first_available = pd.Timestamp(available.min())
        last_available = pd.Timestamp(available.max())

    regimes = regime_values.astype(str).to_numpy()
    pooled_values = values.mean(axis=0)
    counts = pd.Series(0, index=CANONICAL_REGIME_IDS, dtype="int64")
    means = pd.DataFrame(
        index=CANONICAL_REGIME_IDS, columns=names, dtype=float
    )
    for regime_id in CANONICAL_REGIME_IDS:
        mask = regimes == regime_id
        count = int(mask.sum())
        counts.loc[regime_id] = count
        if count == 0:
            if kappa_value == 0.0:
                raise ValueError(
                    "kappa=0 cannot estimate a regime mean when a canonical "
                    f"regime has no training observations: {regime_id}"
                )
            shrunken = pooled_values.copy()
        else:
            sample_mean = values[mask].mean(axis=0)
            if kappa_value == 0.0:
                shrunken = sample_mean
            else:
                shrunken = (
                    count * sample_mean + kappa_value * pooled_values
                ) / (count + kappa_value)
        means.loc[regime_id] = shrunken

    residuals = np.vstack(
        [
            values[row] - means.loc[regimes[row]].to_numpy(dtype=float)
            for row in range(len(values))
        ]
    )
    empirical = residuals.T @ residuals / len(residuals)
    if covariance_method == "ledoit_wolf":
        estimator = LedoitWolf(assume_centered=True, store_precision=False).fit(
            residuals
        )
        covariance_values = estimator.covariance_
        learned_shrinkage: float | None = float(estimator.shrinkage_)
        applied_shrinkage = learned_shrinkage
    elif covariance_method == "empirical":
        covariance_values = empirical
        learned_shrinkage = None
        applied_shrinkage = 0.0
    else:
        target_variance = float(np.trace(empirical) / len(names))
        spherical_target = target_variance * np.eye(len(names))
        covariance_values = (
            (1.0 - shrinkage_value) * empirical
            + shrinkage_value * spherical_target
        )
        learned_shrinkage = None
        applied_shrinkage = shrinkage_value
    covariance_values = _positive_definite_matrix(
        covariance_values, name="shared residual covariance"
    )
    covariance_values = _positive_definite_matrix(
        covariance_values * scale_multiplier_value**2,
        name="scaled shared residual covariance",
    )

    scale_factor = 1.0 if nu is None else (nu - 2.0) / nu
    scale_values = _positive_definite_matrix(
        scale_factor * covariance_values,
        name="distribution scale",
    )
    diagnostics = {
        str(key): int(value) for key, value in (training_diagnostics or {}).items()
    }
    feature_index = pd.Index(names, name="feature_name")
    covariance = pd.DataFrame(
        covariance_values, index=feature_index, columns=feature_index
    )
    scale = pd.DataFrame(scale_values, index=feature_index, columns=feature_index)
    return BlockLikelihoodFit(
        block_id=normalized_block_id,
        feature_names=names,
        state_order=CANONICAL_REGIME_IDS,
        training_count=int(len(vectors)),
        regime_counts=counts,
        pooled_mean=pd.Series(pooled_values, index=feature_index, name="pooled_mean"),
        regime_means=means,
        covariance=covariance,
        distribution_scale=scale,
        kappa=kappa_value,
        degrees_of_freedom=nu,
        scale_multiplier=scale_multiplier_value,
        covariance_method=covariance_method,
        applied_shrinkage=float(applied_shrinkage),
        learned_shrinkage=learned_shrinkage,
        knowledge_cutoff=cutoff,
        first_training_available_at=first_available,
        last_training_available_at=last_available,
        training_diagnostics=diagnostics,
    )


def fit_causal_block_likelihood(
    vectors: pd.DataFrame,
    *,
    block_id: str,
    feature_names: Sequence[str],
    knowledge_cutoff: date | datetime | str | pd.Timestamp,
    kappa: float = 5.0,
    degrees_of_freedom: float | None = 7.0,
    scale_multiplier: float = 1.0,
    covariance_method: str = "ledoit_wolf",
    fixed_shrinkage: float | None = None,
) -> tuple[BlockLikelihoodFit, CausalTrainingSelection]:
    """Select causal training rows and fit one block likelihood."""

    selection = select_causal_training_vectors(
        vectors,
        feature_names=feature_names,
        knowledge_cutoff=knowledge_cutoff,
    )
    fit = fit_block_likelihood(
        selection.eligible,
        block_id=block_id,
        feature_names=selection.feature_names,
        kappa=kappa,
        degrees_of_freedom=degrees_of_freedom,
        scale_multiplier=scale_multiplier,
        covariance_method=covariance_method,
        fixed_shrinkage=fixed_shrinkage,
        knowledge_cutoff=selection.knowledge_cutoff,
        training_diagnostics=selection.diagnostics,
    )
    return fit, selection
