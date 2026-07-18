"""Estimate and propagate Model 01's first-order regime transitions.

The transition process is time-homogeneous and first-order. Estimates may be
recomputed at successive knowledge cutoffs, but every estimate uses only labels
whose point-in-time availability date is on or before its cutoff. Inputs are a
deterministic monthly regime history and Dirichlet prior; outputs include the
posterior-mean transition matrix, row-wise credible intervals, eligible-pair
audits, and propagation helpers for single-month and joint-path probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import beta as _scipy_beta
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal environments
    _scipy_beta = None

from regime_allocation.models.m01_deterministic_composite.pipeline import (
    REGIME_LABELS,
    REGIME_ORDER,
)


REGIME_IDS = tuple(regime.value for regime in REGIME_ORDER)
_REQUIRED_HISTORY_COLUMNS = {
    "reference_month",
    "regime_id",
    "label_available_at",
}


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the incomplete beta function."""

    max_iterations = 300
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) <= epsilon:
            return result
    raise RuntimeError("incomplete beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_scale = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    scale = math.exp(log_scale)
    if x < (a + 1.0) / (a + b + 2.0):
        return scale * _beta_continued_fraction(a, b, x) / a
    return 1.0 - scale * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_ppf(probability: float, a: float, b: float) -> float:
    """Return an exact numerical Beta quantile, preferring SciPy when present."""

    if _scipy_beta is not None:
        return float(_scipy_beta.ppf(probability, a, b))
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(160):
        midpoint = (lower + upper) / 2.0
        if _regularized_incomplete_beta(a, b, midpoint) < probability:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= 1.0e-14:
            break
    return (lower + upper) / 2.0


@dataclass(frozen=True)
class TransitionEstimate:
    """A causally estimated transition matrix and its audit information."""

    state_order: tuple[str, ...]
    alpha: float
    credible_interval_level: float
    knowledge_cutoff: pd.Timestamp
    counts: pd.DataFrame
    mle_probabilities: pd.DataFrame
    posterior_parameters: pd.DataFrame
    posterior_predictive: pd.DataFrame
    ci_lower: pd.DataFrame
    ci_upper: pd.DataFrame
    pairs: pd.DataFrame
    diagnostics: dict[str, int | str | None]

    def to_long_frame(self) -> pd.DataFrame:
        """Return one explicitly labeled row for every possible transition."""

        records: list[dict[str, object]] = []
        row_totals = self.counts.sum(axis=1)
        for source in self.state_order:
            for destination in self.state_order:
                source_regime = next(
                    regime for regime in REGIME_ORDER if regime.value == source
                )
                destination_regime = next(
                    regime for regime in REGIME_ORDER if regime.value == destination
                )
                records.append(
                    {
                        "from_regime_id": source,
                        "from_regime_label": REGIME_LABELS[source_regime],
                        "to_regime_id": destination,
                        "to_regime_label": REGIME_LABELS[destination_regime],
                        "transition_count": int(self.counts.loc[source, destination]),
                        "from_row_total": int(row_totals.loc[source]),
                        "mle_probability": self.mle_probabilities.loc[
                            source, destination
                        ],
                        "posterior_parameter": float(
                            self.posterior_parameters.loc[source, destination]
                        ),
                        "posterior_predictive_probability": float(
                            self.posterior_predictive.loc[source, destination]
                        ),
                        "credible_interval_level": self.credible_interval_level,
                        "credible_interval_lower": float(
                            self.ci_lower.loc[source, destination]
                        ),
                        "credible_interval_upper": float(
                            self.ci_upper.loc[source, destination]
                        ),
                    }
                )
        return pd.DataFrame.from_records(records)


def _as_normalized_timestamp(value: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("knowledge cutoff must be a valid date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED_HISTORY_COLUMNS.difference(history.columns)
    if missing:
        raise ValueError(
            f"missing required history columns: {', '.join(sorted(missing))}"
        )

    normalized = history.copy()
    raw_months = normalized["reference_month"]
    normalized["reference_month"] = pd.to_datetime(raw_months, errors="coerce")
    invalid_months = raw_months.notna() & normalized["reference_month"].isna()
    if invalid_months.any() or normalized["reference_month"].isna().any():
        raise ValueError("reference_month contains an invalid or missing date")
    if (normalized["reference_month"].dt.day != 1).any():
        raise ValueError("reference_month values must be normalized to month start")
    if normalized["reference_month"].duplicated().any():
        raise ValueError("regime history contains duplicate reference months")

    raw_availability = normalized["label_available_at"]
    normalized["label_available_at"] = pd.to_datetime(
        raw_availability, errors="coerce"
    )
    invalid_availability = (
        raw_availability.notna() & normalized["label_available_at"].isna()
    )
    if invalid_availability.any():
        raise ValueError("label_available_at contains an invalid date")

    known_regimes = set(REGIME_IDS)
    supplied_regimes = set(
        normalized.loc[normalized["regime_id"].notna(), "regime_id"].astype(str)
    )
    unknown = sorted(supplied_regimes.difference(known_regimes))
    if unknown:
        raise ValueError(f"unknown regime identifiers: {unknown}")

    classified = normalized["regime_id"].notna()
    if normalized.loc[classified, "label_available_at"].isna().any():
        raise ValueError("classified regimes require label_available_at")

    normalized["regime_id"] = normalized["regime_id"].astype("string")
    return normalized.sort_values("reference_month").reset_index(drop=True)


def _build_pair_audit(
    history: pd.DataFrame,
    *,
    knowledge_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    columns = [
        "source_reference_month",
        "destination_reference_month",
        "source_regime_id",
        "destination_regime_id",
        "source_label_available_at",
        "destination_label_available_at",
        "pair_available_at",
        "included",
        "exclusion_reason",
    ]
    records: list[dict[str, object]] = []
    for position in range(max(len(history) - 1, 0)):
        source = history.iloc[position]
        destination = history.iloc[position + 1]
        source_month = pd.Timestamp(source["reference_month"])
        destination_month = pd.Timestamp(destination["reference_month"])
        source_available = source["label_available_at"]
        destination_available = destination["label_available_at"]
        valid_dates = [
            pd.Timestamp(item)
            for item in (source_available, destination_available)
            if pd.notna(item)
        ]
        pair_available_at = max(valid_dates) if len(valid_dates) == 2 else pd.NaT

        expected_destination = (source_month.to_period("M") + 1).to_timestamp()
        if destination_month != expected_destination:
            exclusion_reason = "nonconsecutive_reference_months"
        elif pd.isna(source["regime_id"]) or pd.isna(destination["regime_id"]):
            exclusion_reason = "missing_regime"
        elif pd.isna(pair_available_at) or pair_available_at > knowledge_cutoff:
            exclusion_reason = "not_available_by_cutoff"
        else:
            exclusion_reason = ""

        records.append(
            {
                "source_reference_month": source_month,
                "destination_reference_month": destination_month,
                "source_regime_id": source["regime_id"],
                "destination_regime_id": destination["regime_id"],
                "source_label_available_at": source_available,
                "destination_label_available_at": destination_available,
                "pair_available_at": pair_available_at,
                "included": exclusion_reason == "",
                "exclusion_reason": exclusion_reason,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def estimate_transition_matrix(
    history: pd.DataFrame,
    *,
    knowledge_cutoff: date | datetime | str | pd.Timestamp,
    alpha: float = 0.5,
    credible_interval_level: float = 0.95,
) -> TransitionEstimate:
    """Estimate a causal Jeffreys-smoothed first-order transition matrix."""

    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and strictly positive")
    if (
        not math.isfinite(credible_interval_level)
        or not 0 < credible_interval_level < 1
    ):
        raise ValueError("credible_interval_level must be between zero and one")

    cutoff = _as_normalized_timestamp(knowledge_cutoff)
    normalized = _normalize_history(history)
    pairs = _build_pair_audit(normalized, knowledge_cutoff=cutoff)

    counts = pd.DataFrame(0, index=REGIME_IDS, columns=REGIME_IDS, dtype="int64")
    included = pairs.loc[pairs["included"]]
    for pair in included.itertuples(index=False):
        counts.loc[str(pair.source_regime_id), str(pair.destination_regime_id)] += 1

    row_totals = counts.sum(axis=1)
    mle = counts.div(row_totals.replace(0, np.nan), axis=0).astype(float)
    posterior_parameters = counts.astype(float) + alpha
    posterior_predictive = posterior_parameters.div(
        posterior_parameters.sum(axis=1), axis=0
    )

    tail_probability = (1.0 - credible_interval_level) / 2.0
    lower = pd.DataFrame(index=REGIME_IDS, columns=REGIME_IDS, dtype=float)
    upper = pd.DataFrame(index=REGIME_IDS, columns=REGIME_IDS, dtype=float)
    for source in REGIME_IDS:
        total_parameter = float(posterior_parameters.loc[source].sum())
        for destination in REGIME_IDS:
            cell_parameter = float(
                posterior_parameters.loc[source, destination]
            )
            other_parameter = total_parameter - cell_parameter
            lower.loc[source, destination] = _beta_ppf(
                tail_probability, cell_parameter, other_parameter
            )
            upper.loc[source, destination] = _beta_ppf(
                1.0 - tail_probability, cell_parameter, other_parameter
            )

    reason_counts = pairs["exclusion_reason"].value_counts()
    latest_destination = (
        pd.Timestamp(included["destination_reference_month"].max())
        if not included.empty
        else None
    )
    diagnostics: dict[str, int | str | None] = {
        "history_rows": len(normalized),
        "adjacent_row_pairs": len(pairs),
        "included_transitions": int(included.shape[0]),
        "excluded_nonconsecutive_pairs": int(
            reason_counts.get("nonconsecutive_reference_months", 0)
        ),
        "excluded_missing_regime_pairs": int(
            reason_counts.get("missing_regime", 0)
        ),
        "excluded_not_available_by_cutoff_pairs": int(
            reason_counts.get("not_available_by_cutoff", 0)
        ),
        "latest_destination_month": (
            latest_destination.date().isoformat()
            if latest_destination is not None
            else None
        ),
    }
    return TransitionEstimate(
        state_order=REGIME_IDS,
        alpha=float(alpha),
        credible_interval_level=float(credible_interval_level),
        knowledge_cutoff=cutoff,
        counts=counts,
        mle_probabilities=mle,
        posterior_parameters=posterior_parameters,
        posterior_predictive=posterior_predictive,
        ci_lower=lower,
        ci_upper=upper,
        pairs=pairs,
        diagnostics=diagnostics,
    )


def estimate_expanding_transition_matrices(
    history: pd.DataFrame,
    *,
    knowledge_cutoffs: Iterable[date | datetime | str | pd.Timestamp],
    alpha: float = 0.5,
    credible_interval_level: float = 0.95,
) -> tuple[TransitionEstimate, ...]:
    """Estimate the same stationary matrix at successive causal cutoffs."""

    return tuple(
        estimate_transition_matrix(
            history,
            knowledge_cutoff=cutoff,
            alpha=alpha,
            credible_interval_level=credible_interval_level,
        )
        for cutoff in knowledge_cutoffs
    )


def _as_transition_array(
    transition_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    if isinstance(transition_matrix, pd.DataFrame):
        if (
            tuple(map(str, transition_matrix.index)) != REGIME_IDS
            or tuple(map(str, transition_matrix.columns)) != REGIME_IDS
        ):
            raise ValueError("transition DataFrame does not use canonical state order")
        matrix = transition_matrix.to_numpy(dtype=float)
    else:
        matrix = np.asarray(transition_matrix, dtype=float)
    expected_shape = (len(REGIME_IDS), len(REGIME_IDS))
    if matrix.shape != expected_shape:
        raise ValueError(f"transition matrix must have shape {expected_shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("transition matrix must contain only finite values")
    if (matrix < 0).any():
        raise ValueError("transition matrix cannot contain negative probabilities")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("transition matrix rows must sum to one")
    return matrix


def propagate_regime_marginal(
    current_probabilities: np.ndarray,
    transition_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Propagate the full current-regime distribution without MAP collapse."""

    probabilities = np.asarray(current_probabilities, dtype=float)
    expected_shape = (len(REGIME_IDS),)
    if probabilities.shape != expected_shape:
        raise ValueError(f"current probabilities must have shape {expected_shape}")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("current probabilities must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("current probabilities must sum to one")
    return probabilities @ _as_transition_array(transition_matrix)


def propagate_joint_path(
    path_posterior: np.ndarray,
    transition_matrix: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Shift ``P(R[m-3:m])`` to the prior ``P(R[m-2:m+1])``."""

    posterior = np.asarray(path_posterior, dtype=float)
    states = len(REGIME_IDS)
    expected_shape = (states, states, states, states)
    if posterior.shape != expected_shape:
        raise ValueError(f"path posterior must have shape {expected_shape}")
    if not np.isfinite(posterior).all():
        raise ValueError("path posterior must contain only finite values")
    if (posterior < 0).any():
        raise ValueError("path posterior cannot contain negative probabilities")
    if not np.isclose(posterior.sum(), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("path posterior must sum to one")

    matrix = _as_transition_array(transition_matrix)
    shifted = np.einsum("abcd,de->bcde", posterior, matrix)
    if not np.isclose(shifted.sum(), 1.0, atol=1e-10, rtol=1e-10):
        raise RuntimeError("shifted path prior did not preserve probability mass")
    return shifted
