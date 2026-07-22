"""Process partial releases of Model 02's score-defining components.

Model 02 defines each monthly score as the fixed equal-weight sum of four
causally standardized first-release components.  This module exposes those
components as release events and converts them into coherent likelihoods for
the existing rolling joint Gaussian score filter.

The component model is deliberately separate from the ordinary evidence-block
model.  Let ``c_m`` contain the eight defining component z-scores and let

``Z_m = W @ c_m``

be the two exact growth/inflation composites, where every nonzero element of
``W`` is one quarter.  A causal Ledoit-Wolf Gaussian estimate for ``c_m``
implies the sequential observation model

``p(c_new | Z_m, c_already_observed)``.

Conditioning every release on components previously processed gives a
factorization of ``p(c_observed | Z_m)``.  A component is therefore used once,
not repeatedly through a succession of unconditional likelihoods.  If a
release completes all four components of an axis, that axis score is imposed
as the exact weighted sum instead of approximating the deterministic identity
with a small measurement variance.  When both axes are complete, the result is
compatible with :func:`condition_on_exact_score` and its later call is
idempotent.

All estimation rows must have a complete component vector whose final score
availability is strictly earlier than the event being scored.  The module has
no file or network access; callers supply the two frozen Model 02 artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
import math

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from regime_allocation.models.m02_soft_composite.joint_filter import (
    JointGaussianState,
    LinearGaussianUpdate,
    update_joint_gaussian,
)
from regime_allocation.models.m02_soft_composite.scores import (
    ALL_COMPONENTS,
    GROWTH_COMPONENTS,
    INFLATION_COMPONENTS,
)


COMPONENT_WEIGHT = 0.25
AXIS_NAMES = ("growth", "inflation")
COMPONENT_AXIS = {
    **{component: "growth" for component in GROWTH_COMPONENTS},
    **{component: "inflation" for component in INFLATION_COMPONENTS},
}
DEFINING_RELEASE_FAMILY = {
    "payrolls": "employment_situation",
    "unemployment_rate": "employment_situation",
    "average_hourly_earnings": "employment_situation",
    "industrial_production": "industrial_production",
    "consumer_activity": "personal_income_and_outlays",
    "core_pce": "personal_income_and_outlays",
    "core_cpi": "consumer_price_index",
    "producer_prices": "producer_price_index",
}


def _timestamp(value: object, *, label: str, normalize: bool = False) -> pd.Timestamp:
    """Return one finite, timezone-naive timestamp."""

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{label} must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize() if normalize else timestamp


def _month(value: object, *, label: str = "reference_month") -> pd.Timestamp:
    """Normalize one timestamp to its timezone-naive calendar month."""

    return _timestamp(value, label=label).to_period("M").to_timestamp()


def _component_names(
    values: Sequence[str],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Validate and canonically order a component-name collection."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of component names")
    names = tuple(str(value) for value in values)
    if not names and not allow_empty:
        raise ValueError(f"{label} cannot be empty")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate components")
    unknown = set(names).difference(ALL_COMPONENTS)
    if unknown:
        raise ValueError(f"{label} contains unknown components: {sorted(unknown)}")
    return tuple(component for component in ALL_COMPONENTS if component in names)


def _finite_mapping(values: Mapping[str, float], *, label: str) -> dict[str, float]:
    """Return a finite component-value mapping in canonical order."""

    names = _component_names(tuple(values), label=label, allow_empty=True)
    output: dict[str, float] = {}
    for component in names:
        value = float(values[component])
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite value")
        output[component] = value
    return output


def score_weight_matrix(
    component_order: Sequence[str] = ALL_COMPONENTS,
) -> np.ndarray:
    """Return the exact two-by-eight equal-weight score map ``W``."""

    order = _component_names(component_order, label="component_order")
    if set(order) != set(ALL_COMPONENTS):
        raise ValueError("component_order must contain every defining component")
    weights = np.zeros((2, len(order)), dtype=float)
    for column, component in enumerate(order):
        axis = AXIS_NAMES.index(COMPONENT_AXIS[component])
        weights[axis, column] = COMPONENT_WEIGHT
    return weights


@dataclass(frozen=True)
class PreparedPartialDefiningData:
    """Causal defining-component events and their complete training history."""

    events: pd.DataFrame
    component_history: pd.DataFrame
    preparation_audit: pd.DataFrame


@dataclass(frozen=True)
class CausalComponentGaussianFit:
    """Ledoit-Wolf Gaussian estimate of the eight defining components."""

    component_order: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    shrinkage: float
    sample_size: int
    knowledge_cutoff: pd.Timestamp
    first_reference_month: pd.Timestamp
    last_reference_month: pd.Timestamp
    latest_training_availability: pd.Timestamp

    def __post_init__(self) -> None:
        order = _component_names(self.component_order, label="component_order")
        if set(order) != set(ALL_COMPONENTS):
            raise ValueError("component fit must contain all defining components")
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        dimension = len(order)
        if mean.shape != (dimension,) or not np.isfinite(mean).all():
            raise ValueError("component mean must be a finite vector")
        if covariance.shape != (dimension, dimension):
            raise ValueError("component covariance has the wrong shape")
        if not np.isfinite(covariance).all() or not np.allclose(
            covariance, covariance.T, atol=1.0e-11, rtol=1.0e-11
        ):
            raise ValueError("component covariance must be finite and symmetric")
        if np.linalg.eigvalsh(covariance).min() <= 0.0:
            raise ValueError("component covariance must be positive definite")
        if self.sample_size < 2:
            raise ValueError("component Gaussian fit requires at least two rows")
        if not 0.0 <= float(self.shrinkage) <= 1.0:
            raise ValueError("Ledoit-Wolf shrinkage must be between zero and one")
        mean = mean.copy()
        covariance = ((covariance + covariance.T) / 2.0).copy()
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "component_order", order)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(
            self,
            "knowledge_cutoff",
            _timestamp(self.knowledge_cutoff, label="knowledge_cutoff"),
        )


@dataclass(frozen=True)
class ConditionalComponentEmission:
    """Conditional Gaussian equation for one new defining-release vector.

    The live equation is

    ``new = intercept + score_loadings @ Z +``
    ``conditioned_loadings @ already_observed + error``.
    """

    response_components: tuple[str, ...]
    conditioned_components: tuple[str, ...]
    intercept: np.ndarray
    score_loadings: np.ndarray
    conditioned_loadings: np.ndarray
    noise_covariance: np.ndarray

    def offset(self, conditioned_values: Mapping[str, float]) -> np.ndarray:
        """Evaluate the intercept after inserting known component values."""

        values = _finite_mapping(conditioned_values, label="conditioned_values")
        if set(values) != set(self.conditioned_components):
            raise ValueError("conditioned_values do not match the emission equation")
        vector = np.asarray(
            [values[component] for component in self.conditioned_components],
            dtype=float,
        )
        return self.intercept + self.conditioned_loadings @ vector


@dataclass(frozen=True)
class PartialDefiningUpdate:
    """Result and lineage from processing one target-month defining event."""

    state: JointGaussianState
    observed_components: Mapping[str, float]
    new_components: tuple[str, ...]
    likelihood_components: tuple[str, ...]
    exact_axes: tuple[str, ...]
    emission: ConditionalComponentEmission | None
    gaussian_update: LinearGaussianUpdate | None
    applied: bool
    reason: str


def prepare_partial_defining_data(
    first_release_components: pd.DataFrame,
    composite_scores: pd.DataFrame,
    *,
    transformed_tolerance: float = 1.0e-10,
    score_tolerance: float = 1.0e-10,
) -> PreparedPartialDefiningData:
    """Join frozen score artifacts into causal partial defining-release events.

    ``first_release_components`` supplies the actual publication date and the
    transformed first-release value.  ``composite_scores`` supplies the
    already-frozen lagged-expanding z-score.  The join is audited against both
    the transformed value and the exact equal-weight composite identity.
    Rows without an available z-score are retained in ``preparation_audit`` but
    are excluded from the live ``events`` table.
    """

    if transformed_tolerance < 0.0 or score_tolerance < 0.0:
        raise ValueError("artifact comparison tolerances must be nonnegative")
    release_required = {
        "reference_month",
        "component",
        "release_date",
        "transformed_value",
    }
    score_required = {
        "reference_month",
        "growth_score",
        "inflation_score",
        "score_available_at",
        *{f"{component}_transformed" for component in ALL_COMPONENTS},
        *{f"{component}_z" for component in ALL_COMPONENTS},
    }
    missing_release = release_required.difference(first_release_components.columns)
    missing_scores = score_required.difference(composite_scores.columns)
    if missing_release:
        raise ValueError(
            "first-release table omits columns: " + ", ".join(sorted(missing_release))
        )
    if missing_scores:
        raise ValueError(
            "composite-score table omits columns: " + ", ".join(sorted(missing_scores))
        )

    releases = first_release_components.loc[:, list(release_required)].copy()
    releases["reference_month"] = pd.to_datetime(
        releases["reference_month"], errors="coerce"
    ).dt.to_period("M").dt.to_timestamp()
    releases["release_date"] = pd.to_datetime(
        releases["release_date"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    releases["component"] = releases["component"].astype(str)
    unknown = set(releases["component"].dropna()).difference(ALL_COMPONENTS)
    if unknown:
        raise ValueError(f"first-release table contains unknown components: {sorted(unknown)}")
    releases["transformed_value"] = pd.to_numeric(
        releases["transformed_value"], errors="coerce"
    )
    if releases.duplicated(["reference_month", "component"]).any():
        raise ValueError("first-release table contains duplicate month/component rows")

    scores = composite_scores.copy()
    scores["reference_month"] = pd.to_datetime(
        scores["reference_month"], errors="coerce"
    ).dt.to_period("M").dt.to_timestamp()
    scores["score_available_at"] = pd.to_datetime(
        scores["score_available_at"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    if scores["reference_month"].isna().any():
        raise ValueError("composite-score table contains an invalid reference month")
    if scores["reference_month"].duplicated().any():
        raise ValueError("composite-score table contains duplicate reference months")
    numeric_columns = [
        "growth_score",
        "inflation_score",
        *[f"{component}_transformed" for component in ALL_COMPONENTS],
        *[f"{component}_z" for component in ALL_COMPONENTS],
    ]
    scores.loc[:, numeric_columns] = scores.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    z_long = scores.melt(
        id_vars=["reference_month"],
        value_vars=[f"{component}_z" for component in ALL_COMPONENTS],
        var_name="component_z_column",
        value_name="component_z",
    )
    z_long["component"] = z_long["component_z_column"].str.removesuffix("_z")
    transformed_long = scores.melt(
        id_vars=["reference_month"],
        value_vars=[f"{component}_transformed" for component in ALL_COMPONENTS],
        var_name="component_transformed_column",
        value_name="score_transformed_value",
    )
    transformed_long["component"] = transformed_long[
        "component_transformed_column"
    ].str.removesuffix("_transformed")
    joined = releases.merge(
        z_long[["reference_month", "component", "component_z"]],
        on=["reference_month", "component"],
        how="left",
        validate="one_to_one",
    ).merge(
        transformed_long[
            ["reference_month", "component", "score_transformed_value"]
        ],
        on=["reference_month", "component"],
        how="left",
        validate="one_to_one",
    )
    finite_pair = np.isfinite(joined["transformed_value"]) & np.isfinite(
        joined["score_transformed_value"]
    )
    mismatch = finite_pair & ~np.isclose(
        joined["transformed_value"],
        joined["score_transformed_value"],
        atol=transformed_tolerance,
        rtol=1.0e-10,
    )
    if mismatch.any():
        sample = joined.loc[mismatch, ["reference_month", "component"]].iloc[0]
        raise ValueError(
            "transformed component artifacts disagree for "
            f"{sample['component']} in {sample['reference_month']:%Y-%m}"
        )

    joined["axis"] = joined["component"].map(COMPONENT_AXIS)
    joined["release_family"] = joined["component"].map(DEFINING_RELEASE_FAMILY)
    joined["component_weight"] = COMPONENT_WEIGHT
    joined["weighted_contribution"] = COMPONENT_WEIGHT * joined["component_z"]
    joined["event_id"] = (
        joined["release_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + joined["reference_month"].dt.strftime("%Y-%m")
    )
    joined["preparation_status"] = "available"
    invalid_date = joined["reference_month"].isna() | joined["release_date"].isna()
    invalid_value = ~np.isfinite(joined["transformed_value"])
    missing_z = ~np.isfinite(joined["component_z"])
    joined.loc[invalid_date, "preparation_status"] = "invalid_date"
    joined.loc[~invalid_date & invalid_value, "preparation_status"] = (
        "invalid_transformed_value"
    )
    joined.loc[~invalid_date & ~invalid_value & missing_z, "preparation_status"] = (
        "component_z_unavailable"
    )

    events = joined.loc[joined["preparation_status"].eq("available")].copy()
    events["event_component_count"] = events.groupby("event_id")[
        "component"
    ].transform("size")
    event_components = events.groupby("event_id")["component"].transform(
        lambda values: "+".join(
            component for component in ALL_COMPONENTS if component in set(values)
        )
    )
    events["event_components"] = event_components
    events = events.sort_values(
        ["release_date", "reference_month", "component"], kind="mergesort"
    ).reset_index(drop=True)

    history_columns = {
        f"{component}_z": component for component in ALL_COMPONENTS
    }
    history = scores.rename(columns=history_columns).loc[
        :,
        [
            "reference_month",
            *ALL_COMPONENTS,
            "growth_score",
            "inflation_score",
            "score_available_at",
        ],
    ].copy()
    release_wide = releases.pivot(
        index="reference_month", columns="component", values="release_date"
    ).rename(columns={component: f"{component}_available_at" for component in ALL_COMPONENTS})
    history = history.merge(
        release_wide.reset_index(), on="reference_month", how="left", validate="one_to_one"
    )
    history["training_available_at"] = history["score_available_at"]
    complete = np.isfinite(history.loc[:, ALL_COMPONENTS]).all(axis=1)
    if complete.any():
        calculated = history.loc[complete, ALL_COMPONENTS].to_numpy(dtype=float) @ (
            score_weight_matrix().T
        )
        published = history.loc[complete, ["growth_score", "inflation_score"]].to_numpy(
            dtype=float
        )
        if not np.allclose(calculated, published, atol=score_tolerance, rtol=1.0e-10):
            raise ValueError("published scores violate the exact equal-weight identity")
    history = history.sort_values("reference_month").reset_index(drop=True)
    return PreparedPartialDefiningData(
        events=events,
        component_history=history,
        preparation_audit=joined.sort_values(
            ["release_date", "reference_month", "component"], kind="mergesort"
        ).reset_index(drop=True),
    )


def fit_causal_component_gaussian(
    component_history: pd.DataFrame,
    knowledge_cutoff: object,
    *,
    minimum_training_samples: int = 36,
    covariance_eigenvalue_floor: float = 1.0e-8,
) -> CausalComponentGaussianFit:
    """Fit the component Gaussian using only fully known earlier score months."""

    if minimum_training_samples < 9:
        raise ValueError("minimum_training_samples must be at least nine")
    if covariance_eigenvalue_floor <= 0.0 or not math.isfinite(
        covariance_eigenvalue_floor
    ):
        raise ValueError("covariance_eigenvalue_floor must be positive")
    required = {
        "reference_month",
        "training_available_at",
        *ALL_COMPONENTS,
    }
    missing = required.difference(component_history.columns)
    if missing:
        raise ValueError(
            "component history omits columns: " + ", ".join(sorted(missing))
        )
    cutoff = _timestamp(knowledge_cutoff, label="knowledge_cutoff")
    history = component_history.loc[:, list(required)].copy()
    history["reference_month"] = pd.to_datetime(
        history["reference_month"], errors="coerce"
    ).dt.to_period("M").dt.to_timestamp()
    history["training_available_at"] = pd.to_datetime(
        history["training_available_at"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    history.loc[:, ALL_COMPONENTS] = history.loc[:, ALL_COMPONENTS].apply(
        pd.to_numeric, errors="coerce"
    )
    complete = (
        history["reference_month"].notna()
        & history["training_available_at"].notna()
        & np.isfinite(history.loc[:, ALL_COMPONENTS]).all(axis=1)
    )
    eligible = history.loc[
        complete & history["training_available_at"].lt(cutoff)
    ].sort_values(["training_available_at", "reference_month"], kind="mergesort")
    if len(eligible) < minimum_training_samples:
        raise ValueError(
            f"only {len(eligible)} complete component months are strictly available; "
            f"need {minimum_training_samples}"
        )
    values = eligible.loc[:, ALL_COMPONENTS].to_numpy(dtype=float)
    estimator = LedoitWolf(assume_centered=False).fit(values)
    covariance = np.asarray(estimator.covariance_, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    scale = max(1.0, float(np.max(eigenvalues)))
    floor = covariance_eigenvalue_floor * scale
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    covariance = (covariance + covariance.T) / 2.0
    return CausalComponentGaussianFit(
        component_order=ALL_COMPONENTS,
        mean=np.asarray(estimator.location_, dtype=float),
        covariance=covariance,
        shrinkage=float(estimator.shrinkage_),
        sample_size=int(len(eligible)),
        knowledge_cutoff=cutoff,
        first_reference_month=pd.Timestamp(eligible["reference_month"].min()),
        last_reference_month=pd.Timestamp(eligible["reference_month"].max()),
        latest_training_availability=pd.Timestamp(
            eligible["training_available_at"].max()
        ),
    )


def _psd_pseudoinverse(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric Moore-Penrose inverse for a PSD matrix."""

    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1.0e-10 * scale
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("conditional covariance is not positive semidefinite")
    inverse = np.where(eigenvalues > tolerance, 1.0 / eigenvalues, 0.0)
    return (eigenvectors * inverse) @ eigenvectors.T


def conditional_component_emission(
    fit: CausalComponentGaussianFit,
    response_components: Sequence[str],
    conditioned_components: Sequence[str] = (),
    *,
    covariance_eigenvalue_floor: float = 1.0e-8,
) -> ConditionalComponentEmission:
    """Derive ``p(response | score, conditioned components)`` from one fit."""

    responses = _component_names(response_components, label="response_components")
    conditioned = _component_names(
        conditioned_components, label="conditioned_components", allow_empty=True
    )
    overlap = set(responses).intersection(conditioned)
    if overlap:
        raise ValueError(f"response and conditioned components overlap: {sorted(overlap)}")
    if covariance_eigenvalue_floor <= 0.0:
        raise ValueError("covariance_eigenvalue_floor must be positive")

    order = fit.component_order
    index = {component: position for position, component in enumerate(order)}
    response_selector = np.zeros((len(responses), len(order)))
    for row, component in enumerate(responses):
        response_selector[row, index[component]] = 1.0
    weights = score_weight_matrix(order)
    conditioned_selector = np.zeros((len(conditioned), len(order)))
    for row, component in enumerate(conditioned):
        conditioned_selector[row, index[component]] = 1.0
    design = np.vstack([weights, conditioned_selector])

    mean_y = response_selector @ fit.mean
    mean_d = design @ fit.mean
    covariance_yd = response_selector @ fit.covariance @ design.T
    covariance_dd = design @ fit.covariance @ design.T
    coefficients = covariance_yd @ _psd_pseudoinverse(covariance_dd)
    covariance_yy = response_selector @ fit.covariance @ response_selector.T
    residual = covariance_yy - coefficients @ covariance_yd.T
    residual = (residual + residual.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(residual)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1.0e-9 * scale
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("derived conditional covariance is not positive semidefinite")
    floor = covariance_eigenvalue_floor * scale
    residual = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    residual = (residual + residual.T) / 2.0
    intercept = mean_y - coefficients @ mean_d
    return ConditionalComponentEmission(
        response_components=responses,
        conditioned_components=conditioned,
        intercept=intercept,
        score_loadings=coefficients[:, :2],
        conditioned_loadings=coefficients[:, 2:],
        noise_covariance=residual,
    )


def condition_on_exact_score_axes(
    state: JointGaussianState,
    reference_month: object,
    exact_axis_values: Mapping[str, float],
    *,
    tolerance: float = 1.0e-10,
) -> JointGaussianState:
    """Condition one or both score axes on deterministic component sums.

    ``JointGaussianState.exact_mask`` remains month-level.  A single exact axis
    is represented by a zero covariance row and column; the mask becomes true
    only after both axes are exact.
    """

    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    if not exact_axis_values:
        return state
    unknown = set(exact_axis_values).difference(AXIS_NAMES)
    if unknown:
        raise ValueError(f"unknown score axes: {sorted(unknown)}")
    values = {axis: float(value) for axis, value in exact_axis_values.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("exact axis values must be finite")
    position = state.position(reference_month)
    block_start = 2 * position
    global_indices = [block_start + AXIS_NAMES.index(axis) for axis in values]
    desired = np.asarray([values[axis] for axis in values], dtype=float)

    if state.exact_mask[position]:
        current = state.mean[global_indices]
        if np.allclose(current, desired, atol=tolerance, rtol=0.0):
            return state
        raise ValueError("conflicting exact axis value for an exact score month")

    covariance_scale = max(1.0, float(np.max(np.abs(state.covariance))))
    numerical_zero = 100.0 * np.finfo(float).eps * covariance_scale
    already_exact = np.asarray(
        [
            np.max(np.abs(state.covariance[index, :])) <= numerical_zero
            for index in global_indices
        ],
        dtype=bool,
    )
    if already_exact.any() and not np.allclose(
        state.mean[np.asarray(global_indices)[already_exact]],
        desired[already_exact],
        atol=tolerance,
        rtol=0.0,
    ):
        raise ValueError("conflicting repeated exact score-axis observation")
    active_indices = [
        index for index, is_exact in zip(global_indices, already_exact) if not is_exact
    ]
    active_values = desired[~already_exact]
    if active_indices:
        observation = np.zeros((len(active_indices), len(state.mean)))
        for row, index in enumerate(active_indices):
            observation[row, index] = 1.0
        innovation_covariance = observation @ state.covariance @ observation.T
        inverse = _psd_pseudoinverse(innovation_covariance)
        gain = state.covariance @ observation.T @ inverse
        mean = state.mean + gain @ (active_values - observation @ state.mean)
        covariance = state.covariance - gain @ observation @ state.covariance
        covariance = (covariance + covariance.T) / 2.0
    else:
        mean = state.mean.copy()
        covariance = state.covariance.copy()

    for index, value in zip(global_indices, desired):
        mean[index] = value
        covariance[index, :] = 0.0
        covariance[:, index] = 0.0

    exact_mask = list(state.exact_mask)
    exact_values = state.exact_values.copy()
    monthly_block = slice(block_start, block_start + 2)
    block_is_exact = (
        np.max(np.abs(covariance[monthly_block, :])) <= numerical_zero
    )
    if block_is_exact:
        exact_mask[position] = True
        exact_values[position] = mean[monthly_block]
    return JointGaussianState(
        reference_months=state.reference_months,
        mean=mean,
        covariance=covariance,
        exact_mask=tuple(exact_mask),
        exact_values=exact_values,
    )


def apply_partial_defining_event(
    state: JointGaussianState,
    reference_month: object,
    new_observations: Mapping[str, float],
    *,
    previously_observed: Mapping[str, float],
    component_fit: CausalComponentGaussianFit,
    covariance_eigenvalue_floor: float = 1.0e-8,
) -> PartialDefiningUpdate:
    """Apply one atomic target-month defining release without double counting.

    The caller persists the returned ``observed_components`` by target month.
    Submitting a component already present in ``previously_observed`` is an
    error, including when the repeated value is identical, because accepting it
    would multiply the same information into the posterior twice.
    """

    month = _month(reference_month)
    previous = _finite_mapping(previously_observed, label="previously_observed")
    new = _finite_mapping(new_observations, label="new_observations")
    if not new:
        raise ValueError("new_observations cannot be empty")
    duplicates = set(previous).intersection(new)
    if duplicates:
        raise ValueError(
            "defining components cannot be processed twice: "
            + ", ".join(sorted(duplicates))
        )
    after = {**previous, **new}
    ordered_after = {
        component: after[component] for component in ALL_COMPONENTS if component in after
    }

    if state.exact_mask[state.position(month)]:
        return PartialDefiningUpdate(
            state=state,
            observed_components=MappingProxyType(ordered_after),
            new_components=tuple(new),
            likelihood_components=(),
            exact_axes=(),
            emission=None,
            gaussian_update=None,
            applied=False,
            reason="target_score_exact",
        )

    exact_axes = tuple(
        axis
        for axis, components in (
            ("growth", GROWTH_COMPONENTS),
            ("inflation", INFLATION_COMPONENTS),
        )
        if set(components).issubset(after)
    )
    completing_components = {
        component for component in new if COMPONENT_AXIS[component] in exact_axes
    }
    likelihood_components = tuple(
        component
        for component in ALL_COMPONENTS
        if component in new and component not in completing_components
    )
    conditioned_values = {
        component: after[component]
        for component in ALL_COMPONENTS
        if component in previous or component in completing_components
    }

    exact_values = {}
    for axis in exact_axes:
        components = GROWTH_COMPONENTS if axis == "growth" else INFLATION_COMPONENTS
        exact_values[axis] = COMPONENT_WEIGHT * sum(after[item] for item in components)
    updated_state = condition_on_exact_score_axes(
        state, month, exact_values
    )

    emission: ConditionalComponentEmission | None = None
    gaussian_update: LinearGaussianUpdate | None = None
    if likelihood_components:
        emission = conditional_component_emission(
            component_fit,
            likelihood_components,
            tuple(conditioned_values),
            covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        )
        observation = np.asarray(
            [new[component] for component in emission.response_components], dtype=float
        )
        gaussian_update = update_joint_gaussian(
            updated_state,
            month,
            observation,
            emission.score_loadings,
            emission.noise_covariance,
            intercept=emission.offset(conditioned_values),
        )
        updated_state = gaussian_update.state

    applied = bool(exact_axes) or bool(
        gaussian_update is not None and gaussian_update.applied
    )
    reason = "updated" if applied else "target_score_exact"
    return PartialDefiningUpdate(
        state=updated_state,
        observed_components=MappingProxyType(ordered_after),
        new_components=tuple(component for component in ALL_COMPONENTS if component in new),
        likelihood_components=likelihood_components,
        exact_axes=exact_axes,
        emission=emission,
        gaussian_update=gaussian_update,
        applied=applied,
        reason=reason,
    )
