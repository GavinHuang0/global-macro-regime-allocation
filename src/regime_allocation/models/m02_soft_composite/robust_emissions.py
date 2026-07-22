"""Robust Student-t release-block emissions for Model 02.

This module is an opt-in sensitivity beside :mod:`gaussian_emissions`; it does
not change the linear-Gaussian baseline.  It estimates the observation model

``y[e] = a + H z[q(e)] + C v[e] + epsilon[e]``

with a multivariate Student-t residual whose degrees of freedom are shared by
the responses in one release block.  Coefficients and the residual *scale*
matrix are estimated by iteratively reweighted ridge regression.  For a
``d``-dimensional residual ``r`` the normal--Gamma mixture representation gives

``w = (nu + d) / (nu + r' R^{-1} r)``.

The same formula supplies an auditable event weight for the robust approximate
Kalman update.  The update replaces ``R`` by ``R / w`` and then performs a
Joseph-form Gaussian update.  This is a moment approximation--a Student-t
likelihood times a Gaussian prior is not generally Gaussian--but deliberately
retains the repository's tractable rolling joint-Gaussian state.

All model selection is causal.  Rows at a validation timestamp are held out as
one group, and both the structured ridge penalty and ``nu`` may be selected by
rolling-origin predictive Student-t log likelihood.  ``nu=inf`` is the nested
Gaussian sensitivity.  This module performs no file or network access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from regime_allocation.models.m02_soft_composite.gaussian_emissions import (
    LinearGaussianEmissionSpec,
    select_causal_emission_rows,
)


DEFAULT_DEGREES_OF_FREEDOM_GRID = (4.0, 5.0, 7.0, 10.0, math.inf)


def _validate_degrees_of_freedom(value: float) -> float:
    degrees = float(value)
    if math.isinf(degrees) and degrees > 0.0:
        return degrees
    if not math.isfinite(degrees) or degrees <= 2.0:
        raise ValueError("degrees_of_freedom must exceed two or equal infinity")
    return degrees


def _validate_degrees_grid(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("degrees_of_freedom_grid must be a numeric sequence")
    grid = tuple(_validate_degrees_of_freedom(value) for value in values)
    if not grid:
        raise ValueError("degrees_of_freedom_grid cannot be empty")
    if len(grid) != len(set(grid)):
        raise ValueError("degrees_of_freedom_grid contains duplicates")
    finite = tuple(value for value in grid if math.isfinite(value))
    if any(right <= left for left, right in zip(finite, finite[1:])):
        raise ValueError("finite degrees_of_freedom_grid values must increase")
    if any(math.isinf(value) for value in grid) and not math.isinf(grid[-1]):
        raise ValueError("infinity must be the final degrees-of-freedom candidate")
    return grid


def _symmetric_positive_definite(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    dimension: int,
    label: str,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (dimension, dimension) or not np.isfinite(values).all():
        raise ValueError(f"{label} has an invalid shape or value")
    values = (values + values.T) / 2.0
    if float(np.linalg.eigvalsh(values).min()) <= 0.0:
        raise ValueError(f"{label} must be positive definite")
    return values


def _symmetric_positive_semidefinite(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    dimension: int,
    label: str,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Validate a covariance while preserving legitimately exact state axes.

    Exact-score conditioning can leave the joint state covariance singular.
    That is valid: the strictly positive-definite release scale still makes
    ``H P H' + R`` invertible.  Small negative eigenvalues within a relative
    numerical tolerance are projected to zero; material asymmetry or negative
    eigenvalues remain errors.
    """

    values = np.asarray(matrix, dtype=float)
    if values.shape != (dimension, dimension) or not np.isfinite(values).all():
        raise ValueError(f"{label} has an invalid shape or value")
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.max(np.abs(values - values.T))) > tolerance * scale:
        raise ValueError(f"{label} must be symmetric")
    symmetric = (values + values.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(eigenvalues.min()) < -tolerance * scale:
        raise ValueError(f"{label} must be positive semidefinite")
    if float(eigenvalues.min()) < 0.0:
        eigenvalues = np.maximum(eigenvalues, 0.0)
        symmetric = (eigenvectors * eigenvalues) @ eigenvectors.T
        symmetric = (symmetric + symmetric.T) / 2.0
    return symmetric


def _floor_scale(scale: np.ndarray, floor: float) -> tuple[np.ndarray, float, float]:
    symmetric = (np.asarray(scale, dtype=float) + np.asarray(scale, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    before = float(eigenvalues.min())
    floored = np.maximum(eigenvalues, float(floor))
    result = (eigenvectors * floored) @ eigenvectors.T
    result = (result + result.T) / 2.0
    return result, before, float(np.linalg.eigvalsh(result).min())


def _mahalanobis_squared(
    residuals: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    scale: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(residuals, dtype=float)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[None, :]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("residuals must be a finite vector or matrix")
    covariance = _symmetric_positive_definite(
        scale, dimension=values.shape[1], label="scale"
    )
    solved = np.linalg.solve(covariance, values.T).T
    result = np.sum(values * solved, axis=1)
    return result[0] if was_vector else result


def student_t_event_weight(
    residual: Sequence[float] | np.ndarray,
    scale: Sequence[Sequence[float]] | np.ndarray,
    degrees_of_freedom: float,
) -> float:
    """Return the Student-t latent-precision expectation for one event.

    The weight is exactly one for the Gaussian sensitivity.  A large
    Mahalanobis innovation receives a weight below one; a sufficiently central
    observation can receive a weight modestly above one, as implied by the
    untruncated normal--Gamma mixture rather than an ad-hoc clipped rule.
    """

    degrees = _validate_degrees_of_freedom(degrees_of_freedom)
    values = np.asarray(residual, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("residual must be a nonempty finite vector")
    if math.isinf(degrees):
        return 1.0
    distance = float(_mahalanobis_squared(values, scale))
    return float((degrees + len(values)) / (degrees + distance))


def multivariate_student_t_nll(
    residuals: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    scale: Sequence[Sequence[float]] | np.ndarray,
    degrees_of_freedom: float,
) -> np.ndarray | float:
    """Return multivariate Student-t negative log density values.

    ``scale`` is the Student-t scale matrix, not its covariance.  For finite
    ``nu > 2``, the residual covariance is ``nu / (nu - 2) * scale``.
    """

    degrees = _validate_degrees_of_freedom(degrees_of_freedom)
    values = np.asarray(residuals, dtype=float)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[None, :]
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("residuals must be a nonempty finite vector or matrix")
    scale_array = _symmetric_positive_definite(
        scale, dimension=values.shape[1], label="scale"
    )
    sign, log_determinant = np.linalg.slogdet(scale_array)
    if sign <= 0:
        raise RuntimeError("Student-t scale is not positive definite")
    distances = np.asarray(_mahalanobis_squared(values, scale_array), dtype=float)
    dimension = values.shape[1]
    if math.isinf(degrees):
        result = 0.5 * (
            dimension * math.log(2.0 * math.pi)
            + float(log_determinant)
            + distances
        )
    else:
        constant = (
            math.lgamma((degrees + dimension) / 2.0)
            - math.lgamma(degrees / 2.0)
            - 0.5
            * (
                dimension * math.log(degrees * math.pi)
                + float(log_determinant)
            )
        )
        result = -constant + 0.5 * (degrees + dimension) * np.log1p(
            distances / degrees
        )
    return float(result[0]) if was_vector else result


def _control_array(
    controls: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray | None,
    control_names: tuple[str, ...],
) -> np.ndarray:
    if not control_names:
        if controls is not None and np.asarray(controls).size:
            raise ValueError("controls were supplied to a model without controls")
        return np.empty(0, dtype=float)
    if controls is None:
        raise ValueError("all configured controls must be supplied")
    if isinstance(controls, (Mapping, pd.Series)):
        supplied = dict(controls)
        missing = set(control_names).difference(supplied)
        if missing:
            raise ValueError(f"controls are missing values: {sorted(missing)}")
        values = np.asarray([supplied[name] for name in control_names], dtype=float)
    else:
        values = np.asarray(controls, dtype=float)
    if values.shape != (len(control_names),) or not np.isfinite(values).all():
        raise ValueError("controls have an invalid shape or value")
    return values


def _identifiable_controls(controls: np.ndarray) -> np.ndarray:
    if controls.ndim != 2:
        raise ValueError("controls must be two dimensional")
    if controls.shape[1] == 0:
        return np.zeros(0, dtype=bool)
    minimum = controls.min(axis=0)
    maximum = controls.max(axis=0)
    tolerance = 1.0e-12 * np.maximum(
        1.0, np.maximum(np.abs(minimum), np.abs(maximum))
    )
    return maximum - minimum > tolerance


@dataclass(frozen=True)
class _RobustEstimate:
    intercept: np.ndarray
    state_loadings: np.ndarray
    control_loadings: np.ndarray
    residual_scale: np.ndarray
    residuals: np.ndarray
    observation_weights: np.ndarray
    iterations: int
    converged: bool
    ledoit_wolf_shrinkage: float
    scale_minimum_eigenvalue_before_floor: float
    scale_minimum_eigenvalue_after_floor: float
    dropped_constant_controls: tuple[str, ...]


def _weighted_ridge_fit(
    responses: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    *,
    spec: LinearGaussianEmissionSpec,
    base_lambda: float,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    observations, output_count = responses.shape
    state_count = states.shape[1]
    control_count = controls.shape[1]
    if weights.shape != (observations,) or not np.isfinite(weights).all() or (
        weights <= 0.0
    ).any():
        raise ValueError("IRLS weights must be finite and positive")
    intercept = np.zeros(output_count, dtype=float)
    state_loadings = np.zeros((output_count, state_count), dtype=float)
    control_loadings = np.zeros((output_count, control_count), dtype=float)
    penalties = np.asarray(spec.state_loading_penalties, dtype=float)
    restrictions = np.asarray(spec.exact_zero_mask, dtype=bool)
    active_controls = _identifiable_controls(controls)
    dropped = tuple(
        name
        for name, active in zip(spec.control_names, active_controls)
        if not active
    )
    root_weights = np.sqrt(weights)

    for output in range(output_count):
        allowed = ~restrictions[output]
        design = np.column_stack(
            [
                np.ones(observations, dtype=float),
                states[:, allowed],
                controls[:, active_controls],
            ]
        )
        diagonal = np.zeros(design.shape[1], dtype=float)
        allowed_count = int(allowed.sum())
        diagonal[1 : 1 + allowed_count] = (
            float(base_lambda) * penalties[output, allowed]
        )
        weighted_design = design * root_weights[:, None]
        weighted_response = responses[:, output] * root_weights
        unpenalized = diagonal == 0.0
        if np.linalg.matrix_rank(weighted_design[:, unpenalized]) != int(
            unpenalized.sum()
        ):
            raise ValueError(
                f"unpenalized design is rank deficient for "
                f"{spec.response_names[output]}"
            )
        system = weighted_design.T @ weighted_design + np.diag(diagonal)
        target = weighted_design.T @ weighted_response
        coefficients = np.linalg.solve(system, target)
        intercept[output] = coefficients[0]
        state_loadings[output, allowed] = coefficients[1 : 1 + allowed_count]
        active_count = int(active_controls.sum())
        if active_count:
            control_loadings[output, active_controls] = coefficients[-active_count:]
    return intercept, state_loadings, control_loadings, dropped


def _robust_scale(
    residuals: np.ndarray,
    weights: np.ndarray,
    *,
    floor: float,
) -> tuple[np.ndarray, float, float, float]:
    pseudo_residuals = residuals * np.sqrt(weights)[:, None]
    estimator = LedoitWolf(assume_centered=True, store_precision=False).fit(
        pseudo_residuals
    )
    scale, before, after = _floor_scale(
        np.asarray(estimator.covariance_, dtype=float), floor
    )
    return scale, float(estimator.shrinkage_), before, after


def _fit_fixed_hyperparameters(
    table: pd.DataFrame,
    *,
    spec: LinearGaussianEmissionSpec,
    base_lambda: float,
    degrees_of_freedom: float,
    maximum_iterations: int,
    tolerance: float,
    minimum_weight: float,
) -> _RobustEstimate:
    degrees = _validate_degrees_of_freedom(degrees_of_freedom)
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not math.isfinite(minimum_weight) or not 0.0 < minimum_weight <= 1.0:
        raise ValueError("minimum_weight must belong to (0, 1]")
    responses = table.loc[:, spec.response_names].to_numpy(dtype=float)
    states = table.loc[:, spec.state_names].to_numpy(dtype=float)
    controls = table.loc[:, spec.control_names].to_numpy(dtype=float)
    observations = len(table)
    dimension = len(spec.response_names)
    weights = np.ones(observations, dtype=float)
    converged = math.isinf(degrees)
    iterations = 0

    for iteration in range(1, maximum_iterations + 1):
        intercept, state_loadings, control_loadings, dropped = _weighted_ridge_fit(
            responses,
            states,
            controls,
            spec=spec,
            base_lambda=base_lambda,
            weights=weights,
        )
        residuals = responses - (
            intercept[None, :]
            + states @ state_loadings.T
            + controls @ control_loadings.T
        )
        scale, shrinkage, before, after = _robust_scale(
            residuals,
            weights,
            floor=spec.covariance_eigenvalue_floor,
        )
        iterations = iteration
        if math.isinf(degrees):
            new_weights = np.ones(observations, dtype=float)
        else:
            distances = np.asarray(_mahalanobis_squared(residuals, scale))
            new_weights = (degrees + dimension) / (degrees + distances)
            new_weights = np.maximum(new_weights, minimum_weight)
        relative_change = float(
            np.max(np.abs(new_weights - weights) / np.maximum(1.0, np.abs(weights)))
        )
        weights = new_weights
        if math.isinf(degrees) or relative_change <= tolerance:
            converged = True
            break

    # Refit at the final latent-precision expectation.  This makes the returned
    # coefficient and scale estimates correspond to the recorded weights even
    # when the maximum-iteration guard fired one step before convergence.
    intercept, state_loadings, control_loadings, dropped = _weighted_ridge_fit(
        responses,
        states,
        controls,
        spec=spec,
        base_lambda=base_lambda,
        weights=weights,
    )
    residuals = responses - (
        intercept[None, :]
        + states @ state_loadings.T
        + controls @ control_loadings.T
    )
    scale, shrinkage, before, after = _robust_scale(
        residuals, weights, floor=spec.covariance_eigenvalue_floor
    )
    return _RobustEstimate(
        intercept=intercept,
        state_loadings=state_loadings,
        control_loadings=control_loadings,
        residual_scale=scale,
        residuals=residuals,
        observation_weights=weights,
        iterations=iterations,
        converged=converged,
        ledoit_wolf_shrinkage=shrinkage,
        scale_minimum_eigenvalue_before_floor=before,
        scale_minimum_eigenvalue_after_floor=after,
        dropped_constant_controls=dropped,
    )


def _predict(
    estimate: _RobustEstimate,
    table: pd.DataFrame,
    spec: LinearGaussianEmissionSpec,
) -> np.ndarray:
    states = table.loc[:, spec.state_names].to_numpy(dtype=float)
    controls = table.loc[:, spec.control_names].to_numpy(dtype=float)
    return (
        estimate.intercept[None, :]
        + states @ estimate.state_loadings.T
        + controls @ estimate.control_loadings.T
    )


def _rolling_origin_score(
    table: pd.DataFrame,
    *,
    spec: LinearGaussianEmissionSpec,
    availability_column: str,
    base_lambda: float,
    degrees_of_freedom: float,
    maximum_iterations: int,
    tolerance: float,
    minimum_weight: float,
) -> dict[str, float | int]:
    nll_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    origin_count = 0
    for origin in table[availability_column].drop_duplicates().sort_values():
        training = table[table[availability_column] < origin]
        validation = table[table[availability_column] == origin]
        if len(training) < spec.validation_minimum_training_samples:
            continue
        estimate = _fit_fixed_hyperparameters(
            training,
            spec=spec,
            base_lambda=base_lambda,
            degrees_of_freedom=degrees_of_freedom,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            minimum_weight=minimum_weight,
        )
        residuals = (
            validation.loc[:, spec.response_names].to_numpy(dtype=float)
            - _predict(estimate, validation, spec)
        )
        nll = np.asarray(
            multivariate_student_t_nll(
                residuals, estimate.residual_scale, degrees_of_freedom
            )
        )
        event_weights = np.asarray(
            [
                student_t_event_weight(
                    residual, estimate.residual_scale, degrees_of_freedom
                )
                for residual in residuals
            ],
            dtype=float,
        )
        nll_parts.append(nll)
        weight_parts.append(event_weights)
        origin_count += 1
    nll_values = np.concatenate(nll_parts) if nll_parts else np.empty(0)
    weight_values = np.concatenate(weight_parts) if weight_parts else np.empty(0)
    return {
        "base_lambda": float(base_lambda),
        "degrees_of_freedom": float(degrees_of_freedom),
        "validation_origins": int(origin_count),
        "validation_observations": int(len(nll_values)),
        "total_predictive_student_t_nll": (
            float(nll_values.sum()) if len(nll_values) else math.nan
        ),
        "mean_predictive_student_t_nll": (
            float(nll_values.mean()) if len(nll_values) else math.nan
        ),
        "median_predictive_student_t_nll": (
            float(np.median(nll_values)) if len(nll_values) else math.nan
        ),
        "mean_validation_event_weight": (
            float(weight_values.mean()) if len(weight_values) else math.nan
        ),
        "minimum_validation_event_weight": (
            float(weight_values.min()) if len(weight_values) else math.nan
        ),
    }


@dataclass(frozen=True)
class StudentTObservedEmissionSystem:
    """Observed subvector of a fitted multivariate Student-t block."""

    observed_names: tuple[str, ...]
    omitted_names: tuple[str, ...]
    observation: np.ndarray
    intercept: np.ndarray
    state_loadings: np.ndarray
    control_contribution: np.ndarray
    residual_scale: np.ndarray
    degrees_of_freedom: float

    @property
    def adjusted_observation(self) -> np.ndarray:
        return self.observation - self.intercept - self.control_contribution

    def conditional_log_likelihood(
        self, state: Sequence[float] | np.ndarray
    ) -> float:
        values = np.asarray(state, dtype=float)
        if values.shape != (self.state_loadings.shape[1],) or not np.isfinite(
            values
        ).all():
            raise ValueError("state has an invalid shape or value")
        residual = self.adjusted_observation - self.state_loadings @ values
        return -float(
            multivariate_student_t_nll(
                residual, self.residual_scale, self.degrees_of_freedom
            )
        )

    def approximate_gaussian_system(
        self,
        prior_mean: Sequence[float] | np.ndarray,
        prior_covariance: Sequence[Sequence[float]] | np.ndarray,
        *,
        minimum_event_weight: float = 1.0e-8,
    ) -> "RobustGaussianApproximation":
        """Approximate this Student-t event by one adaptive Gaussian system."""

        mean = np.asarray(prior_mean, dtype=float)
        if mean.shape != (self.state_loadings.shape[1],) or not np.isfinite(mean).all():
            raise ValueError("prior_mean has an invalid shape or value")
        covariance = _symmetric_positive_semidefinite(
            prior_covariance, dimension=len(mean), label="prior_covariance"
        )
        if not math.isfinite(minimum_event_weight) or not (
            0.0 < minimum_event_weight <= 1.0
        ):
            raise ValueError("minimum_event_weight must belong to (0, 1]")
        predictive_mean = (
            self.intercept
            + self.control_contribution
            + self.state_loadings @ mean
        )
        nominal_predictive_scale = (
            self.state_loadings @ covariance @ self.state_loadings.T
            + self.residual_scale
        )
        nominal_predictive_scale = (
            nominal_predictive_scale + nominal_predictive_scale.T
        ) / 2.0
        innovation = self.observation - predictive_mean
        raw_weight = student_t_event_weight(
            innovation, nominal_predictive_scale, self.degrees_of_freedom
        )
        event_weight = max(raw_weight, float(minimum_event_weight))
        effective_noise = self.residual_scale / event_weight
        return RobustGaussianApproximation(
            observation=self.observation.copy(),
            loading=self.state_loadings.copy(),
            intercept=(self.intercept + self.control_contribution).copy(),
            nominal_noise_scale=self.residual_scale.copy(),
            effective_noise_covariance=effective_noise,
            predictive_mean=predictive_mean,
            nominal_predictive_scale=nominal_predictive_scale,
            innovation=innovation,
            mahalanobis_squared=float(
                _mahalanobis_squared(innovation, nominal_predictive_scale)
            ),
            raw_event_weight=float(raw_weight),
            event_weight=float(event_weight),
            minimum_event_weight=float(minimum_event_weight),
            degrees_of_freedom=float(self.degrees_of_freedom),
            approximate_log_predictive_density=-float(
                multivariate_student_t_nll(
                    innovation,
                    nominal_predictive_scale,
                    self.degrees_of_freedom,
                )
            ),
            observed_names=self.observed_names,
            omitted_names=self.omitted_names,
        )


@dataclass(frozen=True)
class RobustGaussianApproximation:
    """Adaptive Gaussian observation used for one Student-t event update."""

    observation: np.ndarray
    loading: np.ndarray
    intercept: np.ndarray
    nominal_noise_scale: np.ndarray
    effective_noise_covariance: np.ndarray
    predictive_mean: np.ndarray
    nominal_predictive_scale: np.ndarray
    innovation: np.ndarray
    mahalanobis_squared: float
    raw_event_weight: float
    event_weight: float
    minimum_event_weight: float
    degrees_of_freedom: float
    approximate_log_predictive_density: float
    observed_names: tuple[str, ...]
    omitted_names: tuple[str, ...]

    def to_joint_filter_inputs(self) -> dict[str, np.ndarray]:
        """Return adaptive arrays accepted by ``update_joint_gaussian``."""

        return {
            "observation": self.observation.copy(),
            "loading": self.loading.copy(),
            "noise_covariance": self.effective_noise_covariance.copy(),
            "intercept": self.intercept.copy(),
        }


@dataclass(frozen=True)
class RobustStateUpdate:
    """Gaussian moment approximation to one Student-t measurement update."""

    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    predictive_mean: np.ndarray
    effective_predictive_covariance: np.ndarray
    innovation: np.ndarray
    kalman_gain: np.ndarray
    approximate_log_predictive_density: float
    mahalanobis_squared: float
    raw_event_weight: float
    event_weight: float
    degrees_of_freedom: float
    observed_names: tuple[str, ...]
    omitted_names: tuple[str, ...]

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return strict-JSON-safe metadata, including the event weight."""

        degrees: float | str = (
            "infinity"
            if math.isinf(self.degrees_of_freedom)
            else float(self.degrees_of_freedom)
        )
        return {
            "approximation": "student_t_scale_mixture_single_weight_gaussian_moment_update",
            "observed_names": list(self.observed_names),
            "omitted_names": list(self.omitted_names),
            "degrees_of_freedom": degrees,
            "mahalanobis_squared": float(self.mahalanobis_squared),
            "raw_event_weight": float(self.raw_event_weight),
            "event_weight": float(self.event_weight),
            "prior_mean": self.prior_mean.tolist(),
            "prior_covariance": self.prior_covariance.tolist(),
            "predictive_mean": self.predictive_mean.tolist(),
            "effective_predictive_covariance": (
                self.effective_predictive_covariance.tolist()
            ),
            "innovation": self.innovation.tolist(),
            "kalman_gain": self.kalman_gain.tolist(),
            "posterior_mean": self.posterior_mean.tolist(),
            "posterior_covariance": self.posterior_covariance.tolist(),
            "approximate_log_predictive_density": float(
                self.approximate_log_predictive_density
            ),
        }


@dataclass(frozen=True)
class LinearStudentTEmissionFit:
    """Causally selected robust linear Student-t release-block fit."""

    spec: LinearGaussianEmissionSpec
    intercept: pd.Series
    state_loadings: pd.DataFrame
    control_loadings: pd.DataFrame
    residual_scale: pd.DataFrame
    selected_base_lambda: float
    selected_degrees_of_freedom: float
    hyperparameter_selection: pd.DataFrame
    hyperparameter_selection_method: str
    training_count: int
    knowledge_cutoff: pd.Timestamp | None
    first_training_available_at: pd.Timestamp
    last_training_available_at: pd.Timestamp
    selection_diagnostics: dict[str, int]
    irls_iterations: int
    irls_converged: bool
    training_weights: pd.Series
    ledoit_wolf_shrinkage: float
    scale_minimum_eigenvalue_before_floor: float
    scale_minimum_eigenvalue_after_floor: float
    dropped_constant_controls: tuple[str, ...]
    residual_mean: pd.Series
    residual_rmse: pd.Series

    @property
    def residual_covariance(self) -> pd.DataFrame:
        """Return the finite-variance residual covariance implied by the scale."""

        if math.isinf(self.selected_degrees_of_freedom):
            multiplier = 1.0
        else:
            multiplier = self.selected_degrees_of_freedom / (
                self.selected_degrees_of_freedom - 2.0
            )
        return self.residual_scale * multiplier

    def observed_system(
        self,
        observation: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray,
        *,
        controls: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray | None = None,
    ) -> StudentTObservedEmissionSystem:
        if isinstance(observation, (Mapping, pd.Series)):
            supplied = dict(observation)
            unknown = set(supplied).difference(self.spec.response_names)
            if unknown:
                raise ValueError(
                    f"observation contains unknown responses: {sorted(unknown)}"
                )
            observed = tuple(
                name
                for name in self.spec.response_names
                if name in supplied and pd.notna(supplied[name])
            )
            values = np.asarray([supplied[name] for name in observed], dtype=float)
        else:
            full = np.asarray(observation, dtype=float)
            if full.shape != (len(self.spec.response_names),):
                raise ValueError("observation has the wrong dimension")
            mask = np.isfinite(full)
            observed = tuple(
                name for name, available in zip(self.spec.response_names, mask) if available
            )
            values = full[mask]
        if not observed or not np.isfinite(values).all():
            raise ValueError("at least one finite response must be observed")
        omitted = tuple(name for name in self.spec.response_names if name not in observed)
        control_values = _control_array(controls, self.spec.control_names)
        return StudentTObservedEmissionSystem(
            observed_names=observed,
            omitted_names=omitted,
            observation=values,
            intercept=self.intercept.loc[list(observed)].to_numpy(dtype=float),
            state_loadings=self.state_loadings.loc[list(observed)].to_numpy(dtype=float),
            control_contribution=(
                self.control_loadings.loc[list(observed)].to_numpy(dtype=float)
                @ control_values
            ),
            residual_scale=self.residual_scale.loc[
                list(observed), list(observed)
            ].to_numpy(dtype=float),
            degrees_of_freedom=self.selected_degrees_of_freedom,
        )

    def update_gaussian_state(
        self,
        prior_mean: Sequence[float] | np.ndarray,
        prior_covariance: Sequence[Sequence[float]] | np.ndarray,
        observation: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray,
        *,
        controls: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray | None = None,
        minimum_event_weight: float = 1.0e-8,
    ) -> RobustStateUpdate:
        """Apply a one-weight Student-t Gaussian moment approximation."""

        mean = np.asarray(prior_mean, dtype=float)
        if mean.shape != (len(self.spec.state_names),) or not np.isfinite(mean).all():
            raise ValueError("prior_mean has an invalid shape or value")
        covariance = _symmetric_positive_semidefinite(
            prior_covariance, dimension=len(mean), label="prior_covariance"
        )
        system = self.observed_system(observation, controls=controls)
        approximation = system.approximate_gaussian_system(
            mean, covariance, minimum_event_weight=minimum_event_weight
        )
        h = approximation.loading
        noise = approximation.effective_noise_covariance
        predictive_covariance = h @ covariance @ h.T + noise
        predictive_covariance = (predictive_covariance + predictive_covariance.T) / 2.0
        gain = np.linalg.solve(predictive_covariance, h @ covariance).T
        posterior_mean = mean + gain @ approximation.innovation
        identity_minus_kh = np.eye(len(mean)) - gain @ h
        posterior_covariance = (
            identity_minus_kh @ covariance @ identity_minus_kh.T
            + gain @ noise @ gain.T
        )
        posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2.0
        return RobustStateUpdate(
            prior_mean=mean,
            prior_covariance=covariance,
            posterior_mean=posterior_mean,
            posterior_covariance=posterior_covariance,
            predictive_mean=approximation.predictive_mean,
            effective_predictive_covariance=predictive_covariance,
            innovation=approximation.innovation,
            kalman_gain=gain,
            approximate_log_predictive_density=(
                approximation.approximate_log_predictive_density
            ),
            mahalanobis_squared=approximation.mahalanobis_squared,
            raw_event_weight=approximation.raw_event_weight,
            event_weight=approximation.event_weight,
            degrees_of_freedom=approximation.degrees_of_freedom,
            observed_names=approximation.observed_names,
            omitted_names=approximation.omitted_names,
        )

    def to_audit_dict(self) -> dict[str, Any]:
        """Return strict-JSON-safe fit, tuning, and IRLS diagnostics."""

        def degree(value: float) -> float | str:
            return "infinity" if math.isinf(value) else float(value)

        rows: list[dict[str, Any]] = []
        for record in self.hyperparameter_selection.to_dict(orient="records"):
            cleaned = dict(record)
            cleaned["degrees_of_freedom"] = degree(
                float(cleaned["degrees_of_freedom"])
            )
            for key, value in tuple(cleaned.items()):
                if isinstance(value, (np.bool_, bool)):
                    cleaned[key] = bool(value)
                elif isinstance(value, (np.integer,)):
                    cleaned[key] = int(value)
                elif isinstance(value, (np.floating,)):
                    cleaned[key] = float(value)
            rows.append(cleaned)
        cutoff = None if self.knowledge_cutoff is None else self.knowledge_cutoff.isoformat()
        quantiles = self.training_weights.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        return {
            "model_equation": "y = intercept + state_loadings @ z + control_loadings @ v + epsilon",
            "distribution": "multivariate_student_t",
            "scale_note": "residual_scale is Student-t scale, not covariance",
            "specification": self.spec.to_audit_dict(),
            "selected_base_lambda": float(self.selected_base_lambda),
            "selected_degrees_of_freedom": degree(
                self.selected_degrees_of_freedom
            ),
            "hyperparameter_selection_method": self.hyperparameter_selection_method,
            "hyperparameter_selection": rows,
            "training_count": int(self.training_count),
            "knowledge_cutoff": cutoff,
            "first_training_available_at": self.first_training_available_at.isoformat(),
            "last_training_available_at": self.last_training_available_at.isoformat(),
            "strict_cutoff_rule": "training_available_at < knowledge_cutoff",
            "rolling_origin_rule": (
                "equal availability timestamps held out together"
                if self.hyperparameter_selection_method
                == "causal_rolling_origin_predictive_student_t_nll"
                else "not_applicable_hyperparameters_fixed_before_refit"
            ),
            "intercept": self.intercept.to_dict(),
            "state_loadings": self.state_loadings.to_dict(orient="index"),
            "control_loadings": self.control_loadings.to_dict(orient="index"),
            "residual_scale": self.residual_scale.to_dict(orient="index"),
            "residual_covariance": self.residual_covariance.to_dict(orient="index"),
            "irls_iterations": int(self.irls_iterations),
            "irls_converged": bool(self.irls_converged),
            "training_weight_quantiles": {
                str(float(key)): float(value) for key, value in quantiles.items()
            },
            "ledoit_wolf_shrinkage": float(self.ledoit_wolf_shrinkage),
            "scale_minimum_eigenvalue_before_floor": float(
                self.scale_minimum_eigenvalue_before_floor
            ),
            "scale_minimum_eigenvalue_after_floor": float(
                self.scale_minimum_eigenvalue_after_floor
            ),
            "dropped_constant_controls": list(self.dropped_constant_controls),
            "residual_mean": self.residual_mean.to_dict(),
            "residual_rmse": self.residual_rmse.to_dict(),
            "selection_diagnostics": {
                key: int(value) for key, value in self.selection_diagnostics.items()
            },
        }


def _assemble_student_t_fit(
    *,
    spec: LinearGaussianEmissionSpec,
    training: pd.DataFrame,
    selection_diagnostics: dict[str, int],
    knowledge_cutoff: pd.Timestamp | None,
    availability_column: str,
    selected_lambda: float,
    selected_degrees: float,
    hyperparameter_selection: pd.DataFrame,
    hyperparameter_selection_method: str,
    estimate: _RobustEstimate,
) -> LinearStudentTEmissionFit:
    """Build the common immutable fit record for tuned and fixed refits."""

    responses = spec.response_names
    residual_mean = estimate.residuals.mean(axis=0)
    residual_rmse = np.sqrt(np.square(estimate.residuals).mean(axis=0))
    return LinearStudentTEmissionFit(
        spec=spec,
        intercept=pd.Series(estimate.intercept, index=responses),
        state_loadings=pd.DataFrame(
            estimate.state_loadings, index=responses, columns=spec.state_names
        ),
        control_loadings=pd.DataFrame(
            estimate.control_loadings, index=responses, columns=spec.control_names
        ),
        residual_scale=pd.DataFrame(
            estimate.residual_scale, index=responses, columns=responses
        ),
        selected_base_lambda=float(selected_lambda),
        selected_degrees_of_freedom=float(selected_degrees),
        hyperparameter_selection=hyperparameter_selection.reset_index(drop=True),
        hyperparameter_selection_method=str(hyperparameter_selection_method),
        training_count=len(training),
        knowledge_cutoff=knowledge_cutoff,
        first_training_available_at=pd.Timestamp(training[availability_column].min()),
        last_training_available_at=pd.Timestamp(training[availability_column].max()),
        selection_diagnostics=selection_diagnostics,
        irls_iterations=estimate.iterations,
        irls_converged=estimate.converged,
        training_weights=pd.Series(
            estimate.observation_weights,
            index=training.index,
            name="student_t_irls_weight",
        ),
        ledoit_wolf_shrinkage=estimate.ledoit_wolf_shrinkage,
        scale_minimum_eigenvalue_before_floor=(
            estimate.scale_minimum_eigenvalue_before_floor
        ),
        scale_minimum_eigenvalue_after_floor=(
            estimate.scale_minimum_eigenvalue_after_floor
        ),
        dropped_constant_controls=estimate.dropped_constant_controls,
        residual_mean=pd.Series(residual_mean, index=responses),
        residual_rmse=pd.Series(residual_rmse, index=responses),
    )


def fit_linear_student_t_emission_fixed(
    table: pd.DataFrame,
    *,
    spec: LinearGaussianEmissionSpec,
    availability_column: str,
    base_lambda: float,
    degrees_of_freedom: float = 7.0,
    knowledge_cutoff: date | datetime | str | pd.Timestamp | None = None,
    maximum_iterations: int = 100,
    tolerance: float = 1.0e-7,
    minimum_weight: float = 1.0e-8,
) -> LinearStudentTEmissionFit:
    """Fit robust coefficients once at preselected hyperparameters.

    The input may already be causally prefiltered.  Supplying
    ``knowledge_cutoff`` additionally applies the repository's strict
    ``availability < cutoff`` rule.  Unlike
    :func:`fit_linear_student_t_emission`, this function performs no
    rolling-origin validation and therefore is appropriate for coefficient
    refits between slower scheduled hyperparameter-selection dates.

    ``base_lambda`` must belong to the spec's frozen grid so an integration
    runner cannot silently refit at an unregistered penalty.  The returned
    fit uses the same type as the tuned API, while its selection table and
    audit dictionary explicitly mark the hyperparameters as fixed.
    """

    selected_lambda = float(base_lambda)
    if (
        not math.isfinite(selected_lambda)
        or selected_lambda < 0.0
        or selected_lambda not in spec.lambda_grid
    ):
        raise ValueError("base_lambda must belong to the spec's frozen lambda_grid")
    selected_degrees = _validate_degrees_of_freedom(degrees_of_freedom)
    selection = select_causal_emission_rows(
        table,
        spec=spec,
        availability_column=availability_column,
        knowledge_cutoff=knowledge_cutoff,
    )
    training = selection.eligible
    if len(training) < spec.minimum_training_samples:
        raise ValueError(
            f"{spec.block_id} needs at least {spec.minimum_training_samples} "
            "complete causal rows"
        )
    estimate = _fit_fixed_hyperparameters(
        training,
        spec=spec,
        base_lambda=selected_lambda,
        degrees_of_freedom=selected_degrees,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        minimum_weight=minimum_weight,
    )
    fixed_selection = pd.DataFrame.from_records(
        [
            {
                "base_lambda": selected_lambda,
                "degrees_of_freedom": selected_degrees,
                "selected": True,
                "selection_method": "fixed_without_rolling_origin_retuning",
                "validation_performed": False,
            }
        ]
    )
    return _assemble_student_t_fit(
        spec=spec,
        training=training,
        selection_diagnostics=selection.diagnostics,
        knowledge_cutoff=selection.knowledge_cutoff,
        availability_column=availability_column,
        selected_lambda=selected_lambda,
        selected_degrees=selected_degrees,
        hyperparameter_selection=fixed_selection,
        hyperparameter_selection_method="fixed_without_rolling_origin_retuning",
        estimate=estimate,
    )


def fit_linear_student_t_emission(
    table: pd.DataFrame,
    *,
    spec: LinearGaussianEmissionSpec,
    availability_column: str,
    knowledge_cutoff: date | datetime | str | pd.Timestamp | None = None,
    degrees_of_freedom_grid: Sequence[float] = (7.0,),
    maximum_iterations: int = 100,
    tolerance: float = 1.0e-7,
    minimum_weight: float = 1.0e-8,
) -> LinearStudentTEmissionFit:
    """Causally tune and fit one robust Student-t release block.

    Pass ``degrees_of_freedom_grid=(7,)`` for the initial fixed-``nu`` model,
    or :data:`DEFAULT_DEGREES_OF_FREEDOM_GRID` for the requested heavy-tail
    sensitivity.  Ridge and tail parameters are selected jointly by the mean
    rolling-origin predictive negative log likelihood.
    """

    degrees_grid = _validate_degrees_grid(degrees_of_freedom_grid)
    selection = select_causal_emission_rows(
        table,
        spec=spec,
        availability_column=availability_column,
        knowledge_cutoff=knowledge_cutoff,
    )
    training = selection.eligible
    if len(training) < spec.minimum_training_samples:
        raise ValueError(
            f"{spec.block_id} needs at least {spec.minimum_training_samples} "
            "complete causal rows"
        )
    summaries = [
        _rolling_origin_score(
            training,
            spec=spec,
            availability_column=availability_column,
            base_lambda=base_lambda,
            degrees_of_freedom=degrees,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            minimum_weight=minimum_weight,
        )
        for degrees in degrees_grid
        for base_lambda in spec.lambda_grid
    ]
    hyperparameters = pd.DataFrame.from_records(summaries)
    enough = (
        hyperparameters["validation_observations"]
        >= spec.minimum_validation_observations
    )
    if not bool(enough.all()):
        raise ValueError(
            f"{spec.block_id} has too few rolling-origin validation observations"
        )
    losses = hyperparameters["mean_predictive_student_t_nll"].to_numpy(dtype=float)
    minimum = float(losses.min())
    tied = np.isclose(losses, minimum, rtol=1.0e-12, atol=1.0e-12)
    tied_rows = hyperparameters.loc[tied].copy()
    # When predictive scores are numerically identical, prefer stronger ridge
    # regularization and the less heavy-tailed (larger-nu) nested model.
    selected_index = tied_rows.sort_values(
        ["base_lambda", "degrees_of_freedom"], ascending=[False, False]
    ).index[0]
    selected_lambda = float(hyperparameters.loc[selected_index, "base_lambda"])
    selected_degrees = float(
        hyperparameters.loc[selected_index, "degrees_of_freedom"]
    )
    hyperparameters["selected"] = hyperparameters.index == selected_index
    estimate = _fit_fixed_hyperparameters(
        training,
        spec=spec,
        base_lambda=selected_lambda,
        degrees_of_freedom=selected_degrees,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        minimum_weight=minimum_weight,
    )
    return _assemble_student_t_fit(
        spec=spec,
        training=training,
        selection_diagnostics=selection.diagnostics,
        knowledge_cutoff=selection.knowledge_cutoff,
        availability_column=availability_column,
        selected_lambda=selected_lambda,
        selected_degrees=selected_degrees,
        hyperparameter_selection=hyperparameters.reset_index(drop=True),
        hyperparameter_selection_method=(
            "causal_rolling_origin_predictive_student_t_nll"
        ),
        estimate=estimate,
    )
