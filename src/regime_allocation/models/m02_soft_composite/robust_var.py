"""Robust VAR(1) sensitivity estimators for Model 02.

This module leaves the production OLS transition estimator unchanged and adds
two residual-robust sensitivity fits for the bivariate released score center
``Z_m = (G_m, I_m)``:

* a multivariate Huber iteratively reweighted least-squares (IRLS) fit; and
* a fixed-degrees-of-freedom multivariate Student-t IRLS fit.

Both fits include an intercept and apply one scalar weight to each monthly
growth/inflation innovation.  A common weight is important: independently
weighting the two response equations would no longer describe one coherent
bivariate innovation distribution.  The estimators operate only on the pairs
provided by the caller.  Causal availability filtering therefore remains the
caller's responsibility and can continue to use ``select_causal_var_pairs``.

The returned :class:`RobustVar1Fit` is a subclass of the existing
:class:`~regime_allocation.models.m02_soft_composite.var_transition.Var1Fit`.
Consequently, its intercept, transition matrix, and ``innovation_covariance``
can be passed directly to the existing Gaussian joint-state forecast.  For a
Student-t fit, IRLS estimates the distribution's scale matrix and the exposed
Gaussian plug-in covariance is ``nu / (nu - 2)`` times that scale matrix.  This
conversion requires ``nu > 2`` and deliberately retains a Gaussian state in
the first robust-transition sensitivity.

These are residual-robust estimators.  They reduce the influence of an unusual
response conditional on its lagged score, but they are not high-breakdown
estimators for extreme leverage points in the lagged predictors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import pandas as pd

from regime_allocation.models.m02_soft_composite.var_transition import (
    Var1Fit,
    fit_var1_ols,
)


_SOURCE_COLUMNS = ("source_growth_score", "source_inflation_score")
_DESTINATION_COLUMNS = (
    "destination_growth_score",
    "destination_inflation_score",
)
_REQUIRED_COLUMNS = frozenset((*_SOURCE_COLUMNS, *_DESTINATION_COLUMNS))
_RESPONSE_DIMENSION = 2


@dataclass(frozen=True)
class RobustVar1Fit(Var1Fit):
    """A VAR fit compatible with ``Var1Fit`` plus robust-fit diagnostics.

    ``event_weights`` and ``squared_mahalanobis_distances`` follow the input
    row order.  For Student-t fits, weights may exceed one for observations
    very near the fitted conditional mean.  ``downweighted_pairs`` counts
    weights strictly below one, not merely weights below the sample median.

    ``working_scale_matrix`` is the covariance-like matrix used to calculate
    IRLS Mahalanobis distances.  It equals ``innovation_covariance`` for the
    Huber and Gaussian fits.  For Student-t fits, the latter is the finite-
    variance Gaussian plug-in covariance and differs by
    ``gaussian_covariance_multiplier = nu / (nu - 2)``.
    """

    estimator: str
    tuning_parameter_name: str
    tuning_parameter: float
    converged: bool
    iterations: int
    maximum_iterations: int
    convergence_tolerance: float
    final_parameter_change: float
    event_weights: np.ndarray
    squared_mahalanobis_distances: np.ndarray
    working_scale_matrix: np.ndarray
    gaussian_covariance_multiplier: float
    effective_sample_size: float
    downweighted_pairs: int
    minimum_weight: float
    median_weight: float
    mean_weight: float
    maximum_weight: float
    covariance_floor_applications: int


def _validated_training_arrays(
    training_pairs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return finite source, destination, and full-rank intercept design."""

    if _REQUIRED_COLUMNS.difference(training_pairs.columns):
        raise ValueError("training pairs omit required score columns")
    source = training_pairs[list(_SOURCE_COLUMNS)].to_numpy(dtype=float)
    destination = training_pairs[list(_DESTINATION_COLUMNS)].to_numpy(dtype=float)
    if source.shape[0] < 5 or source.shape != destination.shape:
        raise ValueError("robust VAR(1) requires at least five bivariate pairs")
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("VAR(1) training scores must be finite")

    design = np.column_stack([np.ones(len(source)), source])
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError("VAR(1) design matrix is rank deficient")
    return source, destination, design, rank


def _validate_iteration_controls(
    *,
    maximum_iterations: int,
    tolerance: float,
    covariance_eigenvalue_floor: float,
) -> None:
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, (int, np.integer))
        or maximum_iterations < 1
    ):
        raise ValueError("maximum_iterations must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if (
        not math.isfinite(covariance_eigenvalue_floor)
        or covariance_eigenvalue_floor <= 0.0
    ):
        raise ValueError(
            "covariance_eigenvalue_floor must be finite and positive"
        )


def _floor_positive_definite(
    covariance: np.ndarray,
    *,
    relative_floor: float,
) -> tuple[np.ndarray, bool]:
    """Symmetrize and apply a scale-relative positive eigenvalue floor."""

    values = np.asarray(covariance, dtype=float)
    if values.shape != (_RESPONSE_DIMENSION, _RESPONSE_DIMENSION):
        raise ValueError("robust VAR covariance must be a 2x2 matrix")
    if not np.isfinite(values).all():
        raise ValueError("robust VAR covariance must be finite")
    symmetric = (values + values.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).eps)
    absolute_floor = relative_floor * scale
    clipped = np.maximum(eigenvalues, absolute_floor)
    floored = bool(np.any(clipped != eigenvalues))
    result = (eigenvectors * clipped) @ eigenvectors.T
    return (result + result.T) / 2.0, floored


def _mahalanobis_squared(
    residuals: np.ndarray,
    scale_matrix: np.ndarray,
) -> np.ndarray:
    solved = np.linalg.solve(scale_matrix, residuals.T).T
    distances = np.einsum("ij,ij->i", residuals, solved)
    # Tiny negative values can occur from floating-point roundoff.
    return np.maximum(distances, 0.0)


def _weighted_coefficients(
    design: np.ndarray,
    destination: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if weights.shape != (len(design),) or not np.isfinite(weights).all():
        raise ValueError("robust VAR weights must be one finite value per pair")
    if np.any(weights <= 0.0):
        raise ValueError("robust VAR weights must be strictly positive")
    square_root_weight = np.sqrt(weights)
    weighted_design = design * square_root_weight[:, None]
    weighted_destination = destination * square_root_weight[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design,
        weighted_destination,
        rcond=None,
    )
    if int(rank) != design.shape[1]:
        raise ValueError("weighted VAR(1) design matrix is rank deficient")
    return np.asarray(coefficients, dtype=float)


def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
    denominator = max(1.0, float(np.linalg.norm(old, ord="fro")))
    return float(np.linalg.norm(new - old, ord="fro") / denominator)


def _r_squared(destination: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    total_sums = np.square(destination - destination.mean(axis=0)).sum(axis=0)
    residual_sums = np.square(residuals).sum(axis=0)
    result = np.full(_RESPONSE_DIMENSION, np.nan, dtype=float)
    positive = total_sums > 0.0
    result[positive] = 1.0 - residual_sums[positive] / total_sums[positive]
    return result


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.square(weights).sum())
    if denominator <= 0.0:
        return 0.0
    return float(weights.sum() ** 2 / denominator)


def _robust_fit_result(
    *,
    design: np.ndarray,
    destination: np.ndarray,
    coefficients: np.ndarray,
    working_scale_matrix: np.ndarray,
    gaussian_covariance_multiplier: float,
    weights: np.ndarray,
    squared_mahalanobis_distances: np.ndarray,
    estimator: str,
    tuning_parameter_name: str,
    tuning_parameter: float,
    converged: bool,
    iterations: int,
    maximum_iterations: int,
    tolerance: float,
    final_parameter_change: float,
    covariance_floor_applications: int,
    rank: int,
) -> RobustVar1Fit:
    residuals = destination - design @ coefficients
    innovation_covariance, covariance_was_floored = _floor_positive_definite(
        working_scale_matrix * gaussian_covariance_multiplier,
        relative_floor=np.finfo(float).eps,
    )
    covariance_floor_applications += int(covariance_was_floored)
    intercept = np.asarray(coefficients[0], dtype=float)
    transition = np.asarray(coefficients[1:].T, dtype=float)
    eigenvalues = np.linalg.eigvals(transition)
    r_squared = _r_squared(destination, residuals)
    immutable_weights = np.asarray(weights, dtype=float).copy()
    immutable_distances = np.asarray(
        squared_mahalanobis_distances, dtype=float
    ).copy()
    immutable_scale = np.asarray(working_scale_matrix, dtype=float).copy()
    immutable_weights.setflags(write=False)
    immutable_distances.setflags(write=False)
    immutable_scale.setflags(write=False)

    return RobustVar1Fit(
        intercept=intercept,
        transition=transition,
        innovation_covariance=innovation_covariance,
        training_pairs=len(design),
        residual_degrees_of_freedom=len(design) - rank,
        design_rank=rank,
        design_condition_number=float(np.linalg.cond(design)),
        spectral_radius=float(np.max(np.abs(eigenvalues))),
        growth_r_squared=float(r_squared[0]),
        inflation_r_squared=float(r_squared[1]),
        residual_mean=np.asarray(residuals.mean(axis=0), dtype=float),
        estimator=estimator,
        tuning_parameter_name=tuning_parameter_name,
        tuning_parameter=float(tuning_parameter),
        converged=bool(converged),
        iterations=int(iterations),
        maximum_iterations=int(maximum_iterations),
        convergence_tolerance=float(tolerance),
        final_parameter_change=float(final_parameter_change),
        event_weights=immutable_weights,
        squared_mahalanobis_distances=immutable_distances,
        working_scale_matrix=immutable_scale,
        gaussian_covariance_multiplier=float(gaussian_covariance_multiplier),
        effective_sample_size=_effective_sample_size(immutable_weights),
        downweighted_pairs=int(np.sum(immutable_weights < 1.0 - 1.0e-12)),
        minimum_weight=float(np.min(immutable_weights)),
        median_weight=float(np.median(immutable_weights)),
        mean_weight=float(np.mean(immutable_weights)),
        maximum_weight=float(np.max(immutable_weights)),
        covariance_floor_applications=int(covariance_floor_applications),
    )


def _wrap_ols_fit(
    training_pairs: pd.DataFrame,
    *,
    maximum_iterations: int,
    tolerance: float,
) -> RobustVar1Fit:
    """Represent the unchanged OLS baseline in the robust audit contract."""

    source, destination, design, rank = _validated_training_arrays(training_pairs)
    del source
    fit = fit_var1_ols(training_pairs)
    coefficients = np.vstack([fit.intercept, fit.transition.T])
    residuals = destination - design @ coefficients
    distances = _mahalanobis_squared(residuals, fit.innovation_covariance)
    return _robust_fit_result(
        design=design,
        destination=destination,
        coefficients=coefficients,
        working_scale_matrix=fit.innovation_covariance,
        gaussian_covariance_multiplier=1.0,
        weights=np.ones(len(design), dtype=float),
        squared_mahalanobis_distances=distances,
        estimator="gaussian_ols",
        tuning_parameter_name="degrees_of_freedom",
        tuning_parameter=math.inf,
        converged=True,
        iterations=1,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        final_parameter_change=0.0,
        covariance_floor_applications=0,
        rank=rank,
    )


def fit_var1_huber(
    training_pairs: pd.DataFrame,
    *,
    threshold: float = 2.5,
    maximum_iterations: int = 100,
    tolerance: float = 1.0e-8,
    covariance_eigenvalue_floor: float = 1.0e-8,
) -> RobustVar1Fit:
    """Fit an intercept VAR(1) with multivariate Huber IRLS.

    If ``d_i`` is the residual Mahalanobis distance for pair ``i``, its IRLS
    weight is ``min(1, threshold / d_i)``.  The weighted innovation covariance
    uses the ordinary residual-degrees-of-freedom correction when every weight
    is one and a continuous weighted analogue otherwise.
    """

    _validate_iteration_controls(
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
    )
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("Huber threshold must be finite and positive")
    _, destination, design, rank = _validated_training_arrays(training_pairs)

    coefficients, _, _, _ = np.linalg.lstsq(design, destination, rcond=None)
    residuals = destination - design @ coefficients
    residual_dof = len(design) - rank
    scale_matrix, was_floored = _floor_positive_definite(
        residuals.T @ residuals / residual_dof,
        relative_floor=covariance_eigenvalue_floor,
    )
    floor_applications = int(was_floored)
    converged = False
    final_change = math.inf
    iterations = 0

    for iterations in range(1, maximum_iterations + 1):
        squared_distances = _mahalanobis_squared(residuals, scale_matrix)
        distances = np.sqrt(squared_distances)
        weights = np.ones(len(design), dtype=float)
        outside = distances > threshold
        weights[outside] = threshold / distances[outside]

        new_coefficients = _weighted_coefficients(design, destination, weights)
        new_residuals = destination - design @ new_coefficients
        # n/(n-rank) is the usual regression finite-sample correction.  Its
        # weighted analogue keeps the denominator positive even for severe
        # but nonzero Huber downweighting.
        weighted_denominator = float(weights.sum()) * residual_dof / len(design)
        new_scale, was_floored = _floor_positive_definite(
            new_residuals.T @ (weights[:, None] * new_residuals)
            / weighted_denominator,
            relative_floor=covariance_eigenvalue_floor,
        )
        floor_applications += int(was_floored)
        final_change = max(
            _relative_change(new_coefficients, coefficients),
            _relative_change(new_scale, scale_matrix),
        )
        coefficients = new_coefficients
        residuals = new_residuals
        scale_matrix = new_scale
        if final_change <= tolerance:
            converged = True
            break

    final_squared_distances = _mahalanobis_squared(residuals, scale_matrix)
    final_distances = np.sqrt(final_squared_distances)
    final_weights = np.ones(len(design), dtype=float)
    outside = final_distances > threshold
    final_weights[outside] = threshold / final_distances[outside]
    return _robust_fit_result(
        design=design,
        destination=destination,
        coefficients=coefficients,
        working_scale_matrix=scale_matrix,
        gaussian_covariance_multiplier=1.0,
        weights=final_weights,
        squared_mahalanobis_distances=final_squared_distances,
        estimator="huber_irls",
        tuning_parameter_name="huber_threshold",
        tuning_parameter=threshold,
        converged=converged,
        iterations=iterations,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        final_parameter_change=final_change,
        covariance_floor_applications=floor_applications,
        rank=rank,
    )


def fit_var1_student_t(
    training_pairs: pd.DataFrame,
    *,
    degrees_of_freedom: float = 7.0,
    maximum_iterations: int = 100,
    tolerance: float = 1.0e-8,
    covariance_eigenvalue_floor: float = 1.0e-8,
) -> RobustVar1Fit:
    """Fit a fixed-``nu`` multivariate Student-t intercept VAR(1) by IRLS.

    For response dimension ``p=2`` and squared residual Mahalanobis distance
    ``delta_i``, the scale-mixture conditional weight is

    ``w_i = (nu + p) / (nu + delta_i)``.

    ``degrees_of_freedom=inf`` deliberately returns the unchanged Gaussian OLS
    baseline in the same audit-rich return type.  Finite ``nu`` must exceed two
    because the existing joint filter consumes a finite Gaussian covariance.
    """

    _validate_iteration_controls(
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        covariance_eigenvalue_floor=covariance_eigenvalue_floor,
    )
    if math.isinf(degrees_of_freedom):
        if degrees_of_freedom < 0.0:
            raise ValueError("Student-t degrees_of_freedom must exceed two")
        return _wrap_ols_fit(
            training_pairs,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
    if not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 2.0:
        raise ValueError("Student-t degrees_of_freedom must exceed two")
    _, destination, design, rank = _validated_training_arrays(training_pairs)

    coefficients, _, _, _ = np.linalg.lstsq(design, destination, rcond=None)
    residuals = destination - design @ coefficients
    # Initialize the t scale so its implied covariance matches the OLS
    # covariance.  Subsequent scale updates are the standard t-regression
    # scale-mixture M step (weighted scatter divided by n).
    gaussian_initial = residuals.T @ residuals / (len(design) - rank)
    covariance_multiplier = degrees_of_freedom / (degrees_of_freedom - 2.0)
    scale_matrix, was_floored = _floor_positive_definite(
        gaussian_initial / covariance_multiplier,
        relative_floor=covariance_eigenvalue_floor,
    )
    floor_applications = int(was_floored)
    converged = False
    final_change = math.inf
    iterations = 0

    for iterations in range(1, maximum_iterations + 1):
        squared_distances = _mahalanobis_squared(residuals, scale_matrix)
        weights = (degrees_of_freedom + _RESPONSE_DIMENSION) / (
            degrees_of_freedom + squared_distances
        )
        new_coefficients = _weighted_coefficients(design, destination, weights)
        new_residuals = destination - design @ new_coefficients
        new_scale, was_floored = _floor_positive_definite(
            new_residuals.T @ (weights[:, None] * new_residuals) / len(design),
            relative_floor=covariance_eigenvalue_floor,
        )
        floor_applications += int(was_floored)
        final_change = max(
            _relative_change(new_coefficients, coefficients),
            _relative_change(new_scale, scale_matrix),
        )
        coefficients = new_coefficients
        residuals = new_residuals
        scale_matrix = new_scale
        if final_change <= tolerance:
            converged = True
            break

    final_squared_distances = _mahalanobis_squared(residuals, scale_matrix)
    final_weights = (degrees_of_freedom + _RESPONSE_DIMENSION) / (
        degrees_of_freedom + final_squared_distances
    )
    return _robust_fit_result(
        design=design,
        destination=destination,
        coefficients=coefficients,
        working_scale_matrix=scale_matrix,
        gaussian_covariance_multiplier=covariance_multiplier,
        weights=final_weights,
        squared_mahalanobis_distances=final_squared_distances,
        estimator="student_t_irls",
        tuning_parameter_name="degrees_of_freedom",
        tuning_parameter=degrees_of_freedom,
        converged=converged,
        iterations=iterations,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
        final_parameter_change=final_change,
        covariance_floor_applications=floor_applications,
        rank=rank,
    )


def robust_var1_weight_audit(
    training_pairs: pd.DataFrame,
    fit: RobustVar1Fit,
) -> pd.DataFrame:
    """Return input-pair lineage, fitted innovations, distances, and weights."""

    if not isinstance(fit, RobustVar1Fit):
        raise TypeError("fit must be a RobustVar1Fit")
    _, destination, design, _ = _validated_training_arrays(training_pairs)
    if fit.training_pairs != len(design):
        raise ValueError("fit and training-pair row counts disagree")
    audit_columns = {
        "var_training_row",
        "var_estimator",
        "fitted_destination_growth_score",
        "fitted_destination_inflation_score",
        "growth_innovation",
        "inflation_innovation",
        "squared_mahalanobis_distance",
        "event_weight",
        "is_downweighted",
    }
    collisions = audit_columns.intersection(training_pairs.columns)
    if collisions:
        raise ValueError(
            "training pairs already contain robust audit columns: "
            + ", ".join(sorted(collisions))
        )
    coefficients = np.vstack([fit.intercept, fit.transition.T])
    fitted = design @ coefficients
    residuals = destination - fitted
    audit = training_pairs.reset_index(drop=True).copy()
    audit.insert(0, "var_training_row", np.arange(len(audit), dtype=int))
    audit["var_estimator"] = fit.estimator
    audit["fitted_destination_growth_score"] = fitted[:, 0]
    audit["fitted_destination_inflation_score"] = fitted[:, 1]
    audit["growth_innovation"] = residuals[:, 0]
    audit["inflation_innovation"] = residuals[:, 1]
    audit["squared_mahalanobis_distance"] = (
        fit.squared_mahalanobis_distances
    )
    audit["event_weight"] = fit.event_weights
    audit["is_downweighted"] = fit.event_weights < 1.0 - 1.0e-12
    return audit


def fit_var1_sensitivity(
    training_pairs: pd.DataFrame,
    *,
    estimator: Literal["ols", "huber", "student_t"],
    huber_threshold: float = 2.5,
    degrees_of_freedom: float = 7.0,
    maximum_iterations: int = 100,
    tolerance: float = 1.0e-8,
    covariance_eigenvalue_floor: float = 1.0e-8,
) -> RobustVar1Fit:
    """Dispatch one OLS, Huber, or Student-t VAR sensitivity fit."""

    if estimator == "ols":
        _validate_iteration_controls(
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        )
        return _wrap_ols_fit(
            training_pairs,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
    if estimator == "huber":
        return fit_var1_huber(
            training_pairs,
            threshold=huber_threshold,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        )
    if estimator == "student_t":
        return fit_var1_student_t(
            training_pairs,
            degrees_of_freedom=degrees_of_freedom,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            covariance_eigenvalue_floor=covariance_eigenvalue_floor,
        )
    raise ValueError("estimator must be one of: ols, huber, student_t")
