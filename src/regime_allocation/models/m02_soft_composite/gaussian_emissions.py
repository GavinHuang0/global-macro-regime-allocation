"""Estimate causal linear-Gaussian release-block emissions for Model 02.

The observation equation for a release block is

``y[e] = a + H z[q(e)] + C v[e] + epsilon[e]``

where ``z`` is the two-score growth/inflation state, ``v`` contains optional
observed controls, and the block residual is multivariate Gaussian.  Each
response has its own structured ridge penalties on the state loadings.  The
intercept and controls are never penalized, and selected state loadings may be
fixed exactly to zero.

The base ridge penalty is selected from a frozen grid with rolling-origin
predictive Gaussian negative log likelihood.  Rows sharing an availability
timestamp are held out together, so same-time observations cannot train one
another.  Final residual covariance uses Ledoit-Wolf shrinkage followed by an
explicit eigenvalue floor.  The module performs no file or network access.
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


DEFAULT_LAMBDA_GRID = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)


def _names(values: Sequence[str], *, label: str, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of column names")
    result = tuple(str(value).strip() for value in values)
    if (not result and not allow_empty) or any(not value for value in result):
        raise ValueError(f"{label} contains an empty name")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate names")
    return result


def _matrix_tuple(
    values: Sequence[Sequence[float | bool]] | np.ndarray,
    *,
    shape: tuple[int, int],
    dtype: type[float] | type[bool],
    default: float | bool,
    label: str,
) -> tuple[tuple[float | bool, ...], ...]:
    if len(values) == 0:
        array = np.full(shape, default, dtype=dtype)
    else:
        array = np.asarray(values, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if dtype is float and (not np.isfinite(array).all() or (array < 0.0).any()):
        raise ValueError(f"{label} must contain finite nonnegative values")
    return tuple(tuple(value.item() for value in row) for row in array)


@dataclass(frozen=True)
class LinearGaussianEmissionSpec:
    """Frozen structural and tuning specification for one release block."""

    block_id: str
    response_names: tuple[str, ...]
    state_names: tuple[str, ...] = ("growth_score", "inflation_score")
    control_names: tuple[str, ...] = ()
    state_loading_penalties: tuple[tuple[float, ...], ...] = ()
    exact_zero_mask: tuple[tuple[bool, ...], ...] = ()
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    minimum_training_samples: int = 24
    validation_minimum_training_samples: int = 18
    minimum_validation_observations: int = 6
    covariance_eigenvalue_floor: float = 1.0e-8

    def __post_init__(self) -> None:
        block_id = str(self.block_id).strip()
        if not block_id:
            raise ValueError("block_id cannot be empty")
        responses = _names(self.response_names, label="response_names")
        states = _names(self.state_names, label="state_names")
        controls = _names(self.control_names, label="control_names", allow_empty=True)
        all_names = responses + states + controls
        if len(all_names) != len(set(all_names)):
            raise ValueError("response, state, and control names must be disjoint")
        if len(states) != 2:
            raise ValueError("Model 02 emissions require growth and inflation states")
        shape = (len(responses), len(states))
        penalties = _matrix_tuple(
            self.state_loading_penalties,
            shape=shape,
            dtype=float,
            default=1.0,
            label="state_loading_penalties",
        )
        restrictions = _matrix_tuple(
            self.exact_zero_mask,
            shape=shape,
            dtype=bool,
            default=False,
            label="exact_zero_mask",
        )
        grid = tuple(float(value) for value in self.lambda_grid)
        if (
            not grid
            or not np.isfinite(grid).all()
            or any(value < 0.0 for value in grid)
            or any(right <= left for left, right in zip(grid, grid[1:]))
        ):
            raise ValueError("lambda_grid must be finite, nonnegative, and increasing")
        if self.minimum_training_samples < 3:
            raise ValueError("minimum_training_samples must be at least three")
        if self.validation_minimum_training_samples < 3:
            raise ValueError(
                "validation_minimum_training_samples must be at least three"
            )
        if self.minimum_validation_observations < 1:
            raise ValueError("minimum_validation_observations must be positive")
        unpenalized_dimension = 1 + len(controls)
        if self.minimum_training_samples <= unpenalized_dimension:
            raise ValueError(
                "minimum_training_samples must exceed the intercept/control dimension"
            )
        if self.validation_minimum_training_samples <= unpenalized_dimension:
            raise ValueError(
                "validation_minimum_training_samples must exceed the "
                "intercept/control dimension"
            )
        if (
            not math.isfinite(float(self.covariance_eigenvalue_floor))
            or self.covariance_eigenvalue_floor <= 0.0
        ):
            raise ValueError("covariance_eigenvalue_floor must be positive")
        object.__setattr__(self, "block_id", block_id)
        object.__setattr__(self, "response_names", responses)
        object.__setattr__(self, "state_names", states)
        object.__setattr__(self, "control_names", controls)
        object.__setattr__(self, "state_loading_penalties", penalties)
        object.__setattr__(self, "exact_zero_mask", restrictions)
        object.__setattr__(self, "lambda_grid", grid)

    def to_audit_dict(self) -> dict[str, Any]:
        """Return a JSON-safe version of the frozen specification."""

        return {
            "block_id": self.block_id,
            "response_names": list(self.response_names),
            "state_names": list(self.state_names),
            "control_names": list(self.control_names),
            "state_loading_penalties": [
                list(row) for row in self.state_loading_penalties
            ],
            "exact_zero_mask": [list(row) for row in self.exact_zero_mask],
            "lambda_grid": list(self.lambda_grid),
            "minimum_training_samples": int(self.minimum_training_samples),
            "validation_minimum_training_samples": int(
                self.validation_minimum_training_samples
            ),
            "minimum_validation_observations": int(
                self.minimum_validation_observations
            ),
            "covariance_eigenvalue_floor": float(
                self.covariance_eigenvalue_floor
            ),
        }


@dataclass(frozen=True)
class CausalEmissionSelection:
    """Complete rows knowable strictly before one information cutoff."""

    eligible: pd.DataFrame
    audit: pd.DataFrame
    knowledge_cutoff: pd.Timestamp | None
    availability_column: str
    diagnostics: dict[str, int]


def _timestamp(value: date | datetime | str | pd.Timestamp, *, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{label} must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def select_causal_emission_rows(
    table: pd.DataFrame,
    *,
    spec: LinearGaussianEmissionSpec,
    availability_column: str,
    knowledge_cutoff: date | datetime | str | pd.Timestamp | None = None,
) -> CausalEmissionSelection:
    """Select complete rows using the strict causal availability rule.

    ``availability_column`` must already represent the later of release
    availability and target-score availability.  When a cutoff is supplied,
    rows available exactly at the cutoff are excluded because the repository's
    point-in-time contract cannot assume a within-timestamp ordering.
    """

    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    required_numeric = (
        spec.response_names + spec.state_names + spec.control_names
    )
    required = set(required_numeric).union({availability_column})
    missing = required.difference(table.columns)
    if missing:
        raise ValueError("emission table omits columns: " + ", ".join(sorted(missing)))
    audit = table.copy()
    raw_availability = audit[availability_column]
    parsed = pd.to_datetime(raw_availability, errors="coerce", utc=True)
    if (raw_availability.notna() & parsed.isna()).any() or parsed.isna().any():
        raise ValueError(f"{availability_column} contains an invalid timestamp")
    audit[availability_column] = parsed.dt.tz_convert(None)
    try:
        numeric = audit.loc[:, required_numeric].apply(
            pd.to_numeric, errors="raise"
        ).astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("emission variables must be numeric") from error
    audit.loc[:, required_numeric] = numeric
    complete = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    cutoff = (
        None if knowledge_cutoff is None else _timestamp(knowledge_cutoff, label="knowledge_cutoff")
    )
    before_cutoff = np.ones(len(audit), dtype=bool)
    if cutoff is not None:
        before_cutoff = (
            audit[availability_column].to_numpy(dtype="datetime64[ns]")
            < cutoff.to_datetime64()
        )
    reasons = np.full(len(audit), "", dtype=object)
    reasons[~complete] = "incomplete_model_vector"
    reasons[complete & ~before_cutoff] = "not_available_strictly_before_cutoff"
    eligible_mask = reasons == ""
    audit["emission_training_eligible"] = eligible_mask
    audit["emission_training_exclusion_reason"] = reasons
    audit["_emission_original_position"] = np.arange(len(audit))
    audit = audit.sort_values(
        [availability_column, "_emission_original_position"], kind="mergesort"
    ).reset_index(drop=True)
    eligible = audit.loc[audit["emission_training_eligible"]].copy().reset_index(
        drop=True
    )
    counts = audit["emission_training_exclusion_reason"].value_counts()
    diagnostics = {
        "input_rows": int(len(audit)),
        "eligible_rows": int(len(eligible)),
        "excluded_incomplete_model_vector": int(
            counts.get("incomplete_model_vector", 0)
        ),
        "excluded_not_available_strictly_before_cutoff": int(
            counts.get("not_available_strictly_before_cutoff", 0)
        ),
    }
    return CausalEmissionSelection(
        eligible=eligible,
        audit=audit,
        knowledge_cutoff=cutoff,
        availability_column=availability_column,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class _FixedLambdaEstimate:
    intercept: np.ndarray
    state_loadings: np.ndarray
    control_loadings: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    ledoit_wolf_shrinkage: float
    covariance_minimum_eigenvalue_before_floor: float
    covariance_minimum_eigenvalue_after_floor: float
    active_control_mask: np.ndarray
    dropped_constant_controls: tuple[str, ...]


def _floor_covariance(covariance: np.ndarray, floor: float) -> tuple[np.ndarray, float, float]:
    symmetric = (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    before = float(eigenvalues.min())
    floored = np.maximum(eigenvalues, floor)
    result = (eigenvectors * floored) @ eigenvectors.T
    result = (result + result.T) / 2.0
    return result, before, float(np.linalg.eigvalsh(result).min())


def _identifiable_control_mask(controls: np.ndarray) -> np.ndarray:
    """Return controls with variation identifiable from the current training fold.

    A control that is constant in a historical fold is collinear with the
    intercept.  It must be disabled for that fold rather than estimated using
    observations that arrive later.  The mask is recomputed independently for
    every rolling-origin fit, so an indicator such as a methodology-break dummy
    activates automatically once both of its values have appeared in training.
    """

    if controls.ndim != 2:
        raise ValueError("controls must be a two-dimensional matrix")
    if controls.shape[1] == 0:
        return np.zeros(0, dtype=bool)
    minimum = controls.min(axis=0)
    maximum = controls.max(axis=0)
    scale = np.maximum(1.0, np.maximum(np.abs(minimum), np.abs(maximum)))
    tolerance = 1.0e-12 * scale
    return (maximum - minimum) > tolerance


def _fit_fixed_lambda(
    table: pd.DataFrame,
    *,
    spec: LinearGaussianEmissionSpec,
    base_lambda: float,
) -> _FixedLambdaEstimate:
    responses = table.loc[:, spec.response_names].to_numpy(dtype=float)
    states = table.loc[:, spec.state_names].to_numpy(dtype=float)
    controls = table.loc[:, spec.control_names].to_numpy(dtype=float)
    observations = len(table)
    output_count = len(spec.response_names)
    state_count = len(spec.state_names)
    control_count = len(spec.control_names)
    intercept = np.zeros(output_count, dtype=float)
    state_loadings = np.zeros((output_count, state_count), dtype=float)
    control_loadings = np.zeros((output_count, control_count), dtype=float)
    penalties = np.asarray(spec.state_loading_penalties, dtype=float)
    restrictions = np.asarray(spec.exact_zero_mask, dtype=bool)
    active_controls = _identifiable_control_mask(controls)
    dropped_controls = tuple(
        name
        for name, is_active in zip(spec.control_names, active_controls)
        if not is_active
    )

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
        allowed_penalties = penalties[output, allowed]
        diagonal[1 : 1 + int(allowed.sum())] = base_lambda * allowed_penalties
        unpenalized = diagonal == 0.0
        if np.linalg.matrix_rank(design[:, unpenalized]) != int(unpenalized.sum()):
            raise ValueError(
                f"unpenalized design is rank deficient for {spec.response_names[output]}"
            )
        system = design.T @ design + np.diag(diagonal)
        target = design.T @ responses[:, output]
        coefficients = np.linalg.solve(system, target)
        intercept[output] = coefficients[0]
        state_loadings[output, allowed] = coefficients[
            1 : 1 + int(allowed.sum())
        ]
        active_control_count = int(active_controls.sum())
        if active_control_count:
            control_loadings[output, active_controls] = coefficients[
                -active_control_count:
            ]

    fitted = (
        intercept[None, :]
        + states @ state_loadings.T
        + controls @ control_loadings.T
    )
    residuals = responses - fitted
    estimator = LedoitWolf(assume_centered=True, store_precision=False).fit(
        residuals
    )
    covariance, before, after = _floor_covariance(
        np.asarray(estimator.covariance_, dtype=float),
        spec.covariance_eigenvalue_floor,
    )
    return _FixedLambdaEstimate(
        intercept=intercept,
        state_loadings=state_loadings,
        control_loadings=control_loadings,
        covariance=covariance,
        residuals=residuals,
        ledoit_wolf_shrinkage=float(estimator.shrinkage_),
        covariance_minimum_eigenvalue_before_floor=before,
        covariance_minimum_eigenvalue_after_floor=after,
        active_control_mask=active_controls,
        dropped_constant_controls=dropped_controls,
    )


def _gaussian_nll(residuals: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0 or not math.isfinite(float(log_determinant)):
        raise RuntimeError("emission covariance is not positive definite")
    solved = np.linalg.solve(covariance, residuals.T).T
    mahalanobis = np.sum(residuals * solved, axis=1)
    dimension = residuals.shape[1]
    return 0.5 * (
        dimension * math.log(2.0 * math.pi)
        + float(log_determinant)
        + mahalanobis
    )


def _predict(
    estimate: _FixedLambdaEstimate,
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


def _rolling_scores(
    table: pd.DataFrame,
    *,
    spec: LinearGaussianEmissionSpec,
    availability_column: str,
    base_lambda: float,
    return_residuals: bool,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    nll_parts: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    origin_count = 0
    unique_times = table[availability_column].drop_duplicates().sort_values()
    for origin in unique_times:
        training = table[table[availability_column] < origin]
        validation = table[table[availability_column] == origin]
        if len(training) < spec.validation_minimum_training_samples:
            continue
        estimate = _fit_fixed_lambda(
            training, spec=spec, base_lambda=base_lambda
        )
        actual = validation.loc[:, spec.response_names].to_numpy(dtype=float)
        residuals = actual - _predict(estimate, validation, spec)
        nll = _gaussian_nll(residuals, estimate.covariance)
        predictive_marginal_sd = np.sqrt(np.diag(estimate.covariance))
        standardized = residuals / predictive_marginal_sd[None, :]
        cholesky = np.linalg.cholesky(estimate.covariance)
        whitened = np.linalg.solve(cholesky, residuals.T).T
        nll_parts.append(nll)
        origin_count += 1
        if return_residuals:
            for row_position, (_, row) in enumerate(validation.iterrows()):
                record: dict[str, object] = {
                    "validation_available_at": pd.Timestamp(origin),
                    "training_rows": int(len(training)),
                    "base_lambda": float(base_lambda),
                    "predictive_gaussian_nll": float(nll[row_position]),
                    "source_index": row.name,
                    "dropped_constant_controls": "|".join(
                        estimate.dropped_constant_controls
                    ),
                }
                for output, name in enumerate(spec.response_names):
                    record[f"actual_{name}"] = float(actual[row_position, output])
                    record[f"residual_{name}"] = float(residuals[row_position, output])
                    record[f"predictive_marginal_sd_{name}"] = float(
                        predictive_marginal_sd[output]
                    )
                    record[f"standardized_residual_{name}"] = float(
                        standardized[row_position, output]
                    )
                    record[f"cholesky_whitened_residual_{name}"] = float(
                        whitened[row_position, output]
                    )
                records.append(record)
    values = np.concatenate(nll_parts) if nll_parts else np.empty(0, dtype=float)
    summary: dict[str, float | int] = {
        "base_lambda": float(base_lambda),
        "validation_origins": int(origin_count),
        "validation_observations": int(len(values)),
        "total_predictive_gaussian_nll": (
            float(values.sum()) if len(values) else math.nan
        ),
        "mean_predictive_gaussian_nll": (
            float(values.mean()) if len(values) else math.nan
        ),
    }
    return summary, pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class ObservedEmissionSystem:
    """Observed release subvector and its exact linear-Gaussian system."""

    observed_names: tuple[str, ...]
    omitted_names: tuple[str, ...]
    observation: np.ndarray
    intercept: np.ndarray
    state_loadings: np.ndarray
    control_contribution: np.ndarray
    residual_covariance: np.ndarray

    @property
    def adjusted_observation(self) -> np.ndarray:
        """Observation after subtracting intercept and known controls."""

        return self.observation - self.intercept - self.control_contribution

    def conditional_log_likelihood(
        self, state: Sequence[float] | np.ndarray
    ) -> float:
        """Return ``log p(y_observed | state, controls)`` for this submodel."""

        values = np.asarray(state, dtype=float)
        if values.shape != (self.state_loadings.shape[1],) or not np.isfinite(
            values
        ).all():
            raise ValueError("state has an invalid shape or value")
        residual = self.adjusted_observation - self.state_loadings @ values
        return -float(
            _gaussian_nll(residual[None, :], self.residual_covariance)[0]
        )

    def to_joint_filter_inputs(self) -> dict[str, np.ndarray]:
        """Return arrays accepted directly by ``update_joint_gaussian``.

        The joint filter expects an unadjusted observation and a single offset.
        Consequently its intercept argument is ``a + C v`` rather than this
        object's fully adjusted observation ``y - a - C v``.
        """

        return {
            "observation": self.observation.copy(),
            "loading": self.state_loadings.copy(),
            "noise_covariance": self.residual_covariance.copy(),
            "intercept": (self.intercept + self.control_contribution).copy(),
        }


@dataclass(frozen=True)
class GaussianStateUpdate:
    """One analytically integrated linear-Gaussian state update."""

    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    predictive_mean: np.ndarray
    predictive_covariance: np.ndarray
    innovation: np.ndarray
    kalman_gain: np.ndarray
    log_predictive_density: float
    observed_names: tuple[str, ...]
    omitted_names: tuple[str, ...]

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe event-update metadata."""

        return {
            "observed_names": list(self.observed_names),
            "omitted_names": list(self.omitted_names),
            "prior_mean": self.prior_mean.tolist(),
            "prior_covariance": self.prior_covariance.tolist(),
            "predictive_mean": self.predictive_mean.tolist(),
            "predictive_covariance": self.predictive_covariance.tolist(),
            "innovation": self.innovation.tolist(),
            "kalman_gain": self.kalman_gain.tolist(),
            "posterior_mean": self.posterior_mean.tolist(),
            "posterior_covariance": self.posterior_covariance.tolist(),
            "log_predictive_density": float(self.log_predictive_density),
        }


@dataclass(frozen=True)
class LinearGaussianEmissionFit:
    """Selected and refitted linear-Gaussian release-block emission."""

    spec: LinearGaussianEmissionSpec
    intercept: pd.Series
    state_loadings: pd.DataFrame
    control_loadings: pd.DataFrame
    residual_covariance: pd.DataFrame
    selected_base_lambda: float
    lambda_selection: pd.DataFrame
    training_count: int
    knowledge_cutoff: pd.Timestamp | None
    first_training_available_at: pd.Timestamp
    last_training_available_at: pd.Timestamp
    selection_diagnostics: dict[str, int]
    ledoit_wolf_shrinkage: float
    covariance_minimum_eigenvalue_before_floor: float
    covariance_minimum_eigenvalue_after_floor: float
    dropped_constant_controls: tuple[str, ...]
    residual_mean: pd.Series
    residual_rmse: pd.Series

    def observed_system(
        self,
        observation: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray,
        *,
        controls: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray | None = None,
    ) -> ObservedEmissionSystem:
        """Return the exact submodel for whichever block features are observed."""

        if isinstance(observation, (Mapping, pd.Series)):
            supplied = dict(observation)
            unknown = set(supplied).difference(self.spec.response_names)
            if unknown:
                raise ValueError(f"observation contains unknown responses: {sorted(unknown)}")
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
            observed = tuple(
                name
                for name, value in zip(self.spec.response_names, full)
                if np.isfinite(value)
            )
            values = full[np.isfinite(full)]
        if not observed or not np.isfinite(values).all():
            raise ValueError("at least one finite response must be observed")
        omitted = tuple(name for name in self.spec.response_names if name not in observed)
        control_values = _control_array(controls, self.spec.control_names)
        return ObservedEmissionSystem(
            observed_names=observed,
            omitted_names=omitted,
            observation=values,
            intercept=self.intercept.loc[list(observed)].to_numpy(dtype=float),
            state_loadings=self.state_loadings.loc[list(observed)].to_numpy(dtype=float),
            control_contribution=(
                self.control_loadings.loc[list(observed)].to_numpy(dtype=float)
                @ control_values
            ),
            residual_covariance=self.residual_covariance.loc[
                list(observed), list(observed)
            ].to_numpy(dtype=float),
        )

    def update_gaussian_state(
        self,
        prior_mean: Sequence[float] | np.ndarray,
        prior_covariance: Sequence[Sequence[float]] | np.ndarray,
        observation: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray,
        *,
        controls: Mapping[str, float] | pd.Series | Sequence[float] | np.ndarray | None = None,
    ) -> GaussianStateUpdate:
        """Condition a Gaussian state on a complete or partial release block."""

        mean = np.asarray(prior_mean, dtype=float)
        covariance = np.asarray(prior_covariance, dtype=float)
        if mean.shape != (len(self.spec.state_names),) or not np.isfinite(mean).all():
            raise ValueError("prior_mean has an invalid shape or value")
        if covariance.shape != (len(mean), len(mean)) or not np.isfinite(covariance).all():
            raise ValueError("prior_covariance has an invalid shape or value")
        covariance = (covariance + covariance.T) / 2.0
        if float(np.linalg.eigvalsh(covariance).min()) <= 0.0:
            raise ValueError("prior_covariance must be positive definite")
        system = self.observed_system(observation, controls=controls)
        h = system.state_loadings
        predictive_mean = (
            system.intercept + system.control_contribution + h @ mean
        )
        predictive_covariance = h @ covariance @ h.T + system.residual_covariance
        predictive_covariance = (predictive_covariance + predictive_covariance.T) / 2.0
        innovation = system.observation - predictive_mean
        gain = np.linalg.solve(predictive_covariance, h @ covariance).T
        posterior_mean = mean + gain @ innovation
        identity_minus_kh = np.eye(len(mean)) - gain @ h
        posterior_covariance = (
            identity_minus_kh @ covariance @ identity_minus_kh.T
            + gain @ system.residual_covariance @ gain.T
        )
        posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2.0
        nll = float(_gaussian_nll(innovation[None, :], predictive_covariance)[0])
        return GaussianStateUpdate(
            prior_mean=mean,
            prior_covariance=covariance,
            posterior_mean=posterior_mean,
            posterior_covariance=posterior_covariance,
            predictive_mean=predictive_mean,
            predictive_covariance=predictive_covariance,
            innovation=innovation,
            kalman_gain=gain,
            log_predictive_density=-nll,
            observed_names=system.observed_names,
            omitted_names=system.omitted_names,
        )

    def to_audit_dict(self) -> dict[str, Any]:
        """Return complete JSON-safe fit and tuning metadata."""

        def optional_timestamp(value: pd.Timestamp | None) -> str | None:
            return None if value is None else value.isoformat()

        return {
            "model_equation": "y = intercept + state_loadings @ z + control_loadings @ v + epsilon",
            "distribution": "multivariate_gaussian",
            "specification": self.spec.to_audit_dict(),
            "training_count": int(self.training_count),
            "knowledge_cutoff": optional_timestamp(self.knowledge_cutoff),
            "first_training_available_at": self.first_training_available_at.isoformat(),
            "last_training_available_at": self.last_training_available_at.isoformat(),
            "strict_cutoff_rule": "training_available_at < knowledge_cutoff",
            "rolling_origin_rule": "equal availability timestamps held out together",
            "selected_base_lambda": float(self.selected_base_lambda),
            "lambda_selection": self.lambda_selection.to_dict(orient="records"),
            "intercept": self.intercept.to_dict(),
            "state_loadings": self.state_loadings.to_dict(orient="index"),
            "control_loadings": self.control_loadings.to_dict(orient="index"),
            "residual_covariance": self.residual_covariance.to_dict(orient="index"),
            "ledoit_wolf_shrinkage": float(self.ledoit_wolf_shrinkage),
            "covariance_minimum_eigenvalue_before_floor": float(
                self.covariance_minimum_eigenvalue_before_floor
            ),
            "covariance_minimum_eigenvalue_after_floor": float(
                self.covariance_minimum_eigenvalue_after_floor
            ),
            "dropped_constant_controls": list(self.dropped_constant_controls),
            "residual_mean": self.residual_mean.to_dict(),
            "residual_rmse": self.residual_rmse.to_dict(),
            "selection_diagnostics": {
                key: int(value) for key, value in self.selection_diagnostics.items()
            },
        }


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


def fit_linear_gaussian_emission(
    table: pd.DataFrame,
    *,
    spec: LinearGaussianEmissionSpec,
    availability_column: str,
    knowledge_cutoff: date | datetime | str | pd.Timestamp | None = None,
) -> LinearGaussianEmissionFit:
    """Select causally available rows, tune ridge, and fit one emission block."""

    selection = select_causal_emission_rows(
        table,
        spec=spec,
        availability_column=availability_column,
        knowledge_cutoff=knowledge_cutoff,
    )
    training = selection.eligible
    if len(training) < spec.minimum_training_samples:
        raise ValueError(
            f"{spec.block_id} needs at least {spec.minimum_training_samples} complete causal rows"
        )
    summaries = []
    for candidate in spec.lambda_grid:
        summary, _ = _rolling_scores(
            training,
            spec=spec,
            availability_column=availability_column,
            base_lambda=candidate,
            return_residuals=False,
        )
        summaries.append(summary)
    lambda_selection = pd.DataFrame.from_records(summaries)
    enough = (
        lambda_selection["validation_observations"]
        >= spec.minimum_validation_observations
    )
    if not bool(enough.all()):
        raise ValueError(
            f"{spec.block_id} has too few rolling-origin validation observations"
        )
    losses = lambda_selection["mean_predictive_gaussian_nll"].to_numpy(dtype=float)
    minimum = float(losses.min())
    tied = np.isclose(losses, minimum, rtol=1.0e-12, atol=1.0e-12)
    selected_lambda = float(
        lambda_selection.loc[tied, "base_lambda"].max()
    )
    lambda_selection["selected"] = lambda_selection["base_lambda"].eq(
        selected_lambda
    )
    estimate = _fit_fixed_lambda(
        training, spec=spec, base_lambda=selected_lambda
    )
    residual_mean = estimate.residuals.mean(axis=0)
    residual_rmse = np.sqrt(np.square(estimate.residuals).mean(axis=0))
    return LinearGaussianEmissionFit(
        spec=spec,
        intercept=pd.Series(estimate.intercept, index=spec.response_names),
        state_loadings=pd.DataFrame(
            estimate.state_loadings,
            index=spec.response_names,
            columns=spec.state_names,
        ),
        control_loadings=pd.DataFrame(
            estimate.control_loadings,
            index=spec.response_names,
            columns=spec.control_names,
        ),
        residual_covariance=pd.DataFrame(
            estimate.covariance,
            index=spec.response_names,
            columns=spec.response_names,
        ),
        selected_base_lambda=selected_lambda,
        lambda_selection=lambda_selection,
        training_count=len(training),
        knowledge_cutoff=selection.knowledge_cutoff,
        first_training_available_at=pd.Timestamp(training[availability_column].min()),
        last_training_available_at=pd.Timestamp(training[availability_column].max()),
        selection_diagnostics=selection.diagnostics,
        ledoit_wolf_shrinkage=estimate.ledoit_wolf_shrinkage,
        covariance_minimum_eigenvalue_before_floor=(
            estimate.covariance_minimum_eigenvalue_before_floor
        ),
        covariance_minimum_eigenvalue_after_floor=(
            estimate.covariance_minimum_eigenvalue_after_floor
        ),
        dropped_constant_controls=estimate.dropped_constant_controls,
        residual_mean=pd.Series(residual_mean, index=spec.response_names),
        residual_rmse=pd.Series(residual_rmse, index=spec.response_names),
    )


def rolling_origin_residuals(
    table: pd.DataFrame,
    *,
    spec: LinearGaussianEmissionSpec,
    availability_column: str,
    base_lambda: float,
    knowledge_cutoff: date | datetime | str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return genuinely out-of-sample residuals for block-dependence tests."""

    if base_lambda not in spec.lambda_grid:
        raise ValueError("base_lambda must belong to the frozen lambda grid")
    selection = select_causal_emission_rows(
        table,
        spec=spec,
        availability_column=availability_column,
        knowledge_cutoff=knowledge_cutoff,
    )
    _, residuals = _rolling_scores(
        selection.eligible,
        spec=spec,
        availability_column=availability_column,
        base_lambda=float(base_lambda),
        return_residuals=True,
    )
    return residuals
