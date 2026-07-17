"""Causal walk-forward orchestration for Model 01's Bayesian filter.

The module joins the already-built point-in-time evidence table to the
deterministic first-release regime history, estimates likelihoods with a
strictly earlier information set, and replays a four-month joint-path filter.
It deliberately contains no file-system or publication logic; the CLI owns
artifact serialization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math

import numpy as np
import pandas as pd

from regime_allocation.models.m01_deterministic_composite.inference import (
    AxisLogLikelihood,
    PATH_MONTHS,
    STATE_IDS,
    condition_path_on_regimes,
    initialize_markov_path,
    path_marginal,
    path_marginals,
    probability_entropy,
    update_path_log_likelihoods,
)
from regime_allocation.models.m01_deterministic_composite.evaluation import (
    evaluate_regime_probabilities,
)
from regime_allocation.models.m01_deterministic_composite.likelihood import (
    BlockLikelihoodFit,
    CausalTrainingSelection,
    fit_block_likelihood,
    select_causal_training_vectors,
)
from regime_allocation.models.m01_deterministic_composite.transition import (
    TransitionEstimate,
    estimate_transition_matrix,
    propagate_joint_path,
    propagate_regime_marginal,
)


PROBABILITY_COLUMNS = tuple(f"probability_{state}" for state in STATE_IDS)


@dataclass(frozen=True)
class LikelihoodSpecification:
    """One frozen likelihood specification used in a replay."""

    specification_id: str
    degrees_of_freedom: float | None = 7.0
    kappa: float = 5.0
    covariance_method: str = "ledoit_wolf"
    fixed_shrinkage: float | None = None
    scale_multiplier: float = 1.0
    minimum_complete_vectors: int = 24

    def __post_init__(self) -> None:
        if not self.specification_id.strip():
            raise ValueError("specification_id cannot be empty")
        if self.minimum_complete_vectors < 2:
            raise ValueError("minimum_complete_vectors must be at least two")

    @property
    def distribution_name(self) -> str:
        return "gaussian" if self.degrees_of_freedom is None else "student_t"

    def to_record(self) -> dict[str, object]:
        return {
            "specification_id": self.specification_id,
            "distribution": self.distribution_name,
            "degrees_of_freedom": self.degrees_of_freedom,
            "kappa": self.kappa,
            "covariance_method": self.covariance_method,
            "fixed_shrinkage": self.fixed_shrinkage,
            "scale_standard_deviation_multiplier": self.scale_multiplier,
            "minimum_complete_vectors": self.minimum_complete_vectors,
        }


@dataclass
class WalkForwardResult:
    """All in-memory artifacts produced by one filter replay."""

    specification: LikelihoodSpecification
    checkpoints: pd.DataFrame
    joint_paths: pd.DataFrame
    marginals: pd.DataFrame
    event_audit: pd.DataFrame
    likelihood_fit_audit: pd.DataFrame
    forecasts: pd.DataFrame
    latest_path: np.ndarray
    latest_anchor_month: pd.Timestamp
    latest_transition_matrix: pd.DataFrame


@dataclass
class ForecastEvaluationResult:
    """Scored forecasts, aggregate metrics, and reliability bins."""

    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    calibration_bins: pd.DataFrame


def _normalize_month(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise").dt.to_period("M").dt.to_timestamp()


def normalize_regime_history(history: pd.DataFrame) -> pd.DataFrame:
    """Validate the deterministic first-release target history."""

    required = {"reference_month", "regime_id", "label_available_at"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            f"regime history is missing columns: {', '.join(sorted(missing))}"
        )
    output = history.copy()
    output["reference_month"] = _normalize_month(output["reference_month"])
    output["label_available_at"] = pd.to_datetime(
        output["label_available_at"], errors="coerce"
    ).dt.normalize()
    if output["reference_month"].duplicated().any():
        raise ValueError("regime history contains duplicate reference months")
    known = set(STATE_IDS)
    supplied = set(output.loc[output["regime_id"].notna(), "regime_id"].astype(str))
    unknown = sorted(supplied.difference(known))
    if unknown:
        raise ValueError(f"regime history contains unknown regimes: {unknown}")
    classified = output["regime_id"].notna()
    if output.loc[classified, "label_available_at"].isna().any():
        raise ValueError("classified regimes require label_available_at")
    output["regime_id"] = output["regime_id"].astype("string")
    return output.sort_values("reference_month").reset_index(drop=True)


def prepare_block_vectors(
    events: pd.DataFrame,
    history: pd.DataFrame,
    *,
    block_configuration: Mapping[str, Mapping[str, object]],
) -> dict[str, pd.DataFrame]:
    """Pivot the long evidence table into complete-or-auditable block vectors."""

    required = {
        "event_id",
        "event_group_id",
        "release_block",
        "release_date",
        "reference_month",
        "feature_name",
        "series_id",
        "feature_value",
        "feature_status",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"event table is missing columns: {', '.join(sorted(missing))}")
    normalized_history = normalize_regime_history(history)
    targets = normalized_history[
        ["reference_month", "regime_id", "label_available_at"]
    ]
    prepared: dict[str, pd.DataFrame] = {}
    metadata = [
        "event_id",
        "event_group_id",
        "release_block",
        "release_date",
        "reference_month",
    ]
    for block_id, raw in block_configuration.items():
        feature_names = tuple(str(item) for item in raw["feature_names"])
        source_series = set(str(item) for item in raw["source_series"])
        selected = events.loc[
            (events["release_block"].astype(str) == block_id)
            & events["feature_name"].astype(str).isin(feature_names)
            & events["series_id"].astype(str).isin(source_series)
            & (events["feature_status"].astype(str) == "available")
        ].copy()
        if selected.empty:
            raise ValueError(f"no available evidence rows were found for {block_id}")
        duplicate = selected.duplicated(["event_id", "feature_name"])
        if duplicate.any():
            raise ValueError(f"{block_id} contains duplicate event-feature rows")
        selected["release_date"] = pd.to_datetime(
            selected["release_date"], errors="raise"
        ).dt.normalize()
        selected["reference_month"] = _normalize_month(selected["reference_month"])
        vectors = (
            selected.pivot(index=metadata, columns="feature_name", values="feature_value")
            .reset_index()
            .rename_axis(columns=None)
        )
        for feature_name in feature_names:
            if feature_name not in vectors:
                vectors[feature_name] = np.nan
        vectors = vectors[[*metadata, *feature_names]]
        vectors = vectors.merge(
            targets,
            on="reference_month",
            how="left",
            validate="many_to_one",
        )
        numeric = vectors.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        vectors.loc[:, feature_names] = numeric
        vectors["complete_vector"] = np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all(axis=1)
        vectors["training_available_at"] = pd.concat(
            [vectors["release_date"], vectors["label_available_at"]], axis=1
        ).max(axis=1)
        vectors["training_available_at"] = vectors["training_available_at"].where(
            vectors["regime_id"].notna() & vectors["label_available_at"].notna()
        )
        vectors["training_candidate"] = (
            vectors["complete_vector"]
            & vectors["regime_id"].notna()
            & vectors["label_available_at"].notna()
        )
        vectors["block_id"] = block_id
        vectors = vectors.sort_values(
            ["release_date", "event_id"], kind="mergesort"
        ).reset_index(drop=True)
        prepared[block_id] = vectors
    return prepared


def _path_months(anchor_month: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    anchor = pd.Timestamp(anchor_month).to_period("M")
    return tuple((anchor - lag).to_timestamp() for lag in range(3, -1, -1))


def _month_axis(reference_month: pd.Timestamp, anchor_month: pd.Timestamp) -> int | None:
    months = _path_months(anchor_month)
    reference = pd.Timestamp(reference_month).to_period("M").to_timestamp()
    try:
        return months.index(reference)
    except ValueError:
        return None


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _confirmation_map(
    history: pd.DataFrame,
    *,
    anchor_month: pd.Timestamp,
    available_before: pd.Timestamp | None = None,
    available_on: pd.Timestamp | None = None,
) -> tuple[dict[int, str], list[pd.Timestamp]]:
    if (available_before is None) == (available_on is None):
        raise ValueError("choose exactly one confirmation-date selector")
    classified = history.loc[history["regime_id"].notna()].copy()
    if available_before is not None:
        selected = classified.loc[
            classified["label_available_at"] < pd.Timestamp(available_before)
        ]
    else:
        selected = classified.loc[
            classified["label_available_at"] == pd.Timestamp(available_on)
        ]
    confirmations: dict[int, str] = {}
    months: list[pd.Timestamp] = []
    for row in selected.itertuples(index=False):
        reference = pd.Timestamp(row.reference_month)
        axis = _month_axis(reference, anchor_month)
        if axis is not None:
            confirmations[axis] = str(row.regime_id)
            months.append(reference)
    return confirmations, months


def build_causal_training_cache(
    block_vectors: Mapping[str, pd.DataFrame],
    *,
    filter_start: str | pd.Timestamp,
    filter_end: str | pd.Timestamp,
    minimum_complete_vectors: int,
) -> dict[tuple[str, int], CausalTrainingSelection]:
    """Precompute each distinct causal training sample once across variants."""

    start = pd.Timestamp(filter_start).normalize()
    end = pd.Timestamp(filter_end).normalize()
    cache: dict[tuple[str, int], CausalTrainingSelection] = {}
    excluded_columns = {
        "event_id",
        "event_group_id",
        "release_block",
        "release_date",
        "reference_month",
        "regime_id",
        "label_available_at",
        "complete_vector",
        "block_id",
        "training_available_at",
        "training_candidate",
    }
    for block_id, vectors in block_vectors.items():
        feature_names = tuple(
            column
            for column in vectors.columns
            if column not in excluded_columns
            and pd.api.types.is_numeric_dtype(vectors[column])
        )
        dates = sorted(
            pd.Timestamp(item).normalize()
            for item in vectors.loc[
                vectors["release_date"].between(start, end, inclusive="both"),
                "release_date",
            ].unique()
        )
        for cutoff in dates:
            count = int(
                (
                    vectors["training_candidate"]
                    & (vectors["training_available_at"] < cutoff)
                ).sum()
            )
            key = (block_id, count)
            if count < minimum_complete_vectors or key in cache:
                continue
            selection = select_causal_training_vectors(
                vectors,
                feature_names=feature_names,
                knowledge_cutoff=cutoff,
            )
            if len(selection.eligible) != count:
                raise RuntimeError("causal training cache count disagrees with audit")
            cache[key] = selection
    return cache


def run_walk_forward_filter(
    history: pd.DataFrame,
    block_vectors: Mapping[str, pd.DataFrame],
    *,
    specification: LikelihoodSpecification,
    filter_start: str | pd.Timestamp,
    filter_end: str | pd.Timestamp,
    transition_alpha: float = 0.5,
    apply_evidence: bool = True,
    store_detailed_paths: bool = True,
    store_checkpoint_artifacts: bool = True,
    store_audits: bool = True,
    transition_estimates: Mapping[pd.Timestamp, TransitionEstimate] | None = None,
    causal_training_cache: (
        Mapping[tuple[str, int], CausalTrainingSelection] | None
    ) = None,
) -> WalkForwardResult:
    """Replay the four-month filter under one likelihood specification."""

    targets = normalize_regime_history(history)
    start = pd.Timestamp(filter_start).normalize()
    end = pd.Timestamp(filter_end).normalize()
    if start > end:
        raise ValueError("filter_start cannot be after filter_end")
    anchor_month = start.to_period("M").to_timestamp()
    cutoff = start - pd.Timedelta(days=1)
    if transition_estimates is None:
        transition = estimate_transition_matrix(
            targets,
            knowledge_cutoff=cutoff,
            alpha=transition_alpha,
        )
    else:
        try:
            transition = transition_estimates[start]
        except KeyError as error:
            raise ValueError("transition cache is missing the initial month") from error
    current_transition = transition.posterior_predictive
    path = initialize_markov_path(np.full(len(STATE_IDS), 1.0 / len(STATE_IDS)), current_transition)
    initial_confirmations, initial_confirmation_months = _confirmation_map(
        targets,
        anchor_month=anchor_month,
        available_before=start,
    )
    if initial_confirmations:
        path = condition_path_on_regimes(path, initial_confirmations).posterior
    confirmed_reference_months = set(initial_confirmation_months)

    event_dates: set[pd.Timestamp] = set()
    vectors_by_date: dict[pd.Timestamp, list[tuple[str, pd.DataFrame]]] = {}
    for block_id, vectors in block_vectors.items():
        dated = vectors.loc[
            (vectors["release_date"] >= start) & (vectors["release_date"] <= end)
        ]
        for release_date, group in dated.groupby("release_date", sort=True):
            date_key = pd.Timestamp(release_date).normalize()
            event_dates.add(date_key)
            vectors_by_date.setdefault(date_key, []).append((block_id, group.copy()))
    confirmation_dates = set(
        pd.Timestamp(item).normalize()
        for item in targets.loc[
            targets["label_available_at"].between(start, end, inclusive="both"),
            "label_available_at",
        ].dropna()
    )
    month_starts = set(pd.date_range(start=anchor_month, end=end, freq="MS"))
    month_ends = set(
        item.normalize()
        for item in pd.date_range(start=anchor_month, end=end, freq="ME")
        if start <= item.normalize() <= end
    )
    timeline = sorted(
        {start, end}.union(event_dates, confirmation_dates, month_starts, month_ends)
    )

    checkpoint_records: list[dict[str, object]] = []
    joint_records: list[dict[str, object]] = []
    marginal_records: list[dict[str, object]] = []
    event_records: list[dict[str, object]] = []
    fit_records: list[dict[str, object]] = []
    forecast_records: list[dict[str, object]] = []
    fit_cache: dict[tuple[str, int], BlockLikelihoodFit] = {}
    previous_checkpoint: str | None = None
    sequence = 0
    transition_cutoff = cutoff
    icsa_release_numbers: dict[pd.Timestamp, int] = {}

    def record_checkpoint(
        checkpoint_type: str,
        as_of_date: pd.Timestamp,
        *,
        phase_order: int,
        event_blocks: Sequence[str] = (),
        event_ids: Sequence[str] = (),
        confirmation_months: Sequence[pd.Timestamp] = (),
    ) -> str:
        nonlocal sequence, previous_checkpoint
        sequence += 1
        checkpoint_id = f"{specification.specification_id}:{sequence:06d}"
        months = _path_months(anchor_month)
        if store_checkpoint_artifacts:
            checkpoint_records.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": previous_checkpoint,
                    "specification_id": specification.specification_id,
                    "as_of_date": as_of_date,
                    "phase_order": phase_order,
                    "checkpoint_type": checkpoint_type,
                    "anchor_month": anchor_month,
                    "event_blocks": "|".join(sorted(set(event_blocks))),
                    "event_ids": "|".join(sorted(set(event_ids))),
                    "confirmation_months": "|".join(
                        item.date().isoformat()
                        for item in sorted(confirmation_months)
                    ),
                    "transition_training_cutoff": transition_cutoff,
                    "transition_training_pairs": int(
                        transition.diagnostics["included_transitions"]
                    ),
                    "path_month_0": months[0],
                    "path_month_1": months[1],
                    "path_month_2": months[2],
                    "path_month_3": months[3],
                    "joint_probability_sum": float(path.sum()),
                    "normalization_error": float(abs(path.sum() - 1.0)),
                    "evidence_enabled": apply_evidence,
                }
            )
            marginals = path_marginals(path)
            for axis, reference_month in enumerate(months):
                marginal = marginals[axis]
                map_position = int(np.argmax(marginal))
                entropy = probability_entropy(marginal)
                for state_position, state_id in enumerate(STATE_IDS):
                    marginal_records.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "specification_id": specification.specification_id,
                            "as_of_date": as_of_date,
                            "checkpoint_type": checkpoint_type,
                            "anchor_month": anchor_month,
                            "reference_month": reference_month,
                            "relative_month": axis - (PATH_MONTHS - 1),
                            "marginal_type": "path",
                            "regime_id": state_id,
                            "probability": float(marginal[state_position]),
                            "entropy": entropy,
                            "map_regime_id": STATE_IDS[map_position],
                            "map_probability": float(marginal[map_position]),
                        }
                    )
            next_month = (anchor_month.to_period("M") + 1).to_timestamp()
            next_probabilities = propagate_regime_marginal(
                marginals[-1], current_transition
            )
            next_map = int(np.argmax(next_probabilities))
            next_entropy = probability_entropy(next_probabilities)
            for state_position, state_id in enumerate(STATE_IDS):
                marginal_records.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "specification_id": specification.specification_id,
                        "as_of_date": as_of_date,
                        "checkpoint_type": checkpoint_type,
                        "anchor_month": anchor_month,
                        "reference_month": next_month,
                        "relative_month": 1,
                        "marginal_type": "transition_forecast",
                        "regime_id": state_id,
                        "probability": float(next_probabilities[state_position]),
                        "entropy": next_entropy,
                        "map_regime_id": STATE_IDS[next_map],
                        "map_probability": float(next_probabilities[next_map]),
                    }
                )
        if store_detailed_paths and store_checkpoint_artifacts:
            for positions in np.ndindex(path.shape):
                joint_records.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "specification_id": specification.specification_id,
                        "as_of_date": as_of_date,
                        "checkpoint_type": checkpoint_type,
                        "month_0": months[0],
                        "month_1": months[1],
                        "month_2": months[2],
                        "month_3": months[3],
                        "regime_0": STATE_IDS[positions[0]],
                        "regime_1": STATE_IDS[positions[1]],
                        "regime_2": STATE_IDS[positions[2]],
                        "regime_3": STATE_IDS[positions[3]],
                        "probability": float(path[positions]),
                    }
                )
        previous_checkpoint = checkpoint_id
        return checkpoint_id

    def record_forecast(
        checkpoint_id: str,
        checkpoint_type: str,
        target_month: pd.Timestamp,
        as_of_date: pd.Timestamp,
        *,
        release_number: int | None = None,
    ) -> None:
        axis = _month_axis(target_month, anchor_month)
        if axis is None:
            return
        probabilities = path_marginal(path, axis)
        row: dict[str, object] = {
            "specification_id": specification.specification_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_type": checkpoint_type,
            "checkpoint_date": as_of_date,
            "target_reference_month": pd.Timestamp(target_month),
            "release_number": release_number,
            "posterior_entropy": probability_entropy(probabilities),
        }
        row.update(
            {
                column: float(probabilities[position])
                for position, column in enumerate(PROBABILITY_COLUMNS)
            }
        )
        forecast_records.append(row)

    initial_id = record_checkpoint("initial", start, phase_order=0)
    record_forecast(initial_id, "month_start", anchor_month, start)

    for current_date in timeline:
        if current_date == start:
            pass
        elif current_date in month_starts:
            record_checkpoint("pre_month_roll", current_date, phase_order=0)
            transition_cutoff = current_date - pd.Timedelta(days=1)
            if transition_estimates is None:
                transition = estimate_transition_matrix(
                    targets,
                    knowledge_cutoff=transition_cutoff,
                    alpha=transition_alpha,
                )
            else:
                try:
                    transition = transition_estimates[current_date]
                except KeyError as error:
                    raise ValueError(
                        f"transition cache is missing {current_date.date()}"
                    ) from error
            current_transition = transition.posterior_predictive
            path = propagate_joint_path(path, current_transition)
            anchor_month = current_date.to_period("M").to_timestamp()
            checkpoint_id = record_checkpoint(
                "post_month_roll", current_date, phase_order=1
            )
            record_forecast(
                checkpoint_id,
                "month_start",
                anchor_month,
                current_date,
            )

        confirmations, confirmation_months = _confirmation_map(
            targets,
            anchor_month=anchor_month,
            available_on=current_date,
        )
        date_groups = vectors_by_date.get(current_date, [])
        if date_groups:
            event_ids = [
                str(event_id)
                for _, group in date_groups
                for event_id in group["event_id"].astype(str)
            ]
            block_ids = [block_id for block_id, _ in date_groups]
            record_checkpoint(
                "pre_release_group",
                current_date,
                phase_order=2,
                event_blocks=block_ids,
                event_ids=event_ids,
            )
            path_before_events = path.copy()
            updates: list[AxisLogLikelihood] = []
            pending_audits: list[dict[str, object]] = []
            for block_id, released in date_groups:
                all_vectors = block_vectors[block_id]
                feature_names = tuple(
                    column
                    for column in all_vectors.columns
                    if column not in {
                        "event_id",
                        "event_group_id",
                        "release_block",
                        "release_date",
                        "reference_month",
                        "regime_id",
                        "label_available_at",
                        "complete_vector",
                        "block_id",
                        "training_available_at",
                        "training_candidate",
                    }
                )
                feature_names = tuple(
                    feature
                    for feature in feature_names
                    if pd.api.types.is_numeric_dtype(all_vectors[feature])
                )
                fit: BlockLikelihoodFit | None = None
                selection = None
                fit_error = ""
                training_count = 0
                if apply_evidence:
                    training_count = int(
                        (
                            all_vectors["training_candidate"]
                            & (all_vectors["training_available_at"] < current_date)
                        ).sum()
                    )
                    if training_count >= specification.minimum_complete_vectors:
                        cache_key = (block_id, training_count)
                        fit = fit_cache.get(cache_key)
                        if fit is None:
                            if causal_training_cache is None:
                                selection = select_causal_training_vectors(
                                    all_vectors,
                                    feature_names=feature_names,
                                    knowledge_cutoff=current_date,
                                )
                            else:
                                try:
                                    selection = causal_training_cache[cache_key]
                                except KeyError as error:
                                    raise ValueError(
                                        "causal training cache is missing "
                                        f"{block_id}/{training_count}"
                                    ) from error
                            if len(selection.eligible) != training_count:
                                raise RuntimeError(
                                    "fast causal training count disagrees with audit"
                                )
                            try:
                                fit = fit_block_likelihood(
                                    selection.eligible,
                                    block_id=block_id,
                                    feature_names=feature_names,
                                    kappa=specification.kappa,
                                    degrees_of_freedom=(
                                        specification.degrees_of_freedom
                                    ),
                                    covariance_method=specification.covariance_method,
                                    fixed_shrinkage=specification.fixed_shrinkage,
                                    scale_multiplier=specification.scale_multiplier,
                                    knowledge_cutoff=current_date,
                                    training_diagnostics=selection.diagnostics,
                                )
                            except ValueError as error:
                                fit_error = str(error)
                            else:
                                fit_cache[cache_key] = fit
                                audit = fit.to_audit_dict()
                                fit_record = (
                                    {
                                        "specification_id": (
                                            specification.specification_id
                                        ),
                                        "fit_id": (
                                            f"{specification.specification_id}:"
                                            f"{block_id}:{training_count}"
                                        ),
                                        "fit_date": current_date,
                                        "block_id": block_id,
                                        "training_count": training_count,
                                        "distribution": audit["distribution"],
                                        "degrees_of_freedom": audit[
                                            "degrees_of_freedom"
                                        ],
                                        "kappa": audit["kappa"],
                                        "covariance_method": audit[
                                            "covariance_method"
                                        ],
                                        "applied_shrinkage": audit[
                                            "applied_shrinkage"
                                        ],
                                        "learned_shrinkage": audit[
                                            "learned_shrinkage"
                                        ],
                                        "scale_standard_deviation_multiplier": audit[
                                            "scale_standard_deviation_multiplier"
                                        ],
                                        "knowledge_cutoff": audit[
                                            "knowledge_cutoff"
                                        ],
                                        "first_training_available_at": audit[
                                            "first_training_available_at"
                                        ],
                                        "last_training_available_at": audit[
                                            "last_training_available_at"
                                        ],
                                        "training_diagnostics": _json(
                                            audit["training_diagnostics"]
                                        ),
                                        "regime_counts": _json(
                                            audit["regime_counts"]
                                        ),
                                        "pooled_mean": _json(audit["pooled_mean"]),
                                        "regime_means": _json(
                                            audit["regime_means"]
                                        ),
                                        "covariance": _json(audit["covariance"]),
                                        "distribution_scale": _json(
                                            audit["distribution_scale"]
                                        ),
                                        "minimum_covariance_eigenvalue": audit[
                                            "covariance_minimum_eigenvalue"
                                        ],
                                    }
                                )
                                if store_audits:
                                    fit_records.append(fit_record)
                for row in released.itertuples(index=False):
                    reference_month = pd.Timestamp(row.reference_month)
                    axis = _month_axis(reference_month, anchor_month)
                    complete = bool(row.complete_vector)
                    audit_row: dict[str, object] = {
                        "specification_id": specification.specification_id,
                        "release_date": current_date,
                        "block_id": block_id,
                        "event_id": str(row.event_id),
                        "event_group_id": str(row.event_group_id),
                        "reference_month": reference_month,
                        "path_axis": axis,
                        "complete_vector": complete,
                        "training_count": training_count,
                        "fit_error": fit_error,
                    }
                    if not apply_evidence:
                        status = "transition_only_suppressed"
                    elif not complete:
                        status = "incomplete_vector"
                    elif axis is None:
                        status = "outside_four_month_path"
                    elif fit is None:
                        status = (
                            "fit_error"
                            if fit_error
                            else "insufficient_training_vectors"
                        )
                    elif reference_month in confirmed_reference_months:
                        status = "already_confirmed_no_op"
                    else:
                        observation = {
                            feature: float(getattr(row, feature))
                            for feature in feature_names
                        }
                        log_likelihoods = fit.log_likelihoods(observation)
                        values = log_likelihoods.to_numpy(dtype=float)
                        updates.append(AxisLogLikelihood(axis=axis, values=values))
                        prior_marginal = path_marginal(path_before_events, axis)
                        log_prior = np.full(len(STATE_IDS), -np.inf)
                        positive = prior_marginal > 0.0
                        log_prior[positive] = np.log(prior_marginal[positive])
                        audit_row["standalone_predictive_log_density"] = float(
                            np.logaddexp.reduce(log_prior + values)
                        )
                        for position, state_id in enumerate(STATE_IDS):
                            audit_row[f"log_likelihood_{state_id}"] = float(
                                values[position]
                            )
                        status = "applied"
                    audit_row["update_status"] = status
                    if store_audits:
                        pending_audits.append(audit_row)
            if updates:
                update = update_path_log_likelihoods(path, updates)
                path = update.posterior
                update_log_normalizer = update.log_normalizer
                entropy_change = update.posterior_entropy - update.prior_entropy
                update_kl = update.kl_divergence
            else:
                update_log_normalizer = np.nan
                entropy_change = 0.0
                update_kl = 0.0
            for audit_row in pending_audits:
                axis_value = audit_row["path_axis"]
                if axis_value is not None:
                    axis = int(axis_value)
                    before = path_marginal(path_before_events, axis)
                    after = path_marginal(path, axis)
                    for position, state_id in enumerate(STATE_IDS):
                        audit_row[f"prior_probability_{state_id}"] = float(
                            before[position]
                        )
                        audit_row[f"posterior_probability_{state_id}"] = float(
                            after[position]
                        )
                audit_row["daily_update_log_normalizer"] = update_log_normalizer
                audit_row["daily_entropy_change"] = entropy_change
                audit_row["daily_kl_divergence"] = update_kl
                event_records.append(audit_row)
            checkpoint_id = record_checkpoint(
                "post_release_group",
                current_date,
                phase_order=3,
                event_blocks=block_ids,
                event_ids=event_ids,
            )
            icsa_targets = sorted(
                {
                    pd.Timestamp(row.reference_month)
                    for block_id, group in date_groups
                    if block_id == "weekly_claims"
                    for row in group.itertuples(index=False)
                }
            )
            for target_month in icsa_targets:
                number = icsa_release_numbers.get(target_month, 0) + 1
                icsa_release_numbers[target_month] = number
                record_forecast(
                    checkpoint_id,
                    "post_icsa_release",
                    target_month,
                    current_date,
                    release_number=number,
                )

        if confirmations:
            checkpoint_id = record_checkpoint(
                "pre_confirmation_group",
                current_date,
                phase_order=4,
                confirmation_months=confirmation_months,
            )
            for target_month in confirmation_months:
                record_forecast(
                    checkpoint_id,
                    "pre_confirmation",
                    target_month,
                    current_date,
                )
            confirmation_update = condition_path_on_regimes(path, confirmations)
            path = confirmation_update.posterior
            confirmed_reference_months.update(confirmation_months)
            record_checkpoint(
                "post_confirmation_group",
                current_date,
                phase_order=5,
                confirmation_months=confirmation_months,
            )

        if current_date in month_ends:
            checkpoint_id = record_checkpoint(
                "month_end", current_date, phase_order=6
            )
            record_forecast(
                checkpoint_id,
                "month_end",
                anchor_month,
                current_date,
            )

    record_checkpoint("latest", end, phase_order=7)
    return WalkForwardResult(
        specification=specification,
        checkpoints=pd.DataFrame.from_records(checkpoint_records),
        joint_paths=pd.DataFrame.from_records(joint_records),
        marginals=pd.DataFrame.from_records(marginal_records),
        event_audit=pd.DataFrame.from_records(event_records),
        likelihood_fit_audit=pd.DataFrame.from_records(fit_records),
        forecasts=pd.DataFrame.from_records(forecast_records),
        latest_path=path.copy(),
        latest_anchor_month=anchor_month,
        latest_transition_matrix=current_transition.copy(),
    )


def attach_forecast_targets(
    forecasts: pd.DataFrame,
    history: pd.DataFrame,
    *,
    evaluation_start: str | pd.Timestamp,
) -> pd.DataFrame:
    """Attach realized targets and mark forecasts eligible for evaluation."""

    required = {
        "specification_id",
        "checkpoint_type",
        "checkpoint_date",
        "target_reference_month",
        *PROBABILITY_COLUMNS,
    }
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"forecast table is missing columns: {sorted(missing)}")
    targets = normalize_regime_history(history)[
        ["reference_month", "regime_id", "label_available_at"]
    ].rename(
        columns={
            "reference_month": "target_reference_month",
            "regime_id": "realized_regime_id",
            "label_available_at": "target_label_available_at",
        }
    )
    output = forecasts.copy()
    output["checkpoint_date"] = pd.to_datetime(
        output["checkpoint_date"], errors="raise"
    ).dt.normalize()
    output["target_reference_month"] = _normalize_month(
        output["target_reference_month"]
    )
    output = output.merge(
        targets,
        on="target_reference_month",
        how="left",
        validate="many_to_one",
    )
    start = pd.Timestamp(evaluation_start).to_period("M").to_timestamp()
    status = np.full(len(output), "eligible", dtype=object)
    missing_truth = output["realized_regime_id"].isna().to_numpy()
    status[missing_truth] = "unclassified_target"
    before_start = (output["target_reference_month"] < start).to_numpy()
    status[(status == "eligible") & before_start] = "before_evaluation_start"
    label_dates = pd.to_datetime(output["target_label_available_at"], errors="coerce")
    pre_confirmation = output["checkpoint_type"].eq("pre_confirmation")
    date_valid = output["checkpoint_date"] < label_dates
    date_valid |= pre_confirmation & output["checkpoint_date"].eq(label_dates)
    status[(status == "eligible") & ~date_valid.to_numpy()] = (
        "not_strictly_pre_confirmation"
    )
    probabilities = output.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    valid_probability = (
        np.isfinite(probabilities).all(axis=1)
        & (probabilities >= 0.0).all(axis=1)
        & np.isclose(probabilities.sum(axis=1), 1.0)
    )
    status[(status == "eligible") & ~valid_probability] = "invalid_probabilities"
    output["evaluation_status"] = status
    output["evaluation_eligible"] = status == "eligible"
    return output


def _forecast_key(frame: pd.DataFrame) -> pd.Series:
    release = frame["release_number"].astype("Int64").astype("string").fillna("")
    return (
        frame["checkpoint_type"].astype(str)
        + "|"
        + frame["target_reference_month"].dt.strftime("%Y-%m-%d")
        + "|"
        + release
    )


def evaluate_forecast_table(
    forecasts: pd.DataFrame,
    *,
    transition_only_specification_id: str = "transition_only",
    calibration_bin_count: int = 10,
) -> ForecastEvaluationResult:
    """Evaluate every specification on rows shared with transition-only."""

    if "evaluation_eligible" not in forecasts:
        raise ValueError("attach_forecast_targets must run before evaluation")
    scored = forecasts.loc[forecasts["evaluation_eligible"]].copy()
    if scored.empty:
        raise ValueError("no eligible forecasts are available for evaluation")
    scored["forecast_key"] = _forecast_key(scored)
    baseline_keys = set(
        scored.loc[
            scored["specification_id"] == transition_only_specification_id,
            "forecast_key",
        ]
    )
    if not baseline_keys:
        raise ValueError("transition-only forecasts are required for skill scores")
    scored["shared_with_transition_only"] = scored["forecast_key"].isin(
        baseline_keys
    )
    metric_records: list[dict[str, object]] = []
    calibration_records: list[dict[str, object]] = []
    grouping = ["specification_id", "checkpoint_type", "release_number"]
    for group_key, group in scored.loc[
        scored["shared_with_transition_only"]
    ].groupby(grouping, dropna=False, sort=True):
        specification_id, checkpoint_type, release_number = group_key
        group = group.sort_values("target_reference_month")
        probabilities = group.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
        truths = group["realized_regime_id"].astype(str).tolist()
        metrics = evaluate_regime_probabilities(
            probabilities,
            truths,
            calibration_bin_count=calibration_bin_count,
        )
        baseline = scored.loc[
            (scored["specification_id"] == transition_only_specification_id)
            & scored["forecast_key"].isin(group["forecast_key"])
        ].sort_values("forecast_key")
        aligned = group.sort_values("forecast_key")
        if len(baseline) != len(aligned):
            raise RuntimeError("transition-only comparison rows are misaligned")
        baseline_metrics = evaluate_regime_probabilities(
            baseline.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float),
            baseline["realized_regime_id"].astype(str).tolist(),
            calibration_bin_count=calibration_bin_count,
        )
        baseline_brier = baseline_metrics.multiclass_brier_score
        brier_skill = (
            1.0 - metrics.multiclass_brier_score / baseline_brier
            if baseline_brier > 0.0
            else np.nan
        )
        metric_records.append(
            {
                "specification_id": specification_id,
                "checkpoint_type": checkpoint_type,
                "release_number": release_number,
                "observation_count": metrics.observation_count,
                "first_target_month": group["target_reference_month"].min(),
                "last_target_month": group["target_reference_month"].max(),
                "negative_log_likelihood": metrics.negative_log_likelihood,
                "multiclass_brier_score": metrics.multiclass_brier_score,
                "map_accuracy": metrics.map_accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "macro_recall": metrics.macro_recall,
                "macro_f1": metrics.macro_f1,
                "growth_axis_brier_score": metrics.growth_axis_brier_score,
                "inflation_axis_brier_score": metrics.inflation_axis_brier_score,
                "top_label_expected_calibration_error": (
                    metrics.expected_calibration_error
                ),
                "classwise_expected_calibration_error": (
                    metrics.classwise_expected_calibration_error
                ),
                "mean_posterior_entropy": metrics.mean_posterior_entropy,
                "transition_only_negative_log_likelihood": (
                    baseline_metrics.negative_log_likelihood
                ),
                "transition_only_multiclass_brier_score": baseline_brier,
                "negative_log_likelihood_difference": (
                    metrics.negative_log_likelihood
                    - baseline_metrics.negative_log_likelihood
                ),
                "brier_skill_vs_transition_only": brier_skill,
            }
        )
        for item in metrics.calibration_bins:
            calibration_records.append(
                {
                    "specification_id": specification_id,
                    "checkpoint_type": checkpoint_type,
                    "release_number": release_number,
                    "calibration_type": "top_label_confidence",
                    "regime_id": "",
                    "bin_index": item.bin_index,
                    "lower_bound": item.lower_bound,
                    "upper_bound": item.upper_bound,
                    "upper_bound_inclusive": item.upper_bound_inclusive,
                    "count": item.count,
                    "mean_probability": item.mean_confidence,
                    "empirical_frequency": item.empirical_accuracy,
                    "absolute_gap": item.absolute_gap,
                }
            )
        for item in metrics.classwise_calibration_bins:
            calibration_records.append(
                {
                    "specification_id": specification_id,
                    "checkpoint_type": checkpoint_type,
                    "release_number": release_number,
                    "calibration_type": "classwise_one_vs_rest",
                    "regime_id": item.regime_id,
                    "bin_index": item.bin_index,
                    "lower_bound": item.lower_bound,
                    "upper_bound": item.upper_bound,
                    "upper_bound_inclusive": item.upper_bound_inclusive,
                    "count": item.count,
                    "mean_probability": item.mean_probability,
                    "empirical_frequency": item.empirical_frequency,
                    "absolute_gap": item.absolute_gap,
                }
            )
    return ForecastEvaluationResult(
        forecasts=scored.sort_values(
            ["specification_id", "checkpoint_type", "target_reference_month"]
        ).reset_index(drop=True),
        metrics=pd.DataFrame.from_records(metric_records),
        calibration_bins=pd.DataFrame.from_records(calibration_records),
    )
