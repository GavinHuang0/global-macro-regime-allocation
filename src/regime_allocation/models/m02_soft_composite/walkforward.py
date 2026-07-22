"""Orchestrate Model 02's causal rolling event-driven Gaussian filter.

The module is deliberately in-memory only.  It turns the canonical long
release-event table into observation-model vectors, joins each historical
vector to the *eventually released* Model 02 score center, and replays two
parallel four-month filters:

``evidence_filter``
    VAR(1) month rolls, release-block Gaussian updates, and exact score-center
    observations.

``transition_only``
    The identical VAR(1) rolls and exact score observations, with release
    evidence suppressed.  This is the causal comparison baseline.

Input schema assumptions are explicit:

* score history has one row per ``reference_month`` and columns
  ``growth_score``, ``inflation_score``, and ``score_available_at``;
* mapping history has one baseline row per month, an availability timestamp,
  ``mapping_status``, and the three unique entries of ``Omega_map``;
* evidence is the repository's canonical long table with one feature per row,
  keyed by release date, original observation date, and reference month; and
* an observation model is selected by its configured ``event_block`` and its
  configured response/control feature names.  This lets the two asynchronous
  inflation-pressure submodels share an event block without being combined.

Training availability is the maximum of release availability, target-score
availability, and every required control's availability.  Fits always apply a
strict ``training_available_at < release_date`` cutoff.  A control that is
constant in the causal training sample is temporarily dropped, because an
unpenalized constant control is collinear with the intercept; it becomes active
automatically once identified.  This matters for the housing methodology
dummy before its November 2022 break.

The filtered state is the score center ``Z``.  Mapping covariance never enters
its VAR propagation or evidence updates; it is added only when states are read
out as soft quadrant probabilities.  Completed composite scores are exact
zero-noise observations of ``Z`` at the end of their availability day.  A
same-day mapping estimate may therefore enter only post-exact checkpoints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from regime_allocation.models.m02_soft_composite.gaussian_emissions import (
    LinearGaussianEmissionFit,
    LinearGaussianEmissionSpec,
    fit_linear_gaussian_emission,
)
from regime_allocation.models.m02_soft_composite.joint_filter import (
    JointGaussianState,
    VarDynamics,
    condition_on_exact_score,
    initialize_joint_exact,
    joint_quadrant_path_probabilities,
    monthly_quadrant_probabilities,
    roll_joint_gaussian,
    update_joint_gaussian,
)
from regime_allocation.models.m02_soft_composite.probability_map import (
    REGIME_ORDER,
    gaussian_quadrant_weights,
)
from regime_allocation.models.m02_soft_composite.var_transition import (
    build_var_pair_audit,
    fit_var1_ols,
    select_causal_var_pairs,
)


FILTER_VARIANTS = ("evidence_filter", "transition_only")
_STATE_NAMES = ("growth_score", "inflation_score")
_MAPPING_COLUMNS = (
    "growth_map_variance",
    "growth_inflation_map_covariance",
    "inflation_map_variance",
)


@dataclass(frozen=True)
class PreparedObservationData:
    """Validated event vectors and supervised emission-training tables."""

    specifications: Mapping[str, LinearGaussianEmissionSpec]
    events: pd.DataFrame
    training_tables: Mapping[str, pd.DataFrame]
    preparation_audit: pd.DataFrame


@dataclass
class GaussianWalkForwardResult:
    """All in-memory products from one Model 02 filter replay."""

    checkpoints: pd.DataFrame
    joint_gaussians: pd.DataFrame
    marginals: pd.DataFrame
    joint_paths: pd.DataFrame
    event_audit: pd.DataFrame
    exact_score_audit: pd.DataFrame
    transition_fit_audit: pd.DataFrame
    emission_fit_audit: pd.DataFrame
    evaluation_rows: pd.DataFrame
    predictive_residuals: pd.DataFrame
    latest_states: Mapping[str, JointGaussianState]
    initial_date: pd.Timestamp
    replay_end: pd.Timestamp


def _naive_timestamp(value: object, *, label: str, normalize: bool = False) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{label} must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize() if normalize else timestamp


def _month(value: object, *, label: str = "reference_month") -> pd.Timestamp:
    timestamp = _naive_timestamp(value, label=label)
    return timestamp.to_period("M").to_timestamp()


def _parse_datetime(series: pd.Series, *, label: str, normalize: bool = False) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{label} contains an invalid or missing timestamp")
    result = parsed.dt.tz_convert(None)
    return result.dt.normalize() if normalize else result


def _normalize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    required = {
        "reference_month",
        "growth_score",
        "inflation_score",
        "score_available_at",
    }
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError("score history omits columns: " + ", ".join(sorted(missing)))
    output = scores.loc[:, list(required)].copy()
    output["reference_month"] = (
        pd.to_datetime(output["reference_month"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    output["score_available_at"] = pd.to_datetime(
        output["score_available_at"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    for column in _STATE_NAMES:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.loc[
        output["reference_month"].notna()
        & output["score_available_at"].notna()
        & np.isfinite(output.loc[:, _STATE_NAMES]).all(axis=1)
    ].copy()
    if output.empty:
        raise ValueError("score history has no complete released score centers")
    if output["reference_month"].duplicated().any():
        raise ValueError("score history contains duplicate complete reference months")
    return output.sort_values("reference_month").reset_index(drop=True)


def _normalize_mapping_history(
    mapping_history: pd.DataFrame,
    *,
    baseline_revision_horizon_months: int,
) -> pd.DataFrame:
    required = {
        "reference_month",
        "score_available_at",
        "mapping_status",
        *_MAPPING_COLUMNS,
    }
    missing = required.difference(mapping_history.columns)
    if missing:
        raise ValueError("mapping history omits columns: " + ", ".join(sorted(missing)))
    output = mapping_history.copy()
    if "is_baseline" in output:
        baseline = output["is_baseline"]
        if baseline.dtype == object:
            baseline = baseline.astype(str).str.lower().map({"true": True, "false": False})
        output = output.loc[baseline.fillna(False).astype(bool)].copy()
    if "revision_horizon_months" in output:
        output = output.loc[
            pd.to_numeric(output["revision_horizon_months"], errors="coerce")
            == int(baseline_revision_horizon_months)
        ].copy()
    output["reference_month"] = (
        pd.to_datetime(output["reference_month"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    output["score_available_at"] = pd.to_datetime(
        output["score_available_at"], errors="coerce", utc=True
    ).dt.tz_convert(None).dt.normalize()
    for column in _MAPPING_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.loc[
        output["reference_month"].notna()
        & output["score_available_at"].notna()
        & output["mapping_status"].astype(str).eq("available")
        & np.isfinite(output.loc[:, _MAPPING_COLUMNS]).all(axis=1)
    ].copy()
    if output.empty:
        raise ValueError("mapping history has no causally available baseline covariance")
    if output["reference_month"].duplicated().any():
        raise ValueError("baseline mapping history contains duplicate reference months")
    for row in output.itertuples(index=False):
        covariance = _mapping_covariance(pd.Series(row._asdict()))
        if np.linalg.eigvalsh(covariance).min() < -1.0e-10:
            raise ValueError("mapping history contains a non-positive-semidefinite covariance")
    return output.sort_values(["score_available_at", "reference_month"]).reset_index(drop=True)


def _mapping_covariance(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            [row["growth_map_variance"], row["growth_inflation_map_covariance"]],
            [row["growth_inflation_map_covariance"], row["inflation_map_variance"]],
        ],
        dtype=float,
    )


def _mapping_asof(
    mappings: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    allow_same_day: bool,
) -> pd.Series | None:
    available = mappings["score_available_at"]
    eligible = available <= as_of if allow_same_day else available < as_of
    if not bool(eligible.any()):
        return None
    return mappings.loc[eligible].sort_values(
        ["score_available_at", "reference_month"], kind="mergesort"
    ).iloc[-1]


def _mapping_for_exact_score(
    mappings: pd.DataFrame,
    reference_month: pd.Timestamp,
    availability_date: pd.Timestamp,
) -> pd.Series | None:
    """Return the target month's own same-day-or-earlier mapping estimate."""

    eligible = mappings.loc[
        mappings["reference_month"].eq(reference_month)
        & mappings["score_available_at"].le(availability_date)
    ]
    if eligible.empty:
        return None
    return eligible.sort_values("score_available_at", kind="mergesort").iloc[-1]


def observation_specifications(config: Mapping[str, Any]) -> dict[str, LinearGaussianEmissionSpec]:
    """Build emission specifications from the frozen YAML-shaped mapping."""

    models = config.get("observation_models")
    emissions = config.get("emissions", {})
    if not isinstance(models, Mapping) or not models:
        raise ValueError("configuration requires observation_models")
    lambda_grid = tuple(float(value) for value in emissions.get("lambda_grid", ()))
    covariance = emissions.get("covariance", {})
    floor = float(covariance.get("eigenvalue_floor", 1.0e-8))
    result: dict[str, LinearGaussianEmissionSpec] = {}
    for model_id, raw in models.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"observation model {model_id} must be a mapping")
        result[str(model_id)] = LinearGaussianEmissionSpec(
            block_id=str(model_id),
            response_names=tuple(str(value) for value in raw["responses"]),
            control_names=tuple(str(value) for value in raw.get("controls", ())),
            state_loading_penalties=tuple(
                tuple(float(value) for value in row)
                for row in raw.get("loading_penalties", ())
            ),
            exact_zero_mask=tuple(
                tuple(bool(value) for value in row)
                for row in raw.get("exact_zero_mask", ())
            ),
            lambda_grid=lambda_grid,
            minimum_training_samples=int(raw["minimum_training_samples"]),
            validation_minimum_training_samples=int(
                raw["validation_minimum_training_samples"]
            ),
            minimum_validation_observations=int(raw["minimum_validation_observations"]),
            covariance_eigenvalue_floor=floor,
        )
    return result


def prepare_observation_data(
    events: pd.DataFrame,
    score_history: pd.DataFrame,
    config: Mapping[str, Any],
) -> PreparedObservationData:
    """Pivot long releases and construct causally dated supervised rows.

    A live event vector may be partial.  Emission training, however, uses only
    rows with every configured response, every configured control, and a
    subsequently released exact target score.  Excluded rows remain visible in
    ``preparation_audit``.
    """

    required = {
        "event_id",
        "event_group_id",
        "release_block",
        "release_date",
        "reference_month",
        "feature_name",
        "feature_value",
        "feature_status",
        "series_id",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("evidence events omit columns: " + ", ".join(sorted(missing)))
    scores = _normalize_scores(score_history)
    score_lookup = scores.set_index("reference_month")
    specs = observation_specifications(config)
    model_config = config["observation_models"]
    long = events.copy()
    long["release_date"] = _parse_datetime(
        long["release_date"], label="release_date", normalize=True
    )
    long["reference_month"] = (
        pd.to_datetime(long["reference_month"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if long["reference_month"].isna().any():
        raise ValueError("evidence events contain an invalid reference_month")
    if "reference_date" in long:
        long["reference_date"] = _parse_datetime(
            long["reference_date"], label="reference_date", normalize=True
        )
    else:
        # Synthetic/legacy callers may supply only a monthly target.  The
        # production event artifact always carries the precise observation
        # date, which is essential for ordering a same-day weekly catch-up
        # batch in the residual diagnostics.
        long["reference_date"] = long["reference_month"]
    long["feature_value"] = pd.to_numeric(long["feature_value"], errors="coerce")
    long = long.loc[
        long["feature_status"].astype(str).eq("available")
        & np.isfinite(long["feature_value"])
    ].copy()

    prepared_events: list[pd.DataFrame] = []
    training_tables: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, object]] = []
    for model_id, spec in specs.items():
        raw = model_config[model_id]
        event_block = str(raw["event_block"])
        names = spec.response_names + spec.control_names
        selected = long.loc[
            long["release_block"].astype(str).eq(event_block)
            & long["feature_name"].astype(str).isin(names)
        ].copy()
        if selected.empty:
            training_tables[model_id] = pd.DataFrame(
                columns=(
                    "event_instance_id",
                    "training_available_at",
                    *spec.response_names,
                    *spec.state_names,
                    *spec.control_names,
                )
            )
            audits.append(
                {
                    "observation_model_id": model_id,
                    "selected_long_rows": 0,
                    "live_event_vectors": 0,
                    "training_candidates": 0,
                    "excluded_incomplete_response": 0,
                    "excluded_incomplete_control": 0,
                    "excluded_score_unavailable": 0,
                }
            )
            continue
        duplicate = selected.duplicated(
            ["event_id", "feature_name"], keep=False
        )
        if duplicate.any():
            sample = selected.loc[
                duplicate, ["event_id", "release_date", "reference_month", "feature_name"]
            ].head(1)
            raise ValueError(
                f"{model_id} has duplicate feature rows for one release/reference pair: "
                f"{sample.to_dict(orient='records')[0]}"
            )

        records: list[dict[str, object]] = []
        for (event_id, release_date, reference_month), group in selected.groupby(
            ["event_id", "release_date", "reference_month"], sort=True
        ):
            if group["reference_date"].nunique() != 1:
                raise ValueError(
                    f"{model_id} event {event_id} spans multiple reference dates"
                )
            values = group.set_index("feature_name")["feature_value"]
            series = group.set_index("feature_name")["series_id"]
            record: dict[str, object] = {
                "observation_model_id": model_id,
                "economic_block": str(raw["economic_block"]),
                "event_block": event_block,
                "release_date": pd.Timestamp(release_date),
                "reference_month": pd.Timestamp(reference_month),
                "reference_date": pd.Timestamp(group["reference_date"].min()),
                "event_ids": str(event_id),
                "event_group_ids": "|".join(
                    sorted(group["event_group_id"].astype(str).unique())
                ),
            }
            record["event_instance_id"] = (
                f"{model_id}:{pd.Timestamp(release_date):%Y-%m-%d}:"
                f"{pd.Timestamp(reference_month):%Y-%m}:"
                f"{hashlib.sha256(str(event_id).encode('utf-8')).hexdigest()[:12]}"
            )
            for name in names:
                record[name] = float(values[name]) if name in values else np.nan
                record[f"{name}__series_id"] = str(series[name]) if name in series else ""
            records.append(record)
        vectors = pd.DataFrame.from_records(records).sort_values(
            ["release_date", "reference_date", "reference_month", "event_instance_id"]
        ).reset_index(drop=True)
        response_complete = np.isfinite(
            vectors.loc[:, spec.response_names].to_numpy(dtype=float)
        ).all(axis=1)
        response_any = np.isfinite(
            vectors.loc[:, spec.response_names].to_numpy(dtype=float)
        ).any(axis=1)
        control_complete = (
            np.isfinite(vectors.loc[:, spec.control_names].to_numpy(dtype=float)).all(axis=1)
            if spec.control_names
            else np.ones(len(vectors), dtype=bool)
        )
        vectors["has_observed_response"] = response_any
        vectors["complete_response_vector"] = response_complete
        vectors["complete_control_vector"] = control_complete
        prepared_events.append(vectors.loc[response_any].copy())

        # Weekly series legitimately contribute several observations to one
        # monthly target.  Monthly blocks instead get one supervised vector per
        # reference month, assembled across asynchronous first-release dates;
        # its availability is the latest component/control release.  Live
        # events remain separate above, so an early partial release can still
        # update the filter with the observed subvector.
        repeated_within_month = bool(
            selected.groupby(["reference_month", "feature_name"]).size().max() > 1
        )
        if repeated_within_month:
            training_base = vectors.copy()
            training_base["response_available_at"] = training_base["release_date"]
            training_base["control_available_at"] = training_base["release_date"].where(
                training_base["complete_control_vector"]
            )
        else:
            training_records: list[dict[str, object]] = []
            for reference_month, group in selected.groupby("reference_month", sort=True):
                if group["feature_name"].duplicated().any():
                    raise ValueError(
                        f"{model_id} has duplicate monthly training features for "
                        f"{pd.Timestamp(reference_month):%Y-%m}"
                    )
                indexed = group.set_index("feature_name")
                training_record: dict[str, object] = {
                    "observation_model_id": model_id,
                    "economic_block": str(raw["economic_block"]),
                    "event_block": event_block,
                    "reference_month": pd.Timestamp(reference_month),
                    "event_instance_id": (
                        f"training:{model_id}:{pd.Timestamp(reference_month):%Y-%m}"
                    ),
                    "event_ids": "|".join(sorted(group["event_id"].astype(str).unique())),
                    "event_group_ids": "|".join(
                        sorted(group["event_group_id"].astype(str).unique())
                    ),
                }
                for name in names:
                    training_record[name] = (
                        float(indexed.loc[name, "feature_value"])
                        if name in indexed.index
                        else np.nan
                    )
                    training_record[f"{name}__series_id"] = (
                        str(indexed.loc[name, "series_id"])
                        if name in indexed.index
                        else ""
                    )
                response_rows = group[group["feature_name"].isin(spec.response_names)]
                control_rows = group[group["feature_name"].isin(spec.control_names)]
                training_record["response_available_at"] = (
                    response_rows["release_date"].max()
                    if len(response_rows) == len(spec.response_names)
                    else pd.NaT
                )
                training_record["control_available_at"] = (
                    control_rows["release_date"].max()
                    if len(control_rows) == len(spec.control_names)
                    else (group["release_date"].min() if not spec.control_names else pd.NaT)
                )
                training_record["release_date"] = group["release_date"].max()
                training_records.append(training_record)
            training_base = pd.DataFrame.from_records(training_records)

        training_base = training_base.merge(
            scores.rename(columns={"score_available_at": "target_score_available_at"}),
            on="reference_month",
            how="left",
            validate="many_to_one",
        )
        training_response_complete = np.isfinite(
            training_base.loc[:, spec.response_names].to_numpy(dtype=float)
        ).all(axis=1)
        training_control_complete = (
            np.isfinite(
                training_base.loc[:, spec.control_names].to_numpy(dtype=float)
            ).all(axis=1)
            if spec.control_names
            else np.ones(len(training_base), dtype=bool)
        )
        score_complete = (
            training_base["target_score_available_at"].notna()
            & np.isfinite(training_base.loc[:, _STATE_NAMES]).all(axis=1)
        )
        training_base["training_candidate"] = (
            training_response_complete & training_control_complete & score_complete
        )
        training_base["training_available_at"] = pd.concat(
            [
                training_base["response_available_at"],
                training_base["target_score_available_at"],
                training_base["control_available_at"],
            ],
            axis=1,
        ).max(axis=1)
        training_base.loc[
            ~training_base["training_candidate"], "training_available_at"
        ] = pd.NaT

        training = training_base.loc[training_base["training_candidate"]].copy()
        training_tables[model_id] = training.reset_index(drop=True)
        audits.append(
            {
                "observation_model_id": model_id,
                "selected_long_rows": int(len(selected)),
                "live_event_vectors": int(response_any.sum()),
                "training_candidates": int(training_base["training_candidate"].sum()),
                "training_vector_policy": (
                    "event_level_repeated_monthly_target"
                    if repeated_within_month
                    else "monthly_reference_vector_max_availability"
                ),
                "excluded_incomplete_response": int((~training_response_complete).sum()),
                "excluded_incomplete_control": int(
                    (training_response_complete & ~training_control_complete).sum()
                ),
                "excluded_score_unavailable": int(
                    (
                        training_response_complete
                        & training_control_complete
                        & ~score_complete
                    ).sum()
                ),
            }
        )

    event_frame = (
        pd.concat(prepared_events, ignore_index=True, sort=False)
        if prepared_events
        else pd.DataFrame()
    )
    if not event_frame.empty:
        event_frame = event_frame.sort_values(
            [
                "release_date",
                "observation_model_id",
                "reference_date",
                "reference_month",
                "event_instance_id",
            ]
        ).reset_index(drop=True)
    return PreparedObservationData(
        specifications=specs,
        events=event_frame,
        training_tables=training_tables,
        preparation_audit=pd.DataFrame.from_records(audits),
    )


def _causal_var_fit(
    pair_audit: pd.DataFrame,
    *,
    source_reference_month: pd.Timestamp,
    cutoff: pd.Timestamp,
    minimum_pairs: int,
) -> tuple[Any, pd.DataFrame]:
    selected = select_causal_var_pairs(
        pair_audit,
        source_reference_month=source_reference_month,
        # The selector is inclusive; subtracting one nanosecond implements the
        # repository-wide strict-before convention without losing date data.
        forecast_available_at=cutoff - pd.Timedelta(nanoseconds=1),
    )
    if len(selected) < minimum_pairs:
        raise ValueError(
            f"VAR(1) needs {minimum_pairs} causal pairs before {cutoff.date()}, "
            f"but only {len(selected)} are available"
        )
    return fit_var1_ols(selected), selected


def _dynamics(fit: Any) -> VarDynamics:
    return VarDynamics(
        intercept=np.asarray(fit.intercept, dtype=float),
        transition=np.asarray(fit.transition, dtype=float),
        innovation_covariance=np.asarray(fit.innovation_covariance, dtype=float),
    )


def _earliest_initialization(
    scores: pd.DataFrame,
    pair_audit: pd.DataFrame,
    *,
    replay_end: pd.Timestamp,
    minimum_pairs: int,
) -> tuple[pd.Timestamp, Any, pd.DataFrame]:
    score_lookup = scores.set_index("reference_month")
    first = pd.Timestamp(scores["reference_month"].min())
    candidate = (first.to_period("M") + 3).to_timestamp()
    while candidate <= replay_end:
        oldest = (candidate.to_period("M") - 3).to_timestamp()
        if oldest in score_lookup.index:
            available = pd.Timestamp(score_lookup.loc[oldest, "score_available_at"])
            if available < candidate:
                try:
                    fit, selected = _causal_var_fit(
                        pair_audit,
                        source_reference_month=(candidate.to_period("M") - 1).to_timestamp(),
                        cutoff=candidate,
                        minimum_pairs=minimum_pairs,
                    )
                except ValueError:
                    pass
                else:
                    return candidate, fit, selected
        candidate = (candidate.to_period("M") + 1).to_timestamp()
    raise ValueError("no feasible four-month initialization exists before replay_end")


def _active_controls(
    training: pd.DataFrame,
    configured: Sequence[str],
    *,
    cutoff: pd.Timestamp,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    causal = training.loc[training["training_available_at"] < cutoff]
    active: list[str] = []
    dropped: list[str] = []
    for name in configured:
        values = causal[name].to_numpy(dtype=float)
        if len(values) and np.isfinite(values).all() and float(np.ptp(values)) > 1.0e-12:
            active.append(name)
        else:
            dropped.append(name)
    return tuple(active), tuple(dropped)


def _fit_signature(
    training: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    active_controls: Sequence[str],
) -> tuple[str, int, pd.Timestamp | None]:
    causal = training.loc[training["training_available_at"] < cutoff].sort_values(
        ["training_available_at", "event_instance_id"], kind="mergesort"
    )
    digest = hashlib.sha256()
    for row in causal.itertuples(index=False):
        digest.update(str(row.event_instance_id).encode("utf-8"))
        digest.update(b"\0")
    digest.update("|".join(active_controls).encode("utf-8"))
    latest = (
        None
        if causal.empty
        else pd.Timestamp(causal["training_available_at"].max())
    )
    return digest.hexdigest(), int(len(causal)), latest


def _fit_record(
    fit: LinearGaussianEmissionFit,
    *,
    fit_id: str,
    fit_date: pd.Timestamp,
    signature: str,
    active_controls: Sequence[str],
    dropped_controls: Sequence[str],
    distinct_target_months: int,
) -> dict[str, object]:
    audit = fit.to_audit_dict()
    return {
        "fit_id": fit_id,
        "fit_date": fit_date,
        "observation_model_id": fit.spec.block_id,
        "training_signature": signature,
        "training_count": fit.training_count,
        "distinct_target_months": int(distinct_target_months),
        "first_training_available_at": fit.first_training_available_at,
        "last_training_available_at": fit.last_training_available_at,
        "selected_base_lambda": fit.selected_base_lambda,
        "active_controls": "|".join(active_controls),
        "dropped_unidentified_controls": "|".join(dropped_controls),
        "ledoit_wolf_shrinkage": fit.ledoit_wolf_shrinkage,
        "fit_audit_json": json.dumps(audit, sort_keys=True, separators=(",", ":")),
    }


def _state_record(
    checkpoint_id: str,
    variant: str,
    state: JointGaussianState,
) -> dict[str, object]:
    record: dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "filter_variant": variant,
    }
    for position, month in enumerate(state.reference_months):
        record[f"reference_month_{position}"] = month
        record[f"growth_mean_{position}"] = float(state.mean[2 * position])
        record[f"inflation_mean_{position}"] = float(state.mean[2 * position + 1])
        record[f"exact_{position}"] = bool(state.exact_mask[position])
    for row in range(8):
        for column in range(8):
            record[f"covariance_{row}_{column}"] = float(state.covariance[row, column])
    return record


def _stable_seed(checkpoint_id: str, variant: str) -> int:
    digest = hashlib.sha256(f"{checkpoint_id}|{variant}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _probability_metrics(
    forecast: Mapping[str, float], target: Mapping[str, float]
) -> tuple[float, float, float]:
    forecast_values = np.asarray([forecast[name] for name in REGIME_ORDER], dtype=float)
    target_values = np.asarray([target[name] for name in REGIME_ORDER], dtype=float)
    safe_forecast = np.clip(forecast_values, 1.0e-15, 1.0)
    safe_target = np.clip(target_values, 1.0e-15, 1.0)
    cross_entropy = -float(np.sum(target_values * np.log(safe_forecast)))
    brier = float(np.sum(np.square(forecast_values - target_values)))
    kl = float(np.sum(target_values * (np.log(safe_target) - np.log(safe_forecast))))
    return cross_entropy, brier, kl


def _score_log_density(
    score: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> float:
    covariance = (covariance + covariance.T) / 2.0
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        return math.nan
    error = score - mean
    return float(
        0.5
        * (
            2.0 * math.log(2.0 * math.pi)
            + logdet
            + error @ np.linalg.solve(covariance, error)
        )
    )


def _evaluation_record(
    *,
    state: JointGaussianState,
    availability_date: pd.Timestamp,
    reference_month: pd.Timestamp,
    exact_score: np.ndarray,
    filter_variant: str,
    evaluation_checkpoint: str,
    same_day_release_evidence_included: bool,
    forecast_mapping: pd.Series | None,
    target_mapping: pd.Series | None,
) -> dict[str, object] | None:
    """Score one non-exact forecast against a newly completed score center.

    The primary checkpoint is captured before any data release carrying the
    score's availability date.  A second, explicitly named sensitivity may
    include release blocks from that date before the exact score is
    conditioned.  This distinction is required because the source artifacts
    contain dates, not reliable intraday timestamps.
    """

    if (
        reference_month < state.reference_months[0]
        or reference_month > state.reference_months[-1]
    ):
        return None
    position = state.position(reference_month)
    if state.exact_mask[position]:
        return None
    mean, covariance = state.marginal(reference_month)
    evaluation: dict[str, object] = {
        "availability_date": availability_date,
        "reference_month": reference_month,
        "filter_variant": filter_variant,
        "evaluation_checkpoint": evaluation_checkpoint,
        "same_day_release_evidence_included": bool(
            same_day_release_evidence_included
        ),
        "growth_error": float(mean[0] - exact_score[0]),
        "inflation_error": float(mean[1] - exact_score[1]),
        "growth_squared_error": float((mean[0] - exact_score[0]) ** 2),
        "inflation_squared_error": float((mean[1] - exact_score[1]) ** 2),
        "score_center_negative_log_predictive_density": _score_log_density(
            exact_score, mean, covariance
        ),
    }
    if forecast_mapping is not None and target_mapping is not None:
        forecast = monthly_quadrant_probabilities(
            state, reference_month, _mapping_covariance(forecast_mapping)
        )
        _, target = gaussian_quadrant_weights(
            exact_score, _mapping_covariance(target_mapping)
        )
        cross_entropy, brier, kl = _probability_metrics(forecast, target)
        evaluation.update(
            {
                "quadrant_cross_entropy_to_exact_score_map": cross_entropy,
                "quadrant_brier_distance_to_exact_score_map": brier,
                "quadrant_kl_divergence_to_exact_score_map": kl,
                "forecast_mapping_reference_month": forecast_mapping[
                    "reference_month"
                ],
                "target_mapping_reference_month": target_mapping[
                    "reference_month"
                ],
            }
        )
        for regime in REGIME_ORDER:
            evaluation[f"forecast_probability_{regime}"] = forecast[regime]
            evaluation[f"target_probability_{regime}"] = target[regime]
    return evaluation


def run_event_driven_filter(
    score_history: pd.DataFrame,
    mapping_history: pd.DataFrame,
    prepared: PreparedObservationData,
    config: Mapping[str, Any],
    *,
    replay_end: object | None = None,
    apply_evidence: bool = True,
    store_joint_paths: bool = True,
    path_checkpoint_types: Sequence[str] = (
        "initial",
        "post_month_roll",
        "pre_exact_score",
        "month_end",
        "latest",
    ),
) -> GaussianWalkForwardResult:
    """Replay the evidence filter and its transition-only baseline causally."""

    scores = _normalize_scores(score_history)
    transition_config = config["transition"]
    minimum_pairs = int(transition_config["minimum_training_pairs"])
    if minimum_pairs < 5:
        raise ValueError("transition minimum_training_pairs must be at least five")
    mapping_config = config["mapping"]
    mappings = _normalize_mapping_history(
        mapping_history,
        baseline_revision_horizon_months=int(
            mapping_config["baseline_revision_horizon_months"]
        ),
    )
    configured_end = config.get("calendar", {}).get("replay_end")
    end = _naive_timestamp(
        replay_end if replay_end is not None else configured_end,
        label="replay_end",
        normalize=True,
    )
    pair_audit = build_var_pair_audit(scores.set_index("reference_month"))
    start, initial_fit, initial_pairs = _earliest_initialization(
        scores,
        pair_audit,
        replay_end=end,
        minimum_pairs=minimum_pairs,
    )
    score_lookup = scores.set_index("reference_month")
    oldest = (start.to_period("M") - 3).to_timestamp()
    oldest_score = score_lookup.loc[oldest, list(_STATE_NAMES)].to_numpy(dtype=float)
    edge = _dynamics(initial_fit)
    initial_state = initialize_joint_exact(oldest, oldest_score, [edge, edge, edge])
    # Condition every later path month whose score was public before the replay
    # began.  Same-day scores are deliberately left for end-of-day processing.
    for month in initial_state.reference_months[1:]:
        if month in score_lookup.index:
            row = score_lookup.loc[month]
            if pd.Timestamp(row["score_available_at"]) < start:
                initial_state = condition_on_exact_score(
                    initial_state, month, row.loc[list(_STATE_NAMES)].to_numpy(dtype=float)
                )
    states: dict[str, JointGaussianState] = {
        "evidence_filter": initial_state,
        "transition_only": initial_state,
    }

    events = prepared.events.copy()
    if not events.empty:
        events = events.loc[
            events["release_date"].between(start, end, inclusive="both")
        ].copy()
    exact_rows = scores.loc[
        scores["score_available_at"].between(start, end, inclusive="both")
    ].copy()
    event_dates = set(events["release_date"]) if not events.empty else set()
    exact_dates = set(exact_rows["score_available_at"])
    month_starts = set(pd.date_range(start=start, end=end, freq="MS"))
    month_ends = set(pd.date_range(start=start, end=end, freq="ME").normalize())
    timeline = sorted({start, end}.union(event_dates, exact_dates, month_starts, month_ends))

    checkpoint_records: list[dict[str, object]] = []
    state_records: list[dict[str, object]] = []
    marginal_records: list[dict[str, object]] = []
    path_records: list[dict[str, object]] = []
    event_records: list[dict[str, object]] = []
    exact_records: list[dict[str, object]] = []
    transition_records: list[dict[str, object]] = []
    emission_records: list[dict[str, object]] = []
    evaluation_records: list[dict[str, object]] = []
    residual_records: list[dict[str, object]] = []
    fit_cache: dict[tuple[str, str, tuple[str, ...]], LinearGaussianEmissionFit] = {}
    fit_id_cache: dict[tuple[str, str, tuple[str, ...]], str] = {}
    sequence = 0
    parent_checkpoint: str | None = None

    def record_transition(
        fit: Any,
        selected: pd.DataFrame,
        *,
        fit_date: pd.Timestamp,
        fit_kind: str,
    ) -> None:
        transition_records.append(
            {
                "transition_fit_id": f"var1:{fit_date:%Y-%m-%d}:{fit_kind}",
                "fit_date": fit_date,
                "fit_kind": fit_kind,
                "strict_training_cutoff": fit_date,
                "training_pairs": int(len(selected)),
                "first_pair_available_at": selected["pair_available_at"].min(),
                "last_pair_available_at": selected["pair_available_at"].max(),
                "latest_destination_reference_month": selected[
                    "destination_reference_month"
                ].max(),
                "intercept_growth": float(fit.intercept[0]),
                "intercept_inflation": float(fit.intercept[1]),
                "a_growth_growth": float(fit.transition[0, 0]),
                "a_growth_inflation": float(fit.transition[0, 1]),
                "a_inflation_growth": float(fit.transition[1, 0]),
                "a_inflation_inflation": float(fit.transition[1, 1]),
                "q_growth": float(fit.innovation_covariance[0, 0]),
                "q_growth_inflation": float(fit.innovation_covariance[0, 1]),
                "q_inflation": float(fit.innovation_covariance[1, 1]),
                "spectral_radius": float(fit.spectral_radius),
                "design_condition_number": float(fit.design_condition_number),
            }
        )

    record_transition(initial_fit, initial_pairs, fit_date=start, fit_kind="initialization")

    def record_checkpoint(
        checkpoint_type: str,
        as_of: pd.Timestamp,
        *,
        phase_order: int,
        allow_same_day_mapping: bool,
        event_models: Sequence[str] = (),
        exact_months: Sequence[pd.Timestamp] = (),
    ) -> str:
        nonlocal sequence, parent_checkpoint
        sequence += 1
        checkpoint_id = f"m02:{sequence:06d}"
        mapping_row = _mapping_asof(
            mappings, as_of, allow_same_day=allow_same_day_mapping
        )
        checkpoint_records.append(
            {
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": parent_checkpoint,
                "as_of_date": as_of,
                "phase_order": int(phase_order),
                "checkpoint_type": checkpoint_type,
                "event_models": "|".join(sorted(set(event_models))),
                "exact_months": "|".join(f"{month:%Y-%m}" for month in sorted(exact_months)),
                "same_day_mapping_allowed": bool(allow_same_day_mapping),
                "mapping_reference_month": (
                    pd.NaT if mapping_row is None else mapping_row["reference_month"]
                ),
                "mapping_available_at": (
                    pd.NaT if mapping_row is None else mapping_row["score_available_at"]
                ),
                "mapping_status": "unavailable" if mapping_row is None else "available",
            }
        )
        omega = None if mapping_row is None else _mapping_covariance(mapping_row)
        for variant, state in states.items():
            state_records.append(_state_record(checkpoint_id, variant, state))
            for position, month in enumerate(state.reference_months):
                mean, latent_covariance = state.marginal(month)
                marginal: dict[str, object] = {
                    "checkpoint_id": checkpoint_id,
                    "filter_variant": variant,
                    "as_of_date": as_of,
                    "checkpoint_type": checkpoint_type,
                    "reference_month": month,
                    "relative_month": position - 3,
                    "growth_mean": float(mean[0]),
                    "inflation_mean": float(mean[1]),
                    "growth_latent_variance": float(latent_covariance[0, 0]),
                    "growth_inflation_latent_covariance": float(latent_covariance[0, 1]),
                    "inflation_latent_variance": float(latent_covariance[1, 1]),
                    "exact_score_center": bool(state.exact_mask[position]),
                    "mapping_status": "unavailable" if omega is None else "available",
                }
                if omega is not None:
                    probabilities = monthly_quadrant_probabilities(state, month, omega)
                    entropy = -sum(
                        value * math.log(value) for value in probabilities.values() if value > 0
                    )
                    marginal["entropy"] = float(entropy)
                    for regime in REGIME_ORDER:
                        marginal[f"probability_{regime}"] = probabilities[regime]
                marginal_records.append(marginal)

            if (
                store_joint_paths
                and omega is not None
                and checkpoint_type in set(path_checkpoint_types)
            ):
                sample_count = int(mapping_config["joint_path_samples"])
                path = joint_quadrant_path_probabilities(
                    state,
                    omega,
                    sobol_samples=sample_count,
                    seed=_stable_seed(checkpoint_id, variant),
                )
                analytic = [
                    monthly_quadrant_probabilities(state, month, omega)
                    for month in state.reference_months
                ]
                sampled = [dict.fromkeys(REGIME_ORDER, 0.0) for _ in range(4)]
                for regimes, probability in path.items():
                    for position, regime in enumerate(regimes):
                        sampled[position][regime] += probability
                error = max(
                    abs(sampled[position][regime] - analytic[position][regime])
                    for position in range(4)
                    for regime in REGIME_ORDER
                )
                for regimes, probability in path.items():
                    path_records.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "filter_variant": variant,
                            "as_of_date": as_of,
                            "checkpoint_type": checkpoint_type,
                            "regime_0": regimes[0],
                            "regime_1": regimes[1],
                            "regime_2": regimes[2],
                            "regime_3": regimes[3],
                            "probability": float(probability),
                            "sobol_samples": sample_count,
                            "maximum_marginal_probability_error": float(error),
                        }
                    )
        parent_checkpoint = checkpoint_id
        return checkpoint_id

    record_checkpoint(
        "initial", start, phase_order=0, allow_same_day_mapping=False
    )

    # Audit exact scores already incorporated at initialization.
    for position, is_exact in enumerate(initial_state.exact_mask):
        if is_exact:
            exact_records.append(
                {
                    "availability_date": start,
                    "reference_month": initial_state.reference_months[position],
                    "filter_variant": "both",
                    "status": "initialized_from_prior_information",
                }
            )

    for current_date in timeline:
        same_day_exact_processed = False
        if current_date > start and current_date in month_starts:
            record_checkpoint(
                "pre_month_roll",
                current_date,
                phase_order=0,
                allow_same_day_mapping=False,
            )
            source_month = states["evidence_filter"].reference_months[-1]
            fit, selected = _causal_var_fit(
                pair_audit,
                source_reference_month=source_month,
                cutoff=current_date,
                minimum_pairs=minimum_pairs,
            )
            dynamics = _dynamics(fit)
            for variant in FILTER_VARIANTS:
                states[variant] = roll_joint_gaussian(states[variant], dynamics)
            record_transition(fit, selected, fit_date=current_date, fit_kind="month_roll")
            record_checkpoint(
                "post_month_roll",
                current_date,
                phase_order=1,
                allow_same_day_mapping=False,
            )

        date_scores = exact_rows.loc[
            exact_rows["score_available_at"] == current_date
        ].sort_values("reference_month")
        pre_day_states = dict(states) if not date_scores.empty else {}
        if not date_scores.empty:
            record_checkpoint(
                "pre_exact_score_day",
                current_date,
                phase_order=2,
                allow_same_day_mapping=False,
                exact_months=tuple(
                    pd.Timestamp(value) for value in date_scores["reference_month"]
                ),
            )

        date_events = (
            events.loc[events["release_date"] == current_date].copy()
            if not events.empty
            else pd.DataFrame()
        )
        if not date_events.empty:
            # Dependence diagnostics must not depend on the arbitrary stable
            # order used to multiply same-day likelihood factors.  Every
            # diagnostic innovation on this date is therefore measured from
            # the shared pre-release state, while the actual filter updates
            # remain sequential (and algebraically order-invariant under the
            # conditional-independence baseline).
            diagnostic_pre_release_state = states["evidence_filter"]
            models_today = tuple(sorted(date_events["observation_model_id"].unique()))
            record_checkpoint(
                "pre_release_group",
                current_date,
                phase_order=3,
                allow_same_day_mapping=False,
                event_models=models_today,
            )
            fits_today: dict[str, LinearGaussianEmissionFit | None] = {}
            fit_status: dict[str, str] = {}
            fit_ids: dict[str, str] = {}
            dropped_today: dict[str, tuple[str, ...]] = {}
            distinct_months_today: dict[str, int] = {}
            required_distinct_today: dict[str, int] = {}
            for model_id in models_today:
                base_spec = prepared.specifications[model_id]
                training = prepared.training_tables[model_id]
                active, dropped = _active_controls(
                    training, base_spec.control_names, cutoff=current_date
                )
                dropped_today[model_id] = dropped
                spec = replace(base_spec, control_names=active)
                signature, count, _ = _fit_signature(
                    training, cutoff=current_date, active_controls=active
                )
                causal_training = training.loc[
                    training["training_available_at"] < current_date
                ]
                distinct_target_months = int(
                    causal_training["reference_month"].nunique()
                )
                required_distinct_months = int(
                    config["observation_models"][model_id].get(
                        "minimum_distinct_target_months", 0
                    )
                )
                distinct_months_today[model_id] = distinct_target_months
                required_distinct_today[model_id] = required_distinct_months
                cache_key = (model_id, signature, active)
                if (
                    count < spec.minimum_training_samples
                    or distinct_target_months < required_distinct_months
                ):
                    fits_today[model_id] = None
                    fit_status[model_id] = (
                        "insufficient_distinct_target_months"
                        if distinct_target_months < required_distinct_months
                        else "insufficient_training_history"
                    )
                    continue
                if cache_key in fit_cache:
                    fits_today[model_id] = fit_cache[cache_key]
                    fit_ids[model_id] = fit_id_cache[cache_key]
                    fit_status[model_id] = "cached_fit"
                    continue
                try:
                    fit = fit_linear_gaussian_emission(
                        training,
                        spec=spec,
                        availability_column="training_available_at",
                        knowledge_cutoff=current_date,
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    fits_today[model_id] = None
                    fit_status[model_id] = f"fit_error:{error}"
                    continue
                fit_id = f"emission:{model_id}:{len(emission_records) + 1:06d}"
                fit_cache[cache_key] = fit
                fit_id_cache[cache_key] = fit_id
                fits_today[model_id] = fit
                fit_ids[model_id] = fit_id
                fit_status[model_id] = "new_fit"
                emission_records.append(
                    _fit_record(
                        fit,
                        fit_id=fit_id,
                        fit_date=current_date,
                        signature=signature,
                        active_controls=active,
                        dropped_controls=dropped,
                        distinct_target_months=distinct_target_months,
                    )
                )

            for row in date_events.sort_values(
                [
                    "observation_model_id",
                    "reference_date",
                    "reference_month",
                    "event_instance_id",
                ]
            ).itertuples(index=False):
                model_id = str(row.observation_model_id)
                reference_month = pd.Timestamp(row.reference_month)
                fit = fits_today[model_id]
                audit: dict[str, object] = {
                    "release_date": current_date,
                    "observation_model_id": model_id,
                    "economic_block": row.economic_block,
                    "event_instance_id": row.event_instance_id,
                    "event_ids": row.event_ids,
                    "reference_month": reference_month,
                    "reference_date": pd.Timestamp(row.reference_date),
                    "fit_status": fit_status[model_id],
                    "fit_id": fit_ids.get(model_id, ""),
                    "dropped_unidentified_controls": "|".join(dropped_today[model_id]),
                    "distinct_target_months": distinct_months_today[model_id],
                    "required_distinct_target_months": required_distinct_today[model_id],
                    "residual_recorded": False,
                }
                state = states["evidence_filter"]
                if reference_month < state.reference_months[0]:
                    audit["update_status"] = "expired_target_outside_path"
                    event_records.append(audit)
                    continue
                if reference_month > state.reference_months[-1]:
                    audit["update_status"] = "future_target_outside_path"
                    event_records.append(audit)
                    continue
                if not apply_evidence:
                    audit["update_status"] = "evidence_suppressed"
                    event_records.append(audit)
                    continue
                if fit is None:
                    audit["update_status"] = (
                        "fit_error"
                        if fit_status[model_id].startswith("fit_error:")
                        else fit_status[model_id]
                    )
                    event_records.append(audit)
                    continue
                observation = {
                    name: getattr(row, name)
                    for name in fit.spec.response_names
                    if pd.notna(getattr(row, name))
                }
                if not observation:
                    audit["update_status"] = "no_observed_response"
                    event_records.append(audit)
                    continue
                controls = {
                    name: getattr(row, name) for name in fit.spec.control_names
                }
                if any(pd.isna(value) for value in controls.values()):
                    audit["update_status"] = "missing_active_control"
                    event_records.append(audit)
                    continue
                try:
                    system = fit.observed_system(
                        observation,
                        controls=(controls if fit.spec.control_names else None),
                    )
                    update = update_joint_gaussian(
                        state,
                        reference_month,
                        **system.to_joint_filter_inputs(),
                    )
                    diagnostic_update = update_joint_gaussian(
                        diagnostic_pre_release_state,
                        reference_month,
                        **system.to_joint_filter_inputs(),
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    audit["update_status"] = f"update_error:{error}"
                    event_records.append(audit)
                    continue
                states["evidence_filter"] = update.state
                audit.update(
                    {
                        "update_status": (
                            "applied" if update.applied else "target_exact_predictive_only"
                        ),
                        "observed_responses": "|".join(system.observed_names),
                        "omitted_responses": "|".join(system.omitted_names),
                        "log_predictive_density": update.log_predictive_density,
                        "diagnostic_log_predictive_density": (
                            diagnostic_update.log_predictive_density
                        ),
                        "diagnostic_state_cutoff": "shared_pre_release_group",
                        "residual_recorded": True,
                        "innovation_json": json.dumps(update.innovation.tolist()),
                        "innovation_covariance_json": json.dumps(
                            update.innovation_covariance.tolist()
                        ),
                        "diagnostic_innovation_json": json.dumps(
                            diagnostic_update.innovation.tolist()
                        ),
                        "diagnostic_innovation_covariance_json": json.dumps(
                            diagnostic_update.innovation_covariance.tolist()
                        ),
                    }
                )
                event_records.append(audit)
                whitened = np.linalg.solve(
                    np.linalg.cholesky(diagnostic_update.innovation_covariance),
                    diagnostic_update.innovation,
                )
                predictive_sd = np.sqrt(
                    np.diag(diagnostic_update.innovation_covariance)
                )
                for position, response_id in enumerate(system.observed_names):
                    residual_records.append(
                        {
                            "release_date": current_date,
                            "reference_month": reference_month,
                            "observation_date": pd.Timestamp(row.reference_date),
                            "observation_model_id": model_id,
                            "block_id": model_id,
                            "economic_block": row.economic_block,
                            "event_instance_id": row.event_instance_id,
                            "response_id": response_id,
                            "source_series_id": getattr(
                                row, f"{response_id}__series_id", ""
                            ),
                            "innovation": float(diagnostic_update.innovation[position]),
                            "predictive_standard_deviation": float(predictive_sd[position]),
                            "standardized_innovation": float(
                                diagnostic_update.innovation[position]
                                / predictive_sd[position]
                            ),
                            "whitened_innovation": float(whitened[position]),
                            "state_update_applied": bool(update.applied),
                            "diagnostic_state_cutoff": "shared_pre_release_group",
                        }
                    )
            record_checkpoint(
                "post_release_group",
                current_date,
                phase_order=4,
                allow_same_day_mapping=False,
                event_models=models_today,
            )

        if not date_scores.empty:
            exact_months = tuple(pd.Timestamp(value) for value in date_scores["reference_month"])
            record_checkpoint(
                "pre_exact_score",
                current_date,
                phase_order=5,
                allow_same_day_mapping=False,
                exact_months=exact_months,
            )
            pre_mapping = _mapping_asof(mappings, current_date, allow_same_day=False)
            same_day_evidence_applied = any(
                record.get("release_date") == current_date
                and record.get("update_status") == "applied"
                for record in event_records
            )
            # Freeze the post-release/pre-exact information set for the whole
            # atomic score batch.  Otherwise conditioning the first of two
            # same-day scores would leak it into the second score's sensitivity
            # evaluation.
            post_release_states = dict(states)
            for row in date_scores.itertuples(index=False):
                reference_month = pd.Timestamp(row.reference_month)
                exact_score = np.asarray([row.growth_score, row.inflation_score], dtype=float)
                target_mapping = _mapping_for_exact_score(
                    mappings, reference_month, current_date
                )
                for variant in FILTER_VARIANTS:
                    primary = _evaluation_record(
                        state=pre_day_states[variant],
                        availability_date=current_date,
                        reference_month=reference_month,
                        exact_score=exact_score,
                        filter_variant=variant,
                        evaluation_checkpoint="strict_pre_day",
                        same_day_release_evidence_included=False,
                        forecast_mapping=pre_mapping,
                        target_mapping=target_mapping,
                    )
                    if primary is not None:
                        evaluation_records.append(primary)
                    state = states[variant]
                    audit: dict[str, object] = {
                        "availability_date": current_date,
                        "reference_month": reference_month,
                        "filter_variant": variant,
                    }
                    if reference_month < state.reference_months[0]:
                        audit["status"] = "expired_target_outside_path"
                        exact_records.append(audit)
                        continue
                    if reference_month > state.reference_months[-1]:
                        audit["status"] = "future_target_outside_path"
                        exact_records.append(audit)
                        continue
                    position = state.position(reference_month)
                    if not state.exact_mask[position]:
                        sensitivity = _evaluation_record(
                            state=post_release_states[variant],
                            availability_date=current_date,
                            reference_month=reference_month,
                            exact_score=exact_score,
                            filter_variant=variant,
                            evaluation_checkpoint=(
                                "post_release_pre_exact_sensitivity"
                            ),
                            same_day_release_evidence_included=(
                                variant == "evidence_filter"
                                and same_day_evidence_applied
                            ),
                            forecast_mapping=pre_mapping,
                            target_mapping=target_mapping,
                        )
                        if sensitivity is not None:
                            evaluation_records.append(sensitivity)
                    try:
                        conditioned = condition_on_exact_score(
                            state, reference_month, exact_score
                        )
                    except ValueError as error:
                        audit["status"] = f"conditioning_error:{error}"
                    else:
                        audit["status"] = (
                            "identical_exact_no_op"
                            if conditioned is state
                            else "conditioned_exactly"
                        )
                        states[variant] = conditioned
                    exact_records.append(audit)
            same_day_exact_processed = True
            record_checkpoint(
                "post_exact_score",
                current_date,
                phase_order=6,
                allow_same_day_mapping=True,
                exact_months=exact_months,
            )

        if current_date in month_ends:
            record_checkpoint(
                "month_end",
                current_date,
                phase_order=7,
                allow_same_day_mapping=same_day_exact_processed,
            )

    record_checkpoint(
        "latest",
        end,
        phase_order=8,
        allow_same_day_mapping=(
            bool((exact_rows["score_available_at"] == end).any())
        ),
    )
    return GaussianWalkForwardResult(
        checkpoints=pd.DataFrame.from_records(checkpoint_records),
        joint_gaussians=pd.DataFrame.from_records(state_records),
        marginals=pd.DataFrame.from_records(marginal_records),
        joint_paths=pd.DataFrame.from_records(path_records),
        event_audit=pd.DataFrame.from_records(event_records),
        exact_score_audit=pd.DataFrame.from_records(exact_records),
        transition_fit_audit=pd.DataFrame.from_records(transition_records),
        emission_fit_audit=pd.DataFrame.from_records(emission_records),
        evaluation_rows=pd.DataFrame.from_records(evaluation_records),
        predictive_residuals=pd.DataFrame.from_records(residual_records),
        latest_states=dict(states),
        initial_date=start,
        replay_end=end,
    )
