"""Fit and publish Model 01's causal stationary transition model."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from regime_allocation.models.m01_deterministic_composite.pipeline import (
    REGIME_LABELS,
    REGIME_ORDER,
)
from regime_allocation.models.m01_deterministic_composite.transition import (
    REGIME_IDS,
    TransitionEstimate,
    estimate_transition_matrix,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("transition configuration must be a mapping")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported transition configuration schema")
    if str(config.get("model_id")) != "m01_deterministic_composite":
        raise ValueError("transition configuration has the wrong model_id")

    transition = config.get("transition")
    if not isinstance(transition, dict):
        raise ValueError("transition configuration must contain a transition mapping")
    frozen_values = {
        "markov_order": 1,
        "path_months": 4,
        "time_homogeneous": True,
        "estimation_window": "expanding",
        "knowledge_cutoff_policy": "latest_label_available_at",
        "require_consecutive_reference_months": True,
        "availability_field": "label_available_at",
    }
    for key, expected in frozen_values.items():
        if transition.get(key) != expected:
            raise ValueError(
                f"Model 01 transition setting {key!r} must equal {expected!r}"
            )
    alpha = float(transition.get("dirichlet_alpha", float("nan")))
    if not math.isclose(alpha, 0.5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Model 01 requires dirichlet_alpha=0.5")
    interval = float(transition.get("credible_interval_level", float("nan")))
    if not math.isclose(interval, 0.95, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Model 01 requires credible_interval_level=0.95")

    source = config.get("source")
    outputs = config.get("outputs")
    if not isinstance(source, dict) or not source.get("regime_history"):
        raise ValueError("transition configuration must identify source regime history")
    required_outputs = {
        "processed_pairs",
        "published_model",
        "published_matrix",
        "published_prior",
    }
    if not isinstance(outputs, dict) or not required_outputs.issubset(outputs):
        raise ValueError("transition configuration is missing required output paths")
    return config, raw


def _nested_frame(
    frame: pd.DataFrame,
    *,
    integer: bool = False,
) -> dict[str, dict[str, int | float | None]]:
    payload: dict[str, dict[str, int | float | None]] = {}
    for source in REGIME_IDS:
        payload[source] = {}
        for destination in REGIME_IDS:
            value = frame.loc[source, destination]
            if pd.isna(value):
                converted: int | float | None = None
            elif integer:
                converted = int(value)
            else:
                converted = float(value)
            payload[source][destination] = converted
    return payload


def _derive_cutoff(history: pd.DataFrame) -> date:
    if "label_available_at" not in history:
        raise ValueError("regime history is missing label_available_at")
    availability = pd.to_datetime(history["label_available_at"], errors="coerce")
    invalid = history["label_available_at"].notna() & availability.isna()
    if invalid.any() or availability.notna().sum() == 0:
        raise ValueError("regime history has no valid label availability cutoff")
    return pd.Timestamp(availability.max()).date()


def _latest_transition_prior(
    history: pd.DataFrame,
    estimate: TransitionEstimate,
    *,
    model_id: str,
    stage_id: str,
) -> dict[str, Any]:
    available = history.copy()
    available["reference_month"] = pd.to_datetime(
        available["reference_month"], errors="coerce"
    )
    available["label_available_at"] = pd.to_datetime(
        available["label_available_at"], errors="coerce"
    )
    available = available.loc[
        available["regime_id"].notna()
        & available["label_available_at"].notna()
        & (available["label_available_at"] <= estimate.knowledge_cutoff)
    ].sort_values("reference_month")
    if available.empty:
        raise ValueError("no confirmed regime is available at the knowledge cutoff")

    latest = available.iloc[-1]
    source_month = pd.Timestamp(latest["reference_month"])
    target_month = (source_month.to_period("M") + 1).to_timestamp()
    source_id = str(latest["regime_id"])
    source_regime = next(regime for regime in REGIME_ORDER if regime.value == source_id)
    probabilities = estimate.posterior_predictive.loc[source_id]
    return {
        "schema_version": 1,
        "model_id": model_id,
        "stage_id": stage_id,
        "status": "transition_only_prior",
        "knowledge_cutoff": estimate.knowledge_cutoff.date().isoformat(),
        "source_reference_month": source_month.date().isoformat(),
        "target_reference_month": target_month.date().isoformat(),
        "conditioning_regime_id": source_id,
        "conditioning_regime_label": REGIME_LABELS[source_regime],
        "probabilities": [
            {
                "regime_id": regime_id,
                "regime_label": REGIME_LABELS[
                    next(regime for regime in REGIME_ORDER if regime.value == regime_id)
                ],
                "probability": float(probabilities.loc[regime_id]),
            }
            for regime_id in REGIME_IDS
        ],
        "probability_sum": float(probabilities.sum()),
        "interpretation": (
            "One-step transition prior conditioned on the latest confirmed "
            "deterministic regime; no event likelihood has been applied."
        ),
    }


def build_transition(
    *,
    project_root: Path,
    config_path: Path,
    knowledge_cutoff: date | None = None,
) -> dict[str, Path]:
    """Estimate and atomically publish the Model 01 transition artifacts."""

    config, config_bytes = _load_config(config_path)
    history_path = project_root / str(config["source"]["regime_history"])
    history_bytes = history_path.read_bytes()
    history = pd.read_csv(history_path)
    cutoff = knowledge_cutoff or _derive_cutoff(history)
    transition_config = config["transition"]
    estimate = estimate_transition_matrix(
        history,
        knowledge_cutoff=cutoff,
        alpha=float(transition_config["dirichlet_alpha"]),
        credible_interval_level=float(
            transition_config["credible_interval_level"]
        ),
    )

    outputs = config["outputs"]
    pairs_path = project_root / str(outputs["processed_pairs"])
    model_path = project_root / str(outputs["published_model"])
    matrix_path = project_root / str(outputs["published_matrix"])
    prior_path = project_root / str(outputs["published_prior"])
    matrix = estimate.to_long_frame()
    prior = _latest_transition_prior(
        history,
        estimate,
        model_id=str(config["model_id"]),
        stage_id=str(config["stage_id"]),
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    model_payload = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "stage_id": config["stage_id"],
        "generated_at_utc": generated_at,
        "source": {
            "regime_history": history_path.relative_to(project_root).as_posix(),
            "regime_history_sha256": _sha256(history_bytes),
            "history_rows": len(history),
        },
        "configuration": {
            "path": config_path.relative_to(project_root).as_posix(),
            "sha256": _sha256(config_bytes),
        },
        "transition_specification": dict(transition_config),
        "knowledge_cutoff": estimate.knowledge_cutoff.date().isoformat(),
        "state_order": [
            {
                "regime_id": regime.value,
                "regime_label": REGIME_LABELS[regime],
            }
            for regime in REGIME_ORDER
        ],
        "diagnostics": estimate.diagnostics,
        "counts": _nested_frame(estimate.counts, integer=True),
        "mle_probabilities": _nested_frame(estimate.mle_probabilities),
        "posterior_parameters": _nested_frame(estimate.posterior_parameters),
        "posterior_predictive": _nested_frame(estimate.posterior_predictive),
        "marginal_credible_intervals": {
            "level": estimate.credible_interval_level,
            "lower": _nested_frame(estimate.ci_lower),
            "upper": _nested_frame(estimate.ci_upper),
            "interpretation": (
                "Marginal credible intervals for individual transition-matrix "
                "parameters under each row's Dirichlet posterior."
            ),
        },
        "published_files": {
            "transition_matrix": matrix_path.relative_to(project_root).as_posix(),
            "latest_transition_prior": prior_path.relative_to(project_root).as_posix(),
        },
    }

    _write_csv(estimate.pairs, pairs_path)
    _write_csv(matrix, matrix_path)
    _write_json(prior, prior_path)
    _write_json(model_payload, model_path)
    return {
        "model": model_path,
        "matrix": matrix_path,
        "pairs": pairs_path,
        "prior": prior_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m01_deterministic_composite_transition.yaml"),
        help="Transition configuration path, relative to project root by default.",
    )
    parser.add_argument(
        "--knowledge-cutoff",
        type=date.fromisoformat,
        default=None,
        help="Optional YYYY-MM-DD causal cutoff; defaults to latest label availability.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    outputs = build_transition(
        project_root=project_root,
        config_path=config_path,
        knowledge_cutoff=args.knowledge_cutoff,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
