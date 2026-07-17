"""Acquire and prepare Model 01's non-defining release evidence.

This command stops at the data boundary: it freezes first-release observations,
constructs causal release features, and writes a canonical event table.  It does
not estimate likelihoods or perform Bayesian filtering.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from regime_allocation.cli.build_m01_dataset import (
    _as_date,
    _download_or_load,
    _provider_raw_dir,
    _sha256,
    _write_csv,
    _write_json,
)
from regime_allocation.data.first_release import extract_first_release_features
from regime_allocation.data.providers import select_vintage_provider
from regime_allocation.data.providers.vintage_matrix import (
    FirstReleaseObservation,
    first_release_observations_from_matrix,
    load_vintage_matrix,
)
from regime_allocation.features.release_evidence import (
    build_claims_release_events,
    build_monthly_release_events,
    validate_event_table,
)


_EXPECTED_BLOCKS: dict[str, dict[str, object]] = {
    "weekly_claims": {
        "release_id": 180,
        "frequency": "weekly",
        "sources": {
            "ICSA": (
                "initial_claims_innovation",
                "expanding_log_ar1_innovation",
            ),
            "CCSA": (
                "continued_claims_innovation",
                "expanding_log_ar1_innovation",
            ),
        },
    },
    "jolts": {
        "release_id": 192,
        "frequency": "monthly",
        "sources": {
            "JTSJOR": ("job_openings_rate_change", "difference"),
            "JTSHIR": ("hires_rate_change", "difference"),
            "JTSQUR": ("quits_rate_change", "difference"),
            "JTSLDR": ("layoffs_discharges_rate_change", "difference"),
        },
    },
    "retail_sales": {
        "release_id": 9,
        "frequency": "monthly",
        "sources": {
            "RSAFS": ("retail_sales_log_change", "log_difference"),
            "RSFSXMV": (
                "retail_sales_ex_motor_vehicles_log_change",
                "log_difference",
            ),
        },
    },
    "housing": {
        "release_id": 27,
        "frequency": "monthly",
        "sources": {
            "HOUST": ("housing_starts_log_change", "log_difference"),
            "PERMIT": ("building_permits_log_change", "log_difference"),
        },
    },
    "durable_goods": {
        "release_id": 95,
        "frequency": "monthly",
        "sources": {
            "DGORDER": ("durable_goods_orders_log_change", "log_difference"),
            "NEWORDER": (
                "core_capital_goods_orders_log_change",
                "log_difference",
            ),
        },
    },
}


FIRST_RELEASE_OBSERVATION_COLUMNS = (
    "series_id",
    "release_block",
    "frequency",
    "reference_date",
    "release_date",
    "value",
    "release_lag_days",
    "eligible_for_feature",
    "feature_eligibility_status",
    "provider_id",
    "source_url",
)


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("evidence configuration must be a mapping")
    if config.get("schema_version") != 1:
        raise ValueError("evidence configuration schema_version must be 1")
    if config.get("model_id") != "m01_deterministic_composite":
        raise ValueError("evidence configuration has an unexpected model_id")

    configured_blocks = config.get("blocks")
    if not isinstance(configured_blocks, dict):
        raise ValueError("evidence configuration blocks must be a mapping")
    if set(configured_blocks) != set(_EXPECTED_BLOCKS):
        missing = sorted(set(_EXPECTED_BLOCKS).difference(configured_blocks))
        extra = sorted(set(configured_blocks).difference(_EXPECTED_BLOCKS))
        raise ValueError(
            f"evidence block mismatch; missing={missing}, extra={extra}"
        )

    for block_name, expected in _EXPECTED_BLOCKS.items():
        block = configured_blocks[block_name]
        if not isinstance(block, dict):
            raise ValueError(f"block {block_name} must be a mapping")
        if int(block.get("release_id", -1)) != expected["release_id"]:
            raise ValueError(f"block {block_name} has an unexpected release_id")
        if str(block.get("frequency")) != expected["frequency"]:
            raise ValueError(f"block {block_name} has an unexpected frequency")
        sources = block.get("sources")
        if not isinstance(sources, list):
            raise ValueError(f"block {block_name} sources must be a list")
        actual_sources: dict[str, tuple[str, str]] = {}
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError(f"block {block_name} has a non-mapping source")
            series_id = str(source.get("series_id"))
            if series_id in actual_sources:
                raise ValueError(f"block {block_name} repeats {series_id}")
            actual_sources[series_id] = (
                str(source.get("feature_name")),
                str(source.get("transform")),
            )
        if actual_sources != expected["sources"]:
            raise ValueError(f"block {block_name} source contract has changed")

    data = config.get("data")
    features = config.get("features")
    outputs = config.get("outputs")
    if not isinstance(data, dict) or not isinstance(features, dict):
        raise ValueError("evidence data and feature settings must be mappings")
    if not isinstance(outputs, dict):
        raise ValueError("evidence outputs must be a mapping")
    observation_start = _as_date(data["observation_start"])
    observation_end = _as_date(data["observation_end"])
    feature_start = _as_date(data["feature_start"])
    vintage_start = _as_date(data["vintage_start"])
    vintage_end = _as_date(data["vintage_end"])
    if observation_start > feature_start or feature_start > observation_end:
        raise ValueError("feature_start must lie within the observation window")
    if observation_start > observation_end or vintage_start > vintage_end:
        raise ValueError("evidence acquisition windows are invalid")
    for key in (
        "monthly_max_release_lag_days",
        "weekly_max_release_lag_days",
    ):
        if int(data[key]) < 0:
            raise ValueError(f"{key} cannot be negative")
    if data.get("archive_start_latest_only") is not True:
        raise ValueError("archive_start_latest_only must be true")
    if int(features["monthly_min_standardization_history"]) < 2:
        raise ValueError("monthly standardization history must be at least 2")
    if int(features["claims_min_ar_history"]) < 3:
        raise ValueError("claims AR history must be at least 3")
    if int(features["claims_min_standardization_history"]) < 2:
        raise ValueError("claims standardization history must be at least 2")
    ddof = int(features["standard_deviation_ddof"])
    if ddof < 0:
        raise ValueError("standard_deviation_ddof cannot be negative")
    required_outputs = {
        "raw_dir",
        "processed_dir",
        "events_file",
        "features_file",
        "first_release_observations_file",
        "manifest",
    }
    missing_outputs = required_outputs.difference(outputs)
    if missing_outputs:
        raise ValueError(f"evidence outputs are missing {sorted(missing_outputs)}")
    return config, raw


def _release_lag_days(
    observation: FirstReleaseObservation,
    *,
    frequency: str,
) -> int:
    reference = pd.Timestamp(observation.reference_date)
    if frequency == "monthly":
        lag_anchor = reference.to_period("M").end_time.normalize().date()
    else:
        lag_anchor = observation.reference_date
    return (observation.release_date - lag_anchor).days


def _normalized_observation_records(
    observations: tuple[FirstReleaseObservation, ...],
    *,
    series_id: str,
    release_block: str,
    frequency: str,
    max_release_lag_days: int,
    archive_start_latest_only: bool,
    provider_id: str,
    source_url: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    first_release_date = min(
        (observation.release_date for observation in observations),
        default=None,
    )
    latest_archive_start_reference = max(
        (
            observation.reference_date
            for observation in observations
            if observation.release_date == first_release_date
        ),
        default=None,
    )
    for observation in observations:
        lag_days = _release_lag_days(observation, frequency=frequency)
        within_general_lag = 0 <= lag_days <= max_release_lag_days
        within_archive_start_policy = (
            not archive_start_latest_only
            or observation.release_date != first_release_date
            or observation.reference_date == latest_archive_start_reference
        )
        if lag_days < 0:
            eligibility_status = "negative_release_lag"
        elif lag_days > max_release_lag_days:
            eligibility_status = "release_lag_exceeds_max"
        elif not within_archive_start_policy:
            eligibility_status = "archive_bootstrap_history"
        else:
            eligibility_status = "eligible"
        records.append(
            {
                "series_id": series_id,
                "release_block": release_block,
                "frequency": frequency,
                "reference_date": observation.reference_date,
                "release_date": observation.release_date,
                "value": observation.value,
                "release_lag_days": lag_days,
                "eligible_for_feature": (
                    within_general_lag and within_archive_start_policy
                ),
                "feature_eligibility_status": eligibility_status,
                "provider_id": provider_id,
                "source_url": source_url,
            }
        )
    return records


def _normalized_observations_frame(
    records: list[dict[str, object]],
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(
        records, columns=FIRST_RELEASE_OBSERVATION_COLUMNS
    )
    if frame.empty:
        raise ValueError("no first-release observations were acquired")
    for column in ("reference_date", "release_date"):
        frame[column] = pd.to_datetime(frame[column])
    frame["eligible_for_feature"] = frame["eligible_for_feature"].astype(bool)
    duplicate = frame.duplicated(["series_id", "reference_date"])
    if duplicate.any():
        raise ValueError("first-release observations contain duplicate references")
    return frame.sort_values(
        ["release_date", "release_block", "series_id", "reference_date"]
    ).reset_index(drop=True)


def _filter_monthly_archive_start_records(
    records: pd.DataFrame,
    observations: tuple[FirstReleaseObservation, ...],
    *,
    archive_start_latest_only: bool,
) -> tuple[pd.DataFrame, int]:
    """Remove archive-bootstrap history before feature standardization."""

    if not archive_start_latest_only or records.empty or not observations:
        return records, 0
    first_release_date = min(item.release_date for item in observations)
    latest_reference = max(
        item.reference_date
        for item in observations
        if item.release_date == first_release_date
    )
    release_dates = pd.to_datetime(records["release_date"]).dt.date
    reference_dates = pd.to_datetime(records["reference_month"]).dt.date
    excluded = (release_dates == first_release_date) & (
        reference_dates != latest_reference
    )
    return records.loc[~excluded].copy(), int(excluded.sum())


def _provider_vintage_start(
    source: Mapping[str, object],
    *,
    configured_start: date,
    provider_id: str,
) -> date:
    """Apply a transport floor without truncating other providers."""

    if provider_id != "alfred_web" or source.get("alfred_vintage_start") is None:
        return configured_start
    return max(configured_start, _as_date(source["alfred_vintage_start"]))


def build_evidence_dataset(
    *,
    project_root: Path,
    config_path: Path,
    refresh: bool = False,
    provider: str = "auto",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Build point-in-time evidence artifacts without Bayesian inference."""

    config, config_bytes = _load_config(config_path)
    data_config = config["data"]
    feature_config = config["features"]
    output_config = config["outputs"]

    feature_start = _as_date(data_config["feature_start"])
    observation_start = _as_date(data_config["observation_start"])
    observation_end = _as_date(data_config["observation_end"])
    vintage_start = _as_date(data_config["vintage_start"])
    vintage_end = _as_date(data_config["vintage_end"])
    monthly_max_lag = int(data_config["monthly_max_release_lag_days"])
    weekly_max_lag = int(data_config["weekly_max_release_lag_days"])
    archive_start_latest_only = bool(data_config["archive_start_latest_only"])
    ddof = int(feature_config["standard_deviation_ddof"])

    legacy_raw_dir = project_root / output_config["raw_dir"]
    processed_dir = project_root / output_config["processed_dir"]
    manifest_path = project_root / output_config["manifest"]
    events_path = processed_dir / output_config["events_file"]
    features_path = processed_dir / output_config["features_file"]
    observations_path = (
        processed_dir / output_config["first_release_observations_file"]
    )

    selection = select_vintage_provider(provider, environ=environ)
    client = selection.client
    raw_dir = _provider_raw_dir(legacy_raw_dir, client.cache_namespace)
    compatible_cache_dirs: tuple[tuple[Path, str], ...] = ()
    if raw_dir != legacy_raw_dir:
        compatible_cache_dirs = ((legacy_raw_dir, "alfred_web"),)

    event_frames: list[pd.DataFrame] = []
    observation_records: list[dict[str, object]] = []
    acquisitions: list[dict[str, object]] = []

    for block_name, block in config["blocks"].items():
        release_id = int(block["release_id"])
        frequency = str(block["frequency"])
        max_lag = weekly_max_lag if frequency == "weekly" else monthly_max_lag
        for source in block["sources"]:
            series_id = str(source["series_id"])
            feature_name = str(source["feature_name"])
            transform = str(source["transform"])
            source_vintage_start = _provider_vintage_start(
                source,
                configured_start=vintage_start,
                provider_id=client.provider_id,
            )
            if source_vintage_start > vintage_end:
                raise ValueError(
                    f"invalid provider vintage window for {series_id}: "
                    f"{source_vintage_start} > {vintage_end}"
                )

            if frequency == "weekly":
                artifact = client.list_first_release_observations(
                    series_id,
                    release_id=release_id,
                    observation_start=observation_start,
                    observation_end=observation_end,
                    vintage_start=source_vintage_start,
                    vintage_end=vintage_end,
                    chunk_cache_dir=raw_dir / "provider_chunks" / series_id,
                    refresh_cache=refresh,
                )
                observations = artifact.observations
                provider_id = artifact.provider_id
                source_url = artifact.source_url
                events = build_claims_release_events(
                    observations,
                    release_block=block_name,
                    feature_name=feature_name,
                    series_id=series_id,
                    provider_id=provider_id,
                    source_url=source_url,
                    min_ar_history=int(feature_config["claims_min_ar_history"]),
                    min_standardization_history=int(
                        feature_config["claims_min_standardization_history"]
                    ),
                    ddof=ddof,
                    max_release_lag_days=weekly_max_lag,
                    archive_start_latest_only=archive_start_latest_only,
                )
                observation_digest = hashlib.sha256()
                for item in observations:
                    observation_digest.update(
                        (
                            f"{item.reference_date.isoformat()},"
                            f"{item.release_date.isoformat()},{item.value:.17g}\n"
                        ).encode("utf-8")
                    )
                acquisitions.append(
                    {
                        "series_id": series_id,
                        "release_id": release_id,
                        "provider": provider_id,
                        "cache_origin": "provider_managed_chunk_cache",
                        "cache_directory": (
                            raw_dir / "provider_chunks" / series_id
                        ).relative_to(project_root).as_posix(),
                        "source_url": source_url,
                        "vintage_start": source_vintage_start.isoformat(),
                        "first_release_observations": len(observations),
                        "normalized_observations_sha256": (
                            observation_digest.hexdigest()
                        ),
                    }
                )
            else:
                acquisition = _download_or_load(
                    client=client,
                    series_id=series_id,
                    release_id=release_id,
                    raw_dir=raw_dir,
                    compatible_cache_dirs=compatible_cache_dirs,
                    observation_start=observation_start,
                    observation_end=observation_end,
                    vintage_start=source_vintage_start,
                    vintage_end=vintage_end,
                    refresh=refresh,
                )
                matrix = load_vintage_matrix(acquisition.content, series_id)
                observations = first_release_observations_from_matrix(matrix)
                provider_id = acquisition.provider_id
                source_url = acquisition.source_url
                transformed = extract_first_release_features(
                    matrix,
                    series_id=series_id,
                    component=block_name,
                    transform=transform,
                    max_release_lag_days=monthly_max_lag,
                )
                diagnostics = dict(
                    transformed.attrs.get("extraction_diagnostics", {})
                )
                transformed, archive_start_rows_excluded = (
                    _filter_monthly_archive_start_records(
                        transformed,
                        observations,
                        archive_start_latest_only=archive_start_latest_only,
                    )
                )
                diagnostics["rows_excluded_archive_bootstrap"] = (
                    archive_start_rows_excluded
                )
                events = build_monthly_release_events(
                    transformed,
                    release_block=block_name,
                    feature_name=feature_name,
                    frequency=frequency,
                    provider_id=provider_id,
                    source_url=source_url,
                    min_standardization_history=int(
                        feature_config["monthly_min_standardization_history"]
                    ),
                    ddof=ddof,
                )
                acquisitions.append(
                    {
                        "series_id": series_id,
                        "release_id": release_id,
                        "provider": provider_id,
                        "cache_origin": acquisition.cache_origin,
                        "path": acquisition.path.relative_to(project_root).as_posix(),
                        "source_url": source_url,
                        "vintage_start": source_vintage_start.isoformat(),
                        "sha256": _sha256(acquisition.content),
                        "bytes": len(acquisition.content),
                        "first_release_observations": len(observations),
                        "extraction_diagnostics": diagnostics,
                    }
                )

            observation_records.extend(
                _normalized_observation_records(
                    observations,
                    series_id=series_id,
                    release_block=block_name,
                    frequency=frequency,
                    max_release_lag_days=max_lag,
                    archive_start_latest_only=archive_start_latest_only,
                    provider_id=provider_id,
                    source_url=source_url,
                )
            )
            event_frames.append(events)

    observations_frame = _normalized_observations_frame(observation_records)
    all_events = pd.concat(event_frames, ignore_index=True)
    all_events = all_events.loc[
        pd.to_datetime(all_events["reference_date"]).dt.date >= feature_start
    ].copy()
    all_events = all_events.sort_values(
        ["release_date", "event_group_id", "reference_date", "feature_name"]
    ).reset_index(drop=True)
    validate_event_table(all_events)
    available_features = all_events.loc[
        all_events["feature_status"] == "available"
    ].copy()
    if available_features.empty:
        raise ValueError("no available release features were produced")

    _write_csv(observations_frame, observations_path)
    _write_csv(all_events, events_path)
    _write_csv(available_features, features_path)

    status_counts = {
        str(status): int(count)
        for status, count in all_events["feature_status"].value_counts().items()
    }
    block_counts = {
        str(block): int(count)
        for block, count in all_events["release_block"].value_counts().items()
    }
    observation_counts_by_series = {
        str(series_id): {
            "total": int(len(group)),
            "eligible_for_feature": int(group["eligible_for_feature"].sum()),
            "ineligible_for_feature": int((~group["eligible_for_feature"]).sum()),
            "eligibility_status_counts": {
                str(status): int(count)
                for status, count in group[
                    "feature_eligibility_status"
                ].value_counts().items()
            },
        }
        for series_id, group in observations_frame.groupby("series_id", sort=True)
    }
    manifest = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "evidence_set_id": config["evidence_set_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "point-in-time first-release acquisition and causal feature "
            "construction only; no likelihood or Bayesian update"
        ),
        "provider_requested": selection.requested,
        "provider_selected": selection.selected,
        "provider_description": client.provider_description,
        "credential_policy": (
            "FRED_API_KEY is read from the runtime environment only and is "
            "never serialized into outputs, cache identities, or logs"
        ),
        "configuration": config_path.relative_to(project_root).as_posix(),
        "configuration_sha256": _sha256(config_bytes),
        "feature_start": feature_start.isoformat(),
        "observation_start": observation_start.isoformat(),
        "observation_end": observation_end.isoformat(),
        "vintage_start": vintage_start.isoformat(),
        "vintage_end": vintage_end.isoformat(),
        "release_lag_policy_days": {
            "monthly_general_max": monthly_max_lag,
            "weekly_general_max": weekly_max_lag,
        },
        "archive_start_policy": "latest_reference_period_only",
        "first_release_observation_rows": len(observations_frame),
        "eligible_first_release_observation_rows": int(
            observations_frame["eligible_for_feature"].sum()
        ),
        "ineligible_first_release_observation_rows": int(
            (~observations_frame["eligible_for_feature"]).sum()
        ),
        "first_release_observation_counts_by_series": (
            observation_counts_by_series
        ),
        "first_release_observation_eligibility_status_counts": {
            str(status): int(count)
            for status, count in observations_frame[
                "feature_eligibility_status"
            ].value_counts().items()
        },
        "event_rows": len(all_events),
        "available_feature_rows": len(available_features),
        "first_event_release_date": all_events["release_date"]
        .min()
        .date()
        .isoformat(),
        "last_event_release_date": all_events["release_date"]
        .max()
        .date()
        .isoformat(),
        "event_status_counts": status_counts,
        "event_block_counts": block_counts,
        "raw_acquisitions": acquisitions,
        "processed_files": [
            {
                "path": observations_path.relative_to(project_root).as_posix(),
                "sha256": _sha256(observations_path.read_bytes()),
                "rows": len(observations_frame),
            },
            {
                "path": events_path.relative_to(project_root).as_posix(),
                "sha256": _sha256(events_path.read_bytes()),
                "rows": len(all_events),
            },
            {
                "path": features_path.relative_to(project_root).as_posix(),
                "sha256": _sha256(features_path.read_bytes()),
                "rows": len(available_features),
            },
        ],
    }
    _write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "events": events_path,
        "features": features_path,
        "observations": observations_path,
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
        default=Path(
            "configs/models/m01_non_defining_release_evidence.yaml"
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Reacquire the selected provider instead of reusing its cache.",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "fred", "alfred"),
        default="auto",
        help=(
            "Acquisition provider. 'auto' prefers FRED when FRED_API_KEY is "
            "set and otherwise uses the keyless ALFRED fallback."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    outputs = build_evidence_dataset(
        project_root=project_root,
        config_path=config_path.resolve(),
        refresh=args.refresh,
        provider=args.provider,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
