"""Build Model 02's point-in-time deterministic composite scores.

The command acquires or reuses provider-neutral FRED/ALFRED vintage matrices,
extracts genuine first-release monthly transformations, applies strictly lagged
expanding standardization, and averages four fixed components for each of the
growth and inflation axes.  Payroll employment uses monthly log growth and the
axis scores receive no trailing smoothing.

This stage intentionally does not assign quadrant labels, estimate regime
probabilities, fit transition dynamics, process evidence releases, or run a
backtest.  All outputs live in the ``m02_soft_composite`` namespace and cannot
overwrite frozen Model 01 artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from regime_allocation.data.dataset_acquisition import (
    MatrixAcquisition,
    as_date,
    download_or_load_vintage_matrix,
    provider_raw_dir,
    sha256,
    write_csv,
    write_json,
)
from regime_allocation.data.first_release import extract_first_release_features
from regime_allocation.data.providers import select_vintage_provider
from regime_allocation.data.providers.vintage_matrix import load_vintage_matrix
from regime_allocation.models.m02_soft_composite.scores import (
    ALL_COMPONENTS,
    GROWTH_COMPONENTS,
    INFLATION_COMPONENTS,
    build_composite_scores,
)


MODEL_ID = "m02_soft_composite"


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load and validate the frozen Model 02 score-definition contract."""

    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("model configuration must be a mapping")
    if str(config.get("model_id")) != MODEL_ID:
        raise ValueError(f"model_id must be {MODEL_ID!r}")

    configured_components = set(config.get("components", {}))
    if configured_components != set(ALL_COMPONENTS):
        missing = set(ALL_COMPONENTS).difference(configured_components)
        extra = configured_components.difference(ALL_COMPONENTS)
        raise ValueError(
            f"component configuration mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    expected_axes = {
        "payrolls": "growth",
        "industrial_production": "growth",
        "consumer_activity": "growth",
        "unemployment_rate": "growth",
        "core_cpi": "inflation",
        "core_pce": "inflation",
        "producer_prices": "inflation",
        "average_hourly_earnings": "inflation",
    }
    configured_axes = {
        component: str(spec.get("axis"))
        for component, spec in config["components"].items()
    }
    if configured_axes != expected_axes:
        raise ValueError("component axes do not match the Model 02 score contract")

    expected_transforms = {
        "payrolls": "log_difference",
        "industrial_production": "log_difference",
        "consumer_activity": "log_difference",
        "unemployment_rate": "negative_difference",
        "core_cpi": "log_difference",
        "core_pce": "log_difference",
        "producer_prices": "log_difference",
        "average_hourly_earnings": "log_difference",
    }
    configured_transforms = {
        component: str(spec.get("transform"))
        for component, spec in config["components"].items()
    }
    if configured_transforms != expected_transforms:
        raise ValueError(
            "component transforms do not match the Model 02 score contract"
        )

    feature_config = config.get("features", {})
    if float(feature_config.get("component_weight", 0.0)) != 0.25:
        raise ValueError("Model 02 requires four fixed component weights of 0.25")
    if bool(feature_config.get("apply_trailing_smoothing", True)):
        raise ValueError("Model 02 score definition forbids trailing smoothing")

    for output_path in config.get("outputs", {}).values():
        if MODEL_ID not in str(output_path):
            raise ValueError("every Model 02 output path must contain its model_id")
    return config, raw


def _cumulative_score_availability(
    releases: pd.DataFrame,
    *,
    scores_available: pd.Series,
) -> pd.Series:
    """Return the latest release among all score prerequisites through a month."""

    required = releases.reindex(columns=list(ALL_COMPONENTS))
    if not required.index.equals(scores_available.index):
        raise ValueError("release and score-availability indices must match")
    cumulative_prerequisites = required.cummax()
    return cumulative_prerequisites.max(axis=1).where(scores_available)


def _axis_score_availability(
    releases: pd.DataFrame,
    *,
    components: tuple[str, ...],
    scores_available: pd.Series,
) -> pd.Series:
    """Return one axis score's causal availability date for every month."""

    required = releases.reindex(columns=list(components))
    if not required.index.equals(scores_available.index):
        raise ValueError("release and score-availability indices must match")
    return required.cummax().max(axis=1).where(scores_available)


def _expected_missing_pairs(config: Mapping[str, Any]) -> set[tuple[str, object]]:
    configured = config.get("expected_missing_component_months", {})
    return {
        (str(component), as_date(month))
        for component, months in configured.items()
        for month in months
    }


def build_scores(
    *,
    project_root: Path,
    config_path: Path,
    refresh: bool = False,
    provider: str = "auto",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Build, audit, and publish Model 02 deterministic composite scores."""

    config, config_bytes = _load_config(config_path)
    data_config = config["data"]
    feature_config = config["features"]
    output_config = config["outputs"]

    reference_start = as_date(data_config["reference_start"])
    reference_end = as_date(data_config["reference_end"])
    observation_start = as_date(data_config["observation_start"])
    vintage_start = as_date(data_config["vintage_start"])
    vintage_end = as_date(data_config["vintage_end"])
    max_release_lag_days = int(data_config["max_release_lag_days"])
    archive_start_latest_only = data_config.get("archive_start_latest_only", False)
    if not isinstance(archive_start_latest_only, bool):
        raise TypeError("data.archive_start_latest_only must be a boolean")

    legacy_raw_dir = project_root / output_config["raw_dir"]
    processed_dir = project_root / output_config["processed_dir"]
    manifest_path = project_root / output_config["manifest"]
    published_dir = project_root / output_config["published_dir"]

    selection = select_vintage_provider(provider, environ=environ)
    client = selection.client
    raw_dir = provider_raw_dir(legacy_raw_dir, client.cache_namespace)
    compatible_cache_dirs: tuple[tuple[Path, str], ...] = ()
    if raw_dir != legacy_raw_dir:
        compatible_cache_dirs = ((legacy_raw_dir, "alfred_web"),)

    all_records: list[pd.DataFrame] = []
    raw_files: list[dict[str, object]] = []
    for component, component_config in config["components"].items():
        transform = str(component_config["transform"])
        for source in component_config["sources"]:
            series_id = str(source["series_id"])
            release_id = int(source["release_id"])
            source_vintage_start = max(
                vintage_start,
                as_date(source.get("vintage_start", vintage_start)),
            )
            source_vintage_end = min(
                vintage_end,
                as_date(source.get("vintage_end", vintage_end)),
            )
            if source_vintage_start > source_vintage_end:
                raise ValueError(
                    f"invalid vintage window for {series_id}: "
                    f"{source_vintage_start} > {source_vintage_end}"
                )

            acquisition: MatrixAcquisition = download_or_load_vintage_matrix(
                client=client,
                series_id=series_id,
                release_id=release_id,
                raw_dir=raw_dir,
                compatible_cache_dirs=compatible_cache_dirs,
                observation_start=observation_start,
                observation_end=reference_end,
                vintage_start=source_vintage_start,
                vintage_end=source_vintage_end,
                refresh=refresh,
            )
            matrix = load_vintage_matrix(acquisition.content, series_id)
            records = extract_first_release_features(
                matrix,
                series_id=series_id,
                component=component,
                transform=transform,
                max_release_lag_days=max_release_lag_days,
                archive_start_latest_only=archive_start_latest_only,
            )
            extraction_diagnostics = dict(
                records.attrs.get("extraction_diagnostics", {})
            )
            if source.get("active_start") is not None:
                records = records[
                    records["reference_month"] >= pd.Timestamp(source["active_start"])
                ]
            if source.get("active_end") is not None:
                records = records[
                    records["reference_month"] <= pd.Timestamp(source["active_end"])
                ]
            records = records.copy()
            records["source_url"] = acquisition.source_url
            all_records.append(records)
            raw_files.append(
                {
                    "series_id": series_id,
                    "release_id": release_id,
                    "provider": acquisition.provider_id,
                    "source_url": acquisition.source_url,
                    "cache_origin": acquisition.cache_origin,
                    "vintage_start": source_vintage_start.isoformat(),
                    "vintage_end": source_vintage_end.isoformat(),
                    "path": acquisition.path.relative_to(project_root).as_posix(),
                    "sha256": sha256(acquisition.content),
                    "bytes": len(acquisition.content),
                    "extraction_diagnostics": extraction_diagnostics,
                }
            )

    long = pd.concat(all_records, ignore_index=True).sort_values(
        ["reference_month", "component", "series_id"]
    )
    duplicates = long.duplicated(["reference_month", "component"], keep=False)
    if duplicates.any():
        sample = long.loc[
            duplicates, ["reference_month", "component", "series_id"]
        ].head()
        raise ValueError(f"overlapping component sources detected:\n{sample}")

    transformed = long.pivot(
        index="reference_month", columns="component", values="transformed_value"
    )
    releases = long.pivot(
        index="reference_month", columns="component", values="release_date"
    )
    computation_start = min(transformed.index.min(), pd.Timestamp(reference_start))
    full_index = pd.date_range(computation_start, pd.Timestamp(reference_end), freq="MS")
    transformed = transformed.reindex(full_index)
    releases = releases.reindex(full_index)

    features = build_composite_scores(
        transformed,
        min_history=int(feature_config["min_history_months"]),
        ddof=int(feature_config["standard_deviation_ddof"]),
    )
    growth_available = features["growth_score"].notna()
    inflation_available = features["inflation_score"].notna()
    scores_available = growth_available & inflation_available
    features["growth_score_available_at"] = _axis_score_availability(
        releases,
        components=GROWTH_COMPONENTS,
        scores_available=growth_available,
    )
    features["inflation_score_available_at"] = _axis_score_availability(
        releases,
        components=INFLATION_COMPONENTS,
        scores_available=inflation_available,
    )
    features["score_available_at"] = pd.concat(
        [
            features["growth_score_available_at"],
            features["inflation_score_available_at"],
        ],
        axis=1,
    ).max(axis=1).where(scores_available)

    publication_index = pd.date_range(
        reference_start, reference_end, freq="MS", name="reference_month"
    )
    publication_panel = features.reindex(publication_index)
    expected_months = int(data_config["expected_reference_months"])
    if len(publication_panel) != expected_months:
        raise ValueError(
            f"publication range contains {len(publication_panel)} months, "
            f"expected {expected_months}"
        )

    component_panel = transformed.reindex(publication_index)
    actual_missing = {
        (component, month.date())
        for component in ALL_COMPONENTS
        for month in component_panel.index[component_panel[component].isna()]
    }
    expected_missing = _expected_missing_pairs(data_config)
    if actual_missing != expected_missing:
        unexpected = sorted(actual_missing.difference(expected_missing))
        absent = sorted(expected_missing.difference(actual_missing))
        raise ValueError(
            "component-panel missingness differs from the frozen contract; "
            f"unexpected={unexpected[:12]}, expected_but_present={absent[:12]}"
        )

    transformed_columns = [f"{component}_transformed" for component in ALL_COMPONENTS]
    score_complete = publication_panel[["growth_score", "inflation_score"]].notna().all(
        axis=1
    )
    component_complete = publication_panel[transformed_columns].notna().all(axis=1)
    publication_panel["data_status"] = "score_available"
    publication_panel.loc[~component_complete, "data_status"] = (
        "missing_component_feature"
    )
    publication_panel.loc[component_complete & ~score_complete, "data_status"] = (
        "insufficient_standardization_history"
    )
    publication_panel.index.name = "reference_month"

    complete_scores = publication_panel.loc[score_complete]
    if complete_scores.empty:
        raise ValueError("no complete Model 02 scores were produced")
    if complete_scores["score_available_at"].isna().any():
        raise ValueError("a complete score is missing its availability date")
    latest = complete_scores.iloc[-1]
    if complete_scores.index.max().date() != reference_end:
        raise ValueError("latest complete score does not equal reference_end")

    public_history = publication_panel[
        [
            "growth_score",
            "inflation_score",
            "growth_score_available_at",
            "inflation_score_available_at",
            "score_available_at",
            "data_status",
        ]
    ].reset_index()
    latest_payload = {
        "model_id": MODEL_ID,
        "stage": "deterministic_composite_scores",
        "reference_month": complete_scores.index.max().date().isoformat(),
        "score_available_at": pd.Timestamp(latest["score_available_at"])
        .date()
        .isoformat(),
        "growth_score": float(latest["growth_score"]),
        "inflation_score": float(latest["inflation_score"]),
        "status": "latest_observed_deterministic_scores",
        "regime_probabilities_computed": False,
    }

    long_path = processed_dir / "first_release_components_long.csv"
    features_path = processed_dir / "composite_scores.csv"
    history_path = published_dir / "score_history.csv"
    latest_path = published_dir / "latest_scores.json"
    write_csv(long, long_path)
    write_csv(publication_panel.reset_index(), features_path)
    write_csv(public_history, history_path)
    write_json(latest_payload, latest_path)

    generated_paths = [long_path, features_path, history_path, latest_path]
    generated_file_hashes = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
        for path in generated_paths
    ]
    combined_hash = hashlib.sha256()
    for item in sorted(raw_files, key=lambda value: str(value["series_id"])):
        combined_hash.update(str(item["sha256"]).encode("ascii"))

    unavailable = publication_panel.loc[~score_complete, ["data_status"]]
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage": "deterministic_composite_score_definition",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(project_root).as_posix(),
        "configuration_sha256": sha256(config_bytes),
        "provider": "provider-neutral point-in-time vintage matrices",
        "provider_policy": "fred_api_preferred_when_FRED_API_KEY_is_present",
        "provider_requested": selection.requested,
        "provider_selected": selection.selected,
        "providers_used": sorted({str(item["provider"]) for item in raw_files}),
        "provider_output": "batched as-of level snapshots by vintage date",
        "archive_start_policy": (
            "latest_reference_period_only"
            if archive_start_latest_only
            else "all_rows_within_release_lag_limit"
        ),
        "data_snapshot_sha256": combined_hash.hexdigest(),
        "reference_start": reference_start.isoformat(),
        "reference_end": reference_end.isoformat(),
        "reference_months": len(publication_panel),
        "complete_score_months": len(complete_scores),
        "first_complete_score_month": complete_scores.index.min().date().isoformat(),
        "latest_complete_score_month": complete_scores.index.max().date().isoformat(),
        "score_definition": {
            "payroll_transform": "100_log_current_over_previous",
            "standardization": "lagged_expanding_mean_and_sample_standard_deviation",
            "minimum_prior_observations": int(feature_config["min_history_months"]),
            "component_weight": float(feature_config["component_weight"]),
            "trailing_smoothing": False,
            "hard_quadrant_labels": False,
            "regime_probabilities": False,
        },
        "unavailable_months": [
            {
                "reference_month": index.date().isoformat(),
                "data_status": str(row["data_status"]),
            }
            for index, row in unavailable.iterrows()
        ],
        "raw_files": raw_files,
        "processed_files": [
            long_path.relative_to(project_root).as_posix(),
            features_path.relative_to(project_root).as_posix(),
        ],
        "published_files": [
            history_path.relative_to(project_root).as_posix(),
            latest_path.relative_to(project_root).as_posix(),
        ],
        "generated_file_hashes": generated_file_hashes,
    }
    write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "components": long_path,
        "scores": features_path,
        "history": history_path,
        "latest": latest_path,
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
        default=Path("configs/models/m02_soft_composite.yaml"),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Reacquire the selected provider's raw matrices.",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "fred", "alfred"),
        default="auto",
        help=(
            "Acquisition provider. 'auto' prefers FRED when FRED_API_KEY is "
            "present and otherwise retains the keyless ALFRED fallback."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments and build Model 02 deterministic score artifacts."""

    args = _parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    outputs = build_scores(
        project_root=project_root,
        config_path=config_path.resolve(),
        refresh=args.refresh,
        provider=args.provider,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
