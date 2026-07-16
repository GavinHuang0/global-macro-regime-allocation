"""Acquire, transform, classify, and publish model 01's historical dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from regime_allocation.data.first_release import extract_first_release_features
from regime_allocation.data.providers import select_vintage_provider
from regime_allocation.data.providers.vintage_matrix import (
    VintageMatrixProvider,
    load_vintage_matrix,
)
from regime_allocation.features.composites import (
    ALL_COMPONENTS,
    build_composites_from_transformed,
)
from regime_allocation.models.m01_deterministic_composite.pipeline import (
    classify_frame,
)


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


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
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("model configuration must be a mapping")
    configured_components = set(config.get("components", {}))
    if configured_components != set(ALL_COMPONENTS):
        missing = set(ALL_COMPONENTS).difference(configured_components)
        extra = configured_components.difference(ALL_COMPONENTS)
        raise ValueError(
            f"component configuration mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    if float(config["features"]["component_weight"]) != 0.25:
        raise ValueError("model 01 requires four fixed component weights of 0.25")
    if str(config["features"].get("zero_tie_policy")) != "up":
        raise ValueError("model 01 requires exact-zero ties to map to 'up'")
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
        raise ValueError("component axes do not match the frozen model 01 contract")
    return config, raw


@dataclass(frozen=True)
class _MatrixAcquisition:
    path: Path
    content: bytes
    provider_id: str
    cache_origin: str
    source_url: str


def _provider_raw_dir(legacy_raw_dir: Path, cache_namespace: str) -> Path:
    """Keep the historical ALFRED directory while isolating new providers."""

    if cache_namespace == "alfred":
        return legacy_raw_dir
    if legacy_raw_dir.parent.name == "alfred":
        return (
            legacy_raw_dir.parent.parent
            / cache_namespace
            / legacy_raw_dir.name
        )
    return legacy_raw_dir.parent / cache_namespace / legacy_raw_dir.name


def _metadata_path(raw_path: Path) -> Path:
    return raw_path.with_name(raw_path.name + ".metadata.json")


def _cached_provider_id(raw_path: Path, fallback: str) -> str:
    metadata_path = _metadata_path(raw_path)
    if not metadata_path.exists():
        return fallback
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        provider_id = str(payload["provider_id"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError(f"invalid acquisition metadata: {metadata_path}") from None
    if not provider_id or provider_id == "None":
        raise ValueError(f"invalid acquisition metadata: {metadata_path}")
    return provider_id


def _download_or_load(
    *,
    client: VintageMatrixProvider,
    series_id: str,
    release_id: int,
    raw_dir: Path,
    compatible_cache_dirs: tuple[tuple[Path, str], ...] = (),
    observation_start: date,
    observation_end: date,
    vintage_start: date,
    vintage_end: date,
    refresh: bool,
) -> _MatrixAcquisition:
    query = {
        "series_id": series_id,
        "release_id": release_id,
        "observation_start": observation_start.isoformat(),
        "observation_end": observation_end.isoformat(),
        "vintage_start": vintage_start.isoformat(),
        "vintage_end": vintage_end.isoformat(),
        "normalized_contract": "level_snapshots_by_vintage",
    }
    query_hash = _sha256(
        json.dumps(query, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )[:12]
    cache_locations = ((raw_dir, client.provider_id),) + compatible_cache_dirs
    seen_cache_dirs: set[Path] = set()
    if not refresh:
        for cache_dir, fallback_provider_id in cache_locations:
            resolved_cache_dir = cache_dir.resolve()
            if resolved_cache_dir in seen_cache_dirs:
                continue
            seen_cache_dirs.add(resolved_cache_dir)
            candidate = cache_dir / (
                f"{series_id}_{query_hash}_levels_by_vintage.zip"
            )
            if candidate.exists():
                return _MatrixAcquisition(
                    path=candidate,
                    content=candidate.read_bytes(),
                    provider_id=_cached_provider_id(
                        candidate, fallback_provider_id
                    ),
                    cache_origin="existing_normalized_cache",
                    source_url=client.series_page_url(series_id),
                )

    raw_path = raw_dir / f"{series_id}_{query_hash}_levels_by_vintage.zip"

    # Reuse normalized matrices produced by the earlier Download Data form
    # transport. Their downstream contract is identical to the current one.
    legacy_query = dict(query)
    legacy_query.pop("normalized_contract")
    legacy_query.update({"units": "lin", "output_type": 2})
    legacy_hash = _sha256(
        json.dumps(
            legacy_query, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )[:12]
    if not refresh:
        for cache_dir, fallback_provider_id in cache_locations:
            legacy_path = cache_dir / (
                f"{series_id}_{legacy_hash}_levels_by_vintage.zip"
            )
            if legacy_path.exists():
                return _MatrixAcquisition(
                    path=legacy_path,
                    content=legacy_path.read_bytes(),
                    provider_id=_cached_provider_id(
                        legacy_path, fallback_provider_id
                    ),
                    cache_origin="legacy_normalized_cache",
                    source_url=client.series_page_url(series_id),
                )

    artifact = client.download_level_matrix(
        series_id,
        release_id=release_id,
        observation_start=observation_start,
        observation_end=observation_end,
        vintage_start=vintage_start,
        vintage_end=vintage_end,
        chunk_cache_dir=raw_dir / "provider_chunks" / series_id,
        refresh_cache=refresh,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
    temporary.write_bytes(artifact.content)
    temporary.replace(raw_path)
    _write_json(
        {
            "schema_version": 1,
            "provider_id": artifact.provider_id,
            "source_url": artifact.source_url,
            "query": query,
            "content_sha256": _sha256(artifact.content),
            "selected_vintage_dates": len(artifact.selected_vintage_dates),
        },
        _metadata_path(raw_path),
    )
    return _MatrixAcquisition(
        path=raw_path,
        content=artifact.content,
        provider_id=artifact.provider_id,
        cache_origin="downloaded",
        source_url=artifact.source_url,
    )


def build_dataset(
    *,
    project_root: Path,
    config_path: Path,
    refresh: bool = False,
    provider: str = "auto",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    config, config_bytes = _load_config(config_path)
    data_config = config["data"]
    feature_config = config["features"]
    output_config = config["outputs"]

    reference_start = _as_date(data_config["reference_start"])
    reference_end = _as_date(data_config["reference_end"])
    observation_start = _as_date(data_config["observation_start"])
    vintage_start = _as_date(data_config["vintage_start"])
    vintage_end = _as_date(data_config["vintage_end"])
    max_release_lag_days = int(data_config["max_release_lag_days"])

    legacy_raw_dir = project_root / output_config["raw_dir"]
    processed_dir = project_root / output_config["processed_dir"]
    manifest_path = project_root / output_config["manifest"]
    published_dir = project_root / output_config["published_dir"]

    selection = select_vintage_provider(provider, environ=environ)
    client = selection.client
    raw_dir = _provider_raw_dir(legacy_raw_dir, client.cache_namespace)
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
                _as_date(source.get("vintage_start", vintage_start)),
            )
            source_vintage_end = min(
                vintage_end,
                _as_date(source.get("vintage_end", vintage_end)),
            )
            if source_vintage_start > source_vintage_end:
                raise ValueError(
                    f"invalid vintage window for {series_id}: "
                    f"{source_vintage_start} > {source_vintage_end}"
                )
            acquisition = _download_or_load(
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
            raw_path = acquisition.path
            raw_payload = acquisition.content
            matrix = load_vintage_matrix(raw_payload, series_id)
            records = extract_first_release_features(
                matrix,
                series_id=series_id,
                component=component,
                transform=transform,
                max_release_lag_days=max_release_lag_days,
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
                    "cache_origin": acquisition.cache_origin,
                    "vintage_start": source_vintage_start.isoformat(),
                    "vintage_end": source_vintage_end.isoformat(),
                    "path": raw_path.relative_to(project_root).as_posix(),
                    "sha256": _sha256(raw_payload),
                    "bytes": len(raw_payload),
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
    full_index = pd.date_range(
        transformed.index.min(), pd.Timestamp(reference_end), freq="MS"
    )
    transformed = transformed.reindex(full_index)
    releases = releases.reindex(full_index)

    features = build_composites_from_transformed(
        transformed,
        min_history=int(feature_config["min_history_months"]),
        smoothing_window=int(feature_config["smoothing_window_months"]),
        ddof=int(feature_config["standard_deviation_ddof"]),
    )
    required_releases = releases[list(ALL_COMPONENTS)]
    complete_release_rows = required_releases.notna().all(axis=1)
    features["label_available_at"] = required_releases.max(axis=1).where(
        complete_release_rows
    )
    classified = classify_frame(features)
    classified.index.name = "reference_month"

    publication_index = pd.date_range(
        reference_start, reference_end, freq="MS", name="reference_month"
    )
    publication_panel = classified.reindex(publication_index)
    component_panel = transformed.reindex(publication_index)
    expected_months = int(data_config["expected_reference_months"])
    if len(publication_panel) != expected_months:
        raise ValueError(
            f"publication range contains {len(publication_panel)} months, "
            f"expected {expected_months}"
        )
    actual_missing = {
        (component, month.date())
        for component in ALL_COMPONENTS
        for month in component_panel.index[component_panel[component].isna()]
    }
    configured_missing = data_config.get(
        "expected_missing_component_months", {}
    )
    expected_missing = {
        (str(component), _as_date(month))
        for component, months in configured_missing.items()
        for month in months
    }
    if actual_missing != expected_missing:
        unexpected = sorted(actual_missing.difference(expected_missing))
        absent = sorted(expected_missing.difference(actual_missing))
        raise ValueError(
            "component-panel missingness differs from the frozen contract; "
            f"unexpected={unexpected[:8]}, expected_but_present={absent[:8]}"
        )

    first_classified = publication_panel["regime_id"].first_valid_index()
    if first_classified is None:
        raise ValueError("no classified regimes were produced")
    public_history = publication_panel.loc[
        first_classified:,
        [
            "growth_raw",
            "inflation_raw",
            "growth_smoothed",
            "inflation_smoothed",
            "regime_id",
            "regime_label",
            "label_available_at",
        ],
    ].copy()
    public_history["data_status"] = "classified"
    unavailable = public_history["regime_id"].isna()
    public_history.loc[
        unavailable & public_history["label_available_at"].isna(),
        "data_status",
    ] = "missing_component_feature"
    public_history.loc[
        unavailable & public_history["label_available_at"].notna(),
        "data_status",
    ] = "unavailable_trailing_window"
    public_history = public_history.reset_index()
    unavailable_history = public_history["regime_id"].isna()

    history_months = pd.DatetimeIndex(public_history["reference_month"])
    expected_history_months = pd.date_range(
        history_months.min(), pd.Timestamp(reference_end), freq="MS"
    )
    if not history_months.equals(expected_history_months):
        raise ValueError("published regime history contains a monthly gap")
    classified_history = public_history.loc[
        public_history["regime_id"].notna()
    ]
    classified_months = pd.DatetimeIndex(
        classified_history["reference_month"]
    )
    if classified_months.max().date() != reference_end:
        raise ValueError(
            "latest classified month does not equal the configured reference end"
        )
    published_labels = public_history["regime_id"].notna()
    if public_history.loc[published_labels, "label_available_at"].isna().any():
        raise ValueError("published regime history contains a missing availability date")

    latest = classified_history.iloc[-1]
    latest_payload = {
        "model_id": config["model_id"],
        "reference_month": latest["reference_month"].date().isoformat(),
        "label_available_at": pd.Timestamp(latest["label_available_at"])
        .date()
        .isoformat(),
        "growth_score": float(latest["growth_smoothed"]),
        "inflation_score": float(latest["inflation_smoothed"]),
        "regime_id": str(latest["regime_id"]),
        "regime_label": str(latest["regime_label"]),
        "status": "latest_confirmed_deterministic_regime",
    }

    long_path = processed_dir / "first_release_components_long.csv"
    features_path = processed_dir / "composite_features_and_regimes.csv"
    public_history_path = published_dir / "regime_history.csv"
    latest_path = published_dir / "latest_confirmed.json"
    _write_csv(long, long_path)
    _write_csv(publication_panel.reset_index(), features_path)
    _write_csv(public_history, public_history_path)
    _write_json(latest_payload, latest_path)

    combined_hash = hashlib.sha256()
    for item in sorted(raw_files, key=lambda value: str(value["series_id"])):
        combined_hash.update(str(item["sha256"]).encode("ascii"))
    manifest = {
        "schema_version": 2,
        "model_id": config["model_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "provider-neutral point-in-time vintage matrices",
        "provider_policy": "fred_api_preferred_when_FRED_API_KEY_is_present",
        "provider_requested": selection.requested,
        "provider_selected": selection.selected,
        "providers_used": sorted(
            {str(item["provider"]) for item in raw_files}
        ),
        "provider_output": "batched as-of level snapshots by vintage date",
        "cache_format": "deterministically merged and normalized ZIP matrices",
        "configuration": config_path.relative_to(project_root).as_posix(),
        "configuration_sha256": _sha256(config_bytes),
        "data_snapshot_sha256": combined_hash.hexdigest(),
        "reference_start": reference_start.isoformat(),
        "reference_end": reference_end.isoformat(),
        "reference_months": len(publication_panel),
        "history_months": len(public_history),
        "classified_months": len(classified_history),
        "first_classified_month": classified_history["reference_month"]
        .min()
        .date()
        .isoformat(),
        "latest_classified_month": classified_history["reference_month"]
        .max()
        .date()
        .isoformat(),
        "unavailable_months": [
            {
                "reference_month": row.reference_month.date().isoformat(),
                "data_status": row.data_status,
            }
            for row in public_history.loc[unavailable_history].itertuples()
        ],
        "raw_files": raw_files,
        "processed_files": [
            long_path.relative_to(project_root).as_posix(),
            features_path.relative_to(project_root).as_posix(),
        ],
        "published_files": [
            public_history_path.relative_to(project_root).as_posix(),
            latest_path.relative_to(project_root).as_posix(),
        ],
    }
    _write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "components": long_path,
        "features": features_path,
        "history": public_history_path,
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
        default=Path("configs/models/m01_deterministic_composite.yaml"),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Reacquire the selected provider's raw matrices instead of using "
            "compatible caches; other providers' artifacts remain untouched."
        ),
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
    outputs = build_dataset(
        project_root=project_root,
        config_path=config_path.resolve(),
        refresh=args.refresh,
        provider=args.provider,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
