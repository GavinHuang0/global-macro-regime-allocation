"""Build Model 02's authenticated real-retail decomposition sensitivity.

The command retrieves matching RSAFS and RRSFS vintage matrices through the
authenticated FRED provider, aligns each series as of common information dates, and
constructs two causal event coordinates: real retail-and-food-services growth
and the implicit nominal-minus-real price change.  It never modifies the
frozen nominal consumer-demand artifact and never serializes ``FRED_API_KEY``.
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
from regime_allocation.features.m02_retail_sensitivities import (
    build_real_retail_decomposition_matrices,
)
from regime_allocation.features.release_evidence import (
    EVENT_TABLE_COLUMNS,
    build_monthly_release_events,
    validate_event_table,
)


def _project_path(project_root: Path, configured: object) -> Path:
    path = Path(str(configured))
    resolved = path if path.is_absolute() else project_root / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"configured path escapes project root: {configured}") from exc
    return resolved


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    content = path.read_bytes()
    payload = yaml.safe_load(content)
    if not isinstance(payload, dict):
        raise ValueError("sensitivity configuration must be a mapping")
    retail = payload.get("retail_sensitivities", {}).get("real_decomposition")
    if not isinstance(retail, Mapping):
        raise ValueError("configuration omits retail real_decomposition")
    if str(retail.get("provider")) != "fred_api":
        raise ValueError("real-retail sensitivity requires the authenticated FRED API")
    if str(retail.get("series_id")) != "RRSFS" or int(retail.get("release_id", -1)) != 9:
        raise ValueError("real-retail sensitivity must use RRSFS release 9")
    return payload, content


def build_retail_sensitivity(
    *,
    project_root: Path,
    config_path: Path,
    refresh: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Acquire and publish the common-vintage retail decomposition events."""

    config, config_bytes = _load_config(config_path)
    retail = config["retail_sensitivities"]["real_decomposition"]
    outputs = config["outputs"]
    selection = select_vintage_provider("fred", environ=environ)
    if selection.selected != "fred_api":
        raise ValueError("FRED_API_KEY is required for the real-retail sensitivity")
    client = selection.client
    raw_root = _project_path(project_root, outputs["retail_raw_dir"])
    raw_dir = provider_raw_dir(raw_root, client.cache_namespace)
    observation_start = as_date(retail["observation_start"])
    observation_end = as_date(retail["observation_end"])
    vintage_start = as_date(retail["vintage_start"])
    vintage_end = as_date(retail["vintage_end"])

    acquisitions = {}
    matrices = {}
    for series_id in ("RSAFS", "RRSFS"):
        acquisition = download_or_load_vintage_matrix(
            client=client,
            series_id=series_id,
            release_id=int(retail["release_id"]),
            raw_dir=raw_dir,
            compatible_cache_dirs=(),
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_start=vintage_start,
            vintage_end=vintage_end,
            refresh=refresh,
        )
        acquisitions[series_id] = acquisition
        matrices[series_id] = load_vintage_matrix(acquisition.content, series_id)

    decomposition = build_real_retail_decomposition_matrices(
        matrices["RSAFS"], matrices["RRSFS"]
    )
    extraction_options = {
        "transform": "log_difference",
        "max_release_lag_days": int(retail["maximum_release_lag_days"]),
        "archive_start_latest_only": bool(retail["archive_start_latest_only"]),
    }
    real_records = extract_first_release_features(
        decomposition.real_levels,
        series_id="RRSFS",
        component="real_retail_and_food_services",
        **extraction_options,
    )
    price_records = extract_first_release_features(
        decomposition.implicit_price_levels,
        series_id="RSAFS_DIV_RRSFS",
        component="implicit_retail_price",
        **extraction_options,
    )
    source_url = " | ".join(
        sorted({item.source_url for item in acquisitions.values()})
    )
    minimum_history = int(retail["minimum_standardization_history"])
    frames = [
        build_monthly_release_events(
            real_records,
            release_block="consumer_demand_real_decomposition",
            feature_name="real_retail_and_food_services_log_change",
            frequency="monthly",
            provider_id="fred_api",
            source_url=source_url,
            min_standardization_history=minimum_history,
            ddof=1,
        ),
        build_monthly_release_events(
            price_records,
            release_block="consumer_demand_real_decomposition",
            feature_name="implicit_retail_price_log_change",
            frequency="monthly",
            provider_id="fred_api",
            source_url=source_url,
            min_standardization_history=minimum_history,
            ddof=1,
        ),
    ]
    events = pd.concat(frames, ignore_index=True).reindex(columns=EVENT_TABLE_COLUMNS)
    events = events.sort_values(
        ["release_date", "event_group_id", "reference_date", "feature_name"]
    ).reset_index(drop=True)
    validate_event_table(events)

    events_path = _project_path(project_root, outputs["retail_events"])
    manifest_path = _project_path(project_root, outputs["retail_manifest"])
    write_csv(events, events_path)
    manifest = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "stage_id": "m02_retail_real_decomposition_sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(project_root).as_posix(),
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "provider_requested": "fred_api",
        "provider_selected": selection.selected,
        "credential_policy": (
            "FRED_API_KEY is read from the runtime environment only and is never "
            "serialized into artifacts, cache identities, or logs"
        ),
        "common_vintage_count": decomposition.common_vintage_count,
        "aligned_information_date_count": (
            decomposition.aligned_information_date_count
        ),
        "cross_series_alignment_policy": (
            "at each union information date, use each series' latest vintage "
            "dated no later than that date"
        ),
        "common_reference_count": decomposition.common_reference_count,
        "event_rows": len(events),
        "available_event_rows": int(events["feature_status"].eq("available").sum()),
        "feature_status_counts": {
            str(key): int(value)
            for key, value in events["feature_status"].value_counts().sort_index().items()
        },
        "first_release_extraction_diagnostics": {
            "real_retail_and_food_services": dict(
                real_records.attrs.get("extraction_diagnostics", {})
            ),
            "implicit_retail_price": dict(
                price_records.attrs.get("extraction_diagnostics", {})
            ),
        },
        "acquisitions": [
            {
                "series_id": series_id,
                "release_id": int(retail["release_id"]),
                "provider": acquisition.provider_id,
                "cache_origin": acquisition.cache_origin,
                "path": acquisition.path.relative_to(project_root).as_posix(),
                "source_url": acquisition.source_url,
                "sha256": sha256(acquisition.content),
                "bytes": len(acquisition.content),
            }
            for series_id, acquisition in sorted(acquisitions.items())
        ],
        "generated_files": [
            {
                "path": events_path.relative_to(project_root).as_posix(),
                "sha256": sha256(events_path.read_bytes()),
                "bytes": events_path.stat().st_size,
            }
        ],
    }
    write_json(manifest, manifest_path)
    return {"events": events_path, "manifest": manifest_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m02_inference_sensitivities.yaml"),
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    outputs = build_retail_sensitivity(
        project_root=project_root,
        config_path=config_path.resolve(),
        refresh=args.refresh,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
