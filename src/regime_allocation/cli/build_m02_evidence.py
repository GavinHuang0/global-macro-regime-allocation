"""Build Model 02's point-in-time release-evidence event table.

This data-only stage verifies and reuses the existing Model 01 first-release
event artifact for ICSA, JOLTS, retail sales, housing, and capital-goods
orders.  It retrieves only four absent series through the explicitly selected
authenticated FRED provider: MORTGAGE30US, ANXAVS, EXPINF1YR, and WPSID61.
Failure of that provider aborts the run; there is no within-run keyless
fallback.

The command constructs causal standardized features, a same-vintage disjoint
motor-sales coordinate, and housing rate controls.  It does not estimate
observation equations, dependence adjustments, or Bayesian posteriors.  Model
01 inputs are read-only and every Model 02 artifact is written under a distinct
namespace with mixed-provenance lineage recorded in the manifest.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
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
from regime_allocation.data.providers.vintage_matrix import (
    FirstReleaseObservation,
    first_release_observations_from_matrix,
    load_vintage_matrix,
)
from regime_allocation.features.m02_release_evidence import (
    build_binary_control_events,
    build_monthly_rate_control_records,
    disjoint_motor_sales_records_from_m01_events,
    extract_contemporaneous_monthly_features,
)
from regime_allocation.features.release_evidence import (
    EVENT_TABLE_COLUMNS,
    build_monthly_release_events,
    validate_event_table,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "m02_release_evidence"

_EXPECTED_BLOCKS = {
    "weekly_labor_stress",
    "monthly_labor_demand",
    "consumer_demand",
    "housing_activity",
    "business_investment",
    "inflation_pressure",
}

_EXPECTED_REUSED = {
    ("weekly_labor_stress", "ICSA"): (
        "weekly_claims",
        "initial_claims_innovation",
        "expanding_log_ar1_innovation",
    ),
    ("monthly_labor_demand", "JTSJOR"): (
        "jolts",
        "job_openings_rate_change",
        "difference",
    ),
    ("monthly_labor_demand", "JTSHIR"): (
        "jolts",
        "hires_rate_change",
        "difference",
    ),
    ("monthly_labor_demand", "JTSQUR"): (
        "jolts",
        "quits_rate_change",
        "difference",
    ),
    ("monthly_labor_demand", "JTSLDR"): (
        "jolts",
        "layoffs_discharges_rate_change",
        "difference",
    ),
    ("consumer_demand", "RSFSXMV"): (
        "retail_sales",
        "retail_sales_ex_motor_vehicles_log_change",
        "log_difference",
    ),
    ("housing_activity", "HOUST"): (
        "housing",
        "housing_starts_log_change",
        "log_difference",
    ),
    ("housing_activity", "PERMIT"): (
        "housing",
        "building_permits_log_change",
        "log_difference",
    ),
    ("business_investment", "NEWORDER"): (
        "durable_goods",
        "core_capital_goods_orders_log_change",
        "log_difference",
    ),
}

_EXPECTED_NEW = {
    ("business_investment", "ANXAVS"): (
        95,
        "monthly",
        "core_capital_goods_shipments_log_change",
        "log_difference",
        "retrospective_after_reference_month_end",
    ),
    ("inflation_pressure", "EXPINF1YR"): (
        500,
        "monthly",
        "one_year_inflation_expectations_change",
        "difference",
        "contemporaneous_within_reference_month",
    ),
    ("inflation_pressure", "WPSID61"): (
        46,
        "monthly",
        "intermediate_materials_prices_log_change",
        "log_difference",
        "retrospective_after_reference_month_end",
    ),
}

NEW_OBSERVATION_COLUMNS = (
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


def _project_path(project_root: Path, configured: object) -> Path:
    root = project_root.resolve()
    candidate = (root / str(configured)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"configured path leaves the project root: {configured}")
    return candidate


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load and validate the frozen Model 02 evidence-source contract."""

    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("Model 02 evidence configuration must be a mapping")
    if config.get("schema_version") != 1:
        raise ValueError("Model 02 evidence schema_version must be 1")
    if config.get("model_id") != MODEL_ID or config.get("stage_id") != STAGE_ID:
        raise ValueError("Model 02 evidence model_id or stage_id is unexpected")
    for key in ("inputs", "provider_policy", "data", "features", "blocks", "outputs"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Model 02 evidence {key} must be a mapping")
    if set(config["blocks"]) != _EXPECTED_BLOCKS:
        raise ValueError("Model 02 evidence block set has changed")

    reused: dict[tuple[str, str], tuple[str, str, str]] = {}
    new: dict[tuple[str, str], tuple[int, str, str, str, str]] = {}
    for block_name, block in config["blocks"].items():
        if not isinstance(block, dict):
            raise ValueError(f"block {block_name} must be a mapping")
        for source in block.get("reused_sources", []):
            key = (block_name, str(source.get("series_id")))
            reused[key] = (
                str(source.get("m01_release_block")),
                str(source.get("feature_name")),
                str(source.get("transform")),
            )
        for source in block.get("new_sources", []):
            key = (block_name, str(source.get("series_id")))
            new[key] = (
                int(source.get("release_id", -1)),
                str(source.get("frequency")),
                str(source.get("feature_name")),
                str(source.get("transform")),
                str(source.get("release_timing")),
            )
    if reused != _EXPECTED_REUSED:
        raise ValueError("Model 02 reused-source contract has changed")
    if new != _EXPECTED_NEW:
        raise ValueError("Model 02 new-source contract has changed")

    derived = config["blocks"]["consumer_demand"].get("derived_sources")
    if not isinstance(derived, list) or len(derived) != 1:
        raise ValueError("consumer demand requires one derived source")
    derived_source = derived[0]
    if (
        str(derived_source.get("series_id")) != "RSAFS_MINUS_RSFSXMV"
        or list(derived_source.get("source_series_ids", [])) != ["RSAFS", "RSFSXMV"]
        or str(derived_source.get("feature_name"))
        != "motor_vehicle_sales_log_change"
        or str(derived_source.get("transform"))
        != "same_vintage_level_difference_then_log_difference"
    ):
        raise ValueError("derived retail-source contract has changed")

    controls = config["blocks"]["housing_activity"].get("controls")
    if not isinstance(controls, list) or len(controls) != 2:
        raise ValueError("housing activity requires two controls")
    mortgage, methodology = controls
    if (
        str(mortgage.get("series_id")) != "MORTGAGE30US"
        or int(mortgage.get("release_id", -1)) != 190
        or str(mortgage.get("transform")) != "reference_month_average_difference"
        or str(mortgage.get("housing_event_alignment"))
        != "latest_component_release_for_reference_month"
        or str(methodology.get("series_id")) != "MORTGAGE30US_METHOD_20221117"
        or str(methodology.get("transform")) != "post_methodology_indicator"
    ):
        raise ValueError("housing-control contract has changed")

    policy = config["provider_policy"]
    if policy.get("new_series_provider") != "fred_api":
        raise ValueError("new Model 02 evidence series require fred_api")
    if policy.get("on_selected_provider_failure") != "stop_without_fallback":
        raise ValueError("selected FRED failures must stop without fallback")
    data = config["data"]
    starts = [as_date(data[key]) for key in ("feature_start", "observation_start")]
    ends = [as_date(data[key]) for key in ("observation_end", "vintage_end")]
    if starts[1] > starts[0] or starts[1] > ends[0]:
        raise ValueError("Model 02 evidence observation window is invalid")
    if as_date(data["vintage_start"]) > ends[1]:
        raise ValueError("Model 02 evidence vintage window is invalid")
    if data.get("archive_start_latest_only") is not True:
        raise ValueError("archive_start_latest_only must be true")
    for key in ("monthly_max_release_lag_days", "weekly_max_release_lag_days"):
        if int(data[key]) < 0:
            raise ValueError(f"{key} cannot be negative")
    feature_config = config["features"]
    ddof = int(feature_config["standard_deviation_ddof"])
    for key in (
        "default_monthly_min_standardization_history",
        "short_history_min_standardization_history",
        "mortgage_control_min_standardization_history",
    ):
        minimum = int(feature_config[key])
        if minimum < 2 or ddof >= minimum:
            raise ValueError(f"invalid causal standardization setting for {key}")
    if as_date(feature_config["mortgage_methodology_change_date"]) != date(
        2022, 11, 17
    ):
        raise ValueError("mortgage methodology date must remain 2022-11-17")
    required_outputs = {
        "raw_dir",
        "processed_dir",
        "events_file",
        "features_file",
        "new_first_release_observations_file",
        "manifest",
    }
    if not required_outputs.issubset(config["outputs"]):
        raise ValueError("Model 02 evidence output paths are incomplete")
    for key in ("raw_dir", "processed_dir", "manifest"):
        if "m02" not in str(config["outputs"][key]):
            raise ValueError("every Model 02 evidence output path must use m02")
    return config, raw


def _load_verified_m01_events(
    project_root: Path,
    input_config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    """Read Model 01 events only after manifest path and hash verification."""

    manifest_path = _project_path(project_root, input_config["m01_evidence_manifest"])
    event_path = _project_path(project_root, input_config["m01_events"])
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("model_id") != "m01_deterministic_composite":
        raise ValueError("input evidence manifest is not Model 01")
    expected_relative = event_path.relative_to(project_root.resolve()).as_posix()
    candidates = [
        item
        for item in manifest.get("processed_files", [])
        if str(item.get("path")) == expected_relative
    ]
    if len(candidates) != 1:
        raise ValueError("Model 01 manifest does not identify the configured event file")
    expected_hash = str(candidates[0].get("sha256"))
    actual_hash = sha256(event_path.read_bytes())
    if actual_hash != expected_hash:
        raise ValueError("Model 01 event artifact hash does not match its manifest")
    events = pd.read_csv(
        event_path,
        parse_dates=["release_date", "reference_date", "reference_month"],
    )
    validate_event_table(events)
    return (
        events,
        manifest,
        {
            "manifest_path": manifest_path.relative_to(project_root).as_posix(),
            "manifest_sha256": sha256(manifest_bytes),
            "event_path": expected_relative,
            "event_sha256": actual_hash,
            "event_rows": len(events),
            "provider_selected": manifest.get("provider_selected"),
        },
    )


def _reblock_reused_events(
    m01_events: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Select the frozen Model 01 rows and assign Model 02 block identities."""

    frames: list[pd.DataFrame] = []
    for block_name, block in config["blocks"].items():
        for source in block.get("reused_sources", []):
            selected = m01_events.loc[
                (m01_events["series_id"] == str(source["series_id"]))
                & (m01_events["release_block"] == str(source["m01_release_block"]))
                & (m01_events["feature_name"] == str(source["feature_name"]))
                & (m01_events["transform"] == str(source["transform"]))
            ].copy()
            if selected.empty:
                raise ValueError(
                    f"Model 01 contains no reusable rows for {source['series_id']}"
                )
            selected["release_block"] = block_name
            release_text = selected["release_date"].dt.strftime("%Y-%m-%d")
            reference_text = selected["reference_date"].dt.strftime("%Y-%m-%d")
            selected["event_group_id"] = block_name + ":" + release_text
            selected["event_id"] = selected["event_group_id"] + ":" + reference_text
            frames.append(selected.reindex(columns=EVENT_TABLE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _observation_records(
    observations: tuple[FirstReleaseObservation, ...],
    *,
    series_id: str,
    release_block: str,
    frequency: str,
    max_release_lag_days: int,
    archive_start_latest_only: bool,
    provider_id: str,
    source_url: str,
    monthly_release_timing: str = "retrospective_after_reference_month_end",
) -> list[dict[str, object]]:
    """Normalize new-series first-release observations and exclusions."""

    earliest = min((item.release_date for item in observations), default=None)
    archive_latest = max(
        (
            item.reference_date
            for item in observations
            if item.release_date == earliest
        ),
        default=None,
    )
    records: list[dict[str, object]] = []
    valid_monthly_timing = {
        "retrospective_after_reference_month_end",
        "contemporaneous_within_reference_month",
    }
    if frequency == "monthly" and monthly_release_timing not in valid_monthly_timing:
        raise ValueError("unsupported monthly release-timing policy")
    for item in observations:
        reference = pd.Timestamp(item.reference_date)
        anchor = (
            reference.to_period("M").end_time.normalize().date()
            if frequency == "monthly"
            else item.reference_date
        )
        lag = (item.release_date - anchor).days
        archive_ok = (
            not archive_start_latest_only
            or item.release_date != earliest
            or item.reference_date == archive_latest
        )
        if (
            frequency == "monthly"
            and monthly_release_timing == "contemporaneous_within_reference_month"
        ):
            if not archive_ok:
                status = "archive_bootstrap_history"
            else:
                month = reference.to_period("M")
                within_month = (
                    month.start_time.date()
                    <= item.release_date
                    <= month.end_time.date()
                )
                status = (
                    "eligible"
                    if within_month
                    else "outside_contemporaneous_reference_month"
                )
        elif lag < 0:
            status = "negative_release_lag"
        elif lag > max_release_lag_days:
            status = "release_lag_exceeds_max"
        elif not archive_ok:
            status = "archive_bootstrap_history"
        else:
            status = "eligible"
        records.append(
            {
                "series_id": series_id,
                "release_block": release_block,
                "frequency": frequency,
                "reference_date": item.reference_date,
                "release_date": item.release_date,
                "value": item.value,
                "release_lag_days": lag,
                "eligible_for_feature": status == "eligible",
                "feature_eligibility_status": status,
                "provider_id": provider_id,
                "source_url": source_url,
            }
        )
    return records


def _observation_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=NEW_OBSERVATION_COLUMNS)
    if frame.empty:
        raise ValueError("no new-series first-release observations were acquired")
    frame["reference_date"] = pd.to_datetime(frame["reference_date"])
    frame["release_date"] = pd.to_datetime(frame["release_date"])
    frame["eligible_for_feature"] = frame["eligible_for_feature"].astype(bool)
    if frame.duplicated(["series_id", "reference_date"]).any():
        raise ValueError("new first-release observations repeat a reference date")
    return frame.sort_values(
        ["release_date", "release_block", "series_id", "reference_date"]
    ).reset_index(drop=True)


def _eligible_observations(
    observations: tuple[FirstReleaseObservation, ...],
    normalized: pd.DataFrame,
    *,
    series_id: str,
) -> tuple[FirstReleaseObservation, ...]:
    eligible = normalized.loc[
        (normalized["series_id"] == series_id)
        & normalized["eligible_for_feature"],
        ["reference_date", "release_date"],
    ]
    keys = {
        (row.reference_date.date(), row.release_date.date())
        for row in eligible.itertuples(index=False)
    }
    return tuple(
        item
        for item in observations
        if (item.reference_date, item.release_date) in keys
    )


def build_evidence_dataset(
    *,
    project_root: Path,
    config_path: Path,
    refresh: bool = False,
    provider: str = "fred",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Build Model 02 evidence events without likelihoods or posteriors."""

    config, config_bytes = _load_config(config_path)
    if provider.strip().lower() not in {"fred", "fred_api"}:
        raise ValueError("Model 02 new evidence must use the selected FRED provider")
    selection = select_vintage_provider("fred", environ=environ)
    client = selection.client
    if selection.selected != "fred_api":
        raise ValueError("Model 02 new evidence resolved to a non-FRED provider")

    data_config = config["data"]
    feature_config = config["features"]
    output_config = config["outputs"]
    feature_start = as_date(data_config["feature_start"])
    observation_start = as_date(data_config["observation_start"])
    observation_end = as_date(data_config["observation_end"])
    vintage_start = as_date(data_config["vintage_start"])
    vintage_end = as_date(data_config["vintage_end"])
    monthly_max_lag = int(data_config["monthly_max_release_lag_days"])
    weekly_max_lag = int(data_config["weekly_max_release_lag_days"])
    archive_policy = bool(data_config["archive_start_latest_only"])
    ddof = int(feature_config["standard_deviation_ddof"])

    m01_events, m01_manifest, m01_lineage = _load_verified_m01_events(
        project_root, config["inputs"]
    )
    reused_events = _reblock_reused_events(m01_events, config)

    processed_dir = _project_path(project_root, output_config["processed_dir"])
    manifest_path = _project_path(project_root, output_config["manifest"])
    events_path = processed_dir / str(output_config["events_file"])
    features_path = processed_dir / str(output_config["features_file"])
    observations_path = processed_dir / str(
        output_config["new_first_release_observations_file"]
    )
    raw_root = _project_path(project_root, output_config["raw_dir"])
    raw_dir = provider_raw_dir(raw_root, client.cache_namespace)

    event_frames: list[pd.DataFrame] = [reused_events]
    observation_records: list[dict[str, object]] = []
    acquisitions: list[dict[str, object]] = []

    monthly_sources: list[tuple[str, Mapping[str, Any]]] = []
    for block_name, block in config["blocks"].items():
        for source in block.get("new_sources", []):
            monthly_sources.append((block_name, source))
    for block_name, source in monthly_sources:
        series_id = str(source["series_id"])
        source_vintage_start = max(
            vintage_start, as_date(source.get("vintage_start", vintage_start))
        )
        acquisition: MatrixAcquisition = download_or_load_vintage_matrix(
            client=client,
            series_id=series_id,
            release_id=int(source["release_id"]),
            raw_dir=raw_dir,
            compatible_cache_dirs=(),
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_start=source_vintage_start,
            vintage_end=vintage_end,
            refresh=refresh,
        )
        matrix = load_vintage_matrix(acquisition.content, series_id)
        observations = first_release_observations_from_matrix(matrix)
        release_timing = str(source["release_timing"])
        if release_timing == "contemporaneous_within_reference_month":
            transformed = extract_contemporaneous_monthly_features(
                matrix,
                series_id=series_id,
                component=block_name,
                transform=str(source["transform"]),
                archive_start_latest_only=archive_policy,
            )
        else:
            transformed = extract_first_release_features(
                matrix,
                series_id=series_id,
                component=block_name,
                transform=str(source["transform"]),
                max_release_lag_days=monthly_max_lag,
                archive_start_latest_only=archive_policy,
            )
        min_history = int(source["min_standardization_history"])
        events = build_monthly_release_events(
            transformed,
            release_block=block_name,
            feature_name=str(source["feature_name"]),
            frequency="monthly",
            provider_id=acquisition.provider_id,
            source_url=acquisition.source_url,
            min_standardization_history=min_history,
            ddof=ddof,
        )
        event_frames.append(events)
        observation_records.extend(
            _observation_records(
                observations,
                series_id=series_id,
                release_block=block_name,
                frequency="monthly",
                max_release_lag_days=monthly_max_lag,
                archive_start_latest_only=archive_policy,
                provider_id=acquisition.provider_id,
                source_url=acquisition.source_url,
                monthly_release_timing=release_timing,
            )
        )
        acquisitions.append(
            {
                "series_id": series_id,
                "release_id": int(source["release_id"]),
                "provider": acquisition.provider_id,
                "cache_origin": acquisition.cache_origin,
                "path": acquisition.path.relative_to(project_root).as_posix(),
                "source_url": acquisition.source_url,
                "vintage_start": source_vintage_start.isoformat(),
                "sha256": sha256(acquisition.content),
                "bytes": len(acquisition.content),
                "first_release_observations": len(observations),
                "minimum_standardization_history": min_history,
                "release_timing": release_timing,
                "extraction_diagnostics": dict(
                    transformed.attrs.get("extraction_diagnostics", {})
                ),
            }
        )

    # Model 01 contains both overlapping retail aggregates. Rebuild only the
    # motor coordinate so the Model 02 consumer block is disjoint.
    retail_config = config["blocks"]["consumer_demand"]["derived_sources"][0]
    motor_records = disjoint_motor_sales_records_from_m01_events(m01_events)
    retail_sources = m01_events.loc[
        m01_events["series_id"].isin(["RSAFS", "RSFSXMV"])
    ]
    providers = sorted(retail_sources["provider_id"].dropna().astype(str).unique())
    if len(providers) != 1:
        raise ValueError("reused retail rows do not have one provider provenance")
    source_urls = sorted(retail_sources["source_url"].dropna().astype(str).unique())
    motor_events = build_monthly_release_events(
        motor_records,
        release_block="consumer_demand",
        feature_name=str(retail_config["feature_name"]),
        frequency="monthly",
        provider_id=providers[0],
        source_url=" | ".join(source_urls),
        min_standardization_history=int(retail_config["min_standardization_history"]),
        ddof=ddof,
    )
    motor_events["transform"] = str(retail_config["transform"])
    event_frames.append(motor_events)

    mortgage_config = config["blocks"]["housing_activity"]["controls"][0]
    mortgage_vintage_start = max(
        vintage_start,
        as_date(mortgage_config.get("vintage_start", vintage_start)),
    )
    mortgage_artifact = client.list_first_release_observations(
        "MORTGAGE30US",
        release_id=int(mortgage_config["release_id"]),
        observation_start=observation_start,
        observation_end=observation_end,
        vintage_start=mortgage_vintage_start,
        vintage_end=vintage_end,
        chunk_cache_dir=raw_dir / "provider_chunks" / "MORTGAGE30US",
        refresh_cache=refresh,
    )
    mortgage_records = _observation_records(
        mortgage_artifact.observations,
        series_id="MORTGAGE30US",
        release_block="housing_activity",
        frequency="weekly",
        max_release_lag_days=weekly_max_lag,
        archive_start_latest_only=archive_policy,
        provider_id=mortgage_artifact.provider_id,
        source_url=mortgage_artifact.source_url,
    )
    observation_records.extend(mortgage_records)
    new_observations = _observation_frame(observation_records)
    eligible_mortgage = _eligible_observations(
        mortgage_artifact.observations,
        new_observations,
        series_id="MORTGAGE30US",
    )

    housing_schedule = reused_events.loc[
        reused_events["release_block"] == "housing_activity",
        ["reference_month", "release_date"],
    ].drop_duplicates()
    # HOUST and PERMIT ordinarily share a publication. During rare catch-up
    # episodes their first appearances can differ. Attach the block-level
    # control only once, when the final component for that reference month is
    # public, so the rate control is neither duplicated nor available early.
    housing_schedule = (
        housing_schedule.groupby("reference_month", as_index=False)["release_date"]
        .max()
        .sort_values("release_date")
    )
    control_records, dummy_records = build_monthly_rate_control_records(
        housing_schedule,
        eligible_mortgage,
        methodology_change_date=as_date(
            feature_config["mortgage_methodology_change_date"]
        ),
    )
    mortgage_events = build_monthly_release_events(
        control_records,
        release_block="housing_activity",
        feature_name=str(mortgage_config["feature_name"]),
        frequency="monthly",
        provider_id=mortgage_artifact.provider_id,
        source_url=mortgage_artifact.source_url,
        min_standardization_history=int(
            mortgage_config["min_standardization_history"]
        ),
        ddof=ddof,
    )
    methodology_config = config["blocks"]["housing_activity"]["controls"][1]
    methodology_events = build_binary_control_events(
        dummy_records,
        release_block="housing_activity",
        feature_name=str(methodology_config["feature_name"]),
        series_id=str(methodology_config["series_id"]),
        provider_id=mortgage_artifact.provider_id,
        source_url=mortgage_artifact.source_url,
    )
    event_frames.extend([mortgage_events, methodology_events])
    observation_digest = hashlib.sha256()
    for item in mortgage_artifact.observations:
        observation_digest.update(
            (
                f"{item.reference_date.isoformat()},"
                f"{item.release_date.isoformat()},{item.value:.17g}\n"
            ).encode("utf-8")
        )
    acquisitions.append(
        {
            "series_id": "MORTGAGE30US",
            "release_id": int(mortgage_config["release_id"]),
            "provider": mortgage_artifact.provider_id,
            "cache_origin": "provider_managed_chunk_cache",
            "cache_directory": (
                raw_dir / "provider_chunks" / "MORTGAGE30US"
            ).relative_to(project_root).as_posix(),
            "source_url": mortgage_artifact.source_url,
            "vintage_start": mortgage_vintage_start.isoformat(),
            "first_release_observations": len(mortgage_artifact.observations),
            "normalized_observations_sha256": observation_digest.hexdigest(),
            "minimum_standardization_history": int(
                mortgage_config["min_standardization_history"]
            ),
        }
    )

    all_events = pd.concat(event_frames, ignore_index=True).reindex(
        columns=EVENT_TABLE_COLUMNS
    )
    all_events = all_events.loc[
        pd.to_datetime(all_events["reference_date"]).dt.date >= feature_start
    ].copy()
    all_events = all_events.sort_values(
        ["release_date", "event_group_id", "reference_date", "feature_name"]
    ).reset_index(drop=True)
    validate_event_table(all_events)
    available = all_events.loc[all_events["feature_status"] == "available"].copy()
    if available.empty:
        raise ValueError("no available Model 02 evidence features were produced")

    write_csv(new_observations, observations_path)
    write_csv(all_events, events_path)
    write_csv(available, features_path)

    generated = [observations_path, events_path, features_path]
    providers_used = sorted(all_events["provider_id"].dropna().astype(str).unique())
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "evidence_set_id": config["evidence_set_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "causal evidence acquisition and feature construction only",
        "configuration": config_path.relative_to(project_root).as_posix(),
        "configuration_sha256": sha256(config_bytes),
        "input_lineage": {"m01_evidence": m01_lineage},
        "mixed_provenance": True,
        "provenance_note": (
            "Reused rows retain the provider recorded by Model 01; only "
            "MORTGAGE30US, ANXAVS, EXPINF1YR, and WPSID61 were acquired "
            "for this stage through fred_api."
        ),
        "new_series_provider_requested": selection.requested,
        "new_series_provider_selected": selection.selected,
        "new_series_failure_policy": "stop_without_fallback",
        "credential_policy": (
            "FRED_API_KEY is read from the runtime environment only and is "
            "never serialized into artifacts, cache identities, or logs"
        ),
        "providers_used_in_events": providers_used,
        "m01_provider_selected": m01_manifest.get("provider_selected"),
        "feature_start": feature_start.isoformat(),
        "observation_end": observation_end.isoformat(),
        "vintage_end": vintage_end.isoformat(),
        "archive_start_policy": "latest_reference_period_only",
        "monthly_release_timing_policy": {
            "ANXAVS": "retrospective_after_reference_month_end",
            "EXPINF1YR": "contemporaneous_within_reference_month",
            "WPSID61": "retrospective_after_reference_month_end",
        },
        "release_lag_days_note": (
            "Monthly release_lag_days remains measured from reference-month "
            "end. Eligible EXPINF1YR rows are contemporaneous and therefore "
            "normally have negative values; all other monthly series retain "
            "the retrospective non-negative-lag rule."
        ),
        "reused_event_rows_before_derived_features": len(reused_events),
        "new_first_release_observation_rows": len(new_observations),
        "eligible_new_first_release_observation_rows": int(
            new_observations["eligible_for_feature"].sum()
        ),
        "new_observation_status_counts": {
            str(key): int(value)
            for key, value in new_observations[
                "feature_eligibility_status"
            ].value_counts().items()
        },
        "event_rows": len(all_events),
        "available_feature_rows": len(available),
        "event_block_counts": {
            str(key): int(value)
            for key, value in all_events["release_block"].value_counts().items()
        },
        "event_status_counts": {
            str(key): int(value)
            for key, value in all_events["feature_status"].value_counts().items()
        },
        "first_event_release_date": all_events["release_date"].min().date().isoformat(),
        "last_event_release_date": all_events["release_date"].max().date().isoformat(),
        "retail_coordinate_policy": (
            "RSFSXMV plus same-vintage RSAFS-minus-RSFSXMV motor component"
        ),
        "housing_rate_control_policy": (
            "first-release weekly values averaged by reference month, "
            "differenced, causally standardized, and aligned to the latest "
            "component release for each housing reference month"
        ),
        "inflation_block_event_policy": "asynchronous_feature_events",
        "raw_acquisitions_new_series_only": sorted(
            acquisitions, key=lambda item: str(item["series_id"])
        ),
        "generated_files": [
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": sha256(path.read_bytes()),
                "bytes": path.stat().st_size,
            }
            for path in generated
        ],
    }
    write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "events": events_path,
        "features": features_path,
        "new_observations": observations_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m02_release_evidence.yaml"),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Reacquire only the four new series from the FRED provider.",
    )
    parser.add_argument(
        "--provider",
        choices=("fred",),
        default="fred",
        help="New-series provider; Model 02 freezes authenticated FRED.",
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
