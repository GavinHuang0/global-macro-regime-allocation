"""Build additive point-in-time evidence rows for six Model 02 experiments.

The selected ``student_t_7_combined`` input artifact is verified and copied
unchanged into an extended event table. Candidate rows live in distinct block
names and cannot affect the selected baseline unless an inference variant
explicitly opts into those model IDs. Authenticated FRED is the only provider
for newly acquired series; failure stops the stage without fallback.

The stage implements six predeclared tests: current-month manufacturing
surveys, continued claims, cleaner consumer quantities, single-family housing,
independent import-price pressure, and a capital-goods backlog ratio. It does
not fit likelihoods, run the Bayesian filter, or promote a candidate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from regime_allocation.cli.build_m02_evidence import _load_verified_m01_events
from regime_allocation.data.dataset_acquisition import (
    as_date,
    download_or_load_vintage_matrix,
    sha256,
    write_csv,
    write_json,
)
from regime_allocation.data.first_release import extract_first_release_features
from regime_allocation.data.providers import select_vintage_provider
from regime_allocation.data.providers.vintage_matrix import (
    FirstReleaseObservation,
    load_vintage_matrix,
)
from regime_allocation.features.m02_evidence_experiments import (
    extract_first_release_levels,
    extract_first_release_log_change,
    same_vintage_log_ratio_matrix,
)
from regime_allocation.features.m02_release_evidence import (
    build_binary_control_events,
    build_monthly_rate_control_records,
)
from regime_allocation.features.release_evidence import (
    EVENT_TABLE_COLUMNS,
    build_monthly_release_events,
    validate_event_table,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "m02_evidence_block_experiments"

_EXPECTED_NEW_SOURCES = {
    "NOCDISA066MSFRBNY": 321,
    "PPCDISA066MSFRBNY": 321,
    "NOCDFSA066MSFRBPHI": 351,
    "PPCDFSA066MSFRBPHI": 351,
    "MARTSSM44W72USS": 9,
    "TOTALSA": 93,
    "HOUST1F": 27,
    "PERMIT1": 27,
    "HSN1F": 97,
    "IREXPETCOM": 188,
    "ANXAUO": 95,
    "ANXAVS": 95,
}


def _project_path(root: Path, configured: object) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / str(configured)).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"configured path leaves the project root: {configured}")
    return candidate


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("evidence experiment configuration must be a mapping")
    if config.get("schema_version") != 1:
        raise ValueError("evidence experiment schema_version must be 1")
    if config.get("model_id") != MODEL_ID or config.get("stage_id") != STAGE_ID:
        raise ValueError("evidence experiment model_id or stage_id is invalid")
    for key in ("sources", "provider_policy", "data", "priorities", "outputs"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"evidence experiment {key} must be a mapping")
    for key in (
        "baseline_evidence_manifest",
        "baseline_evidence_events",
        "baseline_new_observations",
        "m01_evidence_manifest",
        "m01_events",
    ):
        if key not in config["sources"]:
            raise ValueError(f"evidence experiment sources omit {key}")
    if config["provider_policy"].get("provider") != "fred_api":
        raise ValueError("new experiment series require authenticated FRED")
    if config["provider_policy"].get("on_failure") != "stop_without_fallback":
        raise ValueError("provider failure must stop without fallback")

    declared: dict[str, int] = {}
    for priority in config["priorities"].values():
        sources = list(priority.get("sources", ()))
        if "source" in priority:
            sources.append(priority["source"])
        for source in sources:
            if "release_id" in source and str(source.get("series_id")) != "CCSA":
                declared[str(source["series_id"])] = int(source["release_id"])
    if declared != _EXPECTED_NEW_SOURCES:
        raise ValueError("six-priority FRED source contract has changed")
    data = config["data"]
    if as_date(data["observation_start"]) > as_date(data["observation_end"]):
        raise ValueError("observation window is invalid")
    if as_date(data["vintage_start"]) > as_date(data["vintage_end"]):
        raise ValueError("vintage window is invalid")
    if data.get("archive_start_latest_only") is not True:
        raise ValueError("archive_start_latest_only must remain true")
    return config, raw


def _verified_generated_file(
    root: Path,
    manifest_path: Path,
    configured_file: object,
    *,
    expected_stage: str,
) -> tuple[Path, dict[str, Any], bytes]:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("model_id") != MODEL_ID or manifest.get("stage_id") != expected_stage:
        raise ValueError(f"upstream manifest stage mismatch: {manifest_path}")
    path = _project_path(root, configured_file)
    relative = path.relative_to(root.resolve()).as_posix()
    matches = [
        row
        for row in manifest.get("generated_files", ())
        if str(row.get("path")) == relative
    ]
    if len(matches) != 1:
        raise ValueError(f"upstream manifest does not identify {relative}")
    if sha256(path.read_bytes()) != str(matches[0].get("sha256")):
        raise ValueError(f"upstream file hash mismatch: {relative}")
    return path, manifest, manifest_bytes


def _reblock_events(
    rows: pd.DataFrame,
    *,
    release_block: str,
    feature_name: str | None = None,
) -> pd.DataFrame:
    selected = rows.copy().reindex(columns=EVENT_TABLE_COLUMNS)
    if selected.empty:
        raise ValueError(f"cannot reblock an empty event selection: {release_block}")
    selected["release_block"] = release_block
    if feature_name is not None:
        if selected["feature_name"].nunique() != 1:
            raise ValueError("feature_name override requires one source feature")
        selected["feature_name"] = feature_name
    release = pd.to_datetime(selected["release_date"]).dt.strftime("%Y-%m-%d")
    reference = pd.to_datetime(selected["reference_date"]).dt.strftime("%Y-%m-%d")
    selected["event_group_id"] = release_block + ":" + release
    selected["event_id"] = selected["event_group_id"] + ":" + reference
    return selected


def _monthly_events(
    records: pd.DataFrame,
    *,
    release_block: str,
    feature_name: str,
    provider_id: str,
    source_url: str,
    minimum_history: int,
    ddof: int,
) -> pd.DataFrame:
    return build_monthly_release_events(
        records,
        release_block=release_block,
        feature_name=feature_name,
        frequency="monthly",
        provider_id=provider_id,
        source_url=source_url,
        min_standardization_history=minimum_history,
        ddof=ddof,
    )


def _mortgage_observations(frame: pd.DataFrame) -> tuple[FirstReleaseObservation, ...]:
    selected = frame.loc[
        frame["series_id"].astype(str).eq("MORTGAGE30US")
        & frame["eligible_for_feature"].astype(str).str.lower().eq("true")
    ].copy()
    if selected.empty:
        raise ValueError("baseline evidence has no eligible mortgage observations")
    return tuple(
        FirstReleaseObservation(
            reference_date=pd.Timestamp(row.reference_date).date(),
            release_date=pd.Timestamp(row.release_date).date(),
            value=float(row.value),
        )
        for row in selected.sort_values("reference_date").itertuples(index=False)
    )


def _housing_control_events(
    responses: pd.DataFrame,
    mortgage: tuple[FirstReleaseObservation, ...],
    *,
    release_block: str,
    provider_id: str,
    source_url: str,
    minimum_history: int,
    ddof: int,
) -> list[pd.DataFrame]:
    schedule = (
        responses.loc[:, ["reference_month", "release_date"]]
        .drop_duplicates()
        .groupby("reference_month", as_index=False)["release_date"]
        .max()
        .sort_values("release_date")
    )
    changes, dummies = build_monthly_rate_control_records(schedule, mortgage)
    levels = changes.copy()
    levels["previous_value_as_of_release"] = np.nan
    levels["transform"] = "reference_month_average_level"
    levels["transformed_value"] = levels["current_value"]
    level_events = _monthly_events(
        levels,
        release_block=release_block,
        feature_name="mortgage_rate_monthly_average_level_control",
        provider_id=provider_id,
        source_url=source_url,
        minimum_history=minimum_history,
        ddof=ddof,
    )
    change_events = _monthly_events(
        changes,
        release_block=release_block,
        feature_name="mortgage_rate_monthly_average_change_control",
        provider_id=provider_id,
        source_url=source_url,
        minimum_history=minimum_history,
        ddof=ddof,
    )
    dummy_events = build_binary_control_events(
        dummies,
        release_block=release_block,
        feature_name="mortgage_rate_post_2022_11_17_methodology",
        series_id="MORTGAGE30US_METHOD_20221117",
        provider_id=provider_id,
        source_url=source_url,
    )
    return [level_events, change_events, dummy_events]


def _coverage(events: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (block, feature, series_id), group in events.groupby(
        ["release_block", "feature_name", "series_id"], sort=True
    ):
        available = group.loc[group["feature_status"].astype(str).eq("available")]
        records.append(
            {
                "release_block": block,
                "feature_name": feature,
                "series_id": series_id,
                "event_rows": len(group),
                "available_rows": len(available),
                "first_event_release_date": group["release_date"].min(),
                "last_event_release_date": group["release_date"].max(),
                "first_available_release_date": (
                    pd.NaT if available.empty else available["release_date"].min()
                ),
                "last_available_release_date": (
                    pd.NaT if available.empty else available["release_date"].max()
                ),
                "first_reference_month": group["reference_month"].min(),
                "last_reference_month": group["reference_month"].max(),
                "median_release_lag_days": float(
                    pd.to_numeric(group["release_lag_days"], errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def build_evidence_experiments(
    *,
    project_root: Path,
    config_path: Path,
    refresh: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Acquire candidate data and publish isolated extended evidence rows."""

    root = project_root.resolve()
    config_path = _project_path(root, config_path)
    config, config_bytes = _load_config(config_path)
    sources = config["sources"]
    base_manifest_path = _project_path(root, sources["baseline_evidence_manifest"])
    base_events_path, base_manifest, base_manifest_bytes = _verified_generated_file(
        root,
        base_manifest_path,
        sources["baseline_evidence_events"],
        expected_stage="m02_release_evidence",
    )
    base_observations_path, _, _ = _verified_generated_file(
        root,
        base_manifest_path,
        sources["baseline_new_observations"],
        expected_stage="m02_release_evidence",
    )
    base_events = pd.read_csv(
        base_events_path,
        parse_dates=["release_date", "reference_date", "reference_month"],
    )
    validate_event_table(base_events)
    base_observations = pd.read_csv(
        base_observations_path,
        parse_dates=["reference_date", "release_date"],
    )
    m01_events, _, m01_lineage = _load_verified_m01_events(root, sources)

    selection = select_vintage_provider("fred", environ=environ)
    if selection.selected != "fred_api":
        raise ValueError("evidence experiment resolved to a non-FRED provider")
    client = selection.client
    data = config["data"]
    observation_start = as_date(data["observation_start"])
    observation_end = as_date(data["observation_end"])
    vintage_start = as_date(data["vintage_start"])
    vintage_end = as_date(data["vintage_end"])
    maximum_lag = int(data["maximum_release_lag_days"])
    archive_policy = bool(data["archive_start_latest_only"])
    feature_start = as_date(data["feature_start"])
    ddof = int(data["standard_deviation_ddof"])
    outputs = config["outputs"]
    raw_dir = _project_path(root, outputs["raw_dir"])
    processed_dir = _project_path(root, outputs["processed_dir"])

    matrices: dict[str, pd.DataFrame] = {}
    acquisitions: list[dict[str, object]] = []
    for series_id, release_id in _EXPECTED_NEW_SOURCES.items():
        artifact = download_or_load_vintage_matrix(
            client=client,
            series_id=series_id,
            release_id=release_id,
            raw_dir=raw_dir,
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_start=vintage_start,
            vintage_end=vintage_end,
            refresh=refresh,
        )
        matrices[series_id] = load_vintage_matrix(artifact.content, series_id)
        acquisitions.append(
            {
                "series_id": series_id,
                "release_id": release_id,
                "provider": artifact.provider_id,
                "source_url": artifact.source_url,
                "cache_origin": artifact.cache_origin,
                "path": artifact.path.relative_to(root).as_posix(),
                "sha256": sha256(artifact.content),
                "bytes": len(artifact.content),
            }
        )

    event_frames: list[pd.DataFrame] = []
    priorities = config["priorities"]

    survey = priorities["priority_01_manufacturing_surveys"]
    for source in survey["sources"]:
        series_id = str(source["series_id"])
        records = extract_first_release_levels(
            matrices[series_id],
            series_id=series_id,
            component=str(source["release_block"]),
            transform_name=str(source["transform"]),
            contemporaneous=True,
            max_release_lag_days=maximum_lag,
            archive_start_latest_only=archive_policy,
        )
        event_frames.append(
            _monthly_events(
                records,
                release_block=str(source["release_block"]),
                feature_name=str(source["feature_name"]),
                provider_id=client.provider_id,
                source_url=client.series_page_url(series_id),
                minimum_history=int(survey["minimum_standardization_history"]),
                ddof=ddof,
            )
        )

    claims = priorities["priority_02_continuing_claims"]["source"]
    ccsa = m01_events.loc[
        m01_events["series_id"].astype(str).eq(str(claims["series_id"]))
        & m01_events["release_block"].astype(str).eq(str(claims["m01_release_block"]))
        & m01_events["feature_name"].astype(str).eq(str(claims["feature_name"]))
    ].copy()
    event_frames.append(
        _reblock_events(ccsa, release_block=str(claims["release_block"]))
    )

    consumer = priorities["priority_03_clean_consumer_quantities"]
    for source in consumer["sources"]:
        series_id = str(source["series_id"])
        records = extract_first_release_features(
            matrices[series_id],
            series_id=series_id,
            component=str(source["release_block"]),
            transform=str(source["transform"]),
            max_release_lag_days=maximum_lag,
            archive_start_latest_only=archive_policy,
        )
        event_frames.append(
            _monthly_events(
                records,
                release_block=str(source["release_block"]),
                feature_name=str(source["feature_name"]),
                provider_id=client.provider_id,
                source_url=client.series_page_url(series_id),
                minimum_history=int(consumer["minimum_standardization_history"]),
                ddof=ddof,
            )
        )

    housing = priorities["priority_04_single_family_housing"]
    housing_response_frames: dict[str, list[pd.DataFrame]] = {}
    for source in housing["sources"]:
        series_id = str(source["series_id"])
        block = str(source["release_block"])
        records = extract_first_release_features(
            matrices[series_id],
            series_id=series_id,
            component=block,
            transform=str(source["transform"]),
            max_release_lag_days=maximum_lag,
            archive_start_latest_only=archive_policy,
        )
        events = _monthly_events(
            records,
            release_block=block,
            feature_name=str(source["feature_name"]),
            provider_id=client.provider_id,
            source_url=client.series_page_url(series_id),
            minimum_history=int(housing["minimum_standardization_history"]),
            ddof=ddof,
        )
        event_frames.append(events)
        housing_response_frames.setdefault(block, []).append(events)
    mortgage = _mortgage_observations(base_observations)
    mortgage_rows = base_observations.loc[
        base_observations["series_id"].astype(str).eq("MORTGAGE30US")
    ]
    mortgage_providers = sorted(mortgage_rows["provider_id"].dropna().astype(str).unique())
    mortgage_urls = sorted(mortgage_rows["source_url"].dropna().astype(str).unique())
    if len(mortgage_providers) != 1 or not mortgage_urls:
        raise ValueError("mortgage observation provenance is not unique")
    for block, frames in housing_response_frames.items():
        response_events = pd.concat(frames, ignore_index=True, sort=False)
        event_frames.extend(
            _housing_control_events(
                response_events,
                mortgage,
                release_block=block,
                provider_id=mortgage_providers[0],
                source_url=" | ".join(mortgage_urls),
                minimum_history=int(housing["minimum_standardization_history"]),
                ddof=ddof,
            )
        )

    import_priority = priorities["priority_05_import_price_pressure"]
    import_source = import_priority["source"]
    import_id = str(import_source["series_id"])
    import_records = extract_first_release_log_change(
        matrices[import_id],
        series_id=import_id,
        component=str(import_source["release_block"]),
        lag_months=12,
        transform_name=str(import_source["transform"]),
        max_release_lag_days=maximum_lag,
        archive_start_latest_only=archive_policy,
    )
    event_frames.append(
        _monthly_events(
            import_records,
            release_block=str(import_source["release_block"]),
            feature_name=str(import_source["feature_name"]),
            provider_id=client.provider_id,
            source_url=client.series_page_url(import_id),
            minimum_history=int(import_priority["minimum_standardization_history"]),
            ddof=ddof,
        )
    )

    business = priorities["priority_06_capital_goods_backlog"]
    reused = business["reused_orders"]
    orders = base_events.loc[
        base_events["series_id"].astype(str).eq(str(reused["series_id"]))
        & base_events["release_block"].astype(str).eq(
            str(reused["baseline_release_block"])
        )
        & base_events["feature_name"].astype(str).eq(str(reused["feature_name"]))
    ].copy()
    event_frames.append(
        _reblock_events(orders, release_block=str(business["derived"]["release_block"]))
    )
    derived = business["derived"]
    ratio_matrix = same_vintage_log_ratio_matrix(
        matrices["ANXAUO"],
        matrices["ANXAVS"],
        derived_series_id=str(derived["series_id"]),
    )
    ratio_records = extract_first_release_levels(
        ratio_matrix,
        series_id=str(derived["series_id"]),
        component=str(derived["release_block"]),
        transform_name=str(derived["transform"]),
        contemporaneous=False,
        max_release_lag_days=maximum_lag,
        archive_start_latest_only=archive_policy,
    )
    event_frames.append(
        _monthly_events(
            ratio_records,
            release_block=str(derived["release_block"]),
            feature_name=str(derived["feature_name"]),
            provider_id=client.provider_id,
            source_url=(
                client.series_page_url("ANXAUO")
                + " | "
                + client.series_page_url("ANXAVS")
            ),
            minimum_history=int(business["minimum_standardization_history"]),
            ddof=ddof,
        )
    )

    candidate = pd.concat(event_frames, ignore_index=True, sort=False).reindex(
        columns=EVENT_TABLE_COLUMNS
    )
    candidate = candidate.loc[
        pd.to_datetime(candidate["reference_date"]).dt.date >= feature_start
    ].copy()
    candidate = candidate.sort_values(
        ["release_date", "event_group_id", "reference_date", "feature_name"]
    ).reset_index(drop=True)
    validate_event_table(candidate)
    extended = pd.concat([base_events, candidate], ignore_index=True, sort=False).reindex(
        columns=EVENT_TABLE_COLUMNS
    )
    extended = extended.sort_values(
        ["release_date", "event_group_id", "reference_date", "feature_name"]
    ).reset_index(drop=True)
    validate_event_table(extended)
    coverage = _coverage(candidate)

    candidate_path = processed_dir / str(outputs["candidate_events"])
    extended_path = processed_dir / str(outputs["extended_events"])
    coverage_path = processed_dir / str(outputs["coverage"])
    write_csv(candidate, candidate_path)
    write_csv(extended, extended_path)
    write_csv(coverage, coverage_path)
    generated = [candidate_path, extended_path, coverage_path]

    manifest_path = _project_path(root, outputs["manifest"])
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(root).as_posix(),
        "configuration_sha256": sha256(config_bytes),
        "provider_requested": selection.requested,
        "provider_selected": selection.selected,
        "credential_policy": (
            "FRED_API_KEY is read from runtime memory only and is never "
            "serialized into artifacts, paths, cache identities, or logs"
        ),
        "baseline_evidence": {
            "manifest": base_manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": sha256(base_manifest_bytes),
            "events": base_events_path.relative_to(root).as_posix(),
            "events_sha256": sha256(base_events_path.read_bytes()),
            "evidence_set_id": base_manifest.get("evidence_set_id"),
            "rows": len(base_events),
        },
        "m01_lineage": m01_lineage,
        "candidate_rows": len(candidate),
        "candidate_available_rows": int(
            candidate["feature_status"].astype(str).eq("available").sum()
        ),
        "extended_rows": len(extended),
        "candidate_blocks": sorted(candidate["release_block"].astype(str).unique()),
        "candidate_series": sorted(candidate["series_id"].astype(str).unique()),
        "coverage_caveat": (
            "archive bootstrap histories are excluded; coverage begins only "
            "when a first-release vintage is actually observable"
        ),
        "acquisitions": sorted(acquisitions, key=lambda row: str(row["series_id"])),
        "generated_files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path.read_bytes()),
                "bytes": path.stat().st_size,
            }
            for path in generated
        ],
    }
    write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "candidate_events": candidate_path,
        "extended_events": extended_path,
        "coverage": coverage_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m02_evidence_block_experiments.yaml"),
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    outputs = build_evidence_experiments(
        project_root=root,
        config_path=config_path,
        refresh=args.refresh,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
