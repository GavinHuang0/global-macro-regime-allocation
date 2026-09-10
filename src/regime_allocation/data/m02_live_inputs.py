"""Acquire dated operational inputs for the promoted Model 02 graph.

Historical publication builders deliberately require their original snapshots.
This module uses the same feature and mapping functions with a new information
cutoff, records actual coverage, and never writes to publication namespaces.
Credentials are checked before any file or network activity. No observations
are imputed and unexpected historical component gaps stop the build.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from regime_allocation.cli.build_m02_probability_map import _load_config as _mapping_config
from regime_allocation.cli.build_m02_scores import (
    _axis_score_availability,
)
from regime_allocation.cli.build_m02_scores import (
    _load_config as _score_config,
)
from regime_allocation.data.dataset_acquisition import (
    MatrixAcquisition,
    as_date,
    download_or_load_exact_vintage_matrix,
    download_or_load_vintage_matrix,
    sha256,
    write_csv,
    write_json,
)
from regime_allocation.data.first_release import extract_first_release_features
from regime_allocation.data.providers import select_vintage_provider
from regime_allocation.data.providers.vintage_matrix import (
    VintageMatrixProvider,
    load_vintage_matrix,
    vintage_date_from_column,
)
from regime_allocation.features.m02_feature_revision import (
    build_business_investment_revision_events,
)
from regime_allocation.features.m02_retail_sensitivities import (
    build_real_retail_decomposition_matrices,
)
from regime_allocation.features.release_evidence import (
    EVENT_TABLE_COLUMNS,
    build_claims_release_events,
    build_monthly_release_events,
    validate_event_table,
)
from regime_allocation.models.m02_soft_composite.probability_map import (
    causal_quadrant_mapping_history,
    component_disagreement_history,
)
from regime_allocation.models.m02_soft_composite.revisions import (
    aggregate_axis_revision_errors,
    component_revision_errors,
    first_release_standardization_scales,
    required_exact_vintages,
)
from regime_allocation.models.m02_soft_composite.scores import (
    ALL_COMPONENTS,
    GROWTH_COMPONENTS,
    INFLATION_COMPONENTS,
    build_composite_scores,
)

CONFIG_FILES = {
    "scores": "configs/models/m02_soft_composite.yaml",
    "mapping": "configs/models/m02_probability_map.yaml",
    "m01_evidence": "configs/models/m01_non_defining_release_evidence.yaml",
    "m02_evidence": "configs/models/m02_release_evidence.yaml",
    "retail": "configs/models/m02_inference_sensitivities.yaml",
    "feature_revision": "configs/models/m02_feature_revision.yaml",
}

_LOGGER = logging.getLogger(__name__)


def _output_directory(root: Path, requested: Path) -> Path:
    """Require an operational destination; resolve symlinks before checking."""

    output = (root / requested).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("live input output_dir must be contained in the project")
    relative = output.relative_to(root).parts
    allowed = (
        relative[0] == "outputs"
        or relative[:2] == ("results", "operational")
        or relative[:3] == ("data", "processed", "m02_live")
    )
    if not allowed:
        raise ValueError(
            "live inputs must use outputs/, results/operational/, or "
            "data/processed/m02_live/; historical published namespaces are protected"
        )
    # Existing child symlinks must not redirect any acquisition or output file.
    if output.exists():
        for child in output.rglob("*"):
            if not child.resolve().is_relative_to(output):
                raise ValueError("live input output_dir contains a path escaping its namespace")
    return output


def _declaration(root: Path, path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(payload),
        "bytes": len(payload),
    }


def _assert_cutoff(frame: pd.DataFrame, columns: tuple[str, ...], cutoff: date) -> None:
    for column in columns:
        if column not in frame:
            continue
        dates = pd.to_datetime(frame[column], errors="raise")
        if dates.dropna().gt(pd.Timestamp(cutoff)).any():
            raise ValueError(f"{column} contains information after as_of={cutoff}")


def _checked_matrix(
    acquisition: MatrixAcquisition,
    *,
    series_id: str,
    cutoff: date,
    observation_end: date,
    exact_vintages: tuple[date, ...] | None = None,
) -> pd.DataFrame:
    if acquisition.provider_id != "fred_api":
        raise ValueError("operational Model 02 inputs require authenticated FRED provenance")
    matrix = load_vintage_matrix(acquisition.content, series_id, requested_vintages=exact_vintages)
    if any(vintage_date_from_column(str(column)) > cutoff for column in matrix):
        raise ValueError(f"{series_id} vintage matrix contains data after as_of={cutoff}")
    if matrix.index.has_duplicates or matrix.index.max() > pd.Timestamp(observation_end):
        raise ValueError(f"{series_id} vintage matrix has invalid reference-date coverage")
    values = matrix.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{series_id} vintage matrix contains infinite observations")
    metadata = acquisition.path.with_name(acquisition.path.name + ".metadata.json")
    if metadata.exists():
        stored = json.loads(metadata.read_text(encoding="utf-8"))
        if stored.get("content_sha256") != sha256(acquisition.content):
            raise ValueError(f"{series_id} cached matrix hash does not match its metadata")
    return matrix


class _Acquisition:
    def __init__(
        self, root: Path, output: Path, client: VintageMatrixProvider, cutoff: date, refresh: bool
    ) -> None:
        self.root = root
        self.output = output
        self.client = client
        self.cutoff = cutoff
        self.refresh = refresh
        self.records: list[dict[str, Any]] = []

    def matrix(
        self,
        series_id: str,
        *,
        purpose: str,
        release_id: int,
        observation_start: date,
        observation_end: date,
        vintage_start: date,
        vintage_end: date | None = None,
    ) -> tuple[pd.DataFrame, str]:
        end = min(self.cutoff, vintage_end or self.cutoff)
        if vintage_start > end or observation_start > observation_end:
            raise ValueError(f"invalid operational acquisition window for {series_id}")
        _LOGGER.info("Acquiring %s %s through %s", purpose, series_id, end)
        acquired = download_or_load_vintage_matrix(
            client=self.client,
            series_id=series_id,
            release_id=release_id,
            raw_dir=self.output / "raw" / "fred" / purpose,
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_start=vintage_start,
            vintage_end=end,
            refresh=self.refresh,
        )
        matrix = _checked_matrix(
            acquired, series_id=series_id, cutoff=end, observation_end=observation_end
        )
        self.records.append(
            {
                **_declaration(self.root, acquired.path),
                "series_id": series_id,
                "purpose": purpose,
                "provider": acquired.provider_id,
                "source_url": acquired.source_url,
                "cache_origin": acquired.cache_origin,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "vintage_start": vintage_start,
                "vintage_end": end,
                "latest_vintage": max(vintage_date_from_column(str(c)) for c in matrix),
            }
        )
        return matrix, acquired.source_url


def _score_frames(
    components: pd.DataFrame, config: Mapping[str, Any], cutoff: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain incomplete months and audit gaps without repairing their values."""

    if components.empty or components.duplicated(["reference_month", "component"]).any():
        raise ValueError("score components are empty or contain overlapping source months")
    _assert_cutoff(components, ("release_date", "reference_month"), cutoff)
    transformed = components.pivot(
        index="reference_month", columns="component", values="transformed_value"
    ).sort_index()
    releases = components.pivot(
        index="reference_month", columns="component", values="release_date"
    ).sort_index()
    reference_start = pd.Timestamp(config["data"]["reference_start"])
    component_end = transformed.index.max()
    full_index = pd.date_range(
        min(reference_start, transformed.index.min()), component_end, freq="MS"
    )
    transformed = transformed.reindex(index=full_index, columns=list(ALL_COMPONENTS))
    releases = releases.reindex(index=full_index, columns=list(ALL_COMPONENTS))
    features = build_composite_scores(
        transformed,
        min_history=int(config["features"]["min_history_months"]),
        ddof=int(config["features"]["standard_deviation_ddof"]),
    )
    for axis, names in (("growth", GROWTH_COMPONENTS), ("inflation", INFLATION_COMPONENTS)):
        features[f"{axis}_score_available_at"] = _axis_score_availability(
            releases, components=names, scores_available=features[f"{axis}_score"].notna()
        )
    complete = features[["growth_score", "inflation_score"]].notna().all(axis=1)
    features["score_available_at"] = (
        features[["growth_score_available_at", "inflation_score_available_at"]]
        .max(axis=1)
        .where(complete)
    )
    features["data_status"] = "score_available"
    features.loc[~complete, "data_status"] = "insufficient_standardization_history"
    features.loc[transformed.isna().any(axis=1), "data_status"] = "missing_component_feature"
    features = features.loc[features.index >= reference_start].copy()
    features.index.name = "reference_month"
    if features["score_available_at"].notna().sum() == 0:
        raise ValueError("no complete operational Model 02 scores are available")
    _assert_cutoff(
        features,
        ("growth_score_available_at", "inflation_score_available_at", "score_available_at"),
        cutoff,
    )
    declared = {
        (str(component), pd.Timestamp(month))
        for component, months in config["data"].get("expected_missing_component_months", {}).items()
        for month in months
        if pd.Timestamp(month) <= pd.Timestamp(config["data"]["reference_end"])
    }
    gaps: list[dict[str, Any]] = []
    for component in ALL_COMPONENTS:
        missing_months = features.index[features[f"{component}_transformed"].isna()]
        for month in missing_months:
            age = (pd.Timestamp(cutoff) - month.to_period("M").end_time.normalize()).days
            status = (
                "declared_historical_gap"
                if (component, month) in declared
                else "awaiting_release_within_lag_limit"
                if age <= int(config["data"]["max_release_lag_days"])
                else "unexpected_historical_gap"
            )
            gaps.append({"reference_month": month, "component": component, "status": status})
    audit = pd.DataFrame(gaps, columns=["reference_month", "component", "status"])
    return features, audit


def _build_scores(
    acquire: _Acquisition, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = config["data"]
    # Defining sources are retrospective monthly observations. The newest
    # possibly released reference is the month before the information cutoff.
    observation_end = (pd.Timestamp(acquire.cutoff).to_period("M") - 1).start_time.date()
    parts: list[pd.DataFrame] = []
    for component, declaration in config["components"].items():
        for source in declaration["sources"]:
            matrix, source_url = acquire.matrix(
                str(source["series_id"]),
                purpose="defining",
                release_id=int(source["release_id"]),
                observation_start=as_date(data["observation_start"]),
                observation_end=observation_end,
                vintage_start=max(
                    as_date(data["vintage_start"]),
                    as_date(source.get("vintage_start", data["vintage_start"])),
                ),
                vintage_end=as_date(source.get("vintage_end", acquire.cutoff)),
            )
            records = extract_first_release_features(
                matrix,
                series_id=str(source["series_id"]),
                component=component,
                transform=str(declaration["transform"]),
                max_release_lag_days=int(data["max_release_lag_days"]),
                archive_start_latest_only=bool(data["archive_start_latest_only"]),
            )
            acquire.records[-1]["extraction_diagnostics"] = records.attrs.get(
                "extraction_diagnostics", {}
            )
            if "active_start" in source:
                records = records.loc[
                    records["reference_month"].ge(pd.Timestamp(source["active_start"]))
                ]
            if "active_end" in source:
                records = records.loc[
                    records["reference_month"].le(pd.Timestamp(source["active_end"]))
                ]
            records = records.copy()
            records["source_url"] = source_url
            parts.append(records)
    components = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["reference_month", "component", "series_id"])
        .reset_index(drop=True)
    )
    scores, gaps = _score_frames(components, config, acquire.cutoff)
    audit_path = acquire.output / "missing_component_audit.csv"
    write_csv(gaps, audit_path)
    if gaps["status"].eq("unexpected_historical_gap").any():
        raise ValueError(f"unexpected historical component gaps; inspect {audit_path}")
    return scores, components


def _build_mapping(
    acquire: _Acquisition,
    scores: pd.DataFrame,
    components: pd.DataFrame,
    score_config: Mapping[str, Any],
    map_config: Mapping[str, Any],
) -> pd.DataFrame:
    settings = map_config["mapping"]
    horizons = tuple(int(value) for value in settings["revision_horizons_months"])
    complete = scores.index[scores["score_available_at"].notna()]
    scales = first_release_standardization_scales(
        components,
        min_history=int(score_config["features"]["min_history_months"]),
        ddof=int(score_config["features"]["standard_deviation_ddof"]),
    )
    requested = required_exact_vintages(
        components,
        reference_months=complete,
        horizons=horizons,
        knowledge_cutoff=pd.Timestamp(acquire.cutoff),
    )
    matrices: dict[str, pd.DataFrame] = {}
    for series_id, vintages in sorted(requested.items()):
        dates = tuple(value.date() for value in vintages)
        _LOGGER.info("Acquiring %s at %d fixed revision vintages", series_id, len(dates))
        acquired = download_or_load_exact_vintage_matrix(
            client=acquire.client,
            series_id=series_id,
            raw_dir=acquire.output / "raw" / "fred" / "revisions",
            observation_start=as_date(score_config["data"]["observation_start"]),
            observation_end=complete.max().date(),
            vintage_dates=dates,
            refresh=acquire.refresh,
        )
        matrices[series_id] = _checked_matrix(
            acquired,
            series_id=series_id,
            cutoff=acquire.cutoff,
            observation_end=complete.max().date(),
            exact_vintages=dates,
        )
        acquire.records.append(
            {
                **_declaration(acquire.root, acquired.path),
                "series_id": series_id,
                "purpose": "fixed_horizon_revisions",
                "provider": acquired.provider_id,
                "source_url": acquired.source_url,
                "cache_origin": acquired.cache_origin,
                "vintage_count": len(dates),
                "first_vintage": min(dates),
                "latest_vintage": max(dates),
            }
        )
    errors = component_revision_errors(
        components,
        scales=scales,
        exact_vintage_matrices=matrices,
        reference_months=complete,
        horizons=horizons,
        knowledge_cutoff=pd.Timestamp(acquire.cutoff),
    )
    write_csv(errors, acquire.output / "component_revision_audit.csv")
    unexpected = ~errors["revision_status"].isin({"available", "horizon_not_mature"})
    if unexpected.any():
        raise ValueError("incomplete mature revision inputs; inspect component_revision_audit.csv")
    axis_errors = aggregate_axis_revision_errors(
        errors, reference_months=complete, horizons=horizons
    )
    mapping = causal_quadrant_mapping_history(
        scores,
        component_disagreement_history(scores),
        axis_errors,
        horizons=horizons,
        baseline_horizon=int(settings["baseline_revision_horizon_months"]),
        minimum_disagreement_months=int(settings["disagreement"]["minimum_complete_months"]),
        minimum_revision_months=int(settings["revision"]["minimum_complete_months"]),
    )
    available = mapping.loc[
        mapping["revision_horizon_months"].eq(12) & mapping["mapping_status"].eq("available")
    ]
    if available.empty or available["reference_month"].max() != complete.max():
        raise ValueError("latest completed score has no available 12-month probability map")
    write_csv(axis_errors, acquire.output / "axis_revision_audit.csv")
    return mapping


def _monthly_events(
    matrix: pd.DataFrame,
    *,
    series_id: str,
    block: str,
    feature: str,
    source_url: str,
    minimum_history: int,
    maximum_lag: int,
    archive_policy: bool,
    ddof: int,
) -> pd.DataFrame:
    records = extract_first_release_features(
        matrix,
        series_id=series_id,
        component=block,
        transform="log_difference",
        max_release_lag_days=maximum_lag,
        archive_start_latest_only=archive_policy,
    )
    return build_monthly_release_events(
        records,
        release_block=block,
        feature_name=feature,
        frequency="monthly",
        provider_id="fred_api",
        source_url=source_url,
        min_standardization_history=minimum_history,
        ddof=ddof,
    )


def _build_events(acquire: _Acquisition, configs: Mapping[str, Any]) -> pd.DataFrame:
    m01, evidence = configs["m01_evidence"], configs["m02_evidence"]
    old_data, data = m01["data"], evidence["data"]
    _LOGGER.info("Acquiring ICSA first releases through %s", acquire.cutoff)
    claims = acquire.client.list_first_release_observations(
        "ICSA",
        release_id=int(m01["blocks"]["weekly_claims"]["release_id"]),
        observation_start=as_date(old_data["observation_start"]),
        observation_end=acquire.cutoff,
        vintage_start=as_date(old_data["vintage_start"]),
        vintage_end=acquire.cutoff,
        chunk_cache_dir=acquire.output / "raw" / "fred" / "claims",
        refresh_cache=acquire.refresh,
    )
    if claims.provider_id != "fred_api" or not claims.observations:
        raise ValueError("authenticated first-release ICSA observations are unavailable")
    claim_rows = pd.DataFrame([asdict(item) for item in claims.observations])
    _assert_cutoff(claim_rows, ("release_date", "reference_date"), acquire.cutoff)
    claims_path = acquire.output / "raw" / "fred" / "icsa_first_releases.csv"
    write_csv(claim_rows, claims_path)
    acquire.records.append(
        {
            **_declaration(acquire.root, claims_path),
            "series_id": "ICSA",
            "purpose": "evidence",
            "provider": claims.provider_id,
            "source_url": claims.source_url,
            "observation_start": old_data["observation_start"],
            "observation_end": acquire.cutoff,
            "vintage_start": old_data["vintage_start"],
            "vintage_end": acquire.cutoff,
            "latest_release": max(item.release_date for item in claims.observations),
        }
    )
    claims_events = build_claims_release_events(
        claims.observations,
        release_block="weekly_labor_stress",
        feature_name="initial_claims_innovation",
        series_id="ICSA",
        provider_id="fred_api",
        source_url=claims.source_url,
        min_ar_history=int(m01["features"]["claims_min_ar_history"]),
        min_standardization_history=int(m01["features"]["claims_min_standardization_history"]),
        ddof=int(m01["features"]["standard_deviation_ddof"]),
        max_release_lag_days=int(old_data["weekly_max_release_lag_days"]),
        archive_start_latest_only=bool(old_data["archive_start_latest_only"]),
    )
    # Match the original upstream publication filter after standardization.
    claims_events = claims_events.loc[
        claims_events["reference_date"].ge(pd.Timestamp(old_data["feature_start"]))
    ]

    business_parts: list[pd.DataFrame] = []
    for series_id, feature, source, source_data, minimum_history in (
        (
            "NEWORDER",
            "core_capital_goods_orders_log_change",
            {},
            old_data,
            int(m01["features"]["monthly_min_standardization_history"]),
        ),
        (
            "ANXAVS",
            "core_capital_goods_shipments_log_change",
            evidence["blocks"]["business_investment"]["new_sources"][0],
            data,
            int(
                evidence["blocks"]["business_investment"]["new_sources"][0][
                    "min_standardization_history"
                ]
            ),
        ),
    ):
        matrix, url = acquire.matrix(
            series_id,
            purpose="business",
            release_id=95,
            observation_start=as_date(source_data["observation_start"]),
            observation_end=acquire.cutoff,
            vintage_start=max(
                as_date(source_data["vintage_start"]),
                as_date(source.get("vintage_start", source_data["vintage_start"])),
            ),
        )
        part = _monthly_events(
            matrix,
            series_id=series_id,
            block="business_investment",
            feature=feature,
            source_url=url,
            minimum_history=minimum_history,
            maximum_lag=int(source_data["monthly_max_release_lag_days"]),
            archive_policy=bool(source_data["archive_start_latest_only"]),
            ddof=int(evidence["features"]["standard_deviation_ddof"]),
        )
        business_parts.append(
            part.loc[part["reference_date"].ge(pd.Timestamp(data["feature_start"]))]
        )
    business, business_audit = build_business_investment_revision_events(
        pd.concat(business_parts, ignore_index=True)
    )
    write_json(asdict(business_audit), acquire.output / "business_pair_audit.json")

    retail = configs["retail"]["retail_sensitivities"]["real_decomposition"]
    retail_matrices: dict[str, pd.DataFrame] = {}
    urls: list[str] = []
    for series_id in ("RSAFS", "RRSFS"):
        retail_matrices[series_id], url = acquire.matrix(
            series_id,
            purpose="retail",
            release_id=int(retail["release_id"]),
            observation_start=as_date(retail["observation_start"]),
            observation_end=acquire.cutoff,
            vintage_start=as_date(retail["vintage_start"]),
        )
        urls.append(url)
    decomposition = build_real_retail_decomposition_matrices(
        retail_matrices["RSAFS"], retail_matrices["RRSFS"]
    )
    retail_parts = [
        _monthly_events(
            matrix,
            series_id=series_id,
            block="consumer_demand_real_decomposition",
            feature=feature,
            source_url=" | ".join(sorted(set(urls))),
            minimum_history=int(retail["minimum_standardization_history"]),
            maximum_lag=int(retail["maximum_release_lag_days"]),
            archive_policy=bool(retail["archive_start_latest_only"]),
            ddof=1,
        )
        for matrix, series_id, feature in (
            (decomposition.real_levels, "RRSFS", "real_retail_and_food_services_log_change"),
            (
                decomposition.implicit_price_levels,
                "RSAFS_DIV_RRSFS",
                "implicit_retail_price_log_change",
            ),
        )
    ]
    events = pd.concat([claims_events, business, *retail_parts], ignore_index=True, sort=False)
    events = events.sort_values(
        ["release_date", "event_group_id", "reference_date", "feature_name"], kind="stable"
    ).reset_index(drop=True)
    _assert_cutoff(events, ("release_date", "reference_date", "reference_month"), acquire.cutoff)
    validate_event_table(events.reindex(columns=EVENT_TABLE_COLUMNS))
    for block in (
        "weekly_labor_stress",
        "business_investment_activity_pipeline",
        "consumer_demand_real_decomposition",
    ):
        if not (events["release_block"].eq(block) & events["feature_status"].eq("available")).any():
            raise ValueError(f"promoted evidence block has no usable history: {block}")
    return events


def build_live_inputs(
    *,
    project_root: Path,
    output_dir: Path,
    as_of: date,
    environ: Mapping[str, str] | None = None,
    refresh: bool = False,
) -> dict[str, Path]:
    """Build current first-release inputs without modifying published history.

    ``as_of`` is the inclusive macro information cutoff, normally the Sunday
    before a Monday signal. Returned CSV schemas match the historical score,
    mapping, and consolidated-event inputs. Parse their date columns when read.
    A successful manifest is written last; a failed acquisition may leave only
    reusable raw caches and explicit diagnostic artifacts.
    """

    # This selection validates the key and never performs a download or write.
    selection = select_vintage_provider("fred", environ=environ)
    if selection.selected != "fred_api":
        raise ValueError("operational Model 02 requires the authenticated FRED provider")
    cutoff = as_date(as_of)
    if cutoff > datetime.now(UTC).date():
        raise ValueError("as_of cannot be a future date")
    root = project_root.resolve()
    output = _output_directory(root, output_dir)
    configs: dict[str, Any] = {}
    config_sources: list[dict[str, Any]] = []
    for name, relative in CONFIG_FILES.items():
        path = root / relative
        if name == "scores":
            configs[name], _ = _score_config(path)
        elif name == "mapping":
            configs[name], _ = _mapping_config(path)
        else:
            configs[name] = yaml.safe_load(path.read_bytes())
        config_sources.append(_declaration(root, path))
    if cutoff < as_date(configs["scores"]["data"]["reference_start"]):
        raise ValueError("as_of precedes the Model 02 publication history")
    acquire = _Acquisition(root, output, selection.client, cutoff, refresh)
    scores, components = _build_scores(acquire, configs["scores"])
    mapping = _build_mapping(acquire, scores, components, configs["scores"], configs["mapping"])
    events = _build_events(acquire, configs)
    paths = {
        "scores": output / "composite_scores.csv",
        "components": output / "first_release_components_long.csv",
        "mapping": output / "quadrant_probabilities.csv",
        "events": output / "release_events.csv",
        "manifest": output / "live_inputs_manifest.json",
    }
    for key, frame in (
        ("scores", scores.reset_index()),
        ("components", components),
        ("mapping", mapping),
        ("events", events),
    ):
        write_csv(frame, paths[key])
    gaps = pd.read_csv(output / "missing_component_audit.csv")
    completed = scores.loc[scores["score_available_at"].notna()]
    event_coverage = []
    for (block, feature), group in events.groupby(["release_block", "feature_name"], sort=True):
        usable = group.loc[group["feature_status"].eq("available")]
        event_coverage.append(
            {
                "release_block": block,
                "feature_name": feature,
                "rows": len(group),
                "available_rows": len(usable),
                "latest_release": group["release_date"].max(),
                "latest_available_release": usable["release_date"].max(),
                "latest_reference_month": group["reference_month"].max(),
            }
        )
    implementation_paths = [
        "src/regime_allocation/data/m02_live_inputs.py",
        "src/regime_allocation/data/first_release.py",
        "src/regime_allocation/data/dataset_acquisition.py",
        "src/regime_allocation/data/providers/fred_api.py",
        "src/regime_allocation/models/m02_soft_composite/scores.py",
        "src/regime_allocation/models/m02_soft_composite/revisions.py",
        "src/regime_allocation/models/m02_soft_composite/probability_map.py",
        "src/regime_allocation/features/release_evidence.py",
        "src/regime_allocation/features/m02_retail_sensitivities.py",
        "src/regime_allocation/features/m02_feature_revision.py",
    ]
    write_json(
        {
            "schema_version": 1,
            "model_id": "m02_soft_composite",
            "stage_id": "operational_live_inputs_v1",
            "provider_selected": selection.selected,
            "as_of": cutoff,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "historical_publication_reproduction": False,
            "source_configurations": config_sources,
            "implementation_files": [_declaration(root, root / p) for p in implementation_paths],
            "acquired_files": acquire.records,
            "generated_files": [
                _declaration(root, p) for key, p in paths.items() if key != "manifest"
            ],
            "audit_files": [
                _declaration(root, p)
                for p in (
                    output / "missing_component_audit.csv",
                    output / "component_revision_audit.csv",
                    output / "axis_revision_audit.csv",
                    output / "business_pair_audit.json",
                )
            ],
            "coverage": {
                "reference_start": scores.index.min(),
                "component_reference_end": scores.index.max(),
                "complete_score_months": len(completed),
                "latest_complete_score_month": completed.index.max(),
                "latest_complete_score_available_at": completed["score_available_at"].max(),
                "latest_component_release": components["release_date"].max(),
                "latest_evidence_release": events["release_date"].max(),
                "latest_release_used": max(
                    components["release_date"].max(), events["release_date"].max()
                ),
                "missing_component_months": gaps.to_dict(orient="records"),
                "evidence": event_coverage,
                "available_mapping_months_by_horizon": {
                    str(horizon): int(group["mapping_status"].eq("available").sum())
                    for horizon, group in mapping.groupby("revision_horizon_months")
                },
            },
            "model_contract": {
                "score_definition": configs["scores"]["features"],
                "baseline_revision_horizon_months": 12,
                "information_cutoff": "source_vintage_and_release_date_on_or_before_as_of",
                "missing_value_policy": "retain_without_imputation; unexpected_historical_gaps_stop",
                "source_history_policy": "fresh_authenticated_first_release_acquisition",
            },
        },
        paths["manifest"],
    )
    return paths


__all__ = ["build_live_inputs"]
