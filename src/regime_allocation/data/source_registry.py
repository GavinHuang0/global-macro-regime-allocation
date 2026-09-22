"""Load the explicit, versioned M03 source registry without inferring release times."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from regime_allocation.data.source_snapshots import validate_source_url


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently replacing a source declaration."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError("registry mappings require unique string keys")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field} contains control characters")
    return value


def _date(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if isinstance(value, datetime):
        raise ValueError(f"{field} requires a calendar date, not a timestamp")  # noqa: TRY004
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            pass
    raise ValueError(f"{field} requires an ISO calendar date")


def load_source_registry(path: Path | str) -> dict:
    """Return validated definitions with date fields normalized to ISO strings.

    A configured acquisition boundary is not independently verified archive
    coverage. A verified boundary requires a separate dated value and evidence
    URL; no release time is fabricated from a vintage date.
    """

    registry = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(registry, dict):
        raise ValueError("source registry must be a mapping")  # noqa: TRY004
    if type(registry.get("schema_version")) is not int or registry["schema_version"] != 1:
        raise ValueError("unsupported source registry schema_version")
    for field in ("registry_id", "provider_id", "source_definition_version"):
        _text(registry.get(field), field)
    defaults = registry.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("source registry defaults must be a mapping")  # noqa: TRY004
    defaults["observation_start"] = _date(defaults.get("observation_start"), "observation_start")
    lag = defaults.get("max_release_lag_days")
    if type(lag) is not int or lag < 0:
        raise ValueError("max_release_lag_days must be a nonnegative integer")
    if type(defaults.get("archive_start_latest_only")) is not bool:
        raise ValueError("archive_start_latest_only must be boolean")
    if "reference_publication_vintage_end" in registry:
        registry["reference_publication_vintage_end"] = _date(
            registry["reference_publication_vintage_end"], "reference_publication_vintage_end"
        )
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source registry requires a nonempty sources list")
    required = {
        "series_id",
        "component",
        "transform",
        "release_id",
        "role",
        "definition",
        "frequency",
        "source_url",
        "source_definition_version",
        "source_config",
        "observation_start",
        "vintage_start",
        "vintage_end",
        "active_start",
        "active_end",
        "vintage_start_status",
        "verified_archive_start",
        "release_timestamp_status",
    }
    identifiers = set()
    for source in sources:
        if not isinstance(source, dict) or not required.issubset(source):
            raise ValueError("source declaration is missing required fields")
        identifier = _text(source["series_id"], "series_id")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", identifier) or identifier in identifiers:
            raise ValueError("series_id must be a unique FRED series identifier")
        identifiers.add(identifier)
        for field in ("component", "definition", "source_definition_version", "source_config"):
            _text(source[field], field)
        if source["source_definition_version"] != registry["source_definition_version"]:
            raise ValueError("source definition version disagrees with registry version")
        if source["role"] not in {"score", "evidence"}:
            raise ValueError("source role must be score or evidence")
        if source["frequency"] not in {"monthly", "weekly"}:
            raise ValueError("unsupported source frequency")
        if source["role"] == "score" and source["frequency"] != "monthly":
            raise ValueError("score sources must be monthly")
        if source["transform"] not in {
            "log_difference",
            "negative_difference",
            "expanding_log_ar1_innovation",
        }:
            raise ValueError("unsupported source transformation")
        if type(source["release_id"]) is not int or source["release_id"] <= 0:
            raise ValueError("release_id must be a positive integer")
        validate_source_url(source["source_url"])
        for field in ("observation_start", "vintage_start"):
            source[field] = _date(source[field], field)
        for field in ("vintage_end", "active_start", "active_end", "verified_archive_start"):
            source[field] = _date(source[field], field, nullable=True)
        if "reference_alfred_policy_vintage_start" in source:
            source["reference_alfred_policy_vintage_start"] = _date(
                source["reference_alfred_policy_vintage_start"],
                "reference_alfred_policy_vintage_start",
            )
        for start, end in (("vintage_start", "vintage_end"), ("active_start", "active_end")):
            if (
                source[start] is not None
                and source[end] is not None
                and source[start] > source[end]
            ):
                raise ValueError(f"{start} cannot follow {end}")
        if source["frequency"] == "monthly":
            for field in ("active_start", "active_end"):
                if source[field] is not None and not source[field].endswith("-01"):
                    raise ValueError("monthly active ranges must use reference-month starts")
        if source["vintage_start_status"] not in {
            "configured_policy_boundary",
            "verified_archive_boundary",
        }:
            raise ValueError("unsupported vintage_start_status")
        verified = source["verified_archive_start"]
        if verified is not None:
            validate_source_url(source.get("archive_start_evidence_url"))
        if (
            source["vintage_start_status"] == "verified_archive_boundary"
            and verified != source["vintage_start"]
        ):
            raise ValueError("verified vintage boundary requires matching archive evidence")
        if source["release_timestamp_status"] != "archive_date_only_not_verified_intraday":
            raise ValueError("registry schema 1 does not establish verified intraday release times")
    return registry
