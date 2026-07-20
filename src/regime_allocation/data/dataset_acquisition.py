"""Shared, provider-neutral acquisition and atomic artifact helpers.

This module contains only mechanical data-access operations used by versioned
model builders.  It deliberately knows nothing about regime definitions,
component transformations, or model outputs.  Authenticated FRED credentials
remain inside the provider object and never enter cache identities, filenames,
metadata, or exceptions emitted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from regime_allocation.data.providers.vintage_matrix import VintageMatrixProvider


def as_date(value: object) -> date:
    """Convert configuration date-like values to :class:`datetime.date`."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def sha256(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest for provenance records."""

    return hashlib.sha256(payload).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace ``path`` with a date-normalized CSV artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Atomically replace ``path`` with stable, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class MatrixAcquisition:
    """One normalized vintage-matrix artifact and non-secret provenance."""

    path: Path
    content: bytes
    provider_id: str
    cache_origin: str
    source_url: str


def provider_raw_dir(legacy_raw_dir: Path, cache_namespace: str) -> Path:
    """Place provider caches in distinct namespaces without moving old data."""

    if cache_namespace == "alfred":
        return legacy_raw_dir
    if legacy_raw_dir.parent.name == "alfred":
        return legacy_raw_dir.parent.parent / cache_namespace / legacy_raw_dir.name
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


def download_or_load_vintage_matrix(
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
) -> MatrixAcquisition:
    """Download or reuse one normalized same-vintage level matrix.

    Cache keys contain only public query parameters.  The normalized matrix
    contract is shared by the authenticated FRED and keyless ALFRED providers.
    """

    query = {
        "series_id": series_id,
        "release_id": release_id,
        "observation_start": observation_start.isoformat(),
        "observation_end": observation_end.isoformat(),
        "vintage_start": vintage_start.isoformat(),
        "vintage_end": vintage_end.isoformat(),
        "normalized_contract": "level_snapshots_by_vintage",
    }
    query_hash = sha256(
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
            candidate = cache_dir / f"{series_id}_{query_hash}_levels_by_vintage.zip"
            if candidate.exists():
                return MatrixAcquisition(
                    path=candidate,
                    content=candidate.read_bytes(),
                    provider_id=_cached_provider_id(candidate, fallback_provider_id),
                    cache_origin="existing_normalized_cache",
                    source_url=client.series_page_url(series_id),
                )

    raw_path = raw_dir / f"{series_id}_{query_hash}_levels_by_vintage.zip"

    # Reuse the normalized contract written by the earlier ALFRED-form adapter.
    legacy_query = dict(query)
    legacy_query.pop("normalized_contract")
    legacy_query.update({"units": "lin", "output_type": 2})
    legacy_hash = sha256(
        json.dumps(legacy_query, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )[:12]
    if not refresh:
        for cache_dir, fallback_provider_id in cache_locations:
            legacy_path = cache_dir / (
                f"{series_id}_{legacy_hash}_levels_by_vintage.zip"
            )
            if legacy_path.exists():
                return MatrixAcquisition(
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
    write_json(
        {
            "schema_version": 1,
            "provider_id": artifact.provider_id,
            "source_url": artifact.source_url,
            "query": query,
            "content_sha256": sha256(artifact.content),
            "selected_vintage_dates": len(artifact.selected_vintage_dates),
        },
        _metadata_path(raw_path),
    )
    return MatrixAcquisition(
        path=raw_path,
        content=artifact.content,
        provider_id=artifact.provider_id,
        cache_origin="downloaded",
        source_url=artifact.source_url,
    )


def download_or_load_exact_vintage_matrix(
    *,
    client: VintageMatrixProvider,
    series_id: str,
    raw_dir: Path,
    observation_start: date,
    observation_end: date,
    vintage_dates: tuple[date, ...],
    refresh: bool,
) -> MatrixAcquisition:
    """Download or reuse snapshots at explicitly requested as-of dates.

    This contract is used for fixed-horizon revision measurements.  Exact
    vintage dates are public query inputs; credentials remain encapsulated by
    the provider and are excluded from cache identities and metadata.
    """

    selected = tuple(sorted(set(vintage_dates)))
    if not selected:
        raise ValueError("vintage_dates cannot be empty")
    if observation_start > observation_end:
        raise ValueError("observation_start cannot follow observation_end")
    serialized_dates = ",".join(item.isoformat() for item in selected)
    query = {
        "series_id": series_id,
        "observation_start": observation_start.isoformat(),
        "observation_end": observation_end.isoformat(),
        "vintage_dates_sha256": sha256(serialized_dates.encode("ascii")),
        "vintage_count": len(selected),
        "first_vintage": selected[0].isoformat(),
        "last_vintage": selected[-1].isoformat(),
        "normalized_contract": "exact_as_of_level_snapshots",
    }
    query_hash = sha256(
        json.dumps(query, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )[:12]
    raw_path = raw_dir / f"{series_id}_{query_hash}_exact_vintages.zip"
    if raw_path.exists() and not refresh:
        return MatrixAcquisition(
            path=raw_path,
            content=raw_path.read_bytes(),
            provider_id=_cached_provider_id(raw_path, client.provider_id),
            cache_origin="existing_normalized_cache",
            source_url=client.series_page_url(series_id),
        )

    artifact = client.download_level_matrix_at_vintages(
        series_id,
        observation_start=observation_start,
        observation_end=observation_end,
        vintage_dates=selected,
        chunk_cache_dir=raw_dir / "provider_chunks" / series_id,
        refresh_cache=refresh,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
    temporary.write_bytes(artifact.content)
    temporary.replace(raw_path)
    write_json(
        {
            "schema_version": 1,
            "provider_id": artifact.provider_id,
            "source_url": artifact.source_url,
            "query": query,
            "content_sha256": sha256(artifact.content),
            "selected_vintage_dates": len(artifact.selected_vintage_dates),
        },
        _metadata_path(raw_path),
    )
    return MatrixAcquisition(
        path=raw_path,
        content=artifact.content,
        provider_id=artifact.provider_id,
        cache_origin="downloaded",
        source_url=artifact.source_url,
    )
