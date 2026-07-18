"""Verify provider-neutral cache reuse and preservation without network access.

Synthetic normalized vintage matrices stand in for ALFRED and authenticated FRED
downloads. The tests call the dataset acquisition helper and assert that a
compatible cross-provider cache may be reused, while an explicit refresh writes
provider-specific bytes separately and never overwrites the other provider's
artifact. All paths are temporary and no credentials or production cache files
are read.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from regime_allocation.cli.build_m01_dataset import _download_or_load
from regime_allocation.data.providers.alfred_web import AlfredWebDownloadClient
from regime_allocation.data.providers.fred_api import FredApiDownloadClient
from regime_allocation.data.providers.vintage_matrix import (
    DownloadedVintageMatrix,
    encode_vintage_matrix,
)


def _matrix_payload(series_id: str, vintage: date, value: float) -> bytes:
    """Encode a one-cell normalized vintage matrix for cache tests."""
    frame = pd.DataFrame(
        {f"{series_id}_{vintage:%Y%m%d}": [value]},
        index=pd.DatetimeIndex(["2024-01-01"]),
    )
    return encode_vintage_matrix(frame, series_id)


def _arguments(raw_dir: Path) -> dict[str, object]:
    """Return common acquisition arguments rooted in a temporary cache directory."""
    return {
        "series_id": "PAYEMS",
        "release_id": 50,
        "raw_dir": raw_dir,
        "observation_start": date(2024, 1, 1),
        "observation_end": date(2024, 1, 1),
        "vintage_start": date(2024, 2, 2),
        "vintage_end": date(2024, 2, 2),
    }


def _artifact(
    *, provider_id: str, source_url: str, content: bytes
) -> DownloadedVintageMatrix:
    """Wrap normalized bytes in the provider-neutral acquisition record."""
    return DownloadedVintageMatrix(
        series_id="PAYEMS",
        selected_vintage_dates=(date(2024, 2, 2),),
        content=content,
        source_url=source_url,
        provider_id=provider_id,
    )


def test_fred_selection_reuses_an_existing_alfred_normalized_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_dir = tmp_path / "data" / "raw" / "alfred" / "model_01"
    fred_dir = tmp_path / "data" / "raw" / "fred_api" / "model_01"
    old_payload = _matrix_payload("PAYEMS", date(2024, 2, 2), 100.0)

    alfred = AlfredWebDownloadClient()
    monkeypatch.setattr(
        alfred,
        "download_level_matrix",
        lambda *args, **kwargs: _artifact(
            provider_id="alfred_web",
            source_url=alfred.series_page_url("PAYEMS"),
            content=old_payload,
        ),
    )
    original = _download_or_load(
        client=alfred,
        refresh=True,
        **_arguments(legacy_dir),
    )

    fred = FredApiDownloadClient("a" * 32)
    calls = 0

    def unexpected_download(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("compatible cache should prevent a provider call")

    monkeypatch.setattr(fred, "download_level_matrix", unexpected_download)
    reused = _download_or_load(
        client=fred,
        compatible_cache_dirs=((legacy_dir, "alfred_web"),),
        refresh=False,
        **_arguments(fred_dir),
    )

    assert calls == 0
    assert reused.path == original.path
    assert reused.content == old_payload
    assert reused.provider_id == "alfred_web"
    assert reused.cache_origin == "existing_normalized_cache"
    assert not fred_dir.exists()


def test_fred_refresh_writes_separately_and_leaves_alfred_bytes_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_dir = tmp_path / "data" / "raw" / "alfred" / "model_01"
    fred_dir = tmp_path / "data" / "raw" / "fred_api" / "model_01"
    old_payload = _matrix_payload("PAYEMS", date(2024, 2, 2), 100.0)
    new_payload = _matrix_payload("PAYEMS", date(2024, 2, 2), 101.0)

    alfred = AlfredWebDownloadClient()
    monkeypatch.setattr(
        alfred,
        "download_level_matrix",
        lambda *args, **kwargs: _artifact(
            provider_id="alfred_web",
            source_url=alfred.series_page_url("PAYEMS"),
            content=old_payload,
        ),
    )
    original = _download_or_load(
        client=alfred,
        refresh=True,
        **_arguments(legacy_dir),
    )
    original_bytes = original.path.read_bytes()

    fred = FredApiDownloadClient("a" * 32)
    monkeypatch.setattr(
        fred,
        "download_level_matrix",
        lambda *args, **kwargs: _artifact(
            provider_id="fred_api",
            source_url=fred.series_page_url("PAYEMS"),
            content=new_payload,
        ),
    )
    refreshed = _download_or_load(
        client=fred,
        compatible_cache_dirs=((legacy_dir, "alfred_web"),),
        refresh=True,
        **_arguments(fred_dir),
    )

    assert refreshed.provider_id == "fred_api"
    assert refreshed.cache_origin == "downloaded"
    assert refreshed.path.parent == fred_dir
    assert refreshed.content == new_payload
    assert original.path.read_bytes() == original_bytes == old_payload
    assert refreshed.path != original.path
    assert refreshed.path.with_name(
        refreshed.path.name + ".metadata.json"
    ).exists()
