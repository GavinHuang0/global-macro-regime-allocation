"""No-network tests for the authenticated FRED API provider and selector."""

from __future__ import annotations

from datetime import date
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from regime_allocation.data.providers import select_vintage_provider
from regime_allocation.data.providers.alfred_web import AlfredWebDownloadClient
from regime_allocation.data.providers.fred_api import (
    FredApiConfigurationError,
    FredApiDownloadClient,
    FredApiError,
)
from regime_allocation.data.providers.vintage_matrix import load_vintage_matrix


VALID_KEY = "a" * 32


def _official_vintage_zip(
    series_id: str,
    vintages: tuple[date, ...],
    *,
    value_shift: float = 0.0,
) -> bytes:
    """Build the one-CSV ZIP returned by FRED output type 2."""

    columns = [
        f"{series_id}_{vintage:%Y%m%d}" for vintage in vintages
    ]
    frame = pd.DataFrame(
        [
            [date(2023, 12, 1), *[100.0 + value_shift + i for i in range(len(columns))]],
            [date(2024, 1, 1), *[101.0 + value_shift + i for i in range(len(columns))]],
        ],
        columns=["observation_date", *columns],
    )
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("observations.csv", frame.to_csv(index=False))
        archive.writestr(
            "README.txt",
            f"simulated request metadata api_key={VALID_KEY}",
        )
    return buffer.getvalue()


def test_provider_selection_prefers_fred_only_when_a_key_is_available() -> None:
    with_key = select_vintage_provider(
        "auto", environ={"FRED_API_KEY": VALID_KEY}
    )
    without_key = select_vintage_provider("auto", environ={})
    forced_keyless = select_vintage_provider(
        "alfred", environ={"FRED_API_KEY": VALID_KEY}
    )

    assert with_key.requested == "auto"
    assert with_key.selected == "fred_api"
    assert isinstance(with_key.client, FredApiDownloadClient)
    assert without_key.selected == "alfred_web"
    assert isinstance(without_key.client, AlfredWebDownloadClient)
    assert forced_keyless.requested == "alfred"
    assert forced_keyless.selected == "alfred_web"


def test_explicit_fred_requires_a_key_without_disclosing_configuration() -> None:
    with pytest.raises(FredApiConfigurationError) as missing:
        select_vintage_provider("fred", environ={})
    assert str(missing.value) == (
        "FRED_API_KEY is required when the FRED provider is selected"
    )

    invalid_secret = "DO-NOT-PRINT-THIS-INVALID-SECRET"
    with pytest.raises(FredApiConfigurationError) as invalid:
        select_vintage_provider(
            "fred", environ={"FRED_API_KEY": invalid_secret}
        )
    assert invalid_secret not in str(invalid.value)
    assert invalid_secret not in repr(invalid.value)
    assert "32-character lowercase alphanumeric" in str(invalid.value)


def test_client_repr_and_public_source_url_never_contain_the_key() -> None:
    client = FredApiDownloadClient(
        VALID_KEY,
        timeout_seconds=17,
        max_vintages_per_request=4,
        max_attempts=2,
    )

    assert VALID_KEY not in repr(client)
    assert "timeout_seconds=17" in repr(client)
    assert client.series_page_url("PAYEMS") == (
        "https://fred.stlouisfed.org/series/PAYEMS"
    )
    assert VALID_KEY not in client.series_page_url("PAYEMS")


def test_output_type_four_initial_release_dates_are_paginated_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FredApiDownloadClient(VALID_KEY)
    calls: list[dict[str, object]] = []
    pages = {
        0: {
            "count": 3,
            "offset": 0,
            "output_type": 4,
            "file_type": "json",
            "units": "lin",
            "observations": [
                {"realtime_start": "2024-03-08", "date": "2024-02-01"},
                {"realtime_start": "2024-02-02", "date": "2024-01-01"},
            ],
        },
        2: {
            "count": 3,
            "offset": 2,
            "output_type": 4,
            "file_type": "json",
            "units": "lin",
            "observations": [
                {"realtime_start": "2024-04-05", "date": "2024-03-01"},
            ],
        },
    }

    def fake_request_json(
        endpoint: str, params: dict[str, object]
    ) -> dict[str, object]:
        assert endpoint == "series/observations"
        assert "api_key" not in params
        assert params["output_type"] == 4
        assert params["file_type"] == "json"
        calls.append(dict(params))
        return pages[int(params["offset"])]

    monkeypatch.setattr(client, "_request_json", fake_request_json)
    arguments = {
        "observation_start": date(2024, 1, 1),
        "observation_end": date(2024, 3, 1),
        "vintage_start": date(2024, 1, 1),
        "vintage_end": date(2024, 4, 30),
    }

    expected = (
        date(2024, 2, 2),
        date(2024, 3, 8),
        date(2024, 4, 5),
    )
    assert client.list_initial_release_dates("PAYEMS", **arguments) == expected
    assert client.list_initial_release_dates("PAYEMS", **arguments) == expected
    assert [call["offset"] for call in calls] == [0, 2]
    assert all(call["limit"] == 100000 for call in calls)


def test_output_type_two_zip_is_normalized_and_chunk_cache_is_reused_and_refreshed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vintages = (date(2024, 2, 2), date(2024, 3, 8))
    client = FredApiDownloadClient(
        VALID_KEY,
        max_vintages_per_request=1,
        request_pause_seconds=0,
    )
    initial_date_calls = 0
    matrix_calls: list[tuple[date, ...]] = []

    def fake_initial_dates(*_: object, **__: object) -> tuple[date, ...]:
        nonlocal initial_date_calls
        initial_date_calls += 1
        return vintages

    def fake_request_bytes(
        endpoint: str, params: dict[str, object]
    ) -> bytes:
        assert endpoint == "series/observations"
        assert "api_key" not in params
        assert params["output_type"] == 2
        assert params["file_type"] == "csv"
        requested = tuple(
            date.fromisoformat(item)
            for item in str(params["vintage_dates"]).split(",")
        )
        matrix_calls.append(requested)
        return _official_vintage_zip("PAYEMS", requested)

    monkeypatch.setattr(client, "list_initial_release_dates", fake_initial_dates)
    monkeypatch.setattr(client, "_request_bytes", fake_request_bytes)
    arguments = {
        "release_id": 50,
        "observation_start": date(2023, 12, 1),
        "observation_end": date(2024, 1, 1),
        "vintage_start": vintages[0],
        "vintage_end": vintages[-1],
        "chunk_cache_dir": tmp_path,
    }

    first = client.download_level_matrix("PAYEMS", **arguments)
    second = client.download_level_matrix("PAYEMS", **arguments)
    refreshed = client.download_level_matrix(
        "PAYEMS", **arguments, refresh_cache=True
    )

    assert first.provider_id == "fred_api"
    assert first.source_url == "https://fred.stlouisfed.org/series/PAYEMS"
    assert first.selected_vintage_dates == vintages
    assert second.content == first.content == refreshed.content
    assert matrix_calls == [(vintages[0],), (vintages[1],)] * 2
    assert initial_date_calls == 2

    normalized = load_vintage_matrix(first.content, "PAYEMS")
    assert list(normalized.columns) == [
        "PAYEMS_20240202",
        "PAYEMS_20240308",
    ]
    assert list(normalized.index) == [
        pd.Timestamp("2023-12-01"),
        pd.Timestamp("2024-01-01"),
    ]

    cache_files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    assert len(list(tmp_path.glob("*.zip"))) == 2
    assert len(list(tmp_path.glob("initial_release_dates_*.json"))) == 1
    assert cache_files
    assert all(VALID_KEY not in str(path) for path in cache_files)
    for path in cache_files:
        assert VALID_KEY.encode("utf-8") not in path.read_bytes()
    date_cache = json.loads(
        next(tmp_path.glob("initial_release_dates_*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert date_cache["provider_id"] == "fred_api"
    assert "api_key" not in date_cache["query"]


def test_transport_failure_suppresses_the_credential_bearing_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FredApiDownloadClient(VALID_KEY, max_attempts=1)

    def fail(request: object) -> bytes:
        full_url = request.full_url  # type: ignore[attr-defined]
        assert f"api_key={VALID_KEY}" in full_url
        raise HTTPError(full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(client, "_open_once", fail)

    with pytest.raises(FredApiError) as failure:
        client._request_bytes(
            "series/observations",
            {"series_id": "PAYEMS", "file_type": "json"},
        )

    assert str(failure.value) == (
        "FRED API request failed for endpoint series/observations"
    )
    assert VALID_KEY not in str(failure.value)
    assert VALID_KEY not in repr(failure.value)
    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None
    assert failure.value.__suppress_context__ is True


def test_cache_identity_refuses_credential_parameters() -> None:
    client = FredApiDownloadClient(VALID_KEY)

    identity = client._safe_query_identity(
        "series/observations", {"series_id": "PAYEMS", "output_type": 2}
    )

    assert len(identity) == 16
    assert VALID_KEY not in identity
    with pytest.raises(ValueError, match="must not receive credential"):
        client._safe_query_identity(
            "series/observations",
            {"series_id": "PAYEMS", "api_key": VALID_KEY},
        )
