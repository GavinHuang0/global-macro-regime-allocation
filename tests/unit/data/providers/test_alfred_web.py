"""Test the keyless ALFRED vintage adapter with fully mocked HTTP responses.

Synthetic HTML, text calendars, CSV graphs, and ZIP archives exercise vintage
discovery, release-date parsing, matrix normalization, cache compatibility,
chunk boundaries, and reconstruction of earliest first-release observations.
The tests never contact ALFRED and write only temporary cache files. Their purpose
is to ensure provider transport details cannot alter point-in-time semantics.
"""

from __future__ import annotations

from datetime import date
import hashlib
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from zipfile import ZipFile

import pandas as pd
import pytest

from regime_allocation.data.providers.alfred_web import (
    ALFRED_GRAPH_URL,
    AlfredDownloadError,
    AlfredWebDownloadClient,
    load_graph_matrix,
    load_vintage_matrix,
    vintage_date_from_column,
)
from regime_allocation.data.providers.vintage_matrix import (
    DownloadedVintageMatrix,
    encode_vintage_matrix,
)


def _zip_csv(csv_text: str, *, filename: str = "observations.csv") -> bytes:
    """Package CSV text as the in-memory ZIP payload returned by ALFRED."""
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        archive.writestr(filename, csv_text)
    return buffer.getvalue()


def test_list_vintage_dates_parses_only_target_select_and_sorts_unique_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html = b"""
        <select name="unrelated">
          <option value="1999-01-01">ignore me</option>
        </select>
        <select name="form[selected_vintage_dates][]">
          <option value="2020-03-06">March</option>
          <option value="not-a-date">invalid</option>
          <option value="2020-01-10">January</option>
          <option value="2020-03-06">duplicate March</option>
        </select>
        <select name="another-unrelated-select">
          <option value="2020-02-07">also ignore me</option>
        </select>
    """
    client = AlfredWebDownloadClient()
    opened_urls: list[str] = []

    def fake_open(request: object) -> bytes:
        opened_urls.append(request.full_url)  # type: ignore[attr-defined]
        return html

    monkeypatch.setattr(client, "_open", fake_open)

    assert client.list_vintage_dates("PAYEMS") == (
        date(2020, 1, 10),
        date(2020, 3, 6),
    )
    assert opened_urls == [client.series_url("PAYEMS")]


def test_list_release_dates_parses_text_lines_and_sorts_unique_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (
        b"Employment Situation\r\n2020-03-06\r\nnot a date\r\n"
        b"2020-01-10\r\n2020-03-06\r\n"
    )
    client = AlfredWebDownloadClient()
    opened_urls: list[str] = []

    def fake_open(request: object) -> bytes:
        opened_urls.append(request.full_url)  # type: ignore[attr-defined]
        return payload

    monkeypatch.setattr(client, "_open", fake_open)

    expected = (
        date(2020, 1, 10),
        date(2020, 3, 6),
    )
    assert client.list_release_dates(50) == expected
    assert client.list_release_dates(50) == expected
    assert client.list_release_dates(
        50,
        release_start=date(2020, 2, 1),
        release_end=date(2020, 3, 31),
    ) == (date(2020, 3, 6),)
    assert opened_urls == [
        "https://alfred.stlouisfed.org/release/downloaddates?ff=txt&rid=50"
    ]


def test_list_release_dates_rejects_nonpositive_release_id() -> None:
    client = AlfredWebDownloadClient()
    with pytest.raises(ValueError, match="positive"):
        client.list_release_dates(0)


def test_load_vintage_matrix_sorts_rows_and_vintages_and_ignores_other_columns() -> None:
    payload = _zip_csv(
        "\n".join(
            [
                (
                    "observation_date,PAYEMS_20200306,notes,"
                    "PAYEMS_20200110,PAYEMS_20200207"
                ),
                "2020-01-01,102,later revision,.,101",
                "2019-12-01,92,metadata,90,91",
            ]
        )
    )

    actual = load_vintage_matrix(payload, "PAYEMS")

    assert list(actual.columns) == [
        "PAYEMS_20200110",
        "PAYEMS_20200207",
        "PAYEMS_20200306",
    ]
    assert list(actual.index) == [
        pd.Timestamp("2019-12-01"),
        pd.Timestamp("2020-01-01"),
    ]
    assert actual.index.name == "reference_month"
    assert actual.loc[pd.Timestamp("2019-12-01"), "PAYEMS_20200110"] == 90
    assert pd.isna(actual.loc[pd.Timestamp("2020-01-01"), "PAYEMS_20200110"])
    assert "notes" not in actual.columns


def test_load_graph_matrix_preserves_requested_vintage_order() -> None:
    payload = (
        b"observation_date,PAYEMS_20240308,PAYEMS_20240202\n"
        b"2024-01-01,157533,157700\n"
        b"2023-12-01,157304,157347\n"
    )

    actual = load_graph_matrix(
        payload,
        series_id="PAYEMS",
        requested_vintages=(date(2024, 2, 2), date(2024, 3, 8)),
    )

    assert list(actual.columns) == ["PAYEMS_20240202", "PAYEMS_20240308"]
    assert list(actual.index) == [
        pd.Timestamp("2023-12-01"),
        pd.Timestamp("2024-01-01"),
    ]
    assert actual.loc[pd.Timestamp("2024-01-01"), "PAYEMS_20240202"] == 157700


def test_load_graph_matrix_names_a_single_unadorned_series_column() -> None:
    actual = load_graph_matrix(
        b"observation_date,PAYEMS\n2024-01-01,157700\n",
        series_id="PAYEMS",
        requested_vintages=(date(2024, 2, 2),),
    )

    assert list(actual.columns) == ["PAYEMS_20240202"]


def test_load_graph_matrix_rejects_a_vintage_mismatch() -> None:
    with pytest.raises(AlfredDownloadError, match="vintage mismatch"):
        load_graph_matrix(
            b"observation_date,PAYEMS_20240202\n2024-01-01,157700\n",
            series_id="PAYEMS",
            requested_vintages=(date(2024, 2, 2), date(2024, 3, 8)),
        )


def test_download_level_matrix_reuses_and_refreshes_chunk_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vintage = date(2024, 2, 2)
    client = AlfredWebDownloadClient(
        max_vintages_per_request=1,
        request_pause_seconds=0,
    )
    monkeypatch.setattr(client, "list_release_dates", lambda _: (vintage,))
    calls = 0

    def fake_open(_: object) -> bytes:
        nonlocal calls
        calls += 1
        return b"observation_date,PAYEMS\n2024-01-01,157700\n"

    monkeypatch.setattr(client, "_open", fake_open)
    arguments = {
        "release_id": 50,
        "observation_start": date(2024, 1, 1),
        "observation_end": date(2024, 1, 1),
        "vintage_start": vintage,
        "vintage_end": vintage,
        "chunk_cache_dir": tmp_path,
    }

    first = client.download_level_matrix("PAYEMS", **arguments)
    second = client.download_level_matrix("PAYEMS", **arguments)
    refreshed = client.download_level_matrix(
        "PAYEMS", **arguments, refresh_cache=True
    )

    assert calls == 2
    assert first.content == second.content == refreshed.content
    assert len(list(tmp_path.glob("*.csv"))) == 1


def test_historical_chunk_ends_observations_at_its_last_vintage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vintages = (date(2009, 5, 28), date(2009, 6, 4))
    client = AlfredWebDownloadClient(
        max_vintages_per_request=2,
        request_pause_seconds=0,
    )
    monkeypatch.setattr(client, "list_release_dates", lambda _: vintages)

    def fake_open(request: object) -> bytes:
        query = parse_qs(urlparse(request.full_url).query)  # type: ignore[attr-defined]
        assert query["coed"] == ["2009-06-04,2009-06-04"]
        return (
            b"observation_date,PAYEMS_20090528,PAYEMS_20090604\n"
            b"2009-05-01,100,101\n"
        )

    monkeypatch.setattr(client, "_open", fake_open)
    actual = client.download_level_matrix(
        "PAYEMS",
        release_id=50,
        observation_start=date(2000, 1, 1),
        observation_end=date(2026, 7, 16),
        vintage_start=vintages[0],
        vintage_end=vintages[-1],
    )

    assert actual.selected_vintage_dates == vintages


def test_bounded_request_reuses_compatible_unbounded_chunk_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vintage = date(2024, 2, 2)
    observation_start = date(2024, 1, 1)
    observation_end = date(2026, 7, 16)
    client = AlfredWebDownloadClient(
        max_vintages_per_request=1,
        request_pause_seconds=0,
    )
    monkeypatch.setattr(client, "list_release_dates", lambda _: (vintage,))
    legacy_query = {
        "id": "PAYEMS",
        "vintage_date": vintage.isoformat(),
        "cosd": observation_start.isoformat(),
        "coed": observation_end.isoformat(),
    }
    legacy_url = f"{ALFRED_GRAPH_URL}?{urlencode(legacy_query, safe=',')}"
    legacy_key = hashlib.sha256(legacy_url.encode("ascii")).hexdigest()[:16]
    legacy_path = tmp_path / (
        f"{vintage.isoformat()}_{vintage.isoformat()}_{legacy_key}.csv"
    )
    legacy_path.write_bytes(
        b"observation_date,PAYEMS\n2024-01-01,157700\n"
    )

    def fail_open(_: object) -> bytes:
        raise AssertionError("compatible cache should avoid a network request")

    monkeypatch.setattr(client, "_open", fail_open)
    artifact = client.download_level_matrix(
        "PAYEMS",
        release_id=50,
        observation_start=observation_start,
        observation_end=observation_end,
        vintage_start=vintage,
        vintage_end=vintage,
        chunk_cache_dir=tmp_path,
    )

    assert artifact.selected_vintage_dates == (vintage,)


def test_first_release_observations_are_reconstructed_from_earliest_vintage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "ICSA_20240111": [202.0, float("nan")],
            "ICSA_20240118": [204.0, 211.0],
        },
        index=pd.to_datetime(["2024-01-06", "2024-01-13"]),
    )
    artifact = DownloadedVintageMatrix(
        series_id="ICSA",
        selected_vintage_dates=(date(2024, 1, 11), date(2024, 1, 18)),
        content=encode_vintage_matrix(frame, "ICSA"),
        source_url="https://fred.stlouisfed.org/series/ICSA",
        provider_id="alfred_web",
    )
    client = AlfredWebDownloadClient()
    calls: list[dict[str, object]] = []

    def fake_download(
        series_id: str, **kwargs: object
    ) -> DownloadedVintageMatrix:
        assert series_id == "ICSA"
        calls.append(kwargs)
        return artifact

    monkeypatch.setattr(client, "download_level_matrix", fake_download)
    actual = client.list_first_release_observations(
        "ICSA",
        release_id=180,
        observation_start=date(2024, 1, 1),
        observation_end=date(2024, 1, 31),
        vintage_start=date(2024, 1, 1),
        vintage_end=date(2024, 1, 31),
    )

    assert actual.provider_id == "alfred_web"
    assert actual.source_url == "https://fred.stlouisfed.org/series/ICSA"
    rows = [
        (item.reference_date, item.release_date, item.value)
        for item in actual.observations
    ]
    assert rows == [
        (date(2024, 1, 6), date(2024, 1, 11), 202.0),
        (date(2024, 1, 13), date(2024, 1, 18), 211.0),
    ]
    assert len(calls) == 1
    assert calls[0]["release_id"] == 180


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("PAYEMS_20200207", date(2020, 2, 7)),
        ("prefix_with_underscores_19991231", date(1999, 12, 31)),
    ],
)
def test_vintage_date_from_column_uses_the_terminal_date_suffix(
    column: str,
    expected: date,
) -> None:
    assert vintage_date_from_column(column) == expected


def test_vintage_date_from_column_rejects_a_column_without_a_terminal_date() -> None:
    with pytest.raises(ValueError, match="does not end"):
        vintage_date_from_column("PAYEMS_latest")
