"""No-network tests for ALFRED's public vintage-matrix adapter."""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from regime_allocation.data.providers.alfred_web import (
    AlfredDownloadError,
    AlfredWebDownloadClient,
    load_graph_matrix,
    load_vintage_matrix,
    vintage_date_from_column,
)


def _zip_csv(csv_text: str, *, filename: str = "observations.csv") -> bytes:
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
