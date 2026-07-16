"""Credential-free adapter for ALFRED's public web data endpoints.

The official FRED API is the preferred stable interface, but it requires an API
key. This adapter gets official release dates from ALFRED and requests batched
historical snapshots from ``alfredgraph.csv``. Provider responses are cached so
research can be reproduced even if the web interface changes later.
"""

from __future__ import annotations

import hashlib
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd


BASE_URL = "https://alfred.stlouisfed.org/series/downloaddata"
RELEASE_DATES_URL = "https://alfred.stlouisfed.org/release/downloaddates"
ALFRED_GRAPH_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
_VINTAGE_SUFFIX = re.compile(r"(\d{8})$")
_ISO_DATE_LINE = re.compile(r"(?m)^(\d{4}-\d{2}-\d{2})\s*$")


class AlfredDownloadError(RuntimeError):
    """Raised when ALFRED cannot provide a valid vintage matrix."""


class _VintageDateParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside_vintage_select = False
        self.dates: list[date] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "select" and attributes.get("name") == (
            "form[selected_vintage_dates][]"
        ):
            self._inside_vintage_select = True
            return
        if tag == "option" and self._inside_vintage_select:
            value = attributes.get("value")
            if value:
                try:
                    self.dates.append(date.fromisoformat(value))
                except ValueError:
                    pass

    def handle_endtag(self, tag: str) -> None:
        if tag == "select" and self._inside_vintage_select:
            self._inside_vintage_select = False


@dataclass(frozen=True)
class DownloadedVintageMatrix:
    series_id: str
    selected_vintage_dates: tuple[date, ...]
    content: bytes
    source_url: str


class AlfredWebDownloadClient:
    """Download observation-by-vintage matrices without storing credentials."""

    def __init__(
        self,
        *,
        timeout_seconds: int = 90,
        max_vintages_per_request: int = 12,
        max_attempts: int = 5,
        request_pause_seconds: float = 0.75,
        prefer_curl: bool = True,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        if max_vintages_per_request < 1:
            raise ValueError("max_vintages_per_request must be positive")
        self.max_vintages_per_request = max_vintages_per_request
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts
        self.request_pause_seconds = request_pause_seconds
        self.curl_path = (
            shutil.which("curl.exe") or shutil.which("curl")
            if prefer_curl
            else None
        )
        self.user_agent = "global-macro-regime-allocation/0.1 (research project)"
        self._release_date_cache: dict[int, tuple[date, ...]] = {}

    @staticmethod
    def series_url(series_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_]+", series_id):
            raise ValueError(f"invalid FRED series id: {series_id!r}")
        return f"{BASE_URL}?seid={series_id}"

    def _open_once(self, request: Request) -> bytes:
        if self.curl_path is None:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()

        command = [
            self.curl_path,
            "-L",
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--max-time",
            str(self.timeout_seconds),
        ]
        if request.data is not None:
            command.extend(
                [
                    "--header",
                    "Content-Type: application/x-www-form-urlencoded",
                    "--data-binary",
                    "@-",
                ]
            )
        command.append(request.full_url)
        completed = subprocess.run(
            command,
            input=request.data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds + 10,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0:
            raise URLError("curl transport failed")
        return completed.stdout

    def _open(self, request: Request) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return self._open_once(request)
            except (
                HTTPError,
                URLError,
                TimeoutError,
                OSError,
                subprocess.SubprocessError,
            ) as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(5.0 * (attempt + 1) + random.uniform(0.0, 1.0))
        # Never include request bodies in errors. A later API-backed adapter
        # may contain credentials in its body or headers.
        raise AlfredDownloadError("ALFRED download request failed") from last_error

    def list_vintage_dates(self, series_id: str) -> tuple[date, ...]:
        """Parse a series form's vintage picker.

        This remains available as a diagnostic fallback. Production downloads
        use ``list_release_dates`` because the release calendar endpoint is
        much smaller and more reliable than the full series form.
        """

        request = Request(
            self.series_url(series_id),
            headers={"User-Agent": self.user_agent},
        )
        parser = _VintageDateParser()
        parser.feed(self._open(request).decode("utf-8", errors="replace"))
        dates = tuple(sorted(set(parser.dates)))
        if not dates:
            raise AlfredDownloadError(
                f"ALFRED returned no vintage dates for {series_id}"
            )
        return dates

    def list_release_dates(self, release_id: int) -> tuple[date, ...]:
        """Return all official ALFRED dates for one named data release."""

        if release_id < 1:
            raise ValueError("release_id must be positive")
        cached = self._release_date_cache.get(release_id)
        if cached is not None:
            return cached
        url = f"{RELEASE_DATES_URL}?ff=txt&rid={release_id}"
        request = Request(url, headers={"User-Agent": self.user_agent})
        payload = self._open(request).decode("utf-8", errors="replace")
        dates = tuple(
            sorted(
                {
                    date.fromisoformat(raw)
                    for raw in _ISO_DATE_LINE.findall(payload)
                }
            )
        )
        if not dates:
            raise AlfredDownloadError(
                f"ALFRED returned no dates for release {release_id}"
            )
        self._release_date_cache[release_id] = dates
        return dates

    def download_level_matrix(
        self,
        series_id: str,
        *,
        release_id: int,
        observation_start: date,
        observation_end: date,
        vintage_start: date,
        vintage_end: date,
        chunk_cache_dir: Path | None = None,
        refresh_cache: bool = False,
    ) -> DownloadedVintageMatrix:
        available = self.list_release_dates(release_id)
        selected = tuple(
            vintage
            for vintage in available
            if vintage_start <= vintage <= vintage_end
        )
        if not selected:
            raise AlfredDownloadError(
                f"no {series_id} vintages fall in the requested date range"
            )

        matrices: list[pd.DataFrame] = []
        for start in range(0, len(selected), self.max_vintages_per_request):
            chunk = selected[start : start + self.max_vintages_per_request]
            count = len(chunk)
            query = {
                "id": ",".join([series_id] * count),
                "vintage_date": ",".join(item.isoformat() for item in chunk),
                "cosd": ",".join([observation_start.isoformat()] * count),
                "coed": ",".join([observation_end.isoformat()] * count),
            }
            query_string = urlencode(query, safe=",")
            request_url = f"{ALFRED_GRAPH_URL}?{query_string}"
            cache_path: Path | None = None
            if chunk_cache_dir is not None:
                cache_key = hashlib.sha256(
                    request_url.encode("ascii")
                ).hexdigest()[:16]
                cache_path = chunk_cache_dir / (
                    f"{chunk[0].isoformat()}_{chunk[-1].isoformat()}_{cache_key}.csv"
                )

            fetched_remotely = False
            if (
                cache_path is not None
                and cache_path.exists()
                and not refresh_cache
            ):
                chunk_content = cache_path.read_bytes()
            else:
                fetched_remotely = True
                request = Request(
                    request_url,
                    headers={"User-Agent": self.user_agent},
                )
                try:
                    chunk_content = self._open(request)
                except AlfredDownloadError as exc:
                    raise AlfredDownloadError(
                        f"ALFRED failed for {series_id} vintages "
                        f"{chunk[0]} through {chunk[-1]}"
                    ) from exc
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(".csv.tmp")
                    temporary.write_bytes(chunk_content)
                    temporary.replace(cache_path)
            chunk_matrix = load_graph_matrix(
                chunk_content,
                series_id=series_id,
                requested_vintages=chunk,
            )
            returned_dates = {
                vintage_date_from_column(str(column))
                for column in chunk_matrix.columns
            }
            requested_dates = set(chunk)
            if returned_dates != requested_dates:
                missing = sorted(requested_dates.difference(returned_dates))
                unexpected = sorted(returned_dates.difference(requested_dates))
                raise AlfredDownloadError(
                    f"ALFRED vintage mismatch for {series_id}; "
                    f"missing={missing[:3]}, unexpected={unexpected[:3]}"
                )
            matrices.append(chunk_matrix)
            if fetched_remotely and self.request_pause_seconds > 0:
                time.sleep(self.request_pause_seconds)

        merged = pd.concat(matrices, axis=1).sort_index()
        if merged.columns.duplicated().any():
            raise AlfredDownloadError(
                f"duplicate vintage columns returned for {series_id}"
            )
        merged = merged.reindex(
            sorted(merged.columns, key=vintage_date_from_column), axis=1
        )
        merged.index.name = "observation_date"
        csv_payload = merged.to_csv(na_rep=".").encode("utf-8")
        buffer = BytesIO()
        with ZipFile(buffer, mode="w") as archive:
            info = ZipInfo(
                filename=f"{series_id}_levels_by_vintage.csv",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, csv_payload)
        content = buffer.getvalue()
        return DownloadedVintageMatrix(
            series_id=series_id,
            selected_vintage_dates=selected,
            content=content,
            source_url=self.series_url(series_id),
        )


def load_graph_matrix(
    payload: bytes,
    *,
    series_id: str,
    requested_vintages: tuple[date, ...],
) -> pd.DataFrame:
    """Read a batched ``alfredgraph.csv`` response into a vintage matrix."""

    try:
        frame = pd.read_csv(BytesIO(payload))
    except Exception as exc:
        raise AlfredDownloadError(
            f"invalid ALFRED graph CSV for {series_id}"
        ) from exc
    if frame.empty or frame.columns[0] != "observation_date":
        raise AlfredDownloadError(
            f"unexpected ALFRED graph schema for {series_id}"
        )
    if len(requested_vintages) == 1 and series_id in frame.columns:
        only = requested_vintages[0]
        frame = frame.rename(
            columns={series_id: f"{series_id}_{only:%Y%m%d}"}
        )

    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame = frame.set_index("observation_date").sort_index()
    expected_columns = [
        f"{series_id}_{vintage:%Y%m%d}" for vintage in requested_vintages
    ]
    actual_columns = [str(column) for column in frame.columns]
    if set(actual_columns) != set(expected_columns):
        missing = sorted(set(expected_columns).difference(actual_columns))
        unexpected = sorted(set(actual_columns).difference(expected_columns))
        raise AlfredDownloadError(
            f"ALFRED graph vintage mismatch for {series_id}; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    numeric = frame[expected_columns].apply(pd.to_numeric, errors="coerce")
    numeric.index.name = "reference_month"
    return numeric


def load_vintage_matrix(source: bytes | Path, series_id: str) -> pd.DataFrame:
    """Read an ALFRED zipped CSV into a monthly observation-by-vintage matrix."""

    payload = source if isinstance(source, bytes) else source.read_bytes()
    try:
        with ZipFile(BytesIO(payload)) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv")
                and "readme" not in name.lower()
            ]
            if len(candidates) != 1:
                raise AlfredDownloadError(
                    f"expected one data CSV for {series_id}; found {len(candidates)}"
                )
            with archive.open(candidates[0]) as handle:
                frame = pd.read_csv(handle)
    except BadZipFile as exc:
        raise AlfredDownloadError(f"invalid ZIP for {series_id}") from exc

    if frame.empty or frame.columns[0] not in {
        "observation_date",
        "period_start_date",
    }:
        raise AlfredDownloadError(
            f"unexpected vintage-matrix schema for {series_id}"
        )

    observation_column = frame.columns[0]
    frame[observation_column] = pd.to_datetime(frame[observation_column])
    frame = frame.set_index(observation_column).sort_index()
    vintage_columns: list[tuple[date, str]] = []
    expected_column = re.compile(rf"^{re.escape(series_id)}_(\d{{8}})$")
    for column in frame.columns:
        match = expected_column.fullmatch(str(column))
        if match:
            raw_date = match.group(1)
            vintage = date.fromisoformat(
                f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            )
            vintage_columns.append(
                (vintage, column)
            )
    if not vintage_columns:
        raise AlfredDownloadError(f"no vintage columns found for {series_id}")
    dates = [vintage for vintage, _ in vintage_columns]
    if len(dates) != len(set(dates)):
        raise AlfredDownloadError(f"duplicate vintage dates found for {series_id}")

    ordered_columns = [column for _, column in sorted(vintage_columns)]
    numeric = frame[ordered_columns].apply(pd.to_numeric, errors="coerce")
    numeric.index.name = "reference_month"
    return numeric


def vintage_date_from_column(column: str) -> date:
    match = _VINTAGE_SUFFIX.search(column)
    if match is None:
        raise ValueError(f"column does not end in a vintage date: {column}")
    raw = match.group(1)
    return date.fromisoformat(f"{raw[:4]}-{raw[4:6]}-{raw[6:]}")
