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
from datetime import date
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from regime_allocation.data.providers.vintage_matrix import (
    DownloadedFirstReleaseObservations,
    DownloadedVintageMatrix,
    encode_vintage_matrix,
    first_release_observations_from_matrix,
    load_vintage_matrix as _load_vintage_matrix,
    vintage_date_from_column,
)


BASE_URL = "https://alfred.stlouisfed.org/series/downloaddata"
RELEASE_DATES_URL = "https://alfred.stlouisfed.org/release/downloaddates"
ALFRED_GRAPH_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
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


class AlfredWebDownloadClient:
    """Download observation-by-vintage matrices without storing credentials."""

    provider_id = "alfred_web"
    provider_description = "ALFRED public web endpoints (keyless fallback)"
    cache_namespace = "alfred"

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

    @staticmethod
    def series_page_url(series_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_]+", series_id):
            raise ValueError(f"invalid FRED series id: {series_id!r}")
        return f"https://fred.stlouisfed.org/series/{series_id}"

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

    def list_release_dates(
        self,
        release_id: int,
        *,
        release_start: date | None = None,
        release_end: date | None = None,
    ) -> tuple[date, ...]:
        """Return all official ALFRED dates for one named data release."""

        if release_id < 1:
            raise ValueError("release_id must be positive")
        if (
            release_start is not None
            and release_end is not None
            and release_start > release_end
        ):
            raise ValueError("release_start cannot follow release_end")
        cached = self._release_date_cache.get(release_id)
        if cached is None:
            url = f"{RELEASE_DATES_URL}?ff=txt&rid={release_id}"
            request = Request(url, headers={"User-Agent": self.user_agent})
            payload = self._open(request).decode("utf-8", errors="replace")
            cached = tuple(
                sorted(
                    {
                        date.fromisoformat(raw)
                        for raw in _ISO_DATE_LINE.findall(payload)
                    }
                )
            )
            if not cached:
                raise AlfredDownloadError(
                    f"ALFRED returned no dates for release {release_id}"
                )
            self._release_date_cache[release_id] = cached
        return tuple(
            item
            for item in cached
            if (release_start is None or item >= release_start)
            and (release_end is None or item <= release_end)
        )

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
            # A vintage cannot contain observations from after its as-of date.
            # Bounding each historical request prevents ALFRED from scanning a
            # needlessly long future observation window (especially costly for
            # weekly series) while preserving the exact returned matrix.
            chunk_observation_end = min(observation_end, chunk[-1])
            query = {
                "id": ",".join([series_id] * count),
                "vintage_date": ",".join(item.isoformat() for item in chunk),
                "cosd": ",".join([observation_start.isoformat()] * count),
                "coed": ",".join([chunk_observation_end.isoformat()] * count),
            }
            query_string = urlencode(query, safe=",")
            request_url = f"{ALFRED_GRAPH_URL}?{query_string}"
            cache_path: Path | None = None
            compatible_cache_path: Path | None = None
            if chunk_cache_dir is not None:
                cache_key = hashlib.sha256(
                    request_url.encode("ascii")
                ).hexdigest()[:16]
                cache_path = chunk_cache_dir / (
                    f"{chunk[0].isoformat()}_{chunk[-1].isoformat()}_{cache_key}.csv"
                )
                # Reuse chunks created before requests were bounded by their
                # final vintage. Both responses encode the same as-of values.
                legacy_query = dict(query)
                legacy_query["coed"] = ",".join(
                    [observation_end.isoformat()] * count
                )
                legacy_url = (
                    f"{ALFRED_GRAPH_URL}?"
                    f"{urlencode(legacy_query, safe=',')}"
                )
                legacy_key = hashlib.sha256(
                    legacy_url.encode("ascii")
                ).hexdigest()[:16]
                compatible_cache_path = chunk_cache_dir / (
                    f"{chunk[0].isoformat()}_{chunk[-1].isoformat()}_"
                    f"{legacy_key}.csv"
                )

            fetched_remotely = False
            if (
                cache_path is not None
                and cache_path.exists()
                and not refresh_cache
            ):
                chunk_content = cache_path.read_bytes()
            elif (
                compatible_cache_path is not None
                and compatible_cache_path.exists()
                and not refresh_cache
            ):
                chunk_content = compatible_cache_path.read_bytes()
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
        content = encode_vintage_matrix(merged, series_id)
        return DownloadedVintageMatrix(
            series_id=series_id,
            selected_vintage_dates=selected,
            content=content,
            source_url=self.series_page_url(series_id),
            provider_id=self.provider_id,
        )

    def list_first_release_observations(
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
    ) -> DownloadedFirstReleaseObservations:
        """Reconstruct first releases from keyless release-date snapshots."""

        artifact = self.download_level_matrix(
            series_id,
            release_id=release_id,
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_start=vintage_start,
            vintage_end=vintage_end,
            chunk_cache_dir=chunk_cache_dir,
            refresh_cache=refresh_cache,
        )
        matrix = _load_vintage_matrix(artifact.content, series_id)
        observations = first_release_observations_from_matrix(matrix)
        if not observations:
            raise AlfredDownloadError(
                f"ALFRED returned no first-release observations for {series_id}"
            )
        return DownloadedFirstReleaseObservations(
            series_id=series_id,
            observations=observations,
            source_url=artifact.source_url,
            provider_id=artifact.provider_id,
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
    """Backward-compatible ALFRED import for the shared matrix loader."""

    try:
        return _load_vintage_matrix(source, series_id)
    except ValueError as exc:
        raise AlfredDownloadError(str(exc)) from exc
