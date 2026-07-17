"""Authenticated FRED API provider for point-in-time vintage matrices.

The API key is accepted only as an in-memory constructor argument. It is never
included in cache identities, filenames, metadata, subprocess arguments,
public source URLs, or raised exception messages.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from regime_allocation.data.providers.vintage_matrix import (
    DownloadedFirstReleaseObservations,
    DownloadedVintageMatrix,
    FirstReleaseObservation,
    encode_vintage_matrix,
    load_vintage_matrix,
    vintage_date_from_column,
)


FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
_API_KEY = re.compile(r"^[a-z0-9]{32}$")
_SERIES_ID = re.compile(r"^[A-Za-z0-9_]+$")
_TRANSIENT_HTTP_STATUSES = {423, 429, 500, 502, 503, 504}


class FredApiError(RuntimeError):
    """Raised for sanitized FRED API transport or schema failures."""


class FredApiConfigurationError(FredApiError):
    """Raised when the runtime API-key configuration is missing or invalid."""


class FredApiDownloadClient:
    """Retrieve first-release dates and full same-vintage level snapshots."""

    provider_id = "fred_api"
    provider_description = "FRED API v1 (authenticated primary provider)"
    cache_namespace = "fred_api"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: int = 90,
        max_vintages_per_request: int = 100,
        max_attempts: int = 5,
        request_pause_seconds: float = 0.25,
    ) -> None:
        normalized_key = api_key.strip()
        if not _API_KEY.fullmatch(normalized_key):
            raise FredApiConfigurationError(
                "FRED_API_KEY must be a 32-character lowercase alphanumeric key"
            )
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if max_vintages_per_request < 1 or max_vintages_per_request > 1000:
            raise ValueError("max_vintages_per_request must be between 1 and 1000")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if request_pause_seconds < 0:
            raise ValueError("request_pause_seconds cannot be negative")

        self._api_key = normalized_key
        self.timeout_seconds = timeout_seconds
        self.max_vintages_per_request = max_vintages_per_request
        self.max_attempts = max_attempts
        self.request_pause_seconds = request_pause_seconds
        self.user_agent = "global-macro-regime-allocation/0.1 (research project)"
        self._initial_release_cache: dict[
            tuple[str, date, date, date, date], tuple[date, ...]
        ] = {}
        self._initial_observation_cache: dict[
            tuple[str, date, date, date, date],
            tuple[FirstReleaseObservation, ...],
        ] = {}
        self._release_date_cache: dict[int, tuple[date, ...]] = {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(timeout_seconds={self.timeout_seconds}, "
            f"max_vintages_per_request={self.max_vintages_per_request}, "
            f"max_attempts={self.max_attempts})"
        )

    @staticmethod
    def series_page_url(series_id: str) -> str:
        FredApiDownloadClient._validate_series_id(series_id)
        return f"https://fred.stlouisfed.org/series/{series_id}"

    @staticmethod
    def _validate_series_id(series_id: str) -> None:
        if not _SERIES_ID.fullmatch(series_id):
            raise ValueError(f"invalid FRED series id: {series_id!r}")

    @staticmethod
    def _safe_query_identity(endpoint: str, params: Mapping[str, object]) -> str:
        if "api_key" in params:
            raise ValueError("cache identity must not receive credential parameters")
        payload = {
            "endpoint": endpoint,
            "params": {str(key): str(value) for key, value in params.items()},
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:16]

    def _open_once(self, request: Request) -> bytes:
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read()

    def _request_bytes(
        self, endpoint: str, params: Mapping[str, object]
    ) -> bytes:
        if not re.fullmatch(r"[a-z_/]+", endpoint):
            raise ValueError("invalid FRED API endpoint")
        if "api_key" in params:
            raise ValueError("API parameters must not contain api_key")

        request_params = {str(key): str(value) for key, value in params.items()}
        request_params["api_key"] = self._api_key
        url = f"{FRED_API_BASE_URL}/{endpoint}?{urlencode(request_params, safe=',')}"
        request = Request(url, headers={"User-Agent": self.user_agent})

        for attempt in range(self.max_attempts):
            retry = False
            try:
                return self._open_once(request)
            except HTTPError as exc:
                retry = exc.code in _TRANSIENT_HTTP_STATUSES
            except (URLError, TimeoutError, OSError):
                retry = True

            if not retry or attempt + 1 >= self.max_attempts:
                break
            delay = min(30.0, 2.0**attempt) + random.uniform(0.0, 0.25)
            time.sleep(delay)

        # Do not chain the transport exception: HTTPError retains the complete
        # credential-bearing request URL and can expose it in a traceback.
        raise FredApiError(
            f"FRED API request failed for endpoint {endpoint}"
        ) from None

    def _request_json(
        self, endpoint: str, params: Mapping[str, object]
    ) -> dict[str, Any]:
        payload = self._request_bytes(endpoint, params)
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise FredApiError(
                f"FRED API returned invalid JSON for endpoint {endpoint}"
            ) from None
        if not isinstance(decoded, dict):
            raise FredApiError(
                f"FRED API returned an unexpected JSON schema for endpoint {endpoint}"
            )
        return decoded

    def _fetch_initial_release_observations(
        self,
        series_id: str,
        *,
        observation_start: date,
        observation_end: date,
        vintage_start: date,
        vintage_end: date,
    ) -> tuple[FirstReleaseObservation, ...]:
        """Return numeric FRED output-type-4 observations with exact vintages."""

        self._validate_series_id(series_id)
        cache_key = (
            series_id,
            observation_start,
            observation_end,
            vintage_start,
            vintage_end,
        )
        cached = self._initial_observation_cache.get(cache_key)
        if cached is not None:
            return cached

        records: list[FirstReleaseObservation] = []
        offset = 0
        while True:
            params: dict[str, object] = {
                "series_id": series_id,
                "file_type": "json",
                "units": "lin",
                "output_type": 4,
                "observation_start": observation_start.isoformat(),
                "observation_end": observation_end.isoformat(),
                "realtime_start": vintage_start.isoformat(),
                "realtime_end": vintage_end.isoformat(),
                "sort_order": "asc",
                "limit": 100000,
                "offset": offset,
            }
            response = self._request_json("series/observations", params)
            if (
                response.get("output_type") != 4
                or response.get("file_type") != "json"
                or response.get("units") != "lin"
            ):
                raise FredApiError(
                    f"FRED API returned unexpected initial-release metadata "
                    f"for {series_id}"
                )
            observations = response.get("observations")
            if not isinstance(observations, list):
                raise FredApiError(
                    f"FRED API omitted initial observations for {series_id}"
                )
            try:
                total = int(response["count"])
                response_offset = int(response["offset"])
            except (KeyError, TypeError, ValueError):
                raise FredApiError(
                    f"FRED API returned invalid pagination for {series_id}"
                ) from None
            if response_offset != offset:
                raise FredApiError(
                    f"FRED API returned an unexpected offset for {series_id}"
                )

            for observation in observations:
                if not isinstance(observation, dict):
                    raise FredApiError(
                        f"FRED API returned an invalid observation for {series_id}"
                    )
                try:
                    reference_date = date.fromisoformat(str(observation["date"]))
                    release_date = date.fromisoformat(
                        str(observation["realtime_start"])
                    )
                    raw_value = str(observation["value"])
                except (KeyError, ValueError):
                    raise FredApiError(
                        f"FRED API returned an invalid initial observation for "
                        f"{series_id}"
                    ) from None
                if not observation_start <= reference_date <= observation_end:
                    raise FredApiError(
                        f"FRED API returned an out-of-range observation for {series_id}"
                    )
                if not vintage_start <= release_date <= vintage_end:
                    raise FredApiError(
                        f"FRED API returned an out-of-range vintage for {series_id}"
                    )
                if raw_value == ".":
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    raise FredApiError(
                        f"FRED API returned a nonnumeric initial value for {series_id}"
                    ) from None
                records.append(
                    FirstReleaseObservation(
                        reference_date=reference_date,
                        release_date=release_date,
                        value=value,
                    )
                )

            offset += len(observations)
            if offset >= total:
                break
            if not observations:
                raise FredApiError(
                    f"FRED API pagination stalled for {series_id}"
                )

        selected = tuple(sorted(records))
        if not selected:
            raise FredApiError(
                f"FRED API returned no initial observations for {series_id}"
            )
        reference_dates = [item.reference_date for item in selected]
        if len(reference_dates) != len(set(reference_dates)):
            raise FredApiError(
                f"FRED API returned duplicate initial observations for {series_id}"
            )
        self._initial_observation_cache[cache_key] = selected
        return selected

    def list_initial_release_dates(
        self,
        series_id: str,
        *,
        observation_start: date,
        observation_end: date,
        vintage_start: date,
        vintage_end: date,
    ) -> tuple[date, ...]:
        """Return dates attached to FRED output type 4 initial observations."""

        cache_key = (
            series_id,
            observation_start,
            observation_end,
            vintage_start,
            vintage_end,
        )
        cached = self._initial_release_cache.get(cache_key)
        if cached is not None:
            return cached
        observations = self._fetch_initial_release_observations(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_start=vintage_start,
            vintage_end=vintage_end,
        )
        selected = tuple(sorted({item.release_date for item in observations}))
        self._initial_release_cache[cache_key] = selected
        return selected

    def list_release_dates(
        self,
        release_id: int,
        *,
        release_start: date | None = None,
        release_end: date | None = None,
    ) -> tuple[date, ...]:
        """Return the official release calendar exposed by the FRED API."""

        if release_id < 1:
            raise ValueError("release_id must be positive")
        if (
            release_start is not None
            and release_end is not None
            and release_start > release_end
        ):
            raise ValueError("release_start cannot follow release_end")

        available = self._release_date_cache.get(release_id)
        if available is None:
            records: list[date] = []
            offset = 0
            while True:
                response = self._request_json(
                    "release/dates",
                    {
                        "release_id": release_id,
                        "file_type": "json",
                        "include_release_dates_with_no_data": "true",
                        "sort_order": "asc",
                        "limit": 1000,
                        "offset": offset,
                    },
                )
                release_dates = response.get("release_dates")
                if not isinstance(release_dates, list):
                    raise FredApiError(
                        f"FRED API omitted dates for release {release_id}"
                    )
                try:
                    total = int(response["count"])
                    response_offset = int(response["offset"])
                except (KeyError, TypeError, ValueError):
                    raise FredApiError(
                        f"FRED API returned invalid release-date pagination for "
                        f"release {release_id}"
                    ) from None
                if response_offset != offset:
                    raise FredApiError(
                        f"FRED API returned an unexpected release-date offset for "
                        f"release {release_id}"
                    )
                for item in release_dates:
                    if not isinstance(item, dict):
                        raise FredApiError(
                            f"FRED API returned an invalid date for release "
                            f"{release_id}"
                        )
                    try:
                        records.append(date.fromisoformat(str(item["date"])))
                    except (KeyError, ValueError):
                        raise FredApiError(
                            f"FRED API returned an invalid date for release "
                            f"{release_id}"
                        ) from None
                offset += len(release_dates)
                if offset >= total:
                    break
                if not release_dates:
                    raise FredApiError(
                        f"FRED API release-date pagination stalled for release "
                        f"{release_id}"
                    )
            available = tuple(sorted(set(records)))
            if not available:
                raise FredApiError(
                    f"FRED API returned no dates for release {release_id}"
                )
            self._release_date_cache[release_id] = available

        return tuple(
            item
            for item in available
            if (release_start is None or item >= release_start)
            and (release_end is None or item <= release_end)
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
        """Return exact first-release values using FRED output type 4."""

        self._validate_series_id(series_id)
        if release_id < 1:
            raise ValueError("release_id must be positive")
        if observation_start > observation_end:
            raise ValueError("observation_start cannot follow observation_end")
        if vintage_start > vintage_end:
            raise ValueError("vintage_start cannot follow vintage_end")

        query: dict[str, object] = {
            "series_id": series_id,
            "release_id": release_id,
            "observation_start": observation_start.isoformat(),
            "observation_end": observation_end.isoformat(),
            "vintage_start": vintage_start.isoformat(),
            "vintage_end": vintage_end.isoformat(),
            "output_type": 4,
        }
        cache_path: Path | None = None
        if chunk_cache_dir is not None:
            cache_hash = self._safe_query_identity(
                "series/observations/first-release-values", query
            )
            cache_path = chunk_cache_dir / (
                f"first_release_observations_{cache_hash}.json"
            )

        observations: tuple[FirstReleaseObservation, ...]
        if cache_path is not None and cache_path.exists() and not refresh_cache:
            try:
                stored = json.loads(cache_path.read_text(encoding="utf-8"))
                if stored.get("query") != query:
                    raise ValueError("query mismatch")
                observations = tuple(
                    FirstReleaseObservation(
                        reference_date=date.fromisoformat(str(item["reference_date"])),
                        release_date=date.fromisoformat(str(item["release_date"])),
                        value=float(item["value"]),
                    )
                    for item in stored["observations"]
                )
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                raise FredApiError(
                    f"invalid cached first-release observations for {series_id}"
                ) from None
            if not observations:
                raise FredApiError(
                    f"cached first-release observations are empty for {series_id}"
                )
        else:
            if refresh_cache:
                observation_cache_key = (
                    series_id,
                    observation_start,
                    observation_end,
                    vintage_start,
                    vintage_end,
                )
                self._initial_observation_cache.pop(
                    observation_cache_key, None
                )
                self._initial_release_cache.pop(observation_cache_key, None)
            observations = self._fetch_initial_release_observations(
                series_id,
                observation_start=observation_start,
                observation_end=observation_end,
                vintage_start=vintage_start,
                vintage_end=vintage_end,
            )
            if cache_path is not None:
                content = (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "provider_id": self.provider_id,
                            "query": query,
                            "observations": [
                                {
                                    "reference_date": item.reference_date.isoformat(),
                                    "release_date": item.release_date.isoformat(),
                                    "value": item.value,
                                }
                                for item in observations
                            ],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                self._write_atomic(cache_path, content)

        return DownloadedFirstReleaseObservations(
            series_id=series_id,
            observations=observations,
            source_url=self.series_page_url(series_id),
            provider_id=self.provider_id,
        )

    @staticmethod
    def _write_atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

    def _load_or_fetch_initial_dates(
        self,
        series_id: str,
        *,
        observation_start: date,
        observation_end: date,
        vintage_start: date,
        vintage_end: date,
        chunk_cache_dir: Path | None,
        refresh_cache: bool,
    ) -> tuple[date, ...]:
        query: dict[str, object] = {
            "series_id": series_id,
            "observation_start": observation_start.isoformat(),
            "observation_end": observation_end.isoformat(),
            "vintage_start": vintage_start.isoformat(),
            "vintage_end": vintage_end.isoformat(),
            "output_type": 4,
        }
        cache_path: Path | None = None
        if chunk_cache_dir is not None:
            cache_hash = self._safe_query_identity(
                "series/observations/initial-release-dates", query
            )
            cache_path = chunk_cache_dir / f"initial_release_dates_{cache_hash}.json"

        if cache_path is not None and cache_path.exists() and not refresh_cache:
            try:
                stored = json.loads(cache_path.read_text(encoding="utf-8"))
                if stored.get("query") != query:
                    raise ValueError("query mismatch")
                dates = tuple(
                    sorted(
                        date.fromisoformat(str(item)) for item in stored["dates"]
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise FredApiError(
                    f"invalid cached initial-release dates for {series_id}"
                ) from None
            if not dates:
                raise FredApiError(
                    f"cached initial-release dates are empty for {series_id}"
                )
            return dates

        dates = self.list_initial_release_dates(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_start=vintage_start,
            vintage_end=vintage_end,
        )
        if cache_path is not None:
            content = (
                json.dumps(
                    {
                        "schema_version": 1,
                        "provider_id": self.provider_id,
                        "query": query,
                        "dates": [item.isoformat() for item in dates],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            self._write_atomic(cache_path, content)
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
        """Download output-type-2 snapshots at every initial-release date."""

        self._validate_series_id(series_id)
        if release_id < 1:
            raise ValueError("release_id must be positive")
        if observation_start > observation_end:
            raise ValueError("observation_start cannot follow observation_end")
        if vintage_start > vintage_end:
            raise ValueError("vintage_start cannot follow vintage_end")

        selected = self._load_or_fetch_initial_dates(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
            vintage_start=vintage_start,
            vintage_end=vintage_end,
            chunk_cache_dir=chunk_cache_dir,
            refresh_cache=refresh_cache,
        )

        matrices: list[pd.DataFrame] = []
        for start in range(0, len(selected), self.max_vintages_per_request):
            chunk = selected[start : start + self.max_vintages_per_request]
            params: dict[str, object] = {
                "series_id": series_id,
                "file_type": "csv",
                "units": "lin",
                "output_type": 2,
                "observation_start": observation_start.isoformat(),
                "observation_end": observation_end.isoformat(),
                "vintage_dates": ",".join(item.isoformat() for item in chunk),
                "sort_order": "asc",
                "limit": 100000,
                "offset": 0,
            }
            cache_path: Path | None = None
            if chunk_cache_dir is not None:
                cache_hash = self._safe_query_identity(
                    "series/observations/vintage-matrix", params
                )
                cache_path = chunk_cache_dir / (
                    f"{chunk[0].isoformat()}_{chunk[-1].isoformat()}_"
                    f"{cache_hash}.zip"
                )

            fetched_remotely = False
            if cache_path is not None and cache_path.exists() and not refresh_cache:
                payload = cache_path.read_bytes()
            else:
                payload = self._request_bytes("series/observations", params)
                fetched_remotely = True

            try:
                matrix = load_vintage_matrix(
                    payload,
                    series_id,
                    requested_vintages=chunk,
                )
            except ValueError:
                raise FredApiError(
                    f"FRED API returned an invalid vintage matrix for {series_id}"
                ) from None
            returned = {
                vintage_date_from_column(str(column)) for column in matrix.columns
            }
            requested = set(chunk)
            if returned != requested:
                missing = sorted(requested.difference(returned))
                unexpected = sorted(returned.difference(requested))
                raise FredApiError(
                    f"FRED API vintage mismatch for {series_id}; "
                    f"missing={missing[:3]}, unexpected={unexpected[:3]}"
                )
            if fetched_remotely and cache_path is not None:
                # Cache only our normalized one-member ZIP. The official CSV
                # archive may contain provider README metadata, which is not
                # needed downstream and must never be trusted not to echo a
                # credential-bearing request URL.
                self._write_atomic(
                    cache_path,
                    encode_vintage_matrix(matrix, series_id),
                )
            matrices.append(matrix)
            if fetched_remotely and self.request_pause_seconds > 0:
                time.sleep(self.request_pause_seconds)

        merged = pd.concat(matrices, axis=1).sort_index()
        if merged.columns.duplicated().any():
            raise FredApiError(
                f"FRED API returned duplicate vintage columns for {series_id}"
            )
        merged = merged.reindex(
            sorted(merged.columns, key=vintage_date_from_column), axis=1
        )
        return DownloadedVintageMatrix(
            series_id=series_id,
            selected_vintage_dates=selected,
            content=encode_vintage_matrix(merged, series_id),
            source_url=self.series_page_url(series_id),
            provider_id=self.provider_id,
        )
