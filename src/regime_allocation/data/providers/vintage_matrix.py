"""Provider-neutral contracts for point-in-time observation vintages.

The dataclasses and protocol in this module define the boundary between network
adapters and model data preparation. Utilities serialize and parse deterministic
observation-by-vintage matrices and select the earliest numeric appearance of
each reference period. Inputs may be monthly or weekly; outputs preserve both
the economic reference date and publication date so later stages can enforce
causal cutoffs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Protocol
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd


_VINTAGE_SUFFIX = re.compile(r"(\d{8})$")


@dataclass(frozen=True)
class DownloadedVintageMatrix:
    """Normalized provider response used by every point-in-time downloader."""

    series_id: str
    selected_vintage_dates: tuple[date, ...]
    content: bytes
    source_url: str
    provider_id: str = "unknown"


@dataclass(frozen=True, order=True)
class FirstReleaseObservation:
    """One observation frozen at the first selected vintage containing it.

    ``reference_date`` is the economic observation period supplied by FRED.
    It is deliberately not called ``reference_month`` because the same
    contract also represents weekly series such as unemployment claims.
    ``release_date`` is the FRED/ALFRED vintage date on which ``value`` first
    appears in the selected point-in-time archive.
    """

    reference_date: date
    release_date: date
    value: float


@dataclass(frozen=True)
class DownloadedFirstReleaseObservations:
    """Provider response with credential-free provenance for event records."""

    series_id: str
    observations: tuple[FirstReleaseObservation, ...]
    source_url: str
    provider_id: str = "unknown"


class VintageMatrixProvider(Protocol):
    """Small interface required by the Model 01 acquisition pipeline."""

    provider_id: str
    provider_description: str
    cache_namespace: str

    @staticmethod
    def series_page_url(series_id: str) -> str:
        """Return a public, credential-free page describing a series."""

        ...

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
        """Return level snapshots for the requested point-in-time vintages."""

        ...

    def list_release_dates(
        self,
        release_id: int,
        *,
        release_start: date | None = None,
        release_end: date | None = None,
    ) -> tuple[date, ...]:
        """Return official publication dates for a named data release."""

        ...

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
        """Return first-release values for monthly or higher-frequency data."""

        ...


def encode_vintage_matrix(frame: pd.DataFrame, series_id: str) -> bytes:
    """Serialize a vintage matrix to a byte-stable, one-member ZIP archive."""

    normalized = frame.copy()
    normalized.index.name = "observation_date"
    csv_payload = normalized.to_csv(na_rep=".").encode("utf-8")
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        info = ZipInfo(
            filename=f"{series_id}_levels_by_vintage.csv",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        info.compress_type = ZIP_DEFLATED
        archive.writestr(info, csv_payload)
    return buffer.getvalue()


def load_vintage_matrix(
    source: bytes | Path,
    series_id: str,
    *,
    requested_vintages: tuple[date, ...] | None = None,
) -> pd.DataFrame:
    """Read a normalized or official FRED ZIP into a vintage matrix."""

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
                raise ValueError(
                    f"expected one data CSV for {series_id}; found {len(candidates)}"
                )
            with archive.open(candidates[0]) as handle:
                frame = pd.read_csv(handle, low_memory=False)
    except BadZipFile as exc:
        raise ValueError(f"invalid ZIP for {series_id}") from exc

    if frame.empty or frame.columns[0] not in {
        "observation_date",
        "period_start_date",
    }:
        raise ValueError(f"unexpected vintage-matrix schema for {series_id}")

    observation_column = frame.columns[0]
    frame[observation_column] = pd.to_datetime(frame[observation_column])
    frame = frame.set_index(observation_column).sort_index()
    if (
        requested_vintages is not None
        and len(requested_vintages) == 1
        and series_id in frame.columns
    ):
        only = requested_vintages[0]
        frame = frame.rename(columns={series_id: f"{series_id}_{only:%Y%m%d}"})
    vintage_columns: list[tuple[date, str]] = []
    expected_column = re.compile(rf"^{re.escape(series_id)}_(\d{{8}})$")
    for column in frame.columns:
        match = expected_column.fullmatch(str(column))
        if match:
            raw_date = match.group(1)
            vintage = date.fromisoformat(
                f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            )
            vintage_columns.append((vintage, str(column)))
    if not vintage_columns:
        raise ValueError(f"no vintage columns found for {series_id}")
    dates = [vintage for vintage, _ in vintage_columns]
    if len(dates) != len(set(dates)):
        raise ValueError(f"duplicate vintage dates found for {series_id}")

    ordered_columns = [column for _, column in sorted(vintage_columns)]
    if requested_vintages is not None:
        expected_columns = {
            f"{series_id}_{vintage:%Y%m%d}" for vintage in requested_vintages
        }
        if set(ordered_columns) != expected_columns:
            raise ValueError(f"vintage mismatch for {series_id}")
    numeric = frame[ordered_columns].apply(pd.to_numeric, errors="coerce")
    numeric.index.name = "reference_month"
    return numeric


def first_release_observations_from_matrix(
    frame: pd.DataFrame,
) -> tuple[FirstReleaseObservation, ...]:
    """Select each row's earliest numeric observation from a vintage matrix.

    The caller is responsible for ensuring the matrix contains the intended
    release-calendar vintages. Rows that never contain a numeric value are
    omitted. The implementation is independent of observation frequency.
    """

    if frame.index.has_duplicates:
        raise ValueError("vintage matrix contains duplicate reference dates")

    ordered = frame.reindex(
        sorted(frame.columns, key=lambda column: vintage_date_from_column(str(column))),
        axis=1,
    ).sort_index()
    observations: list[FirstReleaseObservation] = []
    for raw_reference_date, row in ordered.iterrows():
        numeric = pd.to_numeric(row, errors="coerce")
        available = numeric[numeric.notna()]
        if available.empty:
            continue
        first_column = str(available.index[0])
        observations.append(
            FirstReleaseObservation(
                reference_date=pd.Timestamp(raw_reference_date).date(),
                release_date=vintage_date_from_column(first_column),
                value=float(available.iloc[0]),
            )
        )
    return tuple(observations)


def vintage_date_from_column(column: str) -> date:
    """Extract the terminal ``YYYYMMDD`` vintage suffix from a column."""

    match = _VINTAGE_SUFFIX.search(column)
    if match is None:
        raise ValueError(f"column does not end in a vintage date: {column}")
    raw = match.group(1)
    return date.fromisoformat(f"{raw[:4]}-{raw[4:6]}-{raw[6:]}")
