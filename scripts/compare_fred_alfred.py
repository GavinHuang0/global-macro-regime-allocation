"""Compare authenticated FRED and keyless ALFRED first-release data.

The comparison downloads into a temporary directory and writes only a compact
JSON/Markdown report. It never reads or changes Model 01's existing raw cache.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from regime_allocation.data.first_release import extract_first_release_features
from regime_allocation.data.providers.alfred_web import (
    AlfredWebDownloadClient,
    load_vintage_matrix as load_alfred_matrix,
)
from regime_allocation.data.providers.fred_api import FredApiDownloadClient
from regime_allocation.data.providers.vintage_matrix import (
    load_vintage_matrix as load_fred_matrix,
)


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    release_id: int
    component: str
    transform: str


SERIES = (
    SeriesSpec("PAYEMS", 50, "payrolls", "difference"),
    SeriesSpec("CPILFESL", 10, "core_cpi", "log_difference"),
)
FIELDS = (
    "current_value",
    "previous_value_as_of_release",
    "transformed_value",
)


def _value_comparison(
    merged: pd.DataFrame,
    field: str,
    *,
    atol: float,
    rtol: float,
) -> dict[str, int | float]:
    left = merged[f"{field}_fred"].to_numpy(dtype=float)
    right = merged[f"{field}_alfred"].to_numpy(dtype=float)
    matches = np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=True)
    both_finite = np.isfinite(left) & np.isfinite(right)
    max_difference = (
        float(np.max(np.abs(left[both_finite] - right[both_finite])))
        if both_finite.any()
        else 0.0
    )
    return {
        "mismatches": int((~matches).sum()),
        "max_absolute_difference": max_difference,
    }


def _matrix_comparison(
    fred: pd.DataFrame,
    alfred: pd.DataFrame,
    *,
    observation_start: date,
    observation_end: date,
    atol: float,
    rtol: float,
) -> dict[str, int | float]:
    fred_columns = set(map(str, fred.columns))
    alfred_columns = set(map(str, alfred.columns))
    common_columns = sorted(fred_columns.intersection(alfred_columns))
    missing_columns = sorted(fred_columns.difference(alfred_columns))
    index = fred.index.union(alfred.index)
    index = index[
        (index >= pd.Timestamp(observation_start))
        & (index <= pd.Timestamp(observation_end))
    ]
    if not common_columns:
        return {
            "fred_vintage_columns": len(fred_columns),
            "common_vintage_columns": 0,
            "fred_vintages_missing_from_alfred": len(missing_columns),
            "cells_compared": 0,
            "cell_mismatches": 0,
            "max_absolute_difference": 0.0,
        }

    left = fred.reindex(index=index, columns=common_columns).to_numpy(dtype=float)
    right = alfred.reindex(index=index, columns=common_columns).to_numpy(dtype=float)
    relevant = np.isfinite(left) | np.isfinite(right)
    matches = np.isclose(left, right, atol=atol, rtol=rtol, equal_nan=True)
    both_finite = np.isfinite(left) & np.isfinite(right)
    max_difference = (
        float(np.max(np.abs(left[both_finite] - right[both_finite])))
        if both_finite.any()
        else 0.0
    )
    return {
        "fred_vintage_columns": len(fred_columns),
        "common_vintage_columns": len(common_columns),
        "fred_vintages_missing_from_alfred": len(missing_columns),
        "cells_compared": int(relevant.sum()),
        "cell_mismatches": int((relevant & ~matches).sum()),
        "max_absolute_difference": max_difference,
    }


def _compare_series(
    spec: SeriesSpec,
    *,
    fred_client: FredApiDownloadClient,
    alfred_client: AlfredWebDownloadClient,
    work_dir: Path,
    observation_start: date,
    comparison_start: date,
    comparison_end: date,
    vintage_start: date,
    vintage_end: date,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    fred_artifact = fred_client.download_level_matrix(
        spec.series_id,
        release_id=spec.release_id,
        observation_start=observation_start,
        observation_end=comparison_end,
        vintage_start=vintage_start,
        vintage_end=vintage_end,
        chunk_cache_dir=work_dir / "fred" / spec.series_id,
        refresh_cache=True,
    )
    alfred_artifact = alfred_client.download_level_matrix(
        spec.series_id,
        release_id=spec.release_id,
        observation_start=observation_start,
        observation_end=comparison_end,
        vintage_start=vintage_start,
        vintage_end=vintage_end,
        chunk_cache_dir=work_dir / "alfred" / spec.series_id,
        refresh_cache=True,
    )
    fred_matrix = load_fred_matrix(fred_artifact.content, spec.series_id)
    alfred_matrix = load_alfred_matrix(alfred_artifact.content, spec.series_id)

    extraction_arguments = {
        "series_id": spec.series_id,
        "component": spec.component,
        "transform": spec.transform,
        "max_release_lag_days": 92,
    }
    fred_features = extract_first_release_features(
        fred_matrix, **extraction_arguments
    )
    alfred_features = extract_first_release_features(
        alfred_matrix, **extraction_arguments
    )
    columns = [
        "reference_month",
        "release_date",
        "current_value",
        "previous_value_as_of_release",
        "transformed_value",
    ]
    fred_features = fred_features.loc[:, columns]
    alfred_features = alfred_features.loc[:, columns]
    merged = fred_features.merge(
        alfred_features,
        on="reference_month",
        how="outer",
        suffixes=("_fred", "_alfred"),
        indicator=True,
    )
    in_window = (
        (merged["reference_month"] >= pd.Timestamp(comparison_start))
        & (merged["reference_month"] <= pd.Timestamp(comparison_end))
    )
    merged = merged.loc[in_window].sort_values("reference_month")
    matched = merged.loc[merged["_merge"] == "both"].copy()
    release_delta = (
        pd.to_datetime(matched["release_date_fred"])
        - pd.to_datetime(matched["release_date_alfred"])
    ).dt.days.abs()

    feature_values = {
        field: _value_comparison(matched, field, atol=atol, rtol=rtol)
        for field in FIELDS
    }
    matrix_values = _matrix_comparison(
        fred_matrix,
        alfred_matrix,
        observation_start=observation_start,
        observation_end=comparison_end,
        atol=atol,
        rtol=rtol,
    )
    result: dict[str, Any] = {
        "series_id": spec.series_id,
        "transform": spec.transform,
        "fred_selected_vintages": len(fred_artifact.selected_vintage_dates),
        "alfred_selected_vintages": len(alfred_artifact.selected_vintage_dates),
        "comparison_months": int(len(merged)),
        "matched_months": int(len(matched)),
        "unmatched_months": int((merged["_merge"] != "both").sum()),
        "release_date_mismatches": int((release_delta != 0).sum()),
        "maximum_release_date_difference_days": (
            int(release_delta.max()) if len(release_delta) else 0
        ),
        "features": feature_values,
        "matrix": matrix_values,
    }
    result["passed"] = bool(
        result["comparison_months"] > 0
        and result["matched_months"] == result["comparison_months"]
        and result["unmatched_months"] == 0
        and result["release_date_mismatches"] == 0
        and all(item["mismatches"] == 0 for item in feature_values.values())
        and matrix_values["fred_vintages_missing_from_alfred"] == 0
        and matrix_values["cells_compared"] > 0
        and matrix_values["cell_mismatches"] == 0
    )
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FRED API versus ALFRED parity check",
        "",
        f"Overall result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        (
            f"Window: {report['comparison_start']} through "
            f"{report['comparison_end']}"
        ),
        "",
        "| Series | Months | Release mismatches | Feature mismatches | "
        "Matrix-cell mismatches | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in report["series"]:
        feature_mismatches = sum(
            value["mismatches"] for value in item["features"].values()
        )
        lines.append(
            f"| `{item['series_id']}` | {item['comparison_months']} | "
            f"{item['release_date_mismatches']} | {feature_mismatches} | "
            f"{item['matrix']['cell_mismatches']} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "The report compares release dates, first-release current and prior "
            "levels, transformed features, and every common as-of matrix cell.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-start", type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument("--comparison-end", type=date.fromisoformat, default=date(2023, 12, 1))
    parser.add_argument("--observation-start", type=date.fromisoformat, default=date(2018, 1, 1))
    parser.add_argument("--vintage-start", type=date.fromisoformat, default=date(2018, 1, 1))
    parser.add_argument("--vintage-end", type=date.fromisoformat, default=date(2024, 3, 31))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/provider-parity"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.comparison_start > args.comparison_end:
        raise ValueError("comparison-start cannot follow comparison-end")
    api_key = os.environ.get("FRED_API_KEY", "")
    fred_client = FredApiDownloadClient(
        api_key,
        max_vintages_per_request=100,
        request_pause_seconds=0.2,
    )
    alfred_client = AlfredWebDownloadClient(
        max_vintages_per_request=12,
        request_pause_seconds=0.2,
    )
    with tempfile.TemporaryDirectory(prefix="fred-alfred-parity-") as temporary:
        work_dir = Path(temporary)
        series_results = [
            _compare_series(
                spec,
                fred_client=fred_client,
                alfred_client=alfred_client,
                work_dir=work_dir,
                observation_start=args.observation_start,
                comparison_start=args.comparison_start,
                comparison_end=args.comparison_end,
                vintage_start=args.vintage_start,
                vintage_end=args.vintage_end,
                atol=1e-12,
                rtol=1e-12,
            )
            for spec in SERIES
        ]
    report = {
        "schema_version": 1,
        "comparison_start": args.comparison_start.isoformat(),
        "comparison_end": args.comparison_end.isoformat(),
        "observation_start": args.observation_start.isoformat(),
        "vintage_start": args.vintage_start.isoformat(),
        "vintage_end": args.vintage_end.isoformat(),
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 1e-12,
        "series": series_results,
        "passed": all(item["passed"] for item in series_results),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "parity_report.json"
    markdown_path = args.output_dir / "parity_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    print(f"JSON report: {json_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

