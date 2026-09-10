"""Acquire current inputs, replay M02, and publish a dated README research snapshot."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from regime_allocation.data.m02_live_inputs import build_live_inputs
from regime_allocation.data.providers.fred_api import FredApiDownloadClient
from regime_allocation.portfolio.m02_live import (
    compute_live_allocation,
    publish_live_files,
    signal_calendar,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--signal-week",
        type=date.fromisoformat,
        help="Monday to reconstruct; defaults to this week in New York",
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh cached macro responses")
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Keep artifacts in outputs; do not update README or results/live",
    )
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    now = datetime.now(UTC)
    monday, price_day = signal_calendar(now)
    week = args.signal_week or monday
    if week.weekday() != 0 or week > monday:
        parser.error("--signal-week must be a Monday no later than this week")
    if week != monday and not args.compute_only:
        parser.error(
            "Historical weeks require --compute-only; they cannot replace the current README"
        )
    # Validate the key before downloads/writes. Never print environment values.
    FredApiDownloadClient(os.environ.get("FRED_API_KEY", ""))
    work = root / "outputs/m02_weekly_live" / week.isoformat()
    inputs = build_live_inputs(
        project_root=root,
        output_dir=work / "inputs",
        as_of=week - timedelta(days=1),
        refresh=args.refresh,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/download_etf_history.py"),
            "--start",
            "2008-01-01",
            "--as-of",
            price_day.isoformat(),
            "--raw-dir",
            str(work / "market/raw"),
            "--processed-dir",
            str(work / "market/processed"),
            "--manifest",
            str(work / "market/manifest.json"),
        ],
        check=True,
        cwd=root,
    )
    artifacts = compute_live_allocation(
        project_root=root,
        input_paths=inputs,
        prices_path=work / "market/processed/etf_daily_prices_long.csv",
        output_dir=work / "allocation",
        signal_week=week,
        now=now,
    )
    if not args.compute_only:
        publish_live_files(root, artifacts, now=datetime.now(UTC))
        print(
            f"Published M02 research snapshot for {week} to README.md and results/live/m02_weekly"
        )
    else:
        print(f"Computed M02 research snapshot in {work / 'allocation'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
