"""Preserve and audit the source inputs for the M03 data experiment."""

import argparse
import logging
from datetime import UTC, date, datetime
from pathlib import Path

from regime_allocation.data.m03_source_audit import build_source_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory under outputs/"
    )
    parser.add_argument(
        "--as-of",
        type=datetime.fromisoformat,
        default=None,
        help="ISO timestamp with timezone; default is current UTC time",
    )
    parser.add_argument("--reference-start", type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument("--reference-end", type=date.fromisoformat)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--import-m02-manifest", type=Path, help="Preserve existing M02 input bytes")
    mode.add_argument(
        "--replay-manifest", type=Path, help="Offline replay of preserved M03 snapshots"
    )
    parser.add_argument(
        "--previous-manifest", type=Path, help="Compare prior snapshots at same cutoff"
    )
    parser.add_argument(
        "--prices-dir", type=Path, help="Directory of existing Yahoo TICKER.json files"
    )
    parser.add_argument(
        "--series", nargs="+", dest="series_ids", help="Subset of registry series IDs"
    )
    parser.add_argument("--availability-mode", choices=("archive", "captured"), default="archive")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.as_of = args.as_of or datetime.now(UTC)
    try:
        manifest = build_source_audit(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Source audit failed: {exc}\n")
    print(f"Source audit: {manifest}")


if __name__ == "__main__":
    main()
