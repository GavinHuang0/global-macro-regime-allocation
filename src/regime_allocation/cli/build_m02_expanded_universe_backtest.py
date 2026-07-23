"""Build the hash-pinned-posterior Model 02 all-local-ETF diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from regime_allocation.portfolio.m02_expanded_universe_backtest import (
    build_m02_expanded_universe_backtest,
)


DEFAULT_CONFIG = Path("configs/models/m02_expanded_universe_backtest.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    """Parse arguments and publish the isolated expanded-universe artifacts."""

    args = _parse_args()
    outputs = build_m02_expanded_universe_backtest(
        project_root=args.project_root,
        config_path=args.config,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
