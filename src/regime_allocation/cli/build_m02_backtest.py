"""Build Model 02's promoted-baseline weekly allocation and causal backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

from regime_allocation.portfolio.m02_pipeline import build_m02_allocation_backtest


DEFAULT_CONFIG = Path("configs/models/m02_regime_allocation_backtest.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    """Parse arguments and publish Model 02 weekly backtest artifacts."""

    args = _parse_args()
    root = args.project_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    outputs = build_m02_allocation_backtest(
        project_root=root,
        config_path=config.resolve(),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
