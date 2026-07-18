"""Build Model 01's posterior-weighted allocation and causal backtest.

The command reads the allocation YAML and delegates to the portfolio pipeline,
which combines archived month-start probabilities with expanding return
estimates, solves constrained long-only targets, executes them at the first
adjusted open, and writes weights, returns, NAV, metrics, comparisons,
sensitivities, and manifests. It performs no data download and never treats a
posterior published after the trading cutoff as executable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from regime_allocation.portfolio.pipeline import build_allocation_backtest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/m01_regime_allocation_backtest.yaml"),
        help="Allocation configuration path, relative to project root by default.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse arguments and publish allocation and backtest artifacts."""
    args = _parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    outputs = build_allocation_backtest(
        project_root=project_root,
        config_path=config_path.resolve(),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
