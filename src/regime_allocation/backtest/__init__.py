"""Shared portfolio backtesting utilities."""

from regime_allocation.backtest.engine import (
    build_daily_nav,
    build_open_to_open_holding_returns,
    drift_weights,
    simulate_monthly_targets,
)
from regime_allocation.backtest.metrics import (
    compute_performance_metrics,
    paired_circular_block_bootstrap,
)

__all__ = [
    "build_daily_nav",
    "build_open_to_open_holding_returns",
    "compute_performance_metrics",
    "drift_weights",
    "paired_circular_block_bootstrap",
    "simulate_monthly_targets",
]
