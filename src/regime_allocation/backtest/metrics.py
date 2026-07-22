"""Summarize and compare monthly regime-allocation backtests.

Inputs are transaction-cost-aware monthly method returns and, where needed,
phase-ordered daily NAV checkpoints. Outputs include cumulative and annualized
return, volatility, Sharpe ratio, maximum drawdown, turnover, and cost totals.
The paired circular block bootstrap resamples the same month indices for both
methods, preserving contemporaneous dependence while estimating uncertainty in
their annualized return difference. These functions evaluate supplied results;
they do not alter execution timing or portfolio weights.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Sequence

import numpy as np
import pandas as pd


_MONTHLY_COLUMNS = {
    "method",
    "reference_month",
    "gross_return",
    "net_return",
    "transaction_cost_rate",
    "gross_turnover",
    "one_way_turnover",
}
_NAV_COLUMNS = {"method", "date", "phase_order", "nav"}
_PAIRED_COMPARISON_COLUMNS = {"method", "reference_month", "net_return"}
_DEFAULT_PERIODS_PER_YEAR = 12


def _geometric_annual_return(returns: np.ndarray, periods_per_year: int) -> float:
    if len(returns) == 0:
        return float("nan")
    total_factor = float(np.prod(1.0 + returns))
    if total_factor <= 0:
        return -1.0
    return total_factor ** (periods_per_year / len(returns)) - 1.0


def _max_drawdown(nav: np.ndarray) -> float:
    if len(nav) == 0:
        return float("nan")
    peaks = np.maximum.accumulate(nav)
    return float(np.min(nav / peaks - 1.0))


def compute_performance_metrics(
    monthly_simulation: pd.DataFrame,
    daily_nav: pd.DataFrame,
    *,
    bil_monthly_returns: pd.Series | None = None,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Compute comparable gross/net, risk, drawdown, and trading metrics."""

    missing_monthly = _MONTHLY_COLUMNS.difference(monthly_simulation.columns)
    if missing_monthly:
        raise ValueError(
            f"monthly simulation is missing columns: {', '.join(sorted(missing_monthly))}"
        )
    missing_nav = _NAV_COLUMNS.difference(daily_nav.columns)
    if missing_nav:
        raise ValueError(f"daily NAV is missing columns: {', '.join(sorted(missing_nav))}")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be strictly positive")
    bil = None
    if bil_monthly_returns is not None:
        bil = bil_monthly_returns.copy()
        bil.index = pd.to_datetime(bil.index)

    records: list[dict[str, object]] = []
    for method, frame in monthly_simulation.groupby("method", sort=False):
        frame = frame.sort_values("reference_month")
        net = frame["net_return"].to_numpy(dtype=float)
        gross = frame["gross_return"].to_numpy(dtype=float)
        if not np.isfinite(net).all() or not np.isfinite(gross).all() or (net <= -1).any():
            raise ValueError(f"{method} contains invalid monthly returns")
        annual_volatility = float(np.std(net, ddof=1) * math.sqrt(periods_per_year))
        annual_mean = float(np.mean(net) * periods_per_year)
        zero_sharpe = annual_mean / annual_volatility if annual_volatility > 0 else float("nan")
        downside = np.minimum(net, 0.0)
        downside_deviation = float(
            math.sqrt(float(np.mean(np.square(downside)))) * math.sqrt(periods_per_year)
        )
        sortino = annual_mean / downside_deviation if downside_deviation > 0 else float("nan")
        method_nav = daily_nav.loc[daily_nav["method"].eq(method)].sort_values(
            ["date", "phase_order"]
        )["nav"].to_numpy(dtype=float)
        drawdown = _max_drawdown(method_nav)
        cagr = _geometric_annual_return(net, periods_per_year)
        gross_cagr = _geometric_annual_return(gross, periods_per_year)
        bil_excess_sharpe = float("nan")
        if bil is not None:
            reference_months = pd.to_datetime(frame["reference_month"])
            aligned_bil = bil.reindex(reference_months).to_numpy(dtype=float)
            if np.isfinite(aligned_bil).all():
                active = net - aligned_bil
                active_volatility = float(np.std(active, ddof=1) * math.sqrt(periods_per_year))
                if active_volatility > 0:
                    bil_excess_sharpe = float(
                        np.mean(active) * periods_per_year / active_volatility
                    )
        records.append(
            {
                "method": method,
                "start_reference_month": pd.Timestamp(frame["reference_month"].min()),
                "end_reference_month": pd.Timestamp(frame["reference_month"].max()),
                "months": int(len(frame)),
                "total_return": float(np.prod(1.0 + net) - 1.0),
                "cagr": cagr,
                "gross_cagr": gross_cagr,
                "annualized_cost_drag": gross_cagr - cagr,
                "annualized_volatility": annual_volatility,
                "sharpe_zero_rate": zero_sharpe,
                "sharpe_excess_bil": bil_excess_sharpe,
                "sortino_zero_rate": sortino,
                "maximum_drawdown": drawdown,
                "calmar_ratio": cagr / abs(drawdown) if drawdown < 0 else float("nan"),
                "worst_month": float(np.min(net)),
                "best_month": float(np.max(net)),
                "positive_month_fraction": float(np.mean(net > 0)),
                "average_one_way_turnover": float(frame["one_way_turnover"].mean()),
                "annualized_one_way_turnover": float(
                    frame["one_way_turnover"].mean() * periods_per_year
                ),
                "maximum_one_way_turnover": float(frame["one_way_turnover"].max()),
                "sum_transaction_cost_rates": float(frame["transaction_cost_rate"].sum()),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("method").reset_index(drop=True)


def _ordered_method_returns(
    monthly_simulation: pd.DataFrame,
    *,
    method: str,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frame = monthly_simulation.loc[
        monthly_simulation["method"].eq(method), ["reference_month", "net_return"]
    ].copy()
    if frame.empty:
        raise ValueError(f"monthly simulation does not contain method {method!r}")
    frame["reference_month"] = pd.to_datetime(frame["reference_month"], errors="coerce")
    if frame["reference_month"].isna().any():
        raise ValueError(f"{method} contains invalid reference months")
    if frame["reference_month"].duplicated().any():
        raise ValueError(f"{method} contains duplicate reference months")
    months = pd.DatetimeIndex(frame["reference_month"])
    if not months.is_monotonic_increasing:
        raise ValueError(f"{method} reference months must be strictly increasing")
    net_returns = pd.to_numeric(frame["net_return"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(net_returns).all() or (net_returns <= -1.0).any():
        raise ValueError(f"{method} contains invalid monthly net returns")
    return months, net_returns


def _circular_block_indices(
    *,
    observations: int,
    block_length: int,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    """Return circular moving-block sample indices, truncated to sample length."""

    blocks_per_resample = math.ceil(observations / block_length)
    generator = np.random.default_rng(seed)
    starts = generator.integers(
        0,
        observations,
        size=(n_resamples, blocks_per_resample),
        endpoint=False,
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % observations
    return indices.reshape(n_resamples, -1)[:, :observations]


def paired_circular_block_bootstrap(
    monthly_simulation: pd.DataFrame,
    *,
    baseline_method: str,
    comparator_methods: Sequence[str],
    block_length: int = 6,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Compare a baseline with other methods using paired circular blocks.

    For comparator ``j``, the paired monthly difference is

    ``baseline net return - comparator net return``.

    A positive estimate therefore favors the baseline.  The function samples
    consecutive circular blocks of paired differences, concatenates enough
    blocks to recover the original sample length, and reports a percentile
    confidence interval for the annualized arithmetic mean difference.  The
    same deterministic bootstrap index matrix is applied to every comparator.
    This preserves paired timing, permits cross-comparator comparisons, and
    makes a result independent of the order in ``comparator_methods``.

    The t-statistic is intentionally labeled naive: it treats monthly paired
    differences as independent and is retained only as a familiar diagnostic.
    The block-bootstrap interval is the dependence-aware headline result.
    """

    missing = _PAIRED_COMPARISON_COLUMNS.difference(monthly_simulation.columns)
    if missing:
        raise ValueError(
            "monthly simulation is missing columns: " + ", ".join(sorted(missing))
        )
    if not isinstance(baseline_method, str) or not baseline_method:
        raise ValueError("baseline_method must be a non-empty string")
    comparators = tuple(comparator_methods)
    if not comparators:
        raise ValueError("comparator_methods cannot be empty")
    if any(not isinstance(method, str) or not method for method in comparators):
        raise ValueError("comparator_methods must contain non-empty strings")
    if len(set(comparators)) != len(comparators):
        raise ValueError("comparator_methods must be unique")
    if baseline_method in comparators:
        raise ValueError("baseline_method cannot also be a comparator")
    if isinstance(block_length, bool) or not isinstance(block_length, Integral):
        raise ValueError("block_length must be an integer")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, Integral):
        raise ValueError("n_resamples must be an integer")
    if block_length <= 0:
        raise ValueError("block_length must be strictly positive")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be strictly positive")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, Integral):
        raise ValueError("periods_per_year must be an integer")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be strictly positive")

    baseline_months, baseline_returns = _ordered_method_returns(
        monthly_simulation,
        method=baseline_method,
    )
    observations = len(baseline_returns)
    if observations < 2:
        raise ValueError("paired comparison requires at least two monthly observations")
    if block_length > observations:
        raise ValueError("block_length cannot exceed the number of paired months")

    comparator_returns: dict[str, np.ndarray] = {}
    for comparator in comparators:
        months, returns = _ordered_method_returns(monthly_simulation, method=comparator)
        if not months.equals(baseline_months):
            raise ValueError(
                f"{comparator} must have identical ordered reference months to "
                f"{baseline_method}"
            )
        comparator_returns[comparator] = returns

    sample_indices = _circular_block_indices(
        observations=observations,
        block_length=int(block_length),
        n_resamples=int(n_resamples),
        seed=int(seed),
    )
    tail_probability = (1.0 - confidence_level) / 2.0
    records: list[dict[str, object]] = []
    for comparator in comparators:
        differences = baseline_returns - comparator_returns[comparator]
        mean_monthly_difference = float(np.mean(differences))
        annualized_mean_difference = periods_per_year * mean_monthly_difference
        bootstrap_annualized_means = (
            differences[sample_indices].mean(axis=1) * periods_per_year
        )
        lower, upper = np.quantile(
            bootstrap_annualized_means,
            [tail_probability, 1.0 - tail_probability],
        )
        monthly_tracking_error = float(np.std(differences, ddof=1))
        zero_dispersion_tolerance = float(
            100.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(differences))))
        )
        if monthly_tracking_error <= zero_dispersion_tolerance:
            monthly_tracking_error = 0.0
        annualized_tracking_error = monthly_tracking_error * math.sqrt(periods_per_year)
        if monthly_tracking_error > 0.0:
            naive_t = float(
                mean_monthly_difference
                / (monthly_tracking_error / math.sqrt(observations))
            )
            information_ratio = annualized_mean_difference / annualized_tracking_error
        else:
            naive_t = float("nan")
            information_ratio = float("nan")
        records.append(
            {
                "baseline_method": baseline_method,
                "comparator_method": comparator,
                "start_reference_month": pd.Timestamp(baseline_months[0]),
                "end_reference_month": pd.Timestamp(baseline_months[-1]),
                "months": observations,
                "block_length": int(block_length),
                "n_resamples": int(n_resamples),
                "confidence_level": float(confidence_level),
                "seed": int(seed),
                "difference_definition": "baseline_minus_comparator_net_return",
                "annualized_mean_difference": annualized_mean_difference,
                "annualized_mean_difference_ci_lower": float(lower),
                "annualized_mean_difference_ci_upper": float(upper),
                "probability_mean_difference_positive": float(
                    np.mean(bootstrap_annualized_means > 0.0)
                ),
                "naive_paired_t_statistic": naive_t,
                "annualized_tracking_error": annualized_tracking_error,
                "information_ratio": information_ratio,
            }
        )
    return pd.DataFrame.from_records(records)
