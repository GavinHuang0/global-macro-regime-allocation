"""Causal first-session-open portfolio accounting.

Targets are formed before the first executable session of a reference month or
Monday-anchored week, traded at that session's adjusted open, and held until
the next period's first-session adjusted open. Cash distributions and splits
are already represented in the provider's adjusted price series and must not
be added a second time. Inputs are validated adjusted prices, target weights,
pre-trade holdings, and one-way costs; outputs are open-to-open holding returns,
drifted weights, transaction-cost-aware returns, and phase-ordered daily NAV. A
pre-trade NAV point ensures the initial rebalance cost enters drawdown
calculations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


_PRICE_COLUMNS = {"date", "ticker", "adjusted_open", "adjusted_close"}
_TARGET_COLUMNS = {"method", "reference_month", "ticker", "target_weight"}
_WEEKLY_TARGET_COLUMNS = {"method", "reference_week", "ticker", "target_weight"}


def _normalize_prices(prices: pd.DataFrame, assets: Sequence[str]) -> pd.DataFrame:
    missing = _PRICE_COLUMNS.difference(prices.columns)
    if missing:
        raise ValueError(f"price data are missing columns: {', '.join(sorted(missing))}")
    normalized = prices.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    if normalized["date"].isna().any():
        raise ValueError("price dates contain invalid values")
    normalized["ticker"] = normalized["ticker"].astype(str)
    requested = tuple(map(str, assets))
    absent = sorted(set(requested).difference(normalized["ticker"].unique()))
    if absent:
        raise ValueError(f"price data are missing requested assets: {absent}")
    normalized = normalized.loc[normalized["ticker"].isin(requested)].copy()
    for column in ("adjusted_open", "adjusted_close"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if normalized[column].isna().any() or (normalized[column] <= 0).any():
            raise ValueError(f"{column} must be finite and strictly positive")
    if normalized.duplicated(["date", "ticker"]).any():
        raise ValueError("price data contain duplicate ticker-date rows")
    counts = normalized.groupby("date")["ticker"].nunique()
    incomplete_dates = counts.loc[counts.ne(len(requested))]
    if not incomplete_dates.empty:
        raise ValueError("assets do not share a complete common daily calendar")
    return normalized.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def build_open_to_open_holding_returns(
    prices: pd.DataFrame,
    *,
    assets: Sequence[str],
) -> pd.DataFrame:
    """Return one adjusted-open holding-period row for every complete month."""

    requested = tuple(map(str, assets))
    normalized = _normalize_prices(prices, requested)
    opens = normalized.pivot(index="date", columns="ticker", values="adjusted_open")
    opens = opens.reindex(columns=requested).sort_index()
    first_session = (
        pd.Series(opens.index, index=opens.index.to_period("M"))
        .groupby(level=0)
        .min()
        .sort_index()
    )
    records: list[dict[str, object]] = []
    for position in range(max(len(first_session) - 1, 0)):
        reference_period = first_session.index[position]
        next_period = first_session.index[position + 1]
        if next_period != reference_period + 1:
            continue
        start_date = pd.Timestamp(first_session.iloc[position])
        end_date = pd.Timestamp(first_session.iloc[position + 1])
        start = opens.loc[start_date].to_numpy(dtype=float)
        end = opens.loc[end_date].to_numpy(dtype=float)
        returns = end / start - 1.0
        record: dict[str, object] = {
            "reference_month": reference_period.to_timestamp(),
            "start_date": start_date,
            "end_date": end_date,
            "trading_sessions": int(((opens.index >= start_date) & (opens.index < end_date)).sum()),
        }
        record.update(
            {
                asset: float(value)
                for asset, value in zip(requested, returns, strict=True)
            }
        )
        records.append(record)
    output = pd.DataFrame.from_records(records)
    if output.empty:
        raise ValueError("price data do not contain a complete open-to-open month")
    values = output.loc[:, list(requested)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("holding-period returns must be finite and greater than -100%")
    return output.sort_values("reference_month").reset_index(drop=True)


def build_weekly_open_to_open_holding_returns(
    prices: pd.DataFrame,
    *,
    assets: Sequence[str],
) -> pd.DataFrame:
    """Return adjusted-open holding periods for complete Monday-anchored weeks.

    Each ``reference_week`` is the calendar Monday that anchors the week. The
    entry is the first common trading session in that week, so a Monday market
    holiday moves execution to Tuesday without changing the reference anchor.
    A week is emitted only after the following week's first common open exists.
    """

    requested = tuple(map(str, assets))
    normalized = _normalize_prices(prices, requested)
    opens = normalized.pivot(index="date", columns="ticker", values="adjusted_open")
    opens = opens.reindex(columns=requested).sort_index()
    first_session = (
        pd.Series(opens.index, index=opens.index.to_period("W-SUN"))
        .groupby(level=0)
        .min()
        .sort_index()
    )
    records: list[dict[str, object]] = []
    for position in range(max(len(first_session) - 1, 0)):
        reference_period = first_session.index[position]
        next_period = first_session.index[position + 1]
        if next_period != reference_period + 1:
            continue
        start_date = pd.Timestamp(first_session.iloc[position])
        end_date = pd.Timestamp(first_session.iloc[position + 1])
        start = opens.loc[start_date].to_numpy(dtype=float)
        end = opens.loc[end_date].to_numpy(dtype=float)
        returns = end / start - 1.0
        record: dict[str, object] = {
            "reference_week": pd.Timestamp(reference_period.start_time),
            "start_date": start_date,
            "end_date": end_date,
            "trading_sessions": int(
                ((opens.index >= start_date) & (opens.index < end_date)).sum()
            ),
        }
        record.update(
            {
                asset: float(value)
                for asset, value in zip(requested, returns, strict=True)
            }
        )
        records.append(record)
    output = pd.DataFrame.from_records(records)
    if output.empty:
        raise ValueError("price data do not contain a complete open-to-open week")
    values = output.loc[:, list(requested)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("holding-period returns must be finite and greater than -100%")
    return output.sort_values("reference_week").reset_index(drop=True)


def drift_weights(weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    """Drift beginning weights through a vector of simple asset returns."""

    weights_array = np.asarray(weights, dtype=float)
    returns_array = np.asarray(asset_returns, dtype=float)
    if weights_array.ndim != 1 or returns_array.shape != weights_array.shape:
        raise ValueError("weights and returns must be same-length one-dimensional arrays")
    if not np.isfinite(weights_array).all() or not np.isfinite(returns_array).all():
        raise ValueError("weights and returns must be finite")
    if (weights_array < -1e-12).any() or not np.isclose(weights_array.sum(), 1.0, atol=1e-10):
        raise ValueError("beginning weights must be long-only and sum to one")
    growth = weights_array * (1.0 + returns_array)
    total = float(growth.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("portfolio gross value must remain strictly positive")
    return growth / total


def _normalize_targets(
    targets: pd.DataFrame,
    *,
    assets: Sequence[str],
) -> pd.DataFrame:
    missing = _TARGET_COLUMNS.difference(targets.columns)
    if missing:
        raise ValueError(f"target weights are missing columns: {', '.join(sorted(missing))}")
    normalized = targets.copy()
    normalized["method"] = normalized["method"].astype(str)
    normalized["ticker"] = normalized["ticker"].astype(str)
    normalized["reference_month"] = pd.to_datetime(
        normalized["reference_month"], errors="coerce"
    )
    if normalized["reference_month"].isna().any():
        raise ValueError("target reference months contain invalid values")
    if (normalized["reference_month"].dt.day != 1).any():
        raise ValueError("target reference months must be normalized to month start")
    normalized["target_weight"] = pd.to_numeric(normalized["target_weight"], errors="coerce")
    if normalized["target_weight"].isna().any() or (normalized["target_weight"] < -1e-12).any():
        raise ValueError("target weights must be finite and non-negative")
    if normalized.duplicated(["method", "reference_month", "ticker"]).any():
        raise ValueError("target weights contain duplicate method-month-ticker rows")
    unknown = sorted(set(normalized["ticker"]).difference(map(str, assets)))
    if unknown:
        raise ValueError(f"targets contain assets outside the simulation universe: {unknown}")
    totals = normalized.groupby(["method", "reference_month"])["target_weight"].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0, atol=1e-9, rtol=1e-9):
        raise ValueError("each method-month target must sum to one")
    return normalized.sort_values(["method", "reference_month", "ticker"], kind="stable")


def _normalize_weekly_targets(
    targets: pd.DataFrame,
    *,
    assets: Sequence[str],
) -> pd.DataFrame:
    missing = _WEEKLY_TARGET_COLUMNS.difference(targets.columns)
    if missing:
        raise ValueError(f"weekly target weights are missing columns: {', '.join(sorted(missing))}")
    normalized = targets.copy()
    normalized["method"] = normalized["method"].astype(str)
    normalized["ticker"] = normalized["ticker"].astype(str)
    normalized["reference_week"] = pd.to_datetime(
        normalized["reference_week"], errors="coerce"
    )
    if normalized["reference_week"].isna().any():
        raise ValueError("target reference weeks contain invalid values")
    if normalized["reference_week"].dt.dayofweek.ne(0).any():
        raise ValueError("target reference weeks must be normalized to Monday")
    normalized["target_weight"] = pd.to_numeric(
        normalized["target_weight"], errors="coerce"
    )
    if (
        normalized["target_weight"].isna().any()
        or (normalized["target_weight"] < -1e-12).any()
    ):
        raise ValueError("target weights must be finite and non-negative")
    if normalized.duplicated(["method", "reference_week", "ticker"]).any():
        raise ValueError("target weights contain duplicate method-week-ticker rows")
    unknown = sorted(set(normalized["ticker"]).difference(map(str, assets)))
    if unknown:
        raise ValueError(f"targets contain assets outside the simulation universe: {unknown}")
    totals = normalized.groupby(["method", "reference_week"])["target_weight"].sum()
    if not np.allclose(totals.to_numpy(dtype=float), 1.0, atol=1e-9, rtol=1e-9):
        raise ValueError("each method-week target must sum to one")
    return normalized.sort_values(["method", "reference_week", "ticker"], kind="stable")


def _cost_vector(
    assets: Sequence[str],
    transaction_costs: float | Mapping[str, float],
) -> np.ndarray:
    if isinstance(transaction_costs, Mapping):
        missing = set(assets).difference(transaction_costs)
        if missing:
            raise ValueError(f"missing transaction costs for assets: {sorted(missing)}")
        values = np.array([transaction_costs[asset] for asset in assets], dtype=float)
    else:
        values = np.full(len(assets), float(transaction_costs), dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("transaction costs must be finite and non-negative")
    return values


def simulate_monthly_targets(
    targets: pd.DataFrame,
    holding_returns: pd.DataFrame,
    *,
    assets: Sequence[str],
    transaction_costs: float | Mapping[str, float],
) -> pd.DataFrame:
    """Apply monthly targets, drift, and one-way per-asset trading costs."""

    universe = tuple(map(str, assets))
    normalized_targets = _normalize_targets(targets, assets=universe)
    required_returns = {"reference_month", "start_date", "end_date", *universe}
    missing = required_returns.difference(holding_returns.columns)
    if missing:
        raise ValueError(f"holding returns are missing columns: {', '.join(sorted(missing))}")
    returns = holding_returns.copy()
    for column in ("reference_month", "start_date", "end_date"):
        returns[column] = pd.to_datetime(returns[column], errors="coerce")
    if returns[list(required_returns)].isna().any().any():
        raise ValueError("holding returns contain missing required values")
    returns = returns.set_index("reference_month").sort_index()
    if not returns.index.is_unique:
        raise ValueError("holding returns contain duplicate reference months")
    costs = _cost_vector(universe, transaction_costs)

    records: list[dict[str, object]] = []
    for method, method_targets in normalized_targets.groupby("method", sort=False):
        pivot = method_targets.pivot(
            index="reference_month", columns="ticker", values="target_weight"
        ).reindex(columns=universe, fill_value=0.0)
        missing_months = pivot.index.difference(returns.index)
        if len(missing_months):
            raise ValueError(
                f"{method} targets lack complete holding returns: {list(missing_months)}"
            )
        previous_target: np.ndarray | None = None
        previous_returns: np.ndarray | None = None
        nav = 1.0
        for reference_month, target_row in pivot.sort_index().iterrows():
            target = target_row.to_numpy(dtype=float)
            if previous_target is None:
                pretrade = np.zeros(len(universe), dtype=float)
            else:
                assert previous_returns is not None
                pretrade = drift_weights(previous_target, previous_returns)
            trades = target - pretrade
            cost_rate = float(np.dot(costs, np.abs(trades)))
            if not 0 <= cost_rate < 1:
                raise RuntimeError("transaction cost rate must remain in [0, 1)")
            period = returns.loc[reference_month]
            asset_returns = period.loc[list(universe)].to_numpy(dtype=float)
            gross_return = float(np.dot(target, asset_returns))
            net_return = (1.0 - cost_rate) * (1.0 + gross_return) - 1.0
            gross_nav_before_cost = nav
            nav *= 1.0 + net_return
            records.append(
                {
                    "method": method,
                    "reference_month": reference_month,
                    "start_date": pd.Timestamp(period["start_date"]),
                    "end_date": pd.Timestamp(period["end_date"]),
                    "gross_return": gross_return,
                    "transaction_cost_rate": cost_rate,
                    "net_return": net_return,
                    "gross_turnover": float(np.abs(trades).sum()),
                    "one_way_turnover": float(0.5 * np.abs(trades).sum()),
                    "nav_before_period": gross_nav_before_cost,
                    "nav_after_period": nav,
                }
            )
            previous_target = target
            previous_returns = asset_returns
    output = pd.DataFrame.from_records(records)
    if output.empty:
        raise ValueError("no target returns were simulated")
    return output.sort_values(["method", "reference_month"], kind="stable").reset_index(drop=True)


def simulate_weekly_targets(
    targets: pd.DataFrame,
    holding_returns: pd.DataFrame,
    *,
    assets: Sequence[str],
    transaction_costs: float | Mapping[str, float],
) -> pd.DataFrame:
    """Apply weekly targets, drift, and one-way per-asset trading costs."""

    universe = tuple(map(str, assets))
    normalized_targets = _normalize_weekly_targets(targets, assets=universe)
    required_returns = {"reference_week", "start_date", "end_date", *universe}
    missing = required_returns.difference(holding_returns.columns)
    if missing:
        raise ValueError(f"holding returns are missing columns: {', '.join(sorted(missing))}")
    returns = holding_returns.copy()
    for column in ("reference_week", "start_date", "end_date"):
        returns[column] = pd.to_datetime(returns[column], errors="coerce")
    if returns[list(required_returns)].isna().any().any():
        raise ValueError("holding returns contain missing required values")
    returns = returns.set_index("reference_week").sort_index()
    if not returns.index.is_unique:
        raise ValueError("holding returns contain duplicate reference weeks")
    costs = _cost_vector(universe, transaction_costs)

    records: list[dict[str, object]] = []
    for method, method_targets in normalized_targets.groupby("method", sort=False):
        pivot = method_targets.pivot(
            index="reference_week", columns="ticker", values="target_weight"
        ).reindex(columns=universe, fill_value=0.0)
        missing_weeks = pivot.index.difference(returns.index)
        if len(missing_weeks):
            raise ValueError(
                f"{method} targets lack complete holding returns: {list(missing_weeks)}"
            )
        ordered_weeks = pivot.index.sort_values()
        if len(ordered_weeks) > 1 and not (
            ordered_weeks[1:] - ordered_weeks[:-1] == pd.Timedelta(days=7)
        ).all():
            raise ValueError(f"{method} targets must contain consecutive calendar weeks")
        previous_target: np.ndarray | None = None
        previous_returns: np.ndarray | None = None
        nav = 1.0
        for reference_week, target_row in pivot.sort_index().iterrows():
            target = target_row.to_numpy(dtype=float)
            if previous_target is None:
                pretrade = np.zeros(len(universe), dtype=float)
            else:
                assert previous_returns is not None
                pretrade = drift_weights(previous_target, previous_returns)
            trades = target - pretrade
            cost_rate = float(np.dot(costs, np.abs(trades)))
            if not 0 <= cost_rate < 1:
                raise RuntimeError("transaction cost rate must remain in [0, 1)")
            period = returns.loc[reference_week]
            asset_returns = period.loc[list(universe)].to_numpy(dtype=float)
            gross_return = float(np.dot(target, asset_returns))
            net_return = (1.0 - cost_rate) * (1.0 + gross_return) - 1.0
            gross_nav_before_cost = nav
            nav *= 1.0 + net_return
            records.append(
                {
                    "method": method,
                    "reference_week": reference_week,
                    "start_date": pd.Timestamp(period["start_date"]),
                    "end_date": pd.Timestamp(period["end_date"]),
                    "gross_return": gross_return,
                    "transaction_cost_rate": cost_rate,
                    "net_return": net_return,
                    "gross_turnover": float(np.abs(trades).sum()),
                    "one_way_turnover": float(0.5 * np.abs(trades).sum()),
                    "nav_before_period": gross_nav_before_cost,
                    "nav_after_period": nav,
                }
            )
            previous_target = target
            previous_returns = asset_returns
    output = pd.DataFrame.from_records(records)
    if output.empty:
        raise ValueError("no target returns were simulated")
    return output.sort_values(["method", "reference_week"], kind="stable").reset_index(
        drop=True
    )


def build_daily_nav(
    targets: pd.DataFrame,
    monthly_simulation: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    assets: Sequence[str],
) -> pd.DataFrame:
    """Create open/close NAV checkpoints consistent with monthly accounting."""

    universe = tuple(map(str, assets))
    normalized_targets = _normalize_targets(targets, assets=universe)
    normalized_prices = _normalize_prices(prices, universe)
    opens = normalized_prices.pivot(index="date", columns="ticker", values="adjusted_open")
    closes = normalized_prices.pivot(index="date", columns="ticker", values="adjusted_close")
    opens = opens.reindex(columns=universe).sort_index()
    closes = closes.reindex(columns=universe).sort_index()
    simulations = monthly_simulation.copy()
    simulations["reference_month"] = pd.to_datetime(simulations["reference_month"])
    simulations = simulations.set_index(["method", "reference_month"])
    if not simulations.index.is_unique:
        raise ValueError("monthly simulation contains duplicate method-month rows")

    records: list[dict[str, object]] = []
    for method, method_targets in normalized_targets.groupby("method", sort=False):
        pivot = method_targets.pivot(
            index="reference_month", columns="ticker", values="target_weight"
        ).reindex(columns=universe, fill_value=0.0).sort_index()
        capital_before_cost = 1.0
        for reference_month, target_row in pivot.iterrows():
            simulation = simulations.loc[(method, reference_month)]
            start_date = pd.Timestamp(simulation["start_date"])
            end_date = pd.Timestamp(simulation["end_date"])
            cost_rate = float(simulation["transaction_cost_rate"])
            capital_after_cost = capital_before_cost * (1.0 - cost_rate)
            records.append(
                {
                    "method": method,
                    "date": start_date,
                    "phase_order": 0,
                    "phase": "pre_trade_open",
                    "reference_month": reference_month,
                    "nav": capital_before_cost,
                }
            )
            records.append(
                {
                    "method": method,
                    "date": start_date,
                    "phase_order": 1,
                    "phase": "post_trade_open",
                    "reference_month": reference_month,
                    "nav": capital_after_cost,
                }
            )
            start_open = opens.loc[start_date].to_numpy(dtype=float)
            weights = target_row.to_numpy(dtype=float)
            session_dates = closes.index[(closes.index >= start_date) & (closes.index < end_date)]
            for session_date in session_dates:
                relatives = closes.loc[session_date].to_numpy(dtype=float) / start_open
                records.append(
                    {
                        "method": method,
                        "date": session_date,
                        "phase_order": 2,
                        "phase": "close",
                        "reference_month": reference_month,
                        "nav": capital_after_cost * float(np.dot(weights, relatives)),
                    }
                )
            end_relatives = opens.loc[end_date].to_numpy(dtype=float) / start_open
            capital_before_cost = capital_after_cost * float(np.dot(weights, end_relatives))
        records.append(
            {
                "method": method,
                "date": end_date,
                "phase_order": 0,
                "phase": "terminal_open",
                "reference_month": (
                    pd.Timestamp(reference_month).to_period("M") + 1
                ).to_timestamp(),
                "nav": capital_before_cost,
            }
        )
    output = pd.DataFrame.from_records(records)
    if output.empty or output["nav"].isna().any() or (output["nav"] <= 0).any():
        raise RuntimeError("daily NAV generation produced invalid values")
    return output.sort_values(
        ["method", "date", "phase_order"], kind="stable"
    ).reset_index(drop=True)


def build_weekly_daily_nav(
    targets: pd.DataFrame,
    weekly_simulation: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    assets: Sequence[str],
) -> pd.DataFrame:
    """Create open/close NAV checkpoints consistent with weekly accounting."""

    universe = tuple(map(str, assets))
    normalized_targets = _normalize_weekly_targets(targets, assets=universe)
    normalized_prices = _normalize_prices(prices, universe)
    opens = normalized_prices.pivot(index="date", columns="ticker", values="adjusted_open")
    closes = normalized_prices.pivot(index="date", columns="ticker", values="adjusted_close")
    opens = opens.reindex(columns=universe).sort_index()
    closes = closes.reindex(columns=universe).sort_index()
    simulations = weekly_simulation.copy()
    simulations["reference_week"] = pd.to_datetime(simulations["reference_week"])
    simulations = simulations.set_index(["method", "reference_week"])
    if not simulations.index.is_unique:
        raise ValueError("weekly simulation contains duplicate method-week rows")

    records: list[dict[str, object]] = []
    for method, method_targets in normalized_targets.groupby("method", sort=False):
        pivot = method_targets.pivot(
            index="reference_week", columns="ticker", values="target_weight"
        ).reindex(columns=universe, fill_value=0.0).sort_index()
        capital_before_cost = 1.0
        for reference_week, target_row in pivot.iterrows():
            simulation = simulations.loc[(method, reference_week)]
            start_date = pd.Timestamp(simulation["start_date"])
            end_date = pd.Timestamp(simulation["end_date"])
            cost_rate = float(simulation["transaction_cost_rate"])
            capital_after_cost = capital_before_cost * (1.0 - cost_rate)
            records.append(
                {
                    "method": method,
                    "date": start_date,
                    "phase_order": 0,
                    "phase": "pre_trade_open",
                    "reference_week": reference_week,
                    "nav": capital_before_cost,
                }
            )
            records.append(
                {
                    "method": method,
                    "date": start_date,
                    "phase_order": 1,
                    "phase": "post_trade_open",
                    "reference_week": reference_week,
                    "nav": capital_after_cost,
                }
            )
            start_open = opens.loc[start_date].to_numpy(dtype=float)
            weights = target_row.to_numpy(dtype=float)
            session_dates = closes.index[(closes.index >= start_date) & (closes.index < end_date)]
            for session_date in session_dates:
                relatives = closes.loc[session_date].to_numpy(dtype=float) / start_open
                records.append(
                    {
                        "method": method,
                        "date": session_date,
                        "phase_order": 2,
                        "phase": "close",
                        "reference_week": reference_week,
                        "nav": capital_after_cost * float(np.dot(weights, relatives)),
                    }
                )
            end_relatives = opens.loc[end_date].to_numpy(dtype=float) / start_open
            capital_before_cost = capital_after_cost * float(np.dot(weights, end_relatives))
        records.append(
            {
                "method": method,
                "date": end_date,
                "phase_order": 0,
                "phase": "terminal_open",
                "reference_week": pd.Timestamp(reference_week) + pd.Timedelta(days=7),
                "nav": capital_before_cost,
            }
        )
    output = pd.DataFrame.from_records(records)
    if output.empty or output["nav"].isna().any() or (output["nav"] <= 0).any():
        raise RuntimeError("daily NAV generation produced invalid values")
    return output.sort_values(
        ["method", "date", "phase_order"], kind="stable"
    ).reset_index(drop=True)
