"""Pinned-posterior Model 02 backtest over every locally available ETF.

This exploratory stage does not rerun the Bayesian model.  It consumes the
published weekly posterior under pinned hashes; this pins the experiment's
input snapshot but does not freeze Model 02.  It verifies that the all-ETF
execution calendar matches the seven-ETF baseline, refits only causal return
moments for the enlarged opportunity set, and compares posterior, pooled,
equal-weight, and 60/40 strategies on their own transaction-cost paths.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from regime_allocation.backtest import (
    build_weekly_daily_nav,
    build_weekly_open_to_open_holding_returns,
    compute_performance_metrics,
    paired_circular_block_bootstrap,
    simulate_weekly_targets,
)
from regime_allocation.portfolio.baselines import (
    equal_weight_target,
    static_60_40_target,
)
from regime_allocation.portfolio.estimation import (
    CANONICAL_REGIME_IDS,
    PROBABILITY_COLUMNS,
)
from regime_allocation.portfolio.m02_pipeline import (
    PERIODS_PER_YEAR,
    _declaration,
    _dynamic_targets,
    _metric_view,
    _weekly_estimator_history,
    _weekly_metric_labels,
    _weight_records,
)
from regime_allocation.portfolio.m02_weekly import (
    build_weekly_execution_schedule,
    derive_m02_regime_history,
)
from regime_allocation.portfolio.optimization import GroupCap
from regime_allocation.portfolio.pipeline import (
    AllocationSpecification,
    BASELINE_METHOD,
    POOLED_METHOD,
    _write_csv,
    _write_json,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "frozen_posterior_expanded_universe_backtest"
EQUAL_METHOD = "equal_weight"
STATIC_METHOD = "static_60_spy_40_agg"
METHODS = (BASELINE_METHOD, POOLED_METHOD, EQUAL_METHOD, STATIC_METHOD)
EXPANDED_ASSETS = (
    "SPY",
    "IEF",
    "TIP",
    "LQD",
    "HYG",
    "BIL",
    "GLD",
    "DBC",
    "UUP",
    "TLT",
    "USO",
    "AGG",
)
ORIGINAL_ASSETS = ("SPY", "IEF", "TIP", "HYG", "BIL", "GLD", "LQD")
NEW_ASSETS = ("DBC", "UUP", "TLT", "USO", "AGG")
FROZEN_MANIFEST_SHA256 = (
    "5821824dcfe6a82347d1df176467b74d0199cb7724bb08f5e07daf08801fbc4a"
)
FROZEN_SIGNAL_SHA256 = (
    "8cb4d27056ec5b024ad5e2cbf4549429da98def25b8602e91275731e8046111f"
)
MARKET_MANIFEST_SHA256 = (
    "b806212f0db312f0315ae8119021612f17f211a3314be0d1966582d033556a69"
)
TRANSACTION_COST = 0.0005
VOLATILITY_CAP = 0.10
PSEUDO_WEEKS = 104.0
MINIMUM_LABELED_WEEKS = 260
EXPECTED_WEEK_COUNT = 445
EXPECTED_TARGET_COUNT = 446
SCHEDULE_COLUMNS = (
    "reference_week",
    "start_date",
    "end_date",
    "is_complete",
)
PERFORMANCE_COMPARISON_COLUMNS = (
    "total_return",
    "cagr",
    "gross_cagr",
    "annualized_cost_drag",
    "annualized_volatility",
    "sharpe_zero_rate",
    "sharpe_excess_bil",
    "sortino_zero_rate",
    "maximum_drawdown",
    "calmar_ratio",
    "worst_week",
    "positive_week_fraction",
    "annualized_one_way_turnover",
    "maximum_one_way_turnover",
    "sum_transaction_cost_rates",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _project_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _read_verified(
    root: Path, relative_path: object, expected_hash: object
) -> tuple[Path, bytes]:
    path = _project_path(root, relative_path)
    raw = path.read_bytes()
    if _sha256(raw) != str(expected_hash):
        raise ValueError(f"frozen source hash mismatch: {path.relative_to(root)}")
    return path, raw


def _manifest_declaration(
    manifest: Mapping[str, Any], *, path: Path, root: Path
) -> Mapping[str, Any]:
    relative = path.relative_to(root).as_posix()
    matches = [
        row
        for section in ("inputs", "output_files")
        for row in manifest.get(section, ())
        if isinstance(row, Mapping) and str(row.get("path")) == relative
    ]
    if len(matches) != 1:
        raise ValueError(f"upstream manifest does not uniquely declare {relative}")
    return matches[0]


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("expanded-universe configuration must be a mapping")
    if (
        int(config.get("schema_version", 0)) != 1
        or config.get("model_id") != MODEL_ID
        or config.get("stage_id") != STAGE_ID
    ):
        raise ValueError("unexpected expanded-universe configuration identity")
    frozen = config.get("frozen_upstream", {})
    if frozen.get("manifest_sha256") != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen Model 02 backtest manifest hash differs")
    if frozen.get("signal_table_sha256") != FROZEN_SIGNAL_SHA256:
        raise ValueError("frozen posterior signal-table hash differs")
    market = config.get("market_data", {})
    if market.get("manifest_sha256") != MARKET_MANIFEST_SHA256:
        raise ValueError("frozen ETF-universe manifest hash differs")
    if market.get("universe_rule") != "all_tickers_in_frozen_local_market_data_manifest":
        raise ValueError("expanded universe must use every frozen local ETF")
    if bool(market.get("external_retrieval_used")):
        raise ValueError("this controlled diagnostic must not mix in retrieved ETFs")
    universe = config.get("universe", {})
    if tuple(map(str, universe.get("strategy_assets", ()))) != EXPANDED_ASSETS:
        raise ValueError("expanded strategy universe differs from the frozen 12 ETFs")
    if tuple(map(str, universe.get("original_strategy_assets", ()))) != ORIGINAL_ASSETS:
        raise ValueError("original seven-ETF universe declaration differs")
    if tuple(map(str, universe.get("newly_admitted_assets", ()))) != NEW_ASSETS:
        raise ValueError("newly admitted ETF declaration differs")
    if tuple(map(str, config.get("methods", ()))) != METHODS:
        raise ValueError("expanded backtest must contain exactly four methods")
    timing = config.get("timing", {})
    if bool(timing.get("posterior_refit")):
        raise ValueError("expanded-universe diagnostic cannot refit the posterior")
    estimation = config.get("estimation", {})
    if (
        estimation.get("return_frequency") != "weekly_open_to_open"
        or int(estimation.get("minimum_labeled_weeks", 0))
        != MINIMUM_LABELED_WEEKS
        or float(estimation.get("regime_mean_pseudo_weeks", -1.0))
        != PSEUDO_WEEKS
        or float(estimation.get("covariance_annualization_factor", -1.0))
        != PERIODS_PER_YEAR
    ):
        raise ValueError("expanded return-estimation policy differs")
    optimization = config.get("optimization", {})
    if (
        optimization.get("objective")
        != "maximize_one_week_expected_return_net_of_estimated_trading_cost"
        or optimization.get("long_only") is not True
        or optimization.get("fully_invested") is not True
        or estimation.get("covariance_estimator") != "ledoit_wolf"
        or float(optimization.get("annualized_volatility_cap", -1.0))
        != VOLATILITY_CAP
        or float(optimization.get("transaction_cost", -1.0))
        != TRANSACTION_COST
    ):
        raise ValueError("expanded optimization risk or cost policy differs")
    caps = {str(key): float(value) for key, value in optimization["asset_caps"].items()}
    if set(caps) != set(EXPANDED_ASSETS):
        raise ValueError("asset caps do not cover exactly the expanded universe")
    expected_solver = {
        "method": "SLSQP",
        "tolerance": 1.0e-12,
        "maximum_iterations": 2000,
        "feasibility_tolerance": 1.0e-7,
        "fallback_order": [
            "hold_pretrade_if_feasible",
            "constrained_minimum_variance",
            "all_BIL",
        ],
    }
    if optimization.get("solver") != expected_solver:
        raise ValueError("expanded optimizer solver policy differs")
    benchmarks = config.get("benchmarks", {})
    equal = benchmarks.get(EQUAL_METHOD, {})
    static = benchmarks.get(STATIC_METHOD, {})
    if (
        equal.get("assets") != "all_twelve_strategy_assets"
        or float(equal.get("weight_per_asset", -1.0)) != 1.0 / 12.0
        or static.get("equity_asset") != "SPY"
        or float(static.get("equity_weight", -1.0)) != 0.60
        or static.get("bond_asset") != "AGG"
        or float(static.get("bond_weight", -1.0)) != 0.40
    ):
        raise ValueError("expanded static benchmark declarations differ")
    uncertainty = config.get("evaluation", {}).get("uncertainty", {})
    if tuple(map(str, uncertainty.get("comparator_methods", ()))) != (
        POOLED_METHOD,
        STATIC_METHOD,
        EQUAL_METHOD,
    ):
        raise ValueError("posterior comparison set differs")
    if (
        int(uncertainty.get("block_length_weeks", 0)) != 26
        or int(uncertainty.get("resamples", 0)) != 10_000
        or float(uncertainty.get("confidence_level", 0.0)) != 0.95
        or int(uncertainty.get("random_seed", -1)) != 20260718
    ):
        raise ValueError("expanded uncertainty protocol differs")
    return config, raw


def _group_caps(config: Mapping[str, Any]) -> tuple[GroupCap, ...]:
    return tuple(
        GroupCap(
            name=str(name),
            assets=tuple(map(str, declaration["assets"])),
            maximum=float(declaration["maximum"]),
        )
        for name, declaration in config["optimization"]["group_caps"].items()
    )


def _verified_sources(
    root: Path, config: Mapping[str, Any]
) -> tuple[
    Mapping[str, Any],
    bytes,
    dict[str, tuple[Path, bytes]],
    tuple[Path, bytes],
]:
    frozen = config["frozen_upstream"]
    upstream_manifest_path, upstream_manifest_bytes = _read_verified(
        root, frozen["manifest"], frozen["manifest_sha256"]
    )
    upstream_manifest = json.loads(upstream_manifest_bytes)
    verified: dict[str, tuple[Path, bytes]] = {}
    for key in (
        "signal_table",
        "optimizer_audit",
        "regime_estimate_audit",
        "original_performance",
        "original_weekly_returns",
        "price_history",
        "composite_scores",
    ):
        path, raw = _read_verified(root, frozen[key], frozen[f"{key}_sha256"])
        declaration = _manifest_declaration(upstream_manifest, path=path, root=root)
        if (
            str(declaration.get("sha256")) != _sha256(raw)
            or int(declaration.get("bytes", -1)) != len(raw)
        ):
            raise ValueError(f"upstream declaration mismatch for {path.relative_to(root)}")
        verified[key] = (path, raw)
    market = config["market_data"]
    market_manifest = _read_verified(
        root, market["manifest"], market["manifest_sha256"]
    )
    return (
        upstream_manifest,
        upstream_manifest_bytes,
        verified,
        market_manifest,
    )


def _validate_market_data(
    *,
    root: Path,
    config: Mapping[str, Any],
    market_manifest_path: Path,
    market_manifest_bytes: bytes,
    price_path: Path,
    price_bytes: bytes,
    prices: pd.DataFrame,
) -> dict[str, object]:
    market_manifest = json.loads(market_manifest_bytes)
    manifest_tickers = tuple(map(str, market_manifest.get("tickers", ())))
    configured = tuple(map(str, config["universe"]["strategy_assets"]))
    actual = tuple(
        prices.loc[:, ["ticker"]]
        .drop_duplicates()["ticker"]
        .astype(str)
        .tolist()
    )
    if manifest_tickers != configured or set(actual) != set(configured):
        raise ValueError(
            "configured and manifest ETF order must match, and the price file "
            "must contain exactly that ETF set"
        )
    relative_price = price_path.relative_to(root).as_posix()
    declarations = [
        row
        for row in market_manifest.get("processed_files", ())
        if str(row.get("path", "")).replace("\\", "/") == relative_price
    ]
    if len(declarations) != 1:
        raise ValueError("market manifest does not uniquely declare the price history")
    declaration = declarations[0]
    if (
        str(declaration.get("sha256")) != _sha256(price_bytes)
        or int(declaration.get("bytes", -1)) != len(price_bytes)
    ):
        raise ValueError("market manifest price declaration differs")
    dates = pd.to_datetime(prices["date"], errors="raise").dt.normalize()
    inspection = prices.assign(date=dates).groupby("ticker", sort=False).agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        rows=("date", "size"),
        unique_dates=("date", "nunique"),
    )
    if (
        inspection["first_date"].nunique() != 1
        or inspection["last_date"].nunique() != 1
        or inspection["rows"].nunique() != 1
        or not inspection["rows"].eq(inspection["unique_dates"]).all()
    ):
        raise ValueError("all expanded ETFs must have identical duplicate-free coverage")
    if prices.duplicated(["ticker", "date"]).any():
        raise ValueError("price history contains duplicate ticker-date rows")
    if prices["adjusted_open"].isna().any():
        raise ValueError("expanded ETF adjusted opens contain missing values")
    common_dates: pd.DatetimeIndex | None = None
    normalized = prices.assign(date=dates)
    for ticker in configured:
        ticker_dates = pd.DatetimeIndex(
            normalized.loc[normalized["ticker"].astype(str).eq(ticker), "date"]
        ).sort_values()
        if common_dates is None:
            common_dates = ticker_dates
        elif not ticker_dates.equals(common_dates):
            raise ValueError("expanded ETFs do not share an identical daily calendar")
    return {
        "manifest": market_manifest_path.relative_to(root).as_posix(),
        "ticker_count": len(configured),
        "first_date": inspection["first_date"].iloc[0],
        "last_date": inspection["last_date"].iloc[0],
        "rows_per_ticker": int(inspection["rows"].iloc[0]),
        "identical_calendar": True,
    }


def _parse_signals(raw: bytes) -> pd.DataFrame:
    signals = pd.read_csv(BytesIO(raw))
    for column in (
        "reference_week",
        "signal_date",
        "regime_reference_month",
        "start_date",
        "end_date",
    ):
        signals[column] = pd.to_datetime(signals[column], errors="coerce")
    if len(signals) != EXPECTED_TARGET_COUNT:
        raise ValueError("frozen signal count differs from the expected 446 weeks")
    if signals["reference_week"].duplicated().any():
        raise ValueError("frozen posterior contains duplicate weekly signals")
    probability = signals.loc[:, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if (
        not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-10)
    ):
        raise ValueError("frozen posterior probabilities are invalid")
    return signals


def _validate_schedule(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    assets: Sequence[str],
) -> pd.DataFrame:
    schedule = build_weekly_execution_schedule(prices, assets=assets)
    schedule = schedule.loc[
        schedule["reference_week"].isin(signals["reference_week"])
    ].reset_index(drop=True)
    frozen = signals.loc[:, list(SCHEDULE_COLUMNS)].reset_index(drop=True).copy()
    candidate = schedule.loc[:, list(SCHEDULE_COLUMNS)].reset_index(drop=True).copy()
    for frame in (frozen, candidate):
        for column in ("reference_week", "start_date", "end_date"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame["is_complete"] = frame["is_complete"].astype(bool)
    if not candidate.equals(frozen):
        raise ValueError("all-ETF weekly schedule differs from the frozen posterior schedule")
    return schedule


def _static_targets(
    *,
    signals: pd.DataFrame,
    assets: Sequence[str],
) -> pd.DataFrame:
    equal = equal_weight_target(assets)
    static = static_60_40_target(
        assets,
        risk_asset="SPY",
        defensive_asset="AGG",
        risk_weight=0.60,
    )
    records: list[dict[str, object]] = []
    for _, signal in signals.iterrows():
        for method, target in ((EQUAL_METHOD, equal), (STATIC_METHOD, static)):
            records.extend(
                _weight_records(
                    method=method,
                    signal=signal,
                    weights=target,
                    simulation_assets=assets,
                    specification_type="benchmark",
                )
            )
    return pd.DataFrame.from_records(records)


def _simulate_methods(
    *,
    weights: pd.DataFrame,
    holding_returns: pd.DataFrame,
    prices: pd.DataFrame,
    assets: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    simulations: list[pd.DataFrame] = []
    targets: list[pd.DataFrame] = []
    complete = weights.loc[~weights["is_live_only"].astype(bool)].copy()
    for method, frame in complete.groupby("method", sort=False):
        target = frame.loc[
            :, ["method", "reference_week", "ticker", "target_weight"]
        ].copy()
        simulations.append(
            simulate_weekly_targets(
                target,
                holding_returns,
                assets=assets,
                transaction_costs=TRANSACTION_COST,
            )
        )
        targets.append(target)
    weekly = pd.concat(simulations, ignore_index=True)
    nav = build_weekly_daily_nav(
        pd.concat(targets, ignore_index=True),
        weekly,
        prices,
        assets=assets,
    )
    return weekly, nav


def _performance(
    weekly: pd.DataFrame,
    nav: pd.DataFrame,
    holding_returns: pd.DataFrame,
) -> pd.DataFrame:
    bil = holding_returns.set_index("reference_week")["BIL"]
    bil.index.name = "reference_month"
    return _weekly_metric_labels(
        compute_performance_metrics(
            _metric_view(weekly),
            nav,
            bil_monthly_returns=bil,
            periods_per_year=PERIODS_PER_YEAR,
        )
    )


def _uncertainty(
    weekly: pd.DataFrame,
    *,
    baseline_method: str,
    comparator_methods: Sequence[str],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    protocol = config["evaluation"]["uncertainty"]
    return _weekly_metric_labels(
        paired_circular_block_bootstrap(
            _metric_view(weekly),
            baseline_method=baseline_method,
            comparator_methods=tuple(comparator_methods),
            block_length=int(protocol["block_length_weeks"]),
            n_resamples=int(protocol["resamples"]),
            confidence_level=float(protocol["confidence_level"]),
            seed=int(protocol["random_seed"]),
            periods_per_year=PERIODS_PER_YEAR,
        )
    )


def _verify_original_expected_means(
    expanded_audit: pd.DataFrame,
    frozen_audit: pd.DataFrame,
) -> float:
    frozen = frozen_audit.loc[
        frozen_audit["method"].astype(str).isin((BASELINE_METHOD, POOLED_METHOD))
    ].copy()
    expanded = expanded_audit.loc[
        expanded_audit["method"].astype(str).isin((BASELINE_METHOD, POOLED_METHOD))
    ].copy()
    columns = [f"expected_weekly_return_{asset}" for asset in ORIGINAL_ASSETS]
    joined = expanded.loc[:, ["method", "reference_week", *columns]].merge(
        frozen.loc[:, ["method", "reference_week", *columns]],
        on=["method", "reference_week"],
        how="inner",
        suffixes=("_expanded", "_frozen"),
        validate="one_to_one",
    )
    expected_rows = 2 * EXPECTED_TARGET_COUNT
    if len(joined) != expected_rows:
        raise ValueError("frozen expected-return audit does not align with expanded rows")
    mismatches = [
        np.abs(
            joined[f"{column}_expanded"].to_numpy(dtype=float)
            - joined[f"{column}_frozen"].to_numpy(dtype=float)
        )
        for column in columns
    ]
    maximum = float(np.max(np.concatenate(mismatches)))
    if maximum > 1.0e-12:
        raise ValueError("original-asset expected returns changed under frozen signals")
    return maximum


def _universe_comparison(
    expanded_performance: pd.DataFrame,
    original_performance: pd.DataFrame,
) -> pd.DataFrame:
    expanded = expanded_performance.set_index("method")
    original = original_performance.set_index("method")
    records: list[dict[str, object]] = []
    for method in METHODS:
        if method not in expanded.index or method not in original.index:
            raise ValueError(f"performance comparison lacks method {method}")
        record: dict[str, object] = {"method": method}
        for metric in PERFORMANCE_COMPARISON_COLUMNS:
            expanded_value = float(expanded.loc[method, metric])
            original_value = float(original.loc[method, metric])
            record[f"frozen_seven_{metric}"] = original_value
            record[f"expanded_twelve_{metric}"] = expanded_value
            record[f"delta_{metric}"] = expanded_value - original_value
        records.append(record)
    return pd.DataFrame.from_records(records)


def _universe_change_uncertainty(
    expanded_weekly: pd.DataFrame,
    original_weekly: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    protocol = config["evaluation"]["uncertainty"]
    materiality = float(
        config["evaluation"]["universe_change"][
            "practical_materiality_threshold_annual_return"
        ]
    )
    records: list[pd.DataFrame] = []
    for method in METHODS:
        expanded_label = f"expanded_twelve::{method}"
        original_label = f"frozen_seven::{method}"
        expanded = expanded_weekly.loc[
            expanded_weekly["method"].astype(str).eq(method),
            ["reference_week", "net_return"],
        ].copy()
        original = original_weekly.loc[
            original_weekly["method"].astype(str).eq(method),
            ["reference_week", "net_return"],
        ].copy()
        if len(expanded) != EXPECTED_WEEK_COUNT or len(original) != EXPECTED_WEEK_COUNT:
            raise ValueError(f"universe comparison has incomplete {method} history")
        expanded["method"] = expanded_label
        original["method"] = original_label
        combined = pd.concat([expanded, original], ignore_index=True)
        result = paired_circular_block_bootstrap(
            _metric_view(combined),
            baseline_method=expanded_label,
            comparator_methods=(original_label,),
            block_length=int(protocol["block_length_weeks"]),
            n_resamples=int(protocol["resamples"]),
            confidence_level=float(protocol["confidence_level"]),
            seed=int(protocol["random_seed"]),
            periods_per_year=PERIODS_PER_YEAR,
        )
        result = _weekly_metric_labels(result)
        result.insert(0, "method", method)
        point = float(result.iloc[0]["annualized_mean_difference"])
        lower = float(result.iloc[0]["annualized_mean_difference_ci_lower"])
        numerical_tolerance = 1.0e-12
        result["statistical_evidence_of_improvement"] = bool(
            point > numerical_tolerance and lower > numerical_tolerance
        )
        result["practical_materiality_threshold"] = materiality
        result["meets_practical_materiality_threshold"] = bool(point >= materiality)
        records.append(result)
    return pd.concat(records, ignore_index=True)


def _validate_static_control(
    expanded_weekly: pd.DataFrame,
    original_weekly: pd.DataFrame,
) -> dict[str, float]:
    expanded = expanded_weekly.loc[
        expanded_weekly["method"].astype(str).eq(STATIC_METHOD)
    ].copy()
    original = original_weekly.loc[
        original_weekly["method"].astype(str).eq(STATIC_METHOD)
    ].copy()
    columns = (
        "gross_return",
        "transaction_cost_rate",
        "net_return",
        "gross_turnover",
        "one_way_turnover",
    )
    joined = expanded.loc[:, ["reference_week", *columns]].merge(
        original.loc[:, ["reference_week", *columns]],
        on="reference_week",
        suffixes=("_expanded", "_frozen"),
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != EXPECTED_WEEK_COUNT:
        raise ValueError("static 60/40 control does not cover every frozen week")
    result = {
        column: float(
            np.max(
                np.abs(
                    joined[f"{column}_expanded"].to_numpy(dtype=float)
                    - joined[f"{column}_frozen"].to_numpy(dtype=float)
                )
            )
        )
        for column in columns
    }
    if max(result.values()) > 1.0e-14:
        raise ValueError("static 60/40 changed under the expanded simulation")
    return result


def _allocation_diagnostics(
    *,
    weights: pd.DataFrame,
    weekly: pd.DataFrame,
    performance: pd.DataFrame,
    optimizer_audit: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = weights.loc[~weights["is_live_only"].astype(bool)].copy()
    caps = {
        str(asset): float(value)
        for asset, value in config["optimization"]["asset_caps"].items()
    }
    utilization = (
        complete.groupby(["method", "ticker"], sort=False)["target_weight"]
        .agg(
            average_target_weight="mean",
            maximum_target_weight="max",
            minimum_target_weight="min",
            weeks="size",
        )
        .reset_index()
    )
    above = (
        complete.assign(above_one_percent=complete["target_weight"].gt(0.01))
        .groupby(["method", "ticker"], sort=False)["above_one_percent"]
        .mean()
        .rename("fraction_weeks_above_one_percent")
        .reset_index()
    )
    utilization = utilization.merge(
        above, on=["method", "ticker"], how="left", validate="one_to_one"
    )
    utilization["asset_cap"] = utilization["ticker"].map(caps)
    utilization["fraction_weeks_at_asset_cap"] = np.where(
        utilization["method"].isin((BASELINE_METHOD, POOLED_METHOD)),
        complete.assign(
            at_cap=np.isclose(
                complete["target_weight"],
                complete["ticker"].map(caps),
                atol=1.0e-6,
            )
        )
        .groupby(["method", "ticker"], sort=False)["at_cap"]
        .transform("mean")
        .groupby([complete["method"], complete["ticker"]], sort=False)
        .first()
        .reindex(
            pd.MultiIndex.from_frame(utilization[["method", "ticker"]])
        )
        .to_numpy(dtype=float),
        np.nan,
    )

    pivot = complete.pivot(
        index=["method", "reference_week"],
        columns="ticker",
        values="target_weight",
    ).reindex(columns=list(EXPANDED_ASSETS), fill_value=0.0)
    method_records: list[dict[str, object]] = []
    performance_by_method = performance.set_index("method")
    group_config = config["optimization"]["group_caps"]
    for method in METHODS:
        frame = pivot.loc[method]
        values = frame.to_numpy(dtype=float)
        hhi = np.square(values).sum(axis=1)
        maximum = values.max(axis=1)
        new_weight = frame.loc[:, list(NEW_ASSETS)].sum(axis=1).to_numpy(dtype=float)
        audit = optimizer_audit.loc[
            optimizer_audit["method"].astype(str).eq(method)
            & optimizer_audit["reference_week"].isin(frame.index)
        ]
        record: dict[str, object] = {
            "method": method,
            "weeks": len(frame),
            "average_hhi": float(hhi.mean()),
            "average_effective_asset_count": float(np.mean(1.0 / hhi)),
            "minimum_effective_asset_count": float(np.min(1.0 / hhi)),
            "average_maximum_single_asset_weight": float(maximum.mean()),
            "maximum_single_asset_weight": float(maximum.max()),
            "average_assets_above_one_percent": float((values > 0.01).sum(axis=1).mean()),
            "average_new_asset_weight": float(new_weight.mean()),
            "maximum_new_asset_weight": float(new_weight.max()),
            "annualized_one_way_turnover": float(
                performance_by_method.loc[method, "annualized_one_way_turnover"]
            ),
            "annualized_cost_drag": float(
                performance_by_method.loc[method, "annualized_cost_drag"]
            ),
            "solver_fallback_rate": (
                float(audit["fallback_used"].astype(bool).mean())
                if not audit.empty
                else np.nan
            ),
            "maximum_constraint_violation": (
                float(audit["maximum_constraint_violation"].max())
                if not audit.empty
                else np.nan
            ),
            "volatility_cap_binding_frequency": (
                float(
                    np.isclose(
                        audit["estimated_annualized_volatility"].to_numpy(dtype=float),
                        VOLATILITY_CAP,
                        atol=1.0e-6,
                    ).mean()
                )
                if not audit.empty
                else np.nan
            ),
        }
        for name, declaration in group_config.items():
            group_weight = frame.loc[:, list(declaration["assets"])].sum(axis=1)
            record[f"group_cap_binding_frequency_{name}"] = (
                float(
                    np.isclose(
                        group_weight.to_numpy(dtype=float),
                        float(declaration["maximum"]),
                        atol=1.0e-6,
                    ).mean()
                )
                if method in (BASELINE_METHOD, POOLED_METHOD)
                else np.nan
            )
        method_records.append(record)

    posterior = pivot.loc[BASELINE_METHOD]
    pooled = pivot.loc[POOLED_METHOD]
    delta = posterior - pooled
    return_pivot = weekly.pivot(
        index="reference_week", columns="method", values="net_return"
    )
    distance = pd.DataFrame(
        {
            "reference_week": posterior.index,
            "posterior_pooled_half_l1_target_distance": (
                0.5 * delta.abs().sum(axis=1)
            ).to_numpy(dtype=float),
            "posterior_new_asset_weight": posterior.loc[:, list(NEW_ASSETS)]
            .sum(axis=1)
            .to_numpy(dtype=float),
            "pooled_new_asset_weight": pooled.loc[:, list(NEW_ASSETS)]
            .sum(axis=1)
            .to_numpy(dtype=float),
            "posterior_minus_pooled_new_asset_weight": delta.loc[
                :, list(NEW_ASSETS)
            ]
            .sum(axis=1)
            .to_numpy(dtype=float),
            "posterior_minus_pooled_net_return": (
                return_pivot[BASELINE_METHOD] - return_pivot[POOLED_METHOD]
            ).reindex(posterior.index).to_numpy(dtype=float),
        }
    )
    for asset in EXPANDED_ASSETS:
        distance[f"active_weight_{asset}"] = delta[asset].to_numpy(dtype=float)
    realized_tracking_error = float(
        distance["posterior_minus_pooled_net_return"].std(ddof=1)
        * np.sqrt(PERIODS_PER_YEAR)
    )
    diagnostics = pd.DataFrame.from_records(method_records)
    diagnostics["posterior_pooled_realized_tracking_error"] = np.where(
        diagnostics["method"].eq(BASELINE_METHOD), realized_tracking_error, np.nan
    )
    diagnostics["posterior_pooled_average_half_l1_target_distance"] = np.where(
        diagnostics["method"].eq(BASELINE_METHOD),
        float(distance["posterior_pooled_half_l1_target_distance"].mean()),
        np.nan,
    )
    return diagnostics, utilization, distance


def _asset_return_diagnostics(
    holding_returns: pd.DataFrame,
    optimizer_audit: pd.DataFrame,
) -> pd.DataFrame:
    complete_weeks = pd.DatetimeIndex(
        pd.to_datetime(holding_returns["reference_week"], errors="raise")
    )
    complete_audit = optimizer_audit.loc[
        optimizer_audit["reference_week"].isin(complete_weeks)
    ].copy()
    records: list[dict[str, object]] = []
    for asset in EXPANDED_ASSETS:
        realized = holding_returns[asset].to_numpy(dtype=float)
        record: dict[str, object] = {
            "ticker": asset,
            "realized_annualized_arithmetic_return": float(
                PERIODS_PER_YEAR * np.mean(realized)
            ),
            "realized_annualized_volatility": float(
                np.sqrt(PERIODS_PER_YEAR) * np.std(realized, ddof=1)
            ),
            "realized_total_return": float(np.prod(1.0 + realized) - 1.0),
        }
        expected_column = f"expected_weekly_return_{asset}"
        for method in (BASELINE_METHOD, POOLED_METHOD):
            expected = complete_audit.loc[
                complete_audit["method"].astype(str).eq(method),
                expected_column,
            ].to_numpy(dtype=float)
            annualized = float(PERIODS_PER_YEAR * np.mean(expected))
            record[f"{method}_average_expected_annual_return"] = annualized
            record[f"{method}_realized_minus_expected_annual_return"] = (
                float(record["realized_annualized_arithmetic_return"]) - annualized
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _risk_estimator_change(
    expanded_estimates: pd.DataFrame,
    frozen_estimates: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["method", "reference_week", "ticker"]
    columns = [*keys, "within_regime_variance", "ledoit_wolf_shrinkage"]
    expanded = (
        expanded_estimates.loc[
            expanded_estimates["method"].astype(str).isin(
                (BASELINE_METHOD, POOLED_METHOD)
            )
            & expanded_estimates["ticker"].astype(str).isin(ORIGINAL_ASSETS),
            columns,
        ]
        .drop_duplicates(keys)
        .copy()
    )
    frozen = (
        frozen_estimates.loc[
            frozen_estimates["method"].astype(str).isin(
                (BASELINE_METHOD, POOLED_METHOD)
            )
            & frozen_estimates["ticker"].astype(str).isin(ORIGINAL_ASSETS),
            columns,
        ]
        .drop_duplicates(keys)
        .copy()
    )
    joined = expanded.merge(
        frozen,
        on=keys,
        how="inner",
        suffixes=("_expanded", "_frozen"),
        validate="one_to_one",
    )
    expected_rows = 2 * EXPECTED_TARGET_COUNT * len(ORIGINAL_ASSETS)
    if len(joined) != expected_rows:
        raise ValueError("expanded and frozen risk-estimator audits do not align")
    frozen_variance = joined["within_regime_variance_frozen"].to_numpy(
        dtype=float
    )
    expanded_variance = joined["within_regime_variance_expanded"].to_numpy(
        dtype=float
    )
    if (frozen_variance <= 0.0).any():
        raise ValueError("frozen within-regime variances must be positive")
    joined["relative_diagonal_variance_change"] = (
        expanded_variance / frozen_variance - 1.0
    )
    return (
        joined.groupby(["method", "reference_week"], sort=False)
        .agg(
            frozen_ledoit_wolf_shrinkage=(
                "ledoit_wolf_shrinkage_frozen",
                "first",
            ),
            expanded_ledoit_wolf_shrinkage=(
                "ledoit_wolf_shrinkage_expanded",
                "first",
            ),
            mean_absolute_relative_original_asset_variance_change=(
                "relative_diagonal_variance_change",
                lambda values: float(np.mean(np.abs(values))),
            ),
            maximum_absolute_relative_original_asset_variance_change=(
                "relative_diagonal_variance_change",
                lambda values: float(np.max(np.abs(values))),
            ),
        )
        .reset_index()
    )


def _latest_payload(
    *,
    signals: pd.DataFrame,
    weights: pd.DataFrame,
    optimizer_audit: pd.DataFrame,
    coverage: Mapping[str, object],
) -> dict[str, object]:
    latest_week = pd.Timestamp(signals["reference_week"].max())
    signal = signals.loc[signals["reference_week"].eq(latest_week)].iloc[0]
    latest_weights = weights.loc[weights["reference_week"].eq(latest_week)]
    latest_audit = optimizer_audit.loc[
        optimizer_audit["reference_week"].eq(latest_week)
    ]

    def target(method: str) -> list[dict[str, object]]:
        rows = latest_weights.loc[
            latest_weights["method"].astype(str).eq(method)
            & latest_weights["target_weight"].gt(1.0e-12)
        ]
        return [
            {"ticker": str(row.ticker), "weight": float(row.target_weight)}
            for row in rows.itertuples(index=False)
        ]

    def audit(method: str) -> Mapping[str, object]:
        rows = latest_audit.loc[latest_audit["method"].astype(str).eq(method)]
        if len(rows) != 1:
            raise ValueError(f"latest optimizer audit lacks one {method} row")
        return rows.iloc[0]

    posterior_audit = audit(BASELINE_METHOD)
    pooled_audit = audit(POOLED_METHOD)
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "status": "exploratory_research_output_not_investment_advice",
        "reference_week": latest_week,
        "signal_date": signal["signal_date"],
        "execution_date": signal["start_date"],
        "holding_period_complete": bool(signal["is_complete"]),
        "posterior_refit": False,
        "universe": list(EXPANDED_ASSETS),
        "market_data_coverage": dict(coverage),
        "posterior": [
            {"regime_id": regime, "probability": float(signal[column])}
            for regime, column in zip(
                CANONICAL_REGIME_IDS, PROBABILITY_COLUMNS, strict=True
            )
        ],
        "targets": {
            BASELINE_METHOD: target(BASELINE_METHOD),
            POOLED_METHOD: target(POOLED_METHOD),
            EQUAL_METHOD: target(EQUAL_METHOD),
            STATIC_METHOD: target(STATIC_METHOD),
        },
        "optimizer_estimates": {
            BASELINE_METHOD: {
                "expected_weekly_return": float(
                    posterior_audit["expected_portfolio_weekly_return"]
                ),
                "estimated_annualized_volatility": float(
                    posterior_audit["estimated_annualized_volatility"]
                ),
            },
            POOLED_METHOD: {
                "expected_weekly_return": float(
                    pooled_audit["expected_portfolio_weekly_return"]
                ),
                "estimated_annualized_volatility": float(
                    pooled_audit["estimated_annualized_volatility"]
                ),
            },
        },
    }


def build_m02_expanded_universe_backtest(
    *, project_root: Path, config_path: Path
) -> dict[str, Path]:
    """Build and publish the isolated four-strategy, all-local-ETF diagnostic."""

    root = project_root.resolve()
    config_path = _project_path(root, config_path)
    config, config_bytes = _load_config(config_path)
    (
        upstream_manifest,
        upstream_manifest_bytes,
        verified,
        (market_manifest_path, market_manifest_bytes),
    ) = _verified_sources(root, config)
    prices = pd.read_csv(BytesIO(verified["price_history"][1]))
    coverage = _validate_market_data(
        root=root,
        config=config,
        market_manifest_path=market_manifest_path,
        market_manifest_bytes=market_manifest_bytes,
        price_path=verified["price_history"][0],
        price_bytes=verified["price_history"][1],
        prices=prices,
    )
    signals = _parse_signals(verified["signal_table"][1])
    _validate_schedule(signals, prices, assets=EXPANDED_ASSETS)
    start_week = pd.Timestamp(config["timing"]["formal_backtest_start"])
    end_week = pd.Timestamp(config["timing"]["formal_backtest_end"])
    if (
        start_week != pd.Timestamp(upstream_manifest["formal_backtest_start"])
        or end_week != pd.Timestamp(upstream_manifest["formal_backtest_end"])
    ):
        raise ValueError("expanded formal window differs from the frozen stage")
    if pd.Timestamp(config["timing"]["latest_live_target_week"]) != pd.Timestamp(
        upstream_manifest["latest_live_target_week"]
    ):
        raise ValueError("expanded latest live target differs from the frozen stage")

    all_holding_returns = build_weekly_open_to_open_holding_returns(
        prices, assets=EXPANDED_ASSETS
    )
    estimator_returns = _weekly_estimator_history(
        all_holding_returns,
        strategy_assets=EXPANDED_ASSETS,
    )
    holding_returns = all_holding_returns.loc[
        all_holding_returns["reference_week"].between(start_week, end_week)
    ].reset_index(drop=True)
    if len(holding_returns) != EXPECTED_WEEK_COUNT:
        raise ValueError("expanded holding-return window differs from 445 weeks")
    scores = pd.read_csv(BytesIO(verified["composite_scores"][1]))
    regime_history = derive_m02_regime_history(scores)
    specifications = (
        AllocationSpecification(
            method=BASELINE_METHOD,
            specification_type="expanded_universe_exploratory",
            parameter="universe_asset_count",
            parameter_value=float(len(EXPANDED_ASSETS)),
            kappa=PSEUDO_WEEKS,
            volatility_cap=VOLATILITY_CAP,
            transaction_cost=TRANSACTION_COST,
            cap_multiplier=1.0,
        ),
        AllocationSpecification(
            method=POOLED_METHOD,
            specification_type="expanded_universe_benchmark",
            parameter="pooled_mean_ablation",
            parameter_value=1.0,
            kappa=PSEUDO_WEEKS,
            volatility_cap=VOLATILITY_CAP,
            transaction_cost=TRANSACTION_COST,
            cap_multiplier=1.0,
        ),
    )
    dynamic_weights, estimate_audit, optimizer_audit = _dynamic_targets(
        specifications=specifications,
        signals=signals,
        estimator_returns=estimator_returns,
        regime_history=regime_history,
        prices=prices,
        strategy_assets=EXPANDED_ASSETS,
        simulation_assets=EXPANDED_ASSETS,
        minimum_observations=MINIMUM_LABELED_WEEKS,
        covariance_annualization_factor=PERIODS_PER_YEAR,
        asset_caps={
            str(asset): float(value)
            for asset, value in config["optimization"]["asset_caps"].items()
        },
        group_caps=_group_caps(config),
    )
    frozen_audit = pd.read_csv(BytesIO(verified["optimizer_audit"][1]))
    frozen_audit["reference_week"] = pd.to_datetime(
        frozen_audit["reference_week"], errors="raise"
    )
    optimizer_audit["reference_week"] = pd.to_datetime(
        optimizer_audit["reference_week"], errors="raise"
    )
    maximum_original_mean_mismatch = _verify_original_expected_means(
        optimizer_audit, frozen_audit
    )
    frozen_estimate_audit = pd.read_csv(
        BytesIO(verified["regime_estimate_audit"][1])
    )
    for frame in (estimate_audit, frozen_estimate_audit):
        frame["reference_week"] = pd.to_datetime(
            frame["reference_week"], errors="raise"
        )
    risk_estimator_change = _risk_estimator_change(
        estimate_audit, frozen_estimate_audit
    )
    weights = pd.concat(
        [
            dynamic_weights,
            _static_targets(signals=signals, assets=EXPANDED_ASSETS),
        ],
        ignore_index=True,
    )
    if set(weights["method"].astype(str)) != set(METHODS):
        raise ValueError("expanded targets contain methods outside the locked four")
    target_counts = weights.groupby("method")["reference_week"].nunique()
    if not target_counts.eq(EXPECTED_TARGET_COUNT).all():
        raise ValueError("each expanded method must have 446 weekly targets")
    weekly, nav = _simulate_methods(
        weights=weights,
        holding_returns=holding_returns,
        prices=prices,
        assets=EXPANDED_ASSETS,
    )
    weekly_counts = weekly.groupby("method")["reference_week"].nunique()
    if not weekly_counts.eq(EXPECTED_WEEK_COUNT).all():
        raise ValueError("each expanded method must have 445 complete weeks")
    performance = _performance(weekly, nav, holding_returns)
    strategy_uncertainty = _uncertainty(
        weekly,
        baseline_method=BASELINE_METHOD,
        comparator_methods=(POOLED_METHOD, STATIC_METHOD, EQUAL_METHOD),
        config=config,
    )

    original_performance = pd.read_csv(BytesIO(verified["original_performance"][1]))
    original_weekly = pd.read_csv(BytesIO(verified["original_weekly_returns"][1]))
    original_weekly["reference_week"] = pd.to_datetime(
        original_weekly["reference_week"], errors="raise"
    )
    original_weekly = original_weekly.loc[
        original_weekly["method"].astype(str).isin(METHODS)
    ].copy()
    universe_comparison = _universe_comparison(
        performance, original_performance
    )
    universe_change = _universe_change_uncertainty(
        weekly,
        original_weekly,
        config=config,
    )
    static_control = _validate_static_control(weekly, original_weekly)
    diagnostics, utilization, posterior_pooled_distance = _allocation_diagnostics(
        weights=weights,
        weekly=weekly,
        performance=performance,
        optimizer_audit=optimizer_audit,
        config=config,
    )
    asset_return_diagnostics = _asset_return_diagnostics(
        holding_returns, optimizer_audit
    )
    latest = _latest_payload(
        signals=signals,
        weights=weights,
        optimizer_audit=optimizer_audit,
        coverage=coverage,
    )

    output = config["outputs"]
    processed = _project_path(root, output["processed_dir"])
    published = _project_path(root, output["published_dir"])
    processed_frames = {
        "holding_returns": holding_returns,
        "regime_estimate_audit": estimate_audit,
        "optimizer_audit": optimizer_audit,
        "weekly_weights": weights,
        "weekly_returns": weekly,
        "daily_nav": nav,
        "performance": performance,
        "strategy_uncertainty": strategy_uncertainty,
        "universe_comparison": universe_comparison,
        "universe_change_uncertainty": universe_change,
        "allocation_diagnostics": diagnostics,
        "asset_utilization": utilization,
        "asset_return_diagnostics": asset_return_diagnostics,
        "risk_estimator_change": risk_estimator_change,
        "posterior_pooled_distance": posterior_pooled_distance,
    }
    processed_paths = {
        name: processed / str(output[name]) for name in processed_frames
    }
    for name, frame in processed_frames.items():
        _write_csv(frame, processed_paths[name])

    published_names = (
        "weekly_weights",
        "weekly_returns",
        "performance",
        "strategy_uncertainty",
        "universe_comparison",
        "universe_change_uncertainty",
        "allocation_diagnostics",
        "asset_utilization",
        "asset_return_diagnostics",
        "risk_estimator_change",
        "posterior_pooled_distance",
    )
    published_paths = {
        name: published / str(output[name]) for name in published_names
    }
    for name in published_names:
        _write_csv(processed_frames[name], published_paths[name])
    latest_path = published / str(output["latest_allocation"])
    _write_json(latest, latest_path)

    primary = strategy_uncertainty.loc[
        strategy_uncertainty["comparator_method"].astype(str).eq(POOLED_METHOD)
    ].iloc[0]
    posterior_change = universe_change.loc[
        universe_change["method"].astype(str).eq(BASELINE_METHOD)
    ].iloc[0]
    pooled_change = universe_change.loc[
        universe_change["method"].astype(str).eq(POOLED_METHOD)
    ].iloc[0]
    posterior_diagnostic = diagnostics.loc[
        diagnostics["method"].astype(str).eq(BASELINE_METHOD)
    ].iloc[0]
    risk_change_summary = {
        "frozen_average_ledoit_wolf_shrinkage": float(
            risk_estimator_change["frozen_ledoit_wolf_shrinkage"].mean()
        ),
        "expanded_average_ledoit_wolf_shrinkage": float(
            risk_estimator_change["expanded_ledoit_wolf_shrinkage"].mean()
        ),
        "average_absolute_relative_original_asset_variance_change": float(
            risk_estimator_change[
                "mean_absolute_relative_original_asset_variance_change"
            ].mean()
        ),
        "maximum_absolute_relative_original_asset_variance_change": float(
            risk_estimator_change[
                "maximum_absolute_relative_original_asset_variance_change"
            ].max()
        ),
        "interpretation": (
            "The universe comparison includes both new feasible assets and the "
            "dimension-induced change in the multivariate Ledoit-Wolf covariance."
        ),
    }
    summary = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "status": "historical_exploratory_diagnostic_not_investment_advice",
        "posterior_refit": False,
        "frozen_posterior": {
            "manifest": config["frozen_upstream"]["manifest"],
            "manifest_sha256": config["frozen_upstream"]["manifest_sha256"],
            "signal_table": config["frozen_upstream"]["signal_table"],
            "signal_table_sha256": config["frozen_upstream"][
                "signal_table_sha256"
            ],
        },
        "universe": {
            "selection_rule": config["market_data"]["universe_rule"],
            "assets": list(EXPANDED_ASSETS),
            "original_assets": list(ORIGINAL_ASSETS),
            "newly_admitted_assets": list(NEW_ASSETS),
            "external_retrieval_used": False,
            "sufficient_for_controlled_diagnostic": True,
            "sufficient_for_production_global_macro_universe": False,
            "covered_exposures": [
                "US equity beta",
                "nominal Treasury and aggregate duration",
                "inflation-linked bonds",
                "investment-grade and high-yield credit",
                "cash",
                "gold",
                "broad commodities and oil",
                "US dollar",
            ],
            "important_missing_exposures": [
                "international developed and emerging-market equities",
                "small-cap and equity styles",
                "non-US sovereign and emerging-market debt",
                "REITs and infrastructure",
                "broader foreign exchange",
                "industrial metals and agriculture",
            ],
            "retrieval_decision": (
                "No download was used for this first controlled test. External "
                "assets should enter a separately predeclared robustness stage."
            ),
        },
        "market_data_coverage": coverage,
        "schedule_validation": {
            "target_weeks": EXPECTED_TARGET_COUNT,
            "complete_backtest_weeks": EXPECTED_WEEK_COUNT,
            "all_twelve_schedule_exactly_matches_frozen_schedule": True,
            "maximum_original_asset_expected_mean_mismatch": (
                maximum_original_mean_mismatch
            ),
            "static_60_40_maximum_accounting_mismatches": static_control,
        },
        "risk_estimator_change": risk_change_summary,
        "methods": list(METHODS),
        "performance": performance.to_dict(orient="records"),
        "posterior_comparisons": strategy_uncertainty.to_dict(orient="records"),
        "primary_posterior_minus_pooled": primary.to_dict(),
        "universe_change": universe_change.to_dict(orient="records"),
        "headline_universe_change": {
            "posterior": posterior_change.to_dict(),
            "pooled": pooled_change.to_dict(),
        },
        "allocation_diagnostics": diagnostics.to_dict(orient="records"),
        "posterior_pooled_average_half_l1_target_distance": float(
            posterior_diagnostic[
                "posterior_pooled_average_half_l1_target_distance"
            ]
        ),
        "posterior_pooled_realized_tracking_error": float(
            posterior_diagnostic["posterior_pooled_realized_tracking_error"]
        ),
        "warnings": [
            "The universe expansion was requested after reviewing prior results "
            "and is exploratory rather than a fresh holdout.",
            "Five of twelve securities overlap materially with other duration or "
            "commodity exposures, so security count overstates independent breadth.",
            "Expanding the Ledoit-Wolf system changes covariance estimates for "
            "the original assets; the universe delta is not a pure new-asset "
            "admission effect.",
            "Equal weight is security-equal rather than asset-class-equal and "
            "therefore overrepresents overlapping bond and commodity exposures.",
            "USO is a futures-based oil vehicle whose roll and path behavior can "
            "differ materially from spot crude oil.",
            "Adjusted ETF history is mutable provider data, not point-in-time market data.",
        ],
    }
    summary_path = published / str(output["summary"])
    _write_json(summary, summary_path)

    implementation_paths = (
        root
        / "src/regime_allocation/portfolio/m02_expanded_universe_backtest.py",
        root / "src/regime_allocation/cli/build_m02_expanded_universe_backtest.py",
        root / "src/regime_allocation/portfolio/m02_pipeline.py",
        root / "src/regime_allocation/portfolio/m02_estimation.py",
        root / "src/regime_allocation/portfolio/m02_weekly.py",
        root / "src/regime_allocation/portfolio/optimization.py",
        root / "src/regime_allocation/portfolio/pipeline.py",
        root / "src/regime_allocation/portfolio/estimation.py",
        root / "src/regime_allocation/portfolio/baselines.py",
        root / "src/regime_allocation/backtest/engine.py",
        root / "src/regime_allocation/backtest/metrics.py",
    )
    generated_paths = [
        *processed_paths.values(),
        *published_paths.values(),
        latest_path,
        summary_path,
    ]
    input_paths = [
        verified[key][0]
        for key in (
            "signal_table",
            "optimizer_audit",
            "regime_estimate_audit",
            "original_performance",
            "original_weekly_returns",
            "price_history",
            "composite_scores",
        )
    ]
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(root).as_posix(),
        "configuration_sha256": _sha256(config_bytes),
        "frozen_upstream_manifest": {
            "path": config["frozen_upstream"]["manifest"],
            "sha256": _sha256(upstream_manifest_bytes),
        },
        "frozen_posterior_signal": {
            "path": config["frozen_upstream"]["signal_table"],
            "sha256": _sha256(verified["signal_table"][1]),
            "refit": False,
        },
        "market_data_manifest": {
            "path": market_manifest_path.relative_to(root).as_posix(),
            "sha256": _sha256(market_manifest_bytes),
        },
        "inputs": [
            _declaration(root, path)
            for path in (*input_paths, market_manifest_path)
        ],
        "implementation_files": [
            _declaration(root, path) for path in implementation_paths
        ],
        "output_files": [_declaration(root, path) for path in generated_paths],
        "universe": list(EXPANDED_ASSETS),
        "original_universe": list(ORIGINAL_ASSETS),
        "newly_admitted_assets": list(NEW_ASSETS),
        "method_count": len(METHODS),
        "methods": list(METHODS),
        "target_weeks": EXPECTED_TARGET_COUNT,
        "weekly_observations": EXPECTED_WEEK_COUNT,
        "formal_backtest_start": start_week,
        "formal_backtest_end": end_week,
        "posterior_refit": False,
        "schedule_matches_frozen_upstream": True,
        "maximum_original_asset_expected_mean_mismatch": (
            maximum_original_mean_mismatch
        ),
        "external_retrieval_used": False,
        "credential_policy": (
            "This stage consumes only hash-pinned local files and never reads "
            "environment credentials or downloads data."
        ),
    }
    manifest_path = _project_path(root, output["manifest"])
    _write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "latest_allocation": latest_path,
        "performance": published_paths["performance"],
        "strategy_uncertainty": published_paths["strategy_uncertainty"],
        "universe_comparison": published_paths["universe_comparison"],
        "universe_change_uncertainty": published_paths[
            "universe_change_uncertainty"
        ],
        "weekly_returns": published_paths["weekly_returns"],
        "weekly_weights": published_paths["weekly_weights"],
    }


__all__ = ["build_m02_expanded_universe_backtest"]
