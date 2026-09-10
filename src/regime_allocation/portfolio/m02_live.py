"""Dated operational replay of the promoted M02 policy, outside frozen archives.

The full portfolio path is replayed from the formal start: costs and pretrade
weights make a latest-week-only optimization a different strategy. Historical
publication builders and their hash-pinned inputs remain unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from regime_allocation.backtest import (
    build_weekly_daily_nav,
    build_weekly_open_to_open_holding_returns,
    compute_performance_metrics,
    simulate_weekly_targets,
)
from regime_allocation.cli.build_m02_current_baseline import (
    BASELINE_ID,
    SELECTED_CANDIDATE_MODELS,
    _load_current_config,
    _run_config,
)
from regime_allocation.cli.build_m02_evidence_experiment_inference import (
    _merge_observation_models,
)
from regime_allocation.cli.build_m02_inference_sensitivities import _load_yaml
from regime_allocation.models.m02_soft_composite.walkforward import prepare_observation_data
from regime_allocation.models.m02_soft_composite.weekly_decision_replay import (
    run_weekly_decision_replay,
)
from regime_allocation.portfolio.baselines import equal_weight_target, static_60_40_target
from regime_allocation.portfolio.m02_pipeline import (
    _allocation_specifications,
    _dynamic_targets,
    _load_config,
    _metric_view,
    _validate_model01_policy_parity,
    _weekly_estimator_history,
    _weekly_metric_labels,
    _weight_records,
)
from regime_allocation.portfolio.m02_weekly import (
    CANONICAL_REGIME_IDS,
    build_weekly_execution_schedule,
    derive_m02_regime_history,
    select_m02_weekly_signals,
)
from regime_allocation.portfolio.pipeline import (
    BASELINE_METHOD,
    POOLED_METHOD,
    AllocationSpecification,
    _group_caps,
)
from regime_allocation.reporting.readme_snapshot import (
    RELEASE_ID,
    load_snapshot,
    render_snapshot,
    replace_snapshot_section,
    validate_snapshot,
)

NEW_YORK = ZoneInfo("America/New_York")
DEFINING_COMPONENTS = {
    "average_hourly_earnings",
    "consumer_activity",
    "core_cpi",
    "core_pce",
    "industrial_production",
    "payrolls",
    "producer_prices",
    "unemployment_rate",
}
EVIDENCE_FEATURES = {
    ("weekly_labor_stress", "initial_claims_innovation"),
    ("consumer_demand_real_decomposition", "implicit_retail_price_log_change"),
    ("consumer_demand_real_decomposition", "real_retail_and_food_services_log_change"),
    ("business_investment_activity_pipeline", "core_capital_goods_shipments_log_change"),
    ("business_investment_activity_pipeline", "core_capital_goods_orders_shipments_gap_log_change"),
}


def signal_calendar(now: datetime) -> tuple[date, date]:
    """Return this Monday and the latest safely completed price calendar day."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone aware")
    local = now.astimezone(NEW_YORK)
    monday = local.date() - timedelta(days=local.weekday())
    # Wait until 16:30 New York; provider quality checks handle holidays/weekends.
    close_day = local.date()
    if (local.hour, local.minute) < (16, 30):
        close_day -= timedelta(days=1)
    return monday, close_day


def live_replay_configuration(root: Path) -> tuple[dict, dict]:
    """Preserve production mechanics while dropping unrelated experiment variants."""
    folder = root / "configs/models"
    current, _ = _load_current_config(folder / "m02_current_baseline.yaml")
    base, _ = _load_yaml(folder / "m02_event_driven_bayesian_filter.yaml")
    frozen, _ = _load_yaml(folder / "m02_inference_sensitivities.yaml")
    feature, _ = _load_yaml(folder / "m02_feature_revision.yaml")
    base = _merge_observation_models(
        base,
        {key: feature["candidate_observation_models"][key] for key in SELECTED_CANDIDATE_MODELS},
    )
    run = _run_config(frozen, current)
    run["variants"] = [
        item
        for item in run["variants"]
        if item["id"] in (BASELINE_ID, "transition_only", "partial_only")
    ]
    if "dependence_diagnostics" in run:
        run["dependence_diagnostics"]["variant_id"] = BASELINE_ID
    return base, run


def freshness_audit(
    events: pd.DataFrame, components: pd.DataFrame, signal_week: date
) -> dict[str, Any]:
    """Reject future inputs and surface missing/stale input families explicitly."""
    cutoff = pd.Timestamp(signal_week)
    latest: dict[str, str] = {}
    reasons: list[str] = []
    if set(components["component"].dropna()) != DEFINING_COMPONENTS:
        raise ValueError("required defining components are missing or unexpected")
    if set(zip(events["release_block"], events["feature_name"], strict=True)) != EVIDENCE_FEATURES:
        raise ValueError("required evidence features are missing or unexpected")
    for name, frame, group, maximum_age in (
        ("evidence", events, ["release_block", "feature_name"], 75),
        ("defining", components, ["component"], 93),
    ):
        if frame.empty:
            raise ValueError(f"{name} inputs are empty")
        dates = pd.to_datetime(frame["release_date"], errors="raise")
        if dates.isna().any() or dates.ge(cutoff).any():
            raise ValueError(f"{name} inputs violate the Monday information cutoff")
        for key, rows in frame.assign(_date=dates).groupby(group):
            usable = (
                rows.loc[rows["feature_status"].eq("available")]
                if name == "evidence"
                else rows.loc[
                    pd.to_numeric(rows["transformed_value"], errors="coerce").map(
                        lambda x: pd.notna(x) and math.isfinite(x)
                    )
                ]
            )
            if usable.empty:
                raise ValueError(f"no usable observations for {key}")
            last = usable["_date"].max()
            label = ":".join(key)
            latest[f"{name}:{label}"] = last.date().isoformat()
            limit = 14 if key[0] == "weekly_labor_stress" else maximum_age
            if (cutoff - last).days > limit:
                reasons.append(f"{label} last usable release {last.date()}")
    if not latest:
        raise ValueError("no dated macro inputs")
    return {
        "status": "stale" if reasons else "fresh",
        "reason": "; ".join(reasons)
        if reasons
        else "All required input families passed release-age checks at the signal cutoff.",
        "latest_release_by_family": latest,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False, default=str) + "\n").encode("utf-8"))


def compute_live_allocation(
    *,
    project_root: Path,
    input_paths: dict[str, Path],
    prices_path: Path,
    output_dir: Path,
    signal_week: date,
    now: datetime | None = None,
) -> dict[str, Path]:
    """Compute a complete causal path and validated latest snapshot, without publishing."""
    if signal_week.weekday() != 0:
        raise ValueError("signal_week must be a Monday")
    current = now or datetime.now(UTC)
    if signal_week > signal_calendar(current)[0]:
        raise ValueError("cannot compute a future signal week")
    root = project_root.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(root / "outputs/m02_weekly_live"):
        raise ValueError("working output must be inside outputs/m02_weekly_live")
    scores = pd.read_csv(input_paths["scores"])
    components = pd.read_csv(input_paths["components"])
    mapping = pd.read_csv(input_paths["mapping"])
    events = pd.read_csv(input_paths["events"])
    freshness = freshness_audit(events, components, signal_week)
    for column in (
        "release_date",
        "reference_date",
        "reference_month",
        "control_source_reference_date",
        "control_source_reference_month",
    ):
        if column in events:
            events[column] = pd.to_datetime(events[column], format="mixed", errors="raise")
    prices = pd.read_csv(prices_path)
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    if prices["date"].max().date() > signal_calendar(current)[1]:
        raise ValueError("price history contains an uncompleted or future session")
    config, _ = _load_config(root / "configs/models/m02_regime_allocation_backtest.yaml")
    template, _ = _load_yaml(root / config["sources"]["model01_allocation_template"])
    _validate_model01_policy_parity(config, template)
    strategy = tuple(config["universe"]["strategy_assets"])
    assets = strategy + tuple(config["universe"]["benchmark_only_assets"])
    schedule = build_weekly_execution_schedule(prices, assets=assets)
    start = pd.Timestamp(config["timing"]["formal_backtest_start"])
    schedule = schedule.loc[schedule["reference_week"].between(start, pd.Timestamp(signal_week))]
    if schedule.empty or schedule["reference_week"].min() != start:
        raise ValueError("price history must cover the complete allocation path from 2018-01-01")
    if schedule["reference_week"].max() != pd.Timestamp(signal_week):
        raise ValueError(
            "This signal week has no completed common ETF session yet; retry after close."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    base, run = live_replay_configuration(root)
    (output_dir / "replay_configuration.yaml").write_text(
        yaml.safe_dump({"base": base, "run": run}), encoding="utf-8"
    )
    print(f"Replaying weekly macro decisions through {signal_week}...", flush=True)
    prepared = prepare_observation_data(events, scores, base)
    replay = run_weekly_decision_replay(
        scores,
        mapping,
        prepared,
        components,
        base,
        run,
        replay_end=pd.Timestamp(signal_week),
        decision_dates=tuple(schedule["reference_week"]),
    )
    decisions = replay.decision_marginals
    decisions.to_csv(output_dir / "decision_marginals.csv", index=False)
    signals = select_m02_weekly_signals(
        decisions, schedule["reference_week"], baseline_variant_id=BASELINE_ID
    )
    signals = signals.merge(schedule, on="reference_week", validate="one_to_one")
    signals = signals.sort_values("reference_week", kind="stable").reset_index(drop=True)
    if len(signals) != len(schedule):
        raise ValueError("replay lacks weekly signals")
    all_returns = build_weekly_open_to_open_holding_returns(prices, assets=assets)
    estimator = _weekly_estimator_history(all_returns, strategy_assets=strategy)
    returns = all_returns.loc[all_returns["reference_week"].ge(start)]
    returns = returns.loc[returns["reference_week"].isin(schedule["reference_week"])]
    baseline = _allocation_specifications(config)[0]
    pooled = AllocationSpecification(
        method=POOLED_METHOD,
        specification_type="benchmark",
        parameter="pooled_mean_ablation",
        parameter_value=1.0,
        kappa=baseline.kappa,
        volatility_cap=baseline.volatility_cap,
        transaction_cost=baseline.transaction_cost,
        cap_multiplier=1.0,
    )
    print(
        f"Optimizing {len(signals)} weeks, including each strategy's prior holdings...", flush=True
    )
    weights, estimates, audit = _dynamic_targets(
        specifications=(baseline, pooled),
        signals=signals,
        estimator_returns=estimator,
        regime_history=derive_m02_regime_history(scores),
        prices=prices,
        strategy_assets=strategy,
        simulation_assets=assets,
        minimum_observations=int(config["estimation"]["minimum_labeled_weeks"]),
        covariance_annualization_factor=float(
            config["optimization"]["covariance_annualization_factor"]
        ),
        asset_caps={str(k): float(v) for k, v in config["optimization"]["asset_caps"].items()},
        group_caps=_group_caps(config),
    )
    static_records = []
    for _, signal in signals.iterrows():
        for method, target in (
            ("equal_weight", equal_weight_target(strategy)),
            ("static_60_spy_40_agg", static_60_40_target(assets, defensive_asset="AGG")),
        ):
            static_records.extend(
                _weight_records(
                    method=method,
                    signal=signal,
                    weights=target,
                    simulation_assets=assets,
                    specification_type="benchmark",
                    audit={"optimization_outcome": "static_formula"},
                )
            )
    weights = pd.concat([weights, pd.DataFrame(static_records)], ignore_index=True)
    complete = weights.loc[~weights["is_live_only"]]
    simulations, targets = [], []
    for method, frame in complete.groupby("method", sort=False):
        target = frame[["method", "reference_week", "ticker", "target_weight"]]
        targets.append(target)
        simulations.append(
            simulate_weekly_targets(
                target, returns, assets=assets, transaction_costs=baseline.transaction_cost
            )
        )
    weekly = pd.concat(simulations, ignore_index=True)
    nav = build_weekly_daily_nav(
        pd.concat(targets, ignore_index=True), weekly, prices, assets=assets
    )
    bil = returns.set_index("reference_week")["BIL"]
    bil.index.name = "reference_month"
    metrics = _weekly_metric_labels(
        compute_performance_metrics(
            _metric_view(weekly), nav, bil_monthly_returns=bil, periods_per_year=52
        )
    )
    for name, frame in {
        "target_weights": weights,
        "return_estimate_audit": estimates,
        "optimizer_audit": audit,
        "weekly_returns": weekly,
        "daily_nav": nav,
        "performance_summary": metrics,
    }.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, lineterminator="\n")
    last_weights = weights.loc[
        weights["method"].eq(BASELINE_METHOD)
        & weights["reference_week"].eq(pd.Timestamp(signal_week))
    ]
    last_audit = audit.loc[
        audit["method"].eq(BASELINE_METHOD) & audit["reference_week"].eq(pd.Timestamp(signal_week))
    ].iloc[0]
    last_signal = signals.iloc[-1]
    cutoff = datetime.combine(signal_week, datetime.min.time(), tzinfo=NEW_YORK)
    snapshot = {
        "schema_version": 1,
        "status": "research_only",
        "model_variant": BASELINE_ID,
        "release_id": RELEASE_ID,
        "generated_at": current.astimezone(UTC).isoformat(),
        "signal_week": signal_week.isoformat(),
        "information_cutoff": cutoff.isoformat(),
        "macro_as_of": max(freshness["latest_release_by_family"].values()),
        "price_as_of": pd.Timestamp(last_audit["estimated_pretrade_as_of"]).date().isoformat(),
        "valid_until": (cutoff + timedelta(days=7)).isoformat(),
        "data_freshness": {k: freshness[k] for k in ("status", "reason")},
        "weights": {
            asset: float(last_weights.set_index("ticker").loc[asset, "target_weight"])
            for asset in strategy
        },
        "quadrant_posterior": {
            regime: float(last_signal[f"probability_{regime}"]) for regime in CANONICAL_REGIME_IDS
        },
        "forecast_weekly_return": float(last_audit["expected_portfolio_weekly_return"]),
        "estimated_annual_volatility": float(last_audit["estimated_annualized_volatility"]),
        "optimization_outcome": str(last_audit["optimization_outcome"]),
    }
    validate_snapshot(snapshot, now=current)
    snapshot_path = output_dir / "latest_allocation.json"
    _write_json(snapshot_path, snapshot)
    declarations = dict(input_paths, prices=prices_path)
    declarations["scripts/download_etf_history.py"] = root / "scripts/download_etf_history.py"
    declarations["pyproject.toml"] = root / "pyproject.toml"
    declarations.update({p.relative_to(root).as_posix(): p for p in (root / "src").rglob("*.py")})
    declarations.update(
        {p.relative_to(root).as_posix(): p for p in (root / "configs/models").glob("*.yaml")}
    )
    hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in sorted(declarations.items())
    }
    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "generated_at": snapshot["generated_at"],
        "signal_week": signal_week,
        "information_cutoff": snapshot["information_cutoff"],
        "execution_date": str(pd.Timestamp(last_signal["start_date"]).date()),
        "latest_price_download_session": str(prices["date"].max().date()),
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyyaml", "tzdata")
        },
        "data_sources": {
            "macro": "https://api.stlouisfed.org/fred",
            "market": "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        },
        "freshness": freshness,
        "sha256_inputs_configs_implementation": hashes,
        "sha256_outputs": {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in ("latest_allocation.json", "performance_summary.csv")
        },
        "history_policy": "Full own-path replay from 2018-01-01; incomplete week excluded from performance.",
        "reconstruction_disclosure": "Newly acquired FRED vintages and mutable Yahoo adjusted prices; "
        "not a byte-identical reproduction of the archived publication. Same-history model selection "
        "was used; the extended backtest is not an untouched validation sample.",
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return {
        name: output_dir / name
        for name in ("latest_allocation.json", "performance_summary.csv", "run_manifest.json")
    }


def publish_live_files(
    root: Path, artifacts: dict[str, Path], *, now: datetime | None = None
) -> None:
    """Validate everything before changing any public file; replace README last."""
    expected = {"latest_allocation.json", "performance_summary.csv", "run_manifest.json"}
    if set(artifacts) != expected:
        raise ValueError("unexpected publication files")
    payloads = {name: path.read_bytes() for name, path in artifacts.items()}
    current = now or datetime.now(UTC)
    snapshot = load_snapshot(artifacts["latest_allocation.json"], now=current)
    rendered = render_snapshot(snapshot, now=current)
    if (
        snapshot["data_freshness"]["status"] != "fresh"
        or current >= datetime.fromisoformat(snapshot["valid_until"])
        or date.fromisoformat(snapshot["signal_week"]) != signal_calendar(current)[0]
    ):
        raise ValueError("stale inputs: keeping the previous successful publication")
    manifest = json.loads(payloads["run_manifest.json"])
    if (
        manifest.get("release_id") != RELEASE_ID
        or manifest.get("signal_week") != snapshot["signal_week"]
        or manifest.get("information_cutoff") != snapshot["information_cutoff"]
    ):
        raise ValueError("manifest does not match the snapshot release and cutoff")
    for name in ("latest_allocation.json", "performance_summary.csv"):
        if (
            manifest.get("sha256_outputs", {}).get(name)
            != hashlib.sha256(payloads[name]).hexdigest()
        ):
            raise ValueError(f"manifest hash mismatch for {name}")
    metrics = pd.read_csv(BytesIO(payloads["performance_summary.csv"]))
    methods = {BASELINE_METHOD, POOLED_METHOD, "equal_weight", "static_60_spy_40_agg"}
    if len(metrics) != 4 or set(metrics["method"]) != methods:
        raise ValueError("performance summary must contain the four expected methods")
    for column in ("cagr", "annualized_volatility", "maximum_drawdown"):
        if (
            column not in metrics
            or not pd.to_numeric(metrics[column], errors="coerce")
            .map(lambda x: pd.notna(x) and math.isfinite(x))
            .all()
        ):
            raise ValueError(f"invalid performance summary {column}")
    readme = root / "README.md"
    updated = replace_snapshot_section(readme.read_bytes().decode("utf-8"), rendered).encode(
        "utf-8"
    )
    destination = root / "results/live/m02_weekly"
    destination.mkdir(parents=True, exist_ok=True)
    # Stage all bytes before replacing public files. A failed calculation never reaches this step.
    staged: list[tuple[Path, Path]] = []
    for name, payload in payloads.items():
        path = destination / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        staged.append((temporary, path))
    temporary_readme = readme.with_suffix(".md.tmp")
    temporary_readme.write_bytes(updated)
    for temporary, path in staged:
        temporary.replace(path)
    temporary_readme.replace(readme)
