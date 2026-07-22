"""Integration contracts for Model 02's promoted-baseline weekly backtest."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/m02_regime_allocation_backtest.yaml"
MANIFEST = ROOT / "data/manifests/m02_regime_allocation_backtest.json"
CURRENT_MANIFEST = ROOT / "data/manifests/m02_current_baseline.json"
PROCESSED = ROOT / "data/processed/m02_regime_allocation_backtest"
PUBLISHED = ROOT / "results/published/m02_regime_allocation_backtest"

BASELINE = "posterior_optimized"
PROMOTED_VARIANT = "student_t_7_reduced_core"
PUBLIC_METHODS = {
    BASELINE,
    "pooled_mean_optimizer",
    "legacy_sharpe_map",
    "equal_weight",
    "static_60_spy_40_agg",
}
PROBABILITY_COLUMNS = [
    "probability_growth_up_inflation_up",
    "probability_growth_down_inflation_up",
    "probability_growth_up_inflation_down",
    "probability_growth_down_inflation_down",
]


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_hashes_promoted_lineage_implementation_and_outputs() -> None:
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["model_id"] == "m02_soft_composite"
    assert manifest["model_variant"] == PROMOTED_VARIANT
    assert manifest["stage_id"] == "weekly_posterior_regime_allocation_backtest"
    assert manifest["configuration"] == CONFIG.relative_to(ROOT).as_posix()
    assert manifest["configuration_sha256"] == _digest(CONFIG)
    upstream = manifest["upstream_current_baseline_manifest"]
    assert upstream["path"] == CURRENT_MANIFEST.relative_to(ROOT).as_posix()
    assert upstream["sha256"] == _digest(CURRENT_MANIFEST)
    assert manifest["formal_backtest_start"] == "2018-01-01T00:00:00"
    assert manifest["formal_backtest_end"] == "2026-07-06T00:00:00"
    assert manifest["latest_live_target_week"] == "2026-07-13T00:00:00"
    assert manifest["weekly_observations"] == 445
    assert manifest["periods_per_year"] == 52

    for section in ("inputs", "implementation_files", "output_files"):
        declarations = manifest[section]
        assert declarations
        paths = [str(row["path"]) for row in declarations]
        assert len(paths) == len(set(paths))
        for declaration in declarations:
            path = ROOT / str(declaration["path"])
            assert path.is_file(), path
            assert declaration["sha256"] == _digest(path), path
            assert declaration["bytes"] == path.stat().st_size, path


def test_weekly_signals_are_pre_release_causal_and_complete() -> None:
    signals = pd.read_csv(
        PROCESSED / "signal_table.csv",
        parse_dates=[
            "reference_week",
            "signal_date",
            "regime_reference_month",
            "mapping_available_at",
            "start_date",
            "end_date",
        ],
    )
    assert len(signals) == 446
    assert signals["reference_week"].is_unique
    assert signals["reference_week"].dt.dayofweek.eq(0).all()
    assert signals["signal_date"].equals(signals["reference_week"])
    assert signals["variant_id"].eq(PROMOTED_VARIANT).all()
    assert signals["model_role"].eq("baseline").all()
    assert signals["same_day_release_evidence_included"].eq(False).all()  # noqa: E712
    assert (signals["mapping_available_at"] < signals["signal_date"]).all()
    assert (
        signals["regime_reference_month"]
        == signals["reference_week"].dt.to_period("M").dt.to_timestamp()
    ).all()
    probabilities = signals[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(len(signals)))
    assert signals.iloc[-1]["reference_week"] == pd.Timestamp("2026-07-13")
    assert bool(signals.iloc[-1]["is_complete"]) is False
    assert pd.isna(signals.iloc[-1]["end_date"])


def test_weekly_accounting_excludes_only_the_live_week() -> None:
    returns = pd.read_csv(
        PUBLISHED / "weekly_returns.csv", parse_dates=["reference_week"]
    )
    weights = pd.read_csv(
        PUBLISHED / "weekly_weights.csv", parse_dates=["reference_week"]
    )
    assert set(returns["method"]) == PUBLIC_METHODS
    assert returns.groupby("method")["reference_week"].nunique().eq(445).all()
    assert returns["reference_week"].max() == pd.Timestamp("2026-07-06")
    assert set(weights["method"]) == PUBLIC_METHODS
    assert weights.groupby("method")["reference_week"].nunique().eq(446).all()
    live = weights.loc[weights["is_live_only"].eq(True)]  # noqa: E712
    assert live["reference_week"].nunique() == 1
    assert live["reference_week"].iloc[0] == pd.Timestamp("2026-07-13")
    totals = weights.groupby(["method", "reference_week"])["target_weight"].sum()
    assert totals.to_numpy() == pytest.approx(np.ones(len(totals)))
    assert (weights["target_weight"] >= -1.0e-12).all()
    optimized_agg = weights.loc[
        weights["method"].eq(BASELINE) & weights["ticker"].eq("AGG"),
        "target_weight",
    ]
    assert optimized_agg.to_numpy() == pytest.approx(np.zeros(len(optimized_agg)))


def test_performance_and_uncertainty_match_the_published_summary() -> None:
    performance = pd.read_csv(PUBLISHED / "performance_summary.csv").set_index(
        "method"
    )
    uncertainty = pd.read_csv(PUBLISHED / "comparison_uncertainty.csv")
    assert set(performance.index) == PUBLIC_METHODS
    assert performance["weeks"].eq(445).all()
    baseline = performance.loc[BASELINE]
    pooled = performance.loc["pooled_mean_optimizer"]
    assert baseline["total_return"] == pytest.approx(1.087885057674606)
    assert baseline["cagr"] == pytest.approx(0.08983053504339189)
    assert baseline["annualized_volatility"] == pytest.approx(0.10207084262032795)
    assert pooled["total_return"] == pytest.approx(1.222593313036703)
    assert pooled["cagr"] == pytest.approx(0.0978220652097368)
    assert uncertainty["block_length"].eq(26).all()
    assert uncertainty["n_resamples"].eq(10_000).all()
    pooled_row = uncertainty.loc[
        uncertainty["comparator_method"].eq("pooled_mean_optimizer")
    ].iloc[0]
    assert pooled_row["annualized_mean_difference"] == pytest.approx(
        -0.007607642732047061
    )
    assert pooled_row["annualized_mean_difference_ci_lower"] < 0.0
    assert pooled_row["annualized_mean_difference_ci_upper"] > 0.0


def test_optimizer_audit_is_feasible_and_uses_prior_day_knowledge() -> None:
    audit = pd.read_csv(
        PROCESSED / "optimizer_audit.csv",
        parse_dates=["reference_week", "signal_date", "estimated_pretrade_as_of"],
    )
    estimates = pd.read_csv(
        PROCESSED / "regime_estimate_audit.csv",
        parse_dates=["reference_week", "signal_date", "knowledge_cutoff"],
    )
    assert audit["maximum_constraint_violation"].le(1.0e-7).all()
    assert audit["weight_sum"].to_numpy() == pytest.approx(np.ones(len(audit)))
    assert audit["optimization_outcome"].notna().all()
    assert estimates["training_count"].ge(60).all()
    assert (
        estimates["knowledge_cutoff"]
        == estimates["signal_date"] - pd.Timedelta(days=1)
    ).all()
    pretrade = audit["estimated_pretrade_as_of"].notna()
    assert (
        audit.loc[pretrade, "estimated_pretrade_as_of"]
        < audit.loc[pretrade, "signal_date"]
    ).all()


def test_latest_target_is_the_incomplete_week_and_matches_public_weights() -> None:
    latest = json.loads(
        (PUBLISHED / "latest_allocation.json").read_text(encoding="utf-8")
    )
    assert latest["model_variant"] == PROMOTED_VARIANT
    assert latest["reference_week"] == "2026-07-13T00:00:00"
    assert latest["signal_date"] == "2026-07-13T00:00:00"
    assert latest["execution_date"] == "2026-07-13T00:00:00"
    assert latest["holding_period_complete"] is False
    assert latest["baseline_parameters"]["rebalance_frequency"] == "weekly"
    target = {row["ticker"]: row["weight"] for row in latest["target_weights"]}
    assert target == pytest.approx(
        {"SPY": 0.35, "HYG": 0.15, "GLD": 0.25, "LQD": 0.25}
    )
    assert sum(row["probability"] for row in latest["posterior"]) == pytest.approx(1.0)

