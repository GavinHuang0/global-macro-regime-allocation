"""Integration contracts for the frozen-posterior 12-ETF diagnostic."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/m02_expanded_universe_backtest.yaml"
MANIFEST = ROOT / "data/manifests/m02_expanded_universe_backtest.json"
PROCESSED = ROOT / "data/processed/m02_expanded_universe_backtest"
PUBLISHED = ROOT / "results/published/m02_expanded_universe_backtest"
UPSTREAM_MANIFEST = ROOT / "data/manifests/m02_regime_allocation_backtest.json"
FROZEN_SIGNAL = (
    ROOT / "data/processed/m02_regime_allocation_backtest/signal_table.csv"
)

POSTERIOR = "posterior_optimized"
POOLED = "pooled_mean_optimizer"
EQUAL = "equal_weight"
STATIC = "static_60_spy_40_agg"
METHODS = {POSTERIOR, POOLED, EQUAL, STATIC}
ASSETS = (
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


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_freezes_posterior_and_hashes_all_artifacts() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == 1
    assert manifest["model_id"] == "m02_soft_composite"
    assert manifest["stage_id"] == "frozen_posterior_expanded_universe_backtest"
    assert manifest["configuration_sha256"] == _digest(CONFIG)
    assert manifest["frozen_upstream_manifest"] == {
        "path": UPSTREAM_MANIFEST.relative_to(ROOT).as_posix(),
        "sha256": (
            "5821824dcfe6a82347d1df176467b74d0199cb7724bb08f5e07daf08801fbc4a"
        ),
    }
    assert _digest(UPSTREAM_MANIFEST) == manifest["frozen_upstream_manifest"][
        "sha256"
    ]
    assert manifest["frozen_posterior_signal"] == {
        "path": FROZEN_SIGNAL.relative_to(ROOT).as_posix(),
        "sha256": (
            "8cb4d27056ec5b024ad5e2cbf4549429da98def25b8602e91275731e8046111f"
        ),
        "refit": False,
    }
    assert _digest(FROZEN_SIGNAL) == manifest["frozen_posterior_signal"]["sha256"]
    assert manifest["posterior_refit"] is False
    assert manifest["schedule_matches_frozen_upstream"] is True
    assert manifest["external_retrieval_used"] is False
    assert manifest["universe"] == list(ASSETS)
    assert manifest["methods"] == [POSTERIOR, POOLED, EQUAL, STATIC]
    assert manifest["target_weeks"] == 446
    assert manifest["weekly_observations"] == 445
    assert manifest["maximum_original_asset_expected_mean_mismatch"] <= 1.0e-12

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


def test_targets_use_all_twelve_assets_and_obey_locked_constraints() -> None:
    weights = pd.read_csv(
        PUBLISHED / "weekly_weights.csv",
        parse_dates=["reference_week", "signal_date", "execution_date"],
    )
    audit = pd.read_csv(
        PROCESSED / "optimizer_audit.csv",
        parse_dates=["reference_week", "signal_date", "estimated_pretrade_as_of"],
    )
    estimates = pd.read_csv(
        PROCESSED / "regime_estimate_audit.csv",
        parse_dates=[
            "reference_week",
            "signal_date",
            "knowledge_cutoff",
            "first_training_reference_week",
            "last_training_reference_week",
        ],
    )

    assert set(weights["method"]) == METHODS
    assert set(weights["ticker"]) == set(ASSETS)
    assert weights.groupby("method")["reference_week"].nunique().eq(446).all()
    assert weights.groupby(["method", "reference_week"])["ticker"].nunique().eq(12).all()
    totals = weights.groupby(["method", "reference_week"])["target_weight"].sum()
    assert totals.to_numpy() == pytest.approx(np.ones(len(totals)))

    equal = weights.loc[weights["method"].eq(EQUAL), "target_weight"]
    assert equal.to_numpy() == pytest.approx(np.full(len(equal), 1.0 / 12.0))
    static = weights.loc[weights["method"].eq(STATIC)]
    static_average = static.groupby("ticker")["target_weight"].mean().to_dict()
    assert static_average == pytest.approx(
        {asset: 0.6 if asset == "SPY" else 0.4 if asset == "AGG" else 0.0 for asset in ASSETS}
    )

    optimized = weights.loc[weights["method"].isin((POSTERIOR, POOLED))]
    caps = {
        "SPY": 0.35,
        "IEF": 0.50,
        "TIP": 0.40,
        "LQD": 0.40,
        "HYG": 0.25,
        "BIL": 1.00,
        "GLD": 0.25,
        "DBC": 0.25,
        "UUP": 0.25,
        "TLT": 0.25,
        "USO": 0.10,
        "AGG": 0.50,
    }
    for ticker, cap in caps.items():
        assert optimized.loc[
            optimized["ticker"].eq(ticker), "target_weight"
        ].le(cap + 1.0e-10).all()
    pivot = optimized.pivot(
        index=["method", "reference_week"],
        columns="ticker",
        values="target_weight",
    )
    assert (pivot["SPY"] + pivot["HYG"]).le(0.50 + 1.0e-10).all()
    assert (pivot["HYG"] + pivot["LQD"]).le(0.50 + 1.0e-10).all()
    assert (
        pivot["IEF"] + pivot["TIP"] + pivot["LQD"] + pivot["TLT"] + pivot["AGG"]
    ).le(0.75 + 1.0e-10).all()
    assert (pivot["IEF"] + pivot["TLT"] + pivot["AGG"]).le(0.60 + 1.0e-10).all()
    assert (pivot["GLD"] + pivot["DBC"] + pivot["USO"]).le(0.40 + 1.0e-10).all()
    assert (pivot["DBC"] + pivot["USO"]).le(0.30 + 1.0e-10).all()

    assert len(audit) == 2 * 446
    assert audit["optimization_outcome"].eq("optimal").all()
    assert audit["fallback_used"].eq(False).all()  # noqa: E712
    assert audit["maximum_constraint_violation"].le(1.0e-7).all()
    assert audit["weight_sum"].to_numpy() == pytest.approx(np.ones(len(audit)))
    assert estimates["training_count"].ge(260).all()
    assert (
        estimates["knowledge_cutoff"]
        == estimates["signal_date"] - pd.Timedelta(days=1)
    ).all()
    assert (
        estimates["last_training_reference_week"] < estimates["signal_date"]
    ).all()


def test_performance_and_paired_inference_match_published_results() -> None:
    performance = pd.read_csv(PUBLISHED / "performance_summary.csv").set_index(
        "method"
    )
    uncertainty = pd.read_csv(PUBLISHED / "strategy_uncertainty.csv")
    change = pd.read_csv(PUBLISHED / "universe_change_uncertainty.csv")

    assert set(performance.index) == METHODS
    assert performance["weeks"].eq(445).all()
    posterior = performance.loc[POSTERIOR]
    pooled = performance.loc[POOLED]
    assert posterior["total_return"] == pytest.approx(1.0571408584534825)
    assert posterior["cagr"] == pytest.approx(0.0879429790864263)
    assert posterior["annualized_volatility"] == pytest.approx(
        0.0994296527250163
    )
    assert posterior["sharpe_zero_rate"] == pytest.approx(0.8982364873334456)
    assert pooled["total_return"] == pytest.approx(1.0112907224550365)
    assert pooled["cagr"] == pytest.approx(0.0850811767485060)
    assert performance.loc[EQUAL, "cagr"] == pytest.approx(0.0564943909344692)
    assert performance.loc[STATIC, "cagr"] == pytest.approx(0.0959515824931500)

    posterior_pooled = uncertainty.loc[
        uncertainty["comparator_method"].eq(POOLED)
    ].iloc[0]
    assert posterior_pooled["annualized_mean_difference"] == pytest.approx(
        0.0024545205249174
    )
    assert posterior_pooled[
        "annualized_mean_difference_ci_lower"
    ] == pytest.approx(-0.0021395180358249)
    assert posterior_pooled[
        "annualized_mean_difference_ci_upper"
    ] == pytest.approx(0.0070489333189481)
    assert posterior_pooled["probability_mean_difference_positive"] == pytest.approx(
        0.8588
    )

    posterior_change = change.loc[change["method"].eq(POSTERIOR)].iloc[0]
    pooled_change = change.loc[change["method"].eq(POOLED)].iloc[0]
    assert posterior_change["annualized_mean_difference"] == pytest.approx(
        -0.0102385642173618
    )
    assert pooled_change["annualized_mean_difference"] == pytest.approx(
        -0.0125060188262041
    )
    assert not bool(posterior_change["statistical_evidence_of_improvement"])
    assert not bool(pooled_change["statistical_evidence_of_improvement"])
    assert not change["statistical_evidence_of_improvement"].any()


def test_static_control_latest_target_and_failure_attribution_are_auditable() -> None:
    expanded = pd.read_csv(
        PUBLISHED / "weekly_returns.csv", parse_dates=["reference_week"]
    )
    frozen = pd.read_csv(
        ROOT / "results/published/m02_regime_allocation_backtest/weekly_returns.csv",
        parse_dates=["reference_week"],
    )
    expanded_static = expanded.loc[expanded["method"].eq(STATIC)].set_index(
        "reference_week"
    )
    frozen_static = frozen.loc[frozen["method"].eq(STATIC)].set_index(
        "reference_week"
    )
    assert expanded_static.index.equals(frozen_static.index)
    for column in (
        "gross_return",
        "transaction_cost_rate",
        "net_return",
        "gross_turnover",
        "one_way_turnover",
    ):
        assert expanded_static[column].to_numpy() == pytest.approx(
            frozen_static[column].to_numpy(), abs=1.0e-14
        )

    latest = json.loads(
        (PUBLISHED / "latest_allocation.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (PUBLISHED / "expanded_universe_summary.json").read_text(encoding="utf-8")
    )
    assert latest["posterior_refit"] is False
    assert latest["reference_week"] == "2026-07-13T00:00:00"
    assert latest["holding_period_complete"] is False
    assert latest["universe"] == list(ASSETS)
    assert len(latest["targets"][EQUAL]) == 12
    assert summary["universe"]["external_retrieval_used"] is False
    assert summary["universe"]["sufficient_for_controlled_diagnostic"] is True
    assert (
        summary["universe"]["sufficient_for_production_global_macro_universe"]
        is False
    )
    assert summary["schedule_validation"][
        "all_twelve_schedule_exactly_matches_frozen_schedule"
    ]
    assert (
        summary["schedule_validation"][
            "maximum_original_asset_expected_mean_mismatch"
        ]
        <= 1.0e-12
    )

    utilization = pd.read_csv(PUBLISHED / "asset_utilization.csv")
    posterior_tlt = utilization.loc[
        utilization["method"].eq(POSTERIOR)
        & utilization["ticker"].eq("TLT")
    ].iloc[0]
    pooled_tlt = utilization.loc[
        utilization["method"].eq(POOLED)
        & utilization["ticker"].eq("TLT")
    ].iloc[0]
    assert posterior_tlt["average_target_weight"] == pytest.approx(
        0.1795389799555156
    )
    assert pooled_tlt["average_target_weight"] == pytest.approx(
        0.2062835322701436
    )
    asset_returns = pd.read_csv(PUBLISHED / "asset_return_diagnostics.csv").set_index(
        "ticker"
    )
    assert asset_returns.loc[
        "TLT", "realized_annualized_arithmetic_return"
    ] == pytest.approx(-0.0082238788292045)
    assert asset_returns.loc[
        "TLT", "posterior_optimized_average_expected_annual_return"
    ] == pytest.approx(0.0474501753931716)
    risk_change = summary["risk_estimator_change"]
    assert risk_change["frozen_average_ledoit_wolf_shrinkage"] == pytest.approx(
        0.0382976739445414
    )
    assert risk_change["expanded_average_ledoit_wolf_shrinkage"] == pytest.approx(
        0.0175780682018612
    )
    assert risk_change[
        "average_absolute_relative_original_asset_variance_change"
    ] == pytest.approx(0.0292374142837738)
