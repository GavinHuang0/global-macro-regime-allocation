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
ANCHORED = "pooled_anchor_posterior_25pct"
PROMOTED_VARIANT = "student_t_7_reduced_core"
PUBLIC_METHODS = {
    BASELINE,
    ANCHORED,
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
    assert manifest["benchmark_count"] == 4
    assert manifest["alternative_strategy_count"] == 1
    assert manifest["anchored_active_sleeve"] == {
        "method": ANCHORED,
        "role": "exploratory_not_promoted",
        "replaces_promoted_baseline": False,
        "construction": "convex_combination_of_weekly_targets",
        "pooled_core_method": "pooled_mean_optimizer",
        "pooled_core_fraction": 0.75,
        "posterior_sleeve_method": BASELINE,
        "posterior_sleeve_fraction": 0.25,
        "transaction_cost_policy": "simulate_consolidated_target_path",
    }
    assert manifest["horizon_alignment"] == {
        "estimation_return_frequency": "weekly_open_to_open",
        "holding_return_frequency": "weekly_open_to_open",
        "aligned": True,
        "minimum_labeled_weeks": 260,
        "regime_mean_pseudo_weeks": 104.0,
        "covariance_annualization_factor": 52.0,
    }

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
    anchored = weights.loc[weights["method"].eq(ANCHORED)]
    assert anchored["specification_type"].eq("anchored_strategy").all()
    assert anchored["pooled_core_fraction"].eq(0.75).all()
    assert anchored["posterior_sleeve_fraction"].eq(0.25).all()
    pivot = weights.loc[
        weights["method"].isin((BASELINE, "pooled_mean_optimizer", ANCHORED))
    ].pivot(index=["reference_week", "ticker"], columns="method", values="target_weight")
    expected_anchor = (
        0.75 * pivot["pooled_mean_optimizer"] + 0.25 * pivot[BASELINE]
    )
    assert pivot[ANCHORED].to_numpy() == pytest.approx(expected_anchor.to_numpy())
    active_identity = pivot[ANCHORED] - pivot["pooled_mean_optimizer"]
    full_active = pivot[BASELINE] - pivot["pooled_mean_optimizer"]
    assert active_identity.to_numpy() == pytest.approx(
        (0.25 * full_active).to_numpy()
    )
    caps = {
        "SPY": 0.35,
        "IEF": 0.50,
        "TIP": 0.40,
        "HYG": 0.25,
        "BIL": 1.00,
        "GLD": 0.25,
        "LQD": 0.40,
        "AGG": 0.0,
    }
    for ticker, cap in caps.items():
        assert anchored.loc[anchored["ticker"].eq(ticker), "target_weight"].le(
            cap + 1.0e-10
        ).all()
    anchor_pivot = anchored.pivot(
        index="reference_week", columns="ticker", values="target_weight"
    )
    assert (anchor_pivot["SPY"] + anchor_pivot["HYG"]).le(0.50 + 1.0e-10).all()
    assert (anchor_pivot["HYG"] + anchor_pivot["LQD"]).le(0.50 + 1.0e-10).all()
    assert (anchor_pivot["IEF"] + anchor_pivot["TIP"] + anchor_pivot["LQD"]).le(
        0.75 + 1.0e-10
    ).all()

    return_pivot = returns.loc[
        returns["method"].isin((BASELINE, "pooled_mean_optimizer", ANCHORED))
    ].pivot(index="reference_week", columns="method", values="gross_return")
    expected_gross = (
        0.75 * return_pivot["pooled_mean_optimizer"]
        + 0.25 * return_pivot[BASELINE]
    )
    assert return_pivot[ANCHORED].to_numpy() == pytest.approx(
        expected_gross.to_numpy(), abs=1.0e-14
    )
    anchor_returns = returns.loc[returns["method"].eq(ANCHORED)]
    expected_net = (
        (1.0 - anchor_returns["transaction_cost_rate"])
        * (1.0 + anchor_returns["gross_return"])
        - 1.0
    )
    assert anchor_returns["net_return"].to_numpy() == pytest.approx(
        expected_net.to_numpy()
    )


def test_performance_and_uncertainty_match_the_published_summary() -> None:
    performance = pd.read_csv(PUBLISHED / "performance_summary.csv").set_index(
        "method"
    )
    uncertainty = pd.read_csv(PUBLISHED / "comparison_uncertainty.csv")
    assert set(performance.index) == PUBLIC_METHODS
    assert performance["weeks"].eq(445).all()
    baseline = performance.loc[BASELINE]
    pooled = performance.loc["pooled_mean_optimizer"]
    anchored = performance.loc[ANCHORED]
    assert baseline["total_return"] == pytest.approx(1.2360208644390345)
    assert baseline["cagr"] == pytest.approx(0.09859502405745069)
    assert baseline["annualized_volatility"] == pytest.approx(0.104169886675157)
    assert pooled["total_return"] == pytest.approx(1.2307803607618921)
    assert pooled["cagr"] == pytest.approx(0.09829384274557884)
    assert anchored["total_return"] == pytest.approx(1.2321738661604904)
    assert anchored["cagr"] == pytest.approx(0.09837399098006361)
    assert anchored["annualized_volatility"] == pytest.approx(0.1048090914705943)
    assert performance.loc["equal_weight", "total_return"] == pytest.approx(
        0.6585749519044164
    )
    assert performance.loc["legacy_sharpe_map", "total_return"] == pytest.approx(
        0.5409061349076736
    )
    assert performance.loc[
        "static_60_spy_40_agg", "total_return"
    ] == pytest.approx(1.1903942191641814)
    assert uncertainty["block_length"].eq(26).all()
    assert uncertainty["n_resamples"].eq(10_000).all()
    assert len(uncertainty) == 6
    pooled_row = uncertainty.loc[
        uncertainty["baseline_method"].eq(BASELINE)
        & uncertainty["comparator_method"].eq("pooled_mean_optimizer")
    ].iloc[0]
    assert pooled_row["annualized_mean_difference"] == pytest.approx(
        0.0001870659160750313
    )
    assert pooled_row["annualized_mean_difference_ci_lower"] < 0.0
    assert pooled_row["annualized_mean_difference_ci_upper"] > 0.0
    posterior_anchor = uncertainty.loc[
        uncertainty["baseline_method"].eq(BASELINE)
        & uncertainty["comparator_method"].eq(ANCHORED)
    ].iloc[0]
    anchor_pooled = uncertainty.loc[
        uncertainty["baseline_method"].eq(ANCHORED)
        & uncertainty["comparator_method"].eq("pooled_mean_optimizer")
    ].iloc[0]
    assert posterior_anchor["annualized_mean_difference"] > 0.0
    assert posterior_anchor["annualized_mean_difference"] == pytest.approx(
        0.00013713981221743173
    )
    assert posterior_anchor["annualized_mean_difference_ci_lower"] < 0.0
    assert posterior_anchor["annualized_mean_difference_ci_upper"] > 0.0
    assert anchor_pooled["annualized_mean_difference"] > 0.0
    assert anchor_pooled["annualized_mean_difference"] == pytest.approx(
        4.992610385759956e-05
    )
    assert anchor_pooled["annualized_mean_difference_ci_lower"] < 0.0
    assert anchor_pooled["annualized_mean_difference_ci_upper"] > 0.0


def test_optimizer_audit_is_feasible_and_uses_prior_day_knowledge() -> None:
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
    assert {
        "pseudo_weeks",
        "expected_portfolio_weekly_return",
        "net_expected_weekly_return",
    }.issubset(audit.columns)
    assert {
        "kappa",
        "expected_portfolio_monthly_return",
        "net_expected_monthly_return",
    }.isdisjoint(audit.columns)
    assert {
        "pseudo_weeks",
        "first_training_reference_week",
        "last_training_reference_week",
    }.issubset(estimates.columns)
    assert {
        "kappa",
        "first_training_holding_month",
        "last_training_holding_month",
    }.isdisjoint(estimates.columns)
    assert audit["maximum_constraint_violation"].le(1.0e-7).all()
    assert audit["weight_sum"].to_numpy() == pytest.approx(np.ones(len(audit)))
    assert audit["optimization_outcome"].notna().all()
    assert estimates["training_count"].ge(260).all()
    assert (
        estimates["knowledge_cutoff"]
        == estimates["signal_date"] - pd.Timedelta(days=1)
    ).all()
    assert estimates["first_training_reference_week"].dt.dayofweek.eq(0).all()
    assert estimates["last_training_reference_week"].dt.dayofweek.eq(0).all()
    assert (
        estimates["last_training_reference_week"] < estimates["signal_date"]
    ).all()
    baseline_estimates = estimates.loc[estimates["method"].eq(BASELINE)]
    first_fit = baseline_estimates.loc[
        baseline_estimates["reference_week"].eq(pd.Timestamp("2018-01-01"))
    ]
    latest_fit = baseline_estimates.loc[
        baseline_estimates["reference_week"].eq(pd.Timestamp("2026-07-13"))
    ]
    assert first_fit["training_count"].eq(517).all()
    assert first_fit["first_training_reference_week"].eq(
        pd.Timestamp("2008-01-07")
    ).all()
    assert first_fit["last_training_reference_week"].eq(
        pd.Timestamp("2017-11-27")
    ).all()
    assert latest_fit["training_count"].eq(952).all()
    assert latest_fit["last_training_reference_week"].eq(
        pd.Timestamp("2026-05-25")
    ).all()
    assert audit["net_expected_weekly_return"].to_numpy() == pytest.approx(
        (
            audit["expected_portfolio_weekly_return"]
            - audit["estimated_transaction_cost"]
        ).to_numpy()
    )
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
    assert latest["baseline_parameters"]["estimation_return_frequency"] == (
        "weekly_open_to_open"
    )
    assert latest["baseline_parameters"]["minimum_labeled_weeks"] == 260
    assert latest["baseline_parameters"]["regime_mean_pseudo_weeks"] == 104.0
    assert "estimated_weekly_return" in latest
    assert "estimated_monthly_return" not in latest
    target = {row["ticker"]: row["weight"] for row in latest["target_weights"]}
    assert target == pytest.approx(
        {"SPY": 0.35, "HYG": 0.15, "GLD": 0.25, "LQD": 0.25}
    )
    anchored = latest["exploratory_anchored_strategy"]
    assert anchored["method"] == ANCHORED
    assert anchored["role"] == "exploratory_not_promoted"
    assert anchored["replaces_promoted_baseline"] is False
    assert anchored["pooled_core_fraction"] == 0.75
    assert anchored["posterior_sleeve_fraction"] == 0.25
    anchored_target = {
        row["ticker"]: row["weight"] for row in anchored["target_weights"]
    }
    assert anchored_target == pytest.approx(target)
    assert sum(row["probability"] for row in latest["posterior"]) == pytest.approx(1.0)


def test_summary_declares_aligned_weekly_forecast_and_holding_horizons() -> None:
    summary = json.loads(
        (PUBLISHED / "backtest_summary.json").read_text(encoding="utf-8")
    )

    assert summary["estimation_return_frequency"] == "weekly_open_to_open"
    assert summary["holding_return_frequency"] == "weekly_open_to_open"
    assert summary["forecast_horizon"] == summary["holding_horizon"] == "one_week"
    assert summary["forecast_holding_horizons_aligned"] is True
    assert summary["minimum_labeled_weeks"] == 260
    assert summary["regime_mean_pseudo_weeks"] == 104.0
    assert summary["covariance_annualization_factor"] == 52.0
    assert "monthly_estimation_retained" not in summary
    sleeve = summary["exploratory_anchored_strategy"]
    assert sleeve["method"] == ANCHORED
    assert sleeve["role"] == "exploratory_not_promoted"
    assert sleeve["replaces_promoted_baseline"] is False
    assert sleeve["pooled_core_fraction"] == 0.75
    assert sleeve["posterior_sleeve_fraction"] == 0.25


def test_sensitivity_artifact_names_shrinkage_in_weekly_units() -> None:
    sensitivity = pd.read_csv(PUBLISHED / "sensitivity_metrics.csv")

    assert "pseudo_weeks" in sensitivity
    assert "kappa" not in sensitivity
    kappa_rows = sensitivity.loc[
        sensitivity["parameter"].eq("regime_mean_pseudo_weeks")
    ]
    assert set(kappa_rows["parameter_value"]) == {0.0, 52.0, 208.0, 416.0}
    assert set(kappa_rows["pseudo_weeks"]) == {0.0, 52.0, 208.0, 416.0}
