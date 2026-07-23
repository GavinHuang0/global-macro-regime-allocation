"""Integration contracts for the frozen-posterior active/oracle diagnostic."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/m02_active_optimizer_diagnostic.yaml"
MANIFEST = ROOT / "data/manifests/m02_active_optimizer_diagnostic.json"
UPSTREAM_MANIFEST = ROOT / "data/manifests/m02_regime_allocation_backtest.json"
PROCESSED = ROOT / "data/processed/m02_active_optimizer_diagnostic"
PUBLISHED = ROOT / "results/published/m02_active_optimizer_diagnostic"
FROZEN_SIGNAL = (
    ROOT / "data/processed/m02_regime_allocation_backtest/signal_table.csv"
)

ACTIVE = "pooled_relative_posterior_active_te100bp"
ORACLE = "oracle_regime_pooled_relative_active_te100bp"
POOLED = "pooled_mean_optimizer"
STATIC = "static_60_spy_40_agg"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_freezes_the_posterior_and_hashes_every_artifact() -> None:
    manifest = _manifest()

    assert manifest["schema_version"] == 1
    assert manifest["model_id"] == "m02_soft_composite"
    assert (
        manifest["stage_id"]
        == "benchmark_relative_active_optimizer_diagnostic"
    )
    assert manifest["configuration_sha256"] == _digest(CONFIG)
    assert manifest["frozen_upstream_manifest"] == {
        "path": UPSTREAM_MANIFEST.relative_to(ROOT).as_posix(),
        "sha256": (
            "5821824dcfe6a82347d1df176467b74d0199cb7724bb08f5e07daf08801fbc4a"
        ),
    }
    assert _digest(UPSTREAM_MANIFEST) == (
        "5821824dcfe6a82347d1df176467b74d0199cb7724bb08f5e07daf08801fbc4a"
    )
    assert manifest["frozen_posterior_signal"] == {
        "path": FROZEN_SIGNAL.relative_to(ROOT).as_posix(),
        "sha256": (
            "8cb4d27056ec5b024ad5e2cbf4549429da98def25b8602e91275731e8046111f"
        ),
        "refit": False,
    }
    assert _digest(FROZEN_SIGNAL) == manifest["frozen_posterior_signal"]["sha256"]
    assert manifest["active_method"] == ACTIVE
    assert manifest["active_weeks"] == 445
    assert manifest["significance_gate_passed"] is False
    assert manifest["oracle_triggered"] is True
    assert manifest["oracle_method"] == ORACLE
    assert manifest["oracle_primary_weeks"] == 405

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


def test_active_targets_and_own_path_accounting_satisfy_the_locked_limits() -> None:
    weights = pd.read_csv(
        PUBLISHED / "active_weekly_weights.csv",
        parse_dates=["reference_week"],
    )
    audit = pd.read_csv(
        PROCESSED / "active_optimizer_audit.csv",
        parse_dates=["reference_week"],
    )
    returns = pd.read_csv(
        PUBLISHED / "active_weekly_returns.csv",
        parse_dates=["reference_week"],
    )

    assert weights["method"].eq(ACTIVE).all()
    assert weights["reference_week"].nunique() == 446
    assert weights.groupby("reference_week")["target_weight"].sum().to_numpy() == (
        pytest.approx(np.ones(446))
    )
    assert weights.groupby("reference_week")["active_weight"].sum().to_numpy() == (
        pytest.approx(np.zeros(446), abs=1.0e-10)
    )
    assert weights["active_weight"].abs().le(0.05 + 1.0e-9).all()
    one_way = (
        weights.groupby("reference_week")["active_weight"].apply(
            lambda values: 0.5 * values.abs().sum()
        )
    )
    assert one_way.le(0.10 + 1.0e-9).all()
    assert weights.loc[weights["ticker"].eq("AGG"), "target_weight"].eq(0.0).all()

    assert len(audit) == 446
    assert audit["optimization_outcome"].eq("optimal").all()
    assert audit["fallback_used"].eq(False).all()  # noqa: E712
    assert audit["annualized_tracking_error"].le(0.01 + 1.0e-7).all()
    assert audit["annualized_total_volatility"].le(0.10 + 1.0e-7).all()
    assert audit["maximum_absolute_active_weight"].le(0.05 + 1.0e-7).all()
    assert audit["one_way_active_exposure"].le(0.10 + 1.0e-7).all()
    assert audit["maximum_constraint_violation"].le(1.0e-7).all()

    assert set(returns["method"]) == {ACTIVE, POOLED, STATIC}
    assert returns.groupby("method")["reference_week"].nunique().eq(445).all()
    active_returns = returns.loc[returns["method"].eq(ACTIVE)]
    expected_net = (
        (1.0 - active_returns["transaction_cost_rate"])
        * (1.0 + active_returns["gross_return"])
        - 1.0
    )
    assert active_returns["net_return"].to_numpy() == pytest.approx(
        expected_net.to_numpy()
    )


def test_active_strategy_fails_the_locked_gate_and_triggers_the_oracle() -> None:
    performance = pd.read_csv(PUBLISHED / "active_performance.csv").set_index(
        "method"
    )
    uncertainty = pd.read_csv(PUBLISHED / "active_uncertainty.csv")
    summary = json.loads(
        (PUBLISHED / "diagnostic_summary.json").read_text(encoding="utf-8")
    )

    active = performance.loc[ACTIVE]
    assert active["total_return"] == pytest.approx(1.1502136646707122)
    assert active["cagr"] == pytest.approx(0.0935830858935973)
    assert active["annualized_volatility"] == pytest.approx(
        0.0993605389814555
    )
    assert active["sharpe_zero_rate"] == pytest.approx(0.9508637173931768)
    assert active["maximum_drawdown"] == pytest.approx(-0.2087887392472091)
    assert active["annualized_one_way_turnover"] == pytest.approx(
        0.18895246362555
    )
    pooled = uncertainty.loc[
        uncertainty["comparator_method"].eq(POOLED)
    ].iloc[0]
    static = uncertainty.loc[
        uncertainty["comparator_method"].eq(STATIC)
    ].iloc[0]
    assert pooled["annualized_mean_difference"] == pytest.approx(
        -0.0048845088436938
    )
    assert pooled["annualized_mean_difference_ci_lower"] == pytest.approx(
        -0.0081337609260972
    )
    assert pooled["annualized_mean_difference_ci_upper"] == pytest.approx(
        -0.0016529507378339
    )
    assert static["annualized_mean_difference_ci_lower"] < 0.0
    assert static["annualized_mean_difference_ci_upper"] > 0.0
    assert summary["frozen_posterior"]["posterior_refit"] is False
    assert (
        summary["frozen_posterior"][
            "maximum_recomputed_expected_mean_mismatch"
        ]
        <= 1.0e-12
    )
    assert summary["significance_gate"]["passed"] is False
    assert summary["significance_gate"]["oracle_triggered"] is True


def test_oracle_is_hindsight_only_contiguous_and_still_loses_to_pooled() -> None:
    truth = pd.read_csv(
        PUBLISHED / "oracle_truth_audit.csv",
        parse_dates=[
            "reference_week",
            "signal_date",
            "regime_reference_month",
            "oracle_label_available_at",
        ],
    )
    weights = pd.read_csv(
        PUBLISHED / "oracle_weekly_weights.csv",
        parse_dates=["reference_week"],
    )
    audit = pd.read_csv(PROCESSED / "oracle_optimizer_audit.csv")
    performance = pd.read_csv(
        PUBLISHED / "oracle_performance.csv"
    ).set_index("method")
    uncertainty = pd.read_csv(PUBLISHED / "oracle_uncertainty.csv")

    assert len(truth) == 405
    assert truth["reference_week"].iloc[0] == pd.Timestamp("2018-01-01")
    assert truth["reference_week"].iloc[-1] == pd.Timestamp("2025-09-29")
    assert truth["reference_week"].diff().dropna().eq(pd.Timedelta(days=7)).all()
    assert truth["regime_reference_month"].nunique() == 93
    assert (
        truth["oracle_regime_id"]
        .ne(truth["oracle_regime_id"].shift())
        .sum()
        == 58
    )
    assert truth["future_information_used"].all()
    assert truth["available_at_signal"].eq(False).all()  # noqa: E712
    assert weights["method"].eq(ORACLE).all()
    assert weights["reference_week"].nunique() == 405
    assert audit["optimization_outcome"].eq("optimal").all()
    assert audit["maximum_constraint_violation"].le(1.0e-7).all()

    oracle = performance.loc[ORACLE]
    assert oracle["total_return"] == pytest.approx(1.0277520779227496)
    assert oracle["cagr"] == pytest.approx(0.0950127885100213)
    assert oracle["annualized_volatility"] == pytest.approx(
        0.0982612442407537
    )
    oracle_pooled = uncertainty.loc[
        uncertainty["baseline_method"].eq(ORACLE)
        & uncertainty["comparator_method"].eq(POOLED)
    ].iloc[0]
    assert oracle_pooled["annualized_mean_difference"] == pytest.approx(
        -0.0053905189846888
    )
    assert oracle_pooled["annualized_mean_difference_ci_lower"] == pytest.approx(
        -0.0091771468742089
    )
    assert oracle_pooled["annualized_mean_difference_ci_upper"] == pytest.approx(
        -0.0016186806155967
    )
    oracle_active = uncertainty.loc[
        uncertainty["baseline_method"].eq(ORACLE)
        & uncertainty["comparator_method"].eq(ACTIVE)
    ].iloc[0]
    assert oracle_active["annualized_mean_difference_ci_lower"] < 0.0
    assert oracle_active["annualized_mean_difference_ci_upper"] > 0.0
