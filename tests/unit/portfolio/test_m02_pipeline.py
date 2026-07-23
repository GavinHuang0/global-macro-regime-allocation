"""Lock the Model 02 weekly allocation policy and namespace."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import regime_allocation.portfolio.m02_pipeline as m02_pipeline
from regime_allocation.portfolio.m02_pipeline import (
    PERIODS_PER_YEAR,
    PROMOTED_BASELINE_ID,
    _allocation_specifications,
    _load_config,
    _validate_model01_policy_parity,
    _weekly_estimator_history,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/models/m02_regime_allocation_backtest.yaml"
M01_TEMPLATE = ROOT / "configs/models/m01_regime_allocation_backtest.yaml"


def test_weekly_config_reuses_the_model01_portfolio_policy() -> None:
    config, raw = _load_config(CONFIG)
    template = yaml.safe_load(M01_TEMPLATE.read_text(encoding="utf-8"))

    assert raw
    _validate_model01_policy_parity(config, template)
    specifications = _allocation_specifications(config)
    assert len(specifications) == 11
    assert specifications[0].kappa == 104.0
    assert specifications[0].volatility_cap == 0.10
    assert specifications[0].transaction_cost == 0.0005
    assert config["signal"]["variant_id"] == PROMOTED_BASELINE_ID
    assert config["estimation"]["return_frequency"] == "weekly_open_to_open"
    assert config["estimation"]["minimum_labeled_weeks"] == 260
    assert (
        config["estimation"]["regime_labels"]["weekly_association"]
        == "calendar_month_containing_reference_week_monday"
    )
    assert (
        config["estimation"]["regime_means"]["estimator"]
        == "pooled_mean_pseudo_week_shrinkage"
    )
    assert config["estimation"]["regime_means"]["pseudo_weeks"] == 104.0
    assert (
        config["estimation"]["regime_means"]["formula"]
        == template["estimation"]["regime_means"]["formula"]
    )
    assert (
        config["estimation"]["regime_means"]["empty_regime_policy"]
        == template["estimation"]["regime_means"]["empty_regime_policy"]
    )
    assert (
        config["optimization"]["objective"]
        == "maximize_one_week_expected_return_net_of_estimated_trading_cost"
    )
    assert config["optimization"]["covariance_annualization_factor"] == 52.0
    assert config["sensitivity"]["parameters"]["regime_mean_pseudo_weeks"] == [
        0.0,
        52.0,
        104.0,
        208.0,
        416.0,
    ]
    assert config["evaluation"]["periods_per_year"] == PERIODS_PER_YEAR == 52
    assert config["evaluation"]["uncertainty"]["block_length_weeks"] == 26
    sleeve = config["additional_strategies"][
        "pooled_anchor_posterior_active_sleeve"
    ]
    assert sleeve["method_id"] == m02_pipeline.ANCHORED_METHOD
    assert sleeve["role"] == "exploratory_not_promoted"
    assert sleeve["replaces_promoted_baseline"] is False
    assert sleeve["pooled_core_fraction"] == 0.75
    assert sleeve["posterior_sleeve_fraction"] == 0.25


def test_weekly_outputs_use_a_separate_model02_namespace() -> None:
    config, _ = _load_config(CONFIG)
    outputs = config["outputs"]

    assert PurePosixPath(str(outputs["processed_dir"])) == PurePosixPath(
        "data/processed/m02_regime_allocation_backtest"
    )
    assert PurePosixPath(str(outputs["published_dir"])) == PurePosixPath(
        "results/published/m02_regime_allocation_backtest"
    )
    assert PurePosixPath(str(outputs["manifest"])) == PurePosixPath(
        "data/manifests/m02_regime_allocation_backtest.json"
    )
    assert "m01_regime_allocation_backtest" not in str(outputs)
    assert "m02_soft_composite/current" not in str(outputs)


def test_weekly_estimator_history_excludes_a_source_truncated_leading_week() -> None:
    returns = pd.DataFrame(
        {
            "reference_week": pd.to_datetime(
                ["2007-12-31", "2008-01-07", "2008-01-14"]
            ),
            "start_date": pd.to_datetime(
                ["2008-01-02", "2008-01-07", "2008-01-14"]
            ),
            "end_date": pd.to_datetime(
                ["2008-01-07", "2008-01-14", "2008-01-22"]
            ),
            "SPY": [0.01, 0.02, 0.03],
            "IEF": [0.00, 0.01, -0.01],
        }
    )

    history = _weekly_estimator_history(
        returns,
        strategy_assets=("SPY", "IEF"),
    )

    assert history["reference_week"].tolist() == [
        pd.Timestamp("2008-01-07"),
        pd.Timestamp("2008-01-14"),
    ]
    assert history["return_available_at"].tolist() == [
        pd.Timestamp("2008-01-14"),
        pd.Timestamp("2008-01-22"),
    ]
    assert "start_date" not in history


def _active_sleeve_source_weights() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    dates = (pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-13"))
    source_weights = {
        m02_pipeline.POOLED_METHOD: ((0.6, 0.4), (0.5, 0.5)),
        m02_pipeline.BASELINE_METHOD: ((0.2, 0.8), (1.0, 0.0)),
    }
    for method, weekly_weights in source_weights.items():
        for reference_week, values in zip(dates, weekly_weights, strict=True):
            for ticker, weight in zip(("SPY", "IEF"), values, strict=True):
                records.append(
                    {
                        "method": method,
                        "specification_type": "source",
                        "reference_week": reference_week,
                        "signal_date": reference_week,
                        "execution_date": reference_week,
                        "regime_reference_month": pd.Timestamp("2020-01-01"),
                        "ticker": ticker,
                        "target_weight": weight,
                        "is_live_only": reference_week == dates[-1],
                        "posterior_map_regime": "growth_up_inflation_up",
                        "optimization_outcome": "optimal",
                    }
                )
    return pd.DataFrame.from_records(records)


def test_anchored_active_sleeve_is_exact_75_25_target_blend() -> None:
    source = _active_sleeve_source_weights()

    anchored = m02_pipeline._anchored_active_sleeve_targets(source)

    expected = {
        (pd.Timestamp("2020-01-06"), "SPY"): 0.50,
        (pd.Timestamp("2020-01-06"), "IEF"): 0.50,
        (pd.Timestamp("2020-01-13"), "SPY"): 0.625,
        (pd.Timestamp("2020-01-13"), "IEF"): 0.375,
    }
    actual = anchored.set_index(["reference_week", "ticker"])["target_weight"]
    assert actual.to_dict() == pytest.approx(expected)
    assert anchored["method"].eq(m02_pipeline.ANCHORED_METHOD).all()
    assert anchored["specification_type"].eq("anchored_strategy").all()
    assert anchored["pooled_core_fraction"].eq(0.75).all()
    assert anchored["posterior_sleeve_fraction"].eq(0.25).all()
    assert anchored.groupby("reference_week")["target_weight"].sum().eq(1.0).all()
    assert anchored.loc[
        anchored["reference_week"].eq(pd.Timestamp("2020-01-13")),
        "is_live_only",
    ].all()
    identity = anchored["posterior_active_deviation"] - 0.25 * (
        anchored["posterior_sleeve_target_weight"]
        - anchored["pooled_core_target_weight"]
    )
    assert identity.to_numpy() == pytest.approx(np.zeros(len(identity)))


@pytest.mark.parametrize("failure", ["missing", "duplicate", "metadata"])
def test_anchored_active_sleeve_rejects_misaligned_source_legs(failure: str) -> None:
    source = _active_sleeve_source_weights()
    active = source["method"].eq(m02_pipeline.BASELINE_METHOD)
    if failure == "missing":
        source = source.drop(source.index[active][0])
        match = "different week/ticker keys"
    elif failure == "duplicate":
        source = pd.concat([source, source.loc[[source.index[active][0]]]])
        match = "duplicate keys"
    else:
        source.loc[source.index[active][0], "signal_date"] = pd.Timestamp("2020-01-07")
        match = "metadata differ"

    with pytest.raises(ValueError, match=match):
        m02_pipeline._anchored_active_sleeve_targets(source)


def test_dynamic_targets_annualizes_weekly_covariance_by_52(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = ("SPY", "IEF")
    weekly_covariance = pd.DataFrame(
        [[0.002, 0.0003], [0.0003, 0.001]],
        index=assets,
        columns=assets,
    )
    moments = SimpleNamespace(
        expected_mean=pd.Series([0.001, 0.0005], index=assets),
        covariance=weekly_covariance,
    )
    captured: dict[str, object] = {}

    class FakeResult:
        weight_by_asset = {"SPY": 0.5, "IEF": 0.5}
        audit = SimpleNamespace(outcome="optimal")

        @staticmethod
        def as_array() -> np.ndarray:
            return np.array([0.5, 0.5])

    def fake_optimize(**kwargs: object) -> FakeResult:
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(
        m02_pipeline,
        "fit_causal_weekly_regime_return_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        m02_pipeline,
        "posterior_mixture_moments",
        lambda *args, **kwargs: moments,
    )
    monkeypatch.setattr(m02_pipeline, "optimize_long_only", fake_optimize)
    monkeypatch.setattr(
        m02_pipeline,
        "_record_estimate",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        m02_pipeline,
        "_record_optimizer",
        lambda **kwargs: {"method": "posterior_optimized"},
    )
    signal = {
        "reference_week": pd.Timestamp("2020-01-06"),
        "signal_date": pd.Timestamp("2020-01-06"),
        "start_date": pd.Timestamp("2020-01-06"),
        "regime_reference_month": pd.Timestamp("2020-01-01"),
        "is_complete": True,
    }
    signal.update({column: 0.25 for column in m02_pipeline.PROBABILITY_COLUMNS})
    specification = m02_pipeline.AllocationSpecification(
        method=m02_pipeline.BASELINE_METHOD,
        specification_type="baseline",
        parameter="baseline",
        parameter_value=1.0,
        kappa=104.0,
        volatility_cap=0.10,
        transaction_cost=0.0005,
        cap_multiplier=1.0,
    )

    m02_pipeline._dynamic_targets(
        specifications=(specification,),
        signals=pd.DataFrame([signal]),
        estimator_returns=pd.DataFrame(),
        regime_history=pd.DataFrame(),
        prices=pd.DataFrame({"date": [pd.Timestamp("2020-01-03")]}),
        strategy_assets=assets,
        simulation_assets=assets,
        minimum_observations=260,
        covariance_annualization_factor=52.0,
        asset_caps={asset: 1.0 for asset in assets},
        group_caps=(),
    )

    np.testing.assert_allclose(
        captured["annualized_covariance"],
        weekly_covariance.to_numpy() * 52.0,
    )
    np.testing.assert_allclose(
        captured["expected_monthly_returns"],
        moments.expected_mean.to_numpy(),
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("evaluation", "zero_rate_sharpe", False),
        ("uncertainty", "resamples", 999),
        ("uncertainty", "confidence_level", 0.90),
    ),
)
def test_model01_policy_parity_rejects_evaluation_drift(
    section: str,
    key: str,
    value: object,
) -> None:
    config, _ = _load_config(CONFIG)
    template = yaml.safe_load(M01_TEMPLATE.read_text(encoding="utf-8"))
    changed = deepcopy(config)
    target = (
        changed["evaluation"]
        if section == "evaluation"
        else changed["evaluation"]["uncertainty"]
    )
    target[key] = value

    with pytest.raises(ValueError, match="evaluation|uncertainty"):
        _validate_model01_policy_parity(changed, template)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (
            ("estimation", "regime_labels", "weekly_association"),
            "calendar_month_containing_return_end",
        ),
        (
            ("optimization", "objective"),
            "maximize_one_month_expected_return_net_of_estimated_trading_cost",
        ),
        (
            ("estimation", "regime_means", "estimator"),
            "pooled_mean_pseudo_month_shrinkage",
        ),
        (("estimation", "regime_means", "formula"), "regime_mean"),
        (("estimation", "regime_means", "empty_regime_policy"), "error"),
        (("estimation", "minimum_labeled_weeks"), 259),
        (("estimation", "regime_means", "pseudo_weeks"), 103.0),
        (
            ("sensitivity", "parameters", "regime_mean_pseudo_weeks"),
            [0.0, 52.0, 104.0, 208.0, 417.0],
        ),
        (
            (
                "additional_strategies",
                "pooled_anchor_posterior_active_sleeve",
                "posterior_sleeve_fraction",
            ),
            0.30,
        ),
    ),
    ids=(
        "weekly-association",
        "weekly-objective",
        "weekly-estimator-name",
        "shrinkage-formula",
        "empty-regime-policy",
        "minimum-weekly-history",
        "weekly-kappa",
        "weekly-kappa-sensitivity",
        "posterior-sleeve-fraction",
    ),
)
def test_model01_policy_parity_rejects_weekly_translation_drift(
    path: tuple[str, ...],
    value: object,
) -> None:
    config, _ = _load_config(CONFIG)
    template = yaml.safe_load(M01_TEMPLATE.read_text(encoding="utf-8"))
    changed = deepcopy(config)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        _validate_model01_policy_parity(changed, template)
