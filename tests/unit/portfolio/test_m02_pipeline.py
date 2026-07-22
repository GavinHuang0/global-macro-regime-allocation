"""Lock the Model 02 weekly allocation policy and namespace."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath

import pytest
import yaml

from regime_allocation.portfolio.m02_pipeline import (
    PERIODS_PER_YEAR,
    PROMOTED_BASELINE_ID,
    _allocation_specifications,
    _load_config,
    _validate_model01_policy_parity,
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
    assert specifications[0].kappa == 24.0
    assert specifications[0].volatility_cap == 0.10
    assert specifications[0].transaction_cost == 0.0005
    assert config["signal"]["variant_id"] == PROMOTED_BASELINE_ID
    assert config["estimation"]["return_frequency"] == "monthly_open_to_open"
    assert config["optimization"]["covariance_annualization_factor"] == 12.0
    assert config["evaluation"]["periods_per_year"] == PERIODS_PER_YEAR == 52
    assert config["evaluation"]["uncertainty"]["block_length_weeks"] == 26


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
