"""Lock the public Model 01 Bayesian-filter configuration contract.

The tests read the repository's frozen YAML file and assert the baseline
Student-t parameters, one-at-a-time sensitivity grid, and the decision to use
only ICSA in the weekly-claims likelihood. They produce no artifacts and do not
run inference; their role is to make accidental specification drift visible.
"""

from __future__ import annotations

from pathlib import Path

from regime_allocation.cli.build_m01_inference import (
    _load_config,
    _sensitivity_specifications,
)


CONFIG_PATH = Path("configs/models/m01_event_driven_bayesian_filter.yaml")


def test_frozen_inference_configuration_and_sensitivity_grid_load() -> None:
    config, raw = _load_config(CONFIG_PATH)
    specifications = _sensitivity_specifications(config)

    assert raw
    assert specifications[0].specification_id == "baseline"
    assert specifications[0].degrees_of_freedom == 7.0
    assert specifications[0].kappa == 5.0
    assert specifications[0].covariance_method == "ledoit_wolf"
    assert len(specifications) == 16
    identifiers = {item.specification_id for item in specifications}
    assert "sensitivity_gaussian" in identifiers
    assert "sensitivity_kappa_0" in identifiers
    assert "sensitivity_covariance_empirical" in identifiers
    assert "sensitivity_scale_1.25" in identifiers


def test_weekly_claims_likelihood_is_explicitly_icsa_only() -> None:
    config, _ = _load_config(CONFIG_PATH)

    claims = config["blocks"]["weekly_claims"]
    assert claims["source_series"] == ["ICSA"]
    assert claims["feature_names"] == ["initial_claims_innovation"]
