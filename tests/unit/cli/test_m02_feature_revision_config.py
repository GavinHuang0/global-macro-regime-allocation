"""Configuration contracts for the Model 02 feature-revision experiment."""

from __future__ import annotations

from pathlib import Path

from regime_allocation.cli.build_m02_feature_revision import (
    _flat_comparisons,
    _load_feature_config,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/models/m02_feature_revision.yaml"


def test_feature_revision_has_frozen_grid_and_user_exclusions() -> None:
    config, raw = _load_feature_config(CONFIG)
    assert raw
    assert len(config["variants"]) == 26
    comparisons = _flat_comparisons(config)
    assert len(comparisons) == 23
    controls = set(config["model_selection"]["frozen_controls"])
    evidence_sets = config["evidence_sets"]
    variants = {row["id"]: row for row in config["variants"]}
    for variant_id, variant in variants.items():
        if variant_id in controls:
            continue
        models = set(evidence_sets[variant["evidence_set"]]["observation_models"])
        assert "monthly_labor_demand" not in models
        assert "housing_activity" not in models


def test_business_revision_is_not_the_legacy_response_pair() -> None:
    config, _ = _load_feature_config(CONFIG)
    candidate = config["candidate_observation_models"]
    joint = candidate["business_activity_pipeline_joint"]
    assert joint["event_block"] == "business_investment_activity_pipeline"
    assert joint["responses"] == [
        "core_capital_goods_shipments_log_change",
        "core_capital_goods_orders_shipments_gap_log_change",
    ]
    assert joint["exact_zero_mask"][1] == [False, True]
    revised = set(
        config["evidence_sets"]["business_revised_joint"]["observation_models"]
    )
    legacy = set(config["evidence_sets"]["business_legacy"]["observation_models"])
    assert "business_activity_pipeline_joint" in revised
    assert "business_investment" not in revised
    assert "business_investment" in legacy


def test_comparison_atomicity_is_explicitly_conservative() -> None:
    config, _ = _load_feature_config(CONFIG)
    comparisons = {row["id"]: row for row in _flat_comparisons(config)}
    assert comparisons[
        "consumer_revised_full__vs__consumer_legacy"
    ]["primary_atomic"] is False
    assert comparisons[
        "consumer_without_real__vs__consumer_revised_full"
    ]["primary_atomic"] is False
    assert comparisons[
        "business_revised_joint__vs__business_legacy"
    ]["primary_atomic"] is False
    assert comparisons[
        "business_activity_only__vs__business_none"
    ]["primary_atomic"] is True
    assert comparisons[
        "claims_conditional_joint__vs__claims_independent"
    ]["primary_atomic"] is False


def test_conditional_claims_contract_does_not_advertise_unenforced_guard() -> None:
    config, _ = _load_feature_config(CONFIG)
    conditional = config["candidate_observation_models"][
        "weekly_continued_claims_conditional"
    ]
    assert conditional["minimum_training_samples"] == 104
    assert "minimum_distinct_target_months" not in conditional
    assert conditional["controls"] == [
        "same_publication_initial_claims_innovation_control"
    ]
    assert conditional["exact_zero_mask"] == [[False, True]]
