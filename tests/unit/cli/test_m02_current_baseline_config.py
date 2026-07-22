"""Lock the compact current-baseline publication and replay contract.

These tests intentionally stop at configuration assembly. They protect the
five-model governance surface, the reduced-core evidence allowlist, the
distinction between engine and public roles, conservative comparison labels,
and the rule that promotion writes to a new namespace instead of overwriting
any retained sensitivity stage.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from regime_allocation.cli.build_m02_current_baseline import (
    BASELINE_ID,
    BASELINE_OBSERVATION_MODELS,
    ENGINE_BENCHMARK_IDS,
    EXPECTATIONS_OBSERVATION_MODELS,
    EXPECTATIONS_SENSITIVITY_ID,
    LEGACY_BASELINE_ID,
    PUBLICATION_BENCHMARK_IDS,
    _flat_comparisons,
    _load_current_config,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/models/m02_current_baseline.yaml"


def test_current_stage_declares_only_the_compact_five_model_replay() -> None:
    """The public stage contains one baseline, three benchmarks, and one sensitivity."""

    config, raw = _load_current_config(CONFIG)
    assert raw

    variants = {str(row["id"]): row for row in config["variants"]}
    assert set(variants) == {
        BASELINE_ID,
        EXPECTATIONS_SENSITIVITY_ID,
        "transition_only",
        "partial_only",
        LEGACY_BASELINE_ID,
    }
    assert len(variants) == 5

    selection = config["model_selection"]
    assert selection["baseline"] == BASELINE_ID
    assert tuple(selection["engine_major_benchmarks"]) == ENGINE_BENCHMARK_IDS
    assert tuple(selection["publication_benchmarks"]) == PUBLICATION_BENCHMARK_IDS
    assert selection["sensitivities"] == [EXPECTATIONS_SENSITIVITY_ID]

    publication_roles = {
        variant_id: str(row["publication_role"])
        for variant_id, row in variants.items()
    }
    assert publication_roles == {
        BASELINE_ID: "baseline",
        "transition_only": "major_benchmark",
        "partial_only": "major_benchmark",
        LEGACY_BASELINE_ID: "major_benchmark",
        EXPECTATIONS_SENSITIVITY_ID: "sensitivity",
    }

    comparisons = _flat_comparisons(config)
    assert len(comparisons) == 4
    assert {
        (str(row["candidate_id"]), str(row["reference_id"]))
        for row in comparisons
    } == {
        (BASELINE_ID, "transition_only"),
        (BASELINE_ID, "partial_only"),
        (BASELINE_ID, LEGACY_BASELINE_ID),
        (EXPECTATIONS_SENSITIVITY_ID, BASELINE_ID),
    }


def test_reduced_core_allowlist_and_expectations_sensitivity_are_exact() -> None:
    """No rejected evidence family can leak into the selected baseline."""

    config, _ = _load_current_config(CONFIG)
    variants = {str(row["id"]): row for row in config["variants"]}
    evidence_sets = config["evidence_sets"]

    baseline_models = frozenset(
        str(value)
        for value in evidence_sets[
            str(variants[BASELINE_ID]["evidence_set"])
        ]["observation_models"]
    )
    sensitivity_models = frozenset(
        str(value)
        for value in evidence_sets[
            str(variants[EXPECTATIONS_SENSITIVITY_ID]["evidence_set"])
        ]["observation_models"]
    )

    assert baseline_models == BASELINE_OBSERVATION_MODELS == frozenset(
        {
            "weekly_labor_stress",
            "consumer_real_implicit_joint",
            "business_activity_pipeline_joint",
        }
    )
    assert sensitivity_models == EXPECTATIONS_OBSERVATION_MODELS
    assert sensitivity_models - baseline_models == {"inflation_expectations"}
    assert baseline_models - sensitivity_models == set()

    forbidden_models = {
        # JOLTS and aggregate housing.
        "monthly_labor_demand",
        "housing_activity",
        # Standalone price and expectations blocks.
        "inflation_expectations",
        "inflation_input_costs",
        "inflation_import_prices",
        # Vehicle-sales and continued-claims alternatives.
        "consumer_vehicle_units",
        "weekly_continued_claims",
        "weekly_continued_claims_conditional",
    }
    assert baseline_models.isdisjoint(forbidden_models)


def test_only_the_atomic_expectations_add_one_is_labeled_primary_atomic() -> None:
    """Multi-model baseline declarations are descriptive, not atomic tests."""

    config, _ = _load_current_config(CONFIG)
    comparisons = {
        str(row["id"]): row for row in _flat_comparisons(config)
    }

    assert comparisons[
        f"{BASELINE_ID}__vs__transition_only"
    ]["primary_atomic"] is False
    assert comparisons[
        f"{BASELINE_ID}__vs__partial_only"
    ]["primary_atomic"] is False
    assert comparisons[
        f"{BASELINE_ID}__vs__{LEGACY_BASELINE_ID}"
    ]["primary_atomic"] is False
    assert comparisons[
        f"{EXPECTATIONS_SENSITIVITY_ID}__vs__{BASELINE_ID}"
    ]["primary_atomic"] is True


def test_current_outputs_use_a_new_namespace_and_leave_history_addressable() -> None:
    """Promotion outputs cannot alias any immutable sensitivity namespace."""

    config, _ = _load_current_config(CONFIG)
    outputs = config["outputs"]

    assert config["stage_id"] == "current_baseline"
    assert PurePosixPath(str(outputs["processed_dir"])) == PurePosixPath(
        "data/processed/m02_soft_composite/current_baseline"
    )
    assert PurePosixPath(str(outputs["published_dir"])) == PurePosixPath(
        "results/published/m02_soft_composite/current"
    )
    assert PurePosixPath(str(outputs["manifest"])) == PurePosixPath(
        "data/manifests/m02_current_baseline.json"
    )

    archived = config["archived_sensitivity_namespaces"]
    archived_published_dirs = {
        PurePosixPath(str(stage["published_dir"])) for stage in archived.values()
    }
    archived_manifests = {
        PurePosixPath(str(stage["manifest"])) for stage in archived.values()
    }
    assert PurePosixPath(str(outputs["published_dir"])) not in archived_published_dirs
    assert PurePosixPath(str(outputs["manifest"])) not in archived_manifests
    assert set(archived) == {
        "inference_sensitivities",
        "evidence_block_inference",
        "existing_block_attribution",
        "feature_revision",
    }
