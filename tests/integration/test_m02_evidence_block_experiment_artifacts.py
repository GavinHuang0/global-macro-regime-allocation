"""Lineage, causality, and scope checks for six Model 02 evidence tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_MANIFEST = ROOT / "data/manifests/m02_evidence_block_experiments.json"
INFERENCE_MANIFEST = ROOT / "data/manifests/m02_evidence_block_inference.json"
PROCESSED = ROOT / "data/processed/m02_soft_composite/evidence_block_inference"
PUBLISHED = ROOT / "results/published/m02_soft_composite/evidence_block_inference"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_generated_and_implementation_hashes_match() -> None:
    for manifest_path in (DATA_MANIFEST, INFERENCE_MANIFEST):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generated = [
            *manifest.get("generated_files", ()),
            *manifest.get("generated_file_hashes", ()),
        ]
        assert generated
        for artifact in generated:
            path = ROOT / artifact["path"]
            assert path.is_file()
            assert _digest(path) == artifact["sha256"]
            assert path.stat().st_size == artifact["bytes"]
        for source in manifest.get("implementation_file_hashes", ()):
            path = ROOT / source["path"]
            assert path.is_file()
            assert _digest(path) == source["sha256"]
            assert path.stat().st_size == source["bytes"]


def test_frozen_baseline_and_candidate_roles_are_explicit() -> None:
    manifest = json.loads(INFERENCE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["baseline_invariance_passed"] is True
    assert manifest["model_selection"]["baseline"] == "student_t_7_combined"
    assert manifest["model_selection"]["major_benchmarks"] == [
        "transition_only",
        "partial_only",
    ]

    invariance = pd.read_csv(PUBLISHED / "baseline_invariance.csv")
    assert len(invariance) == 64
    assert invariance["passed"].astype(bool).all()
    registry = pd.read_csv(PUBLISHED / "model_registry.csv").set_index("variant_id")
    assert registry.loc["student_t_7_combined", "model_role"] == "baseline"
    assert registry.loc["student_t_7_combined", "evidence_set_id"] == "legacy_baseline"
    sensitivity = registry.loc[registry["model_role"].eq("sensitivity")]
    assert sensitivity["emission_family"].eq("student_t_7").all()
    assert sensitivity["var_method"].eq("ols").all()
    assert sensitivity["partial_defining_releases"].astype(bool).all()


def test_every_candidate_profile_has_realized_causal_updates() -> None:
    coverage = pd.read_csv(PUBLISHED / "profile_coverage.csv")
    candidate = coverage.loc[coverage["variant_id"].eq("candidate_all_six")]
    required = {
        "manufacturing_empire_state",
        "manufacturing_philadelphia",
        "weekly_continued_claims",
        "consumer_retail_ex_auto_gas",
        "consumer_vehicle_units",
        "housing_single_family_construction",
        "housing_new_home_sales",
        "inflation_import_prices",
        "business_investment_backlog",
    }
    selected = candidate.loc[candidate["observation_model_id"].isin(required)]
    assert set(selected["observation_model_id"]) == required
    assert selected["selected_long_rows"].gt(0).all()
    assert selected["training_candidates"].gt(0).all()
    assert selected["applied_updates"].gt(0).all()


def test_bootstrap_and_dependence_diagnostics_are_fully_audited() -> None:
    bootstrap = pd.read_csv(PUBLISHED / "paired_block_bootstrap.csv")
    primary = bootstrap.loc[
        bootstrap["evaluation_sample"].eq("full_sample")
        & bootstrap["evaluation_checkpoint"].eq("before_any_defining_release")
    ]
    core = primary.loc[primary["holm_family_member"].astype(bool)]
    assert core["filter_variant"].nunique() == 6
    assert core["metric"].nunique() == 4
    assert core["holm_adjusted_p_value"].notna().all()
    assert primary["replications"].eq(5000).all()
    assert primary["block_length_months"].eq(12).all()

    residuals = pd.read_csv(
        PROCESSED / "dependence_residuals.csv.gz",
        parse_dates=["fit_cutoff", "last_training_available_at"],
    )
    assert len(residuals) == 2667
    assert (residuals["last_training_available_at"] < residuals["fit_cutoff"]).all()
    assert residuals["target_usage"].eq(
        "retrospective_completed_first_release_score_only"
    ).all()

    dependence = json.loads(
        (PUBLISHED / "dependence_summary.json").read_text(encoding="utf-8")
    )
    assert dependence["live_filter_usage"] == "none"
    assert dependence["valid_cross_model_tests"] > 0
    assert dependence["serial_bh_q_below_0_05"] > 0
    claims = pd.read_csv(PUBLISHED / "dependence_same_publication.csv")
    assert claims["alignment"].eq("exact_common_publication_date").all()
    assert claims["sample_count"].eq(472).all()


def test_secret_is_absent_from_published_experiment_artifacts() -> None:
    # The environment-variable *name* may appear in a credential-policy
    # disclosure; query parameters or serialized key fields may not.
    forbidden = (b"api_key=", b"api_key%3D", b'"api_key":')
    paths = [DATA_MANIFEST, INFERENCE_MANIFEST, *PROCESSED.glob("*"), *PUBLISHED.glob("*")]
    for path in paths:
        if not path.is_file():
            continue
        content = path.read_bytes()
        for marker in forbidden:
            assert marker not in content
