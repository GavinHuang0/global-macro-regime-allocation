"""Integration contracts for the generated existing-block attribution stage."""

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/m02_existing_block_attribution.json"
PROCESSED = ROOT / "data/processed/m02_soft_composite/existing_block_attribution"
PUBLISHED = ROOT / "results/published/m02_soft_composite/existing_block_attribution"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_manifest_hashes_and_frozen_control_invariance() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["stage_id"] == "existing_block_attribution"
    assert manifest["baseline_invariance_passed"] is True
    assert manifest["model_selection"]["atomic_comparisons"] == 14
    assert manifest["model_selection"]["replacement_comparisons"] == 3
    for declaration in manifest["generated_file_hashes"]:
        path = ROOT / declaration["path"]
        assert path.is_file()
        assert _digest(path) == declaration["sha256"]
        assert path.stat().st_size == declaration["bytes"]
    for declaration in manifest["implementation_file_hashes"]:
        path = ROOT / declaration["path"]
        assert path.is_file()
        assert _digest(path) == declaration["sha256"]

    invariance = pd.read_csv(PUBLISHED / "baseline_invariance.csv")
    assert len(invariance) == manifest["coverage"]["invariance_checks"]
    assert invariance["passed"].astype(bool).all()
    assert set(invariance["variant_id"]) == {
        "transition_only",
        "partial_only",
        "student_t_7_combined",
    }


def test_atomic_grid_references_and_bootstrap_contract() -> None:
    comparisons = pd.read_csv(PUBLISHED / "paired_comparisons.csv")
    primary = comparisons.loc[
        comparisons["primary_atomic"].astype(bool)
        & comparisons["comparison_scope"].eq("full_common_calendar")
        & comparisons["evaluation_checkpoint"].eq("before_any_defining_release")
    ]
    assert len(primary) == 14
    assert primary["comparison_id"].nunique() == 14
    assert set(primary.loc[primary["experiment_arm"].eq("add_one"), "reference_id"]) == {
        "partial_only"
    }
    assert set(
        primary.loc[primary["experiment_arm"].eq("leave_one_out"), "reference_id"]
    ) == {"student_t_7_combined"}
    assert primary["paired_months"].eq(183).all()

    bootstrap = pd.read_csv(PUBLISHED / "paired_block_bootstrap.csv")
    atomic = bootstrap.loc[bootstrap["primary_atomic"].astype(bool)]
    assert set(atomic["replications"]) == {5000}
    assert set(atomic["block_length_months"]) == {12}
    assert set(atomic["evaluation_sample"]) == {
        "full_sample",
        "excluding_2020_03_through_2020_05",
    }
    assert set(
        atomic.loc[atomic["evaluation_sample"].eq("full_sample"), "calendar_segments"]
    ) == {1}
    assert set(
        atomic.loc[
            atomic["evaluation_sample"].eq(
                "excluding_2020_03_through_2020_05"
            ),
            "calendar_segments",
        ]
    ) == {2}
    assert atomic["holm_adjusted_p_value_primary_atomic"].notna().all()
    assert atomic["comparison_id"].nunique() == 14
    assert atomic["metric"].nunique() == 4
    assert atomic["evaluation_checkpoint"].nunique() == 4


def test_evidence_allowlists_and_causal_fit_cutoffs() -> None:
    registry = pd.read_csv(PUBLISHED / "model_registry.csv")
    audit = pd.read_csv(PROCESSED / "event_update_audit.csv.gz")
    applied = audit.loc[audit["update_status"].eq("applied")]
    all_models = {
        "weekly_labor_stress",
        "monthly_labor_demand",
        "consumer_demand",
        "housing_activity",
        "business_investment",
        "inflation_expectations",
        "inflation_input_costs",
    }
    for row in registry.loc[registry["variant_id"].str.startswith("add_one_")].itertuples():
        block = row.variant_id.removeprefix("add_one_")
        observed = set(
            applied.loc[applied["variant_id"].eq(row.variant_id), "observation_model_id"]
        )
        assert observed == {block}
    for row in registry.loc[
        registry["variant_id"].str.startswith("leave_one_out_")
    ].itertuples():
        omitted = row.variant_id.removeprefix("leave_one_out_")
        observed = set(
            applied.loc[applied["variant_id"].eq(row.variant_id), "observation_model_id"]
        )
        assert omitted not in observed
        assert observed == all_models - {omitted}

    fits = pd.read_csv(PROCESSED / "emission_fit_audit.csv")
    cutoff = pd.to_datetime(fits["strict_fit_cutoff"], errors="raise")
    latest = pd.to_datetime(fits["last_training_available_at"], errors="raise")
    assert (latest < cutoff).all()


def test_replacement_contrasts_and_public_summary_are_complete() -> None:
    replacements = pd.read_csv(PUBLISHED / "replacement_block_bootstrap.csv")
    assert set(replacements["candidate_id"]) == {
        "priority_03_consumer",
        "priority_04_housing",
        "priority_06_backlog",
    }
    assert replacements["holm_adjusted_p_value_replacement_candidates"].notna().all()
    summary = pd.read_csv(PUBLISHED / "block_attribution_summary.csv")
    assert len(summary) == 8
    assert summary["primary_atomic"].astype(bool).sum() == 7
    assert summary["applied_updates"].ge(0).all()


def test_published_artifacts_do_not_contain_credentials() -> None:
    forbidden = ("FRED_API_KEY=", "api_key=", "client_secret", "refresh_token")
    for path in [MANIFEST, *PUBLISHED.glob("*")]:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            assert not any(token.lower() in text for token in forbidden)
