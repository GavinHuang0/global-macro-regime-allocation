"""Lineage and causality checks for published Model 02 sensitivity artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed/m02_soft_composite/inference_sensitivities"
PUBLISHED = ROOT / "results/published/m02_soft_composite/inference_sensitivities"
MANIFEST = ROOT / "data/manifests/m02_inference_sensitivities.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_hashes_and_required_variants() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["stage_id"] == "partial_defining_and_robustness_sensitivities"
    for artifact in manifest["generated_file_hashes"]:
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert _digest(path) == artifact["sha256"]
        assert path.stat().st_size == artifact["bytes"]
    for source in manifest["implementation_file_hashes"]:
        path = ROOT / source["path"]
        assert path.is_file()
        assert _digest(path) == source["sha256"]
        assert path.stat().st_size == source["bytes"]
    latest = pd.read_csv(PUBLISHED / "latest_marginals.csv")
    required = {
        "transition_only",
        "partial_only",
        "student_t_7_combined",
        "selected_tail_combined",
        "huber_var_combined",
        "student_t_var_combined",
        "retail_shrinkage_combined",
        "retail_real_decomposition_combined",
    }
    assert required.issubset(set(latest["variant_id"]))
    assert "retail_stress_interaction_combined" not in set(latest["variant_id"])


def test_published_model_roles_match_the_selected_contract() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    method = json.loads((PUBLISHED / "method_summary.json").read_text(encoding="utf-8"))
    registry = pd.read_csv(PUBLISHED / "model_registry.csv").set_index("variant_id")

    for selection in (manifest["model_selection"], method["model_selection"]):
        assert selection["baseline"] == "student_t_7_combined"
        assert selection["major_benchmarks"] == ["transition_only", "partial_only"]
        assert selection["variant_roles"]["student_t_7_combined"] == "baseline"
        assert selection["variant_roles"]["transition_only"] == "major_benchmark"
        assert selection["variant_roles"]["partial_only"] == "major_benchmark"

    assert registry.loc["student_t_7_combined", "model_role"] == "baseline"
    assert set(
        registry.index[registry["model_role"].eq("major_benchmark")]
    ) == {"transition_only", "partial_only"}
    selected = {"student_t_7_combined", "transition_only", "partial_only"}
    assert registry.loc[~registry.index.isin(selected), "model_role"].eq(
        "sensitivity"
    ).all()

    for filename, identifier in (
        ("evaluation_summary.csv", "filter_variant"),
        ("evaluation_subperiod_summary.csv", "filter_variant"),
        ("latest_marginals.csv", "variant_id"),
    ):
        frame = pd.read_csv(PUBLISHED / filename)
        actual = frame[[identifier, "model_role"]].drop_duplicates().set_index(identifier)
        expected = registry.loc[actual.index, ["model_role"]]
        expected.index.name = identifier
        pd.testing.assert_frame_equal(actual.sort_index(), expected.sort_index())

    paired = pd.read_csv(PUBLISHED / "paired_comparisons.csv")
    assert "comparison_baseline" not in paired
    assert {
        "comparison_reference",
        "model_role",
        "reference_model_role",
    }.issubset(paired.columns)
    baseline_pairs = paired.loc[
        paired["comparison_scope"].eq("baseline_vs_major_benchmark")
    ]
    assert baseline_pairs["filter_variant"].eq("student_t_7_combined").all()
    assert set(baseline_pairs["comparison_reference"]) == {
        "transition_only",
        "partial_only",
    }
    sensitivity_pairs = paired.loc[
        paired["comparison_scope"].eq("sensitivity_vs_selected_baseline")
    ]
    assert sensitivity_pairs["model_role"].eq("sensitivity").all()
    assert sensitivity_pairs["comparison_reference"].eq(
        "student_t_7_combined"
    ).all()


def test_transition_only_reproduces_frozen_predecessor_replay() -> None:
    sensitivity = pd.read_csv(
        PROCESSED / "evaluation_rows.csv", parse_dates=["reference_month"]
    )
    baseline = pd.read_csv(
        ROOT
        / "data/processed/m02_soft_composite/bayesian_filter/evaluation_rows.csv",
        parse_dates=["reference_month"],
    )
    sensitivity = sensitivity.loc[
        sensitivity["filter_variant"].eq("transition_only")
        & sensitivity["evaluation_checkpoint"].eq("strict_pre_final_score_day")
    ]
    baseline = baseline.loc[
        baseline["filter_variant"].eq("transition_only")
        & baseline["evaluation_checkpoint"].eq("strict_pre_day")
    ]
    columns = [
        "growth_error",
        "inflation_error",
        "score_center_negative_log_predictive_density",
    ]
    paired = sensitivity[["reference_month", *columns]].merge(
        baseline[["reference_month", *columns]],
        on="reference_month",
        validate="one_to_one",
        suffixes=("_sensitivity", "_baseline"),
    )
    assert len(paired) == len(sensitivity) == len(baseline) == 188
    for column in columns:
        np.testing.assert_allclose(
            paired[f"{column}_sensitivity"],
            paired[f"{column}_baseline"],
            atol=1.0e-10,
            rtol=1.0e-12,
        )


def test_incomplete_latest_month_receives_live_partial_defining_updates() -> None:
    score_manifest = json.loads(
        (ROOT / "data/manifests/m02_soft_composite.json").read_text(encoding="utf-8")
    )
    assert score_manifest["latest_complete_score_month"] == "2026-05-01"
    assert score_manifest["latest_component_reference_month"] == "2026-06-01"

    scores = pd.read_csv(
        ROOT / "data/processed/m02_soft_composite/composite_scores.csv"
    )
    june = scores.loc[scores["reference_month"].eq("2026-06-01")].iloc[0]
    assert pd.isna(june["growth_score"])
    assert pd.isna(june["inflation_score"])
    available = {
        column.removesuffix("_z")
        for column in scores.columns
        if column.endswith("_z") and pd.notna(june[column])
    }
    assert available == {
        "average_hourly_earnings",
        "core_cpi",
        "industrial_production",
        "payrolls",
        "producer_prices",
        "unemployment_rate",
    }

    partial = pd.read_csv(PROCESSED / "partial_defining_audit.csv.gz")
    june_updates = partial.loc[
        partial["reference_month"].eq("2026-06-01")
        & partial["variant_id"].eq("partial_only")
        & partial["status"].eq("applied")
    ]
    observed = {
        component
        for value in june_updates["components"]
        for component in str(value).split("|")
    }
    assert observed == available

    latest = pd.read_csv(PUBLISHED / "latest_marginals.csv")
    current = latest.loc[latest["reference_month"].eq("2026-07-01")].set_index(
        "variant_id"
    )
    assert not np.allclose(
        current.loc["partial_only", ["growth_mean", "inflation_mean"]],
        current.loc["transition_only", ["growth_mean", "inflation_mean"]],
    )


def test_all_recorded_training_cutoffs_are_strictly_causal() -> None:
    partial = pd.read_csv(
        PROCESSED / "partial_defining_audit.csv.gz",
        parse_dates=["release_date", "latest_component_training_availability"],
    )
    fitted_partial = partial["latest_component_training_availability"].notna()
    assert (
        partial.loc[fitted_partial, "latest_component_training_availability"]
        < partial.loc[fitted_partial, "release_date"]
    ).all()

    emissions = pd.read_csv(
        PROCESSED / "emission_fit_audit.csv",
        parse_dates=["strict_fit_cutoff", "last_training_available_at"],
    )
    assert (
        emissions["last_training_available_at"] < emissions["strict_fit_cutoff"]
    ).all()

    transitions = pd.read_csv(
        PROCESSED / "transition_fit_audit.csv",
        parse_dates=["fit_date", "last_pair_available_at"],
    )
    assert (transitions["last_pair_available_at"] < transitions["fit_date"]).all()
    assert transitions["converged"].all()


def test_robust_weights_and_final_score_policy_are_audited() -> None:
    events = pd.read_csv(PROCESSED / "event_update_audit.csv.gz")
    weights = pd.to_numeric(events["event_weight"], errors="coerce").dropna()
    assert len(weights) > 1_000
    assert np.isfinite(weights).all()
    assert (weights > 0.0).all()
    assert (weights < 0.1).any()
    assert events["weight_state_cutoff"].dropna().eq("shared_pre_release_day").all()
    applied = events.loc[events["update_status"].eq("applied")]
    assert pd.to_numeric(applied["training_count"], errors="coerce").notna().all()

    partial = pd.read_csv(PROCESSED / "partial_defining_audit.csv.gz")
    final_policy = "skipped_block_completing_both_scores_exact_end_of_day"
    assert partial["status"].eq(final_policy).any()
    exact = pd.read_csv(PROCESSED / "exact_score_audit.csv")
    assert exact["status"].isin(
        {"conditioned_exactly", "identical_exact_no_op", "target_outside_path"}
    ).all()


def test_tail_and_stress_sensitivities_are_honestly_scoped() -> None:
    tails = pd.read_csv(PUBLISHED / "tail_summary.csv")
    finite = tails.loc[tails["sensitivity_status"].eq("evaluated")]
    assert {4.0, 5.0, 7.0, 10.0}.issubset(set(finite["degrees_of_freedom"]))
    assert np.isinf(finite["degrees_of_freedom"]).any()
    sparse = tails.loc[
        tails["profile_id"].eq("base:inflation_expectations")
    ]
    assert sparse["sensitivity_status"].eq(
        "insufficient_causal_annual_origin_history"
    ).all()

    stress = pd.read_csv(PROCESSED / "retail_stress_identification_audit.csv")
    assert len(stress) == 242
    assert int(stress["positive_stress"].sum()) == 1
    method = json.loads((PUBLISHED / "method_summary.json").read_text(encoding="utf-8"))
    assert method["stress_interaction"]["enabled"] is False
    assert method["stress_interaction"]["positive_stress_events"] == 1


def test_no_generated_artifact_contains_secret_fields() -> None:
    forbidden = (b"FRED_API_KEY", b"api_key=", b"api_key%3D")
    paths = [MANIFEST, *PROCESSED.glob("*"), *PUBLISHED.glob("*")]
    for path in paths:
        if not path.is_file():
            continue
        content = path.read_bytes()
        for marker in forbidden:
            assert marker not in content
