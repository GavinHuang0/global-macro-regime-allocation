"""Integration contracts for the promoted Model 02 current-baseline artifacts.

These tests treat ``current_baseline`` as a publication and governance layer,
not as a rewrite of any historical experiment.  They verify the complete
manifest lineage, the distinction between engine and public model roles, the
exact reduced evidence graph, point-in-time fitting, frozen-control
invariance, bootstrap design, and retention of prior sensitivity namespaces.

The promotion was selected after inspecting the same historical development
sample.  Accordingly, the artifacts must describe it as an operational
baseline choice rather than as fresh out-of-sample validation.
"""

from __future__ import annotations

import gzip
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/m02_current_baseline.yaml"
MANIFEST = ROOT / "data/manifests/m02_current_baseline.json"
PROCESSED = ROOT / "data/processed/m02_soft_composite/current_baseline"
PUBLISHED = ROOT / "results/published/m02_soft_composite/current"

BASELINE_ID = "student_t_7_reduced_core"
EXPECTATIONS_ID = "student_t_7_reduced_core_with_expectations"
PUBLIC_BENCHMARK_IDS = {
    "transition_only",
    "partial_only",
    "student_t_7_combined",
}
ENGINE_BENCHMARK_IDS = {"transition_only", "partial_only"}
CURRENT_VARIANT_IDS = {
    BASELINE_ID,
    EXPECTATIONS_ID,
    *PUBLIC_BENCHMARK_IDS,
}
BASELINE_OBSERVATION_MODELS = {
    "weekly_labor_stress",
    "consumer_real_implicit_joint",
    "business_activity_pipeline_joint",
}
FORBIDDEN_BASELINE_MODELS = {
    "monthly_labor_demand",
    "housing_activity",
    "inflation_input_costs",
    "inflation_import_prices",
    "inflation_expectations",
    "consumer_demand",
    "consumer_vehicle_units",
    "business_investment",
    "weekly_continued_claims_conditional",
}
ARCHIVED_STAGE_IDS = {
    "inference_sensitivities",
    "evidence_block_inference",
    "existing_block_attribution",
    "feature_revision",
}


def _digest(path: Path) -> str:
    """Return the byte-level SHA-256 digest used by the stage manifest."""

    return sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    """Load the immutable manifest emitted by the current-baseline build."""

    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _artifact_bytes(path: Path) -> bytes:
    """Read an artifact's logical bytes, expanding gzip for hygiene checks."""

    if path.suffix == ".gz":
        with gzip.open(path, "rb") as source:
            return source.read()
    return path.read_bytes()


def _sorted_latest(frame: pd.DataFrame) -> pd.DataFrame:
    """Put latest-marginal rows in a stable order for exact view checks."""

    return frame.sort_values(
        ["variant_id", "reference_month"], kind="stable"
    ).reset_index(drop=True)


def test_manifest_hashes_full_lineage_and_generated_stage() -> None:
    """Every declared input and output must still match its recorded bytes."""

    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["model_id"] == "m02_soft_composite"
    assert manifest["stage_id"] == "current_baseline"
    assert manifest["configuration"] == CONFIG.relative_to(ROOT).as_posix()
    assert manifest["configuration_sha256"] == _digest(CONFIG)
    assert manifest["base_configuration_sha256"] == _digest(
        ROOT / "configs/models/m02_event_driven_bayesian_filter.yaml"
    )
    assert manifest["frozen_sensitivity_configuration_sha256"] == _digest(
        ROOT / "configs/models/m02_inference_sensitivities.yaml"
    )

    upstream = manifest["upstream_manifest"]
    assert upstream["path"] == "data/manifests/m02_feature_revision.json"
    upstream_path = ROOT / upstream["path"]
    assert upstream_path.is_file()
    assert upstream["sha256"] == _digest(upstream_path)

    for section in ("source_files", "implementation_files", "generated_files"):
        declarations = manifest[section]
        assert declarations, section
        declared_paths = set()
        for declaration in declarations:
            path = ROOT / declaration["path"]
            assert path.is_file(), path
            assert declaration["path"] not in declared_paths
            declared_paths.add(declaration["path"])
            assert declaration["sha256"] == _digest(path), path
            assert declaration["bytes"] == path.stat().st_size, path

    source_paths = {row["path"] for row in manifest["source_files"]}
    assert {
        "results/published/m02_soft_composite/feature_revision/model_registry.csv",
        "data/processed/m02_soft_composite/feature_revision/evaluation_rows.csv",
        "data/processed/m02_soft_composite/feature_revision/event_update_audit.csv.gz",
        (
            "data/processed/m02_soft_composite/feature_revision/"
            "partial_defining_audit.csv.gz"
        ),
    }.issubset(source_paths)
    assert manifest["coverage"]["variants"] == 5
    assert manifest["coverage"]["comparisons"] == 4


def test_current_registry_has_one_baseline_three_public_benchmarks() -> None:
    """Publication roles may augment, but must not overwrite, engine roles."""

    current = pd.read_csv(PUBLISHED / "current_registry.csv")
    engine = pd.read_csv(PROCESSED / "engine_registry.csv")
    assert len(current) == len(engine) == 5
    assert current["variant_id"].is_unique
    assert set(current["variant_id"]) == CURRENT_VARIANT_IDS
    assert set(engine["variant_id"]) == CURRENT_VARIANT_IDS

    public_roles = current.groupby("publication_role")["variant_id"].agg(set)
    assert public_roles.to_dict() == {
        "baseline": {BASELINE_ID},
        "major_benchmark": PUBLIC_BENCHMARK_IDS,
        "sensitivity": {EXPECTATIONS_ID},
    }
    engine_roles = current.groupby("engine_model_role")["variant_id"].agg(set)
    assert engine_roles.to_dict() == {
        "baseline": {BASELINE_ID},
        "major_benchmark": ENGINE_BENCHMARK_IDS,
        "sensitivity": {"student_t_7_combined", EXPECTATIONS_ID},
    }
    pd.testing.assert_series_equal(
        current.set_index("variant_id")["engine_model_role"].sort_index(),
        engine.set_index("variant_id")["model_role"].sort_index(),
        check_names=False,
    )


def test_reduced_core_and_expectations_sensitivity_are_exactly_realized() -> None:
    """The promoted graph is sparse and expectations is its only add-one arm."""

    audit = pd.read_csv(PROCESSED / "event_update_audit.csv.gz")
    baseline = audit.loc[audit["variant_id"].eq(BASELINE_ID)]
    expectations = audit.loc[audit["variant_id"].eq(EXPECTATIONS_ID)]
    assert not baseline.empty
    assert not expectations.empty

    observed_baseline = set(baseline["observation_model_id"])
    observed_expectations = set(expectations["observation_model_id"])
    assert observed_baseline == BASELINE_OBSERVATION_MODELS
    assert FORBIDDEN_BASELINE_MODELS.isdisjoint(observed_baseline)
    assert observed_expectations == {
        *BASELINE_OBSERVATION_MODELS,
        "inflation_expectations",
    }
    assert observed_expectations - observed_baseline == {"inflation_expectations"}

    expectation_rows = expectations.loc[
        expectations["observation_model_id"].eq("inflation_expectations")
    ]
    assert len(expectation_rows) == 29
    assert expectation_rows["update_status"].eq("applied").sum() == 3
    assert expectation_rows["update_status"].str.startswith("fit_error:").sum() == 26
    assert expectation_rows["update_status"].eq("applied").mean() < 0.15

    current = pd.read_csv(PUBLISHED / "current_registry.csv").set_index("variant_id")
    mechanics = [
        "non_defining_evidence",
        "partial_defining_releases",
        "emission_family",
        "var_method",
        "retail_method",
    ]
    pd.testing.assert_series_equal(
        current.loc[BASELINE_ID, mechanics],
        current.loc[EXPECTATIONS_ID, mechanics],
        check_names=False,
    )


def test_latest_public_files_are_exact_role_filtered_views() -> None:
    """Current headline files must exclude non-current roles and extra models."""

    latest = pd.read_csv(PROCESSED / "latest_marginals.csv")
    baseline = pd.read_csv(PUBLISHED / "baseline_latest_marginals.csv")
    benchmarks = pd.read_csv(PUBLISHED / "benchmark_latest_marginals.csv")

    assert set(baseline["variant_id"]) == {BASELINE_ID}
    assert baseline["publication_role"].eq("baseline").all()
    assert set(benchmarks["variant_id"]) == PUBLIC_BENCHMARK_IDS
    assert benchmarks["publication_role"].eq("major_benchmark").all()
    assert EXPECTATIONS_ID not in set(baseline["variant_id"])
    assert EXPECTATIONS_ID not in set(benchmarks["variant_id"])
    assert len(baseline) == 4
    assert len(benchmarks) == 12

    for frame in (baseline, benchmarks):
        assert frame["as_of_date"].nunique() == 1
        assert (
            frame.groupby("variant_id")["relative_month"].agg(set)
            == {-3, -2, -1, 0}
        ).all()

    shared_columns = [
        column for column in latest.columns if column != "model_role"
    ]
    expected_baseline = latest.loc[latest["variant_id"].eq(BASELINE_ID)]
    expected_benchmarks = latest.loc[
        latest["variant_id"].isin(PUBLIC_BENCHMARK_IDS)
    ]
    pd.testing.assert_frame_equal(
        _sorted_latest(baseline)[shared_columns],
        _sorted_latest(expected_baseline)[shared_columns],
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        _sorted_latest(benchmarks)[shared_columns],
        _sorted_latest(expected_benchmarks)[shared_columns],
        check_dtype=False,
    )


def test_frozen_controls_pass_every_semantic_invariance_check() -> None:
    """Promotion must leave all three historical control outputs unchanged."""

    manifest = _manifest()
    invariance = pd.read_csv(PROCESSED / "baseline_invariance.csv")
    assert manifest["frozen_control_invariance_passed"] is True
    assert len(invariance) == 216
    assert len(invariance) == manifest["coverage"][
        "frozen_control_invariance_checks"
    ]
    assert invariance["passed"].eq(True).all()  # noqa: E712
    assert set(invariance["variant_id"]) == PUBLIC_BENCHMARK_IDS

    expected_common = {
        "evaluation_rows",
        "exact_score_audit",
        "latest_marginals",
        "partial_defining_audit",
    }
    for variant_id in PUBLIC_BENCHMARK_IDS:
        observed = set(
            invariance.loc[invariance["variant_id"].eq(variant_id), "artifact"]
        )
        assert expected_common.issubset(observed)
    predecessor = set(
        invariance.loc[
            invariance["variant_id"].eq("student_t_7_combined"), "artifact"
        ]
    )
    assert "event_update_audit" in predecessor


def test_all_estimation_cutoffs_are_strictly_point_in_time() -> None:
    """No release, transition pair, or component may train on same-day news."""

    emissions = pd.read_csv(
        PROCESSED / "emission_fit_audit.csv",
        parse_dates=["strict_fit_cutoff", "last_training_available_at"],
    )
    assert not emissions.empty
    assert emissions["last_training_available_at"].notna().all()
    assert (
        emissions["last_training_available_at"] < emissions["strict_fit_cutoff"]
    ).all()

    transitions = pd.read_csv(
        PROCESSED / "transition_fit_audit.csv",
        parse_dates=["fit_date", "last_pair_available_at"],
    )
    assert not transitions.empty
    assert transitions["last_pair_available_at"].notna().all()
    assert (transitions["last_pair_available_at"] < transitions["fit_date"]).all()
    assert transitions["converged"].eq(True).all()  # noqa: E712

    partial = pd.read_csv(
        PROCESSED / "partial_defining_audit.csv.gz",
        parse_dates=[
            "release_date",
            "component_fit_knowledge_cutoff",
            "latest_component_training_availability",
        ],
    )
    trained = partial["latest_component_training_availability"].notna()
    assert trained.any()
    assert partial.loc[trained, "component_fit_knowledge_cutoff"].eq(
        partial.loc[trained, "release_date"]
    ).all()
    assert (
        partial.loc[trained, "latest_component_training_availability"]
        < partial.loc[trained, "component_fit_knowledge_cutoff"]
    ).all()


def test_paired_bootstrap_has_the_complete_declared_design() -> None:
    """All four comparisons use the same locked samples and resampling policy."""

    bootstrap = pd.read_csv(PUBLISHED / "paired_block_bootstrap.csv")
    expected_comparisons = {
        f"{BASELINE_ID}__vs__transition_only",
        f"{BASELINE_ID}__vs__partial_only",
        f"{BASELINE_ID}__vs__student_t_7_combined",
        f"{EXPECTATIONS_ID}__vs__{BASELINE_ID}",
    }
    assert set(bootstrap["comparison_id"]) == expected_comparisons
    assert set(bootstrap["evaluation_sample"]) == {
        "full_sample",
        "excluding_2020_03_through_2020_05",
    }
    assert set(bootstrap["evaluation_checkpoint"]) == {
        "before_any_defining_release",
        "after_employment_situation",
        "after_midmonth_defining_releases",
        "strict_pre_final_score_day",
    }
    assert set(bootstrap["metric"]) == {
        "quadrant_cross_entropy_to_exact_score_map",
        "quadrant_brier_distance_to_exact_score_map",
        "score_center_negative_log_predictive_density",
        "hard_quadrant_correct",
    }
    assert len(bootstrap) == 4 * 2 * 4 * 4 == 128
    assert bootstrap["replications"].eq(5000).all()
    assert bootstrap["block_length_months"].eq(12).all()
    assert bootstrap["random_seed"].eq(20260722).all()
    assert bootstrap.loc[
        bootstrap["evaluation_sample"].eq("full_sample"), "calendar_segments"
    ].eq(1).all()
    assert bootstrap.loc[
        bootstrap["evaluation_sample"].eq(
            "excluding_2020_03_through_2020_05"
        ),
        "calendar_segments",
    ].eq(2).all()
    assert bootstrap.loc[
        bootstrap["evaluation_sample"].eq("full_sample"), "paired_months"
    ].eq(183).all()
    assert bootstrap.loc[
        bootstrap["evaluation_sample"].eq(
            "excluding_2020_03_through_2020_05"
        ),
        "paired_months",
    ].eq(180).all()

    atomic = bootstrap.groupby("comparison_id")["primary_atomic"].first()
    assert atomic.drop(f"{EXPECTATIONS_ID}__vs__{BASELINE_ID}").eq(False).all()
    assert bool(atomic.loc[f"{EXPECTATIONS_ID}__vs__{BASELINE_ID}"]) is True


def test_sensitivity_catalog_retains_variants_and_archived_namespaces() -> None:
    """Current status remains concise while every older search stays linked."""

    catalog = pd.read_csv(PUBLISHED / "sensitivity_catalog.csv")
    assert catalog["catalog_id"].is_unique
    model_variants = catalog.loc[catalog["entry_type"].eq("model_variant")]
    archived = catalog.loc[catalog["entry_type"].eq("archived_stage")]
    assert len(model_variants) == 24
    assert len(archived) == 4
    assert set(archived["source_stage"]) == ARCHIVED_STAGE_IDS
    assert set(model_variants["publication_role"]) == {"sensitivity"}
    assert set(model_variants.loc[
        model_variants["source_stage"].eq("current_baseline"), "variant_id"
    ]) == {EXPECTATIONS_ID}
    assert (
        model_variants["source_stage"].eq("feature_revision").sum() == 23
    )

    for row in archived.itertuples(index=False):
        assert (ROOT / row.config_path).is_file()
        assert (ROOT / row.manifest_path).is_file()
        assert (ROOT / row.results_path).is_dir()


def test_method_summary_discloses_selection_scope_and_no_credentials() -> None:
    """Public metadata must state same-history selection and serialize no keys."""

    manifest = _manifest()
    method = json.loads(
        (PUBLISHED / "method_summary.json").read_text(encoding="utf-8")
    )
    assert method["selected_baseline"] == BASELINE_ID
    assert method["predecessor_baseline"] == "student_t_7_combined"
    assert set(method["publication_benchmarks"]) == PUBLIC_BENCHMARK_IDS
    assert set(method["engine_major_benchmarks"]) == ENGINE_BENCHMARK_IDS
    assert method["current_sensitivities"] == [EXPECTATIONS_ID]
    assert set(method["baseline_observation_models"]) == BASELINE_OBSERVATION_MODELS
    assert method["historical_sensitivity_count"] == 23
    assert method["archived_sensitivity_stage_count"] == 4
    assert method["frozen_control_invariance_passed"] is True
    scope = method["validation_scope"].lower()
    assert "same-history" in scope
    assert "not a fresh out-of-sample validation" in scope
    assert method["credential_policy"] == "offline; no environment credential is read"
    assert manifest["credential_policy"] == method["credential_policy"]

    forbidden = (
        b"fred_api_key",
        b"api_key=",
        b"api_key%3d",
        b'"api_key":',
        b"client_secret",
        b"refresh_token",
        b"authorization: bearer",
    )
    artifact_paths = [
        CONFIG,
        MANIFEST,
        *[ROOT / row["path"] for row in manifest["generated_files"]],
    ]
    for path in artifact_paths:
        content = _artifact_bytes(path).lower()
        assert not any(marker in content for marker in forbidden), path
