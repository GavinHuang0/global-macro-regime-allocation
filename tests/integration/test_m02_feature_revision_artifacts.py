"""Integration contracts for the Model 02 feature-revision artifacts.

These checks treat the generated manifest as the stage boundary.  They verify
that the replay is reproducible from its declared lineage, that the three
frozen controls are semantically unchanged, and that the revised evidence
profiles implement the predeclared economics.  In particular, the claims
test reconstructs the exact same-publication ICSA--CCSA pairing while
preserving CCSA's own target month; it does not accept a summary count as a
substitute for correct event dating.
"""

from __future__ import annotations

import gzip
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/models/m02_feature_revision.yaml"
MANIFEST = ROOT / "data/manifests/m02_feature_revision.json"
PROCESSED = ROOT / "data/processed/m02_soft_composite/feature_revision"
PUBLISHED = ROOT / "results/published/m02_soft_composite/feature_revision"

CONTROL_IDS = {
    "transition_only",
    "partial_only",
    "student_t_7_combined",
}
FORBIDDEN_REVISED_MODELS = {"monthly_labor_demand", "housing_activity"}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _artifact_text(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as source:
            return source.read()
    return path.read_bytes()


def test_manifest_hashes_lineage_and_predeclared_scope() -> None:
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["model_id"] == "m02_soft_composite"
    assert manifest["stage_id"] == "feature_revision"
    assert manifest["configuration"] == CONFIG.relative_to(ROOT).as_posix()
    assert manifest["configuration_sha256"] == _digest(CONFIG)
    assert manifest["base_configuration_sha256"] == _digest(
        ROOT / "configs/models/m02_event_driven_bayesian_filter.yaml"
    )
    assert manifest["frozen_sensitivity_configuration_sha256"] == _digest(
        ROOT / "configs/models/m02_inference_sensitivities.yaml"
    )
    assert manifest["coverage"]["variants"] == 26
    assert manifest["coverage"]["comparisons"] == 23
    assert manifest["model_selection"] == {
        "selected_baseline": "student_t_7_combined",
        "selected_baseline_changed": False,
        "predeclared_combined_candidate": "all_revised_candidate",
        "status": "development_experiment_no_automatic_promotion",
    }

    for declaration in manifest["upstream_manifests"]:
        path = ROOT / declaration["path"]
        assert path.is_file()
        assert _digest(path) == declaration["sha256"]
    for declaration in manifest["source_files"]:
        path = ROOT / declaration["path"]
        assert path.is_file()
        assert _digest(path) == declaration["sha256"]
        assert path.stat().st_size == declaration["bytes"]
    for declaration in manifest["implementation_files"]:
        path = ROOT / declaration["path"]
        assert path.is_file()
        assert _digest(path) == declaration["sha256"]
        assert path.stat().st_size == declaration["bytes"]
    for declaration in manifest["generated_files"]:
        path = ROOT / declaration["path"]
        assert path.is_file()
        assert _digest(path) == declaration["sha256"]
        assert path.stat().st_size == declaration["bytes"]


def test_frozen_controls_are_semantically_invariant() -> None:
    manifest = _manifest()
    assert manifest["baseline_invariance_passed"] is True
    invariance = pd.read_csv(PUBLISHED / "baseline_invariance.csv")
    assert len(invariance) == manifest["coverage"]["invariance_checks"]
    assert invariance["passed"].eq(True).all()  # noqa: E712 - explicit bool contract

    expected_artifacts = {
        "evaluation_rows",
        "partial_defining_audit",
        "exact_score_audit",
        "latest_marginals",
    }
    for variant_id in CONTROL_IDS:
        observed = set(
            invariance.loc[invariance["variant_id"].eq(variant_id), "artifact"]
        )
        assert expected_artifacts.issubset(observed)
    baseline_artifacts = set(
        invariance.loc[
            invariance["variant_id"].eq("student_t_7_combined"), "artifact"
        ]
    )
    assert "event_update_audit" in baseline_artifacts

    registry = pd.read_csv(PUBLISHED / "model_registry.csv").set_index("variant_id")
    assert len(registry) == 26
    assert registry.index.is_unique
    assert registry.loc["student_t_7_combined", "model_role"] == "baseline"
    assert set(registry.index[registry["model_role"].eq("major_benchmark")]) == {
        "transition_only",
        "partial_only",
    }
    revised = registry.loc[~registry.index.isin(CONTROL_IDS)]
    assert revised["model_role"].eq("sensitivity").all()
    assert revised["emission_family"].eq("student_t_7").all()
    assert revised["var_method"].eq("ols").all()
    assert revised["partial_defining_releases"].eq(True).all()  # noqa: E712


def test_revised_profiles_exclude_jolts_and_aggregate_housing() -> None:
    audit = pd.read_csv(PROCESSED / "event_update_audit.csv.gz")
    revised = audit.loc[~audit["variant_id"].isin(CONTROL_IDS)]
    assert not revised.empty
    assert FORBIDDEN_REVISED_MODELS.isdisjoint(
        set(revised["observation_model_id"])
    )

    coverage = pd.read_csv(PUBLISHED / "profile_coverage.csv")
    revised_coverage = coverage.loc[~coverage["variant_id"].isin(CONTROL_IDS)]
    assert FORBIDDEN_REVISED_MODELS.isdisjoint(
        set(revised_coverage["observation_model_id"])
    )


def test_price_replacement_profiles_are_exactly_realized() -> None:
    audit = pd.read_csv(PROCESSED / "event_update_audit.csv.gz")
    observed = {
        variant_id: set(group["observation_model_id"])
        for variant_id, group in audit.groupby("variant_id", sort=False)
    }
    input_costs = "inflation_input_costs"
    import_prices = "inflation_import_prices"
    assert input_costs in observed["structural_reduced_input"]
    assert import_prices not in observed["structural_reduced_input"]
    assert {input_costs, import_prices}.isdisjoint(
        observed["structural_reduced_no_price"]
    )
    assert import_prices in observed["price_import"]
    assert input_costs not in observed["price_import"]

    revised_with_input_costs = {
        variant_id
        for variant_id, models in observed.items()
        if variant_id not in CONTROL_IDS and input_costs in models
    }
    assert revised_with_input_costs == {"structural_reduced_input"}

    coverage = pd.read_csv(PUBLISHED / "profile_coverage.csv")
    import_coverage = coverage.loc[
        coverage["variant_id"].eq("price_import")
        & coverage["observation_model_id"].eq(import_prices)
    ]
    assert len(import_coverage) == 1
    assert import_coverage.iloc[0]["training_candidates"] > 0
    assert import_coverage.iloc[0]["applied_updates"] > 0


def test_consumer_and_business_redesigns_have_causal_live_updates() -> None:
    coverage = pd.read_csv(PUBLISHED / "profile_coverage.csv")
    expected = {
        ("consumer_real_only", "consumer_real_only"),
        ("consumer_implicit_only", "consumer_implicit_only"),
        ("consumer_vehicle_only", "consumer_vehicle_units"),
        (
            "consumer_real_implicit_joint_only",
            "consumer_real_implicit_joint",
        ),
        ("consumer_revised_full", "consumer_real_implicit_joint"),
        ("consumer_revised_full", "consumer_vehicle_units"),
        ("business_activity_only", "business_activity_only"),
        ("business_pipeline_only", "business_pipeline_only"),
        ("business_revised_joint", "business_activity_pipeline_joint"),
        ("all_revised_candidate", "consumer_real_implicit_joint"),
        ("all_revised_candidate", "consumer_vehicle_units"),
        ("all_revised_candidate", "business_activity_pipeline_joint"),
    }
    indexed = coverage.set_index(["variant_id", "observation_model_id"])
    assert expected.issubset(set(indexed.index))
    selected = indexed.loc[list(sorted(expected))]
    assert selected["selected_long_rows"].gt(0).all()
    assert selected["training_candidates"].gt(0).all()
    assert selected["applied_updates"].gt(0).all()

    events = pd.read_csv(PROCESSED / "consolidated_candidate_event_universe.csv")
    business = events.loc[
        events["release_block"].eq("business_investment_activity_pipeline")
    ]
    assert set(business["feature_name"]) == {
        "core_capital_goods_shipments_log_change",
        "core_capital_goods_orders_shipments_gap_log_change",
    }
    available = business.loc[business["feature_status"].eq("available")]
    assert not available.empty
    assert set(available["feature_name"]) == set(business["feature_name"])
    business_audit = pd.read_csv(PUBLISHED / "business_feature_audit.csv").iloc[0]
    assert business_audit["matched_event_groups"] > 0
    assert business_audit["available_activity_events"] > 0
    assert business_audit["available_pipeline_events"] > 0
    assert business_audit["available_joint_events"] > 0


def test_conditional_claims_pairing_preserves_dates_and_target_months() -> None:
    expected_counts = {
        "continued_rows": 675,
        "paired_rows": 672,
        "unmatched_rows": 3,
        "cross_month_pairs": 154,
        "available_continued_rows": 597,
        "available_paired_rows": 595,
        "available_cross_month_pairs": 137,
        "publication_dates_with_multiple_pairs": 2,
    }
    claims_audit = pd.read_csv(PUBLISHED / "claims_pairing_audit.csv")
    assert len(claims_audit) == 1
    for column, expected in expected_counts.items():
        assert int(claims_audit.iloc[0][column]) == expected
    assert _manifest()["claims_pairing"] == expected_counts

    events = pd.read_csv(
        PROCESSED / "consolidated_candidate_event_universe.csv",
        parse_dates=[
            "release_date",
            "reference_date",
            "reference_month",
            "control_source_reference_date",
            "control_source_reference_month",
        ],
    )
    claims = events.loc[
        events["release_block"].eq("weekly_continued_claims_conditional")
    ].copy()
    # The consolidated table contains heterogeneous rows, so pandas may keep
    # a date column as strings when another block contains a malformed/missing
    # value.  The claims slice itself must be strictly parseable.
    for column in (
        "release_date",
        "reference_date",
        "reference_month",
        "control_source_reference_date",
        "control_source_reference_month",
    ):
        claims[column] = pd.to_datetime(claims[column], errors="coerce")
    response = claims.loc[
        claims["feature_name"].eq("continued_claims_innovation")
    ].copy()
    control = claims.loc[
        claims["feature_name"].eq(
            "same_publication_initial_claims_innovation_control"
        )
    ].copy()
    assert response["event_id"].nunique() == len(response) == 675
    assert control["event_id"].nunique() == len(control) == 672
    assert set(control["event_id"]).issubset(set(response["event_id"]))
    assert response["release_date"].notna().all()
    assert response["series_id"].eq("CCSA").all()
    assert control["series_id"].eq("ICSA").all()

    paired = response.loc[
        response["conditional_pair_status"].eq(
            "paired_exact_same_publication_plus_7d"
        )
    ].copy()
    unmatched = response.loc[
        response["conditional_pair_status"].eq("unmatched_exact_week_offset")
    ]
    assert len(paired) == 672
    assert len(unmatched) == 3
    assert unmatched["control_source_reference_date"].isna().all()
    assert (
        paired["control_source_reference_date"] - paired["reference_date"]
        == pd.Timedelta(days=7)
    ).all()
    assert paired["reference_month"].eq(
        paired["reference_date"].dt.to_period("M").dt.to_timestamp()
    ).all()
    assert paired["control_source_reference_month"].eq(
        paired["control_source_reference_date"].dt.to_period("M").dt.to_timestamp()
    ).all()
    paired_rows = paired[
        ["event_id", "release_date", "reference_date", "reference_month"]
    ].merge(
        control[
            ["event_id", "release_date", "reference_date", "reference_month"]
        ],
        on="event_id",
        how="left",
        validate="one_to_one",
        suffixes=("_response", "_control"),
    )
    assert paired_rows["release_date_response"].eq(
        paired_rows["release_date_control"]
    ).all()
    assert paired_rows["reference_date_response"].eq(
        paired_rows["reference_date_control"]
    ).all()
    assert paired_rows["reference_month_response"].eq(
        paired_rows["reference_month_control"]
    ).all()
    cross_month = paired["control_source_reference_month"].ne(
        paired["reference_month"]
    )
    assert int(cross_month.sum()) == 154
    available_events = set(
        claims.loc[claims["feature_status"].eq("available"), "event_id"]
        .value_counts()
        .loc[lambda count: count.eq(2)]
        .index
    )
    assert len(available_events) == 595
    assert int(paired.loc[cross_month, "event_id"].isin(available_events).sum()) == 137
    assert int(paired.groupby("release_date").size().gt(1).sum()) == 2

    event_audit = pd.read_csv(PROCESSED / "event_update_audit.csv.gz")
    applied = event_audit.loc[
        event_audit["observation_model_id"].eq(
            "weekly_continued_claims_conditional"
        )
        & event_audit["update_status"].eq("applied")
    ]
    assert not applied.empty
    assert applied["observed_responses"].eq("continued_claims_innovation").all()


def test_all_training_cutoffs_are_strictly_point_in_time() -> None:
    emissions = pd.read_csv(
        PROCESSED / "emission_fit_audit.csv",
        parse_dates=["strict_fit_cutoff", "last_training_available_at"],
    )
    assert not emissions.empty
    assert (
        emissions["last_training_available_at"] < emissions["strict_fit_cutoff"]
    ).all()

    transitions = pd.read_csv(
        PROCESSED / "transition_fit_audit.csv",
        parse_dates=["fit_date", "last_pair_available_at"],
    )
    assert not transitions.empty
    assert (transitions["last_pair_available_at"] < transitions["fit_date"]).all()
    assert transitions["converged"].eq(True).all()  # noqa: E712

    partial = pd.read_csv(
        PROCESSED / "partial_defining_audit.csv.gz",
        parse_dates=["release_date", "latest_component_training_availability"],
    )
    trained = partial["latest_component_training_availability"].notna()
    assert trained.any()
    assert (
        partial.loc[trained, "latest_component_training_availability"]
        < partial.loc[trained, "release_date"]
    ).all()


def test_paired_bootstrap_and_holm_contract_is_complete() -> None:
    bootstrap = pd.read_csv(PUBLISHED / "paired_block_bootstrap.csv")
    assert bootstrap["comparison_id"].nunique() == 23
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
    assert len(bootstrap) == 23 * 2 * 4 * 4
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

    attribution = pd.read_csv(PUBLISHED / "feature_attribution_summary.csv")
    family_counts = (
        attribution[["comparison_id", "family"]]
        .drop_duplicates()
        .groupby("family")
        .size()
        .to_dict()
    )
    assert family_counts == {
        "business": 5,
        "claims": 6,
        "combined": 2,
        "consumer": 7,
        "price": 3,
    }
    comparison_family = (
        attribution[["comparison_id", "family"]]
        .drop_duplicates()
        .set_index("comparison_id")["family"]
    )
    for family in ("price", "consumer", "business", "claims"):
        members = set(comparison_family.index[comparison_family.eq(family)])
        selected = bootstrap.loc[bootstrap["comparison_id"].isin(members)]
        assert selected[f"holm_adjusted_p_value_{family}"].notna().all()
    combined = set(comparison_family.index[comparison_family.eq("combined")])
    combined_rows = bootstrap.loc[bootstrap["comparison_id"].isin(combined)]
    holm_columns = [
        column
        for column in bootstrap.columns
        if column.startswith("holm_adjusted_p_value_")
    ]
    assert combined_rows[holm_columns].isna().all().all()


def test_generated_artifacts_do_not_serialize_credentials() -> None:
    manifest = _manifest()
    forbidden = (
        b"fred_api_key",
        b"api_key=",
        b"api_key%3d",
        b'"api_key":',
        b"client_secret",
        b"refresh_token",
    )
    paths = [
        MANIFEST,
        *[ROOT / row["path"] for row in manifest["generated_files"]],
    ]
    for path in paths:
        content = _artifact_text(path).lower()
        assert not any(marker in content for marker in forbidden), path
