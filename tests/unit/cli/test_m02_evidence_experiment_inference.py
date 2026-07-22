"""Contract tests for the six-priority Model 02 inference experiment."""

from pathlib import Path

import pandas as pd
import pytest

from regime_allocation.cli.build_m02_evidence_experiment_inference import (
    _artifact_invariance,
    _load_experiment_config,
    _merge_observation_models,
    _paired_block_bootstrap,
    _same_publication_dependence,
)
from regime_allocation.cli.build_m02_evidence_experiments import (
    _load_config as _load_data_config,
)


ROOT = Path(__file__).resolve().parents[3]


def test_production_experiment_contract_is_valid() -> None:
    config, raw = _load_experiment_config(
        ROOT / "configs/models/m02_evidence_block_inference.yaml"
    )

    assert raw
    assert config["model_selection"]["baseline"] == "student_t_7_combined"
    assert config["variants"][-1]["id"] == "candidate_all_six"


def test_production_data_contract_uses_verified_m01_loader_keys() -> None:
    config, raw = _load_data_config(
        ROOT / "configs/models/m02_evidence_block_experiments.yaml"
    )

    assert raw
    assert config["sources"]["m01_events"].endswith(
        "non_defining_release_events.csv"
    )


def test_candidate_models_cannot_overwrite_base_model_ids() -> None:
    with pytest.raises(ValueError, match="overwrite"):
        _merge_observation_models(
            {"observation_models": {"legacy": {"responses": ["x"]}}},
            {"legacy": {"responses": ["y"]}},
        )


def test_invariance_ignores_only_declared_surrogate_column() -> None:
    frozen = pd.DataFrame(
        {
            "variant_id": ["student_t_7_combined"],
            "model_role": ["baseline"],
            "release_date": ["2020-01-02"],
            "reference_month": ["2020-01-01"],
            "event_instance_id": ["event"],
            "observation_model_id": ["block"],
            "fit_id": ["fit:1"],
            "event_weight": [0.75],
        }
    )
    current = frozen.copy()
    current.loc[0, "fit_id"] = "fit:99"

    rows = _artifact_invariance(
        current,
        frozen,
        artifact="event",
        variant_column="variant_id",
        key_columns=(
            "release_date",
            "observation_model_id",
            "reference_month",
            "event_instance_id",
        ),
        ignored_columns=frozenset({"fit_id"}),
    )
    assert all(bool(row["passed"]) for row in rows)

    current.loc[0, "event_weight"] = 0.70
    rows = _artifact_invariance(
        current,
        frozen,
        artifact="event",
        variant_column="variant_id",
        key_columns=(
            "release_date",
            "observation_model_id",
            "reference_month",
            "event_instance_id",
        ),
        ignored_columns=frozenset({"fit_id"}),
    )
    failed = [row for row in rows if not bool(row["passed"])]
    assert [row["column"] for row in failed] == ["event_weight"]


def test_invariance_treats_equal_datetime_resolutions_as_equal() -> None:
    frozen = pd.DataFrame(
        {
            "variant_id": ["student_t_7_combined"],
            "as_of_date": pd.Series(["2026-07-20"], dtype="datetime64[ns]"),
            "reference_month": pd.Series(["2026-07-01"], dtype="datetime64[ns]"),
            "relative_month": [0],
        }
    )
    current = frozen.copy()
    current["as_of_date"] = current["as_of_date"].astype("datetime64[s]")

    rows = _artifact_invariance(
        current,
        frozen,
        artifact="latest",
        variant_column="variant_id",
        key_columns=("reference_month", "relative_month", "as_of_date"),
    )

    assert all(bool(row["passed"]) for row in rows)


def test_paired_block_bootstrap_preserves_delta_direction() -> None:
    months = pd.date_range("2020-01-01", periods=36, freq="MS")
    rows = []
    for variant, loss, accuracy in (
        ("student_t_7_combined", 1.0, 0.0),
        ("priority_02_claims", 0.8, 1.0),
    ):
        for month in months:
            rows.append(
                {
                    "reference_month": month,
                    "filter_variant": variant,
                    "evaluation_checkpoint": "before_any_defining_release",
                    "score_center_negative_log_predictive_density": loss,
                    "hard_quadrant_correct": accuracy,
                }
            )
    config = {
        "evaluation": {
            "information_stage_checkpoints": ["before_any_defining_release"],
            "paired_block_bootstrap": {
                "random_seed": 7,
                "replications": 1000,
                "block_length_months": 6,
                "confidence_level": 0.95,
                "samples": ["full_sample"],
                "variants": ["priority_02_claims"],
                "holm_family": ["priority_02_claims"],
                "metrics": {
                    "score_center_negative_log_predictive_density": "lower",
                    "hard_quadrant_correct": "higher",
                },
            },
        }
    }

    result = _paired_block_bootstrap(
        pd.DataFrame.from_records(rows),
        initial_date=pd.Timestamp("2019-09-01"),
        burn_in_months=4,
        config=config,
    )

    loss = result.loc[
        result["metric"].eq("score_center_negative_log_predictive_density")
    ].iloc[0]
    accuracy = result.loc[result["metric"].eq("hard_quadrant_correct")].iloc[0]
    assert loss["mean_candidate_minus_baseline"] == pytest.approx(-0.2)
    assert loss["bootstrap_probability_candidate_better"] == 1.0
    assert accuracy["mean_candidate_minus_baseline"] == pytest.approx(1.0)
    assert accuracy["bootstrap_probability_candidate_better"] == 1.0


def test_paired_block_bootstrap_is_invariant_to_input_row_order() -> None:
    months = pd.date_range("2020-01-01", periods=24, freq="MS")
    rows = []
    for position, month in enumerate(months):
        rows.extend(
            [
                {
                    "reference_month": month,
                    "filter_variant": "student_t_7_combined",
                    "evaluation_checkpoint": "before_any_defining_release",
                    "loss": 0.0,
                },
                {
                    "reference_month": month,
                    "filter_variant": "priority_02_claims",
                    "evaluation_checkpoint": "before_any_defining_release",
                    "loss": float((position % 6) - 2),
                },
            ]
        )
    config = {
        "evaluation": {
            "information_stage_checkpoints": ["before_any_defining_release"],
            "paired_block_bootstrap": {
                "random_seed": 17,
                "replications": 1000,
                "block_length_months": 4,
                "confidence_level": 0.95,
                "samples": ["full_sample"],
                "variants": ["priority_02_claims"],
                "holm_family": ["priority_02_claims"],
                "metrics": {"loss": "lower"},
            },
        }
    }
    frame = pd.DataFrame.from_records(rows)

    chronological = _paired_block_bootstrap(
        frame,
        initial_date=pd.Timestamp("2019-09-01"),
        burn_in_months=4,
        config=config,
    )
    shuffled = _paired_block_bootstrap(
        frame.sample(frac=1.0, random_state=123).reset_index(drop=True),
        initial_date=pd.Timestamp("2019-09-01"),
        burn_in_months=4,
        config=config,
    )

    pd.testing.assert_frame_equal(chronological, shuffled)


def test_claims_dependence_aligns_exact_publication_dates() -> None:
    dates = pd.date_range("2020-01-02", periods=12, freq="7D")
    rows = []
    for position, date in enumerate(dates):
        rows.extend(
            [
                {
                    "model_id": "weekly_labor_stress",
                    "response_id": "initial_claims_innovation",
                    "validation_available_at": date,
                    "standardized_residual": float(position),
                },
                {
                    "model_id": "weekly_continued_claims",
                    "response_id": "continued_claims_innovation",
                    "validation_available_at": date,
                    "standardized_residual": float(position * 2),
                },
            ]
        )
    config = {
        "dependence_diagnostics": {
            "exact_same_publication_pair": {
                "left_model_id": "weekly_labor_stress",
                "left_response_id": "initial_claims_innovation",
                "right_model_id": "weekly_continued_claims",
                "right_response_id": "continued_claims_innovation",
            }
        }
    }

    result = _same_publication_dependence(pd.DataFrame.from_records(rows), config)

    assert set(result["statistic"]) == {"pearson", "spearman"}
    assert result["sample_count"].eq(12).all()
    assert result["correlation"].eq(1.0).all()
