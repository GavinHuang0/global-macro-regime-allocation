"""Contract tests for the Model 02 existing-block attribution stage."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_allocation.cli.build_m02_existing_block_attribution import (
    _invariance_rows,
    _load_attribution_config,
    _paired_bootstrap,
    _paired_comparisons,
    _segment_aware_circular_means,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/models/m02_existing_block_attribution.yaml"


def test_production_contract_has_exact_atomic_ablation_grid() -> None:
    config, raw = _load_attribution_config(CONFIG)

    assert raw
    atomic = {
        block_id
        for block_id, declaration in config["blocks"].items()
        if declaration["primary_atomic"]
    }
    assert len(atomic) == 7
    assert len(config["comparisons"]) == 16
    assert sum(row["primary_atomic"] for row in config["comparisons"]) == 14
    assert {
        row["reference_id"]
        for row in config["comparisons"]
        if row["experiment_arm"] == "add_one"
    } == {"partial_only"}
    assert {
        row["reference_id"]
        for row in config["comparisons"]
        if row["experiment_arm"] == "leave_one_out"
    } == {"student_t_7_combined"}


def test_contract_rejects_noncomplementary_leave_one_set(tmp_path: Path) -> None:
    config, _ = _load_attribution_config(CONFIG)
    broken = deepcopy(config)
    broken["evidence_sets"]["without_consumer_demand"]["observation_models"].append(
        "consumer_demand"
    )
    import yaml

    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(broken, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="leave-one-out"):
        _load_attribution_config(path)


def _toy_frames() -> dict[str, pd.DataFrame]:
    months = pd.date_range("2020-01-01", periods=24, freq="MS")
    frames: dict[str, pd.DataFrame] = {}
    for variant, offset in (("partial_only", 0.0), ("add_one_test", -0.1)):
        frames[variant] = pd.DataFrame(
            {
                "reference_month": months,
                "evaluation_checkpoint": "before_any_defining_release",
                "score_center_negative_log_predictive_density": [
                    float(position % 5) + offset for position in range(len(months))
                ],
                "quadrant_cross_entropy_to_exact_score_map": 1.0 + offset,
                "quadrant_brier_distance_to_exact_score_map": 0.5 + offset,
                "hard_quadrant_correct": 0.5 - offset,
            }
        )
    return frames


def _toy_config() -> dict[str, object]:
    return {
        "evaluation": {
            "primary_checkpoint": "before_any_defining_release",
            "information_stage_checkpoints": ["before_any_defining_release"],
            "metrics": {
                "score_center_negative_log_predictive_density": "lower",
                "quadrant_cross_entropy_to_exact_score_map": "lower",
                "quadrant_brier_distance_to_exact_score_map": "lower",
                "hard_quadrant_correct": "higher",
            },
            "paired_block_bootstrap": {
                "random_seed": 7,
                "replications": 1000,
                "block_length_months": 4,
                "confidence_level": 0.95,
                "samples": ["full_sample"],
                "holm_families": {"primary_atomic": ["add_one_test__vs__partial_only"]},
            },
        }
    }


def test_add_one_benefit_direction_and_bootstrap_order_invariance() -> None:
    declaration = {
        "candidate_id": "add_one_test",
        "reference_id": "partial_only",
        "block_id": "test",
        "experiment_arm": "add_one",
        "primary_atomic": True,
    }
    frames = _toy_frames()
    comparison = _paired_comparisons(
        frames,
        [declaration],
        event_audit=None,
        blocks={"test": {"observation_models": ["test"]}},
    )
    assert comparison.iloc[0][
        "mean_block_benefit_quadrant_cross_entropy_to_exact_score_map"
    ] == pytest.approx(0.1)
    assert comparison.iloc[0]["mean_block_benefit_hard_quadrant_correct"] == pytest.approx(
        0.1
    )

    first = _paired_bootstrap(frames, [declaration], _toy_config())
    shuffled = {
        key: frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
        for key, frame in frames.items()
    }
    second = _paired_bootstrap(shuffled, [declaration], _toy_config())
    pd.testing.assert_frame_equal(first, second)
    assert first["bootstrap_probability_block_beneficial"].eq(1.0).all()


def test_segment_aware_bootstrap_preserves_each_calendar_segment_size() -> None:
    means = _segment_aware_circular_means(
        (pd.Series([1.0] * 5).to_numpy(), pd.Series([10.0] * 5).to_numpy()),
        replications=100,
        block_length=4,
        rng=np.random.default_rng(9),
    )

    assert means == pytest.approx(np.full(100, 5.5))


def test_invariance_is_parameterized_by_control_variant() -> None:
    frozen = pd.DataFrame(
        {
            "filter_variant": ["partial_only"],
            "reference_month": ["2020-01-01"],
            "evaluation_checkpoint": ["checkpoint"],
            "loss": [1.0],
        }
    )
    current = frozen.copy()
    rows = _invariance_rows(
        current,
        frozen,
        artifact="evaluation",
        variant_column="filter_variant",
        variant_id="partial_only",
        key_columns=("reference_month", "evaluation_checkpoint"),
    )
    assert all(row["passed"] for row in rows)

    current.loc[0, "loss"] = 1.1
    rows = _invariance_rows(
        current,
        frozen,
        artifact="evaluation",
        variant_column="filter_variant",
        variant_id="partial_only",
        key_columns=("reference_month", "evaluation_checkpoint"),
    )
    assert any(not row["passed"] and row["column"] == "loss" for row in rows)


def test_invariance_normalizes_object_and_datetime_missing_values() -> None:
    frozen = pd.DataFrame(
        {
            "variant_id": ["partial_only"],
            "release_date": ["2020-01-02"],
            "reference_month": ["2020-01-01"],
            "knowledge_cutoff": pd.Series([None], dtype=object),
        }
    )
    current = frozen.copy()
    current["knowledge_cutoff"] = pd.to_datetime(current["knowledge_cutoff"])

    rows = _invariance_rows(
        current,
        frozen,
        artifact="partial",
        variant_column="variant_id",
        variant_id="partial_only",
        key_columns=("release_date", "reference_month"),
    )

    assert all(row["passed"] for row in rows)


def test_invariance_normalizes_datetime_resolution_and_blank_serialization() -> None:
    frozen = pd.DataFrame(
        {
            "variant_id": ["partial_only"],
            "as_of_date": pd.Series(["2026-07-20"], dtype="datetime64[ns]"),
            "conditioned_components": [pd.NA],
        }
    )
    current = frozen.copy()
    current["as_of_date"] = current["as_of_date"].astype("datetime64[s]")
    current["conditioned_components"] = ""

    rows = _invariance_rows(
        current,
        frozen,
        artifact="latest",
        variant_column="variant_id",
        variant_id="partial_only",
        key_columns=("as_of_date",),
    )

    assert all(row["passed"] for row in rows)
