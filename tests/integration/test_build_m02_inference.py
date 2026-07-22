"""Integration tests for the Model 02 inference artifact builder.

The tests replace the expensive historical replay with a schema-realistic
in-memory result.  Manifest verification, diagnostics adaptation, output
serialization, strict JSON, and generated-file hashes still execute through
the production CLI code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from regime_allocation.cli import build_m02_inference as cli
from regime_allocation.models.m02_soft_composite.joint_filter import JointGaussianState
from regime_allocation.models.m02_soft_composite.probability_map import REGIME_ORDER
from regime_allocation.models.m02_soft_composite.walkforward import GaussianWalkForwardResult


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs/models/m02_event_driven_bayesian_filter.yaml"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(content)
    return content


def _configured_project(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    sources = config["sources"]
    sources.update(
        {
            "score_manifest": "manifests/score.json",
            "score_features": "inputs/scores.csv",
            "probability_map_manifest": "manifests/map.json",
            "probability_map": "inputs/map.csv",
            "evidence_manifest": "manifests/evidence.json",
            "evidence_events": "inputs/events.csv",
        }
    )
    config["calendar"]["replay_end"] = "2020-04-20"
    config["mapping"]["joint_path_samples"] = 256
    outputs = config["outputs"]
    outputs["processed_dir"] = "data/processed/m02_test/bayesian_filter"
    outputs["published_dir"] = "results/published/m02_test/bayesian_filter"
    outputs["manifest"] = "manifests/m02_event_driven_filter.json"

    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True)
    file_payloads = {
        "scores.csv": b"placeholder\nscore\n",
        "map.csv": b"placeholder\nmap\n",
        "events.csv": b"placeholder\nevent\n",
    }
    for name, payload in file_payloads.items():
        (inputs / name).write_bytes(payload)

    score_manifest_path = tmp_path / sources["score_manifest"]
    score_manifest_bytes = _write_json(
        score_manifest_path,
        {
            "model_id": cli.MODEL_ID,
            "generated_file_hashes": [
                {
                    "path": sources["score_features"],
                    "sha256": _sha(file_payloads["scores.csv"]),
                }
            ],
        },
    )
    _write_json(
        tmp_path / sources["probability_map_manifest"],
        {
            "model_id": cli.MODEL_ID,
            "stage_id": "m02_probability_map",
            "score_manifest_sha256": _sha(score_manifest_bytes),
            "generated_file_hashes": [
                {
                    "path": sources["probability_map"],
                    "sha256": _sha(file_payloads["map.csv"]),
                }
            ],
        },
    )
    _write_json(
        tmp_path / sources["evidence_manifest"],
        {
            "model_id": cli.MODEL_ID,
            "stage_id": "m02_release_evidence",
            "generated_files": [
                {
                    "path": sources["evidence_events"],
                    "sha256": _sha(file_payloads["events.csv"]),
                }
            ],
        },
    )
    config_path = tmp_path / "configs/m02_test.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return tmp_path, config_path, config


def _fake_result() -> GaussianWalkForwardResult:
    months = tuple(pd.date_range("2020-01-01", periods=4, freq="MS"))
    state = JointGaussianState(
        reference_months=months,
        mean=np.asarray([-0.3, 0.2, -0.1, 0.1, 0.0, 0.2, 0.2, 0.3]),
        covariance=np.eye(8) * 0.2,
    )
    checkpoint_id = "m02:000001"
    as_of = pd.Timestamp("2020-04-20")
    checkpoints = pd.DataFrame(
        [
            {
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": None,
                "as_of_date": as_of,
                "phase_order": 7,
                "checkpoint_type": "latest",
                "event_models": "inflation_expectations|inflation_input_costs",
                "exact_months": "",
                "same_day_mapping_allowed": False,
                "mapping_reference_month": pd.Timestamp("2020-03-01"),
                "mapping_available_at": pd.Timestamp("2020-04-05"),
                "mapping_status": "available",
            }
        ]
    )
    marginal_rows: list[dict[str, object]] = []
    for variant in ("evidence_filter", "transition_only"):
        for position, month in enumerate(months):
            row: dict[str, object] = {
                "checkpoint_id": checkpoint_id,
                "filter_variant": variant,
                "as_of_date": as_of,
                "checkpoint_type": "latest",
                "reference_month": month,
                "relative_month": position - 3,
                "growth_mean": state.mean[2 * position],
                "inflation_mean": state.mean[2 * position + 1],
                "growth_latent_variance": 0.2,
                "growth_inflation_latent_covariance": 0.0,
                "inflation_latent_variance": 0.2,
                "exact_score_center": False,
                "mapping_status": "available",
                "entropy": 1.35,
            }
            for regime in REGIME_ORDER:
                row[f"probability_{regime}"] = 0.25
            marginal_rows.append(row)
    marginals = pd.DataFrame(marginal_rows)
    path_row = {
        "checkpoint_id": checkpoint_id,
        "filter_variant": "evidence_filter",
        "as_of_date": as_of,
        "checkpoint_type": "latest",
        "regime_0": REGIME_ORDER[0],
        "regime_1": REGIME_ORDER[1],
        "regime_2": REGIME_ORDER[2],
        "regime_3": REGIME_ORDER[3],
        "probability": 1.0,
        "sobol_samples": 256,
        "maximum_marginal_probability_error": 0.0,
    }
    audit_payload = {
        "specification": {"response_names": ["one_year_inflation_expectations_change"]},
        "intercept": {"one_year_inflation_expectations_change": 0.1},
        "state_loadings": {
            "one_year_inflation_expectations_change": {
                "growth_score": 0.05,
                "inflation_score": 0.8,
            }
        },
        "control_loadings": {"one_year_inflation_expectations_change": {}},
        "residual_covariance": {
            "one_year_inflation_expectations_change": {
                "one_year_inflation_expectations_change": 0.4
            }
        },
    }
    emission_fits = pd.DataFrame(
        [
            {
                "fit_id": "emission:inflation_expectations:000001",
                "fit_date": pd.Timestamp("2020-04-01"),
                "observation_model_id": "inflation_expectations",
                "training_signature": "abc",
                "training_count": 24,
                "distinct_target_months": 24,
                "first_training_available_at": pd.Timestamp("2018-01-01"),
                "last_training_available_at": pd.Timestamp("2020-03-01"),
                "selected_base_lambda": 1.0,
                "active_controls": "",
                "dropped_unidentified_controls": "",
                "ledoit_wolf_shrinkage": 0.2,
                "fit_audit_json": json.dumps(audit_payload),
            }
        ]
    )
    evaluations = pd.DataFrame(
        [
            {
                "availability_date": pd.Timestamp("2020-04-05"),
                "reference_month": pd.Timestamp("2020-02-01"),
                "filter_variant": variant,
                "evaluation_checkpoint": "strict_pre_day",
                "same_day_release_evidence_included": False,
                "growth_error": 0.1,
                "inflation_error": -0.2,
                "growth_squared_error": 0.01,
                "inflation_squared_error": 0.04,
                "score_center_negative_log_predictive_density": 1.2,
                "quadrant_cross_entropy_to_exact_score_map": np.nan,
                "quadrant_brier_distance_to_exact_score_map": np.nan,
                "quadrant_kl_divergence_to_exact_score_map": np.nan,
            }
            for variant in ("evidence_filter", "transition_only")
        ]
    )
    residual_rows: list[dict[str, object]] = []
    residual_months = pd.date_range("2019-01-01", periods=14, freq="MS")
    for position, month in enumerate(residual_months):
        for model_id, response_id, value in (
            ("inflation_expectations", "expectations", np.sin(position / 2.0)),
            ("inflation_input_costs", "input_costs", np.cos(position / 3.0)),
        ):
            residual_rows.append(
                {
                    "release_date": month + pd.Timedelta(days=10),
                    "reference_month": month,
                    "observation_model_id": model_id,
                    # This pre-existing column caused the former duplicate-name bug.
                    "block_id": model_id,
                    "economic_block": "inflation_pressure",
                    "event_instance_id": f"{model_id}:{month:%Y-%m}",
                    "response_id": response_id,
                    "source_series_id": "TEST",
                    "innovation": float(value),
                    "predictive_standard_deviation": 1.0,
                    "standardized_innovation": float(value),
                    "whitened_innovation": float(value),
                    "state_update_applied": True,
                }
            )
    return GaussianWalkForwardResult(
        checkpoints=checkpoints,
        joint_gaussians=pd.DataFrame([{"checkpoint_id": checkpoint_id}]),
        marginals=marginals,
        joint_paths=pd.DataFrame([path_row]),
        event_audit=pd.DataFrame(
            [
                {
                    "release_date": pd.Timestamp("2020-04-10"),
                    "update_status": "applied",
                }
            ]
        ),
        exact_score_audit=pd.DataFrame(
            [
                {
                    "availability_date": pd.Timestamp("2020-04-05"),
                    "reference_month": pd.Timestamp("2020-02-01"),
                    "filter_variant": "both",
                    "status": "conditioned_exactly",
                }
            ]
        ),
        transition_fit_audit=pd.DataFrame(
            [{"transition_fit_id": "var1:2020-04-01", "training_pairs": 60}]
        ),
        emission_fit_audit=emission_fits,
        evaluation_rows=evaluations,
        predictive_residuals=pd.DataFrame(residual_rows),
        latest_states={"evidence_filter": state, "transition_only": state},
        initial_date=pd.Timestamp("2020-01-01"),
        replay_end=as_of,
    )


def _strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def test_build_serializes_every_output_and_hashes_exact_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, config_path, config = _configured_project(tmp_path)
    prepared = SimpleNamespace(
        preparation_audit=pd.DataFrame(
            [{"observation_model_id": "inflation_expectations", "live_event_vectors": 14}]
        )
    )
    result = _fake_result()
    monkeypatch.setattr(cli, "prepare_observation_data", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(cli, "run_event_driven_filter", lambda *_args, **_kwargs: result)

    returned = cli.build_inference(project_root=project_root, config_path=config_path)

    manifest = _strict_json(returned["manifest"])
    generated = manifest["generated_file_hashes"]
    expected_count = len(manifest["processed_files"]) + len(manifest["published_files"])
    assert len(generated) == expected_count
    assert len({row["path"] for row in generated}) == expected_count
    for row in generated:
        artifact = project_root / row["path"]
        assert artifact.is_file()
        assert _sha(artifact.read_bytes()) == row["sha256"]
        assert artifact.stat().st_size == row["bytes"]

    outputs = config["outputs"]
    processed = project_root / outputs["processed_dir"]
    published = project_root / outputs["published_dir"]
    assert (processed / outputs["joint_gaussian_checkpoints"]).read_bytes()[:2] == b"\x1f\x8b"
    assert (processed / outputs["joint_path_checkpoints"]).read_bytes()[:2] == b"\x1f\x8b"
    _strict_json(published / outputs["latest_posterior"])
    filter_summary = _strict_json(published / outputs["published_filter_summary"])
    assert filter_summary["evaluation"][0][
        "mean_quadrant_cross_entropy_to_exact_score_map"
    ] is None

    pairwise = pd.read_csv(published / outputs["published_dependence_pairwise"])
    same_block = pairwise.loc[pairwise["same_economic_block"].astype(bool)]
    assert not same_block.empty
    assert set(same_block["left_block_id"]) == {"inflation_pressure"}
    assert set(same_block["right_block_id"]) == {"inflation_pressure"}
    assert set(same_block["left_model_id"]).union(same_block["right_model_id"]) == {
        "inflation_expectations",
        "inflation_input_costs",
    }


def test_hash_mismatch_stops_before_model_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, config_path, _ = _configured_project(tmp_path)
    (project_root / "inputs/scores.csv").write_text("changed\nvalue\n", encoding="utf-8")

    def must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("model preparation must not run after failed lineage checks")

    monkeypatch.setattr(cli, "prepare_observation_data", must_not_run)
    with pytest.raises(ValueError, match="input hash mismatch"):
        cli.build_inference(project_root=project_root, config_path=config_path)


def test_diagnostics_adapter_uses_economic_block_without_duplicate_columns() -> None:
    residuals = _fake_result().predictive_residuals

    adapted = cli._dependence_residuals(residuals)

    assert adapted.columns.is_unique
    assert set(adapted["block_id"]) == {"inflation_pressure"}
    assert set(adapted["model_id"]) == {
        "inflation_expectations",
        "inflation_input_costs",
    }


def test_default_configuration_and_console_entry_point_are_registered() -> None:
    config, _ = cli._load_config(DEFAULT_CONFIG)
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert config["dependence"]["minimum_pair_observations"] == 12
    assert config["dependence"]["serial_significance_level"] == pytest.approx(0.05)
    assert (
        'build-m02-inference = "regime_allocation.cli.build_m02_inference:main"'
        in pyproject
    )
