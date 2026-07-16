"""End-to-end contract tests for the model 01 transition build."""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from regime_allocation.cli.build_m01_transition import build_transition
from regime_allocation.models.m01_deterministic_composite.pipeline import Regime


STATE_ORDER = [
    Regime.GROWTH_UP_INFLATION_UP.value,
    Regime.GROWTH_DOWN_INFLATION_UP.value,
    Regime.GROWTH_UP_INFLATION_DOWN.value,
    Regime.GROWTH_DOWN_INFLATION_DOWN.value,
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_history(path: Path) -> None:
    """Write labels containing a missing month and a not-yet-known label."""

    regimes = [
        STATE_ORDER[0],
        STATE_ORDER[0],
        STATE_ORDER[1],
        None,
        STATE_ORDER[2],
        STATE_ORDER[2],
        STATE_ORDER[3],
        STATE_ORDER[3],
        STATE_ORDER[0],
    ]
    available_at = [
        "2020-02-05",
        "2020-03-05",
        "2020-04-05",
        None,
        "2020-06-05",
        "2020-07-05",
        "2020-08-10",
        "2020-09-10",
        "2020-10-10",
    ]
    statuses = [
        "classified",
        "classified",
        "classified",
        "missing_component_feature",
        "classified",
        "classified",
        "classified",
        "classified",
        "classified",
    ]
    frame = pd.DataFrame(
        {
            "reference_month": pd.date_range("2020-01-01", periods=9, freq="MS"),
            "regime_id": regimes,
            "label_available_at": available_at,
            "data_status": statuses,
        }
    )
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")


def _write_config(path: Path) -> None:
    config = {
        "schema_version": 1,
        "model_id": "m01_deterministic_composite",
        "stage_id": "fixed_first_order_transition",
        "source": {
            "regime_history": (
                "results/published/m01_deterministic_composite/"
                "regime_history.csv"
            )
        },
        "transition": {
            "markov_order": 1,
            "path_months": 4,
            "time_homogeneous": True,
            "estimation_window": "expanding",
            "dirichlet_alpha": 0.5,
            "credible_interval_level": 0.95,
            "knowledge_cutoff_policy": "latest_label_available_at",
            "require_consecutive_reference_months": True,
            "availability_field": "label_available_at",
        },
        "outputs": {
            "processed_pairs": (
                "data/processed/m01_deterministic_composite/"
                "transition_pairs.csv"
            ),
            "published_model": (
                "results/published/m01_deterministic_composite/"
                "transition_model.json"
            ),
            "published_matrix": (
                "results/published/m01_deterministic_composite/"
                "transition_matrix.csv"
            ),
            "published_prior": (
                "results/published/m01_deterministic_composite/"
                "latest_transition_prior.json"
            ),
        },
    }
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_build_transition_writes_causal_auditable_artifacts(tmp_path: Path) -> None:
    history_path = (
        tmp_path
        / "results"
        / "published"
        / "m01_deterministic_composite"
        / "regime_history.csv"
    )
    config_path = tmp_path / "configs" / "models" / "m01_transition.yaml"
    _write_history(history_path)
    _write_config(config_path)

    outputs = build_transition(
        project_root=tmp_path,
        config_path=config_path,
        knowledge_cutoff=date(2020, 9, 15),
    )

    assert set(outputs) == {"model", "matrix", "pairs", "prior"}
    assert outputs["model"] == (
        tmp_path
        / "results"
        / "published"
        / "m01_deterministic_composite"
        / "transition_model.json"
    )
    assert outputs["matrix"].name == "transition_matrix.csv"
    assert outputs["pairs"].name == "transition_pairs.csv"
    assert outputs["prior"].name == "latest_transition_prior.json"
    assert all(path.exists() for path in outputs.values())

    pairs = pd.read_csv(
        outputs["pairs"],
        parse_dates=[
            "source_reference_month",
            "destination_reference_month",
            "source_label_available_at",
            "destination_label_available_at",
        ],
    )
    # Every adjacent input-row pair remains available for audit, including the
    # missing-label and after-cutoff exclusions.
    assert len(pairs) == 8
    assert int(pairs["included"].sum()) == 5
    assert list(
        pairs.loc[pairs["included"], "source_reference_month"].dt.strftime(
            "%Y-%m-%d"
        )
    ) == [
        "2020-01-01",
        "2020-02-01",
        "2020-05-01",
        "2020-06-01",
        "2020-07-01",
    ]
    assert not pairs.loc[
        pairs["source_reference_month"] == pd.Timestamp("2020-03-01"),
        "included",
    ].item()
    assert not pairs.loc[
        pairs["source_reference_month"] == pd.Timestamp("2020-04-01"),
        "included",
    ].item()
    after_cutoff = pairs.loc[
        pairs["destination_reference_month"] == pd.Timestamp("2020-09-01")
    ].iloc[0]
    assert not after_cutoff["included"]
    assert "cutoff" in after_cutoff["exclusion_reason"]

    matrix = pd.read_csv(outputs["matrix"])
    assert len(matrix) == 16
    assert list(matrix["from_regime_id"].drop_duplicates()) == STATE_ORDER
    for source_regime in STATE_ORDER:
        row = matrix.loc[matrix["from_regime_id"] == source_regime]
        assert list(row["to_regime_id"]) == STATE_ORDER
        assert row["posterior_predictive_probability"].sum() == pytest.approx(1.0)
        assert row["from_row_total"].nunique() == 1

    counts = matrix.pivot(
        index="from_regime_id",
        columns="to_regime_id",
        values="transition_count",
    ).reindex(index=STATE_ORDER, columns=STATE_ORDER)
    assert counts.to_numpy().tolist() == [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]
    down_down_row = matrix.loc[
        matrix["from_regime_id"] == STATE_ORDER[3]
    ]
    assert list(down_down_row["posterior_predictive_probability"]) == pytest.approx(
        [1 / 6, 1 / 6, 1 / 6, 1 / 2]
    )

    model = json.loads(outputs["model"].read_text(encoding="utf-8"))
    assert model["model_id"] == "m01_deterministic_composite"
    assert model["stage_id"] == "fixed_first_order_transition"
    assert model["knowledge_cutoff"] == "2020-09-15"
    assert [item["regime_id"] for item in model["state_order"]] == STATE_ORDER
    assert model["transition_specification"]["dirichlet_alpha"] == pytest.approx(
        0.5
    )
    assert model["marginal_credible_intervals"]["level"] == pytest.approx(0.95)
    assert model["diagnostics"]["adjacent_row_pairs"] == 8
    assert model["diagnostics"]["included_transitions"] == 5
    assert model["source"]["regime_history_sha256"] == _sha256(history_path)
    assert model["configuration"]["sha256"] == _sha256(config_path)

    prior = json.loads(outputs["prior"].read_text(encoding="utf-8"))
    assert prior["status"] == "transition_only_prior"
    assert prior["knowledge_cutoff"] == "2020-09-15"
    assert prior["source_reference_month"] == "2020-08-01"
    assert prior["conditioning_regime_id"] == STATE_ORDER[3]
    assert prior["target_reference_month"] == "2020-09-01"
    assert [item["regime_id"] for item in prior["probabilities"]] == STATE_ORDER
    assert [item["probability"] for item in prior["probabilities"]] == pytest.approx(
        [1 / 6, 1 / 6, 1 / 6, 1 / 2]
    )
    assert "joint" not in json.dumps(prior).lower()


def test_build_transition_defaults_to_latest_label_availability(
    tmp_path: Path,
) -> None:
    history_path = (
        tmp_path
        / "results"
        / "published"
        / "m01_deterministic_composite"
        / "regime_history.csv"
    )
    config_path = tmp_path / "configs" / "models" / "m01_transition.yaml"
    _write_history(history_path)
    _write_config(config_path)

    outputs = build_transition(project_root=tmp_path, config_path=config_path)
    model = json.loads(outputs["model"].read_text(encoding="utf-8"))
    prior = json.loads(outputs["prior"].read_text(encoding="utf-8"))

    assert model["knowledge_cutoff"] == "2020-10-10"
    assert model["diagnostics"]["included_transitions"] == 6
    assert prior["source_reference_month"] == "2020-09-01"
    assert prior["target_reference_month"] == "2020-10-01"
