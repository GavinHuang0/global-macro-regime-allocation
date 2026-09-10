"""Guard operational M02 dates, policy identity, freshness, and publication boundaries."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.scores import ALL_COMPONENTS
from regime_allocation.portfolio.m02_live import (
    freshness_audit,
    live_replay_configuration,
    publish_live_files,
    signal_calendar,
)
from regime_allocation.reporting.readme_snapshot import END_MARKER, START_MARKER

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 9, 9, 16, tzinfo=UTC)
WEEK = date(2026, 9, 7)
FEATURES = (
    ("weekly_labor_stress", "initial_claims_innovation"),
    ("consumer_demand_real_decomposition", "real_retail_and_food_services_log_change"),
    ("consumer_demand_real_decomposition", "implicit_retail_price_log_change"),
    ("business_investment_activity_pipeline", "core_capital_goods_shipments_log_change"),
    ("business_investment_activity_pipeline", "core_capital_goods_orders_shipments_gap_log_change"),
)


@pytest.fixture
def macro_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.DataFrame(
        [
            {
                "release_block": block,
                "feature_name": feature,
                "release_date": "2026-09-03",
                "feature_status": "available",
                "feature_value": 0.1,
            }
            for block, feature in FEATURES
        ]
    )
    components = pd.DataFrame(
        [
            {"component": component, "release_date": "2026-08-28", "transformed_value": 0.1}
            for component in ALL_COMPONENTS
        ]
    )
    return events, components


@pytest.mark.parametrize(
    "instant,week,price_day",
    [
        ("2026-09-07T03:59:59+00:00", "2026-08-31", "2026-09-06"),
        ("2026-09-07T04:00:00+00:00", "2026-09-07", "2026-09-06"),
        ("2026-03-09T04:00:00+00:00", "2026-03-09", "2026-03-08"),
        ("2026-11-02T04:59:59+00:00", "2026-10-26", "2026-11-01"),
        ("2026-11-02T05:00:00+00:00", "2026-11-02", "2026-11-01"),
        ("2026-09-09T20:29:59+00:00", "2026-09-07", "2026-09-08"),
        ("2026-09-09T20:30:00+00:00", "2026-09-07", "2026-09-09"),
    ],
)
def test_signal_calendar_uses_new_york_week_and_completed_close(instant, week, price_day):
    assert signal_calendar(datetime.fromisoformat(instant)) == (
        date.fromisoformat(week),
        date.fromisoformat(price_day),
    )


def test_signal_calendar_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone"):
        signal_calendar(datetime(2026, 9, 9))  # noqa: DTZ001 - exercise naive-input rejection


def test_live_replay_keeps_promoted_parameters_without_requiring_diagnostics() -> None:
    base, run = live_replay_configuration(ROOT)
    variants = {item["id"]: item for item in run["variants"]}

    assert set(variants) == {"student_t_7_reduced_core", "transition_only", "partial_only"}
    baseline = variants["student_t_7_reduced_core"]
    assert baseline["emission"] == "student_t_7"
    assert baseline["var"] == "ols"
    assert baseline["partial_defining_releases"] is True
    assert baseline["evidence_set"] == "reduced_core"
    assert run["evidence_sets"]["reduced_core"]["observation_models"] == [
        "weekly_labor_stress",
        "consumer_real_implicit_joint",
        "business_activity_pipeline_joint",
    ]
    assert base["observation_models"]["business_activity_pipeline_joint"]["responses"] == [
        "core_capital_goods_shipments_log_change",
        "core_capital_goods_orders_shipments_gap_log_change",
    ]


def test_complete_usable_macro_families_pass_freshness(macro_frames) -> None:
    result = freshness_audit(*macro_frames, WEEK)

    assert result["status"] == "fresh"
    assert max(result["latest_release_by_family"].values()) == "2026-09-03"


@pytest.mark.parametrize("kind", ["evidence", "defining"])
def test_missing_required_family_cannot_be_fresh(macro_frames, kind) -> None:
    events, components = macro_frames
    if kind == "evidence":
        events = events.loc[~events["release_block"].eq("weekly_labor_stress")]
    else:
        components = components.loc[~components["component"].eq("core_cpi")]

    with pytest.raises(ValueError, match="missing|Missing|usable|required"):
        freshness_audit(events, components, WEEK)


def test_one_missing_feature_in_joint_evidence_cannot_be_fresh(macro_frames) -> None:
    events, components = macro_frames
    events = events.loc[~events["feature_name"].eq("implicit_retail_price_log_change")]

    with pytest.raises(ValueError, match="missing|Missing|usable|required"):
        freshness_audit(events, components, WEEK)


def test_recent_unavailable_claims_cannot_hide_stale_usable_history(macro_frames) -> None:
    events, components = macro_frames
    latest = events.loc[events["release_block"].eq("weekly_labor_stress")].copy()
    latest["feature_status"] = "insufficient_history"
    latest["feature_value"] = float("nan")
    events.loc[events["release_block"].eq("weekly_labor_stress"), "release_date"] = "2026-08-01"
    events = pd.concat([events, latest], ignore_index=True)

    result = freshness_audit(events, components, WEEK)

    assert result["status"] == "stale"
    assert "2026-08-01" in result["reason"]


@pytest.mark.parametrize("kind", ["evidence", "defining"])
def test_signal_day_information_is_rejected_even_when_unavailable(macro_frames, kind) -> None:
    events, components = macro_frames
    target = events if kind == "evidence" else components
    target.loc[0, "release_date"] = "2026-09-07"

    with pytest.raises(ValueError, match="cutoff"):
        freshness_audit(events, components, WEEK)


@pytest.fixture
def publication(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    snapshot = {
        "schema_version": 1,
        "status": "research_only",
        "model_variant": "student_t_7_reduced_core",
        "release_id": "m02_weekly_live_v1",
        "generated_at": "2026-09-07T22:30:00Z",
        "signal_week": "2026-09-07",
        "information_cutoff": "2026-09-07T00:00:00-04:00",
        "macro_as_of": "2026-09-03",
        "price_as_of": "2026-09-04",
        "valid_until": "2026-09-14T00:00:00-04:00",
        "data_freshness": {"status": "fresh", "reason": "Input coverage checks passed."},
        "weights": {
            "SPY": 0.35,
            "IEF": 0.0,
            "TIP": 0.0,
            "HYG": 0.15,
            "BIL": 0.0,
            "GLD": 0.25,
            "LQD": 0.25,
        },
        "quadrant_posterior": {
            "growth_up_inflation_up": 0.25,
            "growth_down_inflation_up": 0.35,
            "growth_up_inflation_down": 0.20,
            "growth_down_inflation_down": 0.20,
        },
        "forecast_weekly_return": 0.001,
        "estimated_annual_volatility": 0.09,
        "optimization_outcome": "optimal",
    }
    working = tmp_path / "outputs"
    working.mkdir()
    paths = {
        name: working / name
        for name in ("latest_allocation.json", "performance_summary.csv", "run_manifest.json")
    }
    paths["latest_allocation.json"].write_text(json.dumps(snapshot), encoding="utf-8")
    performance = pd.read_csv(
        ROOT / "results/published/m02_regime_allocation_backtest/performance_summary.csv"
    )
    selected = performance.loc[
        performance["method"].isin(
            {"posterior_optimized", "pooled_mean_optimizer", "equal_weight", "static_60_spy_40_agg"}
        )
    ]
    selected.to_csv(paths["performance_summary.csv"], index=False)
    manifest = {
        "schema_version": 1,
        "release_id": snapshot["release_id"],
        "signal_week": snapshot["signal_week"],
        "generated_at": snapshot["generated_at"],
        "information_cutoff": snapshot["information_cutoff"],
        "sha256_outputs": {
            name: hashlib.sha256(paths[name].read_bytes()).hexdigest()
            for name in ("latest_allocation.json", "performance_summary.csv")
        },
    }
    paths["run_manifest.json"].write_text(json.dumps(manifest), encoding="utf-8")
    public = tmp_path / "results/live/m02_weekly"
    public.mkdir(parents=True)
    for name in paths:
        (public / name).write_bytes(f"previous successful {name}".encode())
    (tmp_path / "README.md").write_text(
        f"# Research\n{START_MARKER}\nprevious snapshot\n{END_MARKER}\nOriginal docs\n",
        encoding="utf-8",
    )
    return tmp_path, paths


def _public_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in [root / "README.md", *(root / "results/live/m02_weekly").glob("*")]
        if not path.name.endswith(".tmp")
    }


def test_successful_publication_changes_only_marked_readme_content(publication) -> None:
    root, artifacts = publication
    before = (root / "README.md").read_text(encoding="utf-8")

    publish_live_files(root, artifacts, now=NOW)

    after = (root / "README.md").read_text(encoding="utf-8")
    assert after.split(START_MARKER)[0] == before.split(START_MARKER)[0]
    assert after.split(END_MARKER)[1] == before.split(END_MARKER)[1]
    assert "FRESH at generation" in after
    for name, artifact in artifacts.items():
        assert (root / "results/live/m02_weekly" / name).read_bytes() == artifact.read_bytes()


def test_expired_fresh_flag_cannot_overwrite_current_publication(publication) -> None:
    root, artifacts = publication
    previous = _public_bytes(root)

    with pytest.raises(ValueError, match="expired|stale|current|week"):
        publish_live_files(root, artifacts, now=datetime(2026, 9, 14, 4, tzinfo=UTC))

    assert _public_bytes(root) == previous


@pytest.mark.parametrize(
    "failure", ["snapshot_hash", "performance_hash", "release", "week", "markers"]
)
def test_failed_publication_validation_preserves_all_public_files(publication, failure) -> None:
    root, artifacts = publication
    manifest = json.loads(artifacts["run_manifest.json"].read_bytes())
    if failure.endswith("hash"):
        name = "latest_allocation.json" if failure == "snapshot_hash" else "performance_summary.csv"
        manifest["sha256_outputs"][name] = "0" * 64
    elif failure == "release":
        manifest["release_id"] = "other_release"
    elif failure == "week":
        manifest["signal_week"] = "2026-08-31"
    else:
        (root / "README.md").write_text("# Missing snapshot markers\n", encoding="utf-8")
    artifacts["run_manifest.json"].write_text(json.dumps(manifest), encoding="utf-8")
    previous = _public_bytes(root)

    with pytest.raises(ValueError):
        publish_live_files(root, artifacts, now=NOW)

    assert _public_bytes(root) == previous
