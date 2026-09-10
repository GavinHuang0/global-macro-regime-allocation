"""Offline checks for dated promoted Model 02 input preparation."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_allocation.data import m02_live_inputs as live
from regime_allocation.data.dataset_acquisition import MatrixAcquisition
from regime_allocation.data.providers import FredApiConfigurationError, ProviderSelection
from regime_allocation.data.providers.vintage_matrix import (
    DownloadedFirstReleaseObservations,
    DownloadedVintageMatrix,
    FirstReleaseObservation,
    encode_vintage_matrix,
)
from regime_allocation.models.m02_soft_composite.scores import ALL_COMPONENTS

ROOT = Path(__file__).resolve().parents[3]


def test_missing_credential_fails_before_any_writes(tmp_path: Path) -> None:
    with pytest.raises(FredApiConfigurationError, match="FRED_API_KEY"):
        live.build_live_inputs(
            project_root=tmp_path,
            output_dir=Path("outputs/live"),
            as_of=date(2020, 9, 6),
            environ={},
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "destination",
    [
        "results/published/m02_soft_composite/current",
        "configs/models",
        "src",
        "data/manifests",
        "data/processed/m02_soft_composite",
        "../outside",
        ".",
    ],
)
def test_publication_and_outside_paths_are_protected(tmp_path: Path, destination: str) -> None:
    with pytest.raises(ValueError):
        live._output_directory(tmp_path, Path(destination))
    assert list(tmp_path.iterdir()) == []


def test_future_vintage_is_rejected_instead_of_silently_trimmed(tmp_path: Path) -> None:
    matrix = pd.DataFrame(
        {"PAYEMS_20200908": [1.0, 2.0]},
        index=pd.to_datetime(["2020-06-01", "2020-07-01"]),
    )
    acquired = MatrixAcquisition(
        path=tmp_path / "matrix.zip",
        content=encode_vintage_matrix(matrix, "PAYEMS"),
        provider_id="fred_api",
        cache_origin="test",
        source_url="https://fred.stlouisfed.org/series/PAYEMS",
    )
    with pytest.raises(ValueError, match="after as_of"):
        live._checked_matrix(
            acquired,
            series_id="PAYEMS",
            cutoff=date(2020, 9, 6),
            observation_end=date(2020, 8, 1),
        )


def _components() -> tuple[pd.DataFrame, dict]:
    config, _ = live._score_config(ROOT / live.CONFIG_FILES["scores"])
    config = deepcopy(config)
    config["data"]["reference_start"] = "2010-01-01"
    config["data"]["expected_missing_component_months"] = {}
    months = pd.date_range("2004-01-01", "2012-08-01", freq="MS")
    rows = []
    for j, component in enumerate(ALL_COMPONENTS):
        for i, month in enumerate(months):
            if month == pd.Timestamp("2012-08-01") and component == "core_pce":
                continue
            rows.append(
                {
                    "reference_month": month,
                    "component": component,
                    "series_id": component,
                    "transformed_value": np.sin(i / (3 + j)) + i / 100,
                    "release_date": month + pd.DateOffset(months=1, days=1),
                }
            )
    return pd.DataFrame(rows), config


def test_partial_scores_and_history_gaps_remain_explicit() -> None:
    components, config = _components()
    components = components.loc[
        ~(
            components["reference_month"].eq(pd.Timestamp("2010-04-01"))
            & components["component"].eq("payrolls")
        )
    ]
    scores, gaps = live._score_frames(components, config, date(2012, 9, 6))
    assert pd.isna(scores.loc["2012-08-01", "inflation_score"])
    assert pd.isna(scores.loc["2010-04-01", "growth_score"])
    assert scores.loc["2012-08-01", "growth_score_available_at"] == pd.Timestamp("2012-09-02")
    assert scores.loc[scores["score_available_at"].notna()].index.max() == pd.Timestamp(
        "2012-07-01"
    )
    statuses = gaps.set_index(["reference_month", "component"])["status"]
    assert statuses.loc[(pd.Timestamp("2010-04-01"), "payrolls")] == "unexpected_historical_gap"
    assert (
        statuses.loc[(pd.Timestamp("2012-08-01"), "core_pce")]
        == "awaiting_release_within_lag_limit"
    )


def test_future_release_cannot_enter_score_standardization() -> None:
    components, config = _components()
    components.loc[components.index[0], "release_date"] = pd.Timestamp("2012-09-07")
    with pytest.raises(ValueError, match="release_date contains information after"):
        live._score_frames(components, config, date(2012, 9, 6))


def test_later_values_do_not_change_earlier_scores() -> None:
    components, config = _components()
    before, _ = live._score_frames(components, config, date(2012, 9, 6))
    changed = components.copy()
    changed.loc[changed["reference_month"].ge("2012-01-01"), "transformed_value"] *= 100
    after, _ = live._score_frames(changed, config, date(2012, 9, 6))
    pd.testing.assert_frame_equal(before.loc[:"2011-12-01"], after.loc[:"2011-12-01"])


class _SyntheticFred:
    provider_id = "fred_api"
    cache_namespace = "fred"

    def __init__(self) -> None:
        self.matrix_queries: list[dict] = []
        self.exact_queries: list[dict] = []

    @staticmethod
    def series_page_url(series_id: str) -> str:
        return f"https://fred.stlouisfed.org/series/{series_id}"

    def _matrix(
        self,
        series_id: str,
        observation_start: date,
        observation_end: date,
        vintages: tuple[date, ...],
    ) -> DownloadedVintageMatrix:
        index = pd.date_range(observation_start, observation_end, freq="MS")
        # Different smooth economic series with nonzero transformed variance.
        sequence = index.year.to_numpy() * 12 + index.month.to_numpy() - 1980 * 12
        phase = (sum(map(ord, series_id)) % 19) / 3
        levels = 100 + sequence * 0.07 + np.sin(sequence / 5 + phase)
        values = {}
        for vintage in vintages:
            latest = pd.Timestamp(vintage).to_period("M") - (1 if vintage.day >= 15 else 2)
            revised = levels + np.sin(sequence / 11 + vintage.year) * 0.01
            values[f"{series_id}_{vintage:%Y%m%d}"] = np.where(
                index <= latest.start_time, revised, np.nan
            )
        matrix = pd.DataFrame(values, index=index)
        return DownloadedVintageMatrix(
            series_id=series_id,
            selected_vintage_dates=vintages,
            content=encode_vintage_matrix(matrix, series_id),
            source_url=self.series_page_url(series_id),
            provider_id=self.provider_id,
        )

    def download_level_matrix(self, series_id: str, **query) -> DownloadedVintageMatrix:
        self.matrix_queries.append({"series_id": series_id, **query})
        vintages = tuple(
            timestamp.date()
            for timestamp in (
                pd.date_range(query["vintage_start"], query["vintage_end"], freq="MS")
                + pd.Timedelta(days=14)
            )
            if timestamp.date() <= query["vintage_end"]
        )
        return self._matrix(
            series_id, query["observation_start"], query["observation_end"], vintages
        )

    def download_level_matrix_at_vintages(self, series_id: str, **query) -> DownloadedVintageMatrix:
        self.exact_queries.append({"series_id": series_id, **query})
        return self._matrix(
            series_id, query["observation_start"], query["observation_end"], query["vintage_dates"]
        )

    def list_first_release_observations(
        self, series_id: str, **query
    ) -> DownloadedFirstReleaseObservations:
        index = pd.date_range(query["observation_start"], query["observation_end"], freq="W-SAT")
        observations = tuple(
            FirstReleaseObservation(
                reference_date=reference.date(),
                release_date=(reference + pd.Timedelta(days=5)).date(),
                value=300_000 + 15_000 * np.sin(i / 11) + 5_000 * np.cos(i / 3),
            )
            for i, reference in enumerate(index)
            if (reference + pd.Timedelta(days=5)).date() <= query["vintage_end"]
        )
        return DownloadedFirstReleaseObservations(
            series_id=series_id,
            observations=observations,
            source_url=self.series_page_url(series_id),
            provider_id=self.provider_id,
        )


def test_end_to_end_synthetic_inputs_have_causal_provenance_and_reuse_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative in live.CONFIG_FILES.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    # The manifest hashes implementation sources; create the same read-only tree.
    source = tmp_path / "src"
    shutil.copytree(
        ROOT / "src", source, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
    )
    provider = _SyntheticFred()
    monkeypatch.setattr(
        live,
        "select_vintage_provider",
        lambda *a, **k: ProviderSelection(
            requested="fred",
            selected="fred_api",
            client=provider,
        ),
    )
    published = tmp_path / "results" / "published" / "sentinel.json"
    published.parent.mkdir(parents=True)
    published.write_text("unchanged", encoding="utf-8")
    outputs = live.build_live_inputs(
        project_root=tmp_path,
        output_dir=Path("outputs/m02_live/2020-09-07"),
        as_of=date(2020, 9, 6),
        environ={"FRED_API_KEY": "never-serialized"},
    )
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["as_of"] == "2020-09-06"
    assert manifest["coverage"]["latest_complete_score_month"].startswith("2020-07-01")
    assert "never-serialized" not in outputs["manifest"].read_text(encoding="utf-8")
    assert published.read_text(encoding="utf-8") == "unchanged"
    assert set(outputs) == {"scores", "components", "mapping", "events", "manifest"}
    for query in provider.matrix_queries:
        assert query["vintage_end"] <= date(2020, 9, 6)
        assert query["observation_end"] <= date(2020, 9, 6)
    for query in provider.exact_queries:
        assert max(query["vintage_dates"]) <= date(2020, 9, 6)
    events = pd.read_csv(outputs["events"])
    assert set(events["release_block"]) == {
        "weekly_labor_stress",
        "business_investment_activity_pipeline",
        "consumer_demand_real_decomposition",
    }
    assert pd.to_datetime(events["release_date"]).max() <= pd.Timestamp("2020-09-06")
    mapping = pd.read_csv(outputs["mapping"])
    assert set(mapping["revision_horizon_months"]) == {3, 12}
    assert (
        mapping.loc[mapping["mapping_status"].eq("available"), "probability_sum"].sub(1).abs().max()
        < 1e-10
    )
    original_counts = (len(provider.matrix_queries), len(provider.exact_queries))
    again = live.build_live_inputs(
        project_root=tmp_path,
        output_dir=Path("outputs/m02_live/2020-09-07"),
        as_of=date(2020, 9, 6),
        environ={},
    )
    assert original_counts == (len(provider.matrix_queries), len(provider.exact_queries))
    assert again == outputs
