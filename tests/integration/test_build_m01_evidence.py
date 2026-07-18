"""Exercise the release-evidence build from synthetic provider observations.

The suite replaces network acquisition with deterministic monthly and weekly
release records, invokes the production evidence command, and checks event-table
schema, feature provenance, release timing, archive policies, configuration
validation, and output manifests. Inputs and outputs remain inside temporary test
directories; no cached Model 01 data or external service is touched. These tests
protect the point-in-time evidence contract used by the Bayesian filter.
"""

from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest
import yaml

from regime_allocation.cli import build_m01_evidence
from regime_allocation.cli.build_m01_dataset import _MatrixAcquisition
from regime_allocation.data.providers import ProviderSelection
from regime_allocation.data.providers.vintage_matrix import (
    DownloadedFirstReleaseObservations,
    FirstReleaseObservation,
)
from regime_allocation.features.release_evidence import EVENT_TABLE_COLUMNS


MONTHLY_SERIES = (
    "JTSJOR",
    "JTSHIR",
    "JTSQUR",
    "JTSLDR",
    "RSAFS",
    "RSFSXMV",
    "HOUST",
    "PERMIT",
    "DGORDER",
    "NEWORDER",
)


def _vintage_zip(series_id: str, months: pd.DatetimeIndex) -> bytes:
    """Encode deterministic monthly first releases as an in-memory vintage ZIP."""
    series_number = MONTHLY_SERIES.index(series_id) + 1
    values = [80.0 + 5.0 * series_number]
    for position in range(1, len(months)):
        if series_id.startswith("JTS"):
            step = 0.02 * (((position + series_number) % 7) - 3)
            values.append(max(0.5, values[-1] + step))
        else:
            log_change = 0.1 + 0.04 * (((position + series_number) % 6) - 2.5)
            values.append(values[-1] * math.exp(log_change / 100.0))

    matrix = pd.DataFrame(index=months)
    # JTSJOR simulates a newly selected archive whose first vintage exposes
    # three historical months at once. Only the latest belongs in real-time
    # feature history; the older two remain in the normalized audit table.
    first_vintage_position = 2 if series_id == "JTSJOR" else 0
    for position in range(first_vintage_position, len(months)):
        reference_month = months[position]
        release_date = (
            (reference_month.to_period("M") + 1).to_timestamp()
            + pd.Timedelta(days=14)
        )
        column = f"{series_id}_{release_date:%Y%m%d}"
        vintage_values = [float("nan")] * len(months)
        vintage_values[: position + 1] = values[: position + 1]
        matrix[column] = vintage_values
    matrix.index.name = "observation_date"

    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{series_id}_levels_by_vintage.csv",
            matrix.to_csv(na_rep="."),
        )
    return buffer.getvalue()


def _config() -> dict[str, object]:
    """Build a synthetic configuration that preserves the production block contract."""
    blocks: dict[str, object] = {}
    for block_name, expected in build_m01_evidence._EXPECTED_BLOCKS.items():
        sources = [
            {
                "series_id": series_id,
                "feature_name": feature_name,
                "transform": transform,
            }
            for series_id, (feature_name, transform) in expected["sources"].items()
        ]
        blocks[block_name] = {
            "release_id": expected["release_id"],
            "frequency": expected["frequency"],
            "sources": sources,
        }
    return {
        "schema_version": 1,
        "model_id": "m01_deterministic_composite",
        "evidence_set_id": "synthetic_evidence_v1",
        "data": {
            "feature_start": "1999-01-01",
            "observation_start": "1999-01-01",
            "observation_end": "2001-06-30",
            "vintage_start": "1999-01-01",
            "vintage_end": "2001-08-01",
            "monthly_max_release_lag_days": 92,
            "weekly_max_release_lag_days": 92,
            "archive_start_latest_only": True,
        },
        "features": {
            "monthly_min_standardization_history": 3,
            "claims_min_ar_history": 8,
            "claims_min_standardization_history": 4,
            "standard_deviation_ddof": 1,
        },
        "blocks": blocks,
        "outputs": {
            "raw_dir": "data/raw/alfred/synthetic_evidence",
            "processed_dir": "data/processed/synthetic_evidence",
            "events_file": "events.csv",
            "features_file": "features.csv",
            "first_release_observations_file": "observations.csv",
            "manifest": "data/manifests/synthetic_evidence.json",
        },
    }


class _FakeProvider:
    """Supply deterministic first-release observations through the provider API."""

    provider_id = "fred_api"
    provider_description = "synthetic authenticated FRED provider"
    cache_namespace = "fred_api"

    @staticmethod
    def series_page_url(series_id: str) -> str:
        return f"https://fred.stlouisfed.org/series/{series_id}"

    def list_first_release_observations(
        self,
        series_id: str,
        **_: object,
    ) -> DownloadedFirstReleaseObservations:
        start = date(1999, 1, 16)
        archive_bootstrap = (
            FirstReleaseObservation(
                reference_date=date(1999, 1, 2),
                release_date=date(1999, 1, 14),
                value=260_000.0 if series_id == "ICSA" else 1_710_000.0,
            ),
            FirstReleaseObservation(
                reference_date=date(1999, 1, 9),
                release_date=date(1999, 1, 14),
                value=255_000.0 if series_id == "ICSA" else 1_705_000.0,
            ),
        )
        regular = tuple(
            FirstReleaseObservation(
                reference_date=start + timedelta(days=7 * position),
                release_date=start
                + timedelta(
                    days=620 if 80 <= position <= 87 else 7 * position + 5
                ),
                value=(
                    (250_000.0 if series_id == "ICSA" else 1_700_000.0)
                    * math.exp(
                        0.04 * math.sin(position / 5.0)
                        + 0.006 * ((position % 9) - 4)
                    )
                ),
            )
            for position in range(130)
        )
        observations = archive_bootstrap + regular
        return DownloadedFirstReleaseObservations(
            series_id=series_id,
            observations=observations,
            source_url=self.series_page_url(series_id),
            provider_id=self.provider_id,
        )


def test_build_evidence_dataset_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    months = pd.date_range("1999-01-01", "2001-06-01", freq="MS")
    payloads = {
        series_id: _vintage_zip(series_id, months)
        for series_id in MONTHLY_SERIES
    }
    downloaded: list[str] = []
    fake_provider = _FakeProvider()

    monkeypatch.setattr(
        build_m01_evidence,
        "select_vintage_provider",
        lambda *_args, **_kwargs: ProviderSelection(
            requested="fred",
            selected="fred_api",
            client=fake_provider,
        ),
    )

    def fake_download_or_load(
        *,
        series_id: str,
        raw_dir: Path,
        **_: object,
    ) -> _MatrixAcquisition:
        downloaded.append(series_id)
        raw_path = raw_dir / f"{series_id}_synthetic.zip"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payloads[series_id])
        return _MatrixAcquisition(
            path=raw_path,
            content=payloads[series_id],
            provider_id="fred_api",
            cache_origin="downloaded",
            source_url=fake_provider.series_page_url(series_id),
        )

    monkeypatch.setattr(
        build_m01_evidence,
        "_download_or_load",
        fake_download_or_load,
    )

    config_path = tmp_path / "configs" / "evidence.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(_config(), sort_keys=False),
        encoding="utf-8",
    )
    secret_sentinel = "z" * 32
    outputs = build_m01_evidence.build_evidence_dataset(
        project_root=tmp_path,
        config_path=config_path,
        provider="fred",
        environ={"FRED_API_KEY": secret_sentinel},
    )

    assert set(outputs) == {"manifest", "events", "features", "observations"}
    assert all(path.exists() for path in outputs.values())
    assert downloaded == list(MONTHLY_SERIES)
    assert not list(tmp_path.rglob("*.tmp"))

    events = pd.read_csv(
        outputs["events"],
        parse_dates=["release_date", "reference_date", "reference_month"],
    )
    assert tuple(events.columns) == EVENT_TABLE_COLUMNS
    assert set(events["release_block"]) == set(build_m01_evidence._EXPECTED_BLOCKS)
    assert not events["is_target_defining"].any()
    assert events["reference_date"].min() >= pd.Timestamp("1999-01-01")
    assert set(events.loc[events["frequency"] == "weekly", "transform"]) == {
        "expanding_log_ar1_innovation"
    }
    assert events.loc[
        events["frequency"] == "weekly", "ar_lag1_coefficient"
    ].notna().any()
    assert "likelihood" not in events.columns
    assert "posterior" not in events.columns

    available = pd.read_csv(outputs["features"])
    assert tuple(available.columns) == EVENT_TABLE_COLUMNS
    assert set(available["feature_status"]) == {"available"}
    assert available["feature_value"].notna().all()
    assert set(available["release_block"]) == set(
        build_m01_evidence._EXPECTED_BLOCKS
    )

    observations = pd.read_csv(
        outputs["observations"],
        parse_dates=["reference_date", "release_date"],
    )
    assert tuple(observations.columns) == (
        build_m01_evidence.FIRST_RELEASE_OBSERVATION_COLUMNS
    )
    assert set(observations["series_id"]) == {
        "ICSA",
        "CCSA",
        *MONTHLY_SERIES,
    }
    claims_observations = observations.loc[
        observations["frequency"] == "weekly"
    ]
    archive_start = claims_observations.groupby("series_id")["release_date"].transform(
        "min"
    )
    bootstrap_rows = claims_observations["release_date"] == archive_start
    assert set(
        claims_observations.loc[
            bootstrap_rows, ["release_lag_days", "eligible_for_feature"]
        ].itertuples(index=False, name=None)
    ) == {(5, True), (12, False)}
    later_delayed = claims_observations[
        (claims_observations["release_lag_days"] == 60)
        & (claims_observations["release_date"] != archive_start)
    ]
    assert len(later_delayed) == 2
    assert later_delayed["eligible_for_feature"].all()
    assert 60 in set(
        events.loc[events["frequency"] == "weekly", "release_lag_days"]
    )
    jolts_openings = observations[observations["series_id"] == "JTSJOR"]
    jolts_archive_start = jolts_openings["release_date"].min()
    jolts_bootstrap = jolts_openings[
        jolts_openings["release_date"] == jolts_archive_start
    ].sort_values("reference_date")
    assert list(jolts_bootstrap["eligible_for_feature"]) == [False, False, True]
    assert list(jolts_bootstrap["feature_eligibility_status"]) == [
        "archive_bootstrap_history",
        "archive_bootstrap_history",
        "eligible",
    ]
    retained_jolts_start = events[
        (events["series_id"] == "JTSJOR")
        & (events["release_date"] == jolts_archive_start)
    ]
    assert list(retained_jolts_start["reference_date"]) == [
        jolts_bootstrap["reference_date"].max()
    ]

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["provider_selected"] == "fred_api"
    assert manifest["event_rows"] == len(events)
    assert manifest["available_feature_rows"] == len(available)
    assert manifest["release_lag_policy_days"] == {
        "monthly_general_max": 92,
        "weekly_general_max": 92,
    }
    assert manifest["archive_start_policy"] == "latest_reference_period_only"
    assert manifest["ineligible_first_release_observation_rows"] == 4
    assert manifest["first_release_observation_eligibility_status_counts"] == {
        "archive_bootstrap_history": 4,
        "eligible": len(observations) - 4,
    }
    expected_ineligible = {"ICSA": 1, "CCSA": 1, "JTSJOR": 2}
    assert all(
        counts["ineligible_for_feature"] == expected_ineligible.get(series_id, 0)
        for series_id, counts in manifest[
            "first_release_observation_counts_by_series"
        ].items()
    )
    assert len(manifest["raw_acquisitions"]) == 12
    serialized_outputs = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in outputs.values()
    )
    assert secret_sentinel not in serialized_outputs


def test_evidence_configuration_rejects_feature_contract_drift(
    tmp_path: Path,
) -> None:
    config = _config()
    config["blocks"]["jolts"]["sources"][0]["transform"] = "log_difference"
    config_path = tmp_path / "evidence.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="source contract has changed"):
        build_m01_evidence._load_config(config_path)


def test_alfred_transport_floor_does_not_truncate_fred() -> None:
    source = {"alfred_vintage_start": "2013-08-15"}
    configured = date(1994, 1, 1)

    assert build_m01_evidence._provider_vintage_start(
        source,
        configured_start=configured,
        provider_id="alfred_web",
    ) == date(2013, 8, 15)
    assert build_m01_evidence._provider_vintage_start(
        source,
        configured_start=configured,
        provider_id="fred_api",
    ) == configured
