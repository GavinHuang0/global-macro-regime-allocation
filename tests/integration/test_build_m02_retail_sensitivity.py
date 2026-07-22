"""Exercise the authenticated Model 02 retail sensitivity without network I/O.

The test supplies matching synthetic RSAFS and RRSFS vintage matrices through
the provider-neutral acquisition boundary.  It verifies same-vintage
nominal/real decomposition, strictly backward-looking standardization,
manifest lineage, and the absence of the runtime credential from every
generated artifact.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from regime_allocation.cli import build_m02_retail_sensitivity
from regime_allocation.data.dataset_acquisition import MatrixAcquisition, sha256
from regime_allocation.data.providers import ProviderSelection
from regime_allocation.data.providers.vintage_matrix import encode_vintage_matrix
from regime_allocation.features.release_evidence import validate_event_table


class _FakeFred:
    provider_id = "fred_api"
    provider_description = "synthetic authenticated FRED provider"
    cache_namespace = "fred_api"

    @staticmethod
    def series_page_url(series_id: str) -> str:
        return f"https://fred.stlouisfed.org/series/{series_id}"


def _vintage_matrix(series_id: str, *, nominal: bool) -> bytes:
    months = pd.date_range("2020-01-01", periods=9, freq="MS")
    real_growth = np.array([0.004, 0.006, -0.002, 0.010, 0.003, 0.007, 0.001, 0.009])
    price_growth = np.array([0.001, 0.003, 0.002, -0.001, 0.004, 0.002, 0.005, 0.001])
    real = np.r_[100.0, 100.0 * np.exp(np.cumsum(real_growth))]
    price = np.r_[2.0, 2.0 * np.exp(np.cumsum(price_growth))]
    levels = real * price if nominal else real

    matrix = pd.DataFrame(index=months)
    for position, month in enumerate(months):
        release = (month.to_period("M") + 1).to_timestamp() + pd.Timedelta(days=14)
        values = np.full(len(months), np.nan)
        values[: position + 1] = levels[: position + 1]
        matrix[f"{series_id}_{release:%Y%m%d}"] = values
    matrix.index.name = "reference_month"
    return encode_vintage_matrix(matrix, series_id)


def _config(root: Path) -> Path:
    config = {
        "model_id": "m02_soft_composite",
        "retail_sensitivities": {
            "real_decomposition": {
                "series_id": "RRSFS",
                "release_id": 9,
                "provider": "fred_api",
                "observation_start": "2020-01-01",
                "observation_end": "2020-09-30",
                "vintage_start": "2020-01-01",
                "vintage_end": "2020-11-30",
                "archive_start_latest_only": True,
                "maximum_release_lag_days": 92,
                "minimum_standardization_history": 2,
            }
        },
        "outputs": {
            "retail_raw_dir": "data/raw/m02_retail_sensitivity",
            "retail_events": "data/processed/retail_events.csv",
            "retail_manifest": "data/manifests/retail_manifest.json",
        },
    }
    path = root / "configs" / "retail.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_build_retail_sensitivity_is_causal_and_secret_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sensitive-runtime-key-must-never-be-written"
    config_path = _config(tmp_path)
    payloads = {
        "RSAFS": _vintage_matrix("RSAFS", nominal=True),
        "RRSFS": _vintage_matrix("RRSFS", nominal=False),
    }

    def fake_select(
        provider: str,
        *,
        environ: dict[str, str] | None = None,
    ) -> ProviderSelection:
        assert provider == "fred"
        assert environ == {"FRED_API_KEY": secret}
        return ProviderSelection(provider, "fred_api", _FakeFred())

    def fake_download(
        *,
        series_id: str,
        raw_dir: Path,
        **_: object,
    ) -> MatrixAcquisition:
        content = payloads[series_id]
        path = raw_dir / f"{series_id}_synthetic.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return MatrixAcquisition(
            path=path,
            content=content,
            provider_id="fred_api",
            cache_origin="downloaded",
            source_url=f"https://fred.stlouisfed.org/series/{series_id}",
        )

    monkeypatch.setattr(build_m02_retail_sensitivity, "select_vintage_provider", fake_select)
    monkeypatch.setattr(
        build_m02_retail_sensitivity,
        "download_or_load_vintage_matrix",
        fake_download,
    )
    outputs = build_m02_retail_sensitivity.build_retail_sensitivity(
        project_root=tmp_path,
        config_path=config_path,
        environ={"FRED_API_KEY": secret},
    )

    events = pd.read_csv(outputs["events"], parse_dates=["release_date", "reference_month"])
    validate_event_table(events)
    assert set(events["feature_name"]) == {
        "real_retail_and_food_services_log_change",
        "implicit_retail_price_log_change",
    }
    for _, feature in events.groupby("feature_name"):
        ordered = feature.sort_values("release_date").reset_index(drop=True)
        assert ordered.loc[:1, "feature_status"].eq("standardization_warmup").all()
        third = ordered.iloc[2]
        prior = ordered.loc[:1, "transformed_value"].to_numpy(dtype=float)
        expected = (float(third["transformed_value"]) - prior.mean()) / prior.std(ddof=1)
        assert float(third["feature_value"]) == pytest.approx(expected)
        assert int(third["standardization_prior_count"]) == 2

    wide = events.pivot(
        index="reference_month",
        columns="feature_name",
        values="transformed_value",
    )
    expected_nominal_growth = 100.0 * np.array(
        [0.004, 0.006, -0.002, 0.010, 0.003, 0.007, 0.001, 0.009]
    ) + 100.0 * np.array(
        [0.001, 0.003, 0.002, -0.001, 0.004, 0.002, 0.005, 0.001]
    )
    reconstructed = (
        wide["real_retail_and_food_services_log_change"]
        + wide["implicit_retail_price_log_change"]
    ).to_numpy()
    assert reconstructed == pytest.approx(expected_nominal_growth)

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["provider_selected"] == "fred_api"
    assert manifest["common_vintage_count"] == 9
    assert manifest["aligned_information_date_count"] == 9
    assert manifest["common_reference_count"] == 9
    assert manifest["event_rows"] == 16
    assert manifest["first_release_extraction_diagnostics"][
        "real_retail_and_food_services"
    ]["rows_retained"] == 8
    assert manifest["generated_files"][0]["sha256"] == sha256(
        outputs["events"].read_bytes()
    )
    generated_bytes = b"".join(
        path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert secret.encode("utf-8") not in generated_bytes


def test_retail_config_rejects_a_keyless_provider(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["retail_sensitivities"]["real_decomposition"]["provider"] = "alfred_web"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="authenticated FRED API"):
        build_m02_retail_sensitivity.build_retail_sensitivity(
            project_root=tmp_path,
            config_path=config_path,
            environ={},
        )
