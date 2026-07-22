"""Unit tests for Model 02's common-vintage retail decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.features.m02_retail_sensitivities import (
    build_real_retail_decomposition_matrices,
)


def _matrix(series: str, values: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=pd.to_datetime(["2020-01-01", "2020-02-01"]),
        columns=[f"{series}_20200315", f"{series}_20200415"],
    )


def test_common_vintage_decomposition_preserves_log_change_identity() -> None:
    nominal = _matrix("RSAFS", [[100.0, 101.0], [110.0, 121.2]])
    real = _matrix("RRSFS", [[50.0, 50.5], [52.5, 55.55]])

    result = build_real_retail_decomposition_matrices(nominal, real)

    for suffix in ("20200315", "20200415"):
        nominal_change = np.log(
            nominal.loc[pd.Timestamp("2020-02-01"), f"RSAFS_{suffix}"]
            / nominal.loc[pd.Timestamp("2020-01-01"), f"RSAFS_{suffix}"]
        )
        real_change = np.log(
            result.real_levels.loc[
                pd.Timestamp("2020-02-01"), f"RRSFS_{suffix}"
            ]
            / result.real_levels.loc[
                pd.Timestamp("2020-01-01"), f"RRSFS_{suffix}"
            ]
        )
        price_change = np.log(
            result.implicit_price_levels.loc[
                pd.Timestamp("2020-02-01"), f"RSAFS_DIV_RRSFS_{suffix}"
            ]
            / result.implicit_price_levels.loc[
                pd.Timestamp("2020-01-01"), f"RSAFS_DIV_RRSFS_{suffix}"
            ]
        )
        assert nominal_change == pytest.approx(real_change + price_change)


def test_nonpositive_pair_is_missing_in_both_outputs() -> None:
    nominal = _matrix("RSAFS", [[100.0, 100.0], [0.0, 110.0]])
    real = _matrix("RRSFS", [[50.0, 50.0], [52.0, 52.0]])

    result = build_real_retail_decomposition_matrices(nominal, real)

    assert pd.isna(result.real_levels.loc["2020-02-01", "RRSFS_20200315"])
    assert pd.isna(
        result.implicit_price_levels.loc[
            "2020-02-01", "RSAFS_DIV_RRSFS_20200315"
        ]
    )


def test_disjoint_vintage_dates_are_aligned_causally() -> None:
    nominal = _matrix("RSAFS", [[100.0, 100.0], [110.0, 110.0]])
    real = _matrix("RRSFS", [[50.0, 50.0], [52.0, 52.0]])
    real.columns = ["RRSFS_20200515", "RRSFS_20200615"]

    result = build_real_retail_decomposition_matrices(nominal, real)

    assert result.common_vintage_count == 0
    assert result.aligned_information_date_count == 2


def test_asof_alignment_uses_no_future_cross_series_vintage() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-02-01"])
    nominal = pd.DataFrame(
        {
            "RSAFS_20200315": [100.0, np.nan],
            "RSAFS_20200415": [110.0, 121.0],
        },
        index=index,
    )
    real = pd.DataFrame(
        {
            "RRSFS_20200320": [50.0, np.nan],
            "RRSFS_20200420": [55.0, 58.0],
        },
        index=index,
    )

    result = build_real_retail_decomposition_matrices(nominal, real)

    # On March 20 only the March 15 nominal snapshot was public; the revised
    # January nominal value from April 15 must not leak backward.
    assert result.implicit_price_levels.loc[
        pd.Timestamp("2020-01-01"), "RSAFS_DIV_RRSFS_20200320"
    ] == pytest.approx(2.0)
    assert pd.isna(
        result.implicit_price_levels.loc[
            pd.Timestamp("2020-02-01"), "RSAFS_DIV_RRSFS_20200415"
        ]
    )
    assert result.implicit_price_levels.loc[
        pd.Timestamp("2020-02-01"), "RSAFS_DIV_RRSFS_20200420"
    ] == pytest.approx(121.0 / 58.0)
