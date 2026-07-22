"""Construct point-in-time real-versus-price retail sensitivity features.

Model 02's frozen consumer block uses two nominal advance-retail coordinates.
This module supplies the provider-neutral transformation for a named
sensitivity: real retail-and-food-services growth plus the implicit price
change obtained from nominal RSAFS divided by real RRSFS. RSAFS and RRSFS do
not always publish on the same historical dates. At each information date the
code therefore uses the latest snapshot of each series available on or before
that date. Current and prior levels still come from one snapshot within each
series, so no future vintage can enter the cross-series decomposition.

Network acquisition and causal standardization remain CLI responsibilities.
The functions here only validate and combine in-memory vintage matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_allocation.data.providers.vintage_matrix import vintage_date_from_column


@dataclass(frozen=True)
class RetailDecompositionMatrices:
    """Causally aligned real levels and nominal-to-real implicit-price ratio."""

    real_levels: pd.DataFrame
    implicit_price_levels: pd.DataFrame
    aligned_information_date_count: int
    common_vintage_count: int
    common_reference_count: int


def _columns_by_vintage(frame: pd.DataFrame, *, label: str) -> dict[object, str]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label} vintage matrix must be a non-empty DataFrame")
    if frame.index.has_duplicates:
        raise ValueError(f"{label} vintage matrix repeats a reference date")
    result: dict[object, str] = {}
    for raw_column in frame.columns:
        column = str(raw_column)
        vintage = vintage_date_from_column(column)
        if vintage in result:
            raise ValueError(f"{label} vintage matrix repeats vintage {vintage}")
        result[vintage] = column
    return result


def build_real_retail_decomposition_matrices(
    nominal_levels: pd.DataFrame,
    real_levels: pd.DataFrame,
    *,
    real_series_id: str = "RRSFS",
    implicit_series_id: str = "RSAFS_DIV_RRSFS",
) -> RetailDecompositionMatrices:
    """Align RSAFS/RRSFS snapshots and form a causal price-ratio matrix.

    Reference rows must exist in both inputs. At every date on which either
    archive changes, each series contributes only its latest snapshot dated no
    later than that information date. A nonpositive or missing constituent
    makes that cell unavailable. The ratio's units are arbitrary, but its log
    change is the exact nominal-growth-minus-real-growth identity for the
    causally aligned snapshots.
    """

    nominal_columns = _columns_by_vintage(nominal_levels, label="nominal")
    real_columns = _columns_by_vintage(real_levels, label="real")
    exact_common_vintages = sorted(set(nominal_columns).intersection(real_columns))
    information_dates = sorted(set(nominal_columns).union(real_columns))
    if not information_dates:
        raise ValueError("nominal and real retail matrices have no vintage dates")
    common_index = nominal_levels.index.intersection(real_levels.index).sort_values()
    if common_index.empty:
        raise ValueError("nominal and real retail matrices have no common references")

    # Build the wide outputs in one allocation.  A real archive contains
    # hundreds of publication vintages, so repeated column insertion would
    # fragment the DataFrame and make this otherwise mechanical alignment
    # needlessly slow and noisy.
    real_values: dict[str, pd.Series] = {}
    implicit_values: dict[str, pd.Series] = {}
    latest_nominal: str | None = None
    latest_real: str | None = None
    for vintage in information_dates:
        if vintage in nominal_columns:
            latest_nominal = nominal_columns[vintage]
        if vintage in real_columns:
            latest_real = real_columns[vintage]
        if latest_nominal is None or latest_real is None:
            continue
        nominal = pd.to_numeric(
            nominal_levels.loc[common_index, latest_nominal], errors="coerce"
        ).astype(float)
        real = pd.to_numeric(
            real_levels.loc[common_index, latest_real], errors="coerce"
        ).astype(float)
        valid = nominal.gt(0.0) & real.gt(0.0)
        suffix = pd.Timestamp(vintage).strftime("%Y%m%d")
        real_values[f"{real_series_id}_{suffix}"] = real.where(valid)
        implicit_values[f"{implicit_series_id}_{suffix}"] = nominal.div(real).where(
            valid
        )

    real_output = pd.DataFrame(real_values, index=common_index)
    implicit_output = pd.DataFrame(implicit_values, index=common_index)

    if not np.isfinite(real_output.to_numpy(dtype=float)[~real_output.isna()]).all():
        raise ValueError("real retail matrix contains a non-finite available value")
    if not np.isfinite(
        implicit_output.to_numpy(dtype=float)[~implicit_output.isna()]
    ).all():
        raise ValueError("implicit retail-price matrix contains a non-finite value")
    real_output.index.name = "reference_month"
    implicit_output.index.name = "reference_month"
    return RetailDecompositionMatrices(
        real_levels=real_output,
        implicit_price_levels=implicit_output,
        aligned_information_date_count=len(real_output.columns),
        common_vintage_count=len(exact_common_vintages),
        common_reference_count=len(common_index),
    )


__all__ = [
    "RetailDecompositionMatrices",
    "build_real_retail_decomposition_matrices",
]
