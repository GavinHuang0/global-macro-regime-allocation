"""Test exact-vintage Model 02 score-scale revision measurements."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.revisions import (
    aggregate_axis_revision_errors,
    component_revision_errors,
)
from regime_allocation.models.m02_soft_composite.scores import ALL_COMPONENTS


def test_component_revision_uses_exact_cutoff_and_frozen_scale() -> None:
    month = pd.Timestamp("2020-01-01")
    release = pd.Timestamp("2020-02-10")
    maturity = pd.Timestamp("2020-05-10")
    first_release = pd.DataFrame(
        {
            "reference_month": [month],
            "component": ["payrolls"],
            "series_id": ["PAYEMS"],
            "release_date": [release],
            "transform": ["log_difference"],
            "transformed_value": [100.0 * math.log(101.0 / 100.0)],
        }
    )
    scales = pd.DataFrame({"payrolls": [0.5]}, index=[month])
    matrix = pd.DataFrame(
        {"PAYEMS_20200510": [100.0, 102.0]},
        index=[pd.Timestamp("2019-12-01"), month],
    )

    errors = component_revision_errors(
        first_release,
        scales=scales,
        exact_vintage_matrices={"PAYEMS": matrix},
        reference_months=[month],
        horizons=(3,),
        knowledge_cutoff=maturity,
    )
    row = errors.iloc[0]
    expected_revised = 100.0 * math.log(102.0 / 100.0)

    assert row["revision_status"] == "available"
    assert row["revision_vintage_date"] == maturity
    assert row["revised_transformed_value"] == pytest.approx(expected_revised)
    assert row["standardized_revision_error"] == pytest.approx(
        (expected_revised - first_release.iloc[0]["transformed_value"]) / 0.5
    )


def test_immature_revision_never_uses_an_available_future_matrix_column() -> None:
    month = pd.Timestamp("2020-01-01")
    first_release = pd.DataFrame(
        {
            "reference_month": [month],
            "component": ["payrolls"],
            "series_id": ["PAYEMS"],
            "release_date": [pd.Timestamp("2020-02-10")],
            "transform": ["log_difference"],
            "transformed_value": [1.0],
        }
    )
    scales = pd.DataFrame({"payrolls": [1.0]}, index=[month])
    matrix = pd.DataFrame(
        {"PAYEMS_20210210": [100.0, 101.0]},
        index=[pd.Timestamp("2019-12-01"), month],
    )

    errors = component_revision_errors(
        first_release,
        scales=scales,
        exact_vintage_matrices={"PAYEMS": matrix},
        reference_months=[month],
        horizons=(12,),
        knowledge_cutoff=pd.Timestamp("2020-12-31"),
    )

    assert errors.iloc[0]["revision_status"] == "horizon_not_mature"
    assert pd.isna(errors.iloc[0]["standardized_revision_error"])


def test_axis_revision_requires_all_components_and_uses_fixed_quarters() -> None:
    month = pd.Timestamp("2020-01-01")
    rows = []
    for position, component in enumerate(ALL_COMPONENTS, start=1):
        rows.append(
            {
                "reference_month": month,
                "component": component,
                "horizon_months": 12,
                "revision_available_at": pd.Timestamp("2021-03-01"),
                "standardized_revision_error": float(position),
                "revision_status": "available",
            }
        )
    component_errors = pd.DataFrame(rows)

    aggregated = aggregate_axis_revision_errors(
        component_errors,
        reference_months=[month],
        horizons=(12,),
    ).iloc[0]

    assert aggregated["revision_status"] == "available"
    assert aggregated["growth_revision_error"] == pytest.approx(2.5)
    assert aggregated["inflation_revision_error"] == pytest.approx(6.5)

