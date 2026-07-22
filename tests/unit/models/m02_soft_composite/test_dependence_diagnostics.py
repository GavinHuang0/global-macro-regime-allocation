"""Test Model 02 causal release-block residual dependence diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from regime_allocation.models.m02_soft_composite.dependence_diagnostics import (
    INDEPENDENCE_CAVEAT,
    ResidualColumnSpec,
    aggregate_residuals_to_monthly,
    benjamini_hochberg,
    build_release_block_dependence_report,
    cross_block_correlation_tests,
    normalize_oos_residuals,
    serial_ljung_box_tests,
)


def _row(
    *,
    block: str,
    model: str,
    response: str,
    month: str,
    available: str,
    residual: float,
) -> dict[str, object]:
    return {
        "block_id": block,
        "model_id": model,
        "response_id": response,
        "reference_month": month,
        "validation_available_at": available,
        "standardized_residual": residual,
    }


def test_weekly_claims_are_aggregated_only_for_monthly_panel() -> None:
    rows = [
        _row(
            block="weekly_labor_stress",
            model="icsa_innovation",
            response="ICSA",
            month="2020-01",
            available=f"2020-01-{day:02d}",
            residual=value,
        )
        for day, value in zip((2, 9, 16, 23), (1.0, 2.0, 3.0, 4.0))
    ]
    rows += [
        _row(
            block="weekly_labor_stress",
            model="icsa_innovation",
            response="ICSA",
            month="2020-02",
            available=f"2020-02-{day:02d}",
            residual=value,
        )
        for day, value in zip((6, 13), (-2.0, 2.0))
    ]
    rows.append(
        _row(
            block="consumer_demand",
            model="retail",
            response="ex_auto",
            month="2020-01",
            available="2020-02-14",
            residual=0.25,
        )
    )
    source = pd.DataFrame(rows)
    source_copy = source.copy(deep=True)

    monthly = aggregate_residuals_to_monthly(source)

    pd.testing.assert_frame_equal(source, source_copy)
    claims = monthly.loc[monthly["response_id"] == "ICSA"].reset_index(drop=True)
    assert len(claims) == 2
    assert claims.loc[0, "standardized_residual"] == pytest.approx(2.5)
    assert claims.loc[0, "source_observation_count"] == 4
    assert claims.loc[0, "source_frequency"] == "weekly_aggregated"
    assert claims.loc[0, "validation_available_at"] == pd.Timestamp("2020-01-23")
    retail = monthly.loc[monthly["response_id"] == "ex_auto"].iloc[0]
    assert retail["source_observation_count"] == 1
    assert retail["source_frequency"] == "monthly"


def test_same_publication_weekly_catchup_rows_are_retained_without_observation_date() -> None:
    observation_dates = pd.date_range("2020-01-02", periods=24, freq="7D")
    rows = []
    for position, observation_date in enumerate(observation_dates):
        publication_date = (
            pd.Timestamp("2020-01-30")
            if position < 4
            else observation_date + pd.Timedelta(days=7)
        )
        rows.append(
            _row(
                block="weekly_labor_stress",
                model="icsa_innovation",
                response="ICSA",
                month=observation_date.strftime("%Y-%m"),
                available=publication_date.strftime("%Y-%m-%d"),
                residual=float(position + 1),
            )
        )
    source = pd.DataFrame(rows)

    normalized = normalize_oos_residuals(source)
    monthly = aggregate_residuals_to_monthly(source)
    serial = serial_ljung_box_tests(source)

    assert len(normalized) == len(source)
    assert int(monthly["source_observation_count"].sum()) == len(source)
    assert serial["sample_count"].eq(len(source)).all()
    assert serial["same_publication_date_batch_count"].eq(1).all()
    assert serial["maximum_same_publication_date_batch_size"].eq(4).all()
    assert serial["lag_semantics"].eq(
        "ordered_weekly_observations_same_publication_batches_"
        "calendar_spacing_not_asserted"
    ).all()
    assert serial["within_publication_date_order"].eq(
        "reference_month_then_stable_source_order"
    ).all()


def test_observation_date_restores_weekly_order_inside_catchup_batch() -> None:
    rng = np.random.default_rng(2026)
    observation_dates = pd.date_range("2018-01-04", periods=48, freq="7D")
    residuals = np.empty(len(observation_dates))
    residuals[0] = rng.normal()
    for position in range(1, len(residuals)):
        residuals[position] = 0.8 * residuals[position - 1] + rng.normal(scale=0.4)

    rows = []
    for observation_date, residual in zip(observation_dates, residuals):
        row = _row(
            block="weekly_labor_stress",
            model="icsa_innovation",
            response="ICSA",
            month=observation_date.strftime("%Y-%m"),
            available="2019-01-01",
            residual=float(residual),
        )
        row["observation_date"] = observation_date.strftime("%Y-%m-%d")
        rows.append(row)
    shuffled = pd.DataFrame(rows).sample(frac=1.0, random_state=9).reset_index(drop=True)
    columns = ResidualColumnSpec(observation_date="observation_date")

    normalized = normalize_oos_residuals(shuffled, columns=columns)
    serial = serial_ljung_box_tests(shuffled, columns=columns)

    assert normalized["observation_date"].is_monotonic_increasing
    assert serial["sample_count"].eq(len(shuffled)).all()
    assert serial["same_publication_date_batch_count"].eq(1).all()
    assert serial["maximum_same_publication_date_batch_size"].eq(len(shuffled)).all()
    assert serial["lag_semantics"].eq("calendar_weeks_by_observation_date").all()
    assert serial["calendar_gap_count"].eq(0).all()
    assert serial["calendar_spacing_status"].eq("calendar_week_contiguous").all()
    assert serial["within_publication_date_order"].eq(
        "observation_date_then_stable_source_order_for_ties"
    ).all()
    assert serial.loc[serial["lag"] == 8, "p_value"].iloc[0] < 1.0e-5


def test_nonweekly_duplicate_reference_month_is_rejected() -> None:
    table = pd.DataFrame(
        [
            _row(
                block="housing",
                model="housing_activity",
                response="starts",
                month="2020-01",
                available="2020-02-15",
                residual=0.1,
            ),
            _row(
                block="housing",
                model="housing_activity",
                response="starts",
                month="2020-01",
                available="2020-03-15",
                residual=0.2,
            ),
        ]
    )
    with pytest.raises(ValueError, match="duplicate reference-month"):
        aggregate_residuals_to_monthly(table)
    with pytest.raises(ValueError, match="duplicate reference-month"):
        serial_ljung_box_tests(table)


def test_benjamini_hochberg_preserves_order_and_missing_values() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, np.nan, 0.03])
    np.testing.assert_allclose(
        adjusted[[0, 1, 3]], np.asarray([0.03, 0.04, 0.04])
    )
    assert np.isnan(adjusted[2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg([0.2, 1.1])


def test_cross_block_lag_definition_and_global_bh_family() -> None:
    rng = np.random.default_rng(42)
    months = pd.date_range("2015-01-01", periods=60, freq="MS")
    left = rng.normal(size=len(months))
    right = np.r_[rng.normal(), left[:-1] + rng.normal(scale=0.02, size=59)]
    rows: list[dict[str, object]] = []
    for month, value in zip(months, left):
        rows.append(
            _row(
                block="labor",
                model="labor_demand",
                response="openings",
                month=month.strftime("%Y-%m"),
                available=(month + pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d"),
                residual=float(value),
            )
        )
        # A response estimated jointly in the same observation model must not
        # form a separate pair because its covariance is already modeled.
        rows.append(
            _row(
                block="labor",
                model="labor_demand",
                response="hires",
                month=month.strftime("%Y-%m"),
                available=(month + pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d"),
                residual=float(-value),
            )
        )
        # A distinct asynchronous model in the same economic block must still
        # be tested because it does not share the first model's covariance.
        rows.append(
            _row(
                block="labor",
                model="labor_secondary",
                response="vacancies_alt",
                month=month.strftime("%Y-%m"),
                available=(month + pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d"),
                residual=float(0.5 * value + rng.normal(scale=0.01)),
            )
        )
    for month, value in zip(months, right):
        rows.append(
            _row(
                block="consumer",
                model="retail",
                response="ex_auto",
                month=month.strftime("%Y-%m"),
                available=(month + pd.offsets.MonthBegin(2) + pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
                residual=float(value),
            )
        )
    monthly = aggregate_residuals_to_monthly(pd.DataFrame(rows))

    diagnostics = cross_block_correlation_tests(
        monthly, lags=(-1, 0, 1), minimum_pair_observations=20
    )

    assert not diagnostics.empty
    same_joint_model = (
        diagnostics["left_block_id"].eq(diagnostics["right_block_id"])
        & diagnostics["left_model_id"].eq(diagnostics["right_model_id"])
    )
    assert not same_joint_model.any()
    same_block_different_model = diagnostics["same_economic_block"]
    assert same_block_different_model.any()
    assert (
        diagnostics.loc[same_block_different_model, "left_model_id"]
        != diagnostics.loc[same_block_different_model, "right_model_id"]
    ).all()
    target = diagnostics.loc[
        (diagnostics["left_response_id"] == "ex_auto")
        & (diagnostics["right_response_id"] == "openings")
        & (diagnostics["lag_months"] == -1)
        & (diagnostics["statistic"] == "pearson")
    ].iloc[0]
    assert target["sample_count"] == 59
    assert target["correlation"] > 0.99
    assert target["p_value"] < 1.0e-40
    assert target["q_value"] >= target["p_value"]
    assert diagnostics["q_value"].dropna().between(0.0, 1.0).all()
    assert diagnostics["bh_family"].nunique() == 1
    assert (
        diagnostics["lag_definition"].iloc[0]
        == "right_reference_month = left_reference_month + lag_months"
    )


def test_ljung_box_uses_weekly_claims_before_monthly_aggregation() -> None:
    rng = np.random.default_rng(7)
    weekly_count = 96
    weekly_values = np.empty(weekly_count)
    weekly_values[0] = rng.normal()
    for index in range(1, weekly_count):
        weekly_values[index] = 0.82 * weekly_values[index - 1] + rng.normal(scale=0.5)
    weekly_dates = pd.date_range("2018-01-04", periods=weekly_count, freq="7D")

    monthly_count = 72
    monthly_values = np.empty(monthly_count)
    monthly_values[0] = rng.normal()
    for index in range(1, monthly_count):
        monthly_values[index] = 0.75 * monthly_values[index - 1] + rng.normal(scale=0.6)
    monthly_dates = pd.date_range("2015-01-01", periods=monthly_count, freq="MS")

    rows = [
        _row(
            block="weekly_labor_stress",
            model="icsa_innovation",
            response="ICSA",
            month=date.strftime("%Y-%m"),
            available=date.strftime("%Y-%m-%d"),
            residual=float(value),
        )
        for date, value in zip(weekly_dates, weekly_values)
    ]
    rows.extend(
        _row(
            block="housing_activity",
            model="housing",
            response="starts",
            month=date.strftime("%Y-%m"),
            available=(date + pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d"),
            residual=float(value),
        )
        for date, value in zip(monthly_dates, monthly_values)
    )

    tests = serial_ljung_box_tests(pd.DataFrame(rows))

    claims = tests.loc[tests["response_id"] == "ICSA"]
    assert claims["lag"].tolist() == [1, 4, 8]
    assert claims["frequency"].eq("weekly").all()
    assert claims["sample_count"].eq(weekly_count).all()
    assert claims.loc[claims["lag"] == 8, "p_value"].iloc[0] < 1.0e-6
    assert claims.loc[claims["lag"] == 8, "q_value"].iloc[0] < 1.0e-6
    assert claims["bh_family"].eq("all_valid_serial_series_and_lags").all()
    assert claims["serial_dependence_detected_bh_5pct"].any()
    assert claims["lag_semantics"].eq(
        "ordered_weekly_observations_calendar_spacing_not_asserted"
    ).all()
    monthly = tests.loc[tests["response_id"] == "starts"]
    assert monthly["lag"].tolist() == [1, 3, 6]
    assert monthly["frequency"].eq("monthly").all()
    assert monthly["sample_count"].eq(monthly_count).all()
    assert monthly.loc[monthly["lag"] == 6, "p_value"].iloc[0] < 1.0e-6
    assert monthly["calendar_spacing_status"].eq("calendar_contiguous").all()
    assert monthly["lag_semantics"].eq("calendar_months").all()


def test_monthly_serial_lags_disclose_missing_calendar_periods() -> None:
    months = list(pd.date_range("2018-01-01", periods=30, freq="MS"))
    months.pop(12)
    rows = [
        _row(
            block="housing",
            model="activity",
            response="starts",
            month=month.strftime("%Y-%m"),
            available=(month + pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d"),
            residual=float(np.sin(position)),
        )
        for position, month in enumerate(months)
    ]

    tests = serial_ljung_box_tests(pd.DataFrame(rows))

    assert tests["calendar_gap_count"].eq(1).all()
    assert tests["calendar_spacing_status"].eq("missing_reference_months").all()
    assert tests["lag_semantics"].eq(
        "ordered_monthly_observations_with_calendar_gaps"
    ).all()


def test_short_series_is_reported_not_silently_dropped() -> None:
    rows = [
        _row(
            block="inflation_pressure",
            model="expectations",
            response="EXPINF1YR",
            month=f"2024-{month:02d}",
            available=f"2024-{month:02d}-15",
            residual=float(month),
        )
        for month in range(1, 9)
    ]
    tests = serial_ljung_box_tests(pd.DataFrame(rows))
    assert len(tests) == 3
    assert tests["status"].eq("insufficient_observations").all()
    assert tests["p_value"].isna().all()
    assert tests["q_value"].isna().all()


def test_report_records_non_independence_caveat_and_is_json_safe() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for index, month in enumerate(pd.date_range("2018-01-01", periods=24, freq="MS")):
        for block, model, response in (
            ("consumer", "retail", "ex_auto"),
            ("housing", "activity", "starts"),
        ):
            rows.append(
                _row(
                    block=block,
                    model=model,
                    response=response,
                    month=month.strftime("%Y-%m"),
                    available=(month + pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d"),
                    residual=float(rng.normal() + 0.01 * index),
                )
            )
    report = build_release_block_dependence_report(
        pd.DataFrame(rows), minimum_pair_observations=12
    )

    assert report.metadata["interpretation_caveat"] == INDEPENDENCE_CAVEAT
    assert "does not establish conditional independence" in INDEPENDENCE_CAVEAT
    assert "independent" not in report.cross_block_tests.columns
    assert report.cross_block_tests["correlation_p_q_interpretation"].str.startswith(
        "exploratory_"
    ).all()
    assert report.metadata["correlation_p_q_values"] == "exploratory"
    json.dumps(report.to_audit_dict())


def test_report_labels_correlations_when_serial_dependence_is_detected() -> None:
    rng = np.random.default_rng(31)
    months = pd.date_range("2010-01-01", periods=96, freq="MS")
    autoregressive = np.empty(len(months))
    autoregressive[0] = rng.normal()
    for position in range(1, len(months)):
        autoregressive[position] = (
            0.9 * autoregressive[position - 1] + rng.normal(scale=0.25)
        )
    rows = []
    for month, left, right in zip(months, autoregressive, rng.normal(size=len(months))):
        rows.append(
            _row(
                block="consumer",
                model="retail",
                response="ex_auto",
                month=month.strftime("%Y-%m"),
                available=(month + pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d"),
                residual=float(left),
            )
        )
        rows.append(
            _row(
                block="housing",
                model="activity",
                response="starts",
                month=month.strftime("%Y-%m"),
                available=(month + pd.offsets.MonthBegin(2)).strftime("%Y-%m-%d"),
                residual=float(right),
            )
        )

    report = build_release_block_dependence_report(pd.DataFrame(rows))

    assert report.serial_tests["serial_dependence_detected_bh_5pct"].any()
    assert report.cross_block_tests["serial_dependence_detected_either"].all()
    assert report.cross_block_tests["correlation_p_q_interpretation"].eq(
        "exploratory_serial_dependence_detected"
    ).all()


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("reference_month", "not-a-month", "reference_month"),
        ("validation_available_at", "not-a-date", "validation_available_at"),
        ("standardized_residual", np.inf, "finite numeric"),
    ],
)
def test_invalid_residual_rows_fail_closed(
    column: str, bad_value: object, message: str
) -> None:
    row = _row(
        block="consumer",
        model="retail",
        response="ex_auto",
        month="2020-01",
        available="2020-02-14",
        residual=0.1,
    )
    row[column] = bad_value
    with pytest.raises(ValueError, match=message):
        normalize_oos_residuals(pd.DataFrame([row]))
