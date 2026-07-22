"""Test causal Model 02 regime, signal, and weekly execution input adapters."""

from __future__ import annotations

import pandas as pd
import pytest

from regime_allocation.portfolio.m02_weekly import (
    CANONICAL_REGIME_IDS,
    PROBABILITY_COLUMNS,
    build_weekly_execution_schedule,
    derive_m02_regime_history,
    select_m02_weekly_signals,
)


def test_regime_history_uses_released_exact_scores_and_zero_is_up() -> None:
    scores = pd.DataFrame(
        {
            "reference_month": pd.date_range("2019-12-01", periods=5, freq="MS"),
            "growth_score": [0.5, 0.0, -0.2, 0.3, -0.4],
            "inflation_score": [None, 0.0, 0.1, -0.1, -0.2],
            "score_available_at": [
                None,
                "2020-02-15",
                "2020-03-15",
                "2020-04-15",
                "2020-05-15",
            ],
        }
    )

    history = derive_m02_regime_history(scores)

    assert history.columns.tolist() == [
        "reference_month",
        "regime_id",
        "label_available_at",
    ]
    assert history["regime_id"].tolist() == list(CANONICAL_REGIME_IDS)
    assert history["reference_month"].tolist() == list(
        pd.date_range("2020-01-01", periods=4, freq="MS")
    )
    assert history.loc[0, "label_available_at"] == pd.Timestamp("2020-02-15")


def test_regime_history_rejects_duplicate_or_non_month_start_references() -> None:
    valid = pd.DataFrame(
        {
            "reference_month": ["2020-01-01", "2020-02-01"],
            "growth_score": [0.1, -0.1],
            "inflation_score": [0.2, -0.2],
            "score_available_at": ["2020-02-10", "2020-03-10"],
        }
    )
    duplicate = valid.copy()
    duplicate.loc[1, "reference_month"] = "2020-01-01"
    with pytest.raises(ValueError, match="duplicate reference months"):
        derive_m02_regime_history(duplicate)

    non_month_start = valid.copy()
    non_month_start.loc[1, "reference_month"] = "2020-02-02"
    with pytest.raises(ValueError, match="normalized month starts"):
        derive_m02_regime_history(non_month_start)


def _probabilities(values: tuple[float, float, float, float]) -> dict[str, float]:
    return dict(zip(PROBABILITY_COLUMNS, values, strict=True))


def _decision_rows(*, explicit_weeks: bool = True) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    decisions = (
        (
            "2020-01-03",
            "2020-01-06",
            (("2019-11-01", (0.1, 0.2, 0.3, 0.4)), ("2019-12-01", (0.4, 0.3, 0.2, 0.1))),
        ),
        (
            "2020-01-10",
            "2020-01-13",
            (("2019-12-01", (0.2, 0.3, 0.1, 0.4)), ("2020-01-01", (0.6, 0.2, 0.1, 0.1))),
        ),
    )
    for signal_date, reference_week, marginals in decisions:
        for reference_month, probabilities in marginals:
            row: dict[str, object] = {
                "variant_id": "student_t_7_reduced_core",
                "signal_date": signal_date,
                "regime_reference_month": reference_month,
                **_probabilities(probabilities),
            }
            if explicit_weeks:
                row["reference_week"] = reference_week
            records.append(row)
    records.append(
        {
            "variant_id": "transition_only",
            "signal_date": "2020-01-03",
            "regime_reference_month": "2019-12-01",
            **_probabilities((0.25, 0.25, 0.25, 0.25)),
            **({"reference_week": "2020-01-06"} if explicit_weeks else {}),
        }
    )
    return pd.DataFrame.from_records(records)


def test_weekly_signals_select_baseline_and_newest_current_marginal() -> None:
    signals = select_m02_weekly_signals(
        _decision_rows(),
        ["2020-01-06", "2020-01-13"],
    )

    assert signals["signal_date"].tolist() == [
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-10"),
    ]
    assert signals["regime_reference_month"].tolist() == [
        pd.Timestamp("2019-12-01"),
        pd.Timestamp("2020-01-01"),
    ]
    assert signals.loc[0, list(PROBABILITY_COLUMNS)].tolist() == pytest.approx(
        [0.4, 0.3, 0.2, 0.1]
    )
    assert signals.loc[1, list(PROBABILITY_COLUMNS)].tolist() == pytest.approx(
        [0.6, 0.2, 0.1, 0.1]
    )


def test_weekly_signals_can_asof_select_without_explicit_week_column() -> None:
    signals = select_m02_weekly_signals(
        _decision_rows(explicit_weeks=False),
        ["2020-01-06", "2020-01-13", "2020-01-20"],
    )

    assert signals["signal_date"].tolist() == [
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-10"),
        pd.Timestamp("2020-01-10"),
    ]


def test_weekly_signals_reject_duplicate_or_unnormalized_probabilities() -> None:
    decisions = _decision_rows()
    duplicate = pd.concat([decisions, decisions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate weekly"):
        select_m02_weekly_signals(duplicate, ["2020-01-06", "2020-01-13"])

    invalid = decisions.copy()
    invalid.loc[0, PROBABILITY_COLUMNS[0]] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        select_m02_weekly_signals(invalid, ["2020-01-06", "2020-01-13"])


def test_weekly_schedule_uses_first_common_session_and_retains_live_week() -> None:
    rows: list[dict[str, object]] = []
    for date in ("2020-01-06", "2020-01-14", "2020-01-20"):
        for ticker in ("A", "B"):
            rows.append({"date": date, "ticker": ticker})
    # This date is not common and therefore cannot become an execution date.
    rows.append({"date": "2020-01-07", "ticker": "A"})

    schedule = build_weekly_execution_schedule(
        pd.DataFrame(rows),
        assets=("A", "B"),
    )

    assert schedule["reference_week"].tolist() == list(
        pd.to_datetime(["2020-01-06", "2020-01-13", "2020-01-20"])
    )
    assert schedule["start_date"].tolist() == list(
        pd.to_datetime(["2020-01-06", "2020-01-14", "2020-01-20"])
    )
    assert schedule.loc[0, "end_date"] == pd.Timestamp("2020-01-14")
    assert schedule.loc[1, "end_date"] == pd.Timestamp("2020-01-20")
    assert pd.isna(schedule.loc[2, "end_date"])
    assert schedule["is_complete"].tolist() == [True, True, False]


def test_weekly_schedule_rejects_an_internal_missing_week() -> None:
    prices = pd.DataFrame(
        [
            {"date": date, "ticker": ticker}
            for date in ("2020-01-06", "2020-01-20")
            for ticker in ("A", "B")
        ]
    )

    with pytest.raises(ValueError, match="skips the week"):
        build_weekly_execution_schedule(prices, assets=("A", "B"))
