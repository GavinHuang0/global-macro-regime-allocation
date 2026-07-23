"""Causal Model 02 inputs for weekly portfolio decisions.

This module deliberately stops at input alignment.  It converts released Model
02 composite scores into monthly hard labels, selects one causal current-regime
marginal for each requested calendar week, and derives first-common-session
weekly execution dates from a long ETF price panel.  The weekly return
estimator associates each Monday-anchored holding with the hard label for the
calendar month containing that Monday.  Portfolio estimation, optimization,
and accounting live elsewhere.

Calendar weeks are identified by their Monday date.  A weekly signal may have
been recorded before that Monday, but never after it.  Execution occurs on the
first common adjusted open in the week and a completed holding exits at the
first common adjusted open in the immediately following week.  The allocation
stage omits the leading, partial 2007-12-31 week in the available ETF history.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from regime_allocation.models.m02_soft_composite.probability_map import REGIME_ORDER


CANONICAL_REGIME_IDS = tuple(REGIME_ORDER)
PROBABILITY_COLUMNS = tuple(
    f"probability_{regime_id}" for regime_id in CANONICAL_REGIME_IDS
)
DEFAULT_BASELINE_VARIANT_ID = "student_t_7_reduced_core"

_PROBABILITY_TOLERANCE = 1.0e-10
_VARIANT_COLUMNS = ("variant_id", "filter_variant", "specification_id")

__all__ = [
    "CANONICAL_REGIME_IDS",
    "DEFAULT_BASELINE_VARIANT_ID",
    "PROBABILITY_COLUMNS",
    "build_weekly_execution_schedule",
    "derive_m02_regime_history",
    "select_m02_weekly_signals",
]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")


def _parse_dates(
    values: pd.Series,
    *,
    name: str,
    allow_missing: bool = False,
) -> pd.Series:
    """Return normalized timezone-naive dates while retaining missing values."""

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    invalid = values.notna() & parsed.isna()
    if invalid.any() or (not allow_missing and parsed.isna().any()):
        raise ValueError(f"{name} contains an invalid or missing date")
    return parsed.dt.tz_convert(None).dt.normalize()


def _normalize_months(values: pd.Series, *, name: str) -> pd.Series:
    converted = _parse_dates(values, name=name)
    if (converted.dt.day != 1).any():
        raise ValueError(f"{name} must contain normalized month starts")
    return converted


def _normalize_reference_weeks(values: Sequence[object]) -> pd.DatetimeIndex:
    if isinstance(values, (str, bytes)):
        raw = pd.Series([values], dtype="object")
    else:
        raw = pd.Series(list(values), dtype="object")
    if raw.empty:
        raise ValueError("reference_weeks cannot be empty")
    converted = _parse_dates(raw, name="reference_weeks")
    if converted.dt.weekday.ne(0).any():
        raise ValueError("reference_weeks must contain calendar Monday anchors")
    if converted.duplicated().any():
        raise ValueError("reference_weeks must be unique")
    return pd.DatetimeIndex(converted.sort_values().tolist(), name="reference_week")


def derive_m02_regime_history(composite_scores: pd.DataFrame) -> pd.DataFrame:
    """Return causal monthly hard regimes from released exact Model 02 scores.

    Incomplete warm-up rows are omitted.  A zero score belongs to the ``up``
    side of its axis, matching Model 02's hard-quadrant evaluation helper.
    The weekly estimator assigns each holding the label for the calendar month
    containing its Monday reference week, and separately enforces that both
    the completed return and this label were available by its fit cutoff.
    """

    required = (
        "reference_month",
        "growth_score",
        "inflation_score",
        "score_available_at",
    )
    _require_columns(composite_scores, required, name="Model 02 composite scores")
    scores = composite_scores.loc[:, list(required)].copy()
    scores["reference_month"] = _normalize_months(
        scores["reference_month"], name="score reference_month"
    )
    if scores["reference_month"].duplicated().any():
        raise ValueError("Model 02 composite scores contain duplicate reference months")

    availability = _parse_dates(
        scores["score_available_at"],
        name="score_available_at",
        allow_missing=True,
    )
    for column in ("growth_score", "inflation_score"):
        original = scores[column]
        converted = pd.to_numeric(original, errors="coerce")
        if (original.notna() & converted.isna()).any():
            raise ValueError(f"{column} contains a non-numeric value")
        if np.isinf(converted.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{column} must be finite when supplied")
        scores[column] = converted
    scores["score_available_at"] = availability

    score_values = scores.loc[:, ["growth_score", "inflation_score"]].to_numpy(
        dtype=float
    )
    complete_scores = np.isfinite(score_values).all(axis=1)
    released = scores["score_available_at"].notna().to_numpy()
    if np.any(complete_scores & ~released):
        raise ValueError("complete Model 02 scores require score_available_at")
    if np.any(released & ~complete_scores):
        raise ValueError("score_available_at cannot accompany an incomplete score pair")

    complete = scores.loc[complete_scores & released].copy()
    if complete.empty:
        raise ValueError("Model 02 composite scores contain no complete released score pairs")
    if (complete["score_available_at"] < complete["reference_month"]).any():
        raise ValueError("score_available_at cannot precede its reference month")

    growth_up = complete["growth_score"].ge(0.0)
    inflation_up = complete["inflation_score"].ge(0.0)
    complete["regime_id"] = np.select(
        (
            growth_up & inflation_up,
            ~growth_up & inflation_up,
            growth_up & ~inflation_up,
        ),
        CANONICAL_REGIME_IDS[:3],
        default=CANONICAL_REGIME_IDS[3],
    )
    output = complete.rename(columns={"score_available_at": "label_available_at"})
    return output.loc[:, ["reference_month", "regime_id", "label_available_at"]].sort_values(
        "reference_month"
    ).reset_index(drop=True)


def _selected_baseline_rows(
    decision_marginals: pd.DataFrame,
    *,
    baseline_variant_id: str,
) -> pd.DataFrame:
    if not isinstance(baseline_variant_id, str) or not baseline_variant_id:
        raise ValueError("baseline_variant_id must be a non-empty string")
    variant_column = next(
        (column for column in _VARIANT_COLUMNS if column in decision_marginals.columns),
        None,
    )
    if variant_column is not None:
        selected = decision_marginals.loc[
            decision_marginals[variant_column].astype(str).eq(baseline_variant_id)
        ].copy()
        if selected.empty:
            raise ValueError(
                f"decision marginals do not contain baseline variant {baseline_variant_id!r}"
            )
        return selected
    if "model_role" in decision_marginals.columns:
        selected = decision_marginals.loc[
            decision_marginals["model_role"].astype(str).str.casefold().eq("baseline")
        ].copy()
        if selected.empty:
            raise ValueError("decision marginals do not contain a baseline model_role")
        return selected
    return decision_marginals.copy()


def select_m02_weekly_signals(
    decision_marginals: pd.DataFrame,
    reference_weeks: Sequence[object],
    *,
    baseline_variant_id: str = DEFAULT_BASELINE_VARIANT_ID,
) -> pd.DataFrame:
    """Select one causal current marginal for every requested calendar week.

    ``decision_marginals`` may contain several model variants and several
    rolling reference months per signal.  The selected baseline's latest
    ``signal_date`` available by each Monday is used, followed by its newest
    regime reference month.  If an explicit ``reference_week`` column is
    supplied, candidates are first restricted to that requested week.
    """

    month_column = (
        "regime_reference_month"
        if "regime_reference_month" in decision_marginals.columns
        else "reference_month"
    )
    _require_columns(
        decision_marginals,
        ("signal_date", month_column, *PROBABILITY_COLUMNS),
        name="Model 02 decision marginals",
    )
    weeks = _normalize_reference_weeks(reference_weeks)
    selected = _selected_baseline_rows(
        decision_marginals, baseline_variant_id=baseline_variant_id
    )
    selected["signal_date"] = _parse_dates(
        selected["signal_date"], name="decision signal_date"
    )
    selected["regime_reference_month"] = _normalize_months(
        selected[month_column], name="decision regime_reference_month"
    )
    has_explicit_week = "reference_week" in selected.columns
    if has_explicit_week:
        selected["reference_week"] = _parse_dates(
            selected["reference_week"], name="decision reference_week"
        )
        if selected["reference_week"].dt.weekday.ne(0).any():
            raise ValueError("decision reference_week values must be calendar Mondays")
        if (selected["signal_date"] > selected["reference_week"]).any():
            raise ValueError("a weekly decision signal cannot occur after its reference week")

    probabilities = selected.loc[:, list(PROBABILITY_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    probability_values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(probability_values).all():
        raise ValueError("decision probabilities must be finite")
    if (probability_values < 0.0).any():
        raise ValueError("decision probabilities cannot be negative")
    if not np.allclose(
        probability_values.sum(axis=1),
        1.0,
        atol=_PROBABILITY_TOLERANCE,
        rtol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("each decision probability vector must sum to one")
    selected.loc[:, list(PROBABILITY_COLUMNS)] = probabilities

    newest_allowed = selected["signal_date"].dt.to_period("M").dt.to_timestamp()
    if (selected["regime_reference_month"] > newest_allowed).any():
        raise ValueError("decision marginals cannot describe a future regime month")
    uniqueness = ["signal_date", "regime_reference_month"]
    if has_explicit_week:
        uniqueness.insert(0, "reference_week")
    if selected.duplicated(uniqueness).any():
        raise ValueError(
            "decision marginals contain duplicate weekly signal/reference-month rows"
        )

    records: list[dict[str, object]] = []
    for week in weeks:
        if has_explicit_week:
            candidates = selected.loc[selected["reference_week"].eq(week)]
        else:
            candidates = selected.loc[selected["signal_date"].le(week)]
        if candidates.empty:
            raise ValueError(f"no causal Model 02 decision marginal exists for {week:%Y-%m-%d}")
        signal_date = pd.Timestamp(candidates["signal_date"].max())
        at_signal = candidates.loc[candidates["signal_date"].eq(signal_date)]
        regime_month = pd.Timestamp(at_signal["regime_reference_month"].max())
        current = at_signal.loc[at_signal["regime_reference_month"].eq(regime_month)]
        if len(current) != 1:
            raise ValueError(
                f"weekly signal {week:%Y-%m-%d} does not have one unique newest marginal"
            )
        row = current.iloc[0]
        record: dict[str, object] = {
            "reference_week": pd.Timestamp(week),
            "signal_date": signal_date,
            "regime_reference_month": regime_month,
        }
        record.update({column: float(row[column]) for column in PROBABILITY_COLUMNS})
        records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=[
            "reference_week",
            "signal_date",
            "regime_reference_month",
            *PROBABILITY_COLUMNS,
        ],
    )


def build_weekly_execution_schedule(
    prices: pd.DataFrame,
    *,
    assets: Sequence[str],
    reference_weeks: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Build Monday-anchored first-common-session weekly execution periods.

    The last available week is retained with a missing ``end_date`` and
    ``is_complete=False`` so callers can publish its live target.  A missing
    week inside the observed common-price calendar is rejected rather than
    silently creating a multi-week holding period.  Callers are responsible
    for excluding a leading partial calendar week, such as the 2007-12-31
    anchor in the Model 02 ETF history, from estimator observations.
    """

    _require_columns(prices, ("date", "ticker"), name="price table")
    asset_tuple = tuple(map(str, assets))
    if not asset_tuple or len(set(asset_tuple)) != len(asset_tuple):
        raise ValueError("assets must be non-empty and unique")
    if any(not asset for asset in asset_tuple):
        raise ValueError("asset identifiers cannot be empty")

    selected = prices.loc[:, ["date", "ticker"]].copy()
    selected["date"] = _parse_dates(selected["date"], name="price date")
    selected["ticker"] = selected["ticker"].astype(str)
    if selected.duplicated(["date", "ticker"]).any():
        raise ValueError("price table contains duplicate ticker-date rows")
    selected = selected.loc[selected["ticker"].isin(asset_tuple)].copy()
    absent = sorted(set(asset_tuple).difference(selected["ticker"].unique()))
    if absent:
        raise ValueError(f"price table is missing requested assets: {absent}")

    dates_by_asset = [
        set(selected.loc[selected["ticker"].eq(asset), "date"])
        for asset in asset_tuple
    ]
    common_dates = sorted(set.intersection(*dates_by_asset))
    if not common_dates:
        raise ValueError("requested assets have no common price dates")
    common = pd.DatetimeIndex(common_dates)
    monday_anchors = common - pd.to_timedelta(common.weekday, unit="D")
    start_by_week = (
        pd.Series(common, index=monday_anchors)
        .groupby(level=0, sort=True)
        .min()
    )
    available_weeks = pd.DatetimeIndex(start_by_week.index, name="reference_week")
    weeks = (
        available_weeks
        if reference_weeks is None
        else _normalize_reference_weeks(reference_weeks)
    )
    missing = weeks.difference(available_weeks)
    if len(missing):
        rendered = ", ".join(timestamp.strftime("%Y-%m-%d") for timestamp in missing)
        raise ValueError(f"requested weeks have no common execution session: {rendered}")

    records: list[dict[str, object]] = []
    for week in weeks:
        start_date = pd.Timestamp(start_by_week.loc[week])
        next_week = pd.Timestamp(week) + pd.Timedelta(days=7)
        if next_week in start_by_week.index:
            end_date = pd.Timestamp(start_by_week.loc[next_week])
            is_complete = True
        else:
            later_weeks = available_weeks[available_weeks > week]
            if len(later_weeks):
                raise ValueError(f"common price calendar skips the week after {week:%Y-%m-%d}")
            end_date = pd.NaT
            is_complete = False
        records.append(
            {
                "reference_week": pd.Timestamp(week),
                "start_date": start_date,
                "end_date": end_date,
                "is_complete": is_complete,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["reference_week", "start_date", "end_date", "is_complete"],
    )
