"""Independent arithmetic checks of saved Yahoo daily chart responses.

This module deliberately does not import the downloader or its return calculation.
Both inputs still come from Yahoo: a passing check is internal provider consistency,
not independent-source verification, a point-in-time vintage, or a market-calendar audit.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class _Snapshot:
    ticker: str
    currency: str
    timezone: str
    prices: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    excluded_prices: int
    excluded_actions: int
    sha256: str


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Ambiguous Yahoo JSON: duplicate object key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant: {value}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        # This is a value error in serialized source data, not a Python API type error.
        raise ValueError(f"{label} must be a JSON object")  # noqa: TRY004
    return value


def _one_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{label} must contain exactly one object")
    return _object(value[0], label)


def _positive_number(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, "missing"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "not_numeric"
    try:
        number = float(value)
    except OverflowError:
        return None, "non_finite"
    if not math.isfinite(number):
        return None, "non_finite"
    if number <= 0:
        return number, "non_positive"
    return number, None


def _exchange_day(value: Any, timezone: ZoneInfo, label: str) -> tuple[int, str]:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer Unix timestamp")  # noqa: TRY004
    try:
        day = datetime.fromtimestamp(value, UTC).astimezone(timezone).date()
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"{label} is outside the supported timestamp range") from exc
    return value, day.isoformat()


def _parse_action(kind: str, event_id: str, value: dict, timestamp: int, day: str) -> dict:
    action: dict[str, Any] = {
        "source_event_id": event_id,
        "kind": kind,
        "timestamp": timestamp,
        "date": day,
        "issues": [],
    }
    if kind == "dividends":
        amount, problem = _positive_number(value.get("amount"))
        action["amount"] = amount
        if problem:
            action["issues"].append(f"dividend_amount_{problem}")
    elif kind == "splits":
        numerator, n_problem = _positive_number(value.get("numerator"))
        denominator, d_problem = _positive_number(value.get("denominator"))
        action.update(numerator=numerator, denominator=denominator, split_ratio=None)
        for field, problem in (("numerator", n_problem), ("denominator", d_problem)):
            if problem:
                action["issues"].append(f"split_{field}_{problem}")
        if not action["issues"]:
            ratio = numerator / denominator
            if not math.isfinite(ratio) or ratio <= 0:
                action["issues"].append("split_ratio_out_of_range")
            else:
                action["split_ratio"] = ratio
        # A disagreeing redundant ratio is ambiguous; never silently prefer one field.
        if "splitRatio" in value:
            try:
                parts = value["splitRatio"].split(":")
                if len(parts) != 2:
                    raise ValueError
                left, right = (float(part) for part in parts)
                if not all(math.isfinite(x) and x > 0 for x in (left, right)):
                    raise ValueError
                stated_ratio = left / right
                if not math.isfinite(stated_ratio) or stated_ratio <= 0:
                    raise ValueError
                if action["split_ratio"] is not None and not math.isclose(
                    stated_ratio, action["split_ratio"], rel_tol=1e-12, abs_tol=0.0
                ):
                    action["issues"].append("conflicting_split_ratio")
            except (AttributeError, TypeError, ValueError, OverflowError, ZeroDivisionError):
                action["issues"].append("invalid_split_ratio_text")
    else:
        action["issues"].append("unsupported_action_type")
        # Retain the payload for a refresh diff, without permitting invalid JSON numbers.
        try:
            action["unsupported_payload"] = json.loads(json.dumps(value, allow_nan=False))
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid unsupported action {kind}/{event_id}") from exc
    return action


def _parse(content: bytes, *, ticker: str, as_of: date) -> _Snapshot:
    if not isinstance(content, bytes):
        raise TypeError("content must be saved Yahoo response bytes")
    if type(as_of) is not date:
        raise TypeError("as_of must be a datetime.date, not a datetime or string")
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be non-empty")
    try:
        payload = json.loads(
            content, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed Yahoo chart JSON") from exc
    chart = _object(_object(payload, "payload").get("chart"), "chart")
    if chart.get("error") is not None:
        raise ValueError(f"Yahoo chart reports an error: {chart['error']!r}")
    result = _one_object(chart.get("result"), "chart.result")
    meta = _object(result.get("meta"), "meta")
    if meta.get("symbol") != ticker:
        raise ValueError(f"Yahoo symbol {meta.get('symbol')!r} does not match {ticker!r}")
    if meta.get("dataGranularity") != "1d":
        raise ValueError("Only explicit daily (1d) chart data is supported")
    currency = meta.get("currency")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("Missing price/dividend currency metadata")
    timezone_name = meta.get("exchangeTimezoneName")
    if not isinstance(timezone_name, str) or not timezone_name:
        raise ValueError("Missing exchange timezone; date matching would be ambiguous")
    try:
        timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown exchange timezone {timezone_name!r}") from exc
    timestamps = result.get("timestamp")
    if not isinstance(timestamps, list):
        raise ValueError("timestamp must be an array (possibly empty)")  # noqa: TRY004
    indicators = _object(result.get("indicators"), "indicators")
    quote = _one_object(indicators.get("quote"), "indicators.quote")
    adjclose = _one_object(indicators.get("adjclose"), "indicators.adjclose")
    closes, adjusted = quote.get("close"), adjclose.get("adjclose")
    if any(
        not isinstance(values, list) or len(values) != len(timestamps)
        for values in (closes, adjusted)
    ):
        raise ValueError("close and adjclose arrays must match the timestamp array length")
    cutoff = as_of.isoformat()
    prices: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    excluded_prices = 0
    for index, raw_timestamp in enumerate(timestamps):
        timestamp, day = _exchange_day(raw_timestamp, timezone, f"timestamp[{index}]")
        if day > cutoff:
            excluded_prices += 1
            continue
        if day in seen_dates:
            raise ValueError(f"Duplicate daily date/timestamp at {day}")
        seen_dates.add(day)
        close, close_problem = _positive_number(closes[index])
        adjusted_close, adjusted_problem = _positive_number(adjusted[index])
        prices.append(
            {
                "date": day,
                "timestamp": timestamp,
                "close": close,
                "adjusted_close": adjusted_close,
                "issues": [
                    f"{field}_{problem}"
                    for field, problem in (
                        ("close", close_problem),
                        ("adjusted_close", adjusted_problem),
                    )
                    if problem
                ],
            }
        )
    prices.sort(key=lambda row: row["date"])
    events = _object(result.get("events", {}), "events")
    actions: list[dict[str, Any]] = []
    seen_actions: set[tuple[str, str]] = set()
    excluded_actions = 0
    for kind, raw_events in events.items():
        for event_id, raw_event in _object(raw_events, f"events.{kind}").items():
            event = _object(raw_event, f"events.{kind}.{event_id}")
            timestamp, day = _exchange_day(event.get("date"), timezone, f"action {event_id}.date")
            if day > cutoff:
                excluded_actions += 1
                continue
            identity = (kind, day)
            if identity in seen_actions:
                raise ValueError(f"Ambiguous duplicate {kind} actions on {day}")
            seen_actions.add(identity)
            actions.append(_parse_action(kind, event_id, event, timestamp, day))
    actions.sort(key=lambda action: (action["date"], action["kind"]))
    return _Snapshot(
        ticker,
        currency,
        timezone_name,
        prices,
        actions,
        excluded_prices,
        excluded_actions,
        hashlib.sha256(content).hexdigest(),
    )


def _coverage(snapshot: _Snapshot, as_of: date) -> dict:
    return {
        "cutoff": as_of.isoformat(),
        "cutoff_basis": "exchange-local calendar date, inclusive",
        "first_observation": snapshot.prices[0]["date"] if snapshot.prices else None,
        "last_observation": snapshot.prices[-1]["date"] if snapshot.prices else None,
        "observations": len(snapshot.prices),
        "actions": len(snapshot.actions),
        "observations_after_cutoff_excluded": snapshot.excluded_prices,
        "actions_after_cutoff_excluded": snapshot.excluded_actions,
        "market_calendar_completeness_checked": False,
        "same_day_session_completion_checked": False,
    }


def reconcile_yahoo_chart(
    content: bytes, *, ticker: str, as_of: date, tolerance_bps: float = 5.0
) -> dict:
    """Reconcile every adjacent in-cutoff daily pair using an independent calculation.

    Structural ambiguity raises ValueError. Bad observed values, unsupported/unmatched
    actions, and discrepancies produce a failed report, never an unexplained dropped row.
    A report with no comparable intervals cannot pass. Values after ``as_of`` are not
    inspected or used; full-file hashes and excluded counts still describe that source.

    Yahoo close is already split-adjusted. Cash is used on the provider's event date
    (assumed ex-date), in the reported currency and historical split-adjusted per-share
    basis. No split multiplier or cash-unit conversion is applied. Same-day cash/split
    combinations require unit verification and fail conservatively. Yahoo adjustment
    conventions need not exactly equal a cash-reinvestment identity; tolerance is explicit.
    """
    if (
        isinstance(tolerance_bps, bool)
        or not isinstance(tolerance_bps, (int, float))
        or not math.isfinite(tolerance_bps)
        or tolerance_bps < 0
    ):
        raise ValueError("tolerance_bps must be finite and non-negative")
    snapshot = _parse(content, ticker=ticker, as_of=as_of)
    by_date = {row["date"]: row for row in snapshot.prices}
    actions_by_date: dict[str, list[dict]] = {}
    for action in snapshot.actions:
        actions_by_date.setdefault(action["date"], []).append(action)
    issues = [
        {"date": row["date"], "code": issue, "source": "price"}
        for row in snapshot.prices
        for issue in row["issues"]
    ]
    action_reports: list[dict] = []
    first_day = snapshot.prices[0]["date"] if snapshot.prices else None
    for action in snapshot.actions:
        report = dict(action)
        report["issues"] = list(action["issues"])
        report["status"] = "pending"
        if action["date"] not in by_date:
            report["issues"].append("action_has_no_matching_price_date")
        elif action["date"] == first_day:
            report["issues"].append("action_has_no_prior_price")
        same_day_kinds = {item["kind"] for item in actions_by_date[action["date"]]}
        if {"dividends", "splits"} <= same_day_kinds:
            report["issues"].append("same_day_cash_and_split_units_require_verification")
        if report["issues"]:
            report["status"] = "unreconciled"
        for issue in report["issues"]:
            issues.append({"date": action["date"], "code": issue, "source": action["kind"]})
        action_reports.append(report)
    reports_by_date: dict[str, list[dict]] = {}
    for report in action_reports:
        reports_by_date.setdefault(report["date"], []).append(report)
    intervals: list[dict] = []
    for previous, current in zip(snapshot.prices, snapshot.prices[1:]):
        day = current["date"]
        actions = reports_by_date.get(day, [])
        missing_session_actions = [
            action for action in action_reports if previous["date"] < action["date"] < day
        ]
        row: dict[str, Any] = {
            "previous_date": previous["date"],
            "date": day,
            "calendar_gap_days": (
                date.fromisoformat(day) - date.fromisoformat(previous["date"])
            ).days,
            "previous_close": previous["close"],
            "close": current["close"],
            "previous_adjusted_close": previous["adjusted_close"],
            "adjusted_close": current["adjusted_close"],
            "cash_dividend": sum(
                action.get("amount", 0.0) or 0.0
                for action in actions
                if action["kind"] == "dividends"
            ),
            "split_ratio": next((a["split_ratio"] for a in actions if a["kind"] == "splits"), 1.0),
            "action_kinds": [action["kind"] for action in actions],
            "cash_return": None,
            "adjusted_return": None,
            "difference_bps": None,
            "status": "not_comparable",
            "reasons": [],
        }
        if previous["issues"]:
            row["reasons"].append("invalid_previous_price")
        if current["issues"]:
            row["reasons"].append("invalid_current_price")
        if any(action["issues"] for action in actions):
            row["reasons"].append("invalid_or_unresolved_action")
        if missing_session_actions:
            row["reasons"].append("action_on_missing_intermediate_price_date")
        if not row["reasons"]:
            cash_return = (current["close"] + row["cash_dividend"]) / previous["close"] - 1
            adjusted_return = current["adjusted_close"] / previous["adjusted_close"] - 1
            difference_bps = (adjusted_return - cash_return) * 10_000
            if all(math.isfinite(x) for x in (cash_return, adjusted_return, difference_bps)):
                row.update(
                    cash_return=cash_return,
                    adjusted_return=adjusted_return,
                    difference_bps=difference_bps,
                    status="passed" if abs(difference_bps) <= tolerance_bps else "discrepancy",
                )
            else:
                row["reasons"].append("return_arithmetic_out_of_range")
        if row["status"] == "discrepancy":
            issues.append(
                {
                    "date": day,
                    "code": "adjusted_cash_return_discrepancy",
                    "source": "reconciliation",
                }
            )
        for reason in row["reasons"]:
            issues.append({"date": day, "code": reason, "source": "reconciliation"})
        for action in actions:
            if action["status"] == "pending":
                action["status"] = {
                    "passed": "reconciled",
                    "discrepancy": "discrepancy",
                    "not_comparable": "unreconciled",
                }[row["status"]]
                action["comparison_date"] = day
        intervals.append(row)
    compared = [row for row in intervals if row["difference_bps"] is not None]
    status = "failed" if issues else "passed" if compared else "insufficient_data"
    return {
        "schema_version": "yahoo_action_price_audit_v1",
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "status": status,
        "source_sha256": snapshot.sha256,
        "verification_scope": "independent_calculation_same_provider",
        "independent_source_verified": False,
        "point_in_time_source_vintage_verified": False,
        "currency": snapshot.currency,
        "exchange_timezone": snapshot.timezone,
        "tolerance_bps": float(tolerance_bps),
        "conventions": {
            "cash_return": "(close_t + dividend_t) / close_previous - 1",
            "adjusted_return": "adjusted_close_t / adjusted_close_previous - 1",
            "difference_bps": "10000 * (adjusted_return - cash_return)",
            "split_handling": "Yahoo close is already split adjusted; no split multiplier",
            "cash_units": "provider-reported cash per share, assumed same split basis as close",
            "cash_timing": "provider event date in exchange timezone, assumed ex-dividend date",
            "dividend_identity": "cash identity may differ from provider adjustment convention",
            "first_row": "no prior close; no return computed; any action remains unreconciled",
        },
        "coverage": _coverage(snapshot, as_of),
        "summary": {
            "candidate_intervals": len(intervals),
            "compared_intervals": len(compared),
            "passed_intervals": sum(row["status"] == "passed" for row in intervals),
            "discrepancy_intervals": sum(row["status"] == "discrepancy" for row in intervals),
            "not_comparable_intervals": sum(row["status"] == "not_comparable" for row in intervals),
            "invalid_price_rows": sum(bool(row["issues"]) for row in snapshot.prices),
            "reconciled_actions": sum(row["status"] == "reconciled" for row in action_reports),
            "unreconciled_actions": sum(row["status"] != "reconciled" for row in action_reports),
            "max_absolute_difference_bps": max(
                (abs(row["difference_bps"]) for row in compared), default=None
            ),
        },
        "issues": issues,
        "actions": action_reports,
        "intervals": intervals,
    }


def _action_identity(action: dict) -> tuple[str, str]:
    return action["date"], action["kind"]


def _action_values(action: dict) -> dict:
    # Provider object keys are not economic action identity. Date/type and the
    # reported amount or ratio are compared; redundant text formatting is immaterial.
    return {
        key: value for key, value in action.items() if key not in {"source_event_id", "timestamp"}
    }


def _adjusted_returns(snapshot: _Snapshot) -> tuple[dict[tuple[str, str], float], list[dict]]:
    result, excluded = {}, []
    for previous, current in zip(snapshot.prices, snapshot.prices[1:]):
        if previous["issues"] or current["issues"]:
            excluded.append(
                {
                    "previous_date": previous["date"],
                    "date": current["date"],
                    "reason": "invalid_price",
                }
            )
            continue
        value = current["adjusted_close"] / previous["adjusted_close"] - 1
        if math.isfinite(value):
            result[(previous["date"], current["date"])] = value
        else:
            excluded.append(
                {
                    "previous_date": previous["date"],
                    "date": current["date"],
                    "reason": "return_arithmetic_out_of_range",
                }
            )
    return result, excluded


def compare_yahoo_snapshots(
    old_content: bytes, new_content: bytes, *, ticker: str, as_of: date
) -> dict:
    """Diff saved responses on common support, separating coverage from revisions.

    Exact numeric differences are reported, including harmless historical adjusted
    level rescaling. Common adjacent adjusted-return changes are reported separately;
    expanded coverage and changed raw-byte hashes alone are not called revisions.
    """
    old = _parse(old_content, ticker=ticker, as_of=as_of)
    new = _parse(new_content, ticker=ticker, as_of=as_of)
    if (old.currency, old.timezone) != (new.currency, new.timezone):
        raise ValueError("Cannot compare snapshots with different currency or exchange timezone")
    old_prices = {row["date"]: row for row in old.prices}
    new_prices = {row["date"]: row for row in new.prices}
    common = sorted(old_prices.keys() & new_prices.keys())
    changes = []
    for day in common:
        for field in ("close", "adjusted_close"):
            before, after = old_prices[day][field], new_prices[day][field]
            if before != after:
                delta_bps = (
                    (after / before - 1) * 10_000
                    if before is not None and before > 0 and after is not None
                    else None
                )
                changes.append(
                    {
                        "date": day,
                        "field": field,
                        "old": before,
                        "new": after,
                        "relative_change_bps": delta_bps
                        if delta_bps is not None and math.isfinite(delta_bps)
                        else None,
                    }
                )
    old_returns, old_excluded = _adjusted_returns(old)
    new_returns, new_excluded = _adjusted_returns(new)
    common_intervals = sorted(old_returns.keys() & new_returns.keys())
    return_changes = []
    arithmetic_issues = []
    for pair in common_intervals:
        if old_returns[pair] == new_returns[pair]:
            continue
        difference_bps = (new_returns[pair] - old_returns[pair]) * 10_000
        if not math.isfinite(difference_bps):
            arithmetic_issues.append(
                {
                    "snapshot": "comparison",
                    "date": pair[1],
                    "code": "return_difference_arithmetic_out_of_range",
                }
            )
        return_changes.append(
            {
                "previous_date": pair[0],
                "date": pair[1],
                "old": old_returns[pair],
                "new": new_returns[pair],
                "difference_bps": difference_bps if math.isfinite(difference_bps) else None,
            }
        )
    first_common = common[0] if common else None
    last_common = common[-1] if common else None

    def within_common(action: dict) -> bool:
        return first_common is not None and first_common <= action["date"] <= last_common

    old_actions = {_action_identity(a): _action_values(a) for a in old.actions if within_common(a)}
    new_actions = {_action_identity(a): _action_values(a) for a in new.actions if within_common(a)}
    action_changes = [
        {
            "date": key[0],
            "kind": key[1],
            "change": "added"
            if key not in old_actions
            else "removed"
            if key not in new_actions
            else "modified",
            "old": old_actions.get(key),
            "new": new_actions.get(key),
        }
        for key in sorted(old_actions.keys() | new_actions.keys())
        if old_actions.get(key) != new_actions.get(key)
    ]
    quality_issues = [
        {"snapshot": name, "date": row["date"], "code": problem}
        for name, snapshot in (("old", old), ("new", new))
        for row in snapshot.prices + snapshot.actions
        for problem in row["issues"]
    ]
    quality_issues.extend(arithmetic_issues)
    quality_issues.extend(
        {"snapshot": name, "date": row["date"], "code": row["reason"]}
        for name, excluded in (("old", old_excluded), ("new", new_excluded))
        for row in excluded
        if row["reason"] == "return_arithmetic_out_of_range"
    )
    changed = bool(changes or action_changes)
    return {
        "schema_version": "yahoo_snapshot_diff_v1",
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "status": "failed"
        if quality_issues
        else "insufficient_data"
        if not common
        else "changed_common_support"
        if changed
        else "unchanged_common_support",
        "verification_scope": "same_provider_source_refresh_comparison",
        "independent_source_verified": False,
        "old_source_sha256": old.sha256,
        "new_source_sha256": new.sha256,
        "coverage": {
            "old": _coverage(old, as_of),
            "new": _coverage(new, as_of),
            "common_dates": len(common),
            "common_first_date": first_common,
            "common_last_date": last_common,
            "old_only_dates": sorted(old_prices.keys() - new_prices.keys()),
            "new_only_dates": sorted(new_prices.keys() - old_prices.keys()),
            "old_actions_outside_common_span": [
                _action_values(a) for a in old.actions if not within_common(a)
            ],
            "new_actions_outside_common_span": [
                _action_values(a) for a in new.actions if not within_common(a)
            ],
        },
        "common_support_changed": changed,
        "price_changes": changes,
        "action_changes": action_changes,
        "compared_adjclose_intervals": len(common_intervals),
        "adjusted_return_changes": return_changes,
        "not_comparable_adjclose_intervals": {"old": old_excluded, "new": new_excluded},
        "old_only_valid_adjclose_intervals": [
            list(pair) for pair in sorted(old_returns.keys() - new_returns.keys())
        ],
        "new_only_valid_adjclose_intervals": [
            list(pair) for pair in sorted(new_returns.keys() - old_returns.keys())
        ],
        "quality_issues": quality_issues,
        "interpretation": "Adjusted-level revisions can reflect later distributions or splits; "
        "return changes are separate. This diff does not establish when a value was published.",
    }
