"""Offline, independently specified cash/action checks for saved Yahoo responses."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, date, datetime

import pytest

from regime_allocation.data.source_price_audit import (
    compare_yahoo_snapshots,
    reconcile_yahoo_chart,
)


def _ts(day: int, hour: int = 14) -> int:
    return int(datetime(2026, 9, day, hour, tzinfo=UTC).timestamp())


def _payload(closes=(100.0, 101.0, 102.0), adjusted=None, days=(8, 9, 10), events=None) -> dict:
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "TEST",
                        "currency": "USD",
                        "dataGranularity": "1d",
                        "exchangeTimezoneName": "America/New_York",
                    },
                    "timestamp": [_ts(day) for day in days],
                    "indicators": {
                        "quote": [{"close": list(closes)}],
                        "adjclose": [
                            {"adjclose": list(adjusted if adjusted is not None else closes)}
                        ],
                    },
                    "events": events or {},
                }
            ],
        }
    }


def _result(payload: dict) -> dict:
    return payload["chart"]["result"][0]


def _bytes(payload: dict) -> bytes:
    return json.dumps(payload, allow_nan=False).encode()


def _audit(payload: dict, *, as_of=date(2026, 9, 10), tolerance_bps=5.0) -> dict:
    report = reconcile_yahoo_chart(
        _bytes(payload), ticker="TEST", as_of=as_of, tolerance_bps=tolerance_bps
    )
    # The public artifact must be strict JSON, not NaN-bearing Python-only output.
    json.dumps(report, allow_nan=False)
    return report


def _dividend(day=9, amount=1.0) -> dict:
    return {"dividends": {"cash": {"date": _ts(day), "amount": amount}}}


def _split(day=9, numerator=1.0, denominator=8.0) -> dict:
    return {
        "splits": {
            "split": {
                "date": _ts(day),
                "numerator": numerator,
                "denominator": denominator,
                "splitRatio": f"{numerator}:{denominator}",
            }
        }
    }


def test_ordinary_cash_distribution_is_reconciled_on_ex_date() -> None:
    # Price falls $1 for a $1 ex-date distribution: economic total return is zero.
    payload = _payload(closes=(100, 99, 100), adjusted=(99, 99, 100), events=_dividend())
    report = _audit(payload)
    assert report["status"] == "passed"
    assert report["independent_source_verified"] is False
    assert report["summary"]["compared_intervals"] == 2
    assert report["actions"][0]["status"] == "reconciled"
    row = report["intervals"][0]
    assert row["cash_return"] == row["adjusted_return"] == 0
    assert row["cash_dividend"] == 1
    assert report["intervals"][1]["cash_dividend"] == 0


def test_split_adjusted_close_never_applies_split_twice() -> None:
    # Mirrors saved USO's 2020 reverse-split convention: both closes are on the
    # post-split share basis, although a 1:8 event appears in the action feed.
    report = _audit(_payload(closes=(17.04, 18, 19.12), events=_split()))
    assert report["status"] == "passed"
    assert report["intervals"][0]["split_ratio"] == 0.125
    assert report["intervals"][0]["cash_return"] == pytest.approx(18 / 17.04 - 1)
    assert report["actions"][0]["status"] == "reconciled"


def test_unadjusted_split_prices_conflicting_with_adjusted_prices_fail() -> None:
    report = _audit(_payload(closes=(2, 16, 17), adjusted=(16, 16, 17), events=_split()))
    assert report["status"] == "failed"
    assert report["intervals"][0]["status"] == "discrepancy"
    assert report["actions"][0]["status"] == "discrepancy"


def test_dividends_are_not_rescaled_again_for_later_split() -> None:
    events = {**_dividend(amount=1.0), **_split(day=10)}
    report = _audit(_payload(closes=(100, 99, 99), adjusted=(99, 99, 99), events=events))
    assert report["status"] == "passed"
    assert report["intervals"][0]["cash_dividend"] == 1
    assert report["intervals"][1]["cash_return"] == 0


def test_wrong_dividend_unit_or_amount_is_a_discrepancy() -> None:
    report = _audit(
        _payload(closes=(100, 99, 99), adjusted=(99, 99, 99), events=_dividend(amount=8))
    )
    assert report["status"] == "failed"
    assert report["intervals"][0]["difference_bps"] == pytest.approx(-700)


def test_wrong_action_date_is_detected_in_both_intervals() -> None:
    report = _audit(_payload(closes=(100, 99, 99), adjusted=(99, 99, 99), events=_dividend(day=10)))
    assert report["summary"]["discrepancy_intervals"] == 2
    assert report["status"] == "failed"


def test_simultaneous_cash_and_split_require_explicit_unit_verification() -> None:
    report = _audit(
        _payload(closes=(100, 99, 99), adjusted=(99, 99, 99), events={**_dividend(), **_split()})
    )
    assert report["status"] == "failed"
    assert report["intervals"][0]["status"] == "not_comparable"
    assert all(
        "same_day_cash_and_split_units_require_verification" in row["issues"]
        for row in report["actions"]
    )


@pytest.mark.parametrize("field", ["close", "adjclose"])
@pytest.mark.parametrize(
    "bad_value,reason",
    [
        (None, "missing"),
        (0, "non_positive"),
        (-1, "non_positive"),
        (True, "not_numeric"),
        ("100", "not_numeric"),
    ],
)
def test_bad_prices_do_not_bridge_or_drop_intervals(field, bad_value, reason) -> None:
    payload = _payload()
    indicators = _result(payload)["indicators"]
    indicator = indicators["quote"][0] if field == "close" else indicators["adjclose"][0]
    indicator[field][1] = bad_value
    report = _audit(payload)
    assert report["status"] == "failed"
    assert len(report["intervals"]) == 2
    assert report["summary"]["not_comparable_intervals"] == 2
    assert report["summary"]["compared_intervals"] == 0
    assert any(issue["code"].endswith(reason) for issue in report["issues"])


def test_out_of_range_json_number_is_invalid_data_with_strict_json_report() -> None:
    content = _bytes(_payload()).replace(b"101.0", b"1e999")
    report = reconcile_yahoo_chart(content, ticker="TEST", as_of=date(2026, 9, 10))
    assert report["status"] == "failed"
    assert any(issue["code"].endswith("non_finite") for issue in report["issues"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("days", [(8, 8, 10), (8, 9, 9)])
def test_duplicate_daily_dates_are_rejected(days) -> None:
    with pytest.raises(ValueError, match="Duplicate daily"):
        _audit(_payload(days=days))


def test_different_timestamps_on_one_exchange_day_are_ambiguous() -> None:
    payload = _payload()
    _result(payload)["timestamp"][1] = _ts(8, 20)
    with pytest.raises(ValueError, match="Duplicate daily"):
        _audit(payload)


def test_duplicate_action_dates_are_not_silently_summed() -> None:
    events = _dividend()
    events["dividends"]["second"] = {"date": _ts(9, 20), "amount": 1}
    with pytest.raises(ValueError, match="Ambiguous duplicate dividends"):
        _audit(_payload(events=events))


@pytest.mark.parametrize("day", [7, 8, 10])
def test_unmatched_and_first_row_actions_prevent_pass(day) -> None:
    report = _audit(_payload(closes=(100, 101), days=(8, 9), events=_dividend(day)))
    assert report["status"] == "failed"
    assert report["actions"][0]["status"] == "unreconciled"


def test_action_in_missing_intermediate_session_blocks_bridged_comparison() -> None:
    report = _audit(_payload(closes=(100, 100), days=(8, 10), events=_dividend()))
    assert report["status"] == "failed"
    assert report["intervals"][0]["reasons"] == ["action_on_missing_intermediate_price_date"]


def test_unsupported_capital_gain_is_not_treated_as_zero_cash() -> None:
    report = _audit(_payload(events={"capitalGains": {"gain": {"date": _ts(9), "amount": 1.0}}}))
    assert report["status"] == "failed"
    assert report["actions"][0]["issues"] == ["unsupported_action_type"]
    assert report["intervals"][0]["status"] == "not_comparable"


@pytest.mark.parametrize("amount", [None, 0, -1, "1", True])
def test_invalid_dividend_is_reported(amount) -> None:
    report = _audit(_payload(events=_dividend(amount=amount)))
    assert report["status"] == "failed"
    assert report["actions"][0]["issues"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("numerator", 0),
        ("denominator", None),
        ("splitRatio", "8:1"),
        ("splitRatio", "8:0"),
        ("splitRatio", "garbage"),
    ],
)
def test_invalid_or_conflicting_split_fields_prevent_pass(field, value) -> None:
    events = _split()
    events["splits"]["split"][field] = value
    report = _audit(_payload(events=events))
    assert report["status"] == "failed"
    assert report["actions"][0]["issues"]


def test_no_comparisons_is_insufficient_not_passed() -> None:
    for closes, days in [((100,), (8,)), ((), ())]:
        report = _audit(_payload(closes=closes, days=days))
        assert report["status"] == "insufficient_data"
        assert report["summary"]["compared_intervals"] == 0


def test_discrepancy_without_reported_action_is_visible() -> None:
    report = _audit(_payload(adjusted=(100, 99, 102)))
    assert report["status"] == "failed"
    assert report["summary"]["discrepancy_intervals"] == 2
    assert report["intervals"][0]["difference_bps"] == pytest.approx(-200)


def test_tolerance_is_explicit_and_respected() -> None:
    payload = _payload(closes=(100, 100), adjusted=(100, 100.04), days=(8, 9))
    assert _audit(payload, tolerance_bps=5)["status"] == "passed"
    assert _audit(payload, tolerance_bps=3)["status"] == "failed"


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf"), True, "5"])
def test_invalid_tolerance_is_rejected(tolerance) -> None:
    with pytest.raises(ValueError, match="tolerance_bps"):
        _audit(_payload(), tolerance_bps=tolerance)


def test_future_prices_and_actions_cannot_change_eligible_reconciliation() -> None:
    original = _payload(closes=(100, 99, 100), adjusted=(99, 99, 100), events=_dividend())
    future = deepcopy(original)
    result = _result(future)
    result["timestamp"].extend([_ts(11), _ts(11)])  # Future duplicates are out of scope.
    result["indicators"]["quote"][0]["close"].extend([None, -100])
    result["indicators"]["adjclose"][0]["adjclose"].extend([None, 0])
    result["events"]["capitalGains"] = {"future": {"date": _ts(11), "amount": "invalid"}}
    result["events"]["dividends"]["future"] = {"date": _ts(11), "amount": -1}
    before, after = _audit(original), _audit(future)
    for key in ("status", "summary", "intervals", "actions", "issues"):
        assert before[key] == after[key]
    assert after["coverage"]["last_observation"] == "2026-09-10"
    assert after["coverage"]["observations_after_cutoff_excluded"] == 2
    assert after["coverage"]["actions_after_cutoff_excluded"] == 2
    assert after["source_sha256"] != before["source_sha256"]


def test_exchange_timezone_controls_cutoff_and_event_matching() -> None:
    payload = _payload(closes=(100, 99), adjusted=(99, 99), days=(8, 9), events=_dividend())
    # 00:00 UTC on Sep10 is Sep9 in New York, still eligible for Sep9.
    _result(payload)["timestamp"][1] = _ts(10, 0)
    _result(payload)["events"]["dividends"]["cash"]["date"] = _ts(10, 0)
    report = _audit(payload, as_of=date(2026, 9, 9))
    assert report["status"] == "passed"
    assert report["actions"][0]["date"] == "2026-09-09"


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", "WRONG"),
        ("currency", None),
        ("dataGranularity", "1wk"),
        ("exchangeTimezoneName", None),
        ("exchangeTimezoneName", "NOT_A_TIMEZONE"),
    ],
)
def test_identity_or_time_basis_ambiguity_raises(field, value) -> None:
    payload = _payload()
    _result(payload)["meta"][field] = value
    with pytest.raises(ValueError):
        _audit(payload)


def test_malformed_lengths_and_missing_event_dates_raise() -> None:
    payload = _payload()
    _result(payload)["indicators"]["quote"][0]["close"].pop()
    with pytest.raises(ValueError, match="array length"):
        _audit(payload)
    with pytest.raises(ValueError, match="integer Unix timestamp"):
        _audit(_payload(events={"unknown": {"action": {"amount": 1}}}))


@pytest.mark.parametrize("content", [b'{"chart": {}, "chart": {}}', b"not json", b'{"chart": NaN}'])
def test_malformed_json_and_duplicate_keys_raise(content) -> None:
    with pytest.raises(ValueError):
        reconcile_yahoo_chart(content, ticker="TEST", as_of=date(2026, 9, 10))


def _diff(old: dict, new: dict, as_of=date(2026, 9, 10)) -> dict:
    result = compare_yahoo_snapshots(_bytes(old), _bytes(new), ticker="TEST", as_of=as_of)
    json.dumps(result, allow_nan=False)
    return result


def test_snapshot_diff_separates_extended_coverage_from_revisions() -> None:
    old = _payload(closes=(100, 101), days=(8, 9))
    new = _payload()
    report = _diff(old, new)
    assert report["status"] == "unchanged_common_support"
    assert report["coverage"]["new_only_dates"] == ["2026-09-10"]
    assert report["price_changes"] == report["action_changes"] == []
    assert report["compared_adjclose_intervals"] == 1


def test_snapshot_diff_reports_adjusted_level_rescaling_separately_from_returns() -> None:
    old = _payload(closes=(100, 100, 100))
    new = _payload(closes=(100, 100, 100), adjusted=(50, 50, 50))
    report = _diff(old, new)
    assert report["status"] == "changed_common_support"
    assert len(report["price_changes"]) == 3
    assert report["adjusted_return_changes"] == []
    assert all(row["field"] == "adjusted_close" for row in report["price_changes"])


def test_snapshot_diff_reports_real_return_and_action_changes() -> None:
    old = _payload(closes=(100, 99, 99), adjusted=(99, 99, 99), events=_dividend(amount=1))
    new = _payload(closes=(100, 98, 99), adjusted=(98, 98, 99), events=_dividend(amount=2))
    report = _diff(old, new)
    assert report["common_support_changed"] is True
    assert len(report["adjusted_return_changes"]) == 1
    action = report["action_changes"][0]
    assert action["change"] == "modified"
    assert action["old"]["amount"] == 1
    assert action["new"]["amount"] == 2


def test_snapshot_action_id_rename_is_not_economic_revision() -> None:
    old = _payload(events=_dividend())
    new = deepcopy(old)
    events = _result(new)["events"]["dividends"]
    events["renamed-key"] = events.pop("cash")
    assert _diff(old, new)["status"] == "unchanged_common_support"


def test_snapshot_added_action_within_common_support_is_revision() -> None:
    report = _diff(_payload(), _payload(events=_dividend()))
    assert report["action_changes"][0]["change"] == "added"


def test_snapshot_added_action_beyond_old_span_is_coverage() -> None:
    report = _diff(_payload(closes=(100, 101), days=(8, 9)), _payload(events=_dividend(day=10)))
    assert report["status"] == "unchanged_common_support"
    assert len(report["coverage"]["new_actions_outside_common_span"]) == 1


def test_snapshot_no_common_support_cannot_pass() -> None:
    report = _diff(_payload(closes=(100,), days=(8,)), _payload(closes=(100,), days=(9,)))
    assert report["status"] == "insufficient_data"


def test_snapshot_invalid_prices_are_reported_even_if_unchanged() -> None:
    payload = _payload(closes=(100, None, 102))
    report = _diff(payload, payload)
    assert report["status"] == "failed"
    assert len(report["quality_issues"]) == 4  # Close and adjusted close in each snapshot.
    assert report["compared_adjclose_intervals"] == 0


def test_snapshot_revisions_after_cutoff_are_ignored() -> None:
    old = _payload()
    new = _payload(closes=(100, 101, 1), adjusted=(100, 101, 1), events=_dividend(day=10))
    report = _diff(old, new, as_of=date(2026, 9, 9))
    assert report["status"] == "unchanged_common_support"
    assert report["price_changes"] == report["action_changes"] == []


def test_snapshot_currency_mismatch_is_not_called_a_price_revision() -> None:
    old, new = _payload(), _payload()
    _result(new)["meta"]["currency"] = "EUR"
    with pytest.raises(ValueError, match="currency"):
        _diff(old, new)


def test_extreme_finite_prices_do_not_produce_non_json_return_values() -> None:
    payload = _payload(closes=(1e-300, 1e300), days=(8, 9))
    report = _audit(payload)
    assert report["status"] == "failed"
    assert report["intervals"][0]["reasons"] == ["return_arithmetic_out_of_range"]
    diff = _diff(payload, payload)
    assert diff["status"] == "failed"
    assert diff["not_comparable_adjclose_intervals"]["old"][0]["reason"] == (
        "return_arithmetic_out_of_range"
    )


def test_extreme_return_difference_is_reported_without_json_infinity() -> None:
    old = _payload(closes=(1, 1), days=(8, 9))
    new = _payload(closes=(1, 1e308), days=(8, 9))
    diff = _diff(old, new)
    assert diff["status"] == "failed"
    assert diff["adjusted_return_changes"][0]["difference_bps"] is None
    assert diff["quality_issues"][0]["code"] == "return_difference_arithmetic_out_of_range"
