"""Offline regression checks for recovering Yahoo's incomplete latest daily row."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
AS_OF = date(2026, 9, 14)
AFTER_CLOSE = datetime(2026, 9, 14, 21, tzinfo=UTC)
PRICE_COLUMNS = ("open", "high", "low", "close", "adjusted_close")


@pytest.fixture(scope="module")
def downloader() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "etf_history_download_tests", ROOT / "scripts" / "download_etf_history.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timestamp(day: date, hour: int = 13, minute: int = 30) -> int:
    return int(datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC).timestamp())


def _payload(days: tuple[date, ...] = (date(2026, 9, 10), date(2026, 9, 11), AS_OF)) -> dict:
    values = [100.0 + day.day for day in days]
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "SPY",
                        "currency": "USD",
                        "exchangeName": "PCX",
                        "fullExchangeName": "NYSEArca",
                        "instrumentType": "ETF",
                        "exchangeTimezoneName": "America/New_York",
                        "dataGranularity": "1d",
                        "currentTradingPeriod": {
                            "regular": {
                                "start": _timestamp(AS_OF),
                                "end": _timestamp(AS_OF, 20, 0),
                            }
                        },
                    },
                    "timestamp": [_timestamp(day) for day in days],
                    "indicators": {
                        "quote": [
                            {
                                "open": values.copy(),
                                "high": [value + 1 for value in values],
                                "low": [value - 1 for value in values],
                                "close": values.copy(),
                                "volume": [1000 + day.day for day in days],
                            }
                        ],
                        "adjclose": [{"adjclose": values.copy()}],
                    },
                    "events": {},
                }
            ],
        }
    }


def _result(payload: dict) -> dict:
    return payload["chart"]["result"][0]


def _series(payload: dict, column: str) -> list:
    indicators = _result(payload)["indicators"]
    if column == "adjusted_close":
        return indicators["adjclose"][0]["adjclose"]
    return indicators["quote"][0][column]


def _bytes(payload: dict) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def _recover(downloader: ModuleType, payload: dict, raw_path: Path, **kwargs):
    prices, actions, metadata = downloader._parse_chart("SPY", _bytes(payload))
    return downloader._recover_latest_session(
        "SPY",
        prices,
        actions,
        metadata,
        as_of=AS_OF,
        timeout=15,
        raw_path=raw_path,
        now=kwargs.pop("now", AFTER_CLOSE),
        **kwargs,
    )


def test_complete_history_does_not_request_or_write_fallback(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    download = Mock(side_effect=AssertionError("Complete history needs no fallback"))
    monkeypatch.setattr(downloader, "_download_latest_session", download)
    payload = _payload()
    expected, _, _ = downloader._parse_chart("SPY", _bytes(payload))
    actual, audit = _recover(downloader, payload, tmp_path / "daily.json")
    pd.testing.assert_frame_equal(actual, expected)
    assert audit is None
    download.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("missing_field", PRICE_COLUMNS)
def test_only_missing_latest_cell_is_filled_from_saved_daily_response(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_field: str,
) -> None:
    original = _payload()
    # Historical adjustments and action records must survive recovery unchanged.
    _series(original, "adjusted_close")[0] = 109.75
    historical_event = {"date": _timestamp(date(2026, 9, 11)), "amount": 0.25}
    _result(original)["events"] = {"dividends": {"historical": historical_event}}
    expected, _, _ = downloader._parse_chart("SPY", _bytes(original))
    _series(original, missing_field)[-1] = None
    before = deepcopy(original)
    fallback = _bytes(_payload((AS_OF,)))
    download = Mock(return_value=fallback)
    monkeypatch.setattr(downloader, "_download_latest_session", download)
    raw_path = tmp_path / "SPY.latest-session.json"
    prices, actions, metadata = downloader._parse_chart("SPY", _bytes(original))
    prices_before = prices.copy(deep=True)
    actions_before = actions.copy(deep=True)

    actual, audit = downloader._recover_latest_session(
        "SPY",
        prices,
        actions,
        metadata,
        as_of=AS_OF,
        timeout=15,
        raw_path=raw_path,
        now=AFTER_CLOSE,
    )

    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    pd.testing.assert_frame_equal(prices, prices_before)
    pd.testing.assert_frame_equal(actions, actions_before)
    assert original == before
    download.assert_called_once_with("SPY", 15)
    assert raw_path.read_bytes() == fallback
    assert audit is not None
    assert str(audit["date"]) == AS_OF.isoformat()
    assert audit["missing_fields"] == [missing_field]
    assert str(audit["raw_file"]) == str(raw_path)
    assert "range=1d" in audit["source_url"]
    assert "interval=1d" in audit["source_url"]


@pytest.mark.parametrize("historical_index", [0, 1])
def test_historical_missing_prices_are_not_repaired(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    historical_index: int,
) -> None:
    payload = _payload()
    _series(payload, "adjusted_close")[historical_index] = None
    _series(payload, "adjusted_close")[-1] = None
    download = Mock()
    monkeypatch.setattr(downloader, "_download_latest_session", download)
    with pytest.raises(RuntimeError):
        _recover(downloader, payload, tmp_path / "daily.json")
    download.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_duplicate_dates_fail_before_daily_fallback(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _payload((date(2026, 9, 11), AS_OF, AS_OF))
    _series(payload, "adjusted_close")[-1] = None
    download = Mock()
    monkeypatch.setattr(downloader, "_download_latest_session", download)
    with pytest.raises(RuntimeError):
        _recover(downloader, payload, tmp_path / "daily.json")
    download.assert_not_called()


def test_missing_previous_session_is_not_replaced_with_requested_session(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _payload((date(2026, 9, 10), date(2026, 9, 11)))
    _series(payload, "adjusted_close")[-1] = None
    download = Mock()
    monkeypatch.setattr(downloader, "_download_latest_session", download)
    with pytest.raises(RuntimeError):
        _recover(downloader, payload, tmp_path / "daily.json")
    download.assert_not_called()


@pytest.mark.parametrize("field", PRICE_COLUMNS)
def test_incomplete_daily_fallback_fails_and_preserves_raw_response(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    payload = _payload()
    _series(payload, "close")[-1] = None
    _series(payload, "adjusted_close")[-1] = None
    daily = _payload((AS_OF,))
    _series(daily, field)[0] = None
    # A tempting real-time metadata value must never substitute for a daily observation.
    _result(daily)["meta"]["regularMarketPrice"] = 114.0
    content = _bytes(daily)
    download = Mock(return_value=content)
    monkeypatch.setattr(downloader, "_download_latest_session", download)
    raw_path = tmp_path / "daily.json"
    with pytest.raises(RuntimeError):
        _recover(downloader, payload, raw_path)
    download.assert_called_once_with("SPY", 15)
    assert raw_path.read_bytes() == content


@pytest.mark.parametrize("bad_value", [0, -1, float("inf"), float("nan")])
def test_non_positive_or_non_finite_fallback_prices_are_rejected(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bad_value: float,
) -> None:
    payload = _payload()
    _series(payload, "adjusted_close")[-1] = None
    daily = _payload((AS_OF,))
    _series(daily, "adjusted_close")[0] = bad_value
    monkeypatch.setattr(downloader, "_download_latest_session", Mock(return_value=_bytes(daily)))
    with pytest.raises(RuntimeError):
        _recover(downloader, payload, tmp_path / "daily.json")


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume"])
def test_conflicting_existing_observations_are_rejected(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    payload = _payload()
    _series(payload, "adjusted_close")[-1] = None
    daily = _payload((AS_OF,))
    _series(daily, field)[0] += 1
    monkeypatch.setattr(downloader, "_download_latest_session", Mock(return_value=_bytes(daily)))
    with pytest.raises(RuntimeError):
        _recover(downloader, payload, tmp_path / "daily.json")


@pytest.mark.parametrize(
    "key,value",
    [
        ("symbol", "IEF"),
        ("currency", "EUR"),
        ("exchangeName", "NMS"),
        ("exchangeTimezoneName", "Europe/London"),
        ("instrumentType", "EQUITY"),
        ("dataGranularity", "5m"),
    ],
)
def test_fallback_identity_and_daily_granularity_must_match(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    key: str,
    value: str,
) -> None:
    payload = _payload()
    _series(payload, "adjusted_close")[-1] = None
    daily = _payload((AS_OF,))
    _result(daily)["meta"][key] = value
    content = _bytes(daily)
    monkeypatch.setattr(downloader, "_download_latest_session", Mock(return_value=content))
    raw_path = tmp_path / "daily.json"
    with pytest.raises(RuntimeError):
        _recover(downloader, payload, raw_path)
    assert raw_path.read_bytes() == content


@pytest.mark.parametrize("days", [(date(2026, 9, 11),), (date(2026, 9, 15),), (AS_OF, AS_OF)])
def test_fallback_must_contain_exactly_the_requested_session(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    days: tuple[date, ...],
) -> None:
    payload = _payload()
    _series(payload, "adjusted_close")[-1] = None
    daily = _payload(days)
    monkeypatch.setattr(downloader, "_download_latest_session", Mock(return_value=_bytes(daily)))
    with pytest.raises(RuntimeError):
        _recover(downloader, payload, tmp_path / "daily.json")


@pytest.mark.parametrize("session_problem", ["still_open", "too_early", "wrong_session"])
def test_fallback_requires_a_settled_requested_session(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    session_problem: str,
) -> None:
    payload = _payload()
    _series(payload, "adjusted_close")[-1] = None
    daily = _payload((AS_OF,))
    now = AFTER_CLOSE
    if session_problem == "still_open":
        now = datetime(2026, 9, 14, 19, tzinfo=UTC)
    elif session_problem == "too_early":
        now = datetime(2026, 9, 14, 20, 29, 59, tzinfo=UTC)
    else:
        _result(daily)["meta"]["currentTradingPeriod"]["regular"]["end"] = _timestamp(
            date(2026, 9, 11), 20, 0
        )
    monkeypatch.setattr(downloader, "_download_latest_session", Mock(return_value=_bytes(daily)))
    with pytest.raises(RuntimeError):
        _recover(downloader, payload, tmp_path / "daily.json", now=now)


def test_exact_thirty_minute_boundary_accepts_daily_row_without_overwriting_close(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _payload()
    _series(payload, "adjusted_close")[-1] = None
    original_close = _series(payload, "close")[-1]
    daily = _payload((AS_OF,))
    # A value inside the comparison tolerance verifies that observed cells remain intact.
    _series(daily, "close")[0] = original_close + 0.000001
    monkeypatch.setattr(downloader, "_download_latest_session", Mock(return_value=_bytes(daily)))
    actual, audit = _recover(
        downloader,
        payload,
        tmp_path / "daily.json",
        now=datetime(2026, 9, 14, 20, 30, tzinfo=UTC),
    )
    assert actual.iloc[-1]["close"] == original_close
    assert actual.iloc[-1]["adjusted_close"] == 114.0
    assert audit is not None


@pytest.mark.parametrize("same_action", [True, False])
def test_latest_corporate_actions_must_agree(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    same_action: bool,
) -> None:
    payload = _payload()
    _series(payload, "adjusted_close")[-1] = None
    _result(payload)["events"] = {
        "dividends": {"history-key": {"date": _timestamp(AS_OF), "amount": 0.5}}
    }
    daily = _payload((AS_OF,))
    _result(daily)["events"] = {
        "dividends": {
            "different-key": {"date": _timestamp(AS_OF), "amount": 0.5 if same_action else 0.6}
        }
    }
    monkeypatch.setattr(downloader, "_download_latest_session", Mock(return_value=_bytes(daily)))
    if same_action:
        actual, audit = _recover(downloader, payload, tmp_path / "daily.json")
        assert actual["adjusted_close"].notna().all()
        assert audit is not None
    else:
        with pytest.raises(RuntimeError):
            _recover(downloader, payload, tmp_path / "daily.json")


def test_main_preserves_and_hashes_original_and_recovery_raw_responses(
    downloader: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = _payload()
    _series(original, "close")[-1] = None
    _series(original, "adjusted_close")[-1] = None
    original_content = _bytes(original)
    daily_content = _bytes(_payload((AS_OF,)))
    history_download = Mock(return_value=original_content)
    daily_download = Mock(return_value=daily_content)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return AFTER_CLOSE if tz is None else AFTER_CLOSE.astimezone(tz)

    monkeypatch.setattr(downloader, "datetime", FixedDateTime)
    monkeypatch.setattr(downloader, "TICKERS", ("SPY",))
    monkeypatch.setattr(downloader, "_download_one", history_download)
    monkeypatch.setattr(downloader, "_download_latest_session", daily_download)
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    manifest_path = tmp_path / "manifests" / "etfs.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_etf_history.py",
            "--start",
            "2026-09-10",
            "--as-of",
            AS_OF.isoformat(),
            "--workers",
            "1",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--manifest",
            str(manifest_path),
        ],
    )

    downloader.main()

    history_download.assert_called_once_with("SPY", date(2026, 9, 10), date(2026, 9, 15), 15)
    daily_download.assert_called_once_with("SPY", 15)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_entries = {Path(entry["path"]): entry for entry in manifest["raw_files"]}
    expected_raw = {
        raw_dir / "SPY.json": original_content,
        raw_dir / "SPY.latest_daily.json": daily_content,
    }
    assert set(raw_entries) == set(expected_raw)
    for path, content in expected_raw.items():
        assert path.read_bytes() == content
        assert raw_entries[path]["bytes"] == len(content)
        assert raw_entries[path]["sha256"] == hashlib.sha256(content).hexdigest()

    assert manifest["latest_observation"] == AS_OF.isoformat()
    assert manifest["price_rows"] == 3
    assert manifest["requested_as_of"] == AS_OF.isoformat()
    assert manifest["period_end_is_exclusive"] == "2026-09-15"
    assert len(manifest["latest_daily_recoveries"]) == 1
    assert manifest["latest_daily_recoveries"][0]["missing_fields"] == ["close", "adjusted_close"]

    prices = pd.read_csv(processed_dir / "etf_daily_prices_long.csv")
    assert prices["date"].tolist() == ["2026-09-10", "2026-09-11", "2026-09-14"]
    latest = prices.iloc[-1]
    assert latest["close"] == latest["adjusted_close"] == 114.0
    assert latest["total_return"] == pytest.approx(114.0 / 111.0 - 1)
    assert prices[list(PRICE_COLUMNS)].notna().all().all()
    quality = pd.read_csv(processed_dir / "etf_data_quality_summary.csv")
    assert quality["status"].tolist() == ["PASS"]
    assert quality["missing_adjusted_close"].tolist() == [0]
    for entry in manifest["processed_files"]:
        assert hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
