"""Download, validate, and publish the ETF history used by Model 01.

The command queries Yahoo Finance's keyless chart endpoint for the frozen ETF
universe, stores each raw JSON response, and produces analysis-ready long and
wide CSV tables plus a hash manifest and data-quality report. Its required inputs
are the date window and explicit raw, processed, and manifest paths supplied on
the command line.

Adjusted close is treated as the provider's split- and distribution-adjusted
total-return series. The script derives adjusted OHLC values, daily total returns,
monthly returns, and action checks; rejects duplicate, missing, non-positive, or
stale price histories; and reconciles ordinary cash-distribution returns within
a documented tolerance. Persistent outputs are written only to caller-provided
paths. This acquisition is reproducible as code, although Yahoo's mutable adjusted
history means a future download is not guaranteed to reproduce frozen bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TICKERS = (
    "SPY",
    "IEF",
    "TIP",
    "LQD",
    "HYG",
    "BIL",
    "GLD",
    "DBC",
    "UUP",
    "TLT",
    "USO",
    "AGG",
)
SOURCE_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
PRICE_COLUMNS = ("open", "high", "low", "close", "adjusted_close")


def _epoch(day: date) -> int:
    """Convert a UTC calendar date to Yahoo's Unix-epoch query boundary."""
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def _request_url(ticker: str, start: date, end_exclusive: date) -> str:
    """Build a daily chart URL including corporate-action event records."""
    query = urlencode(
        {
            "period1": _epoch(start),
            "period2": _epoch(end_exclusive),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    return f"{SOURCE_ENDPOINT.format(ticker=ticker)}?{query}"


def _download_one(ticker: str, start: date, end_exclusive: date, timeout: int) -> bytes:
    """Return one ticker's raw Yahoo chart response without writing it to disk."""
    request = Request(
        _request_url(ticker, start, end_exclusive),
        headers={"User-Agent": "Mozilla/5.0 (compatible; regime-allocation-research/1.0)"},
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{ticker}: HTTP {response.status}")
        return response.read()


def _latest_session_url(ticker: str) -> str:
    """Request the latest daily bar, whose publication can precede history updates."""
    query = urlencode(
        {"range": "1d", "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
    )
    return f"{SOURCE_ENDPOINT.format(ticker=ticker)}?{query}"


def _download_latest_session(ticker: str, timeout: int) -> bytes:
    """Retrieve a provider daily bar; never substitute an intraday quote."""
    request = Request(
        _latest_session_url(ticker),
        headers={"User-Agent": "Mozilla/5.0 (compatible; regime-allocation-research/1.0)"},
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{ticker}: HTTP {response.status}")
        return response.read()


def _exchange_dates(timestamps: list[int], exchange_timezone: str) -> pd.Series:
    """Map UTC response timestamps to normalized exchange-local session dates."""
    return pd.Series(
        pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(exchange_timezone).date,
        dtype="object",
    )


def _parse_actions(
    ticker: str,
    events: dict[str, Any],
    exchange_timezone: str,
) -> pd.DataFrame:
    """Normalize Yahoo dividend and split dictionaries into auditable rows."""
    records: list[dict[str, Any]] = []
    for item in events.get("dividends", {}).values():
        event_date = (
            pd.Timestamp(item["date"], unit="s", tz="UTC").tz_convert(exchange_timezone).date()
        )
        records.append(
            {
                "date": event_date,
                "ticker": ticker,
                "action_type": "dividend",
                "amount": float(item["amount"]),
                "numerator": np.nan,
                "denominator": np.nan,
                "split_ratio": "",
            }
        )
    for item in events.get("splits", {}).values():
        event_date = (
            pd.Timestamp(item["date"], unit="s", tz="UTC").tz_convert(exchange_timezone).date()
        )
        records.append(
            {
                "date": event_date,
                "ticker": ticker,
                "action_type": "split",
                "amount": np.nan,
                "numerator": float(item.get("numerator", np.nan)),
                "denominator": float(item.get("denominator", np.nan)),
                "split_ratio": str(item.get("splitRatio", "")),
            }
        )
    columns = (
        "date",
        "ticker",
        "action_type",
        "amount",
        "numerator",
        "denominator",
        "split_ratio",
    )
    return pd.DataFrame.from_records(records, columns=columns)


def _parse_chart(ticker: str, content: bytes) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Parse one raw chart response into prices, actions, and provider metadata."""
    payload = json.loads(content)
    chart = payload.get("chart", {})
    if chart.get("error") is not None:
        raise RuntimeError(f"{ticker}: provider error: {chart['error']}")
    results = chart.get("result") or []
    if len(results) != 1:
        raise RuntimeError(f"{ticker}: expected one chart result, found {len(results)}")

    result = results[0]
    meta = result["meta"]
    if meta.get("symbol") != ticker:
        raise RuntimeError(f"{ticker}: provider returned symbol {meta.get('symbol')!r}")
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose", [])
    if not timestamps or len(adjclose) != len(timestamps):
        raise RuntimeError(f"{ticker}: incomplete timestamp/adjusted-close arrays")

    exchange_timezone = meta.get("exchangeTimezoneName", "America/New_York")
    frame = pd.DataFrame(
        {
            "date": _exchange_dates(timestamps, exchange_timezone),
            "ticker": ticker,
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "adjusted_close": adjclose,
            "volume": quote.get("volume", []),
        }
    )
    frame["currency"] = meta.get("currency", "")
    frame["exchange"] = meta.get("exchangeName", "")
    frame["exchange_timezone"] = exchange_timezone
    actions = _parse_actions(ticker, result.get("events", {}), exchange_timezone)
    metadata = {
        "ticker": ticker,
        "currency": meta.get("currency"),
        "exchange": meta.get("exchangeName"),
        "full_exchange_name": meta.get("fullExchangeName"),
        "instrument_type": meta.get("instrumentType"),
        "exchange_timezone": exchange_timezone,
        "data_granularity": meta.get("dataGranularity"),
    }
    return frame, actions, metadata


def _sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a generated artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recover_latest_session(
    ticker: str,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    as_of: date,
    timeout: int,
    raw_path: Path,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Fill a delayed history bar only from a matching, completed provider daily bar.

    Yahoo can publish the latest daily close in its one-day response before it
    appears in multi-day history. Preserve the original history and actions,
    save the recovery response for audit, and fill only missing price cells.
    Historical gaps, conflicting responses, and incomplete sessions remain errors.
    """
    dates = pd.to_datetime(prices["date"]).dt.date
    if dates.duplicated().any():
        raise RuntimeError(f"{ticker}: duplicate daily session dates")
    numeric = prices[list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    missing = numeric.isna()
    if not missing.any().any():
        return prices, None
    missing_dates = dates.loc[missing.any(axis=1)]
    if len(missing_dates) != 1 or missing_dates.iloc[0] != as_of or dates.max() != as_of:
        raise RuntimeError(
            f"{ticker}: missing historical OHLC/adjusted close values on "
            f"{[str(day) for day in missing_dates]}; only the requested latest session can recover"
        )

    content = _download_latest_session(ticker, timeout)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    latest, latest_actions, latest_meta = _parse_chart(ticker, content)
    for key in ("currency", "exchange", "exchange_timezone", "instrument_type", "data_granularity"):
        if latest_meta.get(key) != metadata.get(key):
            raise RuntimeError(f"{ticker}: daily recovery has conflicting {key}")
    if latest_meta["data_granularity"] != "1d":
        raise RuntimeError(f"{ticker}: recovery response is not daily history")
    latest_dates = pd.to_datetime(latest["date"]).dt.date
    if len(latest) != 1 or latest_dates.iloc[0] != as_of:
        raise RuntimeError(f"{ticker}: daily recovery does not match requested session {as_of}")

    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("daily recovery now must be timezone aware")
    try:
        provider_meta = json.loads(content)["chart"]["result"][0]["meta"]
        session_end = datetime.fromtimestamp(
            provider_meta["currentTradingPeriod"]["regular"]["end"], tz=UTC
        )
        session_date = session_end.astimezone(ZoneInfo(latest_meta["exchange_timezone"])).date()
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"{ticker}: daily recovery is missing valid session-end metadata"
        ) from exc
    if session_date != as_of or current < session_end + timedelta(minutes=30):
        raise RuntimeError(f"{ticker}: daily recovery session {as_of} is not safely completed")

    recovered = latest[list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce").iloc[0]
    if not np.isfinite(recovered.to_numpy(dtype=float)).all() or recovered.le(0).any():
        raise RuntimeError(
            f"{ticker}: daily recovery still has missing or invalid prices for {as_of}"
        )
    row_index = missing_dates.index[0]
    for column in (*PRICE_COLUMNS, "volume"):
        original = pd.to_numeric(prices.loc[row_index, column], errors="coerce")
        replacement = pd.to_numeric(latest.iloc[0][column], errors="coerce")
        if pd.notna(original) and (
            not np.isfinite(original)
            or not np.isfinite(replacement)
            or not np.isclose(original, replacement, rtol=1e-7, atol=1e-7)
        ):
            raise RuntimeError(
                f"{ticker}: daily recovery conflicts with observed {column} on {as_of}"
            )

    # Mixing revised corporate actions with unchanged adjusted history is unsafe.
    def session_actions(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        return (
            frame.loc[frame["date"].dt.date.eq(as_of)]
            .sort_values(["action_type", "amount", "split_ratio"])
            .reset_index(drop=True)
        )

    try:
        pd.testing.assert_frame_equal(
            session_actions(actions),
            session_actions(latest_actions),
            check_dtype=False,
            check_exact=False,
            rtol=1e-7,
            atol=1e-7,
        )
    except AssertionError as exc:
        raise RuntimeError(f"{ticker}: daily recovery has conflicting corporate actions") from exc

    fields = [column for column in PRICE_COLUMNS if missing.loc[row_index, column]]
    repaired = prices.copy()
    for column in fields:
        repaired.loc[row_index, column] = recovered[column]
    return repaired, {
        "ticker": ticker,
        "date": as_of.isoformat(),
        "missing_fields": fields,
        "source_url": _latest_session_url(ticker),
        "raw_file": str(raw_path),
        "regular_session_end_utc": session_end.isoformat(),
        "reason": "Latest daily bar published before complete multi-day history",
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV using the pipeline's stable date and float formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.10g")


def _validate_and_enrich(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    requested_start: date,
    as_of: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate raw price history and derive total-return fields and diagnostics.

    Returns the enriched long-form price table, one quality-summary row per ETF,
    and an aggregate adjusted-close versus cash-distribution reconciliation.
    Validation errors raise ``RuntimeError`` before any processed table is used.
    """
    numeric = ("open", "high", "low", "close", "adjusted_close", "volume")
    prices.loc[:, list(numeric)] = prices.loc[:, list(numeric)].apply(
        pd.to_numeric, errors="coerce"
    )
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    prices = prices.loc[
        (prices["date"].dt.date >= requested_start) & (prices["date"].dt.date <= as_of)
    ].copy()

    duplicate_count = int(prices.duplicated(["ticker", "date"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Found {duplicate_count} duplicate ticker-date rows")
    if set(prices["ticker"].unique()) != set(TICKERS):
        raise RuntimeError("Not every requested ticker was returned")
    if prices[list(numeric[:-1])].isna().any().any():
        bad = prices.loc[prices[list(numeric[:-1])].isna().any(axis=1), ["ticker", "date"]]
        raise RuntimeError(f"Missing OHLC/adjusted close values: {bad.head().to_dict('records')}")
    if not np.isfinite(prices[list(PRICE_COLUMNS)].to_numpy(dtype=float)).all():
        raise RuntimeError("Found non-finite prices")
    if (prices[["open", "high", "low", "close", "adjusted_close"]] <= 0).any().any():
        raise RuntimeError("Found non-positive prices")
    if (prices["volume"].dropna() < 0).any():
        raise RuntimeError("Found negative volume")

    prices["adjustment_factor"] = prices["adjusted_close"] / prices["close"]
    for column in ("open", "high", "low"):
        prices[f"adjusted_{column}"] = prices[column] * prices["adjustment_factor"]
    prices["total_return"] = prices.groupby("ticker", sort=False)["adjusted_close"].pct_change(
        fill_method=None
    )
    prices["total_return_index"] = (
        100.0
        * prices["adjusted_close"]
        / prices.groupby("ticker", sort=False)["adjusted_close"].transform("first")
    )

    actions = actions.copy()
    if not actions.empty:
        actions["date"] = pd.to_datetime(actions["date"])
    dividends = (
        actions.loc[actions["action_type"].eq("dividend")]
        .groupby(["ticker", "date"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "cash_distribution"})
    )
    splits = actions.loc[actions["action_type"].eq("split")].copy()
    if not splits.empty:
        splits["split_new_per_old"] = splits["numerator"] / splits["denominator"]
        splits = splits[["ticker", "date", "split_new_per_old"]]
    else:
        splits = pd.DataFrame(columns=["ticker", "date", "split_new_per_old"])
    prices = prices.merge(dividends, on=["ticker", "date"], how="left")
    prices = prices.merge(splits, on=["ticker", "date"], how="left")
    prices["cash_distribution"] = prices["cash_distribution"].fillna(0.0)
    prices["split_new_per_old"] = prices["split_new_per_old"].fillna(1.0)

    tolerance = 1e-7
    ohlc_violation = (prices["high"] + tolerance < prices[["open", "close", "low"]].max(axis=1)) | (
        prices["low"] - tolerance > prices[["open", "close", "high"]].min(axis=1)
    )
    union_dates = pd.Index(prices["date"].unique()).sort_values()
    summaries: list[dict[str, Any]] = []
    for ticker, group in prices.groupby("ticker", sort=False):
        group = group.sort_values("date")
        ticker_actions = actions.loc[actions["ticker"].eq(ticker)]
        first_date = group["date"].min().date()
        last_date = group["date"].max().date()
        status = "PASS"
        notes: list[str] = []
        if first_date > requested_start + timedelta(days=7):
            status = "FAIL"
            notes.append("late start")
        if (as_of - last_date).days > 7:
            status = "FAIL"
            notes.append("stale endpoint")
        ticker_ohlc = int(ohlc_violation.loc[group.index].sum())
        if ticker_ohlc:
            status = "FAIL"
            notes.append("OHLC violation")
        max_abs_return = float(group["total_return"].abs().max())
        if max_abs_return > 0.50:
            status = "REVIEW" if status == "PASS" else status
            notes.append("daily return > 50%")
        summaries.append(
            {
                "ticker": ticker,
                "first_date": first_date,
                "last_date": last_date,
                "observations": len(group),
                "union_trading_days": len(union_dates),
                "missing_vs_union": len(union_dates.difference(group["date"])),
                "dividend_events": int(ticker_actions["action_type"].eq("dividend").sum()),
                "split_events": int(ticker_actions["action_type"].eq("split").sum()),
                "min_adjusted_close": float(group["adjusted_close"].min()),
                "max_adjusted_close": float(group["adjusted_close"].max()),
                "min_daily_total_return": float(group["total_return"].min()),
                "max_daily_total_return": float(group["total_return"].max()),
                "max_abs_daily_total_return": max_abs_return,
                "missing_adjusted_close": int(group["adjusted_close"].isna().sum()),
                "duplicate_dates": int(group.duplicated("date").sum()),
                "ohlc_violations": ticker_ohlc,
                "stale_calendar_days": int((as_of - last_date).days),
                "status": status,
                "notes": "; ".join(notes),
            }
        )
    quality = pd.DataFrame(summaries)
    if quality["status"].eq("FAIL").any():
        failed = quality.loc[quality["status"].eq("FAIL"), ["ticker", "notes"]]
        raise RuntimeError(f"Data validation failed: {failed.to_dict('records')}")

    # On ordinary cash-distribution dates, adjusted-close returns should agree
    # closely with (close + distribution) / previous close - 1.
    previous_close = prices.groupby("ticker", sort=False)["close"].shift(1)
    synthetic_distribution_return = (
        prices["close"] + prices["cash_distribution"]
    ) / previous_close - 1.0
    distribution_mask = prices["cash_distribution"].gt(0) & prices["split_new_per_old"].eq(1)
    distribution_differences = (
        prices.loc[distribution_mask, "total_return"]
        - synthetic_distribution_return.loc[distribution_mask]
    ).abs()
    distribution_check = {
        "events_compared": int(distribution_mask.sum()),
        "max_absolute_return_difference": (
            float(distribution_differences.max()) if len(distribution_differences) else None
        ),
        "differences_over_50bp": int(distribution_differences.gt(0.005).sum()),
    }
    if distribution_check["differences_over_50bp"]:
        raise RuntimeError(
            "Adjusted-close dividend validation exceeded 50 basis points on "
            f"{distribution_check['differences_over_50bp']} events"
        )

    ordered_columns = (
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
        "adjustment_factor",
        "volume",
        "cash_distribution",
        "split_new_per_old",
        "total_return",
        "total_return_index",
        "currency",
        "exchange",
        "exchange_timezone",
    )
    return prices.loc[:, ordered_columns], quality, distribution_check


def main() -> None:
    """Acquire the frozen ETF universe and write validated data plus its manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=date(2008, 1, 1))
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),  # noqa: DTZ011 - caller-local default; weekly CLI supplies this explicitly
    )
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    end_exclusive = args.as_of + timedelta(days=1)
    downloaded: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_download_one, ticker, args.start, end_exclusive, args.timeout): ticker
            for ticker in TICKERS
        }
        for future in as_completed(futures):
            ticker = futures[future]
            downloaded[ticker] = future.result()
            print(f"downloaded {ticker}: {len(downloaded[ticker]):,} bytes", flush=True)
    if set(downloaded) != set(TICKERS):
        raise RuntimeError("Download stopped before all requested tickers completed")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    price_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    provider_metadata: list[dict[str, Any]] = []
    raw_files: list[Path] = []
    daily_recoveries: list[dict[str, Any]] = []
    for ticker in TICKERS:
        raw_path = args.raw_dir / f"{ticker}.json"
        raw_path.write_bytes(downloaded[ticker])
        raw_files.append(raw_path)
        prices, actions, metadata = _parse_chart(ticker, downloaded[ticker])
        prices = prices.loc[prices["date"].ge(args.start) & prices["date"].le(args.as_of)].copy()
        recovery_path = args.raw_dir / f"{ticker}.latest_daily.json"
        prices, recovery = _recover_latest_session(
            ticker,
            prices,
            actions,
            metadata,
            as_of=args.as_of,
            timeout=args.timeout,
            raw_path=recovery_path,
        )
        if recovery is not None:
            raw_files.append(recovery_path)
            daily_recoveries.append(recovery)
            print(
                f"recovered {ticker} {args.as_of}: {', '.join(recovery['missing_fields'])} "
                "from matching completed daily response",
                flush=True,
            )
        price_frames.append(prices)
        action_frames.append(actions)
        provider_metadata.append(metadata)

    prices = pd.concat(price_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)
    prices, quality, distribution_check = _validate_and_enrich(
        prices,
        actions,
        requested_start=args.start,
        as_of=args.as_of,
    )
    actions = actions.sort_values(["date", "ticker", "action_type"], kind="stable")

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    prices_path = args.processed_dir / "etf_daily_prices_long.csv"
    actions_path = args.processed_dir / "etf_corporate_actions.csv"
    quality_path = args.processed_dir / "etf_data_quality_summary.csv"
    adjusted_wide_path = args.processed_dir / "etf_daily_adjusted_close_wide.csv"
    returns_wide_path = args.processed_dir / "etf_daily_total_returns_wide.csv"
    monthly_returns_path = args.processed_dir / "etf_monthly_total_returns_wide.csv"

    _write_csv(prices, prices_path)
    _write_csv(actions, actions_path)
    _write_csv(quality, quality_path)

    adjusted_wide = prices.pivot(index="date", columns="ticker", values="adjusted_close")
    adjusted_wide = adjusted_wide.reindex(columns=TICKERS).reset_index()
    _write_csv(adjusted_wide, adjusted_wide_path)
    returns_wide = prices.pivot(index="date", columns="ticker", values="total_return")
    returns_wide = returns_wide.reindex(columns=TICKERS).reset_index()
    _write_csv(returns_wide, returns_wide_path)

    completed_month_cutoff = pd.Timestamp(args.as_of).to_period("M") - 1
    monthly_prices = prices.copy()
    monthly_prices["month"] = monthly_prices["date"].dt.to_period("M")
    monthly_prices = monthly_prices.loc[monthly_prices["month"].le(completed_month_cutoff)]
    monthly_prices = (
        monthly_prices.sort_values("date")
        .groupby(["ticker", "month"], as_index=False)
        .tail(1)
        .sort_values(["ticker", "month"])
    )
    monthly_prices["monthly_total_return"] = monthly_prices.groupby("ticker", sort=False)[
        "adjusted_close"
    ].pct_change(fill_method=None)
    monthly_wide = monthly_prices.pivot(
        index="month", columns="ticker", values="monthly_total_return"
    ).reindex(columns=TICKERS)
    monthly_wide.index = monthly_wide.index.astype(str)
    monthly_wide = monthly_wide.reset_index()
    _write_csv(monthly_wide, monthly_returns_path)

    output_files = (
        prices_path,
        actions_path,
        quality_path,
        adjusted_wide_path,
        returns_wide_path,
        monthly_returns_path,
    )
    manifest = {
        "dataset": "portfolio_market_data",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "source": "Yahoo Finance chart API",
        "source_url_template": SOURCE_ENDPOINT,
        "requested_start": args.start.isoformat(),
        "requested_as_of": args.as_of.isoformat(),
        "period_end_is_exclusive": end_exclusive.isoformat(),
        "tickers": list(TICKERS),
        "return_definition": "adjusted_close_t / adjusted_close_t-1 - 1",
        "monthly_returns": "completed calendar months only",
        "latest_observation": prices["date"].max().date().isoformat(),
        "price_rows": len(prices),
        "corporate_action_rows": len(actions),
        "provider_metadata": provider_metadata,
        "latest_daily_recoveries": daily_recoveries,
        "distribution_adjustment_check": distribution_check,
        "raw_files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in raw_files
        ],
        "processed_files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
        "caveats": [
            (
                "Adjusted values are back-adjusted by the provider and may change "
                "after future distributions or corrections."
            ),
            (
                "No missing daily returns were forward-filled; portfolio calendar "
                "alignment is deferred to the backtest."
            ),
            "Monthly returns exclude the current partial calendar month.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {len(prices):,} daily rows through {manifest['latest_observation']}")
    print(f"wrote {len(actions):,} corporate actions")
    print(f"dividend return check: {distribution_check}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
