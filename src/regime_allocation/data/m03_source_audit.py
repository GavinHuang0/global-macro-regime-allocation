"""Reproducible M03 source acquisition and audits, isolated from promoted models."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import pandas as pd

from regime_allocation.data.providers import select_vintage_provider
from regime_allocation.data.providers.vintage_matrix import (
    load_vintage_matrix,
    vintage_date_from_column,
)
from regime_allocation.data.source_audits import (
    compare_feature_ledgers,
    compare_observation_ledgers,
    ppi_splice_audit,
)
from regime_allocation.data.source_availability import (
    extract_source_features,
    extract_source_observations,
)
from regime_allocation.data.source_price_audit import (
    compare_yahoo_snapshots,
    reconcile_yahoo_chart,
)
from regime_allocation.data.source_registry import load_source_registry
from regime_allocation.data.source_snapshots import SnapshotStore

LOGGER = logging.getLogger(__name__)
DEFAULT_REGISTRY = Path("configs/data/m03_sources.yaml")
STORE = Path("data/raw/m03_sources")


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _resolve(root: Path, path: Path) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("source audit paths must stay within the project")
    return resolved


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _summary_markdown(report: dict, as_of: datetime) -> bytes:
    rows = [
        "# M03 source audit",
        "",
        f"As of **{as_of.isoformat()}**; availability mode: **{report['availability_mode']}**.",
        "",
        "This is a source-data audit, not a fitted model or allocation result.",
        "",
        "## Macro coverage",
        "",
        "| Source | Retained / requested | First retained reference | Last retained reference |",
        "|---|---:|---|---|",
    ]
    for item in report["coverage"]:
        rows.append(
            f"| {item['series_id']} | {item['retained_rows']} / {item['requested_rows']} | "
            f"{item['first_retained_reference'] or 'None'} | "
            f"{item['last_retained_reference'] or 'None'} |"
        )
    rows += [
        "",
        (
            "Every excluded reference has a reason in the CSV ledgers. Score entries are "
            "unscaled same-vintage changes; evidence entries are archive first-observed levels."
        ),
        "",
        "## PPI splice",
        "",
    ]
    splice = report["ppi_splice"]
    rows.append(
        f"Common reference months before applying the splice: **{splice['common_months']}**. "
        f"Maximum absolute change difference: **{splice['max_absolute_difference']}**."
    )
    rows += ["", "Matching overlap is a diagnostic, not proof of source equivalence."]
    for key, label in (
        ("previous_snapshot_comparison", "Score source changes"),
        ("previous_observation_comparison", "Evidence source changes"),
    ):
        if key in report:
            item = report[key]
            rows += [
                "",
                f"## {label}",
                "",
                (
                    f"At the same cutoff and reference window: {item['common_retained_rows']} "
                    f"common rows; {len(item['added_coverage'])} added; "
                    f"{len(item['lost_coverage'])} lost; "
                    f"{len(item['changed_values_on_common_support'])} value changes; "
                    f"{len(item['changed_availability_on_common_support'])} availability changes."
                ),
            ]
    rows += ["", "## Price and corporate-action checks", ""]
    if report["price_action_audits"]:
        rows += [
            "| Ticker | Status | Compared intervals | Discrepancies | Invalid price rows | Unreconciled actions |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for item in report["price_action_audits"]:
            summary = item.get("summary", {})
            rows.append(
                f"| {item['ticker']} | {item['status']} | "
                f"{summary.get('compared_intervals', 0)} | "
                f"{summary.get('discrepancy_intervals', 'N/A')} | "
                f"{summary.get('invalid_price_rows', 'N/A')} | "
                f"{summary.get('unreconciled_actions', 'N/A')} |"
            )
        failures = [
            (item["ticker"], issue["date"], issue["code"])
            for item in report["price_action_audits"]
            for issue in item.get("issues", [])
            if issue.get("source") == "price"
        ]
        if failures:
            rows += ["", "Flagged price rows:", ""]
            rows.extend(f"- {ticker}, {day}: `{reason}`." for ticker, day, reason in failures)
    else:
        rows.append("No price snapshots supplied.")
    rows += [
        "",
        (
            "These are independent calculations using Yahoo data, not independent-provider "
            "verification or proof of historical tradability. See source_audit.json for "
            "complete comparisons, exclusions, provenance, and limitations."
        ),
        "",
    ]
    return "\n".join(rows).encode()


def _load_matrix(content: bytes, series: str, content_format: str) -> pd.DataFrame:
    if content_format == "vintage_matrix_zip":
        # Preserve invalid nonmissing tokens for the row ledger. The shared
        # historical reader coerces them to NaN, which could wrongly select a
        # later revision as the first release in this experiment.
        with ZipFile(BytesIO(content)) as archive:
            name = next(
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and "readme" not in name.lower()
            )
            csv_content = archive.read(name)
        header = next(csv.reader(StringIO(csv_content.decode("utf-8-sig"))))
        if len(header) != len(set(header)):
            raise ValueError("duplicate vintage CSV headers")
        normalized = load_vintage_matrix(content, series)
        if set(header[1:]) != set(normalized.columns):
            raise ValueError("unexpected vintage CSV columns")
        raw = pd.read_csv(
            BytesIO(csv_content), dtype=object, keep_default_na=False, na_values=["", "."]
        )
        raw.index = pd.to_datetime(raw.iloc[:, 0])
        return raw.loc[:, normalized.columns].sort_index()
    if content_format != "first_release_csv":
        raise ValueError("unsupported source snapshot format")
    rows = pd.read_csv(BytesIO(content))
    if set(rows.columns) != {"reference_date", "release_date", "value"}:
        raise ValueError("unexpected first-release CSV schema")
    if rows["reference_date"].duplicated().any():
        raise ValueError("duplicate first-release reference date")
    rows["reference_date"] = pd.to_datetime(rows["reference_date"])
    rows["column"] = series + "_" + pd.to_datetime(rows["release_date"]).dt.strftime("%Y%m%d")
    return rows.pivot(index="reference_date", columns="column", values="value")


def _import_sources(root: Path, manifest: Path, sources: list[dict], store: SnapshotStore) -> dict:
    """Verify the existing manifest's hashes before preserving its legacy inputs."""
    payload = json.loads(_resolve(root, manifest).read_bytes())
    result = {}
    for source in sources:
        series = source["series_id"]
        candidates = [
            item
            for item in payload["acquired_files"]
            if item["series_id"] == series and item["purpose"] != "fixed_horizon_revisions"
        ]
        if len(candidates) != 1:
            raise ValueError(f"import requires exactly one original source for {series}")
        item = candidates[0]
        content = _resolve(root, Path(item["path"])).read_bytes()
        if _digest(content) != item["sha256"]:
            raise ValueError(f"import hash mismatch for {series}")
        if item["provider"] != "fred_api":
            raise ValueError("source audit requires authenticated FRED provenance")
        content_format = "first_release_csv" if series == "ICSA" else "vintage_matrix_zip"
        query = {
            key: item[key]
            for key in ("observation_start", "observation_end", "vintage_start", "vintage_end")
        }
        receipt = store.capture(
            content,
            provider_id=item["provider"],
            series_id=series,
            source_url=item["source_url"],
            query=query,
            retrieved_at=None,
            retrieval_time_status="unknown_legacy_cache",
            source_definition=source,
        )
        result[series] = {"receipt": receipt, "content_format": content_format}
    return result


def _download_sources(
    sources: list[dict],
    store: SnapshotStore,
    *,
    as_of: datetime,
    reference_end: date,
    client: object,
) -> dict:
    result = {}
    last_vintage = as_of.astimezone(ZoneInfo("America/New_York")).date() - timedelta(days=1)
    for source in sources:
        series = source["series_id"]
        vintage_end = (
            min(last_vintage, date.fromisoformat(source["vintage_end"]))
            if source.get("vintage_end")
            else last_vintage
        )
        query = {
            "observation_start": date.fromisoformat(source["observation_start"]),
            "observation_end": reference_end,
            "vintage_start": date.fromisoformat(source["vintage_start"]),
            "vintage_end": vintage_end,
            "release_id": source["release_id"],
        }
        if query["vintage_start"] > vintage_end:
            result[series] = {"unavailable_reason": "before_configured_archive_boundary"}
            continue
        LOGGER.info("Acquiring immutable source snapshot: %s", series)
        # No mutable provider cache is reused: observed retrieval means these
        # bytes were obtained in this acquisition, not inferred from file mtime.
        if series == "ICSA":
            artifact = client.list_first_release_observations(series, **query)
            rows = pd.DataFrame(
                [asdict(item) for item in artifact.observations],
                columns=["reference_date", "release_date", "value"],
            )
            content = rows.to_csv(index=False, lineterminator="\n").encode()
            content_format = "first_release_csv"
        else:
            artifact = client.download_level_matrix(series, **query)
            content, content_format = artifact.content, "vintage_matrix_zip"
        if artifact.provider_id != "fred_api":
            raise ValueError("source audit requires authenticated FRED provenance")
        if artifact.series_id != series:
            raise ValueError("provider returned a different source identifier")
        receipt = store.capture(
            content,
            provider_id=artifact.provider_id,
            series_id=series,
            source_url=artifact.source_url,
            query={
                key: value.isoformat() if isinstance(value, date) else value
                for key, value in query.items()
            },
            retrieved_at=datetime.now(UTC),
            retrieval_time_status="observed",
            source_definition=source,
        )
        result[series] = {"receipt": receipt, "content_format": content_format}
    return result


def _replay_sources(root: Path, manifest: Path, sources: list[dict], store: SnapshotStore) -> dict:
    previous = json.loads(_resolve(root, manifest).read_bytes())
    result = {}
    for source in sources:
        series = source["series_id"]
        item = previous["sources"][series]
        if "receipt" in item:
            store.read(item["receipt"])
            if item["receipt"]["source_definition"] != source:
                raise ValueError(f"registry definition changed for {series}; use a new acquisition")
        result[series] = item
    return result


def _retrieval_block(receipt: dict, as_of: datetime, mode: str) -> str | None:
    if mode == "captured":
        if receipt["retrieved_at"] is None:
            return "snapshot_retrieval_time_unknown"
        if datetime.fromisoformat(receipt["retrieved_at"]) > as_of:
            return "snapshot_not_retrieved_by_asof"
    return None


def _validate_query_bounds(matrix: pd.DataFrame, query: dict) -> None:
    if matrix.index.min() < pd.Timestamp(
        query["observation_start"]
    ) or matrix.index.max() > pd.Timestamp(query["observation_end"]):
        raise ValueError("source reference rows contradict acquisition query bounds")
    start, end = (
        date.fromisoformat(query["vintage_start"]),
        date.fromisoformat(query["vintage_end"]),
    )
    if any(not start <= vintage_date_from_column(str(column)) <= end for column in matrix.columns):
        raise ValueError("source vintage columns contradict acquisition query bounds")


def _query_coverage(ledger: pd.DataFrame, receipt: dict, as_of: datetime) -> None:
    """Distinguish missing archive values from periods not covered by the request."""
    query = receipt["query"]
    for name in ("observation_start", "observation_end", "vintage_start", "vintage_end"):
        ledger[f"acquired_{name}"] = query[name]
    outside = ledger["reference_date"].lt(pd.Timestamp(query["observation_start"])) | ledger[
        "reference_date"
    ].gt(pd.Timestamp(query["observation_end"]))
    outside &= ~ledger["reason"].isin(["outside_active_support", "future_reference"])
    ledger.loc[outside, "status"] = "excluded"
    ledger.loc[outside, "reason"] = "outside_acquired_observation_range"
    for field in ("current_value", "previous_value_as_of_release", "transformed_value"):
        ledger.loc[outside, field] = float("nan")
    for field in ("archive_vintage_date", "archive_available_at"):
        ledger.loc[outside, field] = pd.NaT
    last_eligible = as_of.astimezone(ZoneInfo("America/New_York")).date() - timedelta(days=1)
    if date.fromisoformat(query["vintage_end"]) < last_eligible:
        missing = ledger["reason"].isin(["absent_reference", "unavailable_by_asof"])
        ledger.loc[missing, "reason"] = "archive_query_ended_before_asof"


def _ledgers(
    sources: list[dict],
    acquired: dict,
    store: SnapshotStore,
    *,
    reference_start: date,
    reference_end: date,
    as_of: datetime,
    availability_mode: str,
    defaults: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    features, observations, ppi, boundaries = [], [], [], []
    for source in sources:
        series = source["series_id"]
        item = acquired[series]
        blocked = item.get("unavailable_reason")
        receipt = item.get("receipt")
        if receipt:
            blocked = blocked or _retrieval_block(receipt, as_of, availability_mode)
        matrix = (
            pd.DataFrame(index=pd.DatetimeIndex([]))
            if blocked
            else _load_matrix(store.read(receipt), series, item["content_format"])
        )
        if not blocked:
            _validate_query_bounds(matrix, receipt["query"])
        common = {
            "series_id": series,
            "component": source["component"],
            "reference_start": reference_start,
            "reference_end": reference_end,
            "as_of": as_of,
            "snapshot_id": receipt["content_sha256"] if receipt else "",
        }
        if source["role"] == "score":
            common.update(
                transform=source["transform"],
                max_release_lag_days=defaults["max_release_lag_days"],
                archive_start_latest_only=defaults["archive_start_latest_only"],
            )
            ledger = extract_source_features(
                matrix,
                **common,
                active_start=date.fromisoformat(source["active_start"])
                if source.get("active_start")
                else None,
                active_end=date.fromisoformat(source["active_end"])
                if source.get("active_end")
                else None,
            )
            if series in {"PPILFE", "WPSFD4131"}:
                ppi_ledger = extract_source_features(matrix, **common)
                if blocked:
                    ppi_ledger["reason"] = blocked
                else:
                    _query_coverage(ppi_ledger, receipt, as_of)
                ppi.append(ppi_ledger)
            features.append(ledger)
        else:
            ledger = extract_source_observations(
                matrix,
                **common,
                frequency="weekly_saturday" if series == "ICSA" else "monthly",
                active_start=date.fromisoformat(source["active_start"])
                if source.get("active_start")
                else None,
                active_end=date.fromisoformat(source["active_end"])
                if source.get("active_end")
                else None,
            )
            observations.append(ledger)
        if blocked:
            ledger["reason"] = blocked
        else:
            _query_coverage(ledger, receipt, as_of)
        ledger["retrieved_at"] = receipt["retrieved_at"] if receipt else None
        ledger["retrieval_time_status"] = (
            receipt["retrieval_time_status"] if receipt else "no_snapshot"
        )
        ledger["source_definition_version"] = source["source_definition_version"]
        ledger["information_available_at"] = ledger["archive_available_at"].where(
            ledger["status"].eq("retained")
        )
        if receipt and availability_mode == "captured" and not blocked:
            retrieved_at = pd.Timestamp(receipt["retrieved_at"])
            later_retrieval = ledger["information_available_at"].lt(retrieved_at)
            ledger.loc[later_retrieval, "information_available_at"] = retrieved_at
        available = ledger.loc[ledger["status"].eq("retained")]
        eligible_vintages = [
            vintage_date_from_column(str(column))
            for column in matrix.columns
            if pd.Timestamp(vintage_date_from_column(str(column)) + timedelta(days=1))
            .tz_localize("America/New_York")
            .tz_convert("UTC")
            <= as_of
        ]
        boundaries.append(
            {
                "series_id": series,
                "role": source["role"],
                "requested_rows": len(ledger),
                "retained_rows": len(available),
                "first_retained_reference": str(available["reference_date"].min().date())
                if len(available)
                else None,
                "last_retained_reference": str(available["reference_date"].max().date())
                if len(available)
                else None,
                "configured_vintage_start": source["vintage_start"],
                "first_visible_archive_vintage": min(eligible_vintages).isoformat()
                if eligible_vintages
                else None,
                "last_visible_archive_vintage": max(eligible_vintages).isoformat()
                if eligible_vintages
                else None,
                "acquisition_query": receipt["query"] if receipt else None,
                "verified_publication_timestamp": None,
                "reasons": {str(k): int(v) for k, v in ledger["reason"].value_counts().items()},
            }
        )
    feature_frame = pd.concat(features, ignore_index=True) if features else pd.DataFrame()
    observation_frame = (
        pd.concat(observations, ignore_index=True) if observations else pd.DataFrame()
    )
    ppi_frame = (
        pd.concat(ppi, ignore_index=True)
        if ppi
        else pd.DataFrame(columns=["series_id", "status", "reference_month"])
    )
    return feature_frame, observation_frame, [boundaries, ppi_splice_audit(ppi_frame)]


def build_source_audit(
    *,
    project_root: Path,
    output_dir: Path,
    as_of: datetime,
    reference_start: date = date(2000, 1, 1),
    reference_end: date | None = None,
    registry_path: Path = DEFAULT_REGISTRY,
    import_m02_manifest: Path | None = None,
    replay_manifest: Path | None = None,
    previous_manifest: Path | None = None,
    prices_dir: Path | None = None,
    series_ids: tuple[str, ...] | None = None,
    availability_mode: str = "archive",
    environ: dict | None = None,
) -> Path:
    """Build a new dated audit; existing runs and published results cannot be replaced."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must include a timezone")
    if availability_mode not in {"archive", "captured"}:
        raise ValueError("availability_mode must be archive or captured")
    if import_m02_manifest and replay_manifest:
        raise ValueError("choose import or replay, not both")
    root = project_root.resolve()
    output = _resolve(root, output_dir)
    if not output.is_relative_to(root / "outputs") or output == root / "outputs":
        raise ValueError("audit output must be a new child directory of outputs")
    if output.exists():
        raise ValueError("audit output already exists; choose a new run directory")
    local_day = as_of.astimezone(ZoneInfo("America/New_York")).date()
    reference_end = reference_end or (local_day.replace(day=1) - timedelta(days=1))
    if reference_start > reference_end:
        raise ValueError("reference_start must not follow reference_end")
    registry_file = _resolve(root, registry_path)
    registry = load_source_registry(registry_file)
    sources = registry["sources"]
    if series_ids is not None:
        unknown = set(series_ids) - {item["series_id"] for item in sources}
        if unknown or not series_ids or len(series_ids) != len(set(series_ids)):
            raise ValueError("series selection must be unique registered identifiers")
        sources = [item for item in sources if item["series_id"] in series_ids]
    client = (
        None
        if import_m02_manifest or replay_manifest
        else select_vintage_provider("fred", environ=environ).client
    )
    store = SnapshotStore(_resolve(root, STORE))
    if import_m02_manifest:
        acquired = _import_sources(root, import_m02_manifest, sources, store)
    elif replay_manifest:
        acquired = _replay_sources(root, replay_manifest, sources, store)
    else:
        acquired = _download_sources(
            sources, store, as_of=as_of, reference_end=reference_end, client=client
        )
    features, observations, (boundaries, splice) = _ledgers(
        sources,
        acquired,
        store,
        reference_start=reference_start,
        reference_end=reference_end,
        as_of=as_of,
        availability_mode=availability_mode,
        defaults=registry["defaults"],
    )
    report = {
        "schema_version": 1,
        "availability_mode": availability_mode,
        "archive_clock": "next_calendar_day_midnight_America/New_York",
        "publication_timestamps": "unverified; archive dates are not release timestamps",
        "price_clock": "completed local calendar dates strictly before as_of New York date",
        "scope": "Source audit and unscaled score changes; no fitted M03 or allocation",
        "coverage": boundaries,
        "ppi_splice": splice,
        "price_action_audits": [],
        "evidence_scope": "Native-frequency archive first-observed levels; no evidence transforms",
    }
    if previous_manifest:
        previous = json.loads(_resolve(root, previous_manifest).read_bytes())
        # Re-evaluate both snapshots with this run's exact cutoff/range/policy.
        # Different old/new run dates must not masquerade as changed coverage.
        old_sources = _replay_sources(root, previous_manifest, sources, store)
        old_features, old_observations, _ = _ledgers(
            sources,
            old_sources,
            store,
            reference_start=reference_start,
            reference_end=reference_end,
            as_of=as_of,
            availability_mode=availability_mode,
            defaults=registry["defaults"],
        )
        if not features.empty:
            report["previous_snapshot_comparison"] = compare_feature_ledgers(old_features, features)
        if not observations.empty:
            report["previous_observation_comparison"] = compare_observation_ledgers(
                old_observations, observations
            )
        report["previous_run_as_of"] = previous["as_of"]
    price_receipts = {}
    price_cutoff = local_day - timedelta(days=1)
    if prices_dir:
        price_files = sorted(_resolve(root, prices_dir).glob("*.json"))
        price_files = [path for path in price_files if "." not in path.stem]
        if not price_files:
            raise ValueError("prices_dir contains no TICKER.json snapshots")
        for path in price_files:
            ticker, content = path.stem, path.read_bytes()
            receipt = store.capture(
                content,
                provider_id="yahoo_chart",
                series_id=ticker,
                source_url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                query={"import_file": str(path.relative_to(root).as_posix())},
                retrieved_at=None,
                retrieval_time_status="unknown_legacy_cache",
                source_definition={
                    "source_definition_version": "yahoo_chart_m03_v1",
                    "definition": "Yahoo daily chart prices and corporate actions",
                    "independent_provider_verification": False,
                },
            )
            price_receipts[ticker] = receipt
    elif replay_manifest:
        price_receipts = json.loads(_resolve(root, replay_manifest).read_bytes()).get("prices", {})
    for ticker, receipt in price_receipts.items():
        content = store.read(receipt)
        if _retrieval_block(receipt, as_of, availability_mode):
            report["price_action_audits"].append(
                {
                    "ticker": ticker,
                    "status": "snapshot_not_demonstrably_retrieved_by_asof",
                }
            )
            continue
        report["price_action_audits"].append(
            reconcile_yahoo_chart(content, ticker=ticker, as_of=price_cutoff)
        )
        if previous_manifest and ticker in previous.get("prices", {}):
            old_receipt = previous["prices"][ticker]
            if _retrieval_block(old_receipt, as_of, availability_mode):
                report.setdefault("price_snapshot_comparisons", []).append(
                    {
                        "ticker": ticker,
                        "status": "previous_snapshot_not_retrieved_by_asof",
                    }
                )
                continue
            old_content = store.read(old_receipt)
            report.setdefault("price_snapshot_comparisons", []).append(
                compare_yahoo_snapshots(
                    old_content,
                    content,
                    ticker=ticker,
                    as_of=price_cutoff,
                )
            )
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, frame in (("feature_ledger.csv", features), ("observation_ledger.csv", observations)):
        if not frame.empty:
            content = frame.to_csv(index=False, lineterminator="\n").encode()
            (output / name).write_bytes(content)
            artifacts[name] = _digest(content)
    report_content = _json_bytes(report)
    (output / "source_audit.json").write_bytes(report_content)
    artifacts["source_audit.json"] = _digest(report_content)
    summary_content = _summary_markdown(report, as_of)
    (output / "summary.md").write_bytes(summary_content)
    artifacts["summary.md"] = _digest(summary_content)
    manifest = {
        "schema_version": 1,
        "stage_id": "m03_source_audit_v1",
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(),
        "reference_start": reference_start.isoformat(),
        "reference_end": reference_end.isoformat(),
        "availability_mode": availability_mode,
        "snapshot_store": STORE.as_posix(),
        "registry_sha256": _digest(registry_file.read_bytes()),
        "registry": registry,
        "sources": acquired,
        "prices": price_receipts,
        "artifacts": artifacts,
        "implementation_sha256": {
            path.name: _digest(path.read_bytes())
            for path in sorted(Path(__file__).parent.glob("source_*.py")) + [Path(__file__)]
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return manifest_path
