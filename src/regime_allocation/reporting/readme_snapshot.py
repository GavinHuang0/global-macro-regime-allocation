"""Validate and publish one dated Model 02 research snapshot in a README.

Schema version 1 deliberately separates the signal cutoff, actual input dates,
generation time, and expiry. Rerendering an old snapshot cannot make it current.
Only the content between the two explicit markers may be replaced.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

START_MARKER = "<!-- M02-LATEST:START -->"
END_MARKER = "<!-- M02-LATEST:END -->"
MODEL_VARIANT = "student_t_7_reduced_core"
RELEASE_ID = "m02_weekly_live_v1"
ASSETS = ("SPY", "IEF", "TIP", "HYG", "BIL", "GLD", "LQD")
QUADRANTS = {
    "growth_up_inflation_up": "Growth up / inflation up",
    "growth_down_inflation_up": "Growth down / inflation up",
    "growth_up_inflation_down": "Growth up / inflation down",
    "growth_down_inflation_down": "Growth down / inflation down",
}
NORMALIZATION_TOLERANCE = 1e-8
NEW_YORK = ZoneInfo("America/New_York")
_FIELDS = {
    "schema_version",
    "status",
    "model_variant",
    "release_id",
    "generated_at",
    "signal_week",
    "information_cutoff",
    "macro_as_of",
    "price_as_of",
    "valid_until",
    "data_freshness",
    "weights",
    "quadrant_posterior",
    "forecast_weekly_return",
    "estimated_annual_volatility",
    "optimization_outcome",
}


def _exact_fields(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")  # noqa: TRY004
    missing = fields - value.keys()
    extra = value.keys() - fields
    if missing or extra:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise ValueError(f"{name} must be a nonempty string of at most 1000 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _timestamp(value: object, name: str) -> datetime:
    text = _text(value, name)
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", text
    ):
        raise ValueError(f"{name} must be an ISO timestamp with an explicit timezone")
    try:
        return datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a valid ISO timestamp") from error


def _day(value: object, name: str) -> date:
    text = _text(value, name)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"{name} must be a YYYY-MM-DD date")
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a valid YYYY-MM-DD date") from error


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite JSON number")  # noqa: TRY004
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite JSON number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite JSON number")
    return numeric


def _distribution(value: object, labels: tuple[str, ...], name: str) -> None:
    distribution = _exact_fields(value, set(labels), name)
    values = [_number(distribution[label], f"{name}.{label}") for label in labels]
    if any(number < 0 or number > 1 for number in values):
        raise ValueError(f"{name} values must be between zero and one")
    if not math.isclose(math.fsum(values), 1.0, abs_tol=NORMALIZATION_TOLERANCE, rel_tol=0):
        raise ValueError(f"{name} must sum to one within {NORMALIZATION_TOLERANCE:g}")


def _clock(now: datetime | None) -> datetime:
    value = datetime.now(UTC) if now is None else now
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must have an explicit timezone")
    return value


def validate_snapshot(snapshot: object, *, now: datetime | None = None) -> Mapping[str, Any]:
    """Reject invalid schema, probabilities, provenance, and noncausal dates.

    Stale snapshots remain valid historical outputs; ``render_snapshot`` marks
    them stale. Dates without times are accepted only for actual input coverage.
    """
    data = _exact_fields(snapshot, _FIELDS, "snapshot")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for key, expected in (
        ("status", "research_only"),
        ("model_variant", MODEL_VARIANT),
        ("release_id", RELEASE_ID),
    ):
        if data[key] != expected:
            raise ValueError(f"{key} must be {expected!r}")
    current = _clock(now)
    generated = _timestamp(data["generated_at"], "generated_at")
    cutoff = _timestamp(data["information_cutoff"], "information_cutoff")
    expiry = _timestamp(data["valid_until"], "valid_until")
    week = _day(data["signal_week"], "signal_week")
    local_cutoff = cutoff.astimezone(NEW_YORK)
    if week.weekday() != 0:
        raise ValueError("signal_week must be a Monday")
    if local_cutoff.date() != week or local_cutoff.time().isoformat() != "00:00:00":
        raise ValueError("information_cutoff must be signal-week Monday 00:00 America/New_York")
    if expiry != local_cutoff + timedelta(days=7):
        raise ValueError("valid_until must be the next Monday 00:00 America/New_York")
    if generated > current:
        raise ValueError("generated_at must not be in the future")
    if generated < cutoff:
        raise ValueError("generated_at must not precede information_cutoff")
    for name in ("macro_as_of", "price_as_of"):
        value = data[name]
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            if _day(value, name) >= week:
                raise ValueError(f"{name} must precede the signal day")
        elif _timestamp(value, name) >= cutoff:
            raise ValueError(f"{name} must precede information_cutoff")
    freshness = _exact_fields(data["data_freshness"], {"status", "reason"}, "data_freshness")
    if freshness["status"] not in ("fresh", "stale"):
        raise ValueError("data_freshness.status must be 'fresh' or 'stale'")
    _text(freshness["reason"], "data_freshness.reason")
    _distribution(data["weights"], ASSETS, "weights")
    _distribution(data["quadrant_posterior"], tuple(QUADRANTS), "quadrant_posterior")
    if _number(data["forecast_weekly_return"], "forecast_weekly_return") < -1:
        raise ValueError("forecast_weekly_return must be at least -1")
    if _number(data["estimated_annual_volatility"], "estimated_annual_volatility") < 0:
        raise ValueError("estimated_annual_volatility must be nonnegative")
    _text(data["optimization_outcome"], "optimization_outcome")
    return data


def _escape(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+.!|~-])", r"\\\1", html.escape(value, quote=False))


def _format_timestamp(value: str) -> str:
    parsed = _timestamp(value, "timestamp")
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def render_snapshot(snapshot: object, *, now: datetime | None = None) -> str:
    """Render a validated snapshot with explicit input coverage and expiry."""
    current = _clock(now)
    data = validate_snapshot(snapshot, now=current)
    expired = current >= _timestamp(data["valid_until"], "valid_until")
    stale = expired or data["data_freshness"]["status"] == "stale"
    freshness = "STALE — refresh required" if stale else "FRESH at generation"
    reasons = [data["data_freshness"]["reason"]]
    if expired:
        reasons.append("The signal week has expired; this is a historical snapshot.")
    lines = [
        "## Model 02 weekly research snapshot",
        "",
        f"**Research only · {freshness}.** Signal week: **{data['signal_week']}**.",
        "",
        f"Data freshness: {_escape(' '.join(reasons))}",
        "",
        "| Snapshot timing | As of |",
        "|---|---|",
        f"| Information cutoff | `{_format_timestamp(data['information_cutoff'])}` |",
        f"| Latest macro release used | `{data['macro_as_of']}` |",
        f"| Final price close used | `{data['price_as_of']}` |",
        f"| Generated | `{_format_timestamp(data['generated_at'])}` |",
        f"| Valid until (exclusive) | `{_format_timestamp(data['valid_until'])}` |",
        "",
        "The cutoff is Monday 00:00 America/New_York; releases on the signal day are excluded.",
        "Price coverage records the final close used by estimation and pretrade holdings.",
        "",
        "| Asset | Research target |",
        "|---|---:|",
    ]
    lines.extend(f"| {asset} | {data['weights'][asset]:.1%} |" for asset in ASSETS)
    lines.extend(["", "| Macro quadrant | Posterior probability |", "|---|---:|"])
    lines.extend(
        f"| {label} | {data['quadrant_posterior'][key]:.1%} |" for key, label in QUADRANTS.items()
    )
    lines.extend(
        [
            "",
            "| Model estimate | Value |",
            "|---|---:|",
            f"| Forecast weekly return | {data['forecast_weekly_return']:+.2%} |",
            f"| Ex ante annualized volatility | {data['estimated_annual_volatility']:.2%} |",
            "",
            (
                f"Optimization outcome: **{_escape(data['optimization_outcome'])}**. "
                "Return and risk are model estimates, not realized performance or guaranteed outcomes."
            ),
            "",
            f"Model: `{MODEL_VARIANT}` · Release: `{RELEASE_ID}`.",
            "This research output is not investment advice.",
        ]
    )
    return "\n".join(lines)


def replace_snapshot_section(readme: str, rendered_snapshot: str) -> str:
    """Replace exactly one marked section, preserving every byte outside it."""
    if readme.count(START_MARKER) != 1 or readme.count(END_MARKER) != 1:
        raise ValueError("README must contain exactly one start marker and one end marker")
    start = readme.index(START_MARKER) + len(START_MARKER)
    end = readme.index(END_MARKER)
    if end < start:
        raise ValueError("README snapshot markers are out of order")
    if START_MARKER in rendered_snapshot or END_MARKER in rendered_snapshot:
        raise ValueError("rendered snapshot must not contain section markers")
    newline = "\r\n" if "\r\n" in readme else "\n"
    content = rendered_snapshot.replace("\r\n", "\n").replace("\n", newline).strip("\r\n")
    return readme[:start] + newline + newline + content + newline + newline + readme[end:]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_snapshot(path: str | Path, *, now: datetime | None = None) -> Mapping[str, Any]:
    """Load strict JSON, including rejection of duplicate keys and NaN/Infinity."""

    def invalid_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant: {value}")

    snapshot = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=invalid_constant,
    )
    return validate_snapshot(snapshot, now=now)


def update_readme_file(
    readme_path: str | Path,
    snapshot_path: str | Path,
    *,
    now: datetime | None = None,
    check: bool = False,
) -> bool:
    """Update only the marked section; return whether its contents differ.

    ``check=True`` performs all validation and comparison without writing.
    """
    current = _clock(now)
    rendered = render_snapshot(load_snapshot(snapshot_path, now=current), now=current)
    path = Path(readme_path)
    original = path.read_bytes()
    updated = replace_snapshot_section(original.decode("utf-8"), rendered).encode("utf-8")
    changed = updated != original
    if changed and not check:
        path.write_bytes(updated)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--check", action="store_true", help="Exit 1 if README needs updating")
    args = parser.parse_args(argv)
    try:
        changed = update_readme_file(args.readme, args.snapshot, check=args.check)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(
        "README snapshot differs"
        if changed and args.check
        else ("README snapshot updated" if changed else "README snapshot is unchanged")
    )
    return int(changed and args.check)


if __name__ == "__main__":
    raise SystemExit(main())
