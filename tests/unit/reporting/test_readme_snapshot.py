"""Prevent malformed or stale research output from silently replacing README content."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from regime_allocation.reporting.readme_snapshot import (
    ASSETS,
    END_MARKER,
    MODEL_VARIANT,
    RELEASE_ID,
    START_MARKER,
    load_snapshot,
    render_snapshot,
    replace_snapshot_section,
    update_readme_file,
    validate_snapshot,
)

NOW = datetime(2026, 9, 9, 16, tzinfo=UTC)


@pytest.fixture
def snapshot() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "research_only",
        "model_variant": MODEL_VARIANT,
        "release_id": RELEASE_ID,
        "generated_at": "2026-09-07T11:30:00Z",
        "signal_week": "2026-09-07",
        "information_cutoff": "2026-09-07T00:00:00-04:00",
        "macro_as_of": "2026-09-04",
        "price_as_of": "2026-09-04",
        "valid_until": "2026-09-14T00:00:00-04:00",
        "data_freshness": {"status": "fresh", "reason": "All input coverage checks passed."},
        "weights": {
            "SPY": 0.35,
            "IEF": 0.0,
            "TIP": 0.0,
            "HYG": 0.15,
            "BIL": 0.0,
            "GLD": 0.25,
            "LQD": 0.25,
        },
        "quadrant_posterior": {
            "growth_up_inflation_up": 0.25,
            "growth_down_inflation_up": 0.35,
            "growth_up_inflation_down": 0.20,
            "growth_down_inflation_down": 0.20,
        },
        "forecast_weekly_return": 0.00172,
        "estimated_annual_volatility": 0.0985,
        "optimization_outcome": "optimal",
    }


def test_render_shows_complete_allocation_dates_and_estimated_quantities(snapshot) -> None:
    rendered = render_snapshot(snapshot, now=NOW)

    assert "FRESH at generation" in rendered
    for asset in ASSETS:
        assert f"| {asset} |" in rendered
    assert "| BIL | 0.0% |" in rendered
    assert "| SPY | 35.0% |" in rendered
    assert "| Growth down / inflation up | 35.0% |" in rendered
    assert "| Forecast weekly return | +0.17% |" in rendered
    assert "| Ex ante annualized volatility | 9.85% |" in rendered
    assert "Latest macro release used | `2026-09-04`" in rendered
    assert "Final price close used | `2026-09-04`" in rendered
    assert "`2026-09-07T00:00:00-04:00`" in rendered
    assert "`2026-09-14T00:00:00-04:00`" in rendered
    assert "releases on the signal day are excluded" in rendered
    assert "Research only" in rendered
    assert "model estimates, not realized performance" in rendered


def test_expired_snapshot_cannot_be_relabelled_current_by_regeneration(snapshot) -> None:
    original = deepcopy(snapshot)
    expired = datetime(2026, 9, 14, 4, tzinfo=UTC)

    rendered = render_snapshot(snapshot, now=expired)

    assert "STALE — refresh required" in rendered
    assert "historical snapshot" in rendered
    assert "FRESH at generation" not in rendered
    assert "2026-09-07T11:30:00Z" in rendered
    assert snapshot == original


def test_known_stale_input_is_never_upgraded_before_expiry(snapshot) -> None:
    snapshot["data_freshness"] = {"status": "stale", "reason": "Price refresh failed."}

    assert "STALE — refresh required" in render_snapshot(snapshot, now=NOW)


def test_valid_until_tracks_new_york_dst_transition(snapshot) -> None:
    snapshot.update(
        {
            "generated_at": "2026-03-02T12:30:00Z",
            "signal_week": "2026-03-02",
            "information_cutoff": "2026-03-02T00:00:00-05:00",
            "macro_as_of": "2026-02-27T08:30:00-05:00",
            "price_as_of": "2026-02-27T16:00:00-05:00",
            "valid_until": "2026-03-09T00:00:00-04:00",
        }
    )
    just_before = datetime(2026, 3, 9, 3, 59, 59, tzinfo=UTC)

    assert "FRESH at generation" in render_snapshot(snapshot, now=just_before)
    assert "STALE" in render_snapshot(snapshot, now=datetime(2026, 3, 9, 4, tzinfo=UTC))
    snapshot["valid_until"] = "2026-03-09T00:00:00-05:00"
    with pytest.raises(ValueError, match="next Monday"):
        validate_snapshot(snapshot, now=just_before)


@pytest.mark.parametrize("field", ["weights", "quadrant_posterior"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -0.01, 1.01, True, "0.2"])
def test_distributions_reject_nonfinite_out_of_range_and_non_numeric(snapshot, field, bad_value):
    label = next(iter(snapshot[field]))
    snapshot[field][label] = bad_value

    with pytest.raises(ValueError, match=field):
        validate_snapshot(snapshot, now=NOW)


@pytest.mark.parametrize("field", ["weights", "quadrant_posterior"])
def test_distributions_require_exact_members_and_normalization(snapshot, field) -> None:
    original = deepcopy(snapshot[field])
    label = next(iter(original))
    snapshot[field][label] += 0.01
    with pytest.raises(ValueError, match="sum to one"):
        validate_snapshot(snapshot, now=NOW)

    snapshot[field] = original.copy()
    del snapshot[field][label]
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_snapshot(snapshot, now=NOW)

    snapshot[field] = {**original, "unknown": 0}
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_snapshot(snapshot, now=NOW)


def test_normalization_allows_only_small_float_roundoff(snapshot) -> None:
    snapshot["weights"]["SPY"] += 5e-9
    validate_snapshot(snapshot, now=NOW)

    snapshot["weights"]["SPY"] += 2e-8
    with pytest.raises(ValueError, match="sum to one"):
        validate_snapshot(snapshot, now=NOW)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema_version", True, "schema_version"),
        ("status", "live_trading", "status"),
        ("model_variant", "transition_only", "model_variant"),
        ("release_id", "old_release", "release_id"),
        ("forecast_weekly_return", float("nan"), "finite"),
        ("forecast_weekly_return", -1.1, "at least -1"),
        ("estimated_annual_volatility", -0.1, "nonnegative"),
        ("estimated_annual_volatility", True, "finite"),
        ("optimization_outcome", "", "nonempty"),
        ("optimization_outcome", "optimal\n| forged | row |", "control"),
        ("generated_at", "2026-09-10T00:00:00Z", "future"),
        ("generated_at", "2026-09-06T00:00:00Z", "precede information_cutoff"),
        ("generated_at", "2026-09-07T11:30:00", "explicit timezone"),
        ("signal_week", "2026-09-08", "Monday"),
        ("information_cutoff", "2026-09-07T00:00:00Z", "America/New_York"),
        ("valid_until", "2026-09-21T00:00:00-04:00", "next Monday"),
        ("macro_as_of", "2026-09-07", "precede the signal day"),
        ("price_as_of", "2026-09-10", "precede the signal day"),
        ("macro_as_of", "2026-09-07T08:30:00-04:00", "precede information_cutoff"),
        ("price_as_of", "2026-09-04T16:00:00", "explicit timezone"),
    ],
)
def test_rejects_invalid_provenance_timing_and_model_estimates(snapshot, field, value, match):
    snapshot[field] = value

    with pytest.raises(ValueError, match=match):
        validate_snapshot(snapshot, now=NOW)


def test_unknown_or_missing_schema_fields_fail_closed(snapshot) -> None:
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_snapshot({**snapshot, "extra": "not declared"}, now=NOW)
    del snapshot["macro_as_of"]
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_snapshot(snapshot, now=NOW)


def test_markdown_content_in_free_text_cannot_inject_rows_or_markers(snapshot) -> None:
    snapshot["optimization_outcome"] = "fallback | <b>hold</b>"
    snapshot["data_freshness"]["reason"] = START_MARKER

    rendered = render_snapshot(snapshot, now=NOW)

    assert "fallback \\| &lt;b&gt;hold&lt;/b&gt;" in rendered
    assert START_MARKER not in rendered


@pytest.mark.parametrize(
    "source",
    [
        "No markers",
        START_MARKER,
        END_MARKER + START_MARKER,
        START_MARKER + START_MARKER + END_MARKER,
        START_MARKER + END_MARKER + END_MARKER,
    ],
)
def test_invalid_marker_layout_is_rejected(source) -> None:
    with pytest.raises(ValueError, match="marker"):
        replace_snapshot_section(source, "new content")


def test_replacement_is_idempotent_and_preserves_all_surrounding_bytes(tmp_path, snapshot) -> None:
    prefix = "\ufeff# Research résumé\r\n\r\nUntouched | table\r\n" + START_MARKER
    suffix = END_MARKER + "\r\n\r\n## Original math\r\n$`x_t`$\r\n"
    readme = tmp_path / "README.md"
    data_file = tmp_path / "snapshot.json"
    readme.write_bytes((prefix + "\r\nold result\r\n" + suffix).encode("utf-8"))
    data_file.write_text(json.dumps(snapshot), encoding="utf-8")

    assert update_readme_file(readme, data_file, now=NOW)
    first = readme.read_bytes()
    assert first.startswith(prefix.encode("utf-8"))
    assert first.endswith(suffix.encode("utf-8"))
    assert b"\n" not in first.replace(b"\r\n", b"")
    assert not update_readme_file(readme, data_file, now=NOW)
    assert readme.read_bytes() == first


def test_check_and_failed_validation_never_modify_readme(tmp_path, snapshot) -> None:
    readme = tmp_path / "README.md"
    data_file = tmp_path / "snapshot.json"
    original = f"Heading\n{START_MARKER}\nold\n{END_MARKER}\nrest".encode()
    readme.write_bytes(original)
    data_file.write_text(json.dumps(snapshot), encoding="utf-8")

    assert update_readme_file(readme, data_file, now=NOW, check=True)
    assert readme.read_bytes() == original
    snapshot["weights"]["SPY"] = 0.9
    data_file.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to one"):
        update_readme_file(readme, data_file, now=NOW)
    assert readme.read_bytes() == original


@pytest.mark.parametrize(
    "source,match",
    [
        ('{"schema_version": 1, "schema_version": 1}', "Duplicate JSON key"),
        ('{"weights": {"SPY": 0, "SPY": 1}}', "Duplicate JSON key"),
        ('{"value": NaN}', "Nonfinite JSON constant"),
        ('{"value": Infinity}', "Nonfinite JSON constant"),
        ("[]", "JSON object"),
    ],
)
def test_loading_rejects_ambiguous_or_nonstandard_json(tmp_path: Path, source, match) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_snapshot(path, now=NOW)
