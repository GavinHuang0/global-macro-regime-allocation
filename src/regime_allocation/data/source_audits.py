"""Coverage-aware comparisons for the M03 source-data experiment."""

from __future__ import annotations

import pandas as pd


def compare_feature_ledgers(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Separate changed coverage from changed values on identical source/month keys."""
    return _compare_ledgers(left, right, reference="reference_month", value="transformed_value")


def compare_observation_ledgers(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Compare native reference periods without confusing weekly and monthly coverage."""
    return _compare_ledgers(left, right, reference="reference_date", value="current_value")


def _compare_ledgers(
    left: pd.DataFrame, right: pd.DataFrame, *, reference: str, value: str
) -> dict:
    keys = ["series_id", reference]
    retained = []
    for frame in (left, right):
        frame = frame.copy()
        frame[reference] = pd.to_datetime(frame[reference])
        if frame.duplicated(keys).any():
            raise ValueError("duplicate source/month in comparison")
        retained.append(frame.loc[frame["status"].eq("retained")].set_index(keys))
    old, new = retained
    common = old.index.intersection(new.index)
    old_values = old.loc[common, value]
    new_values = new.loc[common, value]
    changed = common[old_values.ne(new_values)]
    timing_changed = common[
        pd.to_datetime(old.loc[common, "archive_available_at"], utc=True).ne(
            pd.to_datetime(new.loc[common, "archive_available_at"], utc=True)
        )
    ]

    def labels(index: pd.Index) -> list[dict]:
        return [
            {"series_id": series, reference: month.date().isoformat()} for series, month in index
        ]

    return {
        "common_retained_rows": len(common),
        "added_coverage": labels(new.index.difference(old.index)),
        "lost_coverage": labels(old.index.difference(new.index)),
        "changed_values_on_common_support": labels(changed),
        "changed_availability_on_common_support": labels(timing_changed),
        "interpretation": "Data comparison only; no model or performance comparison.",
    }


def ppi_splice_audit(unspliced: pd.DataFrame) -> dict:
    """Compare first-appearance changes before applying the configured PPI splice.

    Both legs must have been available by the caller's cutoff. Their first
    appearance dates can differ; joint availability is the later of the two.
    This diagnostic does not establish that the series definitions are equal.
    """
    frames = []
    counts = {}
    for series in ("PPILFE", "WPSFD4131"):
        rows = unspliced.loc[
            unspliced["series_id"].eq(series) & unspliced["status"].eq("retained")
        ].copy()
        if rows["reference_month"].duplicated().any():
            raise ValueError("duplicate PPI reference month")
        counts[series] = len(rows)
        frames.append(rows.set_index("reference_month"))
    old, new = frames
    common = old.index.intersection(new.index).sort_values()
    pairs = []
    for month in common:
        old_row, new_row = old.loc[month], new.loc[month]
        pairs.append(
            {
                "reference_month": pd.Timestamp(month).date().isoformat(),
                "PPILFE_change": float(old_row["transformed_value"]),
                "WPSFD4131_change": float(new_row["transformed_value"]),
                "difference_new_minus_old": float(
                    new_row["transformed_value"] - old_row["transformed_value"]
                ),
                "PPILFE_available_at": pd.Timestamp(old_row["archive_available_at"]).isoformat(),
                "WPSFD4131_available_at": pd.Timestamp(new_row["archive_available_at"]).isoformat(),
                "jointly_available_at": max(
                    pd.Timestamp(old_row["archive_available_at"]),
                    pd.Timestamp(new_row["archive_available_at"]),
                ).isoformat(),
            }
        )
    differences = pd.Series([row["difference_new_minus_old"] for row in pairs], dtype=float)
    return {
        "status": "descriptive_overlap" if pairs else "insufficient_common_support",
        "retained_months_before_splice": counts,
        "common_months": len(pairs),
        "old_only_months": len(old.index.difference(new.index)),
        "new_only_months": len(new.index.difference(old.index)),
        "mean_difference": float(differences.mean()) if pairs else None,
        "max_absolute_difference": float(differences.abs().max()) if pairs else None,
        "pairs": pairs,
        "interpretation": (
            "First-appearance log changes on common reference months, each using its own "
            "same-vintage prior level. No level splice, imputation, equivalence test or "
            "automatic source promotion; short overlap cannot establish comparability."
        ),
    }
