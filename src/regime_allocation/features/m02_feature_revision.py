"""Construct deterministic event rows for the Model 02 feature revision.

The feature-revision stage reuses frozen point-in-time event artifacts.  Its
only new event construction is a dependence-aware continued-claims equation.
Initial and continued claims are published together but normally describe
different reference weeks, and 137 verified pairs cross a calendar-month
boundary.  Treating both as responses to one monthly state would therefore
retarget one observation incorrectly.

Instead, the stage uses the chain-rule factorization

``p(ICSA_t | Z[q_I]) p(CCSA_t | Z[q_C], ICSA_t)``.

The original ICSA row remains an ordinary growth-state observation for its own
reference month.  This module creates a second event targeted to CCSA's own
reference month; CCSA is the response and the exact same-publication ICSA
innovation is an observed control.  Pairing requires the same release date and
an ICSA reference week exactly seven days after the CCSA reference week.  No
future value, nearest-date guess, or calendar-month retargeting is allowed.

The module also rotates the two legacy capital-goods responses into an
economically clearer activity/pipeline basis.  Current investment activity is
the existing same-vintage log change in core capital-goods shipments.  The
forward pipeline is

``D_t = 100 * Delta log(orders_t / shipments_t)``

and is standardized against values published strictly before the current
release date.  Thus the transformation uses the same source information as
the legacy block, but tests a different, explicitly forward-looking feature
rather than relabeling the old orders/shipments pair.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


CONDITIONAL_CLAIMS_BLOCK = "weekly_continued_claims_conditional"
CONDITIONAL_CLAIMS_RESPONSE = "continued_claims_innovation"
CONDITIONAL_CLAIMS_CONTROL = "same_publication_initial_claims_innovation_control"
BUSINESS_REVISION_BLOCK = "business_investment_activity_pipeline"
BUSINESS_ACTIVITY_RESPONSE = "core_capital_goods_shipments_log_change"
BUSINESS_PIPELINE_RESPONSE = "core_capital_goods_orders_shipments_gap_log_change"


@dataclass(frozen=True)
class ConditionalClaimsAudit:
    """Coverage counts for exact same-publication claims pairing."""

    continued_rows: int
    paired_rows: int
    unmatched_rows: int
    cross_month_pairs: int
    available_continued_rows: int
    available_paired_rows: int
    available_cross_month_pairs: int
    publication_dates_with_multiple_pairs: int


@dataclass(frozen=True)
class BusinessRevisionAudit:
    """Coverage counts for the exact orders/shipments feature rotation."""

    source_event_groups: int
    matched_event_groups: int
    unmatched_event_groups: int
    available_activity_events: int
    available_pipeline_events: int
    available_joint_events: int


def build_business_investment_revision_events(
    events: pd.DataFrame,
    *,
    minimum_standardization_history: int = 24,
    ddof: int = 1,
) -> tuple[pd.DataFrame, BusinessRevisionAudit]:
    """Return shipments-activity and orders/shipments-pipeline event rows.

    Orders and shipments must share an exact publication date and reference
    month.  Catch-up rows released on one date are standardized atomically:
    none of that date's pipeline values enters another row's prior mean or
    standard deviation.  The activity coordinate retains its already-causal
    upstream standardization; the new pipeline coordinate receives its own
    strictly lagged expanding standardization.
    """

    if minimum_standardization_history < 3:
        raise ValueError("minimum_standardization_history must be at least three")
    if ddof < 0:
        raise ValueError("ddof must be nonnegative")
    required = {
        "event_id",
        "event_group_id",
        "release_block",
        "release_date",
        "reference_date",
        "reference_month",
        "feature_name",
        "transformed_value",
        "feature_value",
        "feature_status",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(
            "business events omit required columns: " + ", ".join(sorted(missing))
        )
    names = {"core_capital_goods_orders_log_change", BUSINESS_ACTIVITY_RESPONSE}
    source = events.loc[
        events["release_block"].astype(str).eq("business_investment")
        & events["feature_name"].astype(str).isin(names)
    ].copy()
    if source.empty:
        raise ValueError("business events omit the legacy orders/shipments block")
    for column in ("release_date", "reference_date", "reference_month"):
        source[column] = pd.to_datetime(source[column], errors="raise").dt.normalize()
    duplicate = source.duplicated(
        ["release_date", "reference_date", "feature_name"], keep=False
    )
    if duplicate.any():
        raise ValueError("business events repeat a feature at one publication target")

    keys = ["release_date", "reference_date", "reference_month"]
    records: list[dict[str, object]] = []
    source_groups = 0
    matched_groups = 0
    for key, group in source.groupby(keys, sort=True, dropna=False):
        source_groups += 1
        by_name = {
            str(row.feature_name): row for row in group.itertuples(index=False)
        }
        if names.difference(by_name):
            continue
        matched_groups += 1
        release_date, reference_date, reference_month = map(pd.Timestamp, key)
        event_id = (
            f"{BUSINESS_REVISION_BLOCK}:{release_date:%Y-%m-%d}:"
            f"{reference_date:%Y-%m-%d}"
        )
        event_group_id = f"{BUSINESS_REVISION_BLOCK}:{release_date:%Y-%m-%d}"
        orders = by_name["core_capital_goods_orders_log_change"]
        shipments = by_name[BUSINESS_ACTIVITY_RESPONSE]

        activity = shipments._asdict()
        activity.update(
            {
                "event_id": event_id,
                "event_group_id": event_group_id,
                "release_block": BUSINESS_REVISION_BLOCK,
                "business_pair_status": "exact_orders_shipments_pair",
            }
        )
        records.append(activity)

        orders_transformed = float(orders.transformed_value)
        shipments_transformed = float(shipments.transformed_value)
        transformed = orders_transformed - shipments_transformed
        orders_current = float(orders.current_value)
        shipments_current = float(shipments.current_value)
        orders_previous = float(orders.previous_value_as_of_release)
        shipments_previous = float(shipments.previous_value_as_of_release)
        current_ratio = (
            orders_current / shipments_current
            if orders_current > 0.0 and shipments_current > 0.0
            else np.nan
        )
        previous_ratio = (
            orders_previous / shipments_previous
            if orders_previous > 0.0 and shipments_previous > 0.0
            else np.nan
        )
        pipeline = orders._asdict()
        pipeline.update(
            {
                "event_id": event_id,
                "event_group_id": event_group_id,
                "release_block": BUSINESS_REVISION_BLOCK,
                "feature_name": BUSINESS_PIPELINE_RESPONSE,
                "series_id": "NEWORDER_DIV_ANXAVS",
                "provider_id": "derived_alfred",
                "source_url": (
                    "https://fred.stlouisfed.org/series/NEWORDER;"
                    "https://fred.stlouisfed.org/series/ANXAVS"
                ),
                "current_value": current_ratio,
                "previous_value_as_of_release": previous_ratio,
                "transform": "same_vintage_orders_shipments_log_ratio_change",
                "transformed_value": transformed,
                "feature_value": np.nan,
                "feature_status": "standardization_pending",
                "business_pair_status": "exact_orders_shipments_pair",
            }
        )
        records.append(pipeline)

    if not records:
        raise ValueError("business events contain no exact orders/shipments pairs")
    output = pd.DataFrame.from_records(records)
    for column in (
        "standardization_prior_count",
        "standardization_prior_mean",
        "standardization_prior_std",
    ):
        if column not in output:
            output[column] = np.nan
    pipeline_mask = output["feature_name"].astype(str).eq(BUSINESS_PIPELINE_RESPONSE)
    pipeline = output.loc[pipeline_mask].sort_values(keys, kind="stable").copy()
    prior_values: list[float] = []
    for _, positions in pipeline.groupby("release_date", sort=True).indices.items():
        locations = np.asarray(positions, dtype=np.int64)
        count = len(prior_values)
        mean = np.nan
        std = np.nan
        if count:
            prior = np.asarray(prior_values, dtype=float)
            mean = float(prior.mean())
            if count > ddof:
                std = float(prior.std(ddof=ddof))
        raw = pd.to_numeric(
            pipeline.iloc[locations]["transformed_value"], errors="coerce"
        ).to_numpy(dtype=float)
        pipeline.iloc[
            locations, pipeline.columns.get_loc("standardization_prior_count")
        ] = count
        pipeline.iloc[
            locations, pipeline.columns.get_loc("standardization_prior_mean")
        ] = mean
        pipeline.iloc[
            locations, pipeline.columns.get_loc("standardization_prior_std")
        ] = std
        finite = np.isfinite(raw)
        if count >= minimum_standardization_history and math.isfinite(std) and std > 0:
            values = (raw - mean) / std
            pipeline.iloc[
                locations, pipeline.columns.get_loc("feature_value")
            ] = values
            pipeline.iloc[
                locations, pipeline.columns.get_loc("feature_status")
            ] = np.where(finite, "available", "source_unavailable")
        else:
            status = (
                "standardization_warmup"
                if count < minimum_standardization_history
                else "standardization_unidentified"
            )
            pipeline.iloc[
                locations, pipeline.columns.get_loc("feature_status")
            ] = np.where(finite, status, "source_unavailable")
        prior_values.extend(raw[finite].tolist())
    output.loc[pipeline.index, pipeline.columns] = pipeline
    output = output.sort_values(
        ["release_date", "reference_date", "event_id", "feature_name"], kind="stable"
    ).reset_index(drop=True)
    status = output.pivot_table(
        index="event_id",
        columns="feature_name",
        values="feature_status",
        aggfunc="first",
    )
    activity_available = status.get(BUSINESS_ACTIVITY_RESPONSE, pd.Series(dtype=str)).eq(
        "available"
    )
    pipeline_available = status.get(BUSINESS_PIPELINE_RESPONSE, pd.Series(dtype=str)).eq(
        "available"
    )
    audit = BusinessRevisionAudit(
        source_event_groups=source_groups,
        matched_event_groups=matched_groups,
        unmatched_event_groups=source_groups - matched_groups,
        available_activity_events=int(activity_available.sum()),
        available_pipeline_events=int(pipeline_available.sum()),
        available_joint_events=int((activity_available & pipeline_available).sum()),
    )
    return output, audit


def _normalize_claim_rows(events: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    required = {
        "event_id",
        "event_group_id",
        "release_block",
        "release_date",
        "reference_date",
        "reference_month",
        "feature_name",
        "series_id",
        "feature_value",
        "feature_status",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(
            "claims events omit required columns: " + ", ".join(sorted(missing))
        )
    selected = events.loc[events["feature_name"].astype(str).eq(feature_name)].copy()
    if selected.empty:
        raise ValueError(f"claims events omit {feature_name}")
    selected["release_date"] = pd.to_datetime(
        selected["release_date"], errors="raise"
    ).dt.normalize()
    selected["reference_date"] = pd.to_datetime(
        selected["reference_date"], errors="raise"
    ).dt.normalize()
    selected["reference_month"] = (
        pd.to_datetime(selected["reference_month"], errors="raise")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    duplicate = selected.duplicated(
        ["release_date", "reference_date", "feature_name"], keep=False
    )
    if duplicate.any():
        raise ValueError(f"claims events repeat {feature_name} for one reference week")
    return selected.sort_values(
        ["release_date", "reference_date", "event_id"], kind="stable"
    ).reset_index(drop=True)


def build_conditional_continued_claims_events(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, ConditionalClaimsAudit]:
    """Return CCSA-response/ICSA-control events with exact causal pairing.

    An unmatched CCSA release is retained as a response-only vector.  The
    existing observation engine will audit it as ``missing_control`` rather
    than silently imputing an ICSA value.  All original metadata columns are
    preserved so the synthetic rows remain compatible with the standard event
    preparation path.
    """

    initial = _normalize_claim_rows(events, "initial_claims_innovation")
    continued = _normalize_claim_rows(events, CONDITIONAL_CLAIMS_RESPONSE)
    initial_lookup = {
        (pd.Timestamp(row.release_date), pd.Timestamp(row.reference_date)): row
        for row in initial.itertuples(index=False)
    }
    records: list[dict[str, object]] = []
    paired = 0
    cross_month = 0
    available_paired = 0
    available_cross_month = 0
    pairs_by_release: dict[pd.Timestamp, int] = {}
    columns = [
        *events.columns,
        "control_source_reference_date",
        "control_source_reference_month",
        "conditional_pair_status",
    ]
    for response_row in continued.itertuples(index=False):
        release_date = pd.Timestamp(response_row.release_date)
        response_reference = pd.Timestamp(response_row.reference_date)
        initial_key = (release_date, response_reference + pd.Timedelta(days=7))
        control_row = initial_lookup.get(initial_key)
        event_id = (
            f"{CONDITIONAL_CLAIMS_BLOCK}:{release_date:%Y-%m-%d}:"
            f"{response_reference:%Y-%m-%d}"
        )
        response = response_row._asdict()
        response.update(
            {
                "event_id": event_id,
                "event_group_id": f"{CONDITIONAL_CLAIMS_BLOCK}:{release_date:%Y-%m-%d}",
                "release_block": CONDITIONAL_CLAIMS_BLOCK,
                "control_source_reference_date": (
                    pd.NaT
                    if control_row is None
                    else pd.Timestamp(control_row.reference_date)
                ),
                "control_source_reference_month": (
                    pd.NaT
                    if control_row is None
                    else pd.Timestamp(control_row.reference_month)
                ),
                "conditional_pair_status": (
                    "unmatched_exact_week_offset"
                    if control_row is None
                    else "paired_exact_same_publication_plus_7d"
                ),
            }
        )
        records.append({column: response.get(column) for column in columns})
        if control_row is None:
            continue
        paired += 1
        both_available = (
            str(response_row.feature_status) == "available"
            and str(control_row.feature_status) == "available"
        )
        if both_available:
            available_paired += 1
        pairs_by_release[release_date] = pairs_by_release.get(release_date, 0) + 1
        if pd.Timestamp(control_row.reference_month) != pd.Timestamp(
            response_row.reference_month
        ):
            cross_month += 1
            if both_available:
                available_cross_month += 1
        control = control_row._asdict()
        control.update(
            {
                "event_id": event_id,
                "event_group_id": f"{CONDITIONAL_CLAIMS_BLOCK}:{release_date:%Y-%m-%d}",
                "release_block": CONDITIONAL_CLAIMS_BLOCK,
                # The control conditions the CCSA equation.  Its own state
                # update still occurs through the untouched ICSA event.
                "reference_date": response_reference,
                "reference_month": pd.Timestamp(response_row.reference_month),
                "feature_name": CONDITIONAL_CLAIMS_CONTROL,
                "control_source_reference_date": pd.Timestamp(
                    control_row.reference_date
                ),
                "control_source_reference_month": pd.Timestamp(
                    control_row.reference_month
                ),
                "conditional_pair_status": "paired_exact_same_publication_plus_7d",
            }
        )
        records.append({column: control.get(column) for column in columns})
    output = pd.DataFrame.from_records(records, columns=columns)
    output = output.sort_values(
        ["release_date", "reference_date", "event_id", "feature_name"],
        kind="stable",
    ).reset_index(drop=True)
    audit = ConditionalClaimsAudit(
        continued_rows=len(continued),
        paired_rows=paired,
        unmatched_rows=len(continued) - paired,
        cross_month_pairs=cross_month,
        available_continued_rows=int(
            continued["feature_status"].astype(str).eq("available").sum()
        ),
        available_paired_rows=available_paired,
        available_cross_month_pairs=available_cross_month,
        publication_dates_with_multiple_pairs=sum(
            count > 1 for count in pairs_by_release.values()
        ),
    )
    return output, audit


__all__ = [
    "BUSINESS_ACTIVITY_RESPONSE",
    "BUSINESS_PIPELINE_RESPONSE",
    "BUSINESS_REVISION_BLOCK",
    "BusinessRevisionAudit",
    "CONDITIONAL_CLAIMS_BLOCK",
    "CONDITIONAL_CLAIMS_CONTROL",
    "CONDITIONAL_CLAIMS_RESPONSE",
    "ConditionalClaimsAudit",
    "build_business_investment_revision_events",
    "build_conditional_continued_claims_events",
]
