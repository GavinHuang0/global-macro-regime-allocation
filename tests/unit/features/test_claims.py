"""Test causal expanding-AR(1) innovations for weekly initial claims.

Deterministic synthetic ICSA levels are used to verify strict-prior OLS fitting,
minimum-history and zero-variance diagnostics, same-publication-date batching,
sequential conditioning, expanding standardization, and invariance to current or
future shocks. Invalid levels, parameters, and release sequences are rejected.
The suite has no external inputs or outputs and protects the claims evidence block
from look-forward fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.features.claims import (
    CLAIMS_AR1_COLUMNS,
    build_claims_ar1_features,
)


def _claims_levels(periods: int = 40) -> pd.Series:
    """Generate positive weekly claims levels with deterministic variation."""
    dates = pd.date_range("2020-01-02", periods=periods, freq="7D")
    time = np.arange(periods, dtype=float)
    log_levels = 11.8 + 0.001 * time + 0.06 * np.sin(time / 2.7)
    return pd.Series(np.exp(log_levels), index=dates, name="ICSA")


def test_output_contract_and_warmup_counts_are_auditable() -> None:
    levels = _claims_levels(periods=12)

    features = build_claims_ar1_features(
        levels,
        min_history=5,
        min_standardization_history=3,
    )

    assert tuple(features.columns) == CLAIMS_AR1_COLUMNS
    pd.testing.assert_index_equal(features.index, levels.index)
    pd.testing.assert_series_equal(features["level"], levels, check_names=False)
    np.testing.assert_allclose(features["log_level"], np.log(levels))
    np.testing.assert_array_equal(
        features["ar_prior_observation_count"], np.arange(len(levels))
    )
    np.testing.assert_array_equal(
        features["ar_transition_count"], np.maximum(np.arange(len(levels)) - 1, 0)
    )
    assert features["forecast_log_level"].iloc[:5].isna().all()
    assert features["forecast_log_level"].iloc[5:].notna().all()
    assert features["standardized_innovation"].iloc[:8].isna().all()
    assert features["standardized_innovation"].iloc[8:].notna().all()


def test_first_forecast_matches_ols_fit_on_prior_observations_only() -> None:
    levels = _claims_levels(periods=10)
    features = build_claims_ar1_features(
        levels,
        min_history=5,
        min_standardization_history=2,
    )
    logs = np.log(levels.to_numpy())

    lagged = logs[:4]
    current = logs[1:5]
    expected_coefficient = np.sum(
        (lagged - lagged.mean()) * (current - current.mean())
    ) / np.sum((lagged - lagged.mean()) ** 2)
    expected_intercept = current.mean() - expected_coefficient * lagged.mean()
    expected_forecast = expected_intercept + expected_coefficient * logs[4]

    first = features.iloc[5]
    assert first["ar_prior_observation_count"] == 5
    assert first["ar_transition_count"] == 4
    assert first["ar_intercept"] == pytest.approx(expected_intercept)
    assert first["ar_lag1_coefficient"] == pytest.approx(expected_coefficient)
    assert first["forecast_log_level"] == pytest.approx(expected_forecast)
    assert first["forecast_level"] == pytest.approx(np.exp(expected_forecast))
    assert first["innovation_log"] == pytest.approx(logs[5] - expected_forecast)


def test_current_and_future_values_cannot_change_current_forecast_or_parameters() -> None:
    levels = _claims_levels(periods=30)
    shocked = levels.copy()
    shocked.iloc[15:] = shocked.iloc[15:] * 25.0

    baseline = build_claims_ar1_features(
        levels,
        min_history=5,
        min_standardization_history=3,
    )
    changed = build_claims_ar1_features(
        shocked,
        min_history=5,
        min_standardization_history=3,
    )

    causal_columns = [
        "ar_intercept",
        "ar_lag1_coefficient",
        "forecast_log_level",
        "forecast_level",
        "innovation_prior_mean",
        "innovation_prior_std",
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[: levels.index[15], causal_columns],
        changed.loc[: levels.index[15], causal_columns],
    )
    assert baseline.loc[levels.index[15], "innovation_log"] != pytest.approx(
        changed.loc[levels.index[15], "innovation_log"]
    )


def test_standardization_uses_only_prior_valid_innovations() -> None:
    features = build_claims_ar1_features(
        _claims_levels(periods=20),
        min_history=5,
        min_standardization_history=3,
        ddof=1,
    )
    position = 11
    prior = features["innovation_log"].iloc[:position].dropna()
    current = features["innovation_log"].iloc[position]
    expected = (current - prior.mean()) / prior.std(ddof=1)

    assert features["standardization_prior_count"].iloc[position] == len(prior)
    assert features["innovation_prior_mean"].iloc[position] == pytest.approx(prior.mean())
    assert features["innovation_prior_std"].iloc[position] == pytest.approx(
        prior.std(ddof=1)
    )
    assert features["standardized_innovation"].iloc[position] == pytest.approx(expected)


def test_future_changes_leave_all_earlier_output_unchanged() -> None:
    levels = _claims_levels(periods=30)
    changed = levels.copy()
    changed.iloc[20:] = changed.iloc[20:] * np.linspace(2.0, 4.0, 10)

    baseline_features = build_claims_ar1_features(
        levels,
        min_history=5,
        min_standardization_history=3,
    )
    changed_features = build_claims_ar1_features(
        changed,
        min_history=5,
        min_standardization_history=3,
    )

    pd.testing.assert_frame_equal(
        baseline_features.iloc[:20],
        changed_features.iloc[:20],
    )


def test_same_date_batch_freezes_fit_and_priors_but_conditions_sequentially() -> None:
    levels = _claims_levels(periods=18)
    release_dates = pd.Series(
        levels.index + pd.Timedelta(days=5),
        index=levels.index,
    )
    catch_up_date = release_dates.iloc[12]
    release_dates.iloc[10:13] = catch_up_date

    features = build_claims_ar1_features(
        levels,
        release_dates=release_dates,
        min_history=5,
        min_standardization_history=3,
    )
    batch = features.iloc[10:13]
    logs = np.log(levels.to_numpy())

    assert batch["ar_prior_observation_count"].eq(10).all()
    assert batch["ar_transition_count"].eq(9).all()
    assert batch["ar_intercept"].nunique() == 1
    assert batch["ar_lag1_coefficient"].nunique() == 1
    assert batch["standardization_prior_count"].nunique() == 1
    assert batch["innovation_prior_mean"].nunique() == 1
    assert batch["innovation_prior_std"].nunique() == 1

    intercept = batch["ar_intercept"].iloc[0]
    coefficient = batch["ar_lag1_coefficient"].iloc[0]
    for position in range(10, 13):
        expected_forecast = intercept + coefficient * logs[position - 1]
        assert features["forecast_log_level"].iloc[position] == pytest.approx(
            expected_forecast
        )

    prior = features["innovation_log"].iloc[:10].dropna()
    for position in range(10, 13):
        expected_z = (
            features["innovation_log"].iloc[position] - prior.mean()
        ) / prior.std(ddof=1)
        assert features["standardized_innovation"].iloc[position] == pytest.approx(
            expected_z
        )


def test_earlier_actual_in_batch_changes_next_conditional_forecast_not_frozen_fit() -> None:
    levels = _claims_levels(periods=18)
    release_dates = pd.Series(
        levels.index + pd.Timedelta(days=5),
        index=levels.index,
    )
    release_dates.iloc[10:13] = release_dates.iloc[12]
    shocked_levels = levels.copy()
    shocked_levels.iloc[10] *= 20.0

    baseline = build_claims_ar1_features(
        levels,
        release_dates=release_dates,
        min_history=5,
        min_standardization_history=3,
    )
    shocked = build_claims_ar1_features(
        shocked_levels,
        release_dates=release_dates,
        min_history=5,
        min_standardization_history=3,
    )

    frozen_columns = [
        "ar_prior_observation_count",
        "ar_transition_count",
        "ar_intercept",
        "ar_lag1_coefficient",
        "standardization_prior_count",
        "innovation_prior_mean",
        "innovation_prior_std",
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[levels.index[10:13], frozen_columns],
        shocked.loc[levels.index[10:13], frozen_columns],
    )
    assert baseline["forecast_log_level"].iloc[10] == pytest.approx(
        shocked["forecast_log_level"].iloc[10]
    )
    assert baseline["forecast_log_level"].iloc[11] != pytest.approx(
        shocked["forecast_log_level"].iloc[11]
    )


def test_constant_history_is_reported_as_unidentified_not_silently_forecast() -> None:
    levels = pd.Series(
        200_000.0,
        index=pd.date_range("2020-01-02", periods=12, freq="7D"),
    )

    features = build_claims_ar1_features(
        levels,
        min_history=5,
        min_standardization_history=2,
    )

    assert features["ar_prior_observation_count"].iloc[-1] == 11
    assert features["ar_transition_count"].iloc[-1] == 10
    assert features["ar_lag1_coefficient"].isna().all()
    assert features["forecast_log_level"].isna().all()
    assert features["innovation_log"].isna().all()
    assert features["standardization_prior_count"].eq(0).all()


@pytest.mark.parametrize(
    ("levels", "error", "message"),
    [
        ([1.0, 2.0, 3.0], TypeError, "pandas Series"),
        (pd.Series([1.0, np.nan, 3.0]), ValueError, "finite, non-missing"),
        (pd.Series([1.0, 0.0, 3.0]), ValueError, "strictly positive"),
        (pd.Series([1.0, np.inf, 3.0]), ValueError, "finite, non-missing"),
        (pd.Series([1.0, "bad", 3.0]), ValueError, "numeric"),
        (
            pd.Series([1.0, 2.0], index=[pd.Timestamp("2020-01-02")] * 2),
            ValueError,
            "unique",
        ),
        (
            pd.Series(
                [1.0, 2.0],
                index=[pd.Timestamp("2020-01-09"), pd.Timestamp("2020-01-02")],
            ),
            ValueError,
            "increasing reference sequence",
        ),
    ],
)
def test_invalid_level_inputs_are_rejected(
    levels: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        build_claims_ar1_features(levels)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"min_history": 2}, ValueError, "at least 3"),
        ({"min_history": 3.5}, TypeError, "integer"),
        ({"min_standardization_history": 1}, ValueError, "at least 2"),
        ({"min_standardization_history": True}, TypeError, "integer"),
        ({"ddof": -1}, ValueError, "at least 0"),
        ({"ddof": 2, "min_standardization_history": 2}, ValueError, "smaller"),
    ],
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        build_claims_ar1_features(_claims_levels(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("release_dates", "message"),
    [
        ([pd.Timestamp("2020-01-01")], "length must match"),
        (
            [
                pd.Timestamp("2020-01-09"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-16"),
            ],
            "monotonically non-decreasing",
        ),
        (
            [pd.Timestamp("2020-01-02"), pd.NaT, pd.Timestamp("2020-01-16")],
            "missing",
        ),
    ],
)
def test_invalid_release_date_sequences_are_rejected(
    release_dates: list[pd.Timestamp],
    message: str,
) -> None:
    levels = _claims_levels(periods=3)
    with pytest.raises(ValueError, match=message):
        build_claims_ar1_features(levels, release_dates=release_dates)


def test_release_date_series_must_align_with_levels_index() -> None:
    levels = _claims_levels(periods=3)
    release_dates = pd.Series(
        levels.index + pd.Timedelta(days=5),
        index=pd.RangeIndex(3),
    )

    with pytest.raises(ValueError, match="index must match"):
        build_claims_ar1_features(levels, release_dates=release_dates)
