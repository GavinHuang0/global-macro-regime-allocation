"""Test causal return alignment, regime estimation, and posterior mixture moments.

Synthetic adjusted-open prices, checkpoints, labels, and monthly returns establish
next-month execution alignment, canonical posterior ordering, pseudo-month mean
shrinkage, global fallback, strict signal cutoffs, and the covariance contribution
from uncertainty between regime means. The module performs no I/O and guards the
information boundary between forecasts and portfolio estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.portfolio.estimation import (
    CANONICAL_REGIME_IDS,
    PROBABILITY_COLUMNS,
    build_adjusted_open_holding_returns,
    extract_post_month_roll_signals,
    fit_causal_regime_return_model,
    fit_regime_return_model,
    posterior_mixture_moments,
)


A, B, C, D = CANONICAL_REGIME_IDS


def _checkpoints() -> pd.DataFrame:
    """Return post-month-roll checkpoints used to select allocation signals."""
    return pd.DataFrame(
        {
            "checkpoint_id": ["before", "signal-1", "signal-2"],
            "specification_id": ["baseline"] * 3,
            "as_of_date": ["2020-01-31", "2020-02-01", "2020-03-01"],
            "phase_order": [6, 1, 1],
            "checkpoint_type": ["month_end", "post_month_roll", "post_month_roll"],
            "anchor_month": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "transition_training_cutoff": [
                "2019-12-31",
                "2020-01-31",
                "2020-02-29",
            ],
        }
    )


def _marginals() -> pd.DataFrame:
    """Return normalized current-month marginals in canonical regime order."""
    records: list[dict[str, object]] = []
    for checkpoint_id, month, probabilities in (
        ("signal-1", "2020-02-01", [0.1, 0.2, 0.3, 0.4]),
        ("signal-2", "2020-03-01", [0.4, 0.3, 0.2, 0.1]),
    ):
        for regime_id, probability in zip(
            CANONICAL_REGIME_IDS, probabilities, strict=True
        ):
            records.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "specification_id": "baseline",
                    "reference_month": month,
                    "relative_month": 0,
                    "marginal_type": "path",
                    "regime_id": regime_id,
                    "probability": probability,
                }
            )
    return pd.DataFrame.from_records(records)


def test_extract_post_month_roll_signals_preserves_canonical_order() -> None:
    signals = extract_post_month_roll_signals(_checkpoints(), _marginals())

    assert signals["holding_month"].tolist() == [
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-03-01"),
    ]
    assert signals.columns[-4:].tolist() == list(PROBABILITY_COLUMNS)
    np.testing.assert_allclose(
        signals.loc[0, list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
        [0.1, 0.2, 0.3, 0.4],
    )


def test_extract_post_month_roll_signals_rejects_non_normalized_probability() -> None:
    marginals = _marginals()
    marginals.loc[marginals.index[0], "probability"] = 0.2

    with pytest.raises(ValueError, match="must sum to one"):
        extract_post_month_roll_signals(_checkpoints(), marginals)


def test_adjusted_open_returns_use_next_month_first_common_open() -> None:
    rows = []
    for ticker, values in {"SPY": [100.0, 110.0, 121.0], "IEF": [50.0, 51.0, 49.0]}.items():
        for date, value in zip(
            ["2020-01-02", "2020-02-03", "2020-03-02"],
            values,
            strict=True,
        ):
            rows.append({"date": date, "ticker": ticker, "adjusted_open": value})
    returns = build_adjusted_open_holding_returns(
        pd.DataFrame(rows), asset_ids=["SPY", "IEF"]
    )

    assert returns["holding_month"].tolist() == [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-01"),
    ]
    assert returns.loc[0, "entry_date"] == pd.Timestamp("2020-01-02")
    assert returns.loc[0, "exit_date"] == pd.Timestamp("2020-02-03")
    assert returns.loc[0, "SPY"] == pytest.approx(0.10)
    assert returns.loc[0, "IEF"] == pytest.approx(0.02)
    assert returns.loc[1, "SPY"] == pytest.approx(0.10)


def _labeled_returns() -> pd.DataFrame:
    """Return monthly returns carrying their causal regime labels."""
    return pd.DataFrame(
        {
            "holding_month": pd.date_range("2019-01-01", periods=6, freq="MS"),
            "regime_id": [A, A, B, B, B, A],
            "SPY": [0.01, 0.03, -0.02, 0.00, 0.01, 0.02],
            "IEF": [0.00, 0.01, 0.02, 0.01, 0.03, -0.01],
        }
    )


def test_regime_means_use_exact_pseudo_month_shrinkage_and_missing_fallback() -> None:
    estimate = fit_regime_return_model(
        _labeled_returns(),
        asset_ids=["SPY", "IEF"],
        kappa=2.0,
        minimum_observations=2,
    )

    global_mean = _labeled_returns()[["SPY", "IEF"]].mean()
    sample_a = _labeled_returns().loc[
        _labeled_returns()["regime_id"].eq(A), ["SPY", "IEF"]
    ].mean()
    expected_a = (3.0 * sample_a + 2.0 * global_mean) / 5.0
    pd.testing.assert_series_equal(
        estimate.regime_means.loc[A], expected_a, check_names=False
    )
    pd.testing.assert_series_equal(
        estimate.regime_means.loc[C], global_mean, check_names=False
    )
    pd.testing.assert_series_equal(
        estimate.regime_means.loc[D], global_mean, check_names=False
    )
    assert estimate.regime_counts.to_dict() == {A: 3, B: 3, C: 0, D: 0}
    assert 0.0 <= estimate.ledoit_wolf_shrinkage <= 1.0
    assert np.linalg.eigvalsh(estimate.shared_within_covariance).min() >= -1e-12


def test_zero_kappa_uses_raw_means_and_global_fallback_for_missing_regimes() -> None:
    estimate = fit_regime_return_model(
        _labeled_returns(),
        asset_ids=["SPY", "IEF"],
        kappa=0.0,
        minimum_observations=2,
    )

    raw_a = _labeled_returns().loc[
        _labeled_returns()["regime_id"].eq(A), ["SPY", "IEF"]
    ].mean()
    global_mean = _labeled_returns()[["SPY", "IEF"]].mean()
    pd.testing.assert_series_equal(
        estimate.regime_means.loc[A], raw_a, check_names=False
    )
    pd.testing.assert_series_equal(
        estimate.regime_means.loc[C], global_mean, check_names=False
    )


def test_causal_fit_excludes_returns_or_labels_after_cutoff() -> None:
    monthly = _labeled_returns().drop(columns="regime_id")
    monthly["return_available_at"] = pd.date_range(
        "2019-02-01", periods=6, freq="MS"
    )
    history = _labeled_returns()[["holding_month", "regime_id"]].rename(
        columns={"holding_month": "reference_month"}
    )
    history["label_available_at"] = pd.date_range(
        "2019-02-15", periods=6, freq="MS"
    )

    estimate = fit_causal_regime_return_model(
        monthly,
        history,
        knowledge_cutoff="2019-06-01",
        asset_ids=["SPY", "IEF"],
        kappa=24.0,
        minimum_observations=2,
    )

    assert estimate.training_count == 4
    assert estimate.last_holding_month == pd.Timestamp("2019-04-01")
    assert estimate.knowledge_cutoff == pd.Timestamp("2019-06-01")


def test_posterior_mixture_adds_between_regime_mean_uncertainty() -> None:
    estimate = fit_regime_return_model(
        _labeled_returns(),
        asset_ids=["SPY", "IEF"],
        kappa=2.0,
        minimum_observations=2,
    )
    posterior = {A: 0.5, B: 0.5, C: 0.0, D: 0.0}

    moments = posterior_mixture_moments(posterior, estimate)

    expected = 0.5 * estimate.regime_means.loc[A] + 0.5 * estimate.regime_means.loc[B]
    pd.testing.assert_series_equal(moments.expected_mean, expected, check_names=False)
    difference = (
        estimate.regime_means.loc[A].to_numpy()
        - estimate.regime_means.loc[B].to_numpy()
    )
    expected_between = 0.25 * np.outer(difference, difference)
    np.testing.assert_allclose(moments.between_regime_covariance, expected_between)
    np.testing.assert_allclose(
        moments.covariance,
        estimate.shared_within_covariance.to_numpy() + expected_between,
    )
    assert np.linalg.eigvalsh(moments.covariance).min() >= -1e-12


def test_nonfinite_training_return_is_rejected() -> None:
    labeled = _labeled_returns()
    labeled.loc[0, "SPY"] = np.nan

    with pytest.raises(ValueError, match="finite and common"):
        fit_regime_return_model(
            labeled,
            asset_ids=["SPY", "IEF"],
            minimum_observations=2,
        )
