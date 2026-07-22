"""Test performance summaries and paired circular-block bootstrap uncertainty.

The inputs are compact synthetic monthly return panels and NAV paths with known
closed-form answers. Tests reconcile compounded return and drawdown, reproduce
bootstrap samples manually, verify deterministic seeds and comparator ordering,
and reject misaligned months or invalid configurations. No result artifacts are
written; the suite protects the statistical reporting layer from silent changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_allocation.backtest import paired_circular_block_bootstrap
from regime_allocation.backtest.metrics import compute_performance_metrics


def test_performance_metrics_reconcile_geometric_return_and_drawdown() -> None:
    monthly = pd.DataFrame(
        {
            "method": ["test"] * 3,
            "reference_month": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "gross_return": [0.10, -0.10, 0.05],
            "net_return": [0.10, -0.10, 0.05],
            "transaction_cost_rate": [0.0, 0.0, 0.0],
            "gross_turnover": [1.0, 0.0, 0.0],
            "one_way_turnover": [0.5, 0.0, 0.0],
        }
    )
    nav = pd.DataFrame(
        {
            "method": ["test"] * 4,
            "date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "phase_order": [0, 1, 1, 1],
            "nav": [1.0, 1.1, 0.99, 1.0395],
        }
    )
    metrics = compute_performance_metrics(monthly, nav).iloc[0]
    assert np.isclose(metrics["total_return"], 1.1 * 0.9 * 1.05 - 1.0)
    assert np.isclose(metrics["maximum_drawdown"], 0.99 / 1.1 - 1.0)
    assert metrics["months"] == 3


def _monthly_methods(method_returns: dict[str, list[float]]) -> pd.DataFrame:
    """Convert aligned method-return lists into the evaluator's long-form table."""
    observations = len(next(iter(method_returns.values())))
    months = pd.date_range("2019-01-01", periods=observations, freq="MS")
    frames = []
    for method, returns in method_returns.items():
        frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "reference_month": months,
                    "net_return": returns,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_paired_bootstrap_matches_manual_circular_block_calculation() -> None:
    baseline = np.array([0.01, 0.03, -0.02, 0.04, 0.00, 0.02, -0.01, 0.05])
    comparator = np.array([0.00, 0.02, -0.01, 0.01, 0.01, 0.00, 0.00, 0.03])
    simulation = _monthly_methods(
        {"posterior": baseline.tolist(), "equal_weight": comparator.tolist()}
    )
    result = paired_circular_block_bootstrap(
        simulation,
        baseline_method="posterior",
        comparator_methods=("equal_weight",),
        block_length=3,
        n_resamples=2_000,
        confidence_level=0.90,
        seed=731,
    ).iloc[0]

    differences = baseline - comparator
    generator = np.random.default_rng(731)
    starts = generator.integers(0, len(differences), size=(2_000, 3), endpoint=False)
    offsets = np.arange(3)
    indices = ((starts[:, :, None] + offsets) % len(differences)).reshape(2_000, -1)[:, :8]
    bootstrap_means = differences[indices].mean(axis=1) * 12
    monthly_std = np.std(differences, ddof=1)

    assert result["annualized_mean_difference"] == pytest.approx(np.mean(differences) * 12)
    assert result["annualized_mean_difference_ci_lower"] == pytest.approx(
        np.quantile(bootstrap_means, 0.05)
    )
    assert result["annualized_mean_difference_ci_upper"] == pytest.approx(
        np.quantile(bootstrap_means, 0.95)
    )
    assert result["probability_mean_difference_positive"] == pytest.approx(
        np.mean(bootstrap_means > 0)
    )
    assert result["naive_paired_t_statistic"] == pytest.approx(
        np.mean(differences) / (monthly_std / np.sqrt(len(differences)))
    )
    assert result["annualized_tracking_error"] == pytest.approx(monthly_std * np.sqrt(12))
    assert result["information_ratio"] == pytest.approx(
        np.mean(differences) * 12 / (monthly_std * np.sqrt(12))
    )


def test_paired_bootstrap_is_deterministic_and_comparator_order_invariant() -> None:
    simulation = _monthly_methods(
        {
            "posterior": [0.01, 0.03, -0.01, 0.02, 0.00, 0.04],
            "equal_weight": [0.00, 0.02, 0.00, 0.01, -0.01, 0.03],
            "sixty_forty": [0.02, 0.01, -0.02, 0.02, 0.01, 0.02],
        }
    )
    forward = paired_circular_block_bootstrap(
        simulation,
        baseline_method="posterior",
        comparator_methods=("equal_weight", "sixty_forty"),
        block_length=2,
        n_resamples=500,
        seed=99,
    ).set_index("comparator_method")
    reverse = paired_circular_block_bootstrap(
        simulation,
        baseline_method="posterior",
        comparator_methods=("sixty_forty", "equal_weight"),
        block_length=2,
        n_resamples=500,
        seed=99,
    ).set_index("comparator_method")

    pd.testing.assert_frame_equal(forward.sort_index(), reverse.sort_index())


def test_paired_bootstrap_constant_difference_reports_degenerate_interval() -> None:
    simulation = _monthly_methods(
        {
            "posterior": [0.02] * 12,
            "equal_weight": [0.01] * 12,
        }
    )
    row = paired_circular_block_bootstrap(
        simulation,
        baseline_method="posterior",
        comparator_methods=("equal_weight",),
        block_length=6,
        n_resamples=100,
        seed=1,
    ).iloc[0]

    assert row["annualized_mean_difference"] == pytest.approx(0.12)
    assert row["annualized_mean_difference_ci_lower"] == pytest.approx(0.12)
    assert row["annualized_mean_difference_ci_upper"] == pytest.approx(0.12)
    assert row["probability_mean_difference_positive"] == 1.0
    assert row["annualized_tracking_error"] == pytest.approx(0.0)
    assert np.isnan(row["naive_paired_t_statistic"])
    assert np.isnan(row["information_ratio"])


def test_paired_bootstrap_supports_52_period_annualization() -> None:
    baseline = np.array([0.010, 0.020, -0.005, 0.015, 0.000, 0.025])
    comparator = np.array([0.005, 0.010, 0.000, 0.010, -0.005, 0.015])
    simulation = _monthly_methods(
        {"posterior": baseline.tolist(), "equal_weight": comparator.tolist()}
    )

    row = paired_circular_block_bootstrap(
        simulation,
        baseline_method="posterior",
        comparator_methods=("equal_weight",),
        block_length=2,
        n_resamples=500,
        seed=17,
        periods_per_year=52,
    ).iloc[0]

    differences = baseline - comparator
    period_std = np.std(differences, ddof=1)
    expected_mean = float(np.mean(differences) * 52)
    expected_tracking_error = float(period_std * np.sqrt(52))
    assert row["annualized_mean_difference"] == pytest.approx(expected_mean)
    assert row["annualized_tracking_error"] == pytest.approx(expected_tracking_error)
    assert row["information_ratio"] == pytest.approx(
        expected_mean / expected_tracking_error
    )


@pytest.mark.parametrize("problem", ["missing", "reordered", "duplicate"])
def test_paired_bootstrap_requires_identical_ordered_unique_months(problem: str) -> None:
    simulation = _monthly_methods(
        {
            "posterior": [0.01, 0.02, 0.03, 0.04],
            "equal_weight": [0.00, 0.01, 0.02, 0.03],
        }
    )
    comparator_mask = simulation["method"].eq("equal_weight")
    comparator_indices = simulation.index[comparator_mask].tolist()
    if problem == "missing":
        simulation = simulation.drop(comparator_indices[-1])
        match = "identical ordered"
    elif problem == "reordered":
        baseline = simulation.loc[~comparator_mask]
        comparator = simulation.loc[comparator_mask].iloc[::-1]
        simulation = pd.concat([baseline, comparator], ignore_index=True)
        match = "strictly increasing"
    else:
        simulation = pd.concat(
            [simulation, simulation.loc[[comparator_indices[-1]]]], ignore_index=True
        )
        match = "duplicate reference months"

    with pytest.raises(ValueError, match=match):
        paired_circular_block_bootstrap(
            simulation,
            baseline_method="posterior",
            comparator_methods=("equal_weight",),
            block_length=2,
            n_resamples=20,
            seed=3,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"comparator_methods": ()}, "cannot be empty"),
        ({"comparator_methods": ("posterior",)}, "cannot also be"),
        ({"comparator_methods": ("equal_weight", "equal_weight")}, "must be unique"),
        ({"block_length": 0}, "strictly positive"),
        ({"block_length": 5}, "cannot exceed"),
        ({"n_resamples": 0}, "strictly positive"),
        ({"confidence_level": 1.0}, "strictly between"),
        ({"seed": -1}, "non-negative"),
        ({"seed": 1.5}, "must be an integer"),
        ({"periods_per_year": 0}, "strictly positive"),
        ({"periods_per_year": 52.5}, "must be an integer"),
        ({"periods_per_year": True}, "must be an integer"),
    ],
)
def test_paired_bootstrap_validates_configuration(
    overrides: dict[str, object],
    match: str,
) -> None:
    simulation = _monthly_methods(
        {
            "posterior": [0.01, 0.02, 0.03, 0.04],
            "equal_weight": [0.00, 0.01, 0.02, 0.03],
        }
    )
    arguments: dict[str, object] = {
        "baseline_method": "posterior",
        "comparator_methods": ("equal_weight",),
        "block_length": 2,
        "n_resamples": 20,
        "confidence_level": 0.95,
        "seed": 7,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=match):
        paired_circular_block_bootstrap(simulation, **arguments)  # type: ignore[arg-type]
