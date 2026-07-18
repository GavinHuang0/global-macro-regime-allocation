"""Test causal regime-conditional release-block likelihood estimation.

Synthetic univariate and multivariate event vectors verify pooled-mean shrinkage,
missing-regime fallback, Ledoit-Wolf and spherical covariance shrinkage, Student-t
scale conversion, Gaussian sensitivities, and agreement with SciPy densities.
Selection tests enforce strict pre-cutoff feature and label availability, while
validation tests reject incomplete or singular inputs. Outputs are in-memory
diagnostics only; the suite guards both statistical definitions and causality.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, multivariate_t

from regime_allocation.models.m01_deterministic_composite.likelihood import (
    CANONICAL_REGIME_IDS,
    fit_block_likelihood,
    fit_causal_block_likelihood,
    select_causal_training_vectors,
)


A, B, C, D = CANONICAL_REGIME_IDS


def _univariate_vectors() -> pd.DataFrame:
    """Return a sparse one-feature sample that omits two canonical regimes."""
    return pd.DataFrame(
        {
            "regime_id": [A, A, B],
            "signal": [0.0, 2.0, 10.0],
        }
    )


def _two_dimensional_vectors() -> pd.DataFrame:
    """Return a nonsingular two-feature sample for covariance checks."""
    return pd.DataFrame(
        {
            "regime_id": [A, A, A, A, A, B, B, B],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, -2.0, -1.0, 1.0],
            "y": [0.0, 2.0, 1.0, 4.0, 3.0, 1.0, -1.0, -2.0],
        }
    )


def _causal_vectors() -> pd.DataFrame:
    """Return event vectors spanning eligible, same-day, and future information."""
    rows = [
        ("eligible-1", "2020-01-02", "2019-12-01", "2020-01-05", A, 1.0),
        ("eligible-2", "2020-01-03", "2019-12-01", "2020-01-06", B, 2.0),
        ("same-day", "2020-01-04", "2019-12-01", "2020-01-10", A, 3.0),
        ("future-release", "2020-01-11", "2020-01-01", "2020-01-06", B, 4.0),
        ("incomplete", "2020-01-04", "2019-12-01", "2020-01-07", A, np.nan),
        ("missing-label-date", "2020-01-04", "2019-12-01", None, A, 5.0),
        ("missing-regime", "2020-01-04", "2019-12-01", "2020-01-06", None, 6.0),
    ]
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "event_id",
            "release_date",
            "reference_month",
            "label_available_at",
            "regime_id",
            "signal",
        ],
    )


def test_regime_means_use_pooled_kappa_pseudo_count_and_cover_missing_states() -> None:
    fit = fit_block_likelihood(
        _univariate_vectors(),
        block_id="example",
        feature_names=["signal"],
        kappa=5.0,
        degrees_of_freedom=7.0,
        covariance_method="empirical",
    )

    assert fit.pooled_mean.loc["signal"] == pytest.approx(4.0)
    assert fit.regime_counts.to_dict() == {A: 2, B: 1, C: 0, D: 0}
    assert fit.regime_means.loc[A, "signal"] == pytest.approx(22.0 / 7.0)
    assert fit.regime_means.loc[B, "signal"] == pytest.approx(5.0)
    assert fit.regime_means.loc[C, "signal"] == pytest.approx(4.0)
    assert fit.regime_means.loc[D, "signal"] == pytest.approx(4.0)


def test_zero_kappa_rejects_an_unobserved_canonical_regime() -> None:
    with pytest.raises(ValueError, match="kappa=0 cannot estimate a regime mean"):
        fit_block_likelihood(
            _univariate_vectors(),
            block_id="example",
            feature_names=["signal"],
            kappa=0.0,
            covariance_method="empirical",
        )


def test_student_t_shape_preserves_the_ledoit_wolf_covariance_interpretation() -> None:
    fit = fit_block_likelihood(
        _two_dimensional_vectors(),
        block_id="example",
        feature_names=["x", "y"],
        degrees_of_freedom=7.0,
        covariance_method="ledoit_wolf",
    )

    np.testing.assert_allclose(
        fit.distribution_scale.to_numpy(),
        (5.0 / 7.0) * fit.covariance.to_numpy(),
    )
    assert np.linalg.eigvalsh(fit.covariance.to_numpy()).min() > 0.0
    assert np.linalg.eigvalsh(fit.distribution_scale.to_numpy()).min() > 0.0
    assert 0.0 <= fit.learned_shrinkage <= 1.0


def test_scale_multiplier_scales_covariance_before_student_t_conversion() -> None:
    baseline = fit_block_likelihood(
        _two_dimensional_vectors(),
        block_id="example",
        feature_names=["x", "y"],
        degrees_of_freedom=7.0,
        scale_multiplier=1.0,
        covariance_method="ledoit_wolf",
    )
    scaled = fit_block_likelihood(
        _two_dimensional_vectors(),
        block_id="example",
        feature_names=["x", "y"],
        degrees_of_freedom=7.0,
        scale_multiplier=0.75,
        covariance_method="ledoit_wolf",
    )

    np.testing.assert_allclose(
        scaled.covariance.to_numpy(),
        0.75**2 * baseline.covariance.to_numpy(),
    )
    np.testing.assert_allclose(
        scaled.distribution_scale.to_numpy(),
        (5.0 / 7.0) * scaled.covariance.to_numpy(),
    )
    assert scaled.scale_multiplier == pytest.approx(0.75)


def test_student_t_log_likelihoods_match_scipy() -> None:
    fit = fit_block_likelihood(
        _two_dimensional_vectors(),
        block_id="example",
        feature_names=["x", "y"],
        degrees_of_freedom=7.0,
        covariance_method="ledoit_wolf",
    )
    observation = np.array([0.5, -0.25])

    actual = fit.log_likelihoods(observation)
    for regime_id in CANONICAL_REGIME_IDS:
        expected = multivariate_t.logpdf(
            observation,
            loc=fit.regime_means.loc[regime_id].to_numpy(),
            shape=fit.distribution_scale.to_numpy(),
            df=7.0,
        )
        assert actual.loc[regime_id] == pytest.approx(expected)


def test_gaussian_sensitivity_log_likelihoods_match_scipy() -> None:
    fit = fit_block_likelihood(
        _two_dimensional_vectors(),
        block_id="example",
        feature_names=["x", "y"],
        degrees_of_freedom=None,
        covariance_method="ledoit_wolf",
    )
    observation = {"x": 0.5, "y": -0.25}

    actual = fit.log_likelihoods(observation)
    for regime_id in CANONICAL_REGIME_IDS:
        expected = multivariate_normal.logpdf(
            np.array([0.5, -0.25]),
            mean=fit.regime_means.loc[regime_id].to_numpy(),
            cov=fit.covariance.to_numpy(),
        )
        assert actual.loc[regime_id] == pytest.approx(expected)
    np.testing.assert_allclose(fit.distribution_scale, fit.covariance)


def test_causal_selection_requires_training_availability_strictly_before_cutoff() -> None:
    selection = select_causal_training_vectors(
        _causal_vectors(),
        feature_names=["signal"],
        knowledge_cutoff="2020-01-10",
    )

    assert selection.eligible["event_id"].tolist() == ["eligible-1", "eligible-2"]
    reasons = selection.audit.set_index("event_id")["training_exclusion_reason"]
    assert reasons.loc["same-day"] == "not_available_strictly_before_cutoff"
    assert reasons.loc["future-release"] == "not_available_strictly_before_cutoff"
    assert reasons.loc["incomplete"] == "incomplete_feature_vector"
    assert reasons.loc["missing-label-date"] == "missing_label_availability"
    assert reasons.loc["missing-regime"] == "missing_regime"
    assert selection.diagnostics == {
        "input_vectors": 7,
        "eligible_vectors": 2,
        "excluded_missing_regime": 1,
        "excluded_missing_label_availability": 1,
        "excluded_incomplete_feature_vector": 1,
        "excluded_not_available_strictly_before_cutoff": 2,
    }


def test_causal_fit_carries_selection_cutoff_and_diagnostics() -> None:
    fit, selection = fit_causal_block_likelihood(
        _causal_vectors(),
        block_id="example",
        feature_names=["signal"],
        knowledge_cutoff="2020-01-10",
        covariance_method="ledoit_wolf",
    )

    assert fit.training_count == 2
    assert fit.knowledge_cutoff == pd.Timestamp("2020-01-10")
    assert fit.first_training_available_at == pd.Timestamp("2020-01-05")
    assert fit.last_training_available_at == pd.Timestamp("2020-01-06")
    assert fit.training_diagnostics == selection.diagnostics


def test_fixed_spherical_covariance_uses_explicit_shrinkage_formula() -> None:
    vectors = _two_dimensional_vectors()
    empirical_fit = fit_block_likelihood(
        vectors,
        block_id="example",
        feature_names=["x", "y"],
        kappa=5.0,
        covariance_method="empirical",
    )
    fixed_fit = fit_block_likelihood(
        vectors,
        block_id="example",
        feature_names=["x", "y"],
        kappa=5.0,
        covariance_method="fixed_spherical",
        fixed_shrinkage=0.25,
    )
    empirical = empirical_fit.covariance.to_numpy()
    target = np.trace(empirical) / 2.0 * np.eye(2)

    np.testing.assert_allclose(
        fixed_fit.covariance.to_numpy(),
        0.75 * empirical + 0.25 * target,
    )
    assert fixed_fit.applied_shrinkage == pytest.approx(0.25)
    assert fixed_fit.learned_shrinkage is None


def test_univariate_ledoit_wolf_and_spherical_shrinkage_equal_empirical_variance() -> None:
    vectors = _univariate_vectors()
    empirical = fit_block_likelihood(
        vectors,
        block_id="claims",
        feature_names=["signal"],
        covariance_method="empirical",
    )
    ledoit_wolf = fit_block_likelihood(
        vectors,
        block_id="claims",
        feature_names=["signal"],
        covariance_method="ledoit_wolf",
    )
    spherical = fit_block_likelihood(
        vectors,
        block_id="claims",
        feature_names=["signal"],
        covariance_method="fixed_spherical",
        fixed_shrinkage=1.0,
    )

    np.testing.assert_allclose(ledoit_wolf.covariance, empirical.covariance)
    np.testing.assert_allclose(spherical.covariance, empirical.covariance)


def test_audit_serialization_is_json_safe_and_complete() -> None:
    fit = fit_block_likelihood(
        _two_dimensional_vectors(),
        block_id="example",
        feature_names=["x", "y"],
        covariance_method="ledoit_wolf",
    )

    audit = fit.to_audit_dict()
    json.dumps(audit)
    assert audit["state_order"] == list(CANONICAL_REGIME_IDS)
    assert audit["distribution"] == "student_t"
    assert audit["student_t_scale_factor"] == pytest.approx(5.0 / 7.0)
    assert audit["scale_standard_deviation_multiplier"] == pytest.approx(1.0)
    assert audit["covariance_minimum_eigenvalue"] > 0.0


@pytest.mark.parametrize("degrees_of_freedom", [0.0, 2.0, -1.0, np.inf, np.nan])
def test_student_t_degrees_of_freedom_must_be_finite_and_exceed_two(
    degrees_of_freedom: float,
) -> None:
    with pytest.raises(ValueError, match="exceed 2"):
        fit_block_likelihood(
            _univariate_vectors(),
            block_id="example",
            feature_names=["signal"],
            degrees_of_freedom=degrees_of_freedom,
            covariance_method="empirical",
        )


@pytest.mark.parametrize("scale_multiplier", [0.0, -1.0, np.inf, np.nan])
def test_scale_multiplier_must_be_finite_and_positive(
    scale_multiplier: float,
) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        fit_block_likelihood(
            _univariate_vectors(),
            block_id="example",
            feature_names=["signal"],
            scale_multiplier=scale_multiplier,
            covariance_method="empirical",
        )


def test_fit_rejects_incomplete_vectors_and_unknown_regimes() -> None:
    incomplete = _univariate_vectors()
    incomplete.loc[0, "signal"] = np.nan
    with pytest.raises(ValueError, match="complete and finite"):
        fit_block_likelihood(
            incomplete,
            block_id="example",
            feature_names=["signal"],
        )

    unknown = _univariate_vectors()
    unknown.loc[0, "regime_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown regime"):
        fit_block_likelihood(
            unknown,
            block_id="example",
            feature_names=["signal"],
        )


def test_fit_rejects_same_day_training_row_when_cutoff_is_supplied() -> None:
    vectors = _univariate_vectors()
    vectors["training_available_at"] = [
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
    ]
    with pytest.raises(ValueError, match="strictly before"):
        fit_block_likelihood(
            vectors,
            block_id="example",
            feature_names=["signal"],
            knowledge_cutoff="2020-01-03",
        )


def test_empirical_covariance_rejects_a_singular_block() -> None:
    vectors = pd.DataFrame(
        {
            "regime_id": [A, A, B],
            "x": [0.0, 1.0, 2.0],
            "duplicate_x": [0.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="positive definite"):
        fit_block_likelihood(
            vectors,
            block_id="example",
            feature_names=["x", "duplicate_x"],
            kappa=5.0,
            covariance_method="empirical",
        )
