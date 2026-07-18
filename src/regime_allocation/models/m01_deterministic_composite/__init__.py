"""Public interface for the frozen Model 01 regime architecture.

Model 01 deterministically labels monthly growth/inflation quadrants, estimates
a causal first-order transition model, updates a joint four-month path with
release-block likelihoods, and evaluates the resulting probability forecasts.
This package re-exports the stable primitives used by CLIs and tests; file I/O
and artifact publication remain outside the mathematical core.
"""

from regime_allocation.models.m01_deterministic_composite.evaluation import (
    CalibrationBin,
    ClasswiseCalibrationBin,
    EvaluationMetrics,
    classwise_calibration_bins,
    classwise_expected_calibration_error,
    evaluate_regime_probabilities,
)
from regime_allocation.models.m01_deterministic_composite.inference import (
    AxisLogLikelihood,
    PathUpdateResult,
    condition_path_on_regimes,
    initialize_markov_path,
    path_marginal,
    path_marginals,
    update_path_log_likelihoods,
)

from regime_allocation.models.m01_deterministic_composite.pipeline import (
    REGIME_LABELS,
    REGIME_ORDER,
    Regime,
    classify_frame,
    classify_regime,
)
from regime_allocation.models.m01_deterministic_composite.transition import (
    TransitionEstimate,
    estimate_expanding_transition_matrices,
    estimate_transition_matrix,
    propagate_joint_path,
    propagate_regime_marginal,
)

__all__ = [
    "AxisLogLikelihood",
    "CalibrationBin",
    "ClasswiseCalibrationBin",
    "EvaluationMetrics",
    "PathUpdateResult",
    "REGIME_LABELS",
    "REGIME_ORDER",
    "Regime",
    "TransitionEstimate",
    "classify_frame",
    "classify_regime",
    "classwise_calibration_bins",
    "classwise_expected_calibration_error",
    "condition_path_on_regimes",
    "estimate_expanding_transition_matrices",
    "estimate_transition_matrix",
    "evaluate_regime_probabilities",
    "initialize_markov_path",
    "path_marginal",
    "path_marginals",
    "propagate_joint_path",
    "propagate_regime_marginal",
    "update_path_log_likelihoods",
]
