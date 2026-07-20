"""Public primitives for Model 02 scores and Gaussian quadrant mapping.

The implementation covers the causal deterministic score definition and the
uncertainty map from those scores to four soft quadrant probabilities. It does
not yet fit score-transition dynamics, process evidence likelihoods, or perform
allocation.
"""

from regime_allocation.models.m02_soft_composite.scores import (
    ALL_COMPONENTS,
    GROWTH_COMPONENTS,
    INFLATION_COMPONENTS,
    build_composite_scores,
    lagged_expanding_moments,
    lagged_expanding_zscore,
)
from regime_allocation.models.m02_soft_composite.probability_map import (
    REGIME_LABELS,
    REGIME_ORDER,
    causal_quadrant_mapping_history,
    component_disagreement_history,
    delete_one_jackknife_variance,
    gaussian_quadrant_weights,
)

__all__ = [
    "ALL_COMPONENTS",
    "GROWTH_COMPONENTS",
    "INFLATION_COMPONENTS",
    "build_composite_scores",
    "lagged_expanding_moments",
    "lagged_expanding_zscore",
    "REGIME_LABELS",
    "REGIME_ORDER",
    "causal_quadrant_mapping_history",
    "component_disagreement_history",
    "delete_one_jackknife_variance",
    "gaussian_quadrant_weights",
]
