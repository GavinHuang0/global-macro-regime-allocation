"""Public primitives for Model 02's score, transition, and Bayesian filter.

Model 02 keeps the released growth/inflation score center continuous, evolves a
rolling four-month joint Gaussian state with a causal VAR(1), updates the state
from shrinkage-estimated linear-Gaussian release models, and adds mapping
uncertainty only when reporting soft quadrant probabilities. Portfolio
allocation remains outside this model package.
"""

from regime_allocation.models.m02_soft_composite.gaussian_emissions import (
    LinearGaussianEmissionFit,
    LinearGaussianEmissionSpec,
    fit_linear_gaussian_emission,
)
from regime_allocation.models.m02_soft_composite.joint_filter import (
    JointGaussianState,
    VarDynamics,
    condition_on_exact_score,
    joint_quadrant_path_probabilities,
    monthly_quadrant_probabilities,
    roll_joint_gaussian,
    update_joint_gaussian,
)

from regime_allocation.models.m02_soft_composite.scores import (
    ALL_COMPONENTS,
    GROWTH_COMPONENTS,
    INFLATION_COMPONENTS,
    build_composite_scores,
    lagged_expanding_moments,
    lagged_expanding_zscore,
)
from regime_allocation.models.m02_soft_composite.var_transition import (
    Var1Fit,
    build_var_pair_audit,
    causal_var1_prior_history,
    fit_var1_ols,
    propagate_var1_prior,
    select_causal_var_pairs,
)
from regime_allocation.models.m02_soft_composite.robust_var import (
    RobustVar1Fit,
    fit_var1_huber,
    fit_var1_sensitivity,
    fit_var1_student_t,
    robust_var1_weight_audit,
)
from regime_allocation.models.m02_soft_composite.robust_emissions import (
    DEFAULT_DEGREES_OF_FREEDOM_GRID,
    LinearStudentTEmissionFit,
    RobustGaussianApproximation,
    RobustStateUpdate,
    fit_linear_student_t_emission,
    fit_linear_student_t_emission_fixed,
    multivariate_student_t_nll,
    student_t_event_weight,
)
from regime_allocation.models.m02_soft_composite.probability_map import (
    REGIME_LABELS,
    REGIME_ORDER,
    causal_quadrant_mapping_history,
    component_disagreement_history,
    delete_one_jackknife_variance,
    gaussian_quadrant_weights,
)
from regime_allocation.models.m02_soft_composite.walkforward import (
    GaussianWalkForwardResult,
    PreparedObservationData,
    observation_specifications,
    prepare_observation_data,
    run_event_driven_filter,
)
from regime_allocation.models.m02_soft_composite.partial_defining import (
    CausalComponentGaussianFit,
    PartialDefiningUpdate,
    PreparedPartialDefiningData,
    apply_partial_defining_event,
    fit_causal_component_gaussian,
    prepare_partial_defining_data,
)
from regime_allocation.models.m02_soft_composite.inference_sensitivities import (
    InferenceSensitivityResult,
    SensitivityVariant,
    annual_rolling_origin_tail_scores,
    causal_annual_hyperparameter_schedule,
    run_inference_sensitivities,
)

__all__ = [
    "ALL_COMPONENTS",
    "GROWTH_COMPONENTS",
    "INFLATION_COMPONENTS",
    "build_composite_scores",
    "lagged_expanding_moments",
    "lagged_expanding_zscore",
    "LinearGaussianEmissionFit",
    "LinearGaussianEmissionSpec",
    "fit_linear_gaussian_emission",
    "JointGaussianState",
    "VarDynamics",
    "condition_on_exact_score",
    "joint_quadrant_path_probabilities",
    "monthly_quadrant_probabilities",
    "roll_joint_gaussian",
    "update_joint_gaussian",
    "REGIME_LABELS",
    "REGIME_ORDER",
    "causal_quadrant_mapping_history",
    "component_disagreement_history",
    "delete_one_jackknife_variance",
    "gaussian_quadrant_weights",
    "Var1Fit",
    "build_var_pair_audit",
    "causal_var1_prior_history",
    "fit_var1_ols",
    "propagate_var1_prior",
    "select_causal_var_pairs",
    "RobustVar1Fit",
    "fit_var1_huber",
    "fit_var1_sensitivity",
    "fit_var1_student_t",
    "robust_var1_weight_audit",
    "DEFAULT_DEGREES_OF_FREEDOM_GRID",
    "LinearStudentTEmissionFit",
    "RobustGaussianApproximation",
    "RobustStateUpdate",
    "fit_linear_student_t_emission",
    "fit_linear_student_t_emission_fixed",
    "multivariate_student_t_nll",
    "student_t_event_weight",
    "GaussianWalkForwardResult",
    "PreparedObservationData",
    "observation_specifications",
    "prepare_observation_data",
    "run_event_driven_filter",
    "CausalComponentGaussianFit",
    "PartialDefiningUpdate",
    "PreparedPartialDefiningData",
    "apply_partial_defining_event",
    "fit_causal_component_gaussian",
    "prepare_partial_defining_data",
    "InferenceSensitivityResult",
    "SensitivityVariant",
    "annual_rolling_origin_tail_scores",
    "causal_annual_hyperparameter_schedule",
    "run_inference_sensitivities",
]
