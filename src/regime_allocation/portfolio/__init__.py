"""Public posterior-weighted return-estimation interface.

The exported helpers align archived regime probabilities with executable ETF
holding returns, fit expanding regime-conditioned moments, and combine them
under a posterior distribution.  All causal fitting functions require an
explicit cutoff so a return or regime label unavailable at the trading decision
cannot enter the estimate.
"""

from regime_allocation.portfolio.estimation import (
    CANONICAL_REGIME_IDS,
    PROBABILITY_COLUMNS,
    PosteriorMixtureMoments,
    RegimeReturnEstimate,
    build_adjusted_open_holding_returns,
    extract_post_month_roll_signals,
    fit_causal_regime_return_model,
    fit_regime_return_model,
    posterior_mixture_moments,
)

__all__ = [
    "CANONICAL_REGIME_IDS",
    "PROBABILITY_COLUMNS",
    "PosteriorMixtureMoments",
    "RegimeReturnEstimate",
    "build_adjusted_open_holding_returns",
    "extract_post_month_roll_signals",
    "fit_causal_regime_return_model",
    "fit_regime_return_model",
    "posterior_mixture_moments",
]
