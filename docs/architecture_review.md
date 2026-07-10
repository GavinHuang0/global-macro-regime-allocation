# Architecture review notes

This document will track issues discovered in the original architecture and
the decisions made during the rebuild.

## Initial issues to investigate

1. Whether the HMM transition matrix is statistically compatible with the
   CPI/NFP-defined macro states.
2. Whether smoothed HMM probabilities leak future information.
3. Whether HMM state-to-regime matching was one-to-one and stable.
4. Whether macro data were aligned to publication dates rather than observation dates.
5. Whether daily forward-filling of monthly releases overweights repeated information.
6. Whether likelihood estimation used target labels inconsistent with the claimed framework.
7. Whether the backtest actually consumed the Bayesian posterior or only the hard CPI/NFP label.
8. Whether model fitting and standardization were performed strictly within each walk-forward window.
9. Whether covariance estimation was stable in 11 dimensions with sparse regimes.
10. Whether score shifting produced economically meaningful portfolio weights.
11. Whether transaction costs, turnover, ETF distributions, and adjusted prices were handled.
12. Whether results are robust to alternative regimes, benchmarks, assets, and sample periods.
