# Model 01 closure audit

> Archived development record. This document explains checks and corrections
> made while closing Model 01; it is not part of the current model
> specification. Current contracts are indexed in
> [`docs/models/m01_deterministic_composite/README.md`](../../models/m01_deterministic_composite/README.md).

## Scope and outcome

This audit closes the frozen Model 01 snapshot dated 18 July 2026. It reviewed
the complete lineage from macro acquisition through target construction,
transition fitting, event likelihoods, joint-path filtering, ETF return
estimation, optimization, execution accounting, published results, tests, and
documentation.

No look-forward leakage was found in the investable month-start signals or in
the 2018--2026 portfolio backtest. Two genuine issues were found and corrected:

1. The first archive snapshot caused 15 active component observations to be
   interpreted as first releases. They were useful archive-bootstrap history,
   but were not observations first published on that date.
2. A `pre_confirmation` forecast on the same date as the defining label release
   could be scored even though the implementation assumes end-of-day
   confirmation. The within-day ordering is artificial without release
   timestamps and is not safely reproducible live.

A smaller accounting presentation issue was also corrected: daily NAV now
records pre-trade and post-trade values at each rebalance, so the initial
formation cost enters maximum drawdown as well as return and terminal wealth.

Every transition, posterior, evaluation, allocation, and backtest output was
rebuilt after these changes. The tracked deterministic manifest was also
upgraded from its stale schema-1 representation to the current schema-2
contract.

## Corrective findings

| Severity | Finding | Consequence before correction | Resolution |
|---|---|---|---|
| High, data lineage | Initial ALFRED/FRED snapshot exposed 2--3 older months for several series | 15 active component rows passed the 92-day lag gate as first releases although they predated the archive's real-time start | `archive_start_latest_only=true`; only the newest reference period in the initial snapshot is eligible |
| Medium, forecast evaluation | Same-date pre-confirmation forecast was eligible | 97 baseline rows were evaluated on the target release date; 32 also followed same-day leading evidence | All scored checkpoints now require `checkpoint_date < label_available_at` |
| Low, risk metric | Daily NAV began after initial cost | Total return was correct, but drawdown could omit the first 5 bp deduction | Added `pre_trade_open` followed by `post_trade_open` NAV checkpoints |
| Preventive, availability | Label date used current-row component releases | Correct for this snapshot, but not robust to a future delayed older prerequisite | Availability now uses the cumulative latest release across all required history |

At the raw-extractor level, 17 rows would otherwise have passed the 92-day gate:
two each for `PAYEMS`, `INDPRO`, `PCEC96`, `UNRATE`, `CPILFESL`, `PPILFE`,
`WPSFD4131`, and `AHETPI`, plus one for `PCEPILFE`. The two `WPSFD4131` rows
precede that source's active splice window and never entered the semantic
component panel, leaving 15 rows that affected Model 01. Examples include
October and November 1989 payroll levels first visible in the 5 January 1990
archive snapshot and September/October 1996 core CPI levels first visible in
the 12 December 1996 snapshot. The extractor now rejects all older rows in an
initial snapshot before applying the ordinary lag and active-source gates.

## Effect of the corrections

- The first classified target moved from July 2005 to August 2005.
- Among 246 target months common to both builds, only February 2011 changed
  quadrant: from growth up / inflation down to growth up / inflation up.
- The frozen history now contains 250 rows, 246 classifications, and four
  explicitly unavailable months.
- Month-end NLL changed from 0.9224 to 0.9170 and Brier score from 0.5193 to
  0.5156. The evaluation sample remains 97 targets because same-date rows were
  removed as an entire checkpoint class, not selectively by outcome.
- Posterior-optimized total return changed from 124.27% to 124.15%; CAGR from
  9.97% to 9.96%. The economic conclusion is unchanged.
- The legacy Sharpe-MAP comparator changed more because it selects one hard
  regime and therefore reacts discontinuously to the corrected February 2011
  label.

## Causality checks performed

### Target and source data

- Reconstructed every same-vintage component transformation from raw vintage
  matrices and matched the processed values.
- Recomputed all lagged expanding means, sample standard deviations, equal
  weights, trailing smooths, quadrant labels, and availability dates.
- Confirmed that the current observation is absent from its own standardization
  sample.
- Confirmed that the configured four-month 2025--2026 unavailable block is
  caused by the documented missing component releases and is never imputed.
- Compared authenticated FRED and keyless ALFRED retrieval on representative
  series in the earlier acquisition audit; normalized point-in-time values
  matched.

### Transition and Bayesian filter

- Recomputed every adjacent transition count and Dirichlet-smoothed row.
- Verified that every walk-forward transition fit uses pairs available strictly
  before its month roll.
- Verified all 881 recorded likelihood fits: historical event dates and target
  label dates are strictly before the scored release date.
- Independently evaluated the multivariate Student-$t$ density against SciPy; the
  maximum numerical discrepancy was below `7e-16`.
- Checked that each saved 256-path posterior is finite, nonnegative, and sums to
  one within tolerance.
- Confirmed that releases are assigned only to a path coordinate matching their
  reference month, same-date releases use an atomic update, and confirmations
  enter only after leading evidence.
- Confirmed that post-confirmation and same-day pre-confirmation states are not
  eligible forecasts.

### Portfolio and performance

- Confirmed every month-start signal is a `post_month_roll` marginal formed
  before the first-session execution open.
- Confirmed every estimation return ended and every regime label was available
  strictly before execution.
- Recomputed open-to-open returns, drifted pretrade weights, transaction costs,
  target feasibility, monthly returns, terminal wealth, CAGR, volatility,
  Sharpe, and drawdown.
- Confirmed long-only/full-investment rules, individual and group caps, and the
  ex-ante volatility ceiling for every optimized target.
- Verified that the latest incomplete month is published as a target but is
  excluded from performance.
- Recomputed all tracked manifest hashes; every recorded hash matched its
  artifact after the final rebuild.

## Frozen empirical interpretation

The month-end event posterior improves the transition-only proper scores by a
small amount: NLL 0.9170 versus 0.9762 and Brier 0.5156 versus 0.5291. It does
not improve transition-only MAP or balanced accuracy. This supports the narrow
claim that the leading evidence modestly improves probability assignment in
this sample, not that it reliably identifies every quadrant.

The portfolio attribution is weaker. Posterior-optimized and pooled-mean
monthly returns correlate 0.9973. The posterior method's annualized arithmetic
mean advantage is 0.184%, with a paired block-bootstrap 95% interval of
[-0.364%, 0.795%]. `SPY` is at its 35% cap and `BIL` at zero in every completed
month; `GLD` is capped in 89 of 102 months. The constraint and universe policy
therefore explains much of the historical outcome. Model 01 does not establish
a statistically reliable posterior-driven allocation edge.

## Remaining non-leakage risks and limitations

These are not hidden forward information, but they matter to interpretation:

- ETF adjusted prices are a mutable current Yahoo snapshot. Corporate-action
  scale factors cancel in interval returns, but exact future reproduction and
  historical universe membership are not guaranteed.
- The fixed ETF universe is researcher-selected and survivor/universe-selection
  bias cannot be ruled out.
- Release likelihoods treat events as conditionally independent given the
  regime. Correlated blocks and repeated weekly claims may overconcentrate the
  posterior.
- The first-order transition law implies geometric state durations and ignores
  structural breaks.
- A deterministic release composite is a forecast target, not latent economic
  ground truth; after defining releases arrive it is confirmed by construction.
- Sensitivity analysis is descriptive. It does not correct for the many choices
  made while developing the architecture.
- The forecast and backtest samples contain too few independent cycles for
  strong statistical claims.

## Verification commands

The final snapshot passed the complete test suite, manifest/hash checks,
independent arithmetic checks, and `git diff --check`. Reproduction commands
and fixed data cutoffs are in the root [`README`](../../../README.md).
