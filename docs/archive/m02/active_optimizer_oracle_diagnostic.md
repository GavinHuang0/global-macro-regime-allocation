# Archived Model 02 benchmark-relative active optimizer and oracle diagnostic

## Decision and scope

This is a historical, exploratory diagnostic. It does not replace the promoted
Model 02 posterior allocation, change the pooled benchmark, or promote a new
strategy.

The experiment asked two sequential questions:

1. Can the hash-frozen Model 02 posterior add net weekly return when it is used
   only to choose tightly limited active weights around the pooled portfolio?
2. If not, would infeasible perfect knowledge of the eventual current-month
   hard quadrant rescue the same active allocation architecture?

The answer to both questions is no. The posterior-relative strategy
significantly underperformed pooled mean and did not significantly outperform
static 60/40. That failure triggered the locked oracle diagnostic. Over the
oracle's 405-week common window, the hindsight-regime strategy also
significantly underperformed pooled mean and was statistically
indistinguishable from both the posterior-relative strategy and static 60/40.

This pattern points primarily to the bridge from a current-month macro
quadrant to next-week ETF returns, including the chosen universe and holding
horizon. It does not support posterior inference error as the main bottleneck
within this architecture.

## Frozen posterior contract

The Bayesian model was not refitted. The diagnostic verifies and consumes the
already published weekly signal table:

- signal table:
  `data/processed/m02_regime_allocation_backtest/signal_table.csv`;
- signal SHA-256:
  `8cb4d27056ec5b024ad5e2cbf4549429da98def25b8602e91275731e8046111f`;
- upstream manifest:
  `data/manifests/m02_regime_allocation_backtest.json`; and
- upstream manifest SHA-256:
  `5821824dcfe6a82347d1df176467b74d0199cb7724bb08f5e07daf08801fbc4a`.

The largest recomputed expected-mean mismatch against the frozen upstream
audit is \(9.9991\times10^{-17}\). The diagnostic therefore changes the
allocation rule, not the posterior or the weekly return estimator.

## Benchmark-relative active optimizer

Let \(b_t\) be the frozen pooled-mean target, and let
\(\mu_t^{\mathrm{post}}\) and \(\mu_t^{\mathrm{pool}}\) be the posterior and
pooled weekly expected-return vectors. The optimizer selects active weights
\(a_t\) and a total target \(w_t\):

$$
w_t=b_t+a_t,
\qquad
\Delta\mu_t=\mu_t^{\mathrm{post}}-\mu_t^{\mathrm{pool}}.
$$

Its objective is

$$
\max_{a_t}\left[
\Delta\mu_t^\top a_t
-\sum_i c_i\left|w_{t,i}-w_{t,i}^{-,\mathrm{pretrade}}\right|
\right].
$$

The cost term is the estimated cost of trading the complete strategy target,
not merely the active sleeve. The target is subsequently simulated as its own
portfolio, with its own holdings drift and realized trades.

The locked active limits are:

| Constraint | Locked value |
|---|---:|
| Annualized tracking error versus pooled | 1.00% |
| Active weight per asset | ±5.00% |
| One-way active exposure, \(\frac12\sum_i|a_{t,i}|\) | 10.00% |
| Sum of active weights | 0 |
| Annualized total volatility | 10.00% |
| Transaction cost | 5 bp one way |

Tracking error and total volatility use the same causally estimated pooled
covariance, annualized by 52. The posterior changes only the expected-return
view.

The total portfolio remains long-only and fully invested. It retains the
published caps of 35% `SPY`, 50% `IEF`, 40% `TIP`, 25% `HYG`, 100% `BIL`,
25% `GLD`, and 40% `LQD`. The unchanged group limits are 50% for
`SPY`+`HYG`, 50% for `HYG`+`LQD`, and 75% for
`IEF`+`TIP`+`LQD`. All 446 targets, including the final live-only target, were
optimal solver solutions; no pooled-target fallback was used.

## Locked significance gate

The active test contains 445 complete weeks from 1 January 2018 through
6 July 2026. Its two required comparators are the otherwise identical pooled
optimizer and static 60% `SPY` / 40% `AGG`.

Inference uses 10,000 paired circular-block-bootstrap resamples, 26-week
blocks, a fixed seed of `20260718`, and two-sided 95% intervals. The statistic
is the annualized arithmetic mean difference in net weekly return. The active
strategy passes only if the lower confidence bound is strictly above zero
against both required comparators.

## Active-strategy result

| Method | Total return | CAGR | Ann. vol. | Zero-rate Sharpe | Max drawdown | Ann. one-way turnover | Ann. cost drag |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pooled mean | 123.0780% | 9.8294% | 10.5038% | 0.94597 | -21.9867% | 26.8876% | 2.9536 bp |
| Posterior-relative active | 115.0214% | 9.3583% | 9.9361% | 0.95086 | -20.8789% | 18.8952% | 2.0667 bp |
| Static 60/40 | 119.0394% | 9.5952% | 11.6221% | 0.84722 | -22.0205% | 28.0789% | 3.0779 bp |

The relative optimizer delivered slightly lower realized volatility and
drawdown than pooled, but it gave up too much return. Its lower turnover and
cost drag also show that excess trading cost is not the explanation for the
underperformance.

| Active minus comparator | Ann. mean difference | 95% interval | Bootstrap fraction above zero | Ann. tracking error | Information ratio |
|---|---:|---:|---:|---:|---:|
| Pooled mean | -0.48845% | [-0.81338%, -0.16530%] | 0.0020 | 0.71007% | -0.68789 |
| Static 60/40 | -0.39862% | [-3.16965%, +2.36354%] | 0.3925 | 4.73710% | -0.08415 |

The interval against pooled is entirely below zero, so the active strategy
significantly underperformed its clean allocation ablation. The interval
against 60/40 includes zero, so there is no statistically significant edge
there either. The both-comparator gate failed and the oracle ran.

## Conditional hindsight-regime oracle

The oracle replaces the weekly posterior vector with a one-hot vector for the
eventually finalized hard quadrant of the current calendar month. A zero
growth or inflation score is classified as up. This information was generally
unavailable at the portfolio decision date, so the oracle is intentionally
infeasible and is not an investable backtest.

Everything else is held fixed:

- return-model training still uses only information causally available before
  each decision;
- the pooled benchmark and incremental-return construction are unchanged;
- the covariance, 1% tracking-error cap, 5% asset active cap, 10% one-way
  active-exposure cap, total-portfolio constraints, optimizer, and 5 bp cost
  are identical to the posterior-relative test; and
- the oracle changes only the current regime-probability vector.

The primary oracle window is the longest contiguous prefix with exact regime
truth. It contains 405 weeks from 1 January 2018 through 29 September 2025,
covering 93 truth months and 58 regime runs. No missing label is bridged or
forward-filled.

## Oracle result

All methods in this table are recomputed on the same 405-week oracle window:

| Method | Total return | CAGR | Ann. vol. | Zero-rate Sharpe | Max drawdown | Ann. one-way turnover | Ann. cost drag |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pooled mean | 110.1630% | 10.0056% | 10.5946% | 0.95392 | -21.9867% | 26.2970% | 2.8934 bp |
| Hindsight-regime oracle | 102.7752% | 9.5013% | 9.8261% | 0.97366 | -20.5823% | 33.7754% | 3.6993 bp |
| Posterior-relative active | 102.6148% | 9.4902% | 10.0029% | 0.95724 | -20.8789% | 18.9408% | 2.0742 bp |
| Static 60/40 | 102.6688% | 9.4939% | 11.9867% | 0.81732 | -22.0205% | 29.3052% | 3.2094 bp |

| Oracle minus comparator | Ann. mean difference | 95% interval | Bootstrap fraction above zero | Ann. tracking error | Information ratio |
|---|---:|---:|---:|---:|---:|
| Pooled mean | -0.53905% | [-0.91771%, -0.16187%] | 0.0020 | 0.94612% | -0.56975 |
| Posterior-relative active | -0.00783% | [-0.35488%, +0.30343%] | 0.4882 | 0.63077% | -0.01241 |
| Static 60/40 | -0.22964% | [-2.73026%, +2.20778%] | 0.4238 | 4.65526% | -0.04933 |

The oracle significantly underperformed pooled mean. Its difference from the
posterior-relative active strategy is almost exactly zero and its intervals
against both that strategy and 60/40 include zero. On the same window, the
posterior-relative strategy also significantly underperformed pooled:
-0.53123% annualized mean with a 95% interval of
[-0.87333%, -0.18227%].

The regime attribution helps locate the failure:

| Eventual hard quadrant | Weeks | Ann. net difference vs pooled | Information ratio vs pooled |
|---|---:|---:|---:|
| Growth up / inflation up | 194 | -0.71580% | -0.97044 |
| Growth down / inflation up | 127 | -1.18870% | -1.15040 |
| Growth up / inflation down | 48 | +0.69602% | +1.31129 |
| Growth down / inflation down | 36 | +1.05845% | +0.61244 |

The two inflation-up quadrants account for 321 of 405 weeks, or 79.3% of the
oracle window, and both produced negative active returns versus pooled. The
two inflation-down quadrants were positive but occurred in only 84 weeks.

## Interpretation

The active optimizer removes the common long-run risk-premium component and
gives the posterior a controlled 1% tracking-error budget. Its failure means
the weak result was not caused only by blending a small posterior sleeve into
the pooled portfolio.

More importantly, perfect hindsight classification of the current hard
quadrant does not improve the result. Within this fixed architecture, that
makes posterior concentration or classification accuracy an unlikely primary
explanation. The more plausible bottlenecks are:

- hard quadrants discard continuous state magnitude, changes, uncertainty,
  and release surprises;
- a current-month state level may not predict the following week's return,
  even when the state is known exactly;
- causally estimated quadrant-conditional weekly means remain noisy and can
  have the wrong sign in the most frequent states; and
- the seven-ETF strategy universe may not expose the macro responses that the
  growth/inflation state is intended to capture.

The oracle is not an upper bound for every possible Bayesian architecture. It
tests only exact classification inside the existing hard-quadrant-to-weekly-
mean bridge. A continuous mixed-frequency return model, posterior changes and
release innovations, longer one- to four-week response horizons,
state-dependent risk, or a broader predeclared macro asset universe could
still perform differently. Those changes require a new locked, nested
walk-forward experiment rather than tuning this historical result.

## Artifacts

The machine-readable design is in
[`m02_active_optimizer_diagnostic.yaml`](../../../configs/models/m02_active_optimizer_diagnostic.yaml),
with lineage and hashes in
[`m02_active_optimizer_diagnostic.json`](../../../data/manifests/m02_active_optimizer_diagnostic.json).
Published outputs are under
[`results/published/m02_active_optimizer_diagnostic/`](../../../results/published/m02_active_optimizer_diagnostic/):

- [`diagnostic_summary.json`](../../../results/published/m02_active_optimizer_diagnostic/diagnostic_summary.json);
- [`active_performance.csv`](../../../results/published/m02_active_optimizer_diagnostic/active_performance.csv);
- [`active_uncertainty.csv`](../../../results/published/m02_active_optimizer_diagnostic/active_uncertainty.csv);
- [`active_weekly_weights.csv`](../../../results/published/m02_active_optimizer_diagnostic/active_weekly_weights.csv);
- [`active_weekly_returns.csv`](../../../results/published/m02_active_optimizer_diagnostic/active_weekly_returns.csv);
- [`oracle_performance.csv`](../../../results/published/m02_active_optimizer_diagnostic/oracle_performance.csv);
- [`oracle_uncertainty.csv`](../../../results/published/m02_active_optimizer_diagnostic/oracle_uncertainty.csv);
- [`oracle_regime_attribution.csv`](../../../results/published/m02_active_optimizer_diagnostic/oracle_regime_attribution.csv);
- [`oracle_truth_audit.csv`](../../../results/published/m02_active_optimizer_diagnostic/oracle_truth_audit.csv);
- [`oracle_weekly_weights.csv`](../../../results/published/m02_active_optimizer_diagnostic/oracle_weekly_weights.csv); and
- [`oracle_weekly_returns.csv`](../../../results/published/m02_active_optimizer_diagnostic/oracle_weekly_returns.csv).

The stage is historical research, selected after reviewing the same backtest,
and is not investment advice.
