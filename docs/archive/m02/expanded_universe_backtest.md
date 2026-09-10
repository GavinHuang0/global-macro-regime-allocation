# Archived Model 02 all-local-ETF universe diagnostic

## Decision and scope

This is an exploratory universe-expansion diagnostic. It does not replace the
promoted Model 02 allocation, refit the Bayesian posterior, or promote a new
strategy.

The test admits every ETF already present in the frozen local market-data
manifest:

`SPY`, `IEF`, `TIP`, `LQD`, `HYG`, `BIL`, `GLD`, `DBC`, `UUP`, `TLT`, `USO`,
and `AGG`.

All 12 have the same 4,664 trading dates from 2 January 2008 through 17 July
2026. The all-asset weekly schedule exactly matches all 446 frozen signal
weeks, so the expansion does not shorten the sample or change a holiday
execution date. No external data were retrieved for this first controlled
test. Adding funds after viewing the result would create another selection
layer; any broader universe should be declared in a separate dataset and
experiment.

The local set is sufficient for this controlled US growth/inflation
diagnostic. It spans equity beta, nominal duration, TIPS, credit, cash, gold,
broad commodities, oil, and the US dollar. It is not a complete production
global-macro universe: developed ex-US and emerging-market equities, equity
styles, non-US and emerging-market debt, real estate and infrastructure,
broader currencies, industrial metals, and agriculture remain absent.

## Frozen posterior and common test

The stage directly consumes, rather than reconstructs, the previously
published weekly posterior:

- upstream manifest SHA-256:
  `5821824dcfe6a82347d1df176467b74d0199cb7724bb08f5e07daf08801fbc4a`;
- signal-table SHA-256:
  `8cb4d27056ec5b024ad5e2cbf4549429da98def25b8602e91275731e8046111f`;
- price-history SHA-256:
  `7cbd321c5af2662065edfc3a5c3c1cdb6101bda10de7862bc09c09041bdcb7f1`;
  and
- ETF-universe manifest SHA-256:
  `b806212f0db312f0315ae8119021612f17f211a3314be0d1966582d033556a69`.

For the original seven assets, the largest recomputed expected-return
difference from the frozen optimizer audit is
$`9.9991\times10^{-17}`$. Static 60/40 gross returns, costs, net returns, and
turnover agree with the frozen run to less than $`10^{-16}`$. These controls
show that the posterior and execution schedule are unchanged.

Expanding a multivariate Ledoit--Wolf estimator from seven to twelve assets
also changes the fitted covariance of the original assets. Average shrinkage
falls from 0.03830 to 0.01758, and the average absolute relative change in
their diagonal variances is 2.92%. The reported universe delta therefore
combines access to five new securities with this dimension-induced change in
the risk estimator; it is not a pure admission effect.

The test retains weekly open-to-next-week-open estimation and holding returns,
260 minimum labeled weeks, 104 pseudo-weeks of regime-mean shrinkage, 52x
covariance annualization, a 10% annualized volatility cap, long-only full
investment, and five basis points of one-way cost. It evaluates only:

1. `posterior_optimized`;
2. `pooled_mean_optimizer`;
3. 1/12 in every ETF, rebalanced weekly; and
4. static 60% `SPY` / 40% `AGG`, rebalanced weekly.

The 60/40 comparator deliberately remains unchanged. Equal weight is
security-equal, not asset-class-equal; it therefore gives multiple slots to
overlapping duration and commodity exposures.

## Predeclared overlap controls

The original caps are retained. The new single-security caps are 25% `DBC`,
25% `UUP`, 25% `TLT`, 10% `USO`, and 50% `AGG`. The 10% oil cap reflects
`USO`'s high volatility and futures roll/path dependence.

The expanded group limits are:

| Group | Assets | Maximum |
|---|---|---:|
| Equity and high yield | `SPY`, `HYG` | 50% |
| Corporate credit | `HYG`, `LQD` | 50% |
| Broad rate sensitive | `IEF`, `TIP`, `LQD`, `TLT`, `AGG` | 75% |
| Nominal Treasury and aggregate | `IEF`, `TLT`, `AGG` | 60% |
| Commodity complex | `GLD`, `DBC`, `USO` | 40% |
| Energy and broad commodities | `DBC`, `USO` | 30% |

These securities are not 12 independent opportunities. Over the formal weekly
sample, correlations include 0.905 for `IEF`--`TLT`, 0.873 for `AGG`--`IEF`,
0.897 for `AGG`--`LQD`, and 0.811 for `DBC`--`USO`. The correlation matrix's
participation-ratio effective dimension is only 4.24.

## Backtest result

The formal window has 445 complete weeks from 1 January 2018 through 6 July
2026.

| Method | Total return | CAGR | Ann. vol. | Zero-rate Sharpe | BIL-excess Sharpe | Max drawdown | Ann. one-way turnover | Ann. cost drag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static 60/40 | 119.04% | **9.60%** | 11.62% | 0.847 | 0.633 | -22.02% | 28.08% | 3.08 bp |
| Posterior optimizer | 105.71% | 8.79% | 9.94% | **0.898** | **0.649** | -22.69% | 35.83% | 3.90 bp |
| Pooled optimizer | 101.13% | 8.51% | 10.13% | 0.857 | 0.612 | -23.92% | 30.04% | 3.26 bp |
| Equal weight | 60.05% | 5.65% | **6.83%** | 0.839 | 0.475 | **-15.15%** | 37.45% | 3.96 bp |

The posterior has the best realized risk-adjusted ratios, but static 60/40 has
the highest return. None of the posterior's paired comparisons is
statistically significant:

| Posterior minus comparator | Ann. mean difference | 95% interval | Bootstrap fraction above zero | Ann. tracking error | Information ratio |
|---|---:|---:|---:|---:|---:|
| Pooled optimizer | +0.245% | [-0.214%, +0.705%] | 0.8588 | 1.165% | 0.211 |
| Static 60/40 | -0.915% | [-4.291%, +2.366%] | 0.2985 | 5.545% | -0.165 |
| Equal weight | +3.198% | [-1.810%, +7.677%] | 0.9050 | 6.953% | 0.460 |

Intervals use 10,000 paired circular-block-bootstrap resamples, 26-week
blocks, a fixed seed of `20260718`, and net weekly returns.

## Did the broader universe help?

No. The point estimates are negative, and neither interval supports an
improvement over the frozen seven-ETF result:

| Method | Seven-ETF CAGR | Twelve-ETF CAGR | CAGR change | Ann. mean change | 95% interval for mean change |
|---|---:|---:|---:|---:|---:|
| Posterior optimizer | 9.86% | 8.79% | -1.07% | -1.024% | [-2.412%, +0.311%] |
| Pooled optimizer | 9.83% | 8.51% | -1.32% | -1.251% | [-2.882%, +0.288%] |
| Equal weight | 6.09% | 5.65% | -0.44% | -0.384% | [-3.941%, +3.172%] |
| Static 60/40 | 9.60% | 9.60% | 0.00% | 0.000% | numerical zero |

Posterior annualized cost drag increased by only 0.69 basis point and pooled
cost drag by 0.30 basis point. Trading cost is therefore not the explanation.

## Why the expansion hurt

The optimizer used almost none of the apparent breadth. Average optimized
weights were:

| Asset | Seven-ETF posterior | Twelve-ETF posterior | Seven-ETF pooled | Twelve-ETF pooled |
|---|---:|---:|---:|---:|
| `SPY` | 34.88% | 34.68% | 34.90% | 34.95% |
| `HYG` | 14.64% | 14.35% | 14.75% | 14.55% |
| `LQD` | 28.84% | 18.29% | 28.84% | 18.02% |
| `GLD` | 21.46% | 14.61% | 21.47% | 11.85% |
| `TLT` | -- | **17.95%** | -- | **20.63%** |

`DBC`, `UUP`, `USO`, and `AGG` received effectively zero optimized weight.
The expansion therefore mostly replaced `LQD` and `GLD` with `TLT`, rather
than adding diversified macro sleeves.

Across the 445 realized weeks, the causal expanding-history estimator expected
`TLT` to return 4.75% annualized under the posterior and 5.67% under pooled
mean. Its realized annualized arithmetic return was -0.82%. By contrast, `GLD`
realized 14.13% against expected values of 7.84% and 6.64%. The allocator
extrapolated the pre-2018 duration bull market and adapted slowly to the
2021--2022 rate break. `TLT` was also historically useful as a covariance
hedge, making it look attractive even though the duration group and
total-volatility caps did not bind.

This is a mean-estimation and representation failure more than a
transaction-cost failure. A linear expected-return objective still produced
corner portfolios with roughly four effective assets.

The posterior did make a useful relative adjustment: compared with pooled, it
held 2.67 percentage points less `TLT` and 2.76 points more `GLD` on average.
Its average half-$`L_1`$ target distance from pooled increased from about 1.22%
in the seven-ETF run to 4.42%, and realized tracking error rose from about
0.37% to 1.17%. This explains why the posterior-minus-pooled point estimate
grew from 1.87 to 24.55 basis points per year. The interval still includes
zero, so the larger expression is not yet a reliable edge.

## Recommended improvements

The next work should improve the allocator/payoff model before adding another
set of securities. The prior exact-regime oracle also lost significantly to
pooled mean within the existing architecture, so better quadrant
classification alone is unlikely to solve the problem.

Allocation changes, in priority order:

1. Allocate hierarchically to independent risk factors--equity beta, duration
   or DV01, credit spread, inflation, commodity, and USD--then permit small
   instrument-level curve or sector tilts. Use one core ETF per factor rather
   than treating `IEF`, `TLT`, and `AGG` as independent assets.
2. Replace raw expanding means in the optimizer with a robust pooled core:
   minimum variance or risk budgets, explicit expected-return uncertainty, and
   confidence-scaled posterior tilts. Shrink views toward zero excess return,
   not only regime means toward a pooled raw mean.
3. Constrain duration in risk units such as DV01 and commodity overlap in
   factor-risk units. Nominal ETF-weight caps did not prevent the stale
   duration bet.
4. Add no-trade bands and slower-moving strategic weights. Costs were small
   here, but they will matter more once views become dynamic.
5. Forecast returns over `BIL` and model instrument-specific economics:
   yield/carry and curve changes for bonds, roll and curve shape for commodity
   futures, and carry/valuation for currencies.

Bayesian architecture changes:

1. Replace hard quadrant-conditioned asset means with a hierarchical dynamic
   factor-return model whose coefficients can decay or change over time.
2. Use continuous growth/inflation levels, posterior changes, release
   surprises, and state uncertainty rather than only sign quadrants.
3. Carry expected-return parameter uncertainty into allocation so weak views
   receive smaller risk budgets automatically.
4. Test one-, two-, and four-week distributed response horizons in a locked,
   nested walk-forward design.
5. If external assets are added, predeclare a factor-complete universe and an
   asset-class-equal or risk-equal benchmark before observing results.

## Artifacts

- configuration:
  `configs/models/m02_expanded_universe_backtest.yaml`;
- content-hash manifest:
  `data/manifests/m02_expanded_universe_backtest.json`;
- published machine-readable results:
  `results/published/m02_expanded_universe_backtest/`; and
- builder:
  `python -m regime_allocation.cli.build_m02_expanded_universe_backtest`.
