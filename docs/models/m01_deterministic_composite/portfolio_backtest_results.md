# Model 01 portfolio-allocation results

## Research-use warning

These are historical research results, not investment advice, a live track
record, or a recommendation to buy or sell any ETF. The backtest uses estimated
expected returns, provider-adjusted historical prices, simplified transaction
costs, and a selected ETF universe. Future performance can differ materially.

The frozen methodology is documented in
[`portfolio_allocation.md`](portfolio_allocation.md), and the machine-readable
specification is
[`m01_regime_allocation_backtest.yaml`](../../../configs/models/m01_regime_allocation_backtest.yaml).
All numbers below come from the published artifacts under
[`results/published/m01_regime_allocation_backtest/`](../../../results/published/m01_regime_allocation_backtest/).

## 1. Frozen run

The formal performance sample contains 102 complete open-to-open holding
months:

- first reference month: January 2018;
- final reference month: June 2026;
- execution: first common US session adjusted open;
- exit: next month's first common US session adjusted open;
- baseline mean shrinkage: 24 pooled pseudo-months;
- covariance: pooled within-regime Ledoit-Wolf plus posterior between-regime
  mean covariance;
- causal ablation: pooled global expected mean with historical-frequency
  between-regime risk, under the same optimizer and execution rules;
- baseline annualized ex-ante volatility cap: 10%;
- transaction cost: five basis points per dollar bought or sold;
- allocation signal: causal `post_month_roll` current-month marginal;
- strategy universe: `SPY`, `IEF`, `TIP`, `HYG`, `BIL`, `GLD`, and `LQD`.
- comparison uncertainty: paired six-month circular block bootstrap with
  10,000 samples and seed `20260718`.

The July 2026 target is published separately but is excluded from performance
because its first-open-to-August-first-open holding return is incomplete.

## 2. Metric definitions

The following distinctions matter when interpreting the table:

- **Total return** compounds monthly net returns over the full sample.
- **CAGR** geometrically annualizes the net terminal wealth.
- **Annualized volatility** is the sample standard deviation of monthly net
  returns multiplied by \(\sqrt{12}\).
- **Zero-rate Sharpe** divides annualized arithmetic mean return by annualized
  volatility. It assumes a zero reference rate.
- **BIL-excess Sharpe** uses monthly portfolio return minus the aligned `BIL`
  return and divides by active-return volatility. It is therefore also an
  information ratio relative to `BIL`.
- **Sortino** uses zero as the minimum acceptable return and the root mean
  square of negative monthly returns as downside deviation.
- **Maximum drawdown** is calculated from daily marked NAV, including
  post-trade opens, adjusted closes, and terminal opens.
- **Calmar** is CAGR divided by the absolute maximum drawdown.
- **Annualized one-way turnover** is 12 times average monthly half-\(L^1\)
  turnover.
- **Annualized cost drag** is gross CAGR minus net CAGR. It is geometric and is
  not simply annual turnover multiplied by the posted cost rate.

The initial formation buys 100% of NAV, so its full \(L^1\) traded notional is
one. The reported half-\(L^1\) convention nevertheless records that initial
one-way turnover as 50%. Daily drawdown accounting begins at the post-cost
formation-open NAV: the five-basis-point initial cost remains in net return,
terminal wealth, and cost drag, but is not itself measured as a drawdown from a
pre-cost peak.

All methods use the same complete holding months and realized transaction-cost
accounting. The optimized target's *estimated* turnover penalty is causal: it
uses drift through the last adjusted close strictly before the
first-calendar-day signal. Realized backtest turnover and cost instead use the
actual weights drifted through the execution adjusted open. The difference is
the overnight or holiday-gap move that was unknowable when the target was
formed.

## 3. Headline comparison

| Method | Total return | CAGR | Ann. vol. | Sharpe, zero | Sharpe vs BIL | Sortino | Max drawdown | Calmar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Posterior optimized | **124.27%** | **9.97%** | 10.14% | 0.991 | **0.747** | 1.592 | -21.16% | **0.471** |
| Pooled-mean optimizer | 120.69% | 9.76% | 10.17% | 0.970 | 0.727 | 1.541 | -21.92% | 0.445 |
| Static 60% `SPY` / 40% `AGG` | 117.41% | 9.57% | 11.21% | 0.874 | 0.652 | 1.368 | -21.60% | 0.443 |
| Equal weight, seven assets | 65.13% | 6.08% | 6.37% | 0.960 | 0.573 | 1.569 | -14.77% | 0.411 |
| Legacy Sharpe-MAP | 50.11% | 4.89% | **4.72%** | **1.037** | 0.514 | **1.716** | **-11.07%** | 0.442 |

| Method | Worst month | Positive months | Ann. one-way turnover | Ann. cost drag |
|---|---:|---:|---:|---:|
| Posterior optimized | -9.44% | 67.65% | 39.46% | 4.34 bp |
| Pooled-mean optimizer | -9.94% | 67.65% | 18.25% | 2.00 bp |
| Static 60% `SPY` / 40% `AGG` | -10.04% | 67.65% | 16.35% | 1.79 bp |
| Equal weight, seven assets | -4.72% | **68.63%** | **15.31%** | **1.62 bp** |
| Legacy Sharpe-MAP | **-3.22%** | 67.65% | 108.67% | 11.41 bp |

The posterior-optimized portfolio produced the highest historical terminal
wealth and CAGR. Relative to the pooled-mean optimizer, however, it ended only
3.58 percentage points higher in cumulative return and 0.21 percentage point
higher in CAGR, with almost the same realized volatility and drawdown but more
than twice the annualized turnover and cost drag. Relative to static 60/40, it
ended 6.86 percentage points higher in cumulative return, with 0.40 percentage
point higher CAGR, 1.06 percentage points lower realized annualized volatility,
and a 0.44 percentage point shallower maximum drawdown. Its BIL-relative Sharpe
and Calmar ratios were the highest of the five methods.

That is not uniform dominance. The legacy Sharpe-MAP comparator had the highest
zero-rate Sharpe and Sortino, the lowest realized volatility, and the shallowest
drawdown, but at much lower absolute return and much higher turnover. Equal
weight also had materially lower volatility and drawdown than the optimized and
60/40 portfolios. The optimized portfolio's 0.991 zero-rate Sharpe was only
slightly above equal weight's 0.960 and below the legacy comparator's 1.037.

The results therefore support a narrower statement than "the regime strategy
won": in this sample, the optimized posterior portfolio had the highest
absolute and BIL-relative point estimates, while the defensive legacy
allocation delivered better protection and a slightly higher zero-rate
risk-adjusted ratio. The close pooled-optimizer result and the uncertainty
analysis below do not support a claim that posterior conditioning produced a
statistically reliable improvement.

## 4. Calendar-year path

The annual path shows that the terminal result did not arise from consistent
year-by-year superiority.

| Calendar year | Posterior optimized | Pooled-mean optimizer | Static 60/40 | Equal weight | Legacy Sharpe-MAP |
|---|---:|---:|---:|---:|---:|
| 2018 | -5.13% | -4.07% | -3.37% | -1.75% | -5.98% |
| 2019 | 24.11% | 23.29% | 23.65% | 14.80% | 15.72% |
| 2020 | 16.71% | 15.63% | 14.65% | 11.81% | 9.88% |
| 2021 | 8.86% | 6.91% | 15.54% | 3.16% | 3.01% |
| 2022 | -12.16% | -11.80% | -15.21% | -9.46% | -7.07% |
| 2023 | 14.59% | 14.89% | 16.17% | 9.09% | 6.91% |
| 2024 | 17.66% | 17.66% | 16.13% | 9.97% | 8.84% |
| 2025 | 24.05% | 24.08% | 13.51% | 15.58% | 11.38% |
| 2026 through June | 2.04% | 2.03% | 5.78% | 1.10% | 1.20% |

The optimized method led 60/40 in five of the nine displayed calendar slices,
including a smaller loss in 2022 and a large relative gain in 2025. It lagged
materially in 2021 and in the first half of 2026. Its worst month was March
2020 at -9.44%, followed by its best month in April 2020 at 9.13%. These sharp
adjacent outcomes are a reminder that monthly allocation cannot eliminate
short-horizon market risk.

## 5. Concentration and corner solutions

The optimized weights are much less diversified than the seven-asset label
might suggest. Across the 102 completed baseline months:

- `SPY` was exactly at its 35% individual cap in all 102 months;
- `BIL` was at zero in all 102 months;
- `GLD` was at its 25% cap in 88 months;
- `IEF` was at zero in 70 months;
- `TIP` was at zero in 71 months;
- the average number of positions above \(10^{-8}\) was 4.54;
- the average effective number of assets,
  \(1/\sum_i w_i^2\), was 3.51.

All 102 completed baseline decisions were published with an `optimal` solver
outcome, so these corners were not solver fallbacks. They are a structural
consequence of maximizing a linear expected-return estimate subject to hard
caps and a volatility ceiling. The group limits frequently make an individual
asset's effective ceiling tighter; for example, with `SPY` at 35%, the 50%
combined `SPY` plus `HYG` cap limits `HYG` to 15% whenever that group constraint
binds.

This concentration weakens a strong causal interpretation. The persistent
`SPY` and `GLD` cap positions indicate that stable estimated mean rankings and
constraints drove much of the target, not only month-to-month changes in
regime probabilities. The pooled-mean optimizer now holds those portfolio
mechanics fixed while removing current posterior conditioning. Its close
performance path and the bootstrap interval below show that the incremental
historical contribution of the posterior was small and imprecisely estimated.

## 6. Pooled-mean ablation and sampling uncertainty

The causal `pooled_mean_optimizer` uses the global expanding mean as expected
return and historical regime frequencies for between-regime covariance. It
uses the same sample, within-regime Ledoit-Wolf covariance, asset and group
caps, 10% volatility ceiling, optimizer, cost regularization, execution, and
realized accounting as the posterior method. It therefore isolates the current
posterior-conditioning step more closely than a static benchmark.

For each comparator, the published effect is 12 times the mean monthly net
return difference, posterior minus comparator. Uncertainty uses paired circular
six-month blocks, 10,000 resamples, a 95% percentile interval, and seed
`20260718`.

| Comparator | Ann. arithmetic mean advantage | 95% block-bootstrap interval | \(\Pr(\Delta>0)\) |
|---|---:|---:|---:|
| Pooled-mean optimizer | 0.187% | [-0.413%, 0.814%] | 0.737 |
| Static 60/40 | 0.255% | [-2.616%, 3.085%] | 0.572 |
| Equal weight | 3.935% | [1.454%, 6.342%] | 0.999 |
| Legacy Sharpe-MAP | 5.152% | [1.826%, 8.262%] | 0.999 |

The intervals versus the pooled optimizer and 60/40 both span zero. The
historical point estimates favor the posterior method, but they do **not**
support a claim of statistically reliable superiority over either comparator
at the stated 95% level. \(\Pr(\Delta>0)\) is a bootstrap fraction, not a
p-value. Even the positive intervals versus equal weight and the legacy rule
do not correct for multiple comparisons, strategy selection, or structural
change and should not be read as proof of future alpha.

## 7. Ex-ante cap versus realized volatility

The 10% volatility ceiling constrains estimated annualized covariance at each
decision. It is not a realized-volatility guarantee. The baseline's realized
annualized volatility was 10.14%, slightly above the 10% ex-ante cap. This is
not an accounting violation: realized returns differ from the covariance
forecast used at portfolio formation.

The volatility-cap sensitivity reinforces the distinction. Tightening the
ex-ante cap to 8% still produced 9.48% realized annualized volatility. Relaxing
the cap to 12% generated exactly the same published performance path as the
10% baseline, indicating that the wider constraint did not change the selected
historical targets. Position and group caps, rather than the volatility ceiling,
often determined the corners.

## 8. Mean-shrinkage sensitivity

The baseline was fixed at 24 pseudo-months before results. The alternatives are
diagnostics, not candidates selected after viewing performance.

| Pseudo-months | Total return | CAGR | Realized vol. | Sharpe, zero | Sharpe vs BIL | Max drawdown | Ann. turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 110.30% | 9.14% | **8.61%** | **1.063** | **0.777** | **-18.15%** | 67.81% |
| 12 | 116.97% | 9.54% | 10.00% | 0.965 | 0.718 | -21.16% | 48.15% |
| **24 baseline** | **124.27%** | **9.97%** | 10.14% | 0.991 | 0.747 | -21.16% | 39.46% |
| 48 | 122.61% | 9.87% | 10.16% | 0.981 | 0.738 | -21.16% | 33.49% |
| 96 | 119.91% | 9.71% | 10.09% | 0.973 | 0.728 | -22.05% | **23.72%** |

The baseline had the highest absolute return and CAGR in this grid, while the
unshrunk variant had the highest Sharpe ratios, lowest realized volatility, and
shallowest drawdown. Stronger shrinkage generally reduced turnover, but its
effect on return and drawdown was not monotonic. The 48-pseudo-month result was
close to the baseline in absolute performance, which argues against treating
24 as a uniquely identified optimum.

## 9. Volatility-cap sensitivity

| Ex-ante cap | Total return | CAGR | Realized vol. | Sharpe, zero | Max drawdown | Ann. turnover |
|---:|---:|---:|---:|---:|---:|---:|
| 8% | 106.34% | 8.90% | 9.48% | 0.950 | **-19.91%** | 48.99% |
| **10% baseline** | **124.27%** | **9.97%** | 10.14% | **0.991** | -21.16% | 39.46% |
| 12% | **124.27%** | **9.97%** | 10.14% | **0.991** | -21.16% | 39.46% |

The 8% variant reduced realized volatility and drawdown, but also reduced
historical return and Sharpe. The identical 10% and 12% results show that
loosening this constraint did not create a more aggressive path in the sample.

## 10. Position- and group-cap sensitivity

The multiplier scales all non-`BIL` individual caps and all group caps; the
`BIL` cap remains 100%. This changes the feasible portfolio set, not merely a
reported diagnostic.

| Cap multiplier | Total return | CAGR | Realized vol. | Sharpe, zero | Sharpe vs BIL | Max drawdown | Ann. turnover | Ann. cost drag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 97.70% | 8.35% | **8.91%** | 0.947 | 0.670 | **-19.01%** | **31.10%** | **3.37 bp** |
| **1.0 baseline** | 124.27% | 9.97% | 10.14% | 0.991 | 0.747 | -21.16% | 39.46% | 4.34 bp |
| 1.2 | **146.02%** | **11.17%** | 10.94% | **1.026** | **0.800** | -21.64% | 53.56% | 5.96 bp |

The return, risk, and turnover profile depends materially on the caps.
Tightening them reduced return, volatility, drawdown, and turnover; loosening
them raised return, volatility, and turnover. The strong 1.2-multiplier point
estimate is an in-sample sensitivity, not evidence that the looser policy is a
better ex-ante choice. This dependence reinforces that the baseline result is
jointly produced by return estimates and declared concentration constraints.

## 11. Transaction-cost and optimizer-policy sensitivity

The transaction-cost parameter has two roles: it is charged to realized
trading, and it penalizes estimated turnover inside the optimizer. These are
therefore joint optimizer-policy and realized-cost sensitivities, not isolated
execution-cost stresses. Changing the rate changes target weights; a zero-cost
run is not merely the baseline return with costs added back.

| One-way cost | Total return | CAGR | Realized vol. | Sharpe, zero | Max drawdown | Ann. turnover | Ann. cost drag |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 bp | 110.20% | 9.38% | 9.63% | 0.982 | -21.23% | 132.94% | 0.00 bp |
| **5 bp baseline** | **124.27%** | **9.97%** | 10.14% | **0.991** | **-21.16%** | 39.46% | 4.34 bp |
| 10 bp | 114.94% | 9.42% | 10.06% | 0.948 | -22.06% | **26.71%** | 5.85 bp |

Removing the penalty increased annualized turnover to 132.94% and changed the
portfolio enough to lower historical return despite charging no realized
cost. Doubling the penalty lowered turnover to 26.71%, but also lowered return
and risk-adjusted performance. In this sample, five basis points behaved as a
material turnover regularizer as well as a cost assumption.

## 12. July 2026 research target

The latest published target uses checkpoint `baseline:003909` from July 1,
2026. It is a live-only research target and is not included in the 102-month
performance table.

### 12.1 Regime probability input

| July 2026 regime | Probability |
|---|---:|
| Growth up / inflation up | 56.78% |
| Growth down / inflation up | 19.11% |
| Growth up / inflation down | 19.54% |
| Growth down / inflation down | 4.57% |

These are the July 1 `post_month_roll` probabilities used for allocation. They
are not the later July 16 event-updated posterior shown elsewhere in the
repository.

### 12.2 Target weights

| ETF | Target weight |
|---|---:|
| `SPY` | 35.00% |
| `IEF` | 0.00% |
| `TIP` | 0.00% |
| `HYG` | 15.00% |
| `BIL` | 0.00% |
| `GLD` | 25.00% |
| `LQD` | 25.00% |

The optimizer reported an expected one-month return of 0.694%, estimated
annualized volatility of 9.65%, and estimated rebalance cost of 0.19 basis
point. These are model estimates, not promised returns or risk limits. The
estimated cost uses the last adjusted close available before the signal; the
realized execution-open cost can differ.

The target binds the 35% `SPY` cap, 25% `GLD` cap, and 50% combined `SPY` plus
`HYG` cap. `IEF`, `TIP`, and `BIL` are at their lower bounds. The optimizer
reported an `optimal` outcome without using a fallback. The concentration is
therefore a feature of the expected-return ranking and constraints, not an
instruction to infer high confidence from the four-state posterior.

## 13. What can and cannot be concluded

The backtest establishes that the complete frozen pipeline can convert causal
regime probabilities into feasible targets, account for execution timing and
costs, and produce auditable historical comparisons. Within January 2018 to
June 2026, its optimized portfolio had the highest terminal return, CAGR,
BIL-relative Sharpe, and Calmar ratio among the five published methods. The
pooled-mean ablation was close, however, and its paired confidence interval
against the posterior method contains zero. The evidence therefore does not
support statistically reliable superiority over the pooled optimizer or
static 60/40.

It does **not** establish that:

- the posterior caused all incremental performance or reliably outperformed
  the pooled-mean optimizer;
- the ETF selection is free of survivorship or selection bias;
- the expected-return estimates will remain stable;
- a 10% estimated volatility cap will hold in realized data;
- adjusted-opening prices could be obtained without additional slippage;
- the best sensitivity row identifies a superior future parameter;
- the July 2026 target is suitable for any investor.

The sample contains a limited number of macro episodes and uses a compact US
ETF universe. Constant costs omit taxes, market impact, and time-varying
spreads. Public results should therefore be presented as a reproducible model
experiment with mixed trade-offs, not as a deployable product or evidence of
guaranteed alpha.

## 14. Published artifacts

- [`backtest_summary.json`](../../../results/published/m01_regime_allocation_backtest/backtest_summary.json):
  run metadata and complete performance records;
- [`performance_summary.csv`](../../../results/published/m01_regime_allocation_backtest/performance_summary.csv):
  comparable method-level metrics;
- [`monthly_returns.csv`](../../../results/published/m01_regime_allocation_backtest/monthly_returns.csv):
  gross returns, costs, net returns, turnover, and NAV by method and month;
- [`monthly_weights.csv`](../../../results/published/m01_regime_allocation_backtest/monthly_weights.csv):
  baseline, benchmark, sensitivity, and live target weights;
- [`sensitivity_metrics.csv`](../../../results/published/m01_regime_allocation_backtest/sensitivity_metrics.csv):
  one-parameter-at-a-time comparisons;
- [`comparison_uncertainty.csv`](../../../results/published/m01_regime_allocation_backtest/comparison_uncertainty.csv):
  paired block-bootstrap active-return estimates and confidence intervals;
- [`latest_allocation.json`](../../../results/published/m01_regime_allocation_backtest/latest_allocation.json):
  July 2026 posterior, target, expected moments, and binding constraints.
