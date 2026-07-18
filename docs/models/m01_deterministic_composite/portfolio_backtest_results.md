# Model 01 portfolio backtest results

## Frozen scope

These results were rebuilt on 18 July 2026 after correcting the deterministic
archive-start policy, forecast-score eligibility, and daily NAV cost checkpoint.
The formal net backtest contains 102 complete open-to-open monthly holdings from
January 2018 through June 2026. The July 2026 target is published separately but
has no complete holding return and is excluded from every performance statistic.

The full allocation contract is in
[`portfolio_allocation.md`](portfolio_allocation.md). Exact public data are in
the [performance](../../../results/published/m01_regime_allocation_backtest/performance_summary.csv),
[monthly return](../../../results/published/m01_regime_allocation_backtest/monthly_returns.csv),
[weight](../../../results/published/m01_regime_allocation_backtest/monthly_weights.csv),
[uncertainty](../../../results/published/m01_regime_allocation_backtest/comparison_uncertainty.csv),
and [sensitivity](../../../results/published/m01_regime_allocation_backtest/sensitivity_metrics.csv)
files.

## Headline performance

| Method | Total return | CAGR | Ann. vol. | Zero-rate Sharpe | BIL-excess Sharpe | Sortino | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Posterior optimized | **124.15%** | **9.96%** | 10.17% | 0.988 | **0.745** | **1.585** | -21.16% |
| Pooled-mean optimizer | 120.69% | 9.76% | 10.17% | 0.970 | 0.727 | 1.541 | -21.92% |
| Static 60% `SPY` / 40% `AGG` | 117.41% | 9.57% | 11.21% | 0.874 | 0.652 | 1.368 | -21.60% |
| Equal weight | 65.13% | 6.08% | 6.37% | 0.960 | 0.573 | 1.569 | -14.77% |
| Legacy Sharpe-MAP | 49.61% | 4.85% | 4.83% | **1.007** | 0.495 | 1.664 | **-11.35%** |

The posterior optimizer has the highest historical absolute return, but not the
highest zero-rate Sharpe or shallowest drawdown. The table should not be read as
a ranking of ex-ante investment products: the methods have materially different
risk and concentration profiles.

The optimized method's annualized one-way turnover is 38.24%; equal weight is
15.32%, pooled optimizer 18.25%, legacy 103.17%, and static 60/40 16.35%.
Annualized modeled cost drag is 4.21 basis points for the posterior strategy.
The initial full purchase is included in returns and drawdown; the reported
half-$L^1$ one-way-turnover convention records it as 50% turnover.

## Incremental posterior attribution

The pooled-mean optimizer is the most informative comparator because it keeps
the return sample, covariance estimator, solver, position/group caps, volatility
ceiling, execution, and cost treatment fixed while removing the current regime
posterior.

| Comparator | Ann. arithmetic mean difference | 95% paired block-bootstrap interval | Bootstrap fraction above zero | Tracking error | Information ratio |
|---|---:|---:|---:|---:|---:|
| Pooled-mean optimizer | 0.184% | [-0.364%, 0.795%] | 73.95% | 0.75% | 0.244 |
| Static 60/40 | 0.252% | [-2.623%, 3.098%] | 57.22% | 4.58% | 0.055 |
| Equal weight | 3.931% | [1.440%, 6.346%] | 99.90% | 4.35% | 0.903 |
| Legacy Sharpe-MAP | 5.183% | [1.940%, 8.248%] | 99.92% | 5.80% | 0.894 |

Intervals use 10,000 paired circular block-bootstrap resamples with six-month
blocks. They quantify uncertainty in the mean monthly return difference under
that resampling design. They are not p-values, do not correct for architecture
selection, and do not make the returns independent.

Posterior and pooled-optimizer monthly returns correlate 0.9973. Their interval
crosses zero, so Model 01 does not establish a statistically reliable allocation
benefit from the current posterior. The large gaps versus lower-risk equal and
legacy portfolios mainly reflect different risk exposure, not a clean forecast
ablation.

## Concentration and binding policy

Across 102 completed targets:

- `SPY` is at its 35% cap in all 102 months;
- `BIL` is zero in all 102 months;
- `GLD` is at its 25% cap in 89 months;
- `IEF` is zero in 74 months and `TIP` is zero in 88 months.

The historical result is therefore heavily influenced by the selected ETF
universe and caps. The volatility ceiling is not always the active constraint;
increasing it from 10% to 12% leaves the baseline target path unchanged.

## Prespecified allocation sensitivities

| Variation | Total return | CAGR | Ann. vol. | Sharpe | Max drawdown | Ann. turnover |
|---|---:|---:|---:|---:|---:|---:|
| Baseline: 24 pseudo-months | 124.15% | 9.96% | 10.17% | 0.988 | -21.16% | 38.24% |
| Mean shrinkage 0 | 110.76% | 9.17% | 8.60% | **1.066** | **-18.10%** | 68.24% |
| Mean shrinkage 48 | 120.32% | 9.74% | 10.16% | 0.968 | -21.16% | 35.56% |
| Mean shrinkage 96 | 119.96% | 9.72% | 10.09% | 0.973 | -22.05% | 23.70% |
| Volatility cap 8% | 107.66% | 8.98% | 9.46% | 0.959 | -19.92% | 49.20% |
| Concentration caps x0.8 | 97.73% | 8.35% | 8.93% | 0.946 | -18.98% | 30.28% |
| Concentration caps x1.2 | **146.03%** | **11.17%** | 10.95% | 1.026 | -21.64% | 51.95% |

Looser caps raise both return and risk in this sample and must not be selected
post hoc as a superior policy. Cost variations are also policy variations:
changing the cost coefficient changes optimizer regularization **and** the
realized charge. The zero-cost variant therefore trades much more and has lower
historical total return than the baseline; it is not a pure fee subtraction.

## Latest allocation

At the 1 July 2026 `post_month_roll` signal, the current-month marginal was:

| Regime | Probability |
|---|---:|
| Growth up / inflation up | 57.18% |
| Growth down / inflation up | 18.98% |
| Growth up / inflation down | 19.28% |
| Growth down / inflation down | 4.56% |

The optimal target was 35% `SPY`, 15% `HYG`, 25% `GLD`, and 25% `LQD`, with
zero in `IEF`, `TIP`, and `BIL`. Estimated annualized volatility was 9.65% and
the modeled rebalance cost was 0.19 basis points. This month-start signal is not
the later 16 July posterior and is not included in the completed backtest.

## Interpretation and limitations

The posterior allocation was historically competitive, but Model 01's strongest
honest conclusion is negative: the backtest does not isolate a robust posterior
edge. The sample contains few macro cycles, the universe and constraints were
researcher-chosen, Yahoo history is a mutable adjusted snapshot, and the
transaction-cost model omits taxes, borrowing, market impact, and operational
failures. Results are a reproducible architecture benchmark for Model 02, not a
live performance claim.
