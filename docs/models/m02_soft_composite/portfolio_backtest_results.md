# Model 02 promoted weekly allocation results

## Status and sample

These are the current published results for the promoted
`posterior_optimized` allocation. They may be revised in a future Model 02
version; Model 02 is not frozen.

The formal net backtest contains 445 complete Monday-anchored,
adjusted-open-to-next-week-adjusted-open holding periods from 1 January 2018
through 6 July 2026. The 13 July 2026 target is published separately because
its next execution open is unavailable in the market-data snapshot.

The macro state is monthly and is sampled each Monday before same-day releases.
The return forecast, rebalance, and holding period are one week. Return moments
use at least 260 causally labeled weeks, 104 pooled pseudo-weeks of
quadrant-mean shrinkage, and 52× covariance annualization.

Full methodology is in
[`portfolio_allocation.md`](portfolio_allocation.md).

## Performance

All figures are net of the declared 5 bp one-way transaction cost.

| Method | Total return | CAGR | Ann. volatility | Zero-rate Sharpe | `BIL`-excess Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Promoted posterior optimizer | 123.6021% | 9.8595% | 10.4170% | 0.95565 | 0.71690 | -21.7545% |
| Pooled-mean optimizer | 123.0780% | 9.8294% | 10.5038% | 0.94597 | 0.70918 | -21.9867% |
| Static 60% `SPY` / 40% `AGG` | 119.0394% | 9.5952% | 11.6221% | 0.84722 | 0.63286 | -22.0205% |
| Equal weight | 65.8575% | 6.0906% | 6.3433% | 0.96434 | 0.57272 | -14.6949% |

The posterior optimizer’s annualized one-way turnover is 29.1922%, and its
modeled annualized cost drag is 3.2077 bp. Realized volatility can exceed the
10% ex-ante ceiling because the constraint applies to the estimated covariance
at formation, not future returns.

## Incremental posterior evidence

The paired effect is the posterior strategy’s annualized arithmetic mean net
return minus the comparator’s corresponding mean. It is not a CAGR difference.
Intervals use 10,000 paired circular block-bootstrap resamples with 26-week
blocks.

| Comparator | Annualized mean difference | 95% interval | Bootstrap fraction above zero |
|---|---:|---:|---:|
| Pooled-mean optimizer | +0.0187066% | [-0.0876424%, +0.1297494%] | 0.6281 |
| Static 60/40 | +0.1085410% | [-2.7601925%, +2.9605243%] | 0.5332 |
| Equal weight | +3.8378628% | [+1.5719340%, +6.0419220%] | 0.9991 |

The posterior-minus-pooled interval includes zero, annualized tracking error is
only 0.37248%, and the information ratio is 0.05022. This is the cleanest
comparison because the pooled strategy preserves the same estimator, optimizer,
constraints, dates, and costs while removing the current posterior. The sample
does not establish a statistically reliable incremental posterior return.

The interval against static 60/40 also includes zero. The interval against
equal weight is above zero, but equal weight has materially different realized
risk and is not a clean posterior ablation. The bootstrap fraction above zero
is descriptive and is not a $p$-value.

## Latest published target

The 13 July 2026 signal is the current incomplete holding period. Its
current-month quadrant probabilities, in canonical order, are:

| Quadrant | Probability |
|---|---:|
| Growth up, inflation up | 26.1970% |
| Growth down, inflation up | 34.5705% |
| Growth up, inflation down | 22.9228% |
| Growth down, inflation down | 16.3098% |

The promoted target is 35% `SPY`, 15% `HYG`, 25% `GLD`, and 25% `LQD`, with
zero in `IEF`, `TIP`, and `BIL`. Its estimated annualized volatility is
9.85487%, estimated one-week return is 0.172453%, and estimated one-rebalance
cost is 0.0003815% of portfolio value.

This target used the Monday pre-release posterior. It must not be mixed with a
later within-week inference snapshot.

## Interpretation and limitations

The promoted posterior allocation produced the highest total return and CAGR
among the four retained methods, but its edge over the pooled optimizer was
economically tiny and statistically inconclusive. Absolute results are shaped
primarily by the selected period, compact universe, concentration limits, and
return estimator.

The inference graph and allocation policy were developed on the same historical
period. There is no untouched holdout. Other limitations include mutable
adjusted-price history, repeated weekly observations sharing monthly quadrant
labels, few independent macro cycles, idealized opening execution, fixed
transaction costs, and plug-in parameter estimates. Results are research
outputs, not investment advice.

## Published artifacts

- [latest allocation](../../../results/published/m02_regime_allocation_backtest/latest_allocation.json)
- [performance summary](../../../results/published/m02_regime_allocation_backtest/performance_summary.csv)
- [paired uncertainty](../../../results/published/m02_regime_allocation_backtest/comparison_uncertainty.csv)
- [weekly returns](../../../results/published/m02_regime_allocation_backtest/weekly_returns.csv)
- [weekly weights](../../../results/published/m02_regime_allocation_backtest/weekly_weights.csv)
- [backtest summary](../../../results/published/m02_regime_allocation_backtest/backtest_summary.json)
