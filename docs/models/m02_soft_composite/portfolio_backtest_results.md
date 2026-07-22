# Model 02 weekly portfolio backtest results

## Frozen result scope

The formal net backtest contains 445 complete Monday-anchored open-to-open
weeks from 1 January 2018 through 6 July 2026. The 13 July 2026 target is
published separately because the frozen ETF snapshot ends on 17 July and does
not contain the following week's execution open. Full methodology is in
[`portfolio_allocation.md`](portfolio_allocation.md), with exact tables in the
[published result directory](../../../results/published/m02_regime_allocation_backtest/).

## Headline performance

| Method | Total return | CAGR | Ann. vol. | Zero-rate Sharpe | BIL-excess Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Posterior optimized | 108.79% | 8.98% | 10.21% | 0.895 | 0.651 | -21.16% |
| Pooled-mean optimizer | **122.26%** | **9.78%** | 10.50% | 0.942 | **0.705** | -21.79% |
| Static 60% `SPY` / 40% `AGG` | 119.04% | 9.60% | 11.62% | 0.847 | 0.633 | -22.02% |
| Equal weight | 65.86% | 6.09% | 6.34% | 0.964 | 0.573 | -14.69% |
| Legacy Sharpe-MAP | 54.09% | 5.18% | 5.20% | **0.999** | 0.521 | **-11.14%** |

The posterior strategy's annualized one-way turnover is 50.22%, and its
modeled annualized cost drag is 5.48 basis points. The realized annualized
volatility can exceed the 10% ex-ante model cap because the cap constrains the
estimated monthly covariance at formation, not future returns.

## Posterior attribution and uncertainty

The pooled-mean optimizer is the cleanest allocation ablation because it keeps
the estimator, optimizer, constraints, dates, and costs while removing the
current posterior. The posterior-minus-pooled annualized arithmetic mean
difference is -0.761%, with a 95% 26-week circular-block-bootstrap interval of
[-2.321%, 0.144%] and bootstrap fraction above zero of 10.28%. The interval
includes zero, so this sample does not establish a reliable incremental return
from the promoted posterior.

Intervals against static 60/40 also include zero. Intervals against equal
weight and legacy Sharpe-MAP are above zero, but those methods have materially
different realized risk and are not clean posterior ablations.

## Prespecified sensitivities

The baseline was not selected from the allocation sweep. Larger shrinkage
values and looser concentration caps produce higher historical return in this
sample; the 96-pseudo-month and 1.2x-cap variants reach 121.64% and 125.01%
total return, respectively. Zero optimizer/realized transaction cost reaches
120.07% but trades far more. These are policy sensitivities, not alternatives
promoted after observing their results.

## Latest target

The 13 July 2026 pre-release Monday signal assigns probabilities 26.20%,
34.57%, 22.92%, and 16.31% to growth-up/inflation-up,
growth-down/inflation-up, growth-up/inflation-down, and
growth-down/inflation-down. The target is 35% `SPY`, 15% `HYG`, 25% `GLD`,
and 25% `LQD`, with zero in `IEF`, `TIP`, and `BIL`. Its estimated annualized
volatility is 9.62%. It is a historical research output, not investment advice.

The honest conclusion is modest: the promoted Model 02 posterior produced a
competitive constrained portfolio, but it underperformed the otherwise
identical pooled-mean ablation and did not isolate a statistically reliable
posterior edge. The result remains sensitive to the chosen universe, caps,
monthly return model, fixed cost assumption, and same-history model promotion.

