# Model 02 weekly portfolio backtest results

## Corrected frozen result scope

The formal net backtest contains 445 complete Monday-anchored open-to-open
weeks from 1 January 2018 through 6 July 2026. The 13 July 2026 target is
published separately because the frozen ETF snapshot ends on 17 July and does
not contain the following week's execution open. Full methodology is in
[`portfolio_allocation.md`](portfolio_allocation.md), with exact tables in the
[published result directory](../../../results/published/m02_regime_allocation_backtest/).

The forecast and holding horizons are now both one week. Return moments use
causally available weekly open-to-open observations, a 260-week minimum, 104
pseudo-weeks of regime-mean shrinkage, and 52x covariance annualization. The
optimizer evaluates expected one-week return net of one estimated rebalance
cost.

The previously published 108.79% total return and 8.98% CAGR are superseded.
Those figures came from a horizon mismatch: weekly holdings were optimized
with a monthly expected-return objective and monthly covariance scaling. The
replacement is a unit-alignment correction, not an improvement in Model 02's
signal quality; the promoted inference baseline and its causal probabilities
did not change.

## Headline performance

| Method | Total return | CAGR | Ann. vol. | Zero-rate Sharpe | BIL-excess Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Posterior optimized | **123.6021%** | **9.8595%** | 10.4170% | 0.95565 | **0.71690** | -21.7545% |
| Exploratory 75/25 pooled/posterior anchor | 123.2173866% | 9.8373991% | 10.4809091% | 0.9485128 | 0.7112097 | -21.9286402% |
| Pooled-mean optimizer | 123.0780% | 9.8294% | 10.5038% | 0.94597 | 0.70918 | -21.9867% |
| Static 60% `SPY` / 40% `AGG` | 119.0394% | 9.5952% | 11.6221% | 0.84722 | 0.63286 | -22.0205% |
| Equal weight | 65.8575% | 6.0906% | 6.3433% | 0.96434 | 0.57272 | -14.6949% |
| Legacy Sharpe-MAP | 54.0906% | 5.1822% | 5.1961% | **0.99882** | 0.52124 | **-11.1369%** |

The three benchmark paths are unchanged by the estimator correction. The
posterior strategy's annualized one-way turnover is 29.1922%, and its modeled
annualized cost drag is 3.2077 basis points. Realized annualized volatility can
exceed the 10% ex-ante model cap because the cap constrains the estimated
weekly covariance at formation, not future returns.

The exploratory `pooled_anchor_posterior_25pct` target is 75% of the pooled
target plus 25% of the posterior target each week. It is simulated as one
portfolio with its own drift and trades, paying five basis points one way on
its own turnover. Annualized one-way turnover is 27.1480535%, and modeled
annualized cost drag is 2.982445 basis points. The original posterior and
pooled-mean paths are unchanged. Because the anchor was introduced after
reviewing their corrected results, it is exploratory and non-promoted. Its
convex targets inherit their linear allocation constraints, but the blend has
not been separately audited against a 10% cap under one common covariance
estimate.

## Posterior attribution and uncertainty

The pooled-mean optimizer is the cleanest allocation ablation because it keeps
the weekly estimator, optimizer, constraints, dates, and costs while removing
the current posterior. Paired uncertainty uses 10,000 circular-block-bootstrap
resamples with 26-week blocks:

| Contrast | Ann. mean difference | 95% interval | Bootstrap fraction above zero |
|---|---:|---:|---:|
| Posterior minus pooled mean | +0.0187066% | [-0.0876424%, +0.1297494%] | 0.6281 |
| Posterior minus exploratory anchor | +0.01371398% | [-0.0660674%, +0.0969896%] | 0.6249 |
| Exploratory anchor minus pooled mean | +0.00499261% | [-0.0214632%, +0.0327264%] | 0.6376 |
| Posterior minus static 60/40 | +0.1085410% | [-2.7601925%, +2.9605243%] | 0.5332 |
| Posterior minus equal weight | +3.8378628% | [+1.5719340%, +6.0419220%] | 0.9991 |
| Posterior minus legacy Sharpe-MAP | +4.7649949% | [+1.4056826%, +7.9942040%] | 0.9962 |

The posterior-minus-pooled interval includes zero and its annualized tracking
error is only 0.37248%, so this sample does not establish a reliable
incremental return from the promoted posterior. The intervals against equal
weight and legacy Sharpe-MAP are above zero, but those methods have materially
different realized risk and are not clean posterior ablations. The interval
against static 60/40 also includes zero.

Both exploratory-anchor intervals include zero. The anchor sits between its
two source strategies by construction, and this sample does not distinguish
its incremental mean return from either one. Its 25% sleeve was chosen after
the corrected results were inspected, so these uncertainty estimates are
descriptive rather than confirmatory.

## Prespecified sensitivities

The baseline was not selected from the allocation sweep. Every row below
changes one policy parameter at a time; the complete precision and deltas are
in
[`sensitivity_metrics.csv`](../../../results/published/m02_regime_allocation_backtest/sensitivity_metrics.csv).

| Setting | Total return | CAGR | Ann. vol. | Zero-rate Sharpe | Ann. one-way turnover |
|---|---:|---:|---:|---:|---:|
| Baseline: 104 pseudo-weeks | 123.6021% | 9.8595% | 10.4170% | 0.95565 | 29.1922% |
| 0 pseudo-weeks | 100.3950% | 8.4618% | 10.0308% | 0.86077 | 41.9669% |
| 52 pseudo-weeks | 117.7748% | 9.5210% | 10.2304% | 0.94099 | 35.8156% |
| 208 pseudo-weeks | 123.6529% | 9.8624% | 10.4359% | 0.95436 | 29.0728% |
| 416 pseudo-weeks | 122.3281% | 9.7862% | 10.5077% | 0.94191 | 28.7417% |
| 8% volatility cap | 87.1072% | 7.5957% | 9.5208% | 0.81720 | 29.8673% |
| 12% volatility cap | 123.6583% | 9.8627% | 10.4184% | 0.95581 | 29.1373% |
| Zero transaction cost | 115.6110% | 9.3933% | 10.1197% | 0.93871 | 156.0514% |
| 10 bp one-way cost | 119.4842% | 9.6211% | 10.3851% | 0.93731 | 21.8151% |
| 0.8x concentration caps | 98.4801% | 8.3401% | 9.1295% | 0.92386 | 28.7346% |
| 1.2x concentration caps | 120.1987% | 9.6628% | 11.0409% | 0.89136 | 30.5440% |

The 208-pseudo-week and 12%-volatility-cap rows are nearly indistinguishable
from baseline, while zero or 52 pseudo-weeks and the tighter cap materially
reduce historical return. Removing the cost penalty changes the optimizer's
portfolio path and raises annualized turnover to 156.05%; it is therefore not
a simple subtraction of realized fees. These remain policy sensitivities, not
alternatives promoted after observing their results.

## Latest target

The 13 July 2026 pre-release Monday signal assigns probabilities 26.1970%,
34.5705%, 22.9228%, and 16.3098% to growth-up/inflation-up,
growth-down/inflation-up, growth-up/inflation-down, and
growth-down/inflation-down. The target is unchanged at 35% `SPY`, 15% `HYG`,
25% `GLD`, and 25% `LQD`, with zero in `IEF`, `TIP`, and `BIL`. Its estimated
annualized volatility is 9.85487%, estimated weekly return is 0.172453%, and
estimated one-rebalance cost is 0.0003815% of portfolio value. It is a
historical research output, not investment advice.

The exploratory anchor has the same 13 July target because the posterior and
pooled-mean source targets are identical that week. This coincidence does not
promote the anchor or alter the published posterior target.

The honest conclusion remains modest: after correcting the horizon mismatch,
the promoted Model 02 posterior slightly outperformed the otherwise identical
pooled-mean ablation, but the difference is economically tiny and not
statistically reliable. The result remains sensitive to the chosen universe,
caps, weekly return estimator, fixed cost assumption, and same-history model
promotion.
