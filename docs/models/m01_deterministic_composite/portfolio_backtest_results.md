# Model 01 portfolio backtest results

These are the frozen results for the promoted `posterior_optimized` allocation.
The formal net backtest contains 102 complete first-open-to-next-first-open
monthly holdings from January 2018 through June 2026. The July 2026 target is
published but excluded because its holding period is incomplete.

The contract is in [`portfolio_allocation.md`](portfolio_allocation.md).
Full-precision results are in the
[`performance summary`](../../../results/published/m01_regime_allocation_backtest/performance_summary.csv)
and
[`backtest summary`](../../../results/published/m01_regime_allocation_backtest/backtest_summary.json).

## Performance

| Method | Total return | CAGR | Annualized volatility | Zero-rate Sharpe | `BIL`-excess Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|
| `posterior_optimized` | **124.15%** | **9.96%** | 10.17% | **0.988** | **0.745** | -21.16% |
| `pooled_mean_optimizer` | 120.69% | 9.76% | 10.17% | 0.970 | 0.727 | -21.92% |
| `static_60_spy_40_agg` | 117.41% | 9.57% | 11.21% | 0.874 | 0.652 | -21.60% |
| `equal_weight` | 65.13% | 6.08% | 6.37% | 0.960 | 0.573 | **-14.77%** |

The posterior strategy's annualized one-way turnover is 38.24%, and its
modeled annualized cost drag is 4.21 basis points. Realized volatility can
exceed the 10% ex-ante cap because the constraint applies to estimated
formation-time covariance.

## Incremental posterior contribution

Paired uncertainty uses 10,000 circular block-bootstrap resamples with
six-month blocks:

| Comparator | Annualized mean difference | 95% interval | Bootstrap fraction above zero |
|---|---:|---:|---:|
| `pooled_mean_optimizer` | +0.184% | [-0.364%, +0.795%] | 73.95% |
| `static_60_spy_40_agg` | +0.252% | [-2.623%, +3.098%] | 57.22% |
| `equal_weight` | +3.931% | [+1.440%, +6.346%] | 99.90% |

The fractions are not $`p`$-values. Posterior and pooled monthly returns have
correlation 0.9973, and their interval includes zero. Because pooled mean
holds the estimator, constraints, execution, and costs fixed while removing
the current posterior, it is the cleanest allocation ablation. Model 01 does
not establish a statistically reliable posterior edge.

The interval versus equal weight is positive, but equal weight runs at much
lower realized risk and is not a clean inference ablation. The 60/40 interval
also includes zero.

Full comparison rows are in
[`comparison_uncertainty.csv`](../../../results/published/m01_regime_allocation_backtest/comparison_uncertainty.csv).

## Portfolio concentration

Across the 102 completed targets, `SPY` is at its 35% cap in every month,
`BIL` is always zero, and `GLD` is at its 25% cap in 89 months. `IEF` is zero
in 74 months and `TIP` in 88 months. The headline result is therefore strongly
shaped by the selected assets and caps rather than uniquely by posterior
variation.

## Latest target

At the 1 July 2026 `post_month_roll` signal,
$`p_{m,r\mid d}`$ was:

| Quadrant | Probability |
|---|---:|
| Growth up / inflation up | 57.18% |
| Growth down / inflation up | 18.98% |
| Growth up / inflation down | 19.28% |
| Growth down / inflation down | 4.56% |

The target is 35% `SPY`, 15% `HYG`, 25% `GLD`, and 25% `LQD`, with zero in
`IEF`, `TIP`, and `BIL`. Estimated annualized volatility is 9.65%, and the
estimated rebalance cost is 0.19 basis points. This target uses the 1 July
signal, not the later 16 July posterior.

Full precision and constraint metadata are in
[`latest_allocation.json`](../../../results/published/m01_regime_allocation_backtest/latest_allocation.json).

## Conclusion

The posterior allocation is historically competitive, but its advantage over
the matched pooled optimizer is economically small and statistically
inconclusive. Results remain sensitive to expected-return estimation, the
compact US ETF universe, binding concentration caps, adjusted-price history,
and simplified transaction costs.

Prespecified policy sensitivities remain available in
[`sensitivity_metrics.csv`](../../../results/published/m01_regime_allocation_backtest/sensitivity_metrics.csv);
they are diagnostics and do not replace the promoted baseline.
