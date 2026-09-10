# Model 02: promoted soft-composite model

Model 02 is the repository’s current published macro-inference and allocation
model. It is closed as the present research baseline but remains revisable
through a future versioned change; it is not frozen.

This model card reports only the promoted baselines and their essential
comparisons.

| Layer | Promoted method | Operating cadence |
|---|---|---|
| Inference | `student_t_7_reduced_core` | Monthly continuous macro state updated as releases arrive |
| Allocation | `posterior_optimized` | Monthly posterior sampled weekly; one-week return and holding horizon |

Model 02 does not make a weekly macro-state forecast. Its latent growth and
inflation state remains monthly. Weekly refers to posterior sampling, return
estimation, rebalancing, and the ETF holding period.

## Promoted inference

Model 02 retains four economically specified components on each axis:

- growth: payroll growth, industrial-production growth, real-consumption
  growth, and the negative unemployment-rate change;
- inflation: core CPI, core PCE prices, core finished-goods PPI, and average
  hourly earnings growth.

Each component uses its first eligible release, reads the current and prior
reference months from the same vintage, and is standardized against at least 60
earlier valid observations. Unlike Model 01, the equal-weight monthly scores
are not smoothed over three months.

Let the completed score be

```math
\boldsymbol z_m=(G_m,I_m)^\top.
```

The promoted filter maintains a four-month joint Gaussian state
$`\boldsymbol Z_{m-3:m}`$, rolls it with a causal expanding OLS VAR(1), updates
it as partial defining releases arrive, and conditions on
$`\boldsymbol z_m`$ exactly when all eight defining components are available.
Mapping uncertainty is added only when the Gaussian state is reported as four
growth–inflation quadrant probabilities.

Non-defining releases use fixed-$`\nu=7`$ Student-$`t`$ observation equations.
The promoted reduced-core graph contains:

1. `weekly_labor_stress`: the initial-claims innovation;
2. `consumer_real_implicit_joint`: real retail activity and its implicit-price
   coordinate; and
3. `business_activity_pipeline_joint`: core capital-goods activity and its
   orders-to-shipments pipeline coordinate.

Every fit uses information available strictly before the release being
processed. The within-day order is month roll, release-group update, then exact
score conditioning.

At the primary `before_any_defining_release` checkpoint, 183 target months from
January 2011 through May 2026 are scored:

| Metric | Promoted baseline mean |
|---|---:|
| Score-center NLPD | 26.854150 |
| Quadrant cross-entropy | 1.535693 |
| Unscaled quadrant Brier loss | 0.150466 |
| MAP quadrant accuracy | 0.377049 |

Relative to `transition_only`, the candidate-minus-reference changes are
-0.067415 NLPD, +0.000418 cross-entropy, +0.000860 Brier loss, and +0.016393
accuracy. Every corresponding 12-month block-bootstrap interval includes zero.
The promoted inference model therefore has no established statistical edge
over its essential benchmarks.

The latest published inference is dated 20 July 2026:

| Reference month | Growth up, inflation up | Growth down, inflation up | Growth up, inflation down | Growth down, inflation down |
|---|---:|---:|---:|---:|
| June 2026 | 0.177380 | 0.161240 | 0.354790 | 0.306591 |
| July 2026 | 0.159020 | 0.306653 | 0.266124 | 0.268204 |

The complete promoted inference contract and comparisons are in
[`inference.md`](inference.md).

## Promoted weekly allocation

Each calendar Monday, the allocation samples the newest monthly marginal after
any deterministic month roll and before same-day releases. It trades at the
first common adjusted open in that Monday-anchored week and exits at the next
week’s first common adjusted open.

The optimizer trades `SPY`, `IEF`, `TIP`, `HYG`, `BIL`, `GLD`, and `LQD`.
`AGG` is used only by the static 60/40 comparator. Causal weekly return moments
require at least 260 labeled weeks; quadrant means receive 104 pooled
pseudo-weeks; covariance is annualized by 52. The long-only optimizer has a 10%
ex-ante annualized volatility ceiling, individual and group caps, and a 5 bp
one-way transaction cost.

The formal sample contains 445 complete holding periods from 1 January 2018
through 6 July 2026:

| Method | Total return | CAGR | Ann. volatility | Zero-rate Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| Promoted posterior optimizer | 123.6021% | 9.8595% | 10.4170% | 0.95565 | -21.7545% |
| Pooled-mean optimizer | 123.0780% | 9.8294% | 10.5038% | 0.94597 | -21.9867% |
| Static 60% `SPY` / 40% `AGG` | 119.0394% | 9.5952% | 11.6221% | 0.84722 | -22.0205% |
| Equal weight | 65.8575% | 6.0906% | 6.3433% | 0.96434 | -14.6949% |

The posterior-minus-pooled annualized arithmetic mean difference is
+0.0187066%, with a 95% interval of [-0.0876424%, +0.1297494%]. The interval
includes zero. The posterior and pooled optimizers are economically almost
indistinguishable in this sample, so the backtest does not establish reliable
incremental value from the posterior.

The incomplete 13 July 2026 target is 35% `SPY`, 15% `HYG`, 25% `GLD`, and
25% `LQD`, with zero in `IEF`, `TIP`, and `BIL`.

The allocation contract and full retained comparison are in
[`portfolio_allocation.md`](portfolio_allocation.md) and
[`portfolio_backtest_results.md`](portfolio_backtest_results.md).

## Interpretation

The promoted inference graph and allocation policy were selected after
inspecting the same historical period. There is no untouched holdout, and none
of the reported comparisons establishes confirmatory out-of-sample
superiority.

The main limitations are:

- linear, time-homogeneous VAR dynamics and plug-in parameter estimates;
- approximate Gaussian state updates for robust Student-$`t`$ evidence;
- uncertain mapping from a monthly macro state to one-week ETF returns;
- repeated weekly training observations sharing one monthly quadrant;
- a compact, overlapping US ETF universe and few independent macro cycles;
- mutable provider-adjusted prices, idealized opening execution, and simplified
  transaction costs.

The outputs are research artifacts, not investment advice.

## Contracts, artifacts, and reproduction

Inference:

- [configuration](../../../configs/models/m02_current_baseline.yaml)
- [manifest](../../../data/manifests/m02_current_baseline.json)
- [published results](../../../results/published/m02_soft_composite/current/)

Allocation:

- [configuration](../../../configs/models/m02_regime_allocation_backtest.yaml)
- [manifest](../../../data/manifests/m02_regime_allocation_backtest.json)
- [published results](../../../results/published/m02_regime_allocation_backtest/)

Rebuild both promoted publications from the repository root:

```powershell
python -m regime_allocation.cli.build_m02_current_baseline --project-root .
python -m regime_allocation.cli.build_m02_backtest --project-root .
```

Non-promoted experiments and diagnostics remain available in the
[Model 02 development archive](../../archive/m02/README.md). None replaced the
promoted inference or allocation baseline.
