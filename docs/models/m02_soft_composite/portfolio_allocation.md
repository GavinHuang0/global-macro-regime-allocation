# Model 02 promoted weekly allocation

## Status and scope

`posterior_optimized` is the current promoted Model 02 allocation method. It
consumes the promoted `student_t_7_reduced_core` inference probabilities and
does not refit or alter the inference model. The allocation is published for
continued research use and may be revised in a future version; Model 02 is not
frozen.

The machine-readable contract is
[`m02_regime_allocation_backtest.yaml`](../../../configs/models/m02_regime_allocation_backtest.yaml),
and the stage ID is `weekly_posterior_regime_allocation_backtest`.

The macro state remains monthly. The posterior is sampled weekly, while the
return-estimation, rebalance, and holding periods are all one week. “Weekly
allocation” therefore does not mean that the inference model forecasts a
weekly macro regime.

## Universe

The optimizer trades seven US-listed ETFs:

| ETF | Primary exposure | Maximum weight |
|---|---|---:|
| `SPY` | US large-cap equity | 35% |
| `IEF` | Intermediate nominal Treasury duration | 50% |
| `TIP` | Inflation-linked Treasury exposure | 40% |
| `HYG` | High-yield corporate credit | 25% |
| `BIL` | Treasury-bill and cash proxy | 100% |
| `GLD` | Gold | 25% |
| `LQD` | Investment-grade corporate credit | 40% |

The group caps are 50% for `SPY` plus `HYG`, 50% for `HYG` plus `LQD`, and
75% for `IEF` plus `TIP` plus `LQD`. `AGG` is available only to the static
60/40 comparator.

Provider-adjusted ETF prices are used as total-return proxies. They are not
point-in-time price vintages and may change after provider corrections.

## Weekly signal and execution

Let $t$ index a Monday-anchored holding period, $m(t)$ be the calendar month
containing that Monday, $d_t^{\mathrm{sig}}$ the Monday signal date, and
$d_t^{\mathrm{exe}}$ the first common adjusted open in that week.

The inference replay is sampled immediately after a deterministic month roll,
when applicable, and before every same-day release, partial defining update,
exact-score observation, or probability-map update. The allocation probability
vector is

$$
\boldsymbol p_{m(t)\mid d_t^-}
=\left(p_{m(t),r\mid d_t^-}\right)_{r\in\mathcal R},
$$

in the canonical quadrant order documented in
[`inference.md`](inference.md). The $d_t^-$ cutoff means that no information
published on the Monday signal date is used.

The portfolio executes at $d_t^{\mathrm{exe}}$ and exits at
$d_{t+1}^{\mathrm{exe}}$. A Monday market holiday therefore delays execution
without adding Monday releases to the target. A live target whose next
execution open is unavailable is published but excluded from performance.

For asset $a$, the simple weekly holding return is

$$
x_{t,a}
=\frac{P_{a,d_{t+1}^{\mathrm{exe}}}^{\mathrm{adj,open}}}
       {P_{a,d_t^{\mathrm{exe}}}^{\mathrm{adj,open}}}-1.
$$

Missing execution prices are not forward-filled.

## Causal return estimation

Each historical weekly return receives the retrospective quadrant implied by
the signs of the completed Model 02 scores for $m(t)$. Exact zero belongs to
the nonnegative side. This hard quadrant is used only to train return moments;
the current allocation always integrates the full soft probability vector.

A historical week enters a fit only when both its ending execution open and
the completed score’s `score_available_at` date are available through the
Sunday before the current Monday signal. Every fit requires at least 260 common
labeled weeks.

Let $\overline{\boldsymbol\mu}_t$ be the pooled weekly mean,
$\overline{\boldsymbol\mu}_{t,r}$ the sample mean for quadrant $r$, and
$n_{t,r}$ its eligible count. The conditional mean uses 104 pooled
pseudo-weeks:

$$
\widetilde{\boldsymbol\mu}_{t,r}
=\frac{
n_{t,r}\overline{\boldsymbol\mu}_{t,r}
+104\overline{\boldsymbol\mu}_t
}{
n_{t,r}+104
}.
$$

Residuals around the unshrunk quadrant means produce one shared Ledoit–Wolf
within-quadrant covariance $\boldsymbol C_t$. The posterior return moments are

$$
\boldsymbol\mu_t^{\mathrm{post}}
=\sum_{r\in\mathcal R}
p_{m(t),r\mid d_t^-}\widetilde{\boldsymbol\mu}_{t,r},
$$

$$
\boldsymbol B_t^{\mathrm{post}}
=\sum_{r\in\mathcal R}p_{m(t),r\mid d_t^-}
\left(
\widetilde{\boldsymbol\mu}_{t,r}-\boldsymbol\mu_t^{\mathrm{post}}
\right)
\left(
\widetilde{\boldsymbol\mu}_{t,r}-\boldsymbol\mu_t^{\mathrm{post}}
\right)^\top,
$$

$$
\boldsymbol\Sigma_t^{\mathrm{week}}
=\boldsymbol C_t+\boldsymbol B_t^{\mathrm{post}},
\qquad
\boldsymbol\Sigma_t^{\mathrm{ann}}
=52\boldsymbol\Sigma_t^{\mathrm{week}}.
$$

The expected return in the objective remains in one-week units.

## Optimization and costs

Let $\mathbf w_t$ be the target and
$\widehat{\mathbf w}_t^-$ the causal pretrade estimate formed from the
previous execution open and the last adjusted close strictly before the Monday
signal. The optimizer solves

$$
\max_{\mathbf w_t}
\quad
\left(\boldsymbol\mu_t^{\mathrm{post}}\right)^\top\mathbf w_t
-\sum_a 0.0005
\left|w_{t,a}-\widehat w_{t,a}^-\right|
$$

subject to full investment, long-only weights, the individual and group caps
above, and

$$
\mathbf w_t^\top
\boldsymbol\Sigma_t^{\mathrm{ann}}
\mathbf w_t
\le 0.10^2.
$$

The 10% annualized volatility ceiling constrains the causal covariance estimate;
it does not guarantee realized volatility. SLSQP solutions are independently
checked. The fallback order is feasible pretrade holdings, constrained minimum
variance, then 100% `BIL`; the run fails if none is feasible.

At the execution open, let $\mathbf w_t^{-,\mathrm{exe}}$ be the previous
holdings after realized drift. The charged cost and net weekly return are

$$
K_t
=\sum_a0.0005
\left|w_{t,a}-w_{t,a}^{-,\mathrm{exe}}\right|,
$$

$$
r_t^{\mathrm{net}}
=(1-K_t)
\left(1+\mathbf w_t^\top\mathbf x_t\right)-1.
$$

The initial formation trade is charged. Reported one-way turnover is half the
$L^1$ traded notional.

## Essential comparisons

All comparators use the same weekly execution calendar and cost accounting:

- `pooled_mean_optimizer` keeps the estimator, optimizer, covariance,
  constraints, and costs but removes the current posterior from the return
  moments;
- `equal_weight` holds $1/7$ in each strategy ETF and rebalances weekly; and
- `static_60_spy_40_agg` holds 60% `SPY` and 40% `AGG` and rebalances weekly.

The pooled optimizer is the cleanest test of the posterior’s incremental
allocation value because it changes the least.

Performance uses 52 periods per year. Paired uncertainty uses 10,000 circular
block-bootstrap resamples with 26-week blocks. These weekly settings are policy
equivalents under the 12/52 annualization convention, not exact calendar
equivalents of the Model 01 monthly settings.

## Artifacts and limitations

- [configuration](../../../configs/models/m02_regime_allocation_backtest.yaml)
- [manifest](../../../data/manifests/m02_regime_allocation_backtest.json)
- [latest allocation](../../../results/published/m02_regime_allocation_backtest/latest_allocation.json)
- [performance summary](../../../results/published/m02_regime_allocation_backtest/performance_summary.csv)
- [weekly returns](../../../results/published/m02_regime_allocation_backtest/weekly_returns.csv)
- [weekly weights](../../../results/published/m02_regime_allocation_backtest/weekly_weights.csv)
- [paired uncertainty](../../../results/published/m02_regime_allocation_backtest/comparison_uncertainty.csv)

Rebuild from the repository root:

```powershell
python -m regime_allocation.cli.build_m02_backtest --project-root .
```

The main limitations are uncertain expected returns, repeated weekly
observations sharing one monthly training quadrant, few independent macro
cycles, a compact and overlapping ETF universe, mutable adjusted-price history,
idealized opening execution, fixed costs without market impact or taxes, and
plug-in inference and return parameters. The output is research, not investment
advice.
