# Model 01 monthly portfolio allocation

The promoted allocation method is `posterior_optimized`, stage
`posterior_regime_allocation_backtest`. Its frozen machine-readable contract is
[`configs/models/m01_regime_allocation_backtest.yaml`](../../../configs/models/m01_regime_allocation_backtest.yaml);
results are in
[`portfolio_backtest_results.md`](portfolio_backtest_results.md).

## Signal, universe, and timing

Let \(t\) index monthly rebalances, \(m(t)\) the corresponding macro reference
month, and \(d_t\) the first calendar day of that month. The signal is the
baseline `post_month_roll` marginal:

\[
p_{m(t),r\mid d_t^-}
=\Pr(R_{m(t)}=r\mid\mathcal D_{d_t^-}),
\qquad r\in\mathcal R.
\]

The roll has occurred, but releases and confirmations dated \(d_t\) have not.
If \(d_t\) is not a trading day, the saved signal is unchanged and execution
waits for the first common session.

The strategy assets are `SPY`, `IEF`, `TIP`, `HYG`, `BIL`, `GLD`, and `LQD`.
`AGG` is available only to the 60/40 comparator. Prices are provider-adjusted
total-return proxies.

The target trades at the first common adjusted open and exits at the next
month's first common adjusted open. The resulting strategy-asset return vector
is \(\mathbf x_t\). Missing execution prices are not filled. A current target
may be published before its exit exists, but it is excluded from performance.

## Causal return estimation

A historical holding period enters rebalance \(t\)'s expanding sample only
when both its ending open and deterministic label were available strictly
before \(d_t\). All seven assets share one complete sample, and at least 60
labeled months are required.

Let \(n_{t,r}\) be the number of eligible months in quadrant \(r\),
\(\overline{\boldsymbol\mu}_{t,r}\) their arithmetic mean return vector, and
\(\overline{\boldsymbol\mu}_t\) the pooled mean. Regime means receive 24
pooled pseudo-months:

\[
\widetilde{\boldsymbol\mu}_{t,r}
=\frac{
n_{t,r}\overline{\boldsymbol\mu}_{t,r}
+24\overline{\boldsymbol\mu}_t
}{n_{t,r}+24}.
\]

An empty regime receives the pooled mean. Residuals around the unshrunk regime
sample means are pooled across regimes and passed to Ledoit–Wolf, producing
one shared monthly within-regime covariance \(\mathbf C_t\).

The promoted posterior moments are

\[
\boldsymbol\mu_t
=\sum_{r\in\mathcal R}
p_{m(t),r\mid d_t^-}\widetilde{\boldsymbol\mu}_{t,r},
\]

\[
\mathbf B_t
=\sum_{r\in\mathcal R}
p_{m(t),r\mid d_t^-}
(\widetilde{\boldsymbol\mu}_{t,r}-\boldsymbol\mu_t)
(\widetilde{\boldsymbol\mu}_{t,r}-\boldsymbol\mu_t)^\top,
\]

\[
\boldsymbol\Sigma_t=\mathbf C_t+\mathbf B_t.
\]

\(\mathbf B_t\) is the between-regime covariance implied by uncertainty in
conditional means. Covariance is annualized as
\(12\boldsymbol\Sigma_t\) for the risk constraint; the objective keeps
one-month expected-return units.

## Optimization

Let \(\mathbf w_t^{-}\) be the causal estimate of pretrade weights formed from
the previous target and the last adjusted close strictly before \(d_t\), and
let \(\mathbf w_t\) be the new target. The optimizer solves

\[
\max_{\mathbf w_t}
\quad
\boldsymbol\mu_t^\top\mathbf w_t
-\sum_a 0.0005\,|w_{t,a}-w_{t,a}^{-}|
\]

subject to long-only full investment,

\[
\mathbf 1^\top\mathbf w_t=1,\qquad \mathbf w_t\ge0,
\]

and

\[
\mathbf w_t^\top(12\boldsymbol\Sigma_t)\mathbf w_t\le0.10^2.
\]

Individual caps are:

| `SPY` | `IEF` | `TIP` | `HYG` | `BIL` | `GLD` | `LQD` |
|---:|---:|---:|---:|---:|---:|---:|
| 35% | 50% | 40% | 25% | 100% | 25% | 40% |

Group caps are:

| Group | Assets | Maximum |
|---|---|---:|
| Equity and high yield | `SPY`, `HYG` | 50% |
| Corporate credit | `HYG`, `LQD` | 50% |
| Rate sensitive | `IEF`, `TIP`, `LQD` | 75% |

SLSQP solutions are independently checked for feasibility. The fixed fallback
order is feasible pretrade holdings, constrained minimum variance, then all
`BIL`. The build fails if none is feasible.

## Execution cost and accounting

The target is formed without knowing the execution open. At that later open,
the previous holdings are drifted using realized asset returns to obtain the
actual pretrade vector \(\mathbf w_t^{-,\mathrm{exec}}\). Realized cost is

\[
K_t
=\sum_a0.0005
\left|w_{t,a}-w_{t,a}^{-,\mathrm{exec}}\right|.
\]

If \(g_t=\mathbf w_t^\top\mathbf x_t\), the net holding return is

\[
r_t^{\mathrm{net}}=(1-K_t)(1+g_t)-1.
\]

The initial formation trade is charged. Daily NAV records pretrade and
post-trade open values, so entry cost is included in return and drawdown.
Reported one-way turnover is half the \(L^1\) change in weights.

## Essential comparisons

`pooled_mean_optimizer` is the clean posterior ablation. It retains the causal
sample, optimizer, caps, costs, and execution, but uses
\(\overline{\boldsymbol\mu}_t\) instead of the current posterior-weighted mean.
Its between-regime covariance uses causal historical regime frequencies.

`equal_weight` holds \(1/7\) in each strategy asset and rebalances monthly.
`static_60_spy_40_agg` holds 60% `SPY` and 40% `AGG`. Both pay the same
realized cost schedule. The 60/40 benchmark has a different opportunity set
because `AGG` is benchmark-only.

The detailed publication retains additional diagnostics, but they are not
promoted allocation alternatives and are omitted from the main comparison.

## Evaluation

Performance uses complete net monthly returns with 12 periods per year.
Headline metrics are total return, CAGR, annualized volatility, zero-rate and
`BIL`-excess Sharpe, daily-NAV maximum drawdown, annualized turnover, and cost
drag.

For comparator \(b\), the annualized mean-return difference is

\[
\Delta^{(b)}
=12\,\overline{
r_t^{\mathrm{posterior}}-r_t^{(b)}
}.
\]

Uncertainty uses 10,000 paired circular block-bootstrap resamples with
six-month blocks and seed `20260718`. An interval containing zero does not
support a reliable mean-return edge under this design.

## Artifacts

Detailed outputs live under
`data/processed/m01_regime_allocation_backtest/`. Public outputs include:

- [Latest allocation](../../../results/published/m01_regime_allocation_backtest/latest_allocation.json)
- [Performance summary](../../../results/published/m01_regime_allocation_backtest/performance_summary.csv)
- [Monthly returns](../../../results/published/m01_regime_allocation_backtest/monthly_returns.csv)
- [Monthly weights](../../../results/published/m01_regime_allocation_backtest/monthly_weights.csv)
- [Paired uncertainty](../../../results/published/m01_regime_allocation_backtest/comparison_uncertainty.csv)
- [Sensitivity summary](../../../results/published/m01_regime_allocation_backtest/sensitivity_metrics.csv)
- [Manifest](../../../data/manifests/m01_regime_allocation_backtest.json)

## Limitations

Expected regime returns are noisy and may not persist. The linear objective
with hard caps can produce corner portfolios, and the 10% ceiling is an
estimated ex-ante constraint rather than a realized-risk guarantee. The ETF
universe is compact, overlapping, and US centric; adjusted market history is
not vintage data. Constant opening execution and five-basis-point costs omit
taxes, impact, borrowing, and operational failures. The finite sample contains
few independent macro cycles.
