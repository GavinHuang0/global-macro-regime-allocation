# Model 01 portfolio allocation and backtest specification

## 1. Scope and frozen specification

This document defines the portfolio-construction and backtest stage attached to
Model 01. The machine-readable source of truth is
[`m01_regime_allocation_backtest.yaml`](../../../configs/models/m01_regime_allocation_backtest.yaml).
The stage ID is `posterior_regime_allocation_backtest`.

The backtest asks one narrow question:

> Given the causal probability distribution over the regime of the month about
> to be traded, can a constrained portfolio formed from historically estimated
> regime returns improve on transparent static and legacy allocation rules?

It does not change Model 01's regime definition, transition model, release
likelihoods, or Bayesian filter. It consumes archived filter checkpoints and
uses the same saved probability vector that would have existed at the
historical decision date. All model fitting, portfolio formation, and cost
accounting are walk-forward.

This is a frozen research specification, not an investment recommendation. No
result is stated in this methodology document.

### 1.1 Notation

The upstream notation follows
[`bayesian_filter.md`](bayesian_filter.md#notation): $m$ is the target
reference month, $\ell$ is another historical reference month, $d$ is a
knowledge date, $\mathcal D_d$ is the information available by that date,
and $R_m\in\mathcal R$ is the month-$m$ regime. The candidate-path symbol
$s$ remains reserved for the upstream four-month filter. In this document,
$a$ and $a'$ index assets, $d_m^{\mathrm{sig}}$ is the month-start signal
date, and $d_m^{\mathrm{exe}}$ is the first common trading session used for
execution. Bold lowercase letters denote asset vectors and bold uppercase
letters denote matrices.

## 2. Investment universe

### 2.1 Strategy assets

The long-only optimizer trades seven US-listed, USD-denominated ETFs:

| ETF | Primary role in the compact universe | Important caveat |
|---|---|---|
| `SPY` | US large-cap equity and broad growth-risk exposure | US equity is only one part of global growth risk |
| `IEF` | Intermediate nominal US Treasury duration | Sensitive to nominal yields and monetary policy, not only growth |
| `TIP` | US inflation-linked Treasury exposure | Returns also depend materially on real yields and indexation mechanics |
| `HYG` | Below-investment-grade corporate credit | Contains both spread risk and interest-rate risk; can behave like equity in stress |
| `BIL` | Treasury-bill ETF and portfolio cash proxy | It is an ETF with expenses and small price risk, not a frictionless cash account |
| `GLD` | Gold exposure | Has no contractual yield and can react to real rates, USD moves, and risk demand |
| `LQD` | Investment-grade corporate credit | Combines spread exposure with substantial nominal duration |

The selection is intentionally compact. It provides equity, nominal-rate,
real-rate, high-yield credit, investment-grade credit, cash-like, and gold
exposures without making the initial model depend on a large universe or a
high-dimensional covariance estimate.

The seven assets are not an exhaustive FICC or global-macro opportunity set.
In particular, the strategy omits explicit currency, commodity-basket, energy,
emerging-market, international-equity, and long-duration Treasury positions.
The selected assets also overlap economically: `IEF`, `TIP`, and `LQD` contain
rate risk, while `SPY` and `HYG` can share growth and risk-appetite exposure.
The covariance model and group caps address some overlap but cannot make the
instruments independent.

### 2.2 Benchmark-only asset

`AGG` is retained solely as the bond leg of the conventional static 60/40
benchmark. It is not available to the optimized strategy. This choice makes
the benchmark recognizable as 60% US equities and 40% broad US
investment-grade bonds, but it also means that the 60/40 comparator does not
use exactly the seven-asset strategy universe. Results must disclose this
difference rather than describing every comparator as an identical-universe
portfolio.

### 2.3 Return data

Portfolio returns use provider-adjusted ETF prices. Splits and cash
distributions are already represented in the adjusted price series and must
not be added again. Adjusted prices are convenient total-return proxies, but
they are back-adjusted by the provider and may change after later corrections
or distributions. They are not point-in-time price vintages.

## 3. Causal regime signal

### 3.1 Selected checkpoint

For holding month $m$, the allocation signal is the Model 01 `baseline`
checkpoint satisfying all of the following:

- checkpoint type: `post_month_roll`;
- marginal type: `path`;
- relative month: `0`;
- reference month: $m$;
- four regimes in the canonical order recorded in the configuration;
- probability sum within $10^{-10}$ of one.

Let the selected probability vector be

$$
p_{m,r}
=\Pr(R_m=r\mid\mathcal D_{(d_m^{\mathrm{sig}})^-}),
\qquad r\in\mathcal R,
$$

where $d_m^{\mathrm{sig}}$ is the first calendar day of month $m$,
$\mathcal R$ is the four-regime set, and
$\mathcal D_{(d_m^{\mathrm{sig}})^-}$ is the causal information carried into
the month roll immediately before that day's operations. The vector containing
the four entries is written $\boldsymbol p_m$.

The `post_month_roll` checkpoint is phase one on the first calendar day. The
previous four-month joint posterior has already been shifted and propagated
through a transition matrix trained through the prior day. Same-day releases
and deterministic confirmations have not yet been processed. Consequently,
the signal contains all earlier release evidence through the prior posterior,
but it does not use data published later on the signal date or later during the
holding month.

If the first calendar day is a weekend or market holiday, the signal remains
the archived first-calendar-day checkpoint and execution waits for the first
common trading session. Releases between those two times are deliberately not
added. This conservative convention keeps the historical and live decision
rules identical.

### 3.2 Why the current-month marginal is used

The strategy trades the month that the propagated marginal describes. It does
not select the maximum-probability state and does not discard uncertainty. The
entire vector $\boldsymbol p_m$ enters the expected-return and covariance
calculations in Section 6.

The joint path remains essential to the upstream filter because delayed
releases can revise earlier months and propagate forward. Portfolio
construction needs only the saved current-month marginal after that joint
filtering and propagation have occurred.

## 4. Execution and holding-period return

Let $d_m^{\mathrm{exe}}$ be the first trading session in month $m$ for which
every required asset has an adjusted-open observation. The target is executed
at the adjusted open on $d_m^{\mathrm{exe}}$ and held until the adjusted open
on $d_{m+1}^{\mathrm{exe}}$.

For asset $a$, the simple monthly holding return is

$$
x_{m,a}
=\frac{P^{\mathrm{adj,open}}_{a,d_{m+1}^{\mathrm{exe}}}}
       {P^{\mathrm{adj,open}}_{a,d_m^{\mathrm{exe}}}}-1.
$$

The vector of all seven asset returns is $\boldsymbol x_m$. The symbol
$P^{\mathrm{adj,open}}_{a,d}$ means asset $a$'s provider-adjusted opening
price on trading date $d$.

This first-open-to-next-first-open definition aligns estimation and realized
performance. It prevents a target formed at the beginning of a month from
receiving any return earned before execution. Assets must share a complete
execution calendar; missing observations are not forward-filled.

The formal evaluation begins with the January 2018 holding month. It ends with
the latest month for which the following month's first common adjusted open is
available. A target can be published for the current incomplete month, but
that month is excluded from performance until its exit open exists.

## 5. Causal estimation sample

At each month-start signal, return parameters are refit on an expanding sample.
A historical holding month $\ell$ is eligible only when both of these facts
were available by the recorded prior-day knowledge cutoff:

1. its open-to-open return $\boldsymbol x_\ell$, which becomes observable at
   $d_{\ell+1}^{\mathrm{exe}}$;
2. its deterministic first-release regime label $R_\ell$, whose
   `label_available_at` date is stored in the public regime history.

Because the cutoff is the day before the new month's signal, these non-strict
comparisons to the cutoff are equivalent to requiring both return and label to
be known strictly before the signal date. The month about to be traded cannot
enter its own estimation sample.

All seven strategy assets use one common monthly sample. No asset-specific
sample sizes or missing-return imputations are permitted. At least 60 common,
causally labeled holding months are required before the production estimator
is fit.

## 6. Regime return model

Let $N_m$ be the number of eligible training months at signal $m$, and let
$n_{m,r}$ be the number assigned to regime $r$. All vectors below have one
element per strategy asset.

### 6.1 Global and regime sample means

Let $\mathcal T_m$ be the set of eligible historical holding months available
at the month-$m$ signal. The pooled sample mean is

$$
\overline{\boldsymbol\mu}_m
=\frac{1}{N_m}\sum_{\ell\in\mathcal T_m}\boldsymbol x_\ell.
$$

For a nonempty regime, the unshrunk sample mean is

$$
\overline{\boldsymbol\mu}_{m,r}
=\frac{1}{n_{m,r}}
 \sum_{\ell\in\mathcal T_m:R_\ell=r}\boldsymbol x_\ell.
$$

Both are seven-element vectors of arithmetic monthly sample means:
$\overline{\boldsymbol\mu}_m$ pools all eligible months, whereas
$\overline{\boldsymbol\mu}_{m,r}$ uses only months labeled $r$.

### 6.2 Pseudo-month shrinkage

The production regime mean uses $\kappa=24$ pooled pseudo-months:

$$
\widetilde{\boldsymbol\mu}_{m,r}
=\frac{n_{m,r}\overline{\boldsymbol\mu}_{m,r}
+\kappa\overline{\boldsymbol\mu}_m}
       {n_{m,r}+\kappa},
\qquad \kappa=24.
$$

The value 24 means that the pooled mean carries the same algebraic weight as
24 observations. It is not 24 duplicated return rows. A well-populated regime
remains close to its own sample mean; a sparse regime is pulled toward the
pooled mean. If a regime has no eligible months, its production mean is the
pooled mean.

### 6.3 Shared within-regime covariance

For every eligible month, define a residual around the corresponding
*unshrunk* regime sample mean:

$$
\boldsymbol\varepsilon_\ell
=\boldsymbol x_\ell-\overline{\boldsymbol\mu}_{m,R_\ell}.
$$

Residuals from all regimes are pooled and passed to the Ledoit-Wolf covariance
estimator with `assume_centered=true`:

$$
\mathbf C_m
=\operatorname{LW}\!\left(
\{\boldsymbol\varepsilon_\ell:\ell\in\mathcal T_m\}
\right).
$$

Here $\operatorname{LW}$ denotes the Ledoit-Wolf shrinkage estimator. Thus
$\mathbf C_m$ is one shared monthly within-regime covariance matrix. Separate
regime covariances are not estimated. Centering on unshrunk sample means is
intentional: pulling a sparse regime mean toward the global mean must not
mechanically appear as additional within-regime volatility.

Ledoit-Wolf shrinkage stabilizes the covariance matrix, but it does not remove
sampling error, structural change, or common factor concentration.

## 7. Posterior mixture moments

### 7.1 Posterior-conditioned estimator

The strategy integrates over every regime rather than selecting a hard state.
The expected one-month return vector is

$$
\boldsymbol\mu_m
=\sum_{r\in\mathcal R}p_{m,r}\widetilde{\boldsymbol\mu}_{m,r}.
$$

The between-regime covariance of the conditional means is

$$
\mathbf B_m
=\sum_{r\in\mathcal R}p_{m,r}
  (\widetilde{\boldsymbol\mu}_{m,r}-\boldsymbol\mu_m)
  (\widetilde{\boldsymbol\mu}_{m,r}-\boldsymbol\mu_m)^\top.
$$

Applying the law of total covariance gives the monthly predictive covariance:

$$
\boldsymbol\Sigma_m^{\mathrm{month}}=\mathbf C_m+\mathbf B_m.
$$

The first term represents shared within-regime return variation. The second
term represents risk caused by uncertainty over regime-dependent mean returns.
This construction prevents a diffuse posterior from being treated as though
its probability-weighted mean were certain.

For the volatility constraint only, the matrix is annualized as

$$
\boldsymbol\Sigma_m^{\mathrm{annual}}
=12\boldsymbol\Sigma_m^{\mathrm{month}}.
$$

Expected returns remain in one-month units in the optimization objective.

### 7.2 Pooled-mean optimizer ablation

`pooled_mean_optimizer` is a causal no-current-posterior ablation. It uses the
same expanding sample, Ledoit-Wolf estimator, optimizer, asset and group caps,
volatility ceiling, execution rule, and transaction-cost treatment as the
posterior strategy. Only the current posterior-conditioned return moments are
removed.

Let the causal historical regime frequency be

$$
f_{m,r}=\frac{n_{m,r}}{N_m}.
$$

The ablation's expected-return vector is the pooled sample mean
$\overline{\boldsymbol\mu}_m$, irrespective of the current posterior. Its
between-regime risk term uses the historical frequencies and the *unshrunk*
regime sample means:

$$
\mathbf B_m^{\mathrm{pool}}
=\sum_{r\in\mathcal R} f_{m,r}
  (\overline{\boldsymbol\mu}_{m,r}-\overline{\boldsymbol\mu}_m)
  (\overline{\boldsymbol\mu}_{m,r}-\overline{\boldsymbol\mu}_m)^\top.
$$

An empty regime receives $\overline{\boldsymbol\mu}_m$ in this calculation
and therefore adds
zero between-regime dispersion. The predictive covariance is

$$
\boldsymbol\Sigma_m^{\mathrm{pool,month}}
=\mathbf C_m+\mathbf B_m^{\mathrm{pool}}.
$$

This comparator is more informative about the incremental contribution of the
current posterior than a static benchmark because portfolio constraints,
risk estimation, cost regularization, and causal training data are held fixed.
It is still an ablation within this selected model and sample, not a causal
proof that any performance difference is attributable only to regime
forecasting.

## 8. Long-only optimization

### 8.1 Objective

Let $\boldsymbol w_m$ be the target-weight vector and
$\widehat{\boldsymbol w}_m^-$ the causal estimate of
pretrade weights available when the target is formed. This estimate drifts the
previous target from its entry adjusted open through the last adjusted close
strictly before the first-calendar-day signal. It does not use the execution
open, which is not yet known. Let $\boldsymbol u_m$ contain auxiliary
absolute-trade variables $u_{m,a}$, and set $c_a=0.0005$ for every
strategy ETF. Let $\boldsymbol 1_7$ be the seven-element vector of ones and let
$\overline w_a$ be asset $a$'s individual maximum weight from Section 8.2.

The baseline problem is

$$
\max_{\boldsymbol w_m,\boldsymbol u_m}
\quad \boldsymbol\mu_m^\top\boldsymbol w_m-\sum_a c_a u_{m,a}
$$

subject to

$$
\begin{aligned}
\boldsymbol 1_7^\top\boldsymbol w_m&=1,\\
0\leq w_{m,a}&\leq \overline w_a,\\
u_{m,a}&\geq w_{m,a}-\widehat w_{m,a}^-,\\
u_{m,a}&\geq -(w_{m,a}-\widehat w_{m,a}^-),\\
\boldsymbol w_m^\top\boldsymbol\Sigma_m^{\mathrm{annual}}
\boldsymbol w_m&\leq 0.10^2,
\end{aligned}
$$

together with the group caps below. At the optimum, the cost penalty makes
$u_{m,a}=|w_{m,a}-\widehat w_{m,a}^-|$. The problem maximizes expected
one-month return net of an estimated, causally available trading cost. This
penalty can differ from the cost realized at the later execution open. The
objective does not include a quadratic risk penalty; risk enters through the
hard volatility and position constraints.

### 8.2 Individual caps

| Asset | Maximum weight |
|---|---:|
| `SPY` | 35% |
| `IEF` | 50% |
| `TIP` | 40% |
| `HYG` | 25% |
| `BIL` | 100% |
| `GLD` | 25% |
| `LQD` | 40% |

### 8.3 Group caps

| Group | Assets | Maximum combined weight |
|---|---|---:|
| Equity and high yield | `SPY`, `HYG` | 50% |
| Corporate credit | `HYG`, `LQD` | 50% |
| Rate sensitive | `IEF`, `TIP`, `LQD` | 75% |

The 10% annualized volatility ceiling is an ex-ante model constraint, not a
guarantee about future realized volatility.

### 8.4 Solver and fallback contract

The constrained problem is solved with SLSQP, with at most 2,000 iterations.
Every returned portfolio is independently checked using a $10^{-7}$
feasibility tolerance. No fallback is allowed to relax the long-only rule, full
investment, an individual cap, a group cap, or the volatility ceiling.

If the primary optimization fails or returns an infeasible vector, the frozen
fallback order is:

1. hold the drifted pretrade portfolio if it is feasible;
2. solve the same constraints for a minimum-variance portfolio;
3. allocate 100% to `BIL` if that target is feasible.

If none is feasible, the run must fail visibly rather than publish a silently
relaxed portfolio.

## 9. Trading costs and portfolio accounting

### 9.1 Pretrade estimate used to form the target

Let $d_m^{\mathrm{close}}$ be the last session with an adjusted close strictly
before the first-calendar-day signal for month $m$. The optimizer estimates
the pretrade weights as

$$
\widehat w_{m,a}^-
=\frac{w_{m-1,a}
       P^{\mathrm{adj,close}}_{a,d_m^{\mathrm{close}}}/
       P^{\mathrm{adj,open}}_{a,d_{m-1}^{\mathrm{exe}}}}
       {\sum_{a'} w_{m-1,a'}
       P^{\mathrm{adj,close}}_{a',d_m^{\mathrm{close}}}/
       P^{\mathrm{adj,open}}_{a',d_{m-1}^{\mathrm{exe}}}}.
$$

Only prices strictly before the signal enter this estimate. This preserves a
causal target when the first calendar day is also the execution session. The
optimizer's reported expected turnover and cost use
$\widehat{\boldsymbol w}_m^-$.

### 9.2 Realized pretrade weights at execution

The execution open can differ from the last known close because of an
overnight or holiday-gap return. After the complete open-to-open return
$x_{m-1,a}$ is realized, the backtest obtains the actual execution-open
pretrade weights as

$$
w_{m,a}^{-,\mathrm{exec}}
=\frac{w_{m-1,a}(1+x_{m-1,a})}
       {\sum_{a'} w_{m-1,a'}(1+x_{m-1,a'})}.
$$

Realized turnover and realized cost are measured against these execution-open
weights, not against the prior target and not against the optimizer's
last-close estimate. This separates causal target formation from accurate
backtest accounting.

### 9.3 Cost convention

The cost rate at a rebalance is

$$
K_m=\sum_a c_a|w_{m,a}-w_{m,a}^{-,\mathrm{exec}}|,
\qquad c_a=0.0005.
$$

Each $c_a$ is a one-way cost per dollar bought or sold. A complete rotation
from one fully invested asset to another has full $L^1$ traded notional of
two and therefore costs ten basis points under uniform five-basis-point rates.
Reported one-way turnover is

$$
T_m=\frac{1}{2}\sum_a
|w_{m,a}-w_{m,a}^{-,\mathrm{exec}}|.
$$

The initial portfolio starts from zero holdings. Its full investment therefore
incurs an initial five-basis-point formation cost when all asset cost rates are
five basis points. The initial full $L^1$ traded notional is one, meaning 100%
of NAV is purchased, but the reported half-$L^1$ one-way-turnover convention
records that formation trade as 0.5, or 50%.

The optimizer uses the same formula with $\widehat w_m^-$ to penalize
estimated turnover. The target's published `estimated_rebalance_cost` is this
causal estimate, not a claim that the execution-open cost is already known.

If $g_m=\boldsymbol w_m^\top\boldsymbol x_m$ is the gross holding-period
return, the net return is

$$
r_m^{\mathrm{net}}=(1-K_m)(1+g_m)-1.
$$

This multiplicative convention charges costs at the entry open before the
holding-period return is earned. The same realized cost convention applies to
the optimized strategy and all comparators. Daily marked NAV records both the
pre-trade and post-trade entry-open values. The initial formation cost is
therefore included in net monthly return, cost drag, terminal wealth, and
maximum drawdown from the initial pre-cost peak.

## 10. Comparison portfolios

### 10.1 Pooled-mean optimizer ablation

`pooled_mean_optimizer` is the constrained causal ablation defined in Section
7.2. It deliberately removes the current posterior from expected returns and
risk mixing while retaining the posterior strategy's optimizer policy and all
other portfolio mechanics.

### 10.2 Equal weight

The equal-weight method allocates $1/7$ to each strategy asset and rebalances
at every first common monthly open. It pays the same per-asset realized costs
as the optimized strategy.

### 10.3 Legacy Sharpe-MAP rule

`legacy_sharpe_map` preserves the original paper's Sharpe score and positivity
shift while removing its hard-label look-ahead problem.

At signal $m$, it selects the MAP regime

$$
r_m^*=\arg\max_r p_{m,r}.
$$

It then uses daily ETF returns from historical calendar months labeled
$r_m^*$, but only when the return date precedes the signal and that month's
deterministic label was known strictly before the signal. All assets share one
complete daily sample. If fewer than 60 eligible daily observations exist, the
target is equal weight.

Otherwise, for each asset,

$$
\mathrm{SR}_a=\sqrt{252}\frac{\overline r_a}{\widehat\sigma_a}.
$$

Zero-variance or nonfinite scores are neutralized to zero. The legacy shift is

$$
\gamma=\left|\min_a \mathrm{SR}_a\right|+0.1,
$$

and the long-only weights are

$$
w_a=\frac{\mathrm{SR}_a+\gamma}
{\sum_{a'}(\mathrm{SR}_{a'}+\gamma)}.
$$

Here $\overline r_a$ and $\widehat\sigma_a$ are the sample mean and sample
standard deviation of eligible daily returns for asset $a$,
$\mathrm{SR}_a$ is its annualized Sharpe score, and $\gamma$ is the legacy
positivity shift added to every score.

This comparator is an adaptation, not a literal reproduction: it uses the new
seven-ETF universe, a causal posterior MAP state, first-open monthly execution,
and the shared transaction-cost engine. It does not use covariance, posterior
averaging, or the optimized strategy's asset, group, and volatility caps.

### 10.4 Static 60/40

`static_60_spy_40_agg` allocates 60% to `SPY` and 40% to `AGG`. It rebalances at
every first common monthly open and pays the same realized cost schedule. `AGG`
is benchmark-only, so the simulation calendar must be common to all strategy
and benchmark assets.

## 11. Sensitivity design

Sensitivity checks are one-parameter-at-a-time variants. Every variant uses
the same backtest dates as the baseline, and every parameter not named by the
variant remains fixed at its baseline value. The baseline was declared before
results and must not be replaced after viewing the sensitivity table.

| Parameter | Baseline | Frozen values |
|---|---:|---|
| Regime-mean pseudo-months $\kappa$ | 24 | 0, 12, 24, 48, 96 |
| Annualized volatility cap | 10% | 8%, 10%, 12% |
| Position/group-cap multiplier | 1.0 | 0.8, 1.0, 1.2 |
| Per-asset one-way transaction cost | 5 bp | 0 bp, 5 bp, 10 bp |

The $\kappa=0$ variant is the no-shrinkage limit for nonempty regimes. Under
the frozen empty-regime policy, an unobserved regime still receives the pooled
mean rather than an undefined vector.

The cap multiplier scales every non-`BIL` individual cap and every group cap,
with each result bounded above by 100%. The `BIL` individual cap remains 100%.
This diagnostic tests whether the result depends materially on the declared
concentration policy.

Transaction-cost variants are joint optimizer-policy and realized-cost
sensitivities. The same rate both regularizes estimated trades in the objective
and is charged to realized traded notional. They therefore do not isolate the
mechanical execution-cost drag of a fixed target sequence.

Sensitivity variants are diagnostics of dependence on declared assumptions.
They are not an in-sample parameter-selection tournament.

## 12. Evaluation metrics

All return and trading metrics use net monthly returns unless explicitly
identified as gross. With $M$ complete months and monthly net returns $r_m$:

### 12.1 Return and risk

- **Total return:**
  $\prod_{m=1}^{M}(1+r_m)-1$.
- **CAGR:**
  $\left[\prod_m(1+r_m)\right]^{12/M}-1$.
- **Gross CAGR:** the same geometric calculation using returns before costs.
- **Annualized cost drag:** gross CAGR minus net CAGR.
- **Annualized volatility:** sample standard deviation of monthly net returns
  multiplied by $\sqrt{12}$.
- **Zero-rate Sharpe:** annualized arithmetic mean net return divided by
  annualized volatility.
- **BIL-excess Sharpe:** annualized arithmetic mean of monthly portfolio return
  minus the aligned `BIL` return, divided by the annualized sample standard
  deviation of that active-return series. Because the denominator is active
  volatility, this is also interpretable as an information ratio versus `BIL`.
- **Zero-rate Sortino:** annualized arithmetic mean net return divided by
  $\sqrt{12}$ times the root mean square of $\min(r_m,0)$.

### 12.2 Drawdown and distribution diagnostics

- **Maximum drawdown:** the minimum ratio of daily marked NAV to its prior
  running peak, minus one. Daily NAV contains post-trade open points, daily
  adjusted-close marks, and the terminal next-month open.
- **Calmar ratio:** CAGR divided by the absolute maximum drawdown when a
  negative drawdown exists.
- **Worst month:** minimum monthly net return.
- **Positive-month fraction:** fraction of complete months with net return
  strictly greater than zero.

The audit tables may additionally retain best month and other descriptive
fields even when they are not headline metrics.

### 12.3 Trading diagnostics

- **Annualized one-way turnover:** 12 times the mean monthly half-$L^1$
  turnover $T_m$.
- **Annualized cost drag:** the geometric difference between gross and net
  CAGR, not merely twelve times average cost.
- The audit output also records gross traded notional, each monthly cost rate,
  average and maximum one-way turnover, and the sum of monthly cost rates.

Metrics must be compared on identical complete holding months. A current
target without a realized next-open exit cannot enter a metric.

### 12.4 Paired sampling uncertainty

For every comparator $b$, the monthly active-return difference is

$$
\delta_m^{(b)}=r_m^{\mathrm{posterior}}-r_m^{(b)}.
$$

Define the annualized arithmetic mean effect as

$$
\Delta^{(b)}=12\overline\delta^{(b)}.
$$

Sampling uncertainty is estimated with a paired circular block bootstrap:
six-month consecutive blocks are sampled with wraparound and truncated to the
original 102-month length. The same 10,000 bootstrap index paths are applied
to the posterior and every comparator, using seed `20260718`. The public table
reports the 2.5th and 97.5th percentiles and the fraction of bootstrap mean
differences strictly above zero, written $\Pr(\Delta^{(b)}>0)$.

That fraction is not a p-value. The confidence interval is the headline
uncertainty summary, and an interval containing zero does not support a claim
of superior expected performance at the stated 95% level. The bootstrap also
does not correct for model selection, multiple comparisons, or structural
change.

## 13. Output and audit contract

Local research tables are written below
`data/processed/m01_regime_allocation_backtest/`. They include the selected
signals, open-to-open returns, regime-estimation audit, optimizer audit, target
weights, strategy returns, daily NAV, performance metrics, and sensitivities.

The tracked manifest is
`data/manifests/m01_regime_allocation_backtest.json`. Small public artifacts
are written below `results/published/m01_regime_allocation_backtest/`, including
the latest target, backtest summary, performance comparison, monthly returns,
monthly weights, sensitivity table, and paired comparison-uncertainty table.

The latest target must identify the signal and execution dates, posterior,
weights, expected moments, active constraints, and whether the holding period
is complete. It must also state that it is research output rather than
investment advice.

## 14. Limitations

1. **Expected returns are difficult to estimate.** Twenty-four pseudo-months
   stabilize sparse regime means, but neither shrinkage nor expanding windows
   guarantee that historical conditional means persist.
2. **The risk model is deliberately restrictive.** One shared within-regime
   covariance assumes that covariance dynamics do not differ by regime. The
   between-regime term captures dispersion in conditional means, not every
   form of regime-dependent volatility or tail dependence.
3. **Posterior uncertainty is conditional on Model 01.** The mixture treats the
   saved regime probabilities and fitted means as inputs. It does not integrate
   over transition-parameter, likelihood-parameter, or return-model estimation
   uncertainty.
4. **Monthly trading leaves information unused.** Daily release updates matter
   through the posterior carried into the next month, but the strategy does
   not rebalance when evidence arrives during the holding month.
5. **The optimizer can prefer corners.** The objective is linear in expected
   return and trading cost, with risk represented by hard constraints. Small
   changes in estimated means can move a solution between binding caps.
6. **The volatility ceiling is model based.** Ten percent is an ex-ante limit
   under the estimated covariance, not a realized-volatility target or
   guarantee.
7. **Cost modeling is simplified.** Constant five-basis-point one-way costs do
   not vary with spreads, opening liquidity, volatility, order size, taxes, or
   market impact. Executing every asset at a provider-adjusted opening price is
   an idealization.
8. **ETF proxies are imperfect.** Fund expenses, tracking difference, roll or
   index methodology, liquidity, and product structure can affect returns
   independently of the intended macro exposure.
9. **Adjusted data are not vintage data.** Historical adjusted prices can be
   revised by the provider. The macro pipeline is point-in-time, but the ETF
   return panel is a downloaded research snapshot rather than an archived
   point-in-time market database.
10. **The universe is selected and narrow.** The backtest begins after ETF data
    become jointly available and does not establish performance across earlier
    inflation or policy regimes. Omitting FX and broad commodities may weaken
    inflation-regime expression.
11. **The 60/40 comparator uses an extra instrument.** `AGG` is benchmark-only,
    so the benchmark and strategy opportunity sets are not identical.
12. **Long-only, fully invested constraints limit expression.** The strategy
    cannot short an unattractive asset class, use leverage, or move outside the
    listed ETFs. `BIL` is the defensive residual but remains an ETF rather than
    literal cash.
13. **Backtest evidence is finite.** A 2018-forward evaluation contains only a
    small number of distinct macro episodes. Headline ratios require subperiod,
    sensitivity, and sampling-uncertainty context.
14. **The pooled ablation is close but not definitive.** It holds portfolio
    mechanics fixed while removing the current posterior, but it remains a
    comparison within one selected universe, constraint set, and historical
    sample. A small or uncertain difference cannot establish either that the
    posterior has no value or that it will add value out of sample.

These limitations are part of the specification. They should accompany public
results rather than being added only when a variant underperforms.
