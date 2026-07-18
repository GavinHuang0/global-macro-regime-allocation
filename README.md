# Global Macro Regime Detection and Allocation

Model 01 is a frozen, point-in-time research pipeline for forecasting monthly
US growth/inflation regimes and using the resulting probability distribution in
a constrained cross-asset ETF allocation. It is an independent rebuild of a
team course project; the original paper and surviving script are preserved in
[`legacy/`](legacy/README.md) but are not imported by the new package.

The model is complete through 18 July 2026. The full repository was reviewed
for data errors, arithmetic errors, and look-forward leakage before this
snapshot was frozen. That review found and corrected two data/evaluation issues:
15 active target-component observations had been mistaken for first releases,
and same-day pre-confirmation snapshots had been eligible for forecast scoring.
Every downstream artifact and result shown below was rebuilt after both fixes.
The detailed closure record is in
[`model_audit.md`](docs/models/m01_deterministic_composite/model_audit.md).

This is research software, not investment advice. The main empirical conclusion
is deliberately modest: leading releases improve the 97-month month-end
probability scores slightly relative to a transition-only model, but the
portfolio's advantage over an otherwise identical pooled-mean optimizer is
small and statistically inconclusive.

## Model versions

| ID | Architecture | Status |
|---|---|---|
| `m01_deterministic_composite` | Deterministic monthly target, event-driven Bayesian filter, posterior allocation | **Frozen** |
| `m02_continuous_state` | Latent continuous growth/inflation state with quadrant probabilities | Planned |
| `m03_switching_state_space` | Regime-switching continuous state-space model | Planned |

## End-to-end architecture

```text
point-in-time first releases
          |
          +--> deterministic monthly growth/inflation target
          |          |
          |          +--> expanding transition matrix --> path prior
          |
          +--> non-defining release events --> Student-t likelihoods
                                             |
                        four-month joint Bayesian filter
                                             |
                             current-month regime posterior
                                             |
                   posterior return moments + constrained optimizer
                                             |
                         next-session open execution and backtest
```

All clocks are publication clocks. An observation indexed to January is not
available in January merely because it describes January. Every fit, forecast,
and trade has an explicit information cutoff.

### Notation used throughout Model 01

The documentation uses one calendar convention throughout:

- $m$ is the monthly reference or target month, and $\ell$ denotes another
  historical reference month;
- $d$ is a calendar knowledge date, while $e$ is a release event published on
  date $d_e$ and describing reference month $m(e)$;
- $R_m\in\mathcal R$ is the regime for month $m$, where $\mathcal R$ is the
  four-regime state space, and $r,i,j\in\mathcal R$ denote candidate regimes;
- $S_m=(R_{m-3},R_{m-2},R_{m-1},R_m)$ is the four-month state path, and $s$
  denotes one of its $4^4=256$ possible regime sequences;
- $\mathcal D_d$ is all information available by date $d$, and
  $q_{m,d}(s)=\Pr(S_m=s\mid\mathcal D_d)$ is the filtered probability of path
  $s$;
- bold lowercase symbols are vectors, bold uppercase symbols are matrices, and
  plain symbols are scalars unless stated otherwise.

Dates and reference months are deliberately different objects: $d_e$ says
when an event became knowable, whereas $m(e)$ says which economic month it
describes.

## 1. Deterministic monthly target

### 1.1 First-release rule

For component series $k$, reference month $m$, and real-time vintage $v$, let
$x_{k,m}^{(v)}$ denote the published level visible in that vintage. First
identify the observation's earliest appearance in the acquired real-time
archive:

$$
\widetilde v_k(m)=
\min\left\{v:x_{k,m}^{(v)}\text{ is available}\right\}.
$$

It is retained as $v_k(m)=\widetilde v_k(m)$ only when its release lag is
between 0 and 92 days and it passes the archive-start rule below. Otherwise the
feature is unavailable; the extractor never searches for a later vintage that
happens to pass the lag test. This preserves the meaning of *first release* and
rejects observations whose first appearance is a backfill.

Both the current and previous levels are read from the same snapshot
$v_k(m)$. This prevents a current first release from being compared with a
later-revised prior-month value. At the beginning of a real-time archive, one
snapshot can expose several old observations at once. Only the latest reference
period in that initial snapshot is retained as a genuine real-time release;
older bootstrap rows are excluded.

### 1.2 Components and transformations

In the table below, $x_m$ abbreviates the component's level for reference month
$m$ as seen in its selected first-release snapshot $v_k(m)$; $x_{m-1}$ is
the prior-month level read from that same snapshot.

| Axis | Component | FRED series | First-release transformation |
|---|---|---|---|
| Growth | Nonfarm payrolls | `PAYEMS` | $x_m-x_{m-1}$ |
| Growth | Industrial production | `INDPRO` | $100\log(x_m/x_{m-1})$ |
| Growth | Real personal consumption | `PCEC96` | $100\log(x_m/x_{m-1})$ |
| Growth | Unemployment rate | `UNRATE` | $-(x_m-x_{m-1})$ |
| Inflation | Core CPI | `CPILFESL` | $100\log(x_m/x_{m-1})$ |
| Inflation | Core PCE price index | `PCEPILFE` | $100\log(x_m/x_{m-1})$ |
| Inflation | Core finished-goods PPI | `PPILFE`, then `WPSFD4131` | $100\log(x_m/x_{m-1})$ |
| Inflation | Average hourly earnings | `AHETPI` | $100\log(x_m/x_{m-1})$ |

The PPI substitution follows the official series break and is spliced in
changes, not levels. All components are seasonally adjusted monthly series.

### 1.3 Lagged standardization, composites, and regimes

Let $u_{k,m}$ be a transformed component and define its available historical
index set as
$\mathcal H_{k,m-1}=\{\ell<m:u_{k,\ell}\text{ is available}\}$.
Its expanding standardization uses only those earlier reference months. Here
$n_{k,m-1}=|\mathcal H_{k,m-1}|$, $\widehat\mu_{k,m-1}$ is their mean, and
$\widehat\sigma_{k,m-1}$ is their sample standard deviation:

$$
\widehat\mu_{k,m-1}=\frac{1}{n_{k,m-1}}
\sum_{\ell\in\mathcal H_{k,m-1}}u_{k,\ell},\qquad
\widehat\sigma_{k,m-1}^2=\frac{1}{n_{k,m-1}-1}
\sum_{\ell\in\mathcal H_{k,m-1}}
(u_{k,\ell}-\widehat\mu_{k,m-1})^2,
$$

$$
z_{k,m}=\frac{u_{k,m}-\widehat\mu_{k,m-1}}
{\widehat\sigma_{k,m-1}},
\qquad n_{k,m-1}\ge60.
$$

The current observation never contributes to its own mean or variance. Let
$\mathcal G$ and $\mathcal I$ be the four-component growth and inflation sets,
respectively. Their equal-weight monthly composites are

$$
C_m^G=\frac14\sum_{k\in\mathcal G}z_{k,m},\qquad
C_m^I=\frac14\sum_{k\in\mathcal I}z_{k,m}.
$$

The superscripts identify the growth and inflation axes; they are not powers.
The model smooths each composite over the current and preceding two months:

$$
G_m=\frac{C_m^G+C_{m-1}^G+C_{m-2}^G}{3},\qquad
I_m=\frac{C_m^I+C_{m-1}^I+C_{m-2}^I}{3}.
$$

The signs define the target:

| $G_m$ | $I_m$ | `regime_id` |
|---:|---:|---|
| $\ge0$ | $\ge0$ | `growth_up_inflation_up` |
| $<0$ | $\ge0$ | `growth_down_inflation_up` |
| $\ge0$ | $<0$ | `growth_up_inflation_down` |
| $<0$ | $<0$ | `growth_down_inflation_down` |

This is a deterministic eventual target, not an assertion that the economic
state is directly observable. The Bayesian stage forecasts this target before
all of its defining releases arrive.

The label availability date is the latest publication date among every
component observation required by the expanding transforms and three-month
window. The implementation enforces this cumulatively, so a delayed historical
release cannot be used early. A month is unclassified if any required component
or trailing score is missing; weights are never redistributed.

The acquired panel contains 312 reference months from June 2000 through May
2026. After the 60-month warm-up, the public history contains 250 months from
August 2005 through May 2026: 246 classified and four unavailable. The
October 2025 federal shutdown left October core CPI and unemployment missing;
the frozen change and smoothing rules propagate that gap through January 2026.

## 2. Transition dynamics and four-month path

Let $T_m$ be the date on which the deterministic label $R_m$ became available.
Model 01 uses a time-homogeneous, first-order Markov transition law. At
information cutoff $d$, only consecutive pairs whose two labels were available
by the end of $d$ are counted. For origin state $i$ and destination state $j$,

$$
N_{ij}^{(d)}=\sum_\ell
\mathbf 1[R_{\ell}=i,R_{\ell+1}=j,T_{\ell}\le d,T_{\ell+1}\le d].
$$

The indicator $\mathbf 1[\cdot]$ equals one when every condition in brackets is
true and zero otherwise, so $N_{ij}^{(d)}$ is a transition count.

Each row receives an independent symmetric Dirichlet prior with
$\alpha=0.5$ in each of the four destination cells. This adds one-half of a
transition as a pseudocount to every possible destination, so an unobserved
transition does not receive exactly zero probability. Defining
$N_i^{(d)}=\sum_{j\in\mathcal R}N_{ij}^{(d)}$, the Dirichlet-smoothed
probability used to predict one next transition is

$$
\overline A_{ij}^{(d)}
=\Pr(R_m=j\mid R_{m-1}=i,\mathcal D_d)
=\frac{N_{ij}^{(d)}+0.5}{N_i^{(d)}+4(0.5)}.
$$

At a month roll, the filter drops the oldest coordinate of $S_m$ and appends a
new regime. Let $d^-$ mean the information available immediately before the
roll on date $d$; operationally, the fitted transition counts end on the prior
calendar day. The prior for the shifted path is

$$
q_{m+1,d}^{-}(r_{m-2:m+1})=
\sum_{r_{m-3}\in\mathcal R}
q_{m,d^-}(r_{m-3:m})
\overline A_{r_m,r_{m+1}}^{(d^-)}.
$$

The superscript $-$ on $q_{m+1,d}^{-}$ means the new date's prior before any
release update. The summation marginalizes the discarded month $R_{m-3}$; the
transition factor appends the candidate state $R_{m+1}=r_{m+1}$.

The four-month window provides fixed-lag smoothing for delayed releases. It
does **not** make the transition law fourth-order; only $R_m$ controls the
transition to $R_{m+1}$.

## 3. Non-defining event evidence

Five release blocks provide evidence before the defining target is confirmed.
The evidence builder engineers 12 features, while the frozen likelihood uses
11 because continued claims (`CCSA`) is retained in the data contract but
excluded from the baseline.

| Block | Baseline features | Regime reference |
|---|---|---|
| Weekly claims | `ICSA` innovation | Month containing the reference week |
| JOLTS | Changes in openings, hires, quits, layoffs/discharges rates | Reported month |
| Retail sales | Log changes in total and ex-motor-vehicle sales | Reported month |
| Housing | Log changes in starts and permits | Reported month |
| Durable goods | Log changes in total and core capital-goods orders | Reported month |

Monthly evidence features use same-vintage differences and are standardized on
release history strictly before the event date. At least 60 earlier
observations are required. For absolute weekly claims, let $x_\tau$ be the log
of the `ICSA` level for weekly observation $\tau$. Immediately before publication
date $d$, an expanding AR(1) is fitted only to observations released earlier:

$$
x_\tau=a_d+\phi_d x_{\tau-1}+\varepsilon_\tau,
\qquad d_\tau<d.
$$

Here $a_d$ and $\phi_d$ are the intercept and persistence coefficient fitted
with the information available before $d$, $d_\tau$ is observation $\tau$'s
release date, and $\varepsilon_\tau$ is its unexplained innovation. The fitted
innovation
$\widehat\varepsilon_\tau=x_\tau-\widehat a_d-\widehat\phi_d x_{\tau-1}$ is
standardized using only earlier innovations. Same-day catch-up observations
share one frozen fit and cannot train one another.

The event artifact contains 6,955 normalized first-release observations, 3,968
events, and 3,275 available feature rows through 16 July 2026. Archive
backfills and delayed batches remain visible in audit tables rather than being
silently converted into ordinary observations.

## 4. Release likelihood and Bayesian update

For release block $b$ and scored date $d$, let $\mathbf y_e\in\mathbb R^{p_b}$
be event $e$'s complete feature vector, where $p_b$ is the number of features
in block $b$, and let $b(e)$ identify the block containing event $e$. The causal
training set is

$$
\mathcal T_b(d)=
\left\{
(\mathbf y_e,R_{m(e)}):
b(e)=b,\ d_e<d,\ T_{m(e)}<d,
\ \mathbf y_e\text{ is complete}
\right\}.
$$

Both inequalities are strict. Today's event cannot train today's likelihood,
and a target confirmed today cannot enter today's fit. At least 24 complete
labeled vectors are required.

For regime $r$, let $n_{b,r}$ be its number of observations in
$\mathcal T_b(d)$, let $\bar{\mathbf y}_{b,r}$ be their mean, and let
$\bar{\mathbf y}_b$ be the mean across the whole block. The regime mean is
shrunk toward the block-wide mean with $\kappa=5$ pseudo-observations:

$$
\widetilde{\boldsymbol\mu}_{b,r}=
\frac{n_{b,r}\bar{\mathbf y}_{b,r}+5\bar{\mathbf y}_b}{n_{b,r}+5}.
$$

This formula gives the block-wide mean the same influence as five additional
observations. For each training event, define the residual
$\mathbf u_e=\mathbf y_e-\widetilde{\boldsymbol\mu}_{b,R_{m(e)}}$. Residuals
from all regimes are pooled. If $\widehat{\mathbf V}_b$ is their empirical
covariance, one Ledoit-Wolf covariance is fitted per block:

$$
\widehat{\mathbf C}_b=(1-\widehat\lambda_b)\widehat{\mathbf V}_b+
\widehat\lambda_b
\frac{\operatorname{tr}(\widehat{\mathbf V}_b)}{p_b}\mathbf I_{p_b}.
$$

The estimated shrinkage intensity $\widehat\lambda_b\in[0,1]$ moves the noisy
empirical covariance toward a spherical matrix; $\mathbf I_{p_b}$ is the
$p_b\times p_b$ identity matrix. Regimes therefore differ in location but
share a stabilized covariance. The baseline likelihood is multivariate
Student-$t$ with $\nu=7$ degrees of freedom. Because the distribution's shape
matrix is not its covariance, Model 01 uses

$$
\boldsymbol\Psi_b=
\frac{\nu-2}{\nu}\widehat{\mathbf C}_b
=\frac57\widehat{\mathbf C}_b,
\qquad
\mathbf y_e\mid R_{m(e)}=r
\sim t_7\!\left(\widetilde{\boldsymbol\mu}_{b,r},
\boldsymbol\Psi_b\right).
$$

Here $t_7(\boldsymbol\mu,\boldsymbol\Psi)$ denotes a multivariate Student-$t$
distribution with location $\boldsymbol\mu$, shape $\boldsymbol\Psi$, and seven
degrees of freedom. For candidate path $s$, $L_e(s)$ is this density evaluated
under the regime occupying month $m(e)$ in that path. Let $\mathcal E_d$ be all
eligible leading events published on date $d$. Their likelihoods are combined
in log space for numerical stability, and the path distribution is normalized
once:

$$
q_{m,d}^{+}(s)=
\frac{
q_{m,d}^{-}(s)\displaystyle\prod_{e\in\mathcal E_d}L_e(s)
}{
\displaystyle\sum_{s'\in\mathcal R^4}
q_{m,d}^{-}(s')\prod_{e\in\mathcal E_d}L_e(s')
}.
$$

The denominator sums the unnormalized weights of all 256 paths, so the updated
probabilities are nonnegative and sum to one.

When all defining first releases for a past target become available, the path
is hard-clamped to the deterministic label at end of day: inconsistent paths
receive probability zero, and the remaining paths are renormalized. This is
historical confirmation, not forecast evidence. Daily order is: month roll,
one combined leading-release update, then one combined end-of-day confirmation.

The baseline is compared with 15 prespecified one-at-a-time sensitivities over
degrees of freedom, mean shrinkage, covariance shrinkage, and scale. These
checks were not used to select the headline specification on the evaluation
sample.

## 5. Forecast checkpoints and metrics

Forecasts are scored only when their checkpoint date is **strictly earlier**
than the target's availability date. Month start, month end, and each numbered
ICSA release are eligible. Same-day pre-confirmation snapshots are retained for
state reconstruction but are diagnostic-only and never scored.

Suppose the evaluation contains $H$ forecast instances. For instance $h$, let
$o_h\in\mathcal R$ be the observed regime and let $p_{h,r}$ be the forecast
probability assigned to regime $r$. The negative log likelihood (NLL) and
multiclass Brier score are

$$
\mathrm{NLL}=-\frac1H\sum_{h=1}^{H}\log p_{h,o_h},\qquad
\mathrm{Brier}=\frac1H\sum_{h=1}^{H}\sum_{r\in\mathcal R}
\left(p_{h,r}-\mathbf 1[o_h=r]\right)^2.
$$

The indicator $\mathbf 1[o_h=r]$ equals one when regime $r$ occurred and zero
otherwise. Lower NLL and Brier scores are better; the unscaled four-class Brier
score ranges from 0 to 2. Maximum-a-posteriori (MAP) accuracy is the fraction
for which the largest-probability regime occurred. Balanced accuracy averages
the recall
within each realized regime, and macro F1 averages the four class-specific F1
scores equally. Calibration error compares stated probabilities with observed
frequencies. Axis Brier scores collapse the four quadrants into the binary
growth-up and inflation-up events. Entropy measures concentration of the
forecast distribution, not correctness.

On the 97 classified targets from January 2018 through May 2026:

| Checkpoint | Model | NLL | Brier | MAP accuracy | Balanced accuracy |
|---|---|---:|---:|---:|---:|
| Month start | Event model | 1.0464 | 0.5725 | 62.89% | 33.92% |
| Month start | Transition only | 1.0741 | 0.5807 | 61.86% | 33.55% |
| Month end | Event model | **0.9170** | **0.5156** | 64.95% | 34.60% |
| Month end | Transition only | 0.9762 | 0.5291 | **65.98%** | **34.97%** |

The month-end event model improves the two proper probability scores, but not
the hard-label metrics. Its Brier skill versus transition only is 2.55%. With
only 97 targets and four imbalanced regimes, this is evidence of a modest
incremental probability improvement, not a definitive forecasting result.

## 6. Portfolio allocation

### 6.1 Universe and timing

The strategy assets are `SPY`, `IEF`, `TIP`, `HYG`, `BIL`, `GLD`, and `LQD`;
`AGG` is used only by the static 60/40 comparator. At the first calendar day of
month $m$, the strategy uses the current-month marginal from the
`post_month_roll` checkpoint. Execution occurs at the adjusted open of the
first common US session, and the holding return runs open-to-open to the next
month's first common session.

An estimation month $\ell$ is eligible only if its ETF holding period ended
strictly before execution and its regime label was available strictly before
the signal. At least 60 common labeled months are required. Current or
incomplete ETF returns never enter estimation.

### 6.2 Posterior return moments

Let $\mathbf x_\ell\in\mathbb R^7$ be the ETF return vector for eligible
historical month $\ell$. At the signal for target month $m$, let
$\bar{\boldsymbol\mu}_m$ be the pooled expanding return mean and
$\bar{\boldsymbol\mu}_{m,r}$ the sample mean among months labeled regime $r$.
If $n_{m,r}$ is the number of eligible months in regime $r$, the regime means
receive 24 pooled pseudo-months:

$$
\widetilde{\boldsymbol\mu}_{m,r}=
\frac{n_{m,r}\bar{\boldsymbol\mu}_{m,r}
+24\bar{\boldsymbol\mu}_m}{n_{m,r}+24}.
$$

Ledoit-Wolf shrinkage is fitted to residuals around the unshrunk regime sample
means, giving one shared monthly within-regime covariance
$\mathbf C_m\in\mathbb R^{7\times7}$. Let
$p_{m,r}=\Pr(R_m=r\mid\mathcal D_{d_m})$ be the month-$m$ regime probability at
its portfolio signal date $d_m$. The posterior-weighted mean is

$$
\boldsymbol\mu_m=
\sum_{r\in\mathcal R}p_{m,r}\widetilde{\boldsymbol\mu}_{m,r},
$$

and the covariance used for risk control is

$$
\mathbf B_m=
\sum_{r\in\mathcal R}p_{m,r}
\left(\widetilde{\boldsymbol\mu}_{m,r}-\boldsymbol\mu_m\right)
\left(\widetilde{\boldsymbol\mu}_{m,r}-\boldsymbol\mu_m\right)^\top,
\qquad
\boldsymbol\Sigma_m=\mathbf C_m+\mathbf B_m.
$$

The matrix $\mathbf B_m$ is the between-regime covariance implied by uncertain
regime means. Adding it implements the law of total covariance: a diffuse
posterior is not treated as though its probability-weighted mean were known
with certainty.

### 6.3 Optimization and costs

Let $\mathbf w$ be a candidate target-weight vector, let $a\in\{1,\ldots,7\}$
index the assets,
$\widehat{\mathbf w}_m^{-}$ the causal estimate of weights immediately before
trading, and $c_a=0.0005$ the cost per dollar moved into or out of asset $a$.
The optimizer solves

$$
\max_{\mathbf w}\quad
\boldsymbol\mu_m^\top\mathbf w-
\sum_{a=1}^{7}c_a\left|w_a-\widehat w_{m,a}^{-}\right|
$$

subject to full investment and long-only weights,

$$
\sum_{a=1}^{7}w_a=1,\qquad w_a\ge0,
$$

as well as individual caps
(`SPY` 35%, `IEF` 50%, `TIP` 40%, `HYG` 25%, `BIL` 100%, `GLD` 25%, `LQD`
40%), group caps (`SPY`+`HYG` 50%, `HYG`+`LQD` 50%,
`IEF`+`TIP`+`LQD` 75%), and

$$
\mathbf w^\top(12\boldsymbol\Sigma_m)\mathbf w\le0.10^2.
$$

The factor 12 annualizes monthly covariance, so the final inequality caps the
model-estimated annualized volatility at 10%.

SLSQP solutions are independently checked for feasibility. The fixed fallback
order is feasible pretrade holdings, constrained minimum variance, then all
`BIL`; the run fails if none is feasible.

Realized execution cost uses $\mathbf w_m$, the selected optimizer target, and
$\mathbf w_m^{-,\mathrm{exec}}$, the previous holdings drifted to the actual
execution open:

$$
K_m=\sum_{a=1}^{7}0.0005
\left|w_{m,a}-w_{m,a}^{-,\mathrm{exec}}\right|.
$$

Thus $K_m$ is the fraction of portfolio value deducted as transaction cost at
the rebalance.

Monthly returns, terminal wealth, and the daily NAV all include initial
formation cost. The NAV records both pre-trade and post-trade open checkpoints,
so maximum drawdown also includes that initial deduction.

Comparators are an otherwise identical pooled-mean optimizer without the
current posterior, monthly equal weight, the legacy regime-MAP Sharpe score and
shift rule, and static 60% `SPY` / 40% `AGG`.

## 7. Frozen backtest results

The formal sample contains 102 complete monthly holdings from January 2018
through June 2026. These are net of the declared transaction costs.

| Method | Total return | CAGR | Ann. vol. | Zero-rate Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Posterior optimized | **124.15%** | **9.96%** | 10.17% | 0.988 | -21.16% |
| Pooled-mean optimizer | 120.69% | 9.76% | 10.17% | 0.970 | -21.92% |
| Static 60% `SPY` / 40% `AGG` | 117.41% | 9.57% | 11.21% | 0.874 | -21.60% |
| Equal weight | 65.13% | 6.08% | 6.37% | 0.960 | -14.77% |
| Legacy Sharpe-MAP | 49.61% | 4.85% | 4.83% | **1.007** | **-11.35%** |

Let $\Delta$ denote the posterior strategy's annualized arithmetic mean return
minus a comparator's corresponding mean. Against the pooled-mean optimizer,
$\Delta$ was 0.184%, with a paired six-month circular-block-bootstrap 95%
interval of $[-0.364\%,0.795\%]$ and $\Pr(\Delta>0)=0.740$. Against static
60/40, the estimate was 0.252%, its interval was
$[-2.623\%,3.098\%]$, and $\Pr(\Delta>0)=0.572$. Both intervals cross zero.
The bootstrap fractions are not $p$-values.

This is the most important attribution check: posterior and pooled-optimizer
monthly returns have correlation 0.9973. `SPY` was at its 35% cap and `BIL` at
zero in all 102 completed months; `GLD` was at its cap in 89 months. The
headline absolute return is therefore heavily shaped by the universe,
constraints, and sample—not demonstrably by a unique posterior edge.

## 8. Latest published research snapshot

The latest confirmed target is May 2026, available 25 June 2026:

- `growth_up_inflation_up`;
- growth score 0.016676;
- inflation score 0.443404.

At the 16 July 2026 information cutoff, the event-driven July marginal was:

| Regime | Probability |
|---|---:|
| Growth up / inflation up | 59.66% |
| Growth down / inflation up | 20.09% |
| Growth up / inflation down | 19.98% |
| Growth down / inflation down | 0.26% |

The July allocation was formed earlier, at the 1 July month roll, from
probabilities 57.18%, 18.98%, 19.28%, and 4.56%, respectively. Its research
target is 35% `SPY`, 15% `HYG`, 25% `GLD`, and 25% `LQD`, with zero in `IEF`,
`TIP`, and `BIL`. Do not mix that month-start portfolio signal with the later
16 July posterior.

Machine-readable outputs:

- [confirmed history](results/published/m01_deterministic_composite/regime_history.csv),
  [transition model](results/published/m01_deterministic_composite/transition_model.json),
  and [transition prior](results/published/m01_deterministic_composite/latest_transition_prior.json);
- [latest posterior](results/published/m01_bayesian_filter/latest_posterior.json),
  [forecast evaluation](results/published/m01_bayesian_filter/evaluation_summary.json),
  and [likelihood sensitivities](results/published/m01_bayesian_filter/sensitivity_metrics.csv);
- [latest allocation](results/published/m01_regime_allocation_backtest/latest_allocation.json),
  [performance](results/published/m01_regime_allocation_backtest/performance_summary.csv),
  [monthly returns](results/published/m01_regime_allocation_backtest/monthly_returns.csv),
  [weights](results/published/m01_regime_allocation_backtest/monthly_weights.csv),
  [allocation sensitivities](results/published/m01_regime_allocation_backtest/sensitivity_metrics.csv),
  and [paired uncertainty](results/published/m01_regime_allocation_backtest/comparison_uncertainty.csv).

## 9. Leakage and integrity controls

| Risk | Frozen control | Audit result |
|---|---|---|
| Revised macro history | Same-vintage first-release levels | Passed |
| Archive bootstrap mistaken for releases | Only latest reference in initial snapshot retained | Corrected; 15 active component rows removed |
| Full-sample standardization | Expanding statistics exclude current month/event | Passed |
| Delayed prerequisite used early | Cumulative component availability date | Passed and hardened |
| Future labels in transitions | Pair availability strictly before roll | Passed for every fit |
| Future events in likelihood | Event and label dates strictly before scored release | Passed for 881 fits |
| Same-day release ordering | Atomic group, one normalization | Passed |
| Confirmation counted as forecast | Score date strictly before target availability | Corrected; same-day rows excluded |
| Smoothed future HMM states | No HMM or full-sample decoding in Model 01 | Not applicable |
| Future return in allocation | Holding end and label availability strictly before execution | Passed |
| Execution-price foresight | Target uses last known close; accounting uses later realized open | Passed |
| Output corruption | Manifests hash inputs and generated artifacts | Passed |

The code and reproduced arithmetic passed independent checks of composite
scores, regime labels, transition counts, Student-$t$ densities, path
normalization, portfolio constraints, costs, metrics, and output hashes.

## 10. Known limitations

- The target is a constructed quadrant of eight releases, not a directly
  observed latent economic truth. Defining releases eventually confirm it.
- The model assumes a time-homogeneous first-order transition matrix and does
  not model explicit regime duration.
- Release likelihoods are conditionally independent across blocks and dates
  given the regime. Repeated weekly claims and correlated macro releases can
  make posterior confidence too high.
- `ICSA` is the only weekly claims feature in the frozen likelihood; `CCSA`
  remains available for later versions.
- Yahoo adjusted history is a current, mutable data snapshot and the ETF
  universe was selected ex ante for this research build, not from a historical
  investable-universe database. This is not direct return leakage, but it limits
  reproducibility and introduces universe-selection risk.
- The ETF universe is compact and US-centric. `GLD`, `SPY`, and credit dominate
  the fitted allocation; this is not a comprehensive global FICC portfolio.
- The 97-target forecast evaluation and 102-month backtest cover few independent
  macro cycles. Sensitivity grids and bootstrap intervals do not eliminate
  model-selection or multiple-testing risk.
- Costs are stylized at five basis points per dollar traded; taxes, borrowing,
  market impact, and live operational failures are absent.
- The repository has no open-source license. Public visibility alone does not
  grant reuse rights; a license should be chosen explicitly before soliciting
  outside contributions.

## 11. Repository organization

```text
configs/models/                         frozen stage specifications
data/manifests/                         tracked provenance and content hashes
data/raw/                               local provider cache; ignored by Git
data/processed/                         local audit tables; ignored by Git
docs/models/m01_deterministic_composite detailed specifications and audit
legacy/                                 original team artifacts and provenance
results/published/                      compact public outputs
scripts/                                ETF acquisition utility
src/regime_allocation/                  data, model, portfolio, and backtest code
tests/                                  unit, causality, and integration tests
```

The README is intended to be sufficient for understanding the architecture.
The model-specific documents preserve formal contracts, data dictionaries,
sensitivity detail, and implementation-level caveats without crowding the
project root.

## 12. Reproduction

The frozen rebuild requires Python 3.11 or newer. The exact snapshot dates are
in the configurations and ETF command below.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

python -m regime_allocation.cli.build_m01_dataset --provider auto
python -m regime_allocation.cli.build_m01_evidence --provider auto
python -m regime_allocation.cli.build_m01_transition
python -m regime_allocation.cli.build_m01_inference

python scripts/download_etf_history.py `
  --start 2008-01-01 `
  --as-of 2026-07-18 `
  --raw-dir data/raw/yahoo_finance/us_cross_asset_etf_universe_v1 `
  --processed-dir data/processed/us_cross_asset_etf_universe_v1 `
  --manifest data/manifests/us_cross_asset_etf_universe_v1.json

python -m regime_allocation.cli.build_m01_backtest
python -m pytest
```

`auto` prefers the official FRED API when `FRED_API_KEY` exists and otherwise
uses the retained keyless ALFRED path. Existing compatible caches are validated
before a network request; authenticated refreshes and keyless artifacts occupy
separate raw-data trees. The key is read only from the environment and is never
written to a manifest, cache identity, URL, or log. See
[`docs/data_access.md`](docs/data_access.md) and [`SECURITY.md`](SECURITY.md).

The manual GitHub Action reproduces all five stages, including ETF acquisition
and the portfolio backtest, and uploads only non-secret artifacts. Raw and
processed local research data are intentionally ignored by Git; tracked
manifests provide their paths and hashes.

## Attribution

The original course project was produced by Zekai Yao, Mianchen Zhang, Gavin
Huang, and Serin Gleave. Model 01 is Gavin Huang's later independent
reimplementation, correction, and extension. This product uses the FRED API but
is not endorsed or certified by the Federal Reserve Bank of St. Louis.
