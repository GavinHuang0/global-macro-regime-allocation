# Global Macro Regime Detection and Allocation

## Model 02: current inference publication

Model 02 is a deliberately modest revision of Model 01, kept in the separate
`m02_soft_composite` configuration, package, data, manifest, documentation, and
result namespaces. Its current causal replay is published through 20 July 2026.
The declared inference baseline is `student_t_7_reduced_core`.

| Publication role | Model ID | Evidence and purpose |
|---|---|---|
| Current baseline | `student_t_7_reduced_core` | ICSA weekly labor stress, joint real-retail/implicit-price consumer evidence, and joint capital-goods activity/pipeline evidence |
| Major benchmark | `transition_only` | OLS VAR(1) transition dynamics only |
| Major benchmark | `partial_only` | Transition dynamics plus partial defining releases |
| Frozen predecessor benchmark | `student_t_7_combined` | The baseline selected by the earlier inference-sensitivity stage, preserved unchanged |
| Current sensitivity | `student_t_7_reduced_core_with_expectations` | Current baseline plus inflation expectations |

The baseline retains a four-month joint Gaussian score state, OLS VAR(1)
dynamics, sequential partial defining-release updates, exact end-of-day
conditioning when a monthly composite is complete, and fixed-$\nu=7$
Student-$t$ non-defining emissions. The [promotion report](docs/models/m02_soft_composite/baseline_promotion.md)
contains the full model hierarchy, evidence definitions, evaluation, latest
probabilities, limitations, and artifact contract.

At the primary `before_any_defining_release` checkpoint, 183 common months are
scored. The current baseline means are:

| NLPD | Quadrant cross-entropy | Quadrant Brier distance | Hard-quadrant accuracy |
|---:|---:|---:|---:|
| 26.854150 | 1.535693 | 0.150466 | 0.377049 |

Candidate-minus-reference deltas versus `transition_only` are $-0.067415$ NLPD,
$+0.000418$ cross-entropy, $+0.000860$ Brier distance, and $+0.016393$
accuracy. Against `partial_only`, they are $-0.066712$, $-0.001260$,
$-0.000051$, and $+0.016393$; against the frozen predecessor they are
$-0.000915$, $-0.000240$, $-0.000118$, and $+0.005464$. Lower is better for
the three losses and higher is better for accuracy. Every corresponding
12-month moving-block bootstrap interval includes zero, so none of these
differences supports a statistical-significance claim.

The graph was selected adaptively after inspecting the same causal history.
There is no untouched holdout, and this operational promotion is not a
confirmatory out-of-sample result. Inflation expectations remains a sensitivity:
only three updates were estimable and its measured effects are negligible. The
[current sensitivity catalog](results/published/m02_soft_composite/current/sensitivity_catalog.csv)
retains that variant, 23 earlier feature variants, and links all four historical
sensitivity namespaces.

As of 20 July 2026, the current baseline reports:

| Reference month | Growth up / inflation up | Growth down / inflation up | Growth up / inflation down | Growth down / inflation down |
|---|---:|---:|---:|---:|
| June 2026 | 0.177380 | 0.161240 | 0.354790 | 0.306591 |
| July 2026 | 0.159020 | 0.306653 | 0.266124 | 0.268204 |

The promoted baseline now has a weekly allocation and backtest in
[`portfolio_allocation.md`](docs/models/m02_soft_composite/portfolio_allocation.md)
and
[`portfolio_backtest_results.md`](docs/models/m02_soft_composite/portfolio_backtest_results.md).
Its forecast and holding horizons are aligned: estimates use Monday-anchored
open-to-next-week-open returns, a 260-week minimum, 104-pseudo-week regime-mean
shrinkage, and 52x covariance annualization. Each holding is associated with
the hard label for its Monday's calendar month, and both its ending open and
label must be available by the Sunday fit cutoff. The optimizer maximizes
expected one-week return net of one estimated rebalance cost.
The formal sample contains 445 complete weeks from January 2018 through the
week of 6 July 2026. The corrected posterior optimizer returned 123.6021% in
total, with 9.8595% CAGR, 10.4170% annualized volatility, 0.95565 zero-rate
Sharpe, 0.71690 BIL-excess Sharpe, and -21.7545% maximum drawdown. The
otherwise identical pooled-mean ablation returned 123.0780%, or 9.8294% CAGR.
Their posterior-minus-pooled annualized mean difference is +0.0187066%, with a
95% 26-week block-bootstrap interval of [-0.0876424%, +0.1297494%] and 0.6281
bootstrap fraction above zero. The interval includes zero, so the weekly result
does not establish a reliable incremental posterior edge. The old 108.79%
total-return and 8.98% CAGR figures are superseded by this horizon-mismatch
correction; the change aligns units and is not a Model 02 signal-quality
improvement.

An exploratory post-result strategy,
`pooled_anchor_posterior_25pct`, blends each weekly target as 75% pooled mean
plus 25% posterior. Simulated on its own drift and trades with the same
five-basis-point one-way cost, it returned 123.2173866%, with 9.8373991% CAGR,
10.4809091% volatility, 0.9485128 zero-rate Sharpe, 0.7112097 BIL-excess
Sharpe, -21.9286402% maximum drawdown, 27.1480535% annualized one-way turnover,
and 2.982445 basis points of annualized cost drag. Posterior minus anchor has
annualized mean +0.01371398% with 95% interval
[-0.0660674%, +0.0969896%] and 0.6249 bootstrap fraction above zero; anchor
minus pooled mean is +0.00499261% with interval
[-0.0214632%, +0.0327264%] and fraction 0.6376. Both intervals include zero.
The source strategies are unchanged, and their latest targets coincide, so the
latest anchor target is unchanged as well. The convex blend inherits their
linear caps but has no separately audited 10% constraint under one common
covariance estimate. The 25% sleeve is exploratory, non-promoted, and not an
independently validated improvement. The benchmark paths and Model 01 remain
frozen below.

### Historical Model 02 reduced-core feature revision

A locked 26-variant, 23-comparison replay tested the first reduced-core
revision without changing the then-selected `student_t_7_combined`. Every experimental arm
removes the legacy JOLTS and aggregate-housing likelihoods while preserving
their source data. The replay also tests import prices against both legacy
input costs and no price block; replaces nominal consumer responses with real
retail activity, an implicit retail-price coordinate, and vehicle units;
rotates capital-goods data into shipments activity and the causally
standardized forward-pipeline feature

$$
D_m=\operatorname{zscore}_{<t_e}
\left[100\Delta\log\left(\frac{\text{orders}_m}{\text{shipments}_m}\right)\right];
$$

and implements the reference-month-safe claims factorization

$$
p(I_t,C_t\mid\boldsymbol Z)
=p(I_t\mid G_{q_I(t)})
 p(C_t\mid I_t,G_{q_C(t)}).
$$

All 216 frozen-control semantic invariance checks pass, and no predeclared
feature contrast survives Holm correction. At the primary checkpoint, the
all-revised development candidate improves mean quadrant cross-entropy by
0.002669, Brier distance by 0.001012, continuous-score NLPD by 0.01549, and
hard-quadrant accuracy by 2.73 percentage points relative to the then-selected
baseline, but every relevant bootstrap interval crosses zero.

Applying the locked component rules gives a narrower recommendation: retain
real retail activity plus its implicit-price coordinate; omit vehicle units;
retain shipments activity plus the orders/shipments pipeline; use no price
block because import prices show mixed later-checkpoint effects; and retain
ICSA alone until a dependence-aware CCSA equation can match the independently
modeled pair. The import and both CCSA designs remain named sensitivities. This
is same-history feature-selection evidence, not an untouched validation, so
that stage did not change its selected baseline. Exact equations, coverage, paired tests,
and limitations are in
[`feature_revision.md`](docs/models/m02_soft_composite/feature_revision.md).

### Historical Model 02 existing-block attribution

The seven non-defining observation models in `student_t_7_combined` have now
been tested in two causal arms: each block added alone to `partial_only`, and
each block removed alone from the then-selected baseline. All 216 frozen-control
invariance checks pass. No full-sample atomic contrast survives Holm correction
across the fourteen add/remove tests.

Weekly initial claims is the clear continuous-score contributor. Excluding
March-May 2020 with the two remaining calendar segments resampled separately,
its standalone and conditional NLPD benefits are 0.01934 and 0.01936. The 95%
intervals are \([0.01566,0.02391]\) and \([0.01575,0.02387]\), and both
fourteen-test Holm-adjusted \(p\)-values are 0.0028. The corresponding
full-sample tests are not significant, and claims slightly worsens Brier
distance. The post-May-2020 segment drives the exclusion sensitivity, so it
does not establish stable value across eras. Consumer demand and legacy
business investment have small, directionally positive effects in both arms.
The lagged monthly labor-demand likelihood, aggregate housing, and
intermediate-materials input costs are neutral-to-negative, while inflation
expectations has only three applied updates and cannot be judged.

The resulting reduced-core recommendation is exploratory: retain initial
claims and, provisionally, legacy consumer demand and orders/shipments; disable
the lagged JOLTS likelihood and intermediate-materials input costs; park
inflation expectations pending usable real-time history; and exclude aggregate
housing. Separate locked arms should test single-family housing, joint
initial/continued claims, an import-price replacement, and response-level
consumer alternatives. That stage did not change its baseline on the same sample
used for feature selection. Exact mathematics, coverage, bootstrap results,
direct replacement tests, and recommendations are in
[`existing_block_attribution.md`](docs/models/m02_soft_composite/existing_block_attribution.md).

### Prior Model 02 evidence-block experiment

Six evidence changes were tested without altering the then-selected inference
architecture: regional manufacturing surveys, continued claims, cleaner
consumer quantities, single-family housing, import prices, and a
capital-goods backlog ratio. The frozen `student_t_7_combined` path passed 64
invariance checks spanning its evaluation rows, semantic event updates, and
latest marginals.

Continued claims produced the best isolated mean change at the primary
pre-defining-release checkpoint: NLPD $-0.0170$, quadrant cross-entropy
$-0.00253$, Brier distance $-0.000934$, and hard-quadrant accuracy
$+2.19$ percentage points relative to the then-selected baseline. Its 12-month block-bootstrap
intervals still cross zero, however, and none of the six priorities survives
Holm correction. Combining all six worsens quadrant cross-entropy by $0.0516$
and Brier distance by $0.0243$; both exploratory 95% intervals are strictly
above zero.

The combined-candidate residual diagnostic also rejects the independent-block
approximation broadly: 140 of 966 valid cross-model tests and 35 of 57 valid
serial tests reject after Benjamini--Hochberg adjustment. Accordingly, no
candidate was promoted, and `student_t_7_combined` remained the baseline at that stage.
The exact feature definitions, causal data windows, bootstrap results, and
dependence tests are documented in
[`evidence_block_experiments.md`](docs/models/m02_soft_composite/evidence_block_experiments.md).

The defining component set is unchanged: four first-release growth indicators
and four first-release inflation indicators. Model 02 makes two score changes:

1. payroll employment is transformed as monthly log growth rather than an
   absolute change in thousands of employees;
2. the equal-weight growth and inflation composites are not subsequently
   averaged over three months.

For component $k$ and reference month $m$, the current and previous levels
come from the same first-release vintage. Log-transformed components use

$$
u_{k,m}=100\log\left(\frac{x_{k,m}}{x_{k,m-1}}\right),
$$

while unemployment uses the negative monthly difference. Each component is
standardized using an expanding mean and sample standard deviation estimated
only from at least 60 strictly earlier observations:

$$
z_{k,m}
=\frac{u_{k,m}-\widehat\mu_{k,m-1}}
{\widehat\sigma_{k,m-1}}.
$$

The unsmoothed scores are

$$
G_m=\frac14\sum_{k\in\mathcal G}z_{k,m},
\qquad
I_m=\frac14\sum_{k\in\mathcal I}z_{k,m}.
$$

All four components are required on each axis; missing data never trigger
dynamic reweighting. No hard quadrant label is assigned.

The authenticated FRED build contains a gapless 318-row component panel from
January 2000 through June 2026 and 249 complete score pairs through May 2026.
Authenticated first-release core PCE begins in July 2000, so January-June 2000
are retained but cannot have a complete eight-component score. With the
60-observation causal warm-up, the first complete score pair is July 2005. The
latest May 2026 scores, available on 25 June 2026, are:

| Score | Value |
|---|---:|
| Growth | 0.027394 |
| Inflation | 0.410476 |

The component feed and completed-score feed have deliberately different
cutoffs. As of 20 July, six June components had been released: payrolls,
unemployment, earnings, core CPI, producer prices, and industrial production.
They update the live June score distribution on their actual July publication
dates even though real consumption and core PCE are still missing, so no June
composite, exact score, VAR training pair, or evaluation target is fabricated.

One diagnostic remains deliberately unresolved: the April 2020 payroll shock
has an expanding z-score of $-103.0223$ and produces a
growth score of $-51.8317$. The score is correct under the unbounded classical
standardization specified here. The current inference baseline preserves this rule,
while a later sensitivity should address crisis-outlier robustness.

The observed score $\boldsymbol s_m=(G_m,I_m)^\top$ is mapped through the
reporting variable

$$
\boldsymbol U_m
=\boldsymbol s_m+\boldsymbol\epsilon_m^{\mathrm{map}},
\qquad
\boldsymbol\epsilon_m^{\mathrm{map}}\sim
\mathcal N(\boldsymbol 0,\boldsymbol\Omega_{\mathrm{map},m}),
$$

where

$$
\boldsymbol\Omega_{\mathrm{map},m}
=\boldsymbol\Omega_{\mathrm{disagreement},m}
+\boldsymbol\Omega_{\mathrm{revision},m}.
$$

Disagreement is the diagonal expanding pool of the two axes' monthly
delete-one-component jackknife variances. Revision covariance is estimated in
the same standardized score units from exact later as-of vintages, using only
mature errors strictly available before each historical score cutoff. The
12-month revision horizon is the probability-map default; the nested 3-month
revision horizon is reported
separately rather than added.

For May 2026, the 12-month-map Gaussian quadrant weights are:

| Quadrant | Probability |
|---|---:|
| Growth up / inflation up | 36.70% |
| Growth down / inflation up | 35.56% |
| Growth up / inflation down | 14.13% |
| Growth down / inflation down | 13.61% |

These are score-definition weights, not transition forecasts or
release-updated posteriors. The 12-month map has 214 causal monthly probability
vectors from June 2008 through May 2026.

The transition and filtering state is the released composite-score center

$$
\boldsymbol Z_m=\boldsymbol s_m=(G_m,I_m)^\top.
$$

Once its complete score is released on date $T_m$, $\boldsymbol Z_m$ is known
exactly. The perturbed reporting variable

$$
\boldsymbol U_m=\boldsymbol Z_m+\boldsymbol\epsilon_m^{\mathrm{map}},
\qquad
\boldsymbol\epsilon_m^{\mathrm{map}}
\sim\mathcal N(\boldsymbol 0,\boldsymbol\Omega_{\mathrm{map},m}),
$$

exists only to translate a score distribution into soft quadrant membership.
Mapping covariance is not propagated as uncertainty in the exact VAR state.

The expanding VAR(1) is fitted to deterministic score centers:

$$
\boldsymbol Z_{m+1}
=\widehat{\boldsymbol c}_m
+\widehat{\boldsymbol A}_m\boldsymbol Z_m
+\boldsymbol\eta_{m+1},
\qquad
\boldsymbol\eta_{m+1}\sim
\mathcal N(\boldsymbol 0,\widehat{\boldsymbol Q}_m).
$$

The next innovation is assumed conditionally independent of the current state.
Given an exact source score, the latent next-month prior is

$$
\boldsymbol Z_{m+1}\mid\mathcal D_{T_m}
\sim\mathcal N\!\left(
\widehat{\boldsymbol c}_m
+\widehat{\boldsymbol A}_m\boldsymbol s_m,
\widehat{\boldsymbol Q}_m
\right).
$$

The standalone transition report adds the latest causally available mapping
covariance only to its quadrant readout:

$$
\operatorname{Var}(\boldsymbol U_{m+1}\mid\mathcal D_{T_m})
=\widehat{\boldsymbol Q}_m
+\boldsymbol\Omega_{m+1}^{\mathrm{proxy}}.
$$

It never uses

$$
\widehat{\boldsymbol A}_m\boldsymbol\Omega_{\mathrm{map},m}
\widehat{\boldsymbol A}_m^\top.
$$

The VAR is refitted at every eligible source cutoff after a 60-pair warm-up,
using only exact consecutive monthly pairs then available. Coefficients may use
earlier deterministic-score pairs even when those months predate the Gaussian
map; a prior is published only when the source month has that map. For the June
2026 target, the corrected quadrant prior is 27.29%
growth-up/inflation-up, 36.33% growth-down/inflation-up, 22.44%
growth-up/inflation-down, and 13.94% growth-down/inflation-down. This prior
became available on 25 June 2026, so its
late-within-month timing must be retained when later evidence updates are
evaluated. None of the historical one-step priors was available at the start of
its target month; a beginning-of-month prior will require a separately specified
multi-step propagation from the latest state then observable.

The plain OLS transition specification deliberately does not clip the April 2020 score or force
VAR stability. That cutoff produces the only unstable historical fit, and the
shock leaves the latest growth innovation variance unusually large. A robust
VAR should therefore be a separately named sensitivity, not an undocumented
change to this transition specification.

### Model 02 release evidence

Non-defining evidence is organized into observation models rather than treated
as one conditionally independent scalar stream:

| Economic block | Observed responses | State-loading policy |
|---|---|---|
| Weekly labor stress | Causal log-AR(1) innovation in initial claims (`ICSA`) | Growth estimated; inflation fixed to zero |
| Monthly labor demand | Changes in JOLTS openings, hires, quits, and layoffs/discharges rates | Both estimated; stronger inflation shrinkage except for quits |
| Consumer demand | Log changes in retail sales excluding motor vehicles and a same-vintage motor-vehicle component | Both estimated |
| Housing activity | Log changes in starts and permits | Both estimated; mortgage-rate change and methodology-break controls |
| Business investment | Log changes in core capital-goods orders and shipments | Both estimated; stronger inflation shrinkage |
| Inflation pressure | Change in one-year inflation expectations; log change in intermediate-materials prices | Primarily inflation; the two asynchronous releases have separate models |

Existing point-in-time Model 01 evidence is hash-verified and reused. Only
`MORTGAGE30US`, `ANXAVS`, `EXPINF1YR`, and `WPSID61` are newly retrieved, using
the authenticated FRED API and never serializing `FRED_API_KEY`. Monthly
features use same-vintage changes and causal expanding standardization.
`EXPINF1YR` is eligible only when first published within its own reference
month; its month-end-relative release lag can therefore be negative without
being look-ahead leakage.

For observation model $b$ and an event referring to month $q(e)$,

$$
\boldsymbol y_e
=\boldsymbol a_b
+\boldsymbol H_b\boldsymbol Z_{q(e)}
+\boldsymbol C_b\boldsymbol v_e
+\boldsymbol\epsilon_e,
\qquad
\boldsymbol\epsilon_e\sim
\mathcal N(\boldsymbol 0,\boldsymbol R_b).
$$

The state loadings in $\boldsymbol H_b$ use response-by-axis ridge penalties;
the intercept and observed controls are unpenalized, and the claims inflation
loading is exactly zero. A positive base penalty is chosen from a frozen grid
by rolling-origin predictive Gaussian negative log likelihood, holding all rows
with the same availability timestamp out together. The final residual
covariance $\boldsymbol R_b$ is a single Ledoit–Wolf estimate for that
observation model, followed by a small eigenvalue floor. Training availability
is the latest of the release, target-score, and control availability dates, and
must be strictly earlier than the event being scored.

### Rolling four-month Bayesian filter

The filter retains an eight-dimensional joint Gaussian for four consecutive
score centers:

$$
\boldsymbol Z_{m-3:m}
=\left(
\boldsymbol Z_{m-3}^\top,
\boldsymbol Z_{m-2}^\top,
\boldsymbol Z_{m-1}^\top,
\boldsymbol Z_m^\top
\right)^\top.
$$

At each month roll it drops the oldest two coordinates, copies the six
overlapping coordinates and their cross-covariances, and appends a VAR forecast;
$\boldsymbol Q$ is added only to the new month's $2\times2$ covariance block.
An event conditions the coordinates for its actual reference month and can
update other months through cross-time covariance. Events are never silently
retargeted to the current month.

On a date with several releases, all emission fits use the common pre-release
information set, then the configured Gaussian factors are applied in stable
model order. Complete composite scores are processed after release evidence at
the end of their availability day as exact, zero-noise observations of
$\boldsymbol Z_m$. Evidence arriving after its target score is exact remains a
predictive-residual diagnostic but cannot alter that target. This timing is
especially important for JOLTS, whose reference-month score is often already
exact by the time the JOLTS release arrives; the event audit, rather than a
retargeting rule, will quantify those no-op updates.

Monthly quadrant probabilities add the latest causally eligible
$\boldsymbol\Omega_{\mathrm{map}}$ only at readout. Four-month path
probabilities are approximated reproducibly with a scrambled Sobol sequence.
The frozen upstream replay adds the four mapping covariances block-diagonally,
which assumes
mapping perturbations are independent across months; that is an explicit
reporting approximation, not a property established by the data.

The frozen Gaussian replay compares the evidence filter with an otherwise
identical transition-only comparator at two explicitly separated information
cutoffs. The
primary forecast is frozen at the start of the completed score's availability
day, after any deterministic calendar roll but before every release sharing
that date. A post-release/pre-exact forecast is retained only as a sensitivity,
because the source artifacts do not provide dependable intraday ordering.
Published metrics are score-center Gaussian negative log predictive density,
growth and inflation RMSE, and cross-entropy, Brier distance, and
Kullback–Leibler divergence relative to the exact score's soft quadrant map.

Across 188 strict-pre-day score completions, the evidence filter did not
uniformly improve on the transition-only comparator:

| Lower-is-better metric | Evidence filter | Transition only | Difference |
|---|---:|---:|---:|
| Growth-score RMSE | 3.441 | 4.238 | -0.797 |
| Inflation-score RMSE | 0.852 | 0.733 | +0.119 |
| Mean Gaussian NLPD | 28.031 | 26.247 | +1.785 |
| Median Gaussian NLPD | 1.770 | 1.817 | -0.047 |
| Mean quadrant cross-entropy | 1.573 | 1.527 | +0.045 |
| Mean quadrant Brier distance | 0.160 | 0.151 | +0.008 |
| Mean quadrant KL divergence | 0.515 | 0.470 | +0.045 |

The apparently better full-sample growth RMSE is dominated by the 2020 shock.
Excluding March--June 2020, growth RMSE is 0.893 for the evidence filter versus
0.745 for transition only. Conversely, the evidence filter has lower NLPD in
134 months, higher NLPD in 45, and ties in 9, but its April 2020 NLPD penalty is
about 783 points larger. The median and mean therefore tell materially
different stories. The post-release/pre-exact sensitivity is very close to the
strict-pre-day result and does not change this conclusion.

Dependence diagnostics use causal whitened predictive innovations measured from
the common pre-release state for every observation sharing a publication date,
so their values do not depend on the stable likelihood-application order.
They report Pearson and Spearman cross-model correlations at monthly lags
$-1$, $0$, and $+1$ with Benjamini–Hochberg adjustment, plus Ljung–Box serial
tests. Weekly claims innovations are averaged within reference month only for
cross-block comparisons; their serial tests retain every observation and use
the original weekly reference date, including same-publication catch-up
batches. Responses fitted
jointly are not redundantly cross-tested, while the two separately fitted
inflation-pressure models remain eligible for comparison. Failure to reject a
test does not establish conditional independence, particularly for short
histories.

The dependence diagnostics reject the frozen Gaussian replay's simple
factorization often:
64 of 342 valid cross-model tests have Benjamini--Hochberg $q<0.05$ (59
Pearson and 5 Spearman), and 28 of 36 valid serial tests do as well. These
statistics are exploratory because serial dependence is widespread, but they
are strong evidence against treating the blocks as cleanly independent. A
later sensitivity should use joint disturbances, block tempering, or a robust
heavy-tailed observation model.

The cross-model result is also crisis-sensitive. Excluding 2020 leaves only 2
significant Pearson tests and no significant Spearman tests after adjustment,
and the maximum absolute Pearson correlation falls from 0.930 to 0.446. Thus
the full-sample rejection should not be read as stable linear dependence in
ordinary periods; it shows that the Gaussian factorization is particularly
fragile around extreme observations.

As of 20 July 2026, the frozen upstream Gaussian evidence-filter probabilities
were 34.0%
growth-down/inflation-up, 25.1% growth-up/inflation-up, 23.6%
growth-up/inflation-down, and 17.2% growth-down/inflation-down. The distribution
is intentionally diffuse; its entropy is 1.358 versus the four-state maximum
of $\log 4\approx1.386$.

That upstream replay remains intentionally Gaussian and non-robust. The April 2020
score outlier inflates VAR process variance and can affect Gaussian emission
fits; robust scaling, robust VAR dynamics, and heavy-tailed observation errors
were therefore introduced only through explicit comparison variants. The next
section reports the baseline-selection decision and the remaining robust-VAR
and heavy-tailed-emission sensitivities;
the defining-score standardization itself remains unchanged.

### Historical partial-release baseline-selection and robustness stage

The selection replay adds information from a score-defining component as
soon as its first release arrives. If $\boldsymbol c_m$ is the eight-component
vector and $\boldsymbol Z_m=\boldsymbol W\boldsymbol c_m$ is the exact
two-score identity, a causal Ledoit--Wolf component Gaussian supplies

$$
p\!\left(
\boldsymbol c_{B,m}\mid
\boldsymbol Z_m,\boldsymbol c_{A,m}
\right)
$$

for a newly released block $B$ conditional on previously released components
$A$. Every fit uses at least 60 complete component months whose final score
was available strictly before the event. Sequential conditioning prevents one
component from being counted repeatedly. The final PCE block completes both
scores; it is skipped as a partial likelihood and the completed score is
conditioned exactly at the end of that day.

The component-event feed may therefore extend beyond the last completed-score
month. In the 20 July snapshot, June has six released components but lacks
real consumption and core PCE. Those observations update the uncertain June
state and, through the joint path covariance, the July state. June remains
ineligible for exact conditioning, transition-model training, or forecast
evaluation until all eight components make both scores observable.

Non-defining release blocks can instead use robust Student-$t$ ridge fits. For
event innovation Mahalanobis distance $\delta_e$, observed dimension $p_e$,
and tail parameter $\nu_b$, the adaptive weight is

$$
w_e=\frac{\nu_b+p_e}{\nu_b+\delta_e},
$$

and the Gaussian moment approximation uses effective measurement scale

$$
\boldsymbol R_e^{\mathrm{eff}}
=
\frac{\boldsymbol R_b}{\max(w_e,10^{-6})}.
$$

The event weight, its unfloored value, and its distance are audited. Same-day
weights use a common pre-release-day state. The predecessor baseline selected
by this historical stage fixes $\nu_b=7$; the selected-tail sensitivity chooses from
$\{4,5,7,10,\infty\}$ using calendar-year rolling origins, and a live year may
use only completed validation folds from earlier years.

Huber and Student-$t$ VAR sensitivities robustly reweight monthly bivariate
innovations but retain the same joint Gaussian score state. Retail tests either
shrink nominal retail's inflation loading ten times as strongly as its growth
loading or decompose nominal growth into real-retail and implicit-price growth.
The proposed claims-stress interaction is not fitted: only one of 242 aligned
retail dates has positive support under the predeclared two-standard-deviation
stress transform.

The primary checkpoint is the start of the final score's release day. After a
four-month burn-in, 183 months from January 2011 through May 2026 are scored:

| Stage role | Variant | Mean NLPD | Growth RMSE | Inflation RMSE | Hard accuracy | Cross-entropy |
|---|---|---:|---:|---:|---:|---:|
| Stage-selected baseline | Fixed-$\nu=7$ combined | 5.721 | 1.411 | 0.495 | 76.0% | 1.302 |
| Major benchmark | Partial only, OLS VAR(1) | 5.836 | 1.447 | 0.487 | **82.5%** | 1.300 |
| Major benchmark | Transition only, OLS VAR(1) | 26.919 | 4.295 | 0.738 | 35.0% | 1.537 |
| Sensitivity | Selected-tail combined | **5.720** | 1.423 | 0.495 | 73.2% | 1.303 |
| Sensitivity | Gaussian combined | 6.270 | **1.076** | 0.557 | 78.7% | 1.301 |
| Sensitivity | Real-retail combined | 10.387 | 1.959 | **0.448** | 72.7% | **1.297** |
| Sensitivity | Stronger-retail-shrinkage combined | 10.446 | 1.959 | 0.478 | 74.3% | 1.301 |
| Sensitivity | Huber-VAR combined | 10.450 | 1.959 | 0.479 | 73.8% | 1.301 |
| Sensitivity | Student-$t$-VAR combined | 11.285 | 2.014 | 0.467 | 74.3% | 1.301 |
| Sensitivity | Gaussian non-defining only | 28.065 | 3.496 | 0.880 | 33.9% | 1.571 |

Most of the late-month gain is partial target revelation. Adding selected-tail
non-defining evidence to partial releases lowers mean NLPD by only about 0.116
and worsens several quadrant metrics. No variant dominates every metric.
The fixed-$\nu=7$ combined model was selected as that stage's baseline because it retains
the full event-driven architecture, has nearly the best mean NLPD, and avoids
the additional annual tail-selection layer. This is a declared model-governance
choice, not a claim that it wins every reported metric.

March--May 2020 dominate the pooled mean. Excluding only those three months,
Student-$t$ VAR has the lowest mean NLPD at $-0.148$, whereas its full-sample
mean is 11.285. This reversal is a diagnostic, not permission to discard the
shock: subperiod choice is post hoc, and the full sample contains only one
pandemic. Neither ranking establishes a stable economic edge.

After processing the six June defining components, the stage-selected
fixed-$\nu=7$/OLS-VAR predecessor's 20 July readout for July 2026 has score means
$(-0.5097,0.0937)$ and quadrant probabilities 19.95%
growth-up/inflation-up, 33.84% growth-down/inflation-up, 24.36%
growth-up/inflation-down, and 21.84% growth-down/inflation-down. This is the
historical stage-selected readout; the benchmark and sensitivity
readouts remain published so that its model uncertainty is visible.

The full baseline, benchmark, and sensitivity specification is in
[`inference_sensitivities.md`](docs/models/m02_soft_composite/inference_sensitivities.md).
Public results are under
[`results/published/m02_soft_composite/inference_sensitivities/`](results/published/m02_soft_composite/inference_sensitivities/),
including the machine-readable
[`model_registry.csv`](results/published/m02_soft_composite/inference_sensitivities/model_registry.csv),
with provenance in
[`data/manifests/m02_inference_sensitivities.json`](data/manifests/m02_inference_sensitivities.json).

The complete mathematical definition, coverage limitations, and artifact
contract are in
[`docs/models/m02_soft_composite/README.md`](docs/models/m02_soft_composite/README.md).
Published score history and provenance are under
[`results/published/m02_soft_composite/`](results/published/m02_soft_composite/)
and
[`data/manifests/m02_soft_composite.json`](data/manifests/m02_soft_composite.json).
The probability-map snapshot is under
[`results/published/m02_soft_composite/probability_map/`](results/published/m02_soft_composite/probability_map/)
with provenance in
[`data/manifests/m02_probability_map.json`](data/manifests/m02_probability_map.json).
The VAR priors are under
[`results/published/m02_soft_composite/transition/`](results/published/m02_soft_composite/transition/)
with provenance in
[`data/manifests/m02_var1_transition.json`](data/manifests/m02_var1_transition.json).
The release-evidence contract is in
[`configs/models/m02_release_evidence.yaml`](configs/models/m02_release_evidence.yaml)
with provenance in
[`data/manifests/m02_release_evidence.json`](data/manifests/m02_release_evidence.json).
The filter specification is frozen in
[`configs/models/m02_event_driven_bayesian_filter.yaml`](configs/models/m02_event_driven_bayesian_filter.yaml);
its public results are under
[`results/published/m02_soft_composite/bayesian_filter/`](results/published/m02_soft_composite/bayesian_filter/),
with provenance in
[`data/manifests/m02_event_driven_bayesian_filter.json`](data/manifests/m02_event_driven_bayesian_filter.json).

---

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
| `m02_soft_composite` | Unsmoothed scores, soft quadrant map, four-month Gaussian joint state, partial defining updates, reduced-core fixed-$\nu=7$ Student-$t$ evidence, OLS VAR(1), weekly posterior allocation | **`student_t_7_reduced_core` current; weekly allocation published** |
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
- Model 02 [latest weekly allocation](results/published/m02_regime_allocation_backtest/latest_allocation.json),
  [performance](results/published/m02_regime_allocation_backtest/performance_summary.csv),
  [weekly returns](results/published/m02_regime_allocation_backtest/weekly_returns.csv),
  [weekly weights](results/published/m02_regime_allocation_backtest/weekly_weights.csv),
  [allocation sensitivities](results/published/m02_regime_allocation_backtest/sensitivity_metrics.csv),
  and [paired uncertainty](results/published/m02_regime_allocation_backtest/comparison_uncertainty.csv).

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
python -m regime_allocation.cli.build_m02_current_baseline
python -m regime_allocation.cli.build_m02_backtest
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
