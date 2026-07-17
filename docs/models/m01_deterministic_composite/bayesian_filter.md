# Model 01 event-driven Bayesian filter

The frozen historical run and sensitivity findings are reported in
[`bayesian_filter_results.md`](bayesian_filter_results.md).

## Scope

This stage turns Model 01's transition-only monthly prior into an event-driven
posterior over four consecutive monthly regimes. It estimates a separate
release likelihood for each leading-evidence block, applies each publication
date's usable evidence, and later incorporates the deterministic target label
as a hard confirmation. It also performs a walk-forward probabilistic
evaluation and a frozen one-parameter-at-a-time sensitivity study.

The baseline is intentionally modest:

- a four-state, first-order monthly transition model;
- a joint posterior over four reference months;
- regime-specific release-block means with pseudo-count shrinkage;
- one Ledoit-Wolf residual covariance matrix shared by all regimes in a block;
- a multivariate Student-$t$ likelihood with seven degrees of freedom;
- complete release vectors for monthly blocks;
- only initial claims (`ICSA`) in the weekly-claims block;
- exact deterministic confirmations at the end of their availability dates.

The frozen machine-readable specification is
[`m01_event_driven_bayesian_filter.yaml`](../../../configs/models/m01_event_driven_bayesian_filter.yaml).
This document defines the baseline before results are examined. It does not
claim that the baseline or any sensitivity variant improves on the transition
model.

This stage remains a regime-inference experiment. It does not optimize a
portfolio, simulate execution, or constitute an investment recommendation.

## 1. State and fixed-lag path

The permanent state order is:

1. `growth_up_inflation_up`;
2. `growth_down_inflation_up`;
3. `growth_up_inflation_down`;
4. `growth_down_inflation_down`.

On a day in calendar month $m$, the filter retains

$$
q_t(a,b,c,d)
=P(R_{m-3}=a,R_{m-2}=b,R_{m-1}=c,R_m=d\mid\mathcal F_t).
$$

There are $4^4=256$ possible paths. The four-month window allows a delayed
release to revise a recent reference month and, through path dependence, the
current-month marginal. It does not make the transition process fourth order.

At initialization, the oldest state has a uniform distribution. The causal
transition matrix available at the initial cutoff propagates that distribution
through the other three months. Deterministic labels whose availability dates
are strictly before the initial cutoff are then clamped atomically, and the
remaining path mass is normalized. This explicit rule avoids estimating an
initial distribution from future regime frequencies.

At the beginning of a new calendar month, the filter shifts with the expanding
transition matrix available strictly before the roll date:

$$
q^-_{m+1}(b,c,d,e)
=\sum_{a=1}^4 q_m(a,b,c,d)\overline A^{(t)}_{de}.
$$

The transition matrix uses the same consecutive-month eligibility and
Dirichlet-$0.5$ smoothing rules specified in
[`transition_model.md`](transition_model.md). It is re-estimated causally at
each monthly roll; the final-sample matrix is never backfilled into an earlier
forecast.

## 2. Evidence blocks

The canonical point-in-time features come from
[`leading_evidence_data.md`](leading_evidence_data.md). The baseline likelihood
uses the following vectors.

| Block | Baseline vector | Dimension |
|---|---|---:|
| `weekly_claims` | `initial_claims_innovation` from `ICSA` | 1 |
| `jolts` | openings-, hires-, quits-, and layoffs-rate changes | 4 |
| `retail_sales` | total and ex-motor-vehicle log changes | 2 |
| `housing` | housing-starts and building-permits log changes | 2 |
| `durable_goods` | total durable-goods and core capital-goods log changes | 2 |

Continued claims (`CCSA`) remain in the gathered evidence data but are excluded
from the initial likelihood. Initial and continued claims have different
reference weeks and can appear in a shared publication batch. Beginning with
`ICSA` keeps the baseline likelihood univariate and avoids silently pretending
that mismatched reference weeks form one contemporaneous vector. A later model
can define an explicit lagged two-series claims observation equation.

For every monthly block, an event is usable only when all configured features
are available for the same reference period and canonical event. The model does
not fit a lower-dimensional marginal when one component is missing. This
complete-vector policy gives every historical observation the same meaning and
prevents the covariance dimension from changing across dates.

For claims, the state reference month is the calendar month containing the
reported reference week. For monthly releases, it is the calendar month of the
reference period. Each feature must have status `available` in the canonical
event table.

## 3. Strictly causal likelihood-training sample

Let $d$ be the publication date of the event currently being processed. For
block $b$, the expanding training set is

$$
\mathcal T_b(d)=
\{(y_e,R_{q(e)}):
d_e<d,\ T_{q(e)}<d,\ y_e\text{ is a complete usable block vector}\},
$$

where:

- $d_e$ is the historical event's publication date;
- $q(e)$ is its regime reference month;
- $T_{q(e)}$ is that deterministic label's `label_available_at` date;
- $y_e\in\mathbb R^{p_b}$ is its causal standardized feature vector.

Both inequalities are strict. An event published today cannot train today's
likelihood, and a label confirmed today cannot enter today's likelihood fit.
All releases sharing $d$ therefore use the same earlier information set.

The baseline requires at least 24 complete labeled vectors in a block. If the
threshold is not met, the covariance is unusable, a required numerical check
fails, or a block mean cannot be constructed, the event is skipped and the
reason is recorded. The implementation must never substitute a full-sample fit
or silently borrow a later label.

The weekly claims observations remain individual training vectors. Multiple
weeks associated with one monthly regime are not averaged into one row. This
gives the weekly block more observations and more update opportunities than a
monthly block; that is an explicit baseline limitation rather than an implied
claim of conditional independence.

## 4. Regime-specific shrunk means

For block $b$ at cutoff $d$, let $n_{b,r}$ be the number of training vectors
assigned to regime $r$. Define the unshrunk regime mean and the global block
mean as

$$
\bar y_{b,r}=\frac{1}{n_{b,r}}
\sum_{e\in\mathcal T_b(d):R_{q(e)}=r}y_e,
$$

$$
\bar y_b=\frac{1}{n_b}\sum_{e\in\mathcal T_b(d)}y_e.
$$

The production mean is

$$
\widetilde\mu_{b,r}
=\frac{n_{b,r}\bar y_{b,r}+\kappa\bar y_b}{n_{b,r}+\kappa},
\qquad \kappa=5.
$$

Thus the block-wide mean contributes five pseudo-observations to each regime.
Well-populated regimes remain data-driven, while a rare or not-yet-observed
regime is pulled toward the global mean. When $n_{b,r}=0$ and $\kappa>0$, its
mean is exactly $\bar y_b$.

This is shrinkage of conditional means, not a Bayesian posterior over mean
parameters. The baseline carries the resulting point estimates into the
filter. Estimation uncertainty is examined indirectly through walk-forward
performance and the $\kappa$ sensitivity grid.

## 5. Shared Ledoit-Wolf residual covariance

For each labeled training vector, construct the residual

$$
e_e=y_e-\widetilde\mu_{b,R_{q(e)}}.
$$

All regimes' residuals are pooled within block $b$. Let
$S_b^{\mathrm{emp}}$ be their maximum-likelihood empirical covariance and let

$$
F_b=\frac{\operatorname{tr}(S_b^{\mathrm{emp}})}{p_b}I_{p_b}
$$

be the spherical target. The Ledoit-Wolf estimate has the form

$$
\widehat C_b^{\mathrm{LW}}
=(1-\widehat\lambda_b)S_b^{\mathrm{emp}}
+\widehat\lambda_b F_b,
\qquad 0\leq\widehat\lambda_b\leq1,
$$

where $\widehat\lambda_b$ is estimated from the causal residual sample. The
residuals are treated as centered around the already fitted regime means; the
covariance routine must not replace those means with a second, future-aware
location fit.

One covariance is shared by the four regimes in a block. This pools scarce
data and prevents four small, unstable covariance matrices, while still
allowing the regimes to differ in location.

For the one-dimensional `ICSA` likelihood, the spherical target equals the
empirical variance. Ledoit-Wolf shrinkage is therefore algebraically a no-op:
it cannot change a $1\times1$ covariance. The implementation still uses the
same estimator interface and records the fitted variance and shrinkage
coefficient. Covariance-estimator sensitivity for this block should be
interpreted as an invariance check, not as three genuinely different models.

## 6. Student-$t$ parameterization

For block dimension $p_b$, Model 01 uses the multivariate Student-$t$ density

$$
f(y\mid\mu,\Psi,\nu)
=\frac{\Gamma((\nu+p_b)/2)}
{\Gamma(\nu/2)(\nu\pi)^{p_b/2}|\Psi|^{1/2}}
\left[
1+\frac{1}{\nu}(y-\mu)^\top\Psi^{-1}(y-\mu)
\right]^{-(\nu+p_b)/2}.
$$

Here $\Psi$ is a shape matrix, not the covariance. For $\nu>2$,

$$
\operatorname{Cov}(Y)=\frac{\nu}{\nu-2}\Psi.
$$

The baseline fixes $\nu=7$ and wants the $t$ covariance to equal the fitted
Ledoit-Wolf residual covariance. It therefore uses

$$
\widehat\Psi_b
=\frac{\nu-2}{\nu}\widehat C_b^{\mathrm{LW}}
=\frac{5}{7}\widehat C_b^{\mathrm{LW}}.
$$

The regime-conditioned block likelihood is

$$
y_e\mid R_{q(e)}=r
\sim t_7(\widetilde\mu_{b,r},\widehat\Psi_b).
$$

Seven degrees of freedom provide heavier tails than a Gaussian without making
the covariance undefined. Every fit must validate finite means, a finite
positive-definite shape matrix, a finite log determinant, and finite log
likelihoods. Calculations are performed in log space.

## 7. Event-level update

For a usable event $e$ from block $b$ referring to month $q(e)$, its likelihood
on candidate path $s$ is

$$
L_e(s)=f_b(y_e\mid R_{q(e)}=r_{q(e)}(s)).
$$

If several eligible events share publication date $d$, their atomic multiplier
is

$$
L_d(s)=\prod_{e:d_e=d}L_e(s).
$$

The posterior is normalized once:

$$
q_d^+(s)
=\frac{q_d^-(s)L_d(s)}{\sum_{s'}q_d^-(s')L_d(s')}.
$$

Computing $\log q+\sum_e\log L_e$ and using log-sum-exp prevents numerical
underflow. The atomic same-day rule removes arbitrary ordering among releases
when the data contain dates but no reliable intraday timestamps. It does not
assert that different release blocks are conditionally independent; the
product likelihood is a baseline approximation whose overconfidence must be
examined through calibration, entropy, and scale sensitivity.

An event outside the current four-month path is recorded and skipped. If its
reference month has already been hard-confirmed, its likelihood is constant
over all surviving paths and cannot change the normalized posterior. It is
likewise recorded as an already-confirmed no-op rather than presented as new
forecast information.

JOLTS is published with a comparatively long lag and will often arrive after
Model 01's deterministic label for the same reference month has been
confirmed. Such rows remain valuable for likelihood estimation once their
labels are causally available, but many will not create live posterior updates.
The audit and evaluation must expose this rather than counting every JOLTS row
as a timely nowcast.

## 8. Hard deterministic confirmations

On date $T_q$, a deterministic regime label for reference month $q$ becomes
available. Its confirmation multiplier is

$$
C_q(s)=\mathbf 1[r_q(s)=R_q^{\mathrm{observed}}].
$$

All confirmations sharing the date are applied atomically:

$$
q^+(s)
=\frac{q^-(s)\prod_q C_q(s)}
{\sum_{s'}q^-(s')\prod_q C_q(s')}.
$$

The denominator must be positive; otherwise the build fails rather than
inventing fallback mass. Missing deterministic labels are never clamped.

Because the source tables usually provide only dates, each calendar day uses
this conservative fixed order:

1. perform a month roll, if applicable;
2. apply all usable leading releases as one atomic group;
3. apply all deterministic confirmations as one atomic end-of-day group.

This preserves a pre-confirmation snapshot when leading evidence and a target
confirmation share a date. It is a documented date-level convention, not a
claim about the releases' actual intraday ordering.

## 9. Posterior checkpoints

The filter saves both all 256 path probabilities and the marginal distribution
for every month in the path at these checkpoint types:

| Checkpoint | Meaning |
|---|---|
| `initial` | Normalized path immediately after causal initialization |
| `pre_month_roll` | Last path before dropping the oldest reference month |
| `post_month_roll` | Transition-only path immediately after the shift |
| `pre_release_group` | Path before the date's atomic leading evidence |
| `post_release_group` | Path after the date's leading evidence |
| `pre_confirmation_group` | Path immediately before hard confirmation |
| `post_confirmation_group` | Path after hard confirmation |
| `month_end` | State after all dated operations through calendar month-end |
| `latest` | Final state at the build's information cutoff |

Each checkpoint receives a stable ID and records its information date,
checkpoint type, anchored calendar month, all four path months, joint
probability sum and normalization error, event or confirmation group ID where
applicable, and whether evidence is enabled. Forecast-scoring eligibility is
recorded separately in `forecast_predictions.csv`. `latest` may duplicate the
probabilities of the last
operational checkpoint; it exists as an explicit publication handoff rather
than a new update.

Saving the full joint distribution is essential. Four separate monthly
marginals cannot reconstruct correlations between adjacent states and are not
sufficient to update a delayed month and propagate its consequences.

## 10. Walk-forward evaluation sample

Formal evaluation begins with the January 2018 reference month. Earlier events
are training and filter warm-up history. A forecast row is eligible only when:

- the target reference month is January 2018 or later;
- the target eventually has a valid deterministic first-release regime;
- the checkpoint occurs strictly before that target's confirmation;
- the probability vector is finite, nonnegative, and normalized.

Post-confirmation marginals are retained for audit and path propagation but are
never scored as forecasts. Otherwise a hard clamp would produce mechanically
perfect accuracy. Events for an already confirmed reference month are also not
given a forecast score.

Metrics are reported separately by meaningful checkpoint slice, including
month roll, month-end, last pre-confirmation forecast, and eligible post-release
groups. Heterogeneous weekly and monthly event snapshots must not be pooled
into one flattering headline average. Every slice reports its number of target
months, forecast rows, date range, and class support.

The comparison baseline replays the same expanding monthly transitions, path
shifts, initial conditions, and hard confirmations but suppresses every leading
release likelihood. Thus any measured skill is attributable to the release
layer rather than a different transition or confirmation history.

## 11. Evaluation metrics

Let $N$ be the number of eligible forecasts in a reported slice,
$p_{i,k}$ the predicted probability of regime $k$, and $y_{i,k}$ its one-hot
realization.

### Negative log likelihood

$$
\mathrm{NLL}
=-\frac{1}{N}\sum_{i=1}^N\log p_{i,y_i}.
$$

NLL is a strictly proper probability score: it rewards probability assigned to
the realized state and strongly penalizes confident errors. Lower is better.
Probabilities are not silently clipped. Assigning exact zero probability to the
realized regime therefore produces infinite NLL, which is serialized as a
missing finite summary value while the raw forecast remains available.

### Unscaled multiclass Brier score

$$
\mathrm{Brier}
=\frac{1}{N}\sum_{i=1}^N\sum_{k=1}^4(p_{i,k}-y_{i,k})^2.
$$

This unscaled convention ranges from 0 for a perfect forecast to 2 for a
completely confident forecast of the wrong class. It measures squared
probability error and is less dominated by a single confident mistake than
NLL. Lower is better.

### MAP accuracy

The maximum-a-posteriori forecast is $\widehat y_i=\arg\max_k p_{i,k}$.
Accuracy is the fraction with $\widehat y_i=y_i$. Ties use the frozen state
order. Accuracy is familiar but discards the rest of the probability vector,
so it is secondary to NLL and Brier score.

### Balanced accuracy / macro recall

For class $k$, recall is

$$
\mathrm{Recall}_k
=\frac{\mathrm{true\ positives}_k}{\mathrm{actual\ observations}_k}.
$$

Balanced accuracy is the unweighted mean of recalls for classes with positive
support in the reported slice. It prevents a common persistent regime from
dominating ordinary accuracy. The scored forecast table retains every realized
and predicted class, so per-class supports and recalls remain reconstructible
when a short slice omits a rare regime.

### Macro F1

For each regime, F1 is the harmonic mean of one-versus-rest precision and
recall. Macro F1 is the equal-weight mean over the union of regimes realized or
predicted in the reported slice. A regime absent from both truth and prediction
is not fabricated as a zero-score class. It evaluates hard classifications
while giving active rare regimes equal weight, but, like accuracy, ignores
probability calibration.

### Top-label and classwise ECE and reliability

Each regime is treated as a one-versus-rest probability forecast. Probabilities
are assigned to ten fixed equal-width bins. For class $k$,

$$
\mathrm{ECE}_k
=\sum_{j=1}^{10}\frac{n_j}{N}
\left|\operatorname{freq}_{j,k}-\overline p_{j,k}\right|,
$$

where $\operatorname{freq}_{j,k}$ is the realized class frequency and
$\overline p_{j,k}$ the mean forecast in bin $j$. Lower is better. The full
reliability table stores bin edges, count, mean probability, observed
frequency, and gap; ECE alone can conceal offsetting or sparse-bin behavior.
The output also reports top-label confidence ECE, which compares the selected
regime's probability with whether that selected regime was correct.

### Growth- and inflation-axis Brier scores

The four probabilities are collapsed to the two economic axes:

$$
p_i^{G+}
=p_{i,\text{growth up, inflation up}}
+p_{i,\text{growth up, inflation down}},
$$

$$
p_i^{I+}
=p_{i,\text{growth up, inflation up}}
+p_{i,\text{growth down, inflation up}}.
$$

Each axis uses the binary squared error

$$
\mathrm{Brier}_{axis}
=\frac{1}{N}\sum_i(p_i^{axis+}-y_i^{axis+})^2,
$$

which lies in $[0,1]$. These scores distinguish failure to infer the growth
direction from failure to infer the inflation direction.

### Posterior entropy

$$
H_i=-\sum_{k=1}^4p_{i,k}\log p_{i,k}.
$$

Entropy lies between 0 and $\log4$. Lower entropy means a sharper forecast, not
necessarily a better one. It is reported with calibration and proper scores so
overconfidence is not mistaken for information.

### Comparison with transition-only

For the Brier score, release-model skill is

$$
\mathrm{Skill}_S=1-\frac{S_{release}}{S_{transition\ only}}.
$$

Positive skill indicates improvement, zero indicates no change, and negative
skill indicates deterioration relative to the matched transition-only replay.
For NLL, the output retains the direct model-minus-baseline difference rather
than a ratio. Both underlying proper scores are retained alongside these
comparisons so the scale remains visible.

## 12. Frozen sensitivity checks

Sensitivity is one parameter at a time. Every non-varied setting returns to the
baseline, and every variant is evaluated on exactly the same eligible forecast
rows as the baseline. The out-of-sample period is not searched for a favorable
combination.

### Tail thickness

$$
\nu\in\{3,5,7,10,30,\mathrm{Gaussian}\}.
$$

For every finite $\nu$, the shape conversion
$\Psi=(\nu-2)C/\nu$ preserves the fitted covariance. The Gaussian variant uses
$\mathcal N(\widetilde\mu,C)$ directly. This checks whether results depend on
extreme-event robustness rather than ordinary location differences.

### Mean shrinkage

$$
\kappa\in\{0,2,5,10,20\}.
$$

$\kappa=0$ is an unshrunk diagnostic. It cannot supply a mean for a regime with
no training observations, so affected fits are skipped rather than silently
assigned the global mean. Larger values increasingly pool regime
locations toward the block-wide mean.

### Covariance shrinkage

The alternatives are:

1. unshrunk empirical residual covariance;
2. data-driven Ledoit-Wolf shrinkage;
3. fixed spherical shrinkage

$$
C_{\lambda}=(1-\lambda)S^{\mathrm{emp}}
+\lambda\frac{\operatorname{tr}(S^{\mathrm{emp}})}{p_b}I,
\qquad \lambda\in\{0.25,0.50,0.75\}.
$$

These checks reveal whether the likelihood gains apparent confidence from a
fragile correlation estimate. All alternatives are identical in one dimension,
so `ICSA` should be invariant apart from numerical tolerance.

### Scale

The optional standard-deviation multipliers are

$$
s\in\{0.75,1.00,1.25\},
\qquad \Psi_s=s^2\Psi.
$$

This deliberately sharpens or flattens every block likelihood without changing
its mean or correlation. It is a direct check for posterior overconfidence
caused by multiplying conditionally dependent evidence.

Every sensitivity specification records a stable ID, the varied parameter,
baseline and alternate values, and all fixed settings. Its metric rows record
the matched eligible forecast count and complete evaluation set. Baseline-equivalent
settings such as $\nu=7$, $\kappa=5$, Ledoit-Wolf covariance, and $s=1$ are
represented once by the single baseline specification rather than duplicated
under every parameter family.

## 13. Output and audit contract

Local reproducible tables are written below
`data/processed/m01_bayesian_filter/`:

- `checkpoint_index.csv`: one row per saved operational checkpoint;
- `joint_path_checkpoints.csv.gz`: all 256 path probabilities per checkpoint;
- `marginal_checkpoints.csv`: four-state marginals for every path month plus a
  one-month transition forecast;
- `event_update_audit.csv`: applied, skipped, and no-op release events;
- `likelihood_fit_audit.csv`: causal sample sizes, means, covariance, shrinkage,
  numerical diagnostics, and training cutoffs;
- `forecast_predictions.csv`: pre-confirmation probability forecasts joined to
  eventual deterministic outcomes;
- `evaluation_metrics.csv`: metric values by model, checkpoint slice, block or
  horizon where applicable;
- `calibration_bins.csv`: classwise and top-label reliability-bin inputs to ECE;
- `sensitivity_specifications.csv`: complete frozen variant definitions;
- `sensitivity_metrics.csv`: sensitivity results on matched forecast rows.

The tracked manifest is
`data/manifests/m01_event_driven_bayesian_filter.json`. It records configuration
and input hashes, causal date coverage, primary artifact row counts, output
hashes, and the exact baseline parameterization. Detailed event outcomes,
likelihood-fit failures, and post-confirmation no-ops remain in the hashed audit
tables. The manifest contains no API credentials or raw authenticated URLs.

Small publication-safe artifacts are written below
`results/published/m01_bayesian_filter/`:

- `latest_posterior.json`, containing the latest four-month marginals and full
  interpretation metadata;
- `evaluation_summary.json`, containing the predefined walk-forward metric
  slices and transition-only comparisons;
- `sensitivity_metrics.csv`, containing the frozen sensitivity summary.

The public latest posterior must state its information cutoff, checkpoint type,
four reference months, state order, confirmation status for each month, and
probability sums. It is a model estimate, not a confirmed label or a trading
recommendation.

## 14. Known limitations

- Release features and deterministic labels share the same broad economy and
  are not conditionally independent given a four-state quadrant.
- Multiplying separate block likelihoods can still produce overconfident
  posteriors. Calibration and scale sensitivity diagnose but do not fully solve
  this structural limitation.
- Causal standardized features inherit expanding means and standard deviations
  affected by COVID and other structural breaks.
- A single shared block covariance stabilizes estimation but rules out
  regime-dependent volatility and correlation.
- Fixed $\nu=7$ and $\kappa=5$ are modeling choices, not facts learned from the
  evaluation period.
- Weekly claims create more updates than monthly blocks and may dominate the
  posterior even when each individual likelihood appears reasonable.
- The complete-vector rule discards partially observed monthly events and some
  historically split releases.
- Date-only ordering cannot reproduce actual intraday information availability.
- JOLTS frequently describes a month whose deterministic regime is already
  known, limiting its live contribution despite its usefulness in estimation.
- Hard confirmations treat the deterministic composite label as error-free;
  they do not integrate uncertainty about source revisions or the regime rule.
- Transition, likelihood, and mean/covariance parameter uncertainty are not
  integrated through the production path posterior.
- The January 2018 evaluation start yields a modest number of monthly targets,
  especially for rare regimes. Metric uncertainty and class support must
  accompany point estimates.

These constraints are deliberate boundaries for Model 01. A continuous latent
growth/inflation model, joint mixed-frequency observation equation, dynamic
covariance, tempered likelihood, or explicit release-time model belongs in a
later architecture rather than an undocumented switch inside this baseline.
