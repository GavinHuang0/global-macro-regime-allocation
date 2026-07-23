# Archived Model 02 evidence-block experiments

This document records six point-in-time evidence changes tested around the
selected `student_t_7_combined` inference model. The experiment does **not**
change the selected Model 02 baseline. Every candidate retains:

- the unsmoothed Model 02 growth and inflation scores;
- the four-month joint Gaussian state;
- OLS VAR(1) transition dynamics;
- partial score-defining first releases;
- fixed-$\nu=7$ Student-$t$ observation models;
- the causal annual ridge schedule; and
- the same mapping and evaluation targets.

Only the declared non-defining evidence set changes. The replay runs through
20 July 2026. Its selected-baseline evaluation rows, semantic event updates,
and latest marginals reproduce the frozen `student_t_7_combined` artifacts at
all 64 invariant checks.

## 1. Six tested priorities

| Priority | Action | Candidate responses | Point-in-time transformation |
|---|---|---|---|
| 1. Manufacturing surveys | Add | Empire State current new orders and prices paid; Philadelphia current new orders and prices paid | Contemporaneous diffusion-index level, causally standardized after 24 earlier observations |
| 2. Continued claims | Add beside initial claims | Continued-claims innovation (`CCSA`) | Expanding log-AR(1) innovation inherited from the verified Model 01 event artifact |
| 3. Cleaner consumer indicators | Replace the broad nominal consumer block | Nominal retail sales excluding autos and gasoline; total vehicle unit sales | Same-vintage monthly log differences, causally standardized after 24 earlier observations |
| 4. Single-family housing | Replace aggregate housing block | Single-family starts and permits; new single-family home sales | Same-vintage monthly log differences, with mortgage-rate level, change, and methodology controls |
| 5. Import-price pressure | Add | Import prices excluding food, fuels, and computers | Same-vintage 12-month log change, used because the index is not seasonally adjusted |
| 6. Capital-goods backlog | Replace shipments | Core capital-goods orders; unfilled-orders-to-shipments ratio | Existing same-vintage orders log change and $100\log(\text{unfilled orders}/\text{shipments})$ from an exact common vintage |

The survey observations describe the current calendar month and can therefore
have a negative month-end-relative release lag without look-ahead. Archived
history is not treated as if it had been observed in real time. The first
usable standardized survey rows are April 2016 for Empire State and June 2017
for Philadelphia. Direct retail-ex-autos-and-gas evidence first becomes usable
in June 2020. These short histories are a feature of the real-time data, not
filled gaps.

The combined candidate uses all six changes simultaneously. The survey-only
experiment also reports Empire and Philadelphia separately to diagnose whether
their information is redundant.

## 2. Unchanged emission model

For observation model $b$, release event $e$, and its reference month $q(e)$,

$$
\boldsymbol y_e
=
\boldsymbol a_b
+\boldsymbol H_b\boldsymbol Z_{q(e)}
+\boldsymbol C_b\boldsymbol v_e
+\boldsymbol\epsilon_e,
$$

where

$$
\boldsymbol\epsilon_e
\sim
t_{\nu=7}(\boldsymbol 0,\boldsymbol R_b).
$$

The response-specific state loadings retain their declared ridge penalties and
exact-zero restrictions. Each fit uses only rows satisfying

$$
T_i^{\mathrm{train}}<d_e,
$$

where $T_i^{\mathrm{train}}$ is the latest availability date of the response,
target score, and controls, and $d_e$ is the candidate event's release date.
The residual scale uses Ledoit--Wolf shrinkage. The release-time Student-$t$
weight is evaluated against the shared pre-release-day state, so same-day
factor order cannot change its value.

The following numbers of candidate updates were applied in the combined
replay after their model-specific warm-ups:

| Candidate observation model | Applied updates |
|---|---:|
| Empire State survey | 99 |
| Philadelphia survey | 85 |
| Continued claims | 491 |
| Retail excluding autos and gasoline | 50 |
| Vehicle unit sales | 114 |
| Single-family construction | 130 |
| New-home sales | 122 |
| Import prices | 147 |
| Capital-goods backlog | 130 |

## 3. Evaluation design

The primary checkpoint is `before_any_defining_release`. It evaluates the
forecast before any component that defines that month's completed composite
has arrived. After the four-month initialization burn-in, 183 paired target
months run from January 2011 through May 2026.

For each candidate, the experiment reports the paired monthly difference

$$
\Delta L_m=L_m^{\mathrm{candidate}}-L_m^{\mathrm{baseline}}.
$$

Negative differences are favorable for score negative log predictive density
(NLPD), quadrant cross-entropy, and quadrant Brier distance. Positive
differences are favorable for hard-quadrant correctness.

Uncertainty is estimated with 5,000 deterministic circular moving-block
bootstrap replications using 12-month blocks. The report includes a 95%
percentile interval, the bootstrap probability that the candidate is better,
and a centered two-sided bootstrap $p$-value. Holm adjustment controls the
family-wise error rate across the six isolated priorities for each metric,
checkpoint, and evaluation sample. The combined candidate is reported as an
exploratory joint test and is not part of that six-test Holm family.

## 4. Primary results

The table gives candidate-minus-baseline differences at the primary
checkpoint. NLPD, cross-entropy, and Brier are losses, so lower is better.
Accuracy is a fraction, so $0.0219$ means about 2.19 percentage points.

| Candidate | $\Delta$ NLPD | $\Delta$ cross-entropy | $\Delta$ Brier | $\Delta$ accuracy |
|---|---:|---:|---:|---:|
| Manufacturing surveys | 0.071872 | 0.047370 | 0.023917 | -0.027322 |
| Continued claims | **-0.017000** | **-0.002526** | **-0.000934** | **0.021858** |
| Consumer quantities | -0.008824 | 0.000114 | 0.000030 | 0.010929 |
| Single-family housing | 0.000004 | -0.000003 | -0.000001 | 0.000000 |
| Import prices | -0.000410 | -0.000083 | -0.000047 | 0.005464 |
| Capital-goods backlog | 0.000030 | 0.000013 | 0.000006 | 0.000000 |
| All six together | 0.062669 | 0.051600 | 0.024278 | -0.010929 |

The strongest isolated mean result is continued claims, but its 95% intervals
still cross zero: NLPD $[-0.07213,0.04282]$, cross-entropy
$[-0.00661,0.00044]$, Brier $[-0.00278,0.00058]$, and accuracy
$[-0.00546,0.05464]$. Its bootstrap probabilities of improvement are 0.881,
0.934, 0.861, and 0.913, respectively. None of the six priorities is
significant after Holm correction at the primary checkpoint.

The consumer replacement has a favorable NLPD interval
$[-0.02427,-0.00033]$, but its centered two-sided $p$-value is 0.078 and its
Holm-adjusted value is 0.468. It is also metric-dependent: quadrant
cross-entropy and Brier are slightly worse. After the employment release, the
consumer candidate's mean NLPD is much worse because its April 2020 loss
increases by approximately 891.8 relative to the baseline. It should not be
promoted from the primary-checkpoint result alone.

The survey pair is worse than either regional survey alone and materially
worsens both quadrant losses. The all-six candidate likewise has strictly
positive 95% intervals for its cross-entropy and Brier deterioration. Housing,
import prices, and the backlog ratio have changes too small to constitute a
practical improvement under the present likelihood.

## 5. Conditional-residual dependence

The combined candidate still multiplies observation-model likelihood factors
as a conditional-independence approximation. To diagnose that approximation,
the experiment computes residuals using a fit selected strictly before each
release and then, **only retrospectively**, conditions the emission mean on the
subsequently completed first-release score:

$$
\boldsymbol e_{b,e}
=
\boldsymbol y_e
-\widehat{\boldsymbol a}_{b,e^-}
-\widehat{\boldsymbol H}_{b,e^-}\boldsymbol Z_{q(e)}^{\mathrm{completed}}
-\widehat{\boldsymbol C}_{b,e^-}\boldsymbol v_e.
$$

Joint response vectors are Cholesky-whitened by their causal Student-$t$ scale.
The completed score is never used to fit the event model, update the live
posterior, select a parameter, or alter the reported forecast. This is a
retrospective factorization diagnostic only.

The diagnostic contains 2,667 residual rows. Of 1,080 cross-model tests, 966
have sufficient data and 140 reject after Benjamini--Hochberg adjustment at
5%. Of 60 serial tests, 57 are valid and 35 reject after adjustment. Important
contemporaneous Pearson correlations include:

| Pair | Correlation | Common months | BH $q$-value |
|---|---:|---:|---:|
| Empire vs Philadelphia prices paid | 0.761 | 81 | $2.12\times10^{-14}$ |
| Empire vs Philadelphia new orders | 0.596 | 81 | $1.81\times10^{-7}$ |
| Retail excluding autos/gas vs vehicle units | 0.580 | 47 | 0.000308 |
| Import prices vs intermediate input costs | 0.453 | 37 | 0.0358 |
| New-home sales vs single-family permits | 0.064 | 128 | 0.755 |
| New-home sales vs single-family starts | 0.054 | 128 | 0.786 |

Initial and continued claims require a separate exact-publication-date check
because their reference weeks differ. Across 472 common publication dates,
their Pearson correlation is 0.102 ($q=0.0272$) and Spearman correlation is
0.118 ($q=0.0207$). These are small effect sizes despite statistical rejection.
The much larger month-aggregated correlation is contaminated by strong serial
dependence and must not be read as the same-day relationship.

These results show redundancy consistent with the combined candidate's poor
performance, especially in the two regional surveys and overlapping price
signals. They do not isolate redundancy as the causal source of that
deterioration. Failure to reject a pair is not proof of conditional
independence; short histories and serial dependence limit the tests. Future
work should whiten correlated block innovations, jointly model redundant
releases, temper their likelihoods, or select one representative from a
correlated family.

## 6. Decision

No candidate is promoted. `student_t_7_combined` remains the selected Model 02
inference baseline, with `transition_only` and `partial_only` as the two major
benchmarks.

The next focused experiment should retain continued claims as the most
promising addition but model its overlap with initial claims explicitly. The
consumer-quantity replacement merits a crisis-robust follow-up. The two
regional surveys should not be multiplied as independent blocks in their
current form.

## 7. Reproduction and artifacts

After setting `FRED_API_KEY` in the runtime environment:

```powershell
build-m02-evidence-experiments
build-m02-evidence-experiment-inference
```

The acquisition stage is the only networked stage. It never writes the key.
The inference stage verifies every upstream hash and performs no network or
credential access.

The principal contracts and outputs are:

- `configs/models/m02_evidence_block_experiments.yaml`;
- `configs/models/m02_evidence_block_inference.yaml`;
- `data/manifests/m02_evidence_block_experiments.json`;
- `data/manifests/m02_evidence_block_inference.json`;
- `results/published/m02_soft_composite/evidence_block_inference/evaluation_summary.csv`;
- `results/published/m02_soft_composite/evidence_block_inference/paired_block_bootstrap.csv`;
- `results/published/m02_soft_composite/evidence_block_inference/dependence_pairwise.csv`;
- `results/published/m02_soft_composite/evidence_block_inference/dependence_serial.csv`;
- `results/published/m02_soft_composite/evidence_block_inference/dependence_same_publication.csv`;
- `results/published/m02_soft_composite/evidence_block_inference/dependence_summary.json`; and
- `results/published/m02_soft_composite/evidence_block_inference/baseline_invariance.csv`.
