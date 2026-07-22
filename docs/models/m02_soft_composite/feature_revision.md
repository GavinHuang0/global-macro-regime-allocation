# Model 02 reduced-core feature revision

This document records the locked Model 02 experiment that removes two weak
legacy observation models, tests import prices as a direct replacement for
input costs, redesigns the consumer and business-investment evidence, and
models initial and continued unemployment claims through a dependence-aware
factorization. It is a development-stage feature experiment around the frozen
`student_t_7_combined` baseline; it does not retroactively redefine that
baseline.

The experiment is entirely offline. It reuses verified point-in-time event
artifacts and does not read an API key or any other credential. Removing an
observation model means removing it from a candidate model's likelihood
allowlist. No source observation, cached vintage, processed event row, or
historical result is deleted.

## 1. Fixed inference architecture

Let

$$
\boldsymbol Z_m=
\begin{bmatrix}
G_m\\
I_m
\end{bmatrix}
$$

denote the growth and inflation composite scores for reference month $m$.
The filter maintains the four-month joint state

$$
\boldsymbol X_m=
\begin{bmatrix}
\boldsymbol Z_{m-3}^{\mathsf T} &
\boldsymbol Z_{m-2}^{\mathsf T} &
\boldsymbol Z_{m-1}^{\mathsf T} &
\boldsymbol Z_m^{\mathsf T}
\end{bmatrix}^{\mathsf T}.
$$

All non-control variants retain the selected baseline's:

- unsmoothed score definition and quadrant-probability map;
- OLS VAR(1) transition model;
- partial first-release updates for score-defining components;
- expanding, strictly causal emission fits;
- fixed-$\nu=7$ Student-$t$ release likelihoods;
- robust release-time update and event-weight audit; and
- four fixed evaluation checkpoints.

For observation model $b$, event $e$, and event reference month $q(e)$, the
general release equation is

$$
\boldsymbol y_e
=
\boldsymbol a_b
+\boldsymbol H_b\boldsymbol Z_{q(e)}
+\boldsymbol C_b\boldsymbol v_e
+\boldsymbol\varepsilon_e,
\qquad
\boldsymbol\varepsilon_e
\sim
t_{\nu=7}(\boldsymbol 0,\boldsymbol R_b),
$$

where $\boldsymbol y_e$ is the response vector, $\boldsymbol v_e$ contains
observed controls, $\boldsymbol H_b$ contains state loadings, and
$\boldsymbol R_b$ is one residual scale matrix for the observation model.
Loading-specific ridge penalties and exact-zero restrictions are described
below. A model fit used for event $e$ includes only training rows whose entire
information set was available strictly before the event's publication date.
Rows sharing the candidate publication timestamp are held out atomically.

## 2. Reduced-core exclusions

Two exclusions are fixed before evaluating the new features:

1. `monthly_labor_demand`, the legacy JOLTS likelihood for openings, hires,
   quits, and layoffs; and
2. `housing_activity`, the legacy aggregate starts-and-permits likelihood.

These models were negative or negligible in the preceding add-one and
leave-one-out attribution and suffered important timing or dependence
limitations. Every new, non-control variant excludes both. The three frozen
controls&mdash;`transition_only`, `partial_only`, and
`student_t_7_combined`&mdash;are replayed with their historical specifications so
that their published artifacts can be checked for invariance. Consequently,
the frozen selected baseline still contains JOLTS and aggregate housing; that
fact does not re-enable them in any reduced-core candidate.

This is an observation-model decision, not a data-destruction operation. The
JOLTS and housing series, release events, manifests, and earlier experiment
results remain in the repository for audit and possible later redesign.

## 3. Import prices versus legacy input costs

The replacement price coordinate uses the import-price index excluding food,
fuels, and computers (`IREXPETCOM`). Because the series is not seasonally
adjusted, its release-time transformation is the same-vintage 12-month log
change

$$
p^{\mathrm{imp}}_e
=
100\log\!\left(
\frac{P^{(v_e)}_m}{P^{(v_e)}_{m-12}}
\right),
$$

where $v_e$ is the vintage available at release $e$. The current and lagged
levels therefore come from the same vintage. The transformed value is
standardized using only earlier publication dates. After a 24-observation
warm-up, it enters an inflation-only emission:

$$
y^{\mathrm{imp}}_e
=a_{\mathrm{imp}}
+h_{\mathrm{imp},I} I_{q(e)}
+\varepsilon_e,
\qquad
h_{\mathrm{imp},G}=0.
$$

The experiment separates three questions:

- **direct replacement:** import prices versus the reduced core retaining
  legacy intermediate-materials input costs;
- **add one:** import prices versus the same reduced core with no price block;
  and
- **legacy add one:** input costs versus that no-price reduced core.

Import prices replace input costs only if the direct swap improves both mean
quadrant cross-entropy and mean Brier distance without material damage at later
checkpoints. If that rule fails, the predeclared fallback is no price block,
not a return to the already weak input-cost likelihood.

## 4. Consumer-demand redesign

### 4.1 Real demand and implicit retail prices

The first consumer event model decomposes nominal retail and food-services
sales (`RSAFS`) into real retail activity (`RRSFS`) and an implicit price
ratio. At every information date, each series contributes only its most recent
vintage dated no later than that date. This matters because the two archives
do not always update on the same historical day.

Let $v_e^N$ and $v_e^R$ be the latest nominal and real archive vintages dated
no later than information date $e$. They need not be the same calendar date.
Define the causally aligned implicit price level

$$
Q_m^{(e)}=\frac{\operatorname{RSAFS}_m^{(v_e^N)}}
                   {\operatorname{RRSFS}_m^{(v_e^R)}}.
$$

The two monthly response coordinates are

$$
r^{\mathrm{real}}_e
=100\log\!\left(
\frac{\operatorname{RRSFS}^{(v_e^R)}_m}
     {\operatorname{RRSFS}^{(v_e^R)}_{m-1}}
\right)
$$

and

$$
p^{\mathrm{retail}}_e
=100\log\!\left(
\frac{Q_m^{(e)}}{Q_{m-1}^{(e)}}
\right).
$$

Within each constituent series, the current and prior levels come from the
same available snapshot. The identity

$$
r^{\mathrm{real}}_e+p^{\mathrm{retail}}_e
=100\log\!\left(
\frac{\operatorname{RSAFS}^{(v_e^N)}_m}
     {\operatorname{RSAFS}^{(v_e^N)}_{m-1}}
\right)
$$

holds for causally aligned snapshots. Each coordinate receives its own
strictly lagged expanding standardization after 60 earlier observations. They
are then estimated jointly with one residual scale matrix. The real-activity
response uses state-loading ridge penalties $(1,10)$ for growth and inflation,
respectively; the implicit-price response uses $(10,1)$. Both loadings remain
estimable, but the economically secondary cross-loading is shrunk more
strongly.

### 4.2 Vehicle quantities

Total vehicle unit sales (`TOTALSA`) supply a separate quantity coordinate:

$$
u^{\mathrm{veh}}_e
=100\log\!\left(
\frac{\operatorname{TOTALSA}^{(v_e)}_m}
     {\operatorname{TOTALSA}^{(v_e)}_{m-1}}
\right).
$$

It is standardized against publication dates strictly before $e$ after a
24-observation warm-up. It loads on growth only; its inflation loading is fixed
exactly to zero.

### 4.3 Consumer attribution

The experiment tests real activity, the implicit price coordinate, and vehicle
units separately as add-one responses against a reduced core with no consumer
model. It also removes each response from the full redesigned consumer profile
and compares the complete redesign with the legacy nominal consumer model.

The leave-one-out contrasts are conditional ablations, not mechanical
coefficient deletions. In particular, removing real activity or implicit
prices changes the dimension of the joint response and refits its residual
scale. The resulting effect includes both the omitted response's information
and the change in the remaining response's fitted likelihood. Add-one and
leave-one-out effects therefore need not agree or add up.

## 5. Business-investment redesign

The legacy model used separate monthly log changes in core capital-goods
orders (`NEWORDER`) and shipments (`ANXAVS`). The redesign rotates the same
point-in-time sources into current activity and a forward-pipeline coordinate.
Orders and shipments must have the same exact release date and reference
month.

The activity response is the existing same-vintage shipment change

$$
a^{\mathrm{cap}}_e
=100\log\!\left(
\frac{S_m^{(v_e)}}{S_{m-1}^{(v_e)}}
\right),
$$

and the pipeline response is

$$
d^{\mathrm{cap}}_e
=100\Delta\log\!\left(\frac{O_m^{(v_e)}}{S_m^{(v_e)}}\right)
=100\log\!\left(
\frac{O_m^{(v_e)}/S_m^{(v_e)}}
     {O_{m-1}^{(v_e)}/S_{m-1}^{(v_e)}}
\right),
$$

where $O$ denotes orders and $S$ denotes shipments. Equivalently,
$d^{\mathrm{cap}}_e$ is the orders log change minus the shipments log change
from the exact matched event.

Activity retains its previously verified causal standardization. The new
pipeline coordinate is standardized as

$$
z^{\mathrm{pipe}}_e
=\frac{d^{\mathrm{cap}}_e-\bar d_{e^-}}{s_{e^-}},
$$

where $\bar d_{e^-}$ and $s_{e^-}$ use only available pipeline observations
with publication date strictly earlier than $e$. All catch-up rows sharing a
publication date are standardized atomically, so no same-day value enters
another row's mean or standard deviation. The warm-up requires 24 earlier
values.

The joint activity/pipeline model uses one residual scale matrix. Activity is
allowed to load on both scores with penalties $(1,10)$, while the pipeline
loads on growth only with its inflation loading fixed exactly to zero. Atomic
add-one and conditional leave-one-out contrasts are reported for both
coordinates, followed by a direct comparison between the joint redesign and
the legacy orders-and-shipments model. As with the consumer joint response,
the conditional ablation also changes and refits the residual covariance; it
is not a pure partial-regression coefficient test.

## 6. Joint initial- and continued-claims evidence

Initial claims (`ICSA`) and continued claims (`CCSA`) are published together
but normally describe different reference weeks. A naive bivariate response
assigned to one reference month would therefore misdate one coordinate. The
implemented chain-rule factorization is

$$
p\!\left(u^I_e\mid\boldsymbol Z_{q_I(e)}\right)
\;p\!\left(
u^C_e\mid\boldsymbol Z_{q_C(e)},u^I_e
\right),
$$

where $u^I_e$ and $u^C_e$ are the existing expanding log-AR(1) innovations,
$q_I(e)$ is the ICSA reference month, and $q_C(e)$ is the CCSA reference
month. The first factor is the existing ICSA likelihood. In the second factor,
CCSA remains targeted to its own reference month and the same-publication ICSA
innovation is an observed control:

$$
u^C_e
=a_C+h_{C,G}G_{q_C(e)}+c_Iu^I_e+\varepsilon^C_e,
\qquad
h_{C,I}=0.
$$

Pairing is permitted only when the two rows have the exact same publication
date and the ICSA reference week is exactly seven days after the CCSA reference
week. The builder does not use a nearest-date match, a future row, or a
calendar-month retargeting rule. Of 675 CCSA rows, 672 have exact pairs and
three remain explicitly unmatched. Of 597 available CCSA rows, 595 have an
available exact pair. There are 137 available pairs whose two reference weeks
fall in different calendar months, demonstrating why their targets must remain
separate. Two publication dates contain multiple exact pairs; the exact
reference-week key resolves those cases.

The study compares ICSA only, conditional CCSA only, their conditional product,
and a comparator that treats ICSA and CCSA as independent likelihood factors.
The conditional construction removes their same-publication linear overlap;
it does not eliminate serial dependence created by using weekly observations
against monthly states. Weekly evidence can still accumulate more rapidly than
monthly evidence, so this remains a limitation rather than a complete
dependence model.

## 7. Causal comparison and evaluation design

The locked design declares 23 candidate-reference contrasts:

- three price-block comparisons;
- seven consumer comparisons;
- five business-investment comparisons;
- six claims comparisons; and
- two descriptive comparisons for the all-revised candidate.

The price, consumer, business, and claims families receive separate Holm
multiplicity adjustments for each metric and evaluation sample. The combined
candidate is descriptive and is not included in an atomic Holm family.

The primary checkpoint is `before_any_defining_release`, before any component
that defines that month's completed composite score has arrived. The other
checkpoints are after the Employment Situation release, after midmonth
defining releases, and immediately before the final score-defining release
day. For a candidate $c$, reference $r$, and lower-is-better loss $L$, the
reported mean benefit is

$$
B_{c,r}=\frac{1}{M}\sum_{m=1}^{M}
\left(L_{r,m}-L_{c,m}\right).
$$

Positive benefit therefore always means that the candidate helps. For hard
quadrant accuracy the subtraction is reversed so the same interpretation
holds. The principal loss is quadrant cross-entropy, corroborated by quadrant
Brier distance. Continuous-score negative log predictive density (NLPD) is a
secondary diagnostic, and hard accuracy is descriptive because it discards
forecast uncertainty.

More precisely, let $\widehat{\boldsymbol p}_m$ be the forecast distribution
over the four quadrants and let $\boldsymbol p_m^*$ be the soft quadrant map
computed from the completed score. The two proper probability losses are

$$
\operatorname{CE}_m
=-\sum_{r=1}^{4}p_{m,r}^*\log\widehat p_{m,r}
$$

and

$$
\operatorname{Brier}_m
=\sum_{r=1}^{4}
\left(\widehat p_{m,r}-p_{m,r}^*\right)^2.
$$

They score the full probability distribution rather than only its largest
entry.

Uncertainty uses 5,000 deterministic, segment-aware circular moving-block
bootstrap replications with 12-month blocks. The full sample and the
March&ndash;May 2020 exclusion sensitivity preserve calendar-contiguous segments;
the bootstrap never treats February and June 2020 as adjacent. Comparisons use
the common eligible calendar, not only months in which the added response
updates the filter. This measures the strategy-level average effect but can
dilute sparse-event effects with many exact zero monthly differences.

## 8. Results and decisions

The corrected replay contains 26 variants and all 23 declared comparisons.
All 216 semantic invariance checks pass for the three frozen controls. The
primary common calendar contains 183 months from January 2011 through May
2026. The main candidate models apply 147 import-price updates, 179 joint
real/implicit-retail updates, 114 vehicle-unit updates, 82 joint
activity/pipeline updates, 490 ICSA updates, and 489 conditional CCSA updates.

The following table reports the principal full-sample results at
`before_any_defining_release`. Every entry is a block benefit, so a positive
number means that the named feature or candidate lowers the loss. `Conditional`
denotes the corresponding leave-one-out effect within the complete redesigned
block.

| Comparison | Cross-entropy benefit | Brier benefit | NLPD benefit |
|---|---:|---:|---:|
| Import prices versus input costs | 0.000124 | 0.0000657 | 0.000540 |
| Import prices versus no price block | 0.0000678 | 0.0000395 | 0.000371 |
| Input costs versus no price block | -0.0000565 | -0.0000262 | -0.000169 |
| Real retail activity, add one | 0.0000309 | 0.0000165 | 0.000127 |
| Real retail activity, conditional | 0.0000596 | 0.0000263 | 0.000240 |
| Implicit retail price, add one | 0.0000966 | 0.0000445 | 0.000423 |
| Implicit retail price, conditional | 0.000137 | 0.0000569 | 0.000552 |
| Vehicle units, add one | -0.0000159 | 0.00000462 | 0.00910 |
| Vehicle units, conditional | -0.0000127 | -0.00000335 | 0.00912 |
| Full consumer redesign versus legacy consumer | 0.0000935 | 0.0000482 | 0.00963 |
| Capital-goods activity, add one | 0.00000261 | 0.00000119 | 0.00000615 |
| Capital-goods activity, conditional | 0.00000367 | 0.00000174 | 0.00000963 |
| Capital-goods pipeline, add one | 0.000000139 | 0.0000000708 | 0.000000290 |
| Capital-goods pipeline, conditional | 0.00000120 | 0.000000616 | 0.00000377 |
| Joint business redesign versus legacy business | 0.000000549 | 0.000000290 | 0.00000189 |
| ICSA alone versus no claims | 0.00102 | -0.0000646 | 0.0658 |
| Conditional CCSA alone versus no claims | 0.00135 | -0.0000940 | 0.00495 |
| Conditional ICSA--CCSA versus ICSA alone | 0.00215 | 0.000758 | 0.00487 |
| Independent ICSA--CCSA versus ICSA alone | 0.00253 | 0.000932 | 0.0170 |
| Conditional versus independent ICSA--CCSA | -0.000382 | -0.000174 | -0.0121 |

None of the atomic proper-score comparisons is significant after its
family-specific Holm adjustment. For example, the pipeline add-one comparison
has favorable percentile intervals for cross-entropy and Brier, but its raw
centered $p$-values are 0.085 and 0.082 and its Holm-adjusted values are 0.427
and 0.409. The signs below are therefore development guidance, not confirmed
out-of-sample effects.

### 8.1 Feature decisions

The predeclared decision rules imply the following reduced-core design for a
future locked replay:

- **JOLTS and aggregate housing remain excluded.** This is the fixed structural
  decision tested by every non-control candidate.
- **Use no price block for now.** Import prices improve both primary proper
  scores relative to input costs and to no price block, while legacy input
  costs are worse than no price block. However, the import-for-input direct
  swap has cross-entropy and Brier benefits of $-0.000477$ and $-0.000578$
  after the midmonth defining releases, and its strict-pre-final Brier benefit
  is $-0.000258$. This fails the predeclared requirement of no material harm at
  later checkpoints. Import prices remain the preferred price sensitivity,
  but are not selected for the reduced core.
- **Keep real retail activity and the implicit retail-price coordinate.** Both
  have positive cross-entropy and Brier benefits in their add-one and
  conditional ablations, satisfying the sign rule. Their joint economic
  decomposition also avoids treating a broad nominal-sales change as pure
  demand.
- **Omit vehicle units.** Vehicle sales fail the cross-entropy sign rule in
  both attribution arms and fail the Brier rule conditionally. Their favorable
  primary NLPD does not override the predeclared emphasis on quadrant proper
  scores.
- **Keep both capital-goods activity and pipeline coordinates.** Each has
  nonnegative cross-entropy and Brier effects in both attribution arms, and the
  joint redesign is slightly better than the legacy orders-and-shipments model
  on all three primary probabilistic losses. The magnitudes are extremely
  small and not statistically secure.
- **Retain ICSA only pending a better dependence model.** The conditional
  ICSA--CCSA product improves cross-entropy and Brier relative to ICSA alone,
  but is worse than the independent two-likelihood comparator on both. It
  therefore fails the exact predeclared promotion rule. The independent and
  conditional CCSA designs remain useful sensitivities; the independent
  version's better mean score does not justify ignoring its more questionable
  conditional-independence assumption.

These choices do not create a newly validated combined model. In particular,
the predeclared `all_revised_candidate` contains import prices, vehicle units,
and conditional CCSA, so it is not the same as the component-wise reduced core
just described. The component-wise selection would require a fresh locked
joint replay and, ultimately, new data for honest validation.

### 8.2 Combined development candidate

At the primary checkpoint, `all_revised_candidate` has the following benefits:

| Reference | Cross-entropy | Brier | NLPD | Hard accuracy |
|---|---:|---:|---:|---:|
| Frozen `student_t_7_combined` | 0.00267 | 0.00101 | 0.0155 | 2.73 percentage points |
| `partial_only` | 0.00369 | 0.000945 | 0.0813 | 3.83 percentage points |

Against the frozen selected baseline, the 95% moving-block intervals are
$[-0.000178,0.00662]$ for cross-entropy,
$[-0.000515,0.00287]$ for Brier, and $[-0.0105,0.0368]$ for NLPD. All include
zero. Against `partial_only`, the hard-accuracy interval is positive, but hard
accuracy is descriptive, the combined comparison is outside the atomic Holm
families, and its principal proper-score intervals include zero.

The candidate's later-checkpoint full-sample NLPD is also unstable. Its benefit
relative to the frozen baseline is $-4.83$ after the employment release and
$-3.01$ after midmonth defining releases, whereas the sensitivity excluding
March--May 2020 becomes favorable. This is another reason not to promote the
combined profile from its favorable primary means.

## 9. Interpretation limits

The following limitations apply even when a mean comparison is favorable:

- This experiment was motivated by earlier results from the same historical
  sample. It is development evidence, not an untouched validation set.
- The all-revised profile combines changes selected and interpreted on this
  history. Its same-sample performance cannot support an unbiased production
  claim or automatic baseline promotion.
- Add-one and leave-one-out comparisons answer different questions. Joint
  response covariance, sequential nonlinear updates, overlapping information,
  and ridge refits prevent their effects from adding linearly.
- Many event models have short causal histories or long warm-ups. A small
  full-calendar effect can reflect limited usable releases as well as weak
  economics.
- The Student-$t$ emission, state loadings, residual scale, and VAR parameters
  are plug-in estimates. The filter does not integrate parameter uncertainty.
- The block factorization is still an approximation. Conditional CCSA handles
  one important contemporaneous dependency, but consumer, price, labor, and
  business information can remain correlated across blocks and through time.
- First-release vintage reconstruction prevents known future revisions from
  entering earlier events, but archive starts and historical release coverage
  still limit how completely old real-time information sets can be recovered.
- Holm-adjusted significance and interval exclusion are required for strong
  support. Directionally favorable means with intervals crossing zero remain
  provisional.

The implementation contract is
[`configs/models/m02_feature_revision.yaml`](../../../configs/models/m02_feature_revision.yaml).
Published comparison and audit tables are under
[`results/published/m02_soft_composite/feature_revision`](../../../results/published/m02_soft_composite/feature_revision/).
