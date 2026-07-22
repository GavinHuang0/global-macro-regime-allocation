# Model 02 existing-block attribution

This document records the causal add-one and leave-one-out attribution of the
non-defining evidence already used by `student_t_7_combined`. The experiment
does not change the selected Model 02 baseline. It reuses the original
point-in-time evidence table and reads no network credential.

## 1. Questions and estimands

Let \(\mathcal E\) denote the seven observation models enabled by the selected
baseline and let \(b\in\mathcal E\) be one model. Two experiments answer
different questions:

$$
\mathcal E_b^{\mathrm{add}}=\{b\},
\qquad
\mathcal E_b^{\mathrm{leave}}=\mathcal E\setminus\{b\}.
$$

The add-one variant enables only \(b\) on top of `partial_only`. It estimates
the block's standalone value. The leave-one-out variant removes only \(b\)
from `student_t_7_combined`. It estimates the block's conditional value in the
presence of all the other legacy blocks.

For a lower-is-better loss \(L\), define

$$
A_b=L(\texttt{partial\_only})
    -L(\texttt{partial\_only}+b),
$$

and

$$
C_b=L(\texttt{baseline}-b)
    -L(\texttt{baseline}).
$$

For accuracy, the subtraction is reversed where necessary so that positive
\(A_b\) and positive \(C_b\) always mean that the block helps. Add-one and
leave-one-out effects need not add to the baseline's total change because the
Bayesian updates are nonlinear, sequential, and informationally dependent.

Every candidate retains:

- fixed-\(\nu=7\) Student-\(t\) emissions;
- OLS VAR(1) transition dynamics;
- partial first-release updates for score-defining components;
- the baseline ridge schedules and covariance estimators;
- the same four-month joint state, mapping uncertainty, and replay dates; and
- the baseline nominal-retail specification.

Only the non-defining observation-model allowlist changes.

## 2. Attributed blocks and live coverage

The baseline has seven asynchronous observation models. Inflation expectations
and input costs share an economic inflation-pressure theme but are attributed
separately because they have different publication histories and likelihoods.
They are also tested jointly as an exploratory pair.

| Block | Responses | Applied updates | Active reference months | Main limitation |
|---|---|---:|---:|---|
| Weekly labor stress | Initial-claims log-AR(1) innovation | 490 | 114 | Weekly frequency and serial dependence can overweight the signal |
| Monthly labor demand | Openings, hires, quits, and layoffs-rate changes | 14 | 14 | 82 releases arrive after the referenced score is already exact |
| Consumer demand | Ex-motor-vehicle retail growth and motor-vehicle-sales growth | 182 | 182 | Two nominal responses may mix quantity and price information |
| Housing activity | Aggregate starts and permits with mortgage controls | 129 | 129 | Broad aggregates and correlated responses |
| Business investment | Core capital-goods orders and shipments | 82 | 82 | 105 early events fail the causal warm-up |
| Inflation expectations | One-year expected-inflation change | 3 | 3 | Real-time history begins too late for meaningful attribution |
| Inflation input costs | Intermediate-materials price growth | 40 | 40 | Short history and overlap with other price information |

Near-zero effects for monthly labor demand or inflation expectations must not
be interpreted as evidence that JOLTS or expectations lack economic value.
They primarily expose a timing and history problem in the current event model.

## 3. Evaluation and inference

The primary checkpoint is `before_any_defining_release`. It evaluates the
regime nowcast before the current month's score-defining data enter. There are
183 common eligible months spanning January 2011 through May 2026.

The principal metric is quadrant cross-entropy, with quadrant Brier distance
as corroboration. Continuous-score negative log predictive density (NLPD)
diagnoses the Gaussian score forecast. Hard-quadrant accuracy is secondary
because it discards posterior uncertainty.

For every candidate-reference pair, the experiment resamples calendar-ordered
monthly deltas with 5,000 circular moving-block replications and 12-month
blocks. It reports the full sample and a sample excluding March-May 2020. For
the exclusion sensitivity, the pre-March and post-May calendar segments are
resampled separately; February and June are never treated as adjacent months.
The primary multiplicity correction is Holm's procedure across all fourteen
atomic add-one and leave-one-out contrasts for each metric, checkpoint, and
sample. Sparse active-month results are descriptive only; they are not fed to
the block bootstrap as though calendar gaps were adjacent months.

All 216 control-invariance checks pass across `transition_only`,
`partial_only`, and `student_t_7_combined`. The checks cover evaluation rows,
partial defining updates, exact-score updates, latest marginals, and the
selected baseline's semantic non-defining event updates.

## 4. Primary attribution results

The table reports block-benefit units. Positive numbers mean the block helps;
negative numbers mean the model is better without it.

| Block | \(A_b\), NLPD | \(C_b\), NLPD | \(A_b\), cross-entropy | \(C_b\), cross-entropy | \(A_b\), Brier | \(C_b\), Brier |
|---|---:|---:|---:|---:|---:|---:|
| Weekly labor stress | 0.065786 | 0.065813 | 0.001018 | 0.001025 | -0.0000685 | -0.0000668 |
| Monthly labor demand | -0.0000223 | -0.0000362 | -0.0000139 | -0.0000177 | -0.00000713 | -0.00000826 |
| Consumer demand | 0.000187 | 0.000195 | 0.0000648 | 0.0000690 | 0.0000304 | 0.0000327 |
| Housing activity | -0.00000389 | 0.00000462 | -0.00000166 | -0.00000134 | -0.000000799 | -0.000000669 |
| Business investment | 0.0000240 | 0.00000999 | 0.00000990 | 0.00000410 | 0.00000451 | 0.00000190 |
| Inflation expectations | -0.00000323 | 0.000000591 | 0.000000279 | 0.00000263 | -0.0000000130 | 0.00000140 |
| Inflation input costs | -0.000190 | -0.000170 | -0.0000633 | -0.0000568 | -0.0000270 | -0.0000263 |

No atomic contrast is significant after the fourteen-test Holm correction in
the full sample. In the segment-aware sensitivity excluding March-May 2020,
weekly labor stress has a positive NLPD benefit in both arms:

$$
A_{\mathrm{claims}}=0.01934,
\qquad
C_{\mathrm{claims}}=0.01936.
$$

Their 95% block-bootstrap intervals are approximately
\([0.01566,0.02391]\) and \([0.01575,0.02387]\). The centered two-sided
\(p\)-value is 0.0002 and the fourteen-test Holm-adjusted \(p\)-value is
0.0028 in both arms. This result is specific to continuous-score NLPD. Weekly
claims slightly improves quadrant cross-entropy and hard accuracy but slightly
worsens Brier distance, so it is the strongest block without being a uniform
winner on every metric. The pre-2020 paired effects are mostly zero and the
post-May-2020 segment drives this sensitivity; preserving each segment's fixed
sample share makes the interval tighter, but does not demonstrate temporally
stable value across eras. The nonsignificant full-sample result therefore
remains the primary uncertainty statement.

The selected baseline's full-sample change relative to `partial_only` is an
NLPD benefit of 0.065797, a cross-entropy benefit of 0.001020, a Brier benefit
of -0.0000672, and an accuracy benefit of 1.09 percentage points. Initial
claims alone produces 0.065786, 0.001018, -0.0000685, and 1.09 percentage
points, respectively. Thus initial claims explains virtually the entire net
movement from `partial_only` to the selected baseline; the other six blocks
nearly cancel in aggregate.

Consumer demand and business investment have small positive benefits in both
arms across the three probabilistic losses. Their intervals include zero and
their multiplicity-adjusted \(p\)-values equal one. The agreement of both arms
is evidence of direction, not proof of material forecast value.

Monthly labor demand and input costs are negative in both arms. Housing is
also negative on both quadrant losses. None is statistically distinguishable
from zero after multiplicity correction, but their signs and coverage identify
specific architectural changes worth testing.

## 5. Direct tests of the prior replacement candidates

The preceding six-priority experiment compared each replacement with the full
legacy baseline. This stage additionally compares replacements with the
matching reduced core:

| Replacement | Reference | NLPD benefit | Cross-entropy benefit | Brier benefit | Accuracy benefit |
|---|---|---:|---:|---:|---:|
| Ex-autos-and-gas retail plus vehicle units | Baseline without legacy consumer block | 0.009019 | -0.0000451 | 0.00000245 | 1.09 percentage points |
| Single-family starts, permits, and new-home sales | Baseline without legacy housing block | 0.000000743 | 0.00000142 | 0.000000708 | 0.00 percentage points |
| Orders plus backlog-to-shipments ratio | Baseline without legacy business block | -0.0000203 | -0.00000846 | -0.00000407 | 0.00 percentage points |

The consumer replacement improves mean score NLPD relative to having no
consumer block; its 95% percentile-bootstrap interval for the
candidate-minus-reference NLPD is \([-0.0247,-0.00054]\). Its centered test is
weaker (raw \(p=0.076\), three-test Holm-adjusted \(p=0.229\)), and its quadrant
cross-entropy is slightly worse. It is therefore a useful continuous-score
candidate, not a clear replacement for the legacy consumer likelihood.

The single-family housing replacement is directionally better than no housing
block on both quadrant losses, but its effect is extremely small and not
multiplicity-adjusted significant. The backlog replacement is worse than no
business-investment block on all three probabilistic losses.

## 6. Combined recommendation

The following recommendations combine this attribution with the preceding
six-priority evidence experiment. They define the next locked candidate; they
do not retroactively promote a new baseline on the same sample used for model
selection.

### Keep

- **Weekly initial claims:** this is the only block with a material,
  outlier-excluded, multiplicity-adjusted NLPD contribution. Preserve the
  log-AR(1) innovation, but address weekly serial overcounting.
- **Legacy consumer demand, provisionally:** it is small but positive in both
  attribution arms and has broad live coverage. Run response-level ablations
  before replacing either coordinate.
- **Legacy business investment:** orders and shipments are small but
  directionally useful. The backlog-ratio replacement performs worse than
  having no business block.

### Remove or disable in the next reduced-core candidate

- **Lagged monthly labor-demand likelihood:** it is negative in both arms and
  only 14 JOLTS events reach the live filter. This is a rejection of the timing
  architecture, not of JOLTS economics. Reintroduce it only after retargeting
  the release to a current/future score or constructing a genuinely leading
  transformation.
- **Intermediate-materials input-cost block:** it is negative in both arms on
  NLPD, cross-entropy, and Brier. Test a direct swap to import-price pressure
  rather than retaining both as independent likelihood factors.
- **Inflation-expectations block until adequate history exists:** three applied
  updates cannot support a performance claim. Keep collecting it, but do not
  count it as validated baseline evidence.

### Modify and retest

- **Housing:** remove the legacy aggregate block from the reduced core and test
  the single-family replacement as a predeclared candidate. Its advantage is
  currently too small to promote directly.
- **Labor stress:** continued claims was the best prior addition, but it should
  be modeled jointly with initial claims or applied as a correlated/tempered
  update rather than as another independent weekly likelihood.
- **Consumer demand:** the alternative ex-autos-and-gas retail plus
  vehicle-unit block improves continuous NLPD but not cross-entropy. A
  response-level or joint nominal-real
  decomposition is preferable to wholesale replacement.
- **Inflation pressure:** test import prices as a replacement for the harmful
  input-cost block inside one joint price-pressure likelihood. Do not simply
  append another correlated inflation factor.

### Do not add in the tested form

- the Empire State and Philadelphia survey pair;
- the capital-goods backlog ratio;
- the all-six evidence combination; or
- separate independent price blocks whose residual dependence is untreated.

The decision matrix is therefore:

| Current or proposed block | Decision for the next locked candidate | Reason |
|---|---|---|
| Initial claims | Keep, but modify | Strongest score-density evidence; weekly dependence and a small Brier deterioration remain |
| Consumer demand | Keep provisionally; modify | Broad coverage and small positive effects; alternative specification is metric-dependent |
| Core capital-goods orders and shipments | Keep | Small positive effects in both arms; backlog replacement is worse |
| Lagged JOLTS labor demand | Disable and retarget | Negative current implementation and only 14 live updates |
| Aggregate housing | Remove from reduced core; test single-family replacement | Legacy quadrant effects are negative; replacement is positive but negligible |
| Intermediate-materials input costs | Remove; test import-price replacement | Negative in both arms and short history |
| Inflation expectations | Park pending history | Three applied updates are not enough for inference |
| Continued claims | Add only as a joint/orthogonalized claims candidate | Best isolated prior addition, but correlated and not yet significant |
| Regional surveys | Do not add as tested | Redundant/harmful in the preceding experiment |

At later checkpoints, after defining releases have already narrowed the score
posterior, the quadrant effects of initial claims, consumer demand, and business
investment often attenuate or reverse. A future filter should therefore test
stage-dependent tempering rather than applying every non-defining likelihood at
full strength after more direct evidence has arrived.

The next defensible experiment should first lock a reduced-core reference
containing partial defining releases, weekly initial claims, provisional legacy
consumer demand, and provisional legacy business investment. Against that
common reference, separate predeclared arms should test single-family housing,
joint initial/continued claims, an import-price replacement for input costs,
and response-level consumer alternatives. These candidates require a new
causal replay and must not be evaluated by summing the individual effects in
this document.

## 7. Artifacts and limitations

The public artifacts are under
`results/published/m02_soft_composite/existing_block_attribution/`. The hashed
lineage manifest is `data/manifests/m02_existing_block_attribution.json`.

This experiment attributes observation models, not individual responses
inside a multivariate model. It cannot identify openings versus hires versus
quits, starts versus permits, or orders versus shipments. A block marked for
modification needs a response-level ablation before an individual series is
deleted. Finally, all architecture choices were motivated and evaluated on the
same history. Any reduced-core specification remains exploratory until locked
and assessed prospectively or with nested time-series model selection.
