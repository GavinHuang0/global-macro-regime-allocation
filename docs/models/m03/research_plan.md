# Model 03: pipeline limitations and research plan

**Status:** research initialization; architecture and promotion criteria remain
to be specified. **Assessment date:** 14 September 2026.

This document audits Model 02 and defines the questions that Model 03 should
resolve. It is a research plan, not a new fitted model or a claim of improved
performance. Model 01 remains the frozen monthly benchmark; the promoted
Model 02 inference and allocation remain the operational reference.

The central question is whether information available at a portfolio decision
improves forecasts of subsequent asset returns or risk, and whether those
improvements survive portfolio constraints and implementation costs. A more
accurate description of the economy is useful, but does not establish that link
by itself.

**Evidence** below means an implemented assumption or a recorded result.
**Likely causes** are hypotheses unless a controlled experiment isolates them.
**Proposed improvements** are candidates for testing, not adopted solutions.
Numerical evidence comes from the linked historical publications and their
stated windows; it is not presented as a fresh Model 03 backtest.

## What the existing research establishes

| Evidence | Recorded result | Implication for Model 03 |
|---|---|---|
| [Promoted M02 inference][inference] | At the primary early-information checkpoint, 183 months from January 2011 through May 2026; all reported paired intervals include zero against the essential benchmarks | Additional evidence has not established a reliable early forecast improvement |
| [Seven-ETF allocation][allocation-results] | Across 445 weeks, posterior minus pooled annualized arithmetic net return is +0.01871 percentage points, with a 95% interval of [-0.08764, +0.12975] | Absolute portfolio performance does not demonstrate incremental macro value |
| [Benchmark-relative active allocation][active-oracle] | Across 445 weeks, active posterior minus pooled is -0.48845 percentage points annually, with a 95% interval of [-0.81338, -0.16530] | Giving the posterior an explicit active-risk budget did not rescue this implementation |
| [Exact-quadrant oracle][oracle-uncertainty] | Across its separate 405-week window, hindsight quadrant minus pooled is -0.53905 percentage points annually, with a 95% interval of [-0.91771, -0.16187] | Perfect classification is insufficient within the existing quadrant-to-weekly-return architecture |
| [Twelve-ETF expansion][expanded-universe] | Posterior CAGR fell from 9.86% to 8.79%; the interval for the annualized mean change still includes zero. Effective correlation dimension was only 4.24 | More securities did not automatically create independent opportunities or improve the estimator |

The oracle is intentionally infeasible. It changes the current quadrant vector
while preserving the causal return estimator and constrained active allocator.
It is **not** an upper bound for all uses of macro information. The active
strategies also had lower realized volatility than pooled, so their return
shortfalls do not establish inferiority under every possible risk objective.

The strongest working diagnosis is a weakness in the **representation and
estimation of the relationship between macro information and future returns**.
Inference, calibration, and execution still need improvement, but none should
be assumed to be the sole cause of the allocation result.

## Pipeline audit

The stages below follow the flow from source observations to published results.
The acceptance checks describe evidence to collect; numerical thresholds must
be fixed in an experiment specification before inspecting its test outcomes.

### 1. Source data, vintages, and availability

**Implementation:** the initial [M03 source-data layer](source_data.md) now
provides a versioned registry, immutable snapshots, availability and exclusion
ledgers, and source comparisons. It remains a separate data experiment;
verified intraday publication times and independent market-data validation
are still open research work.

**Current limitations.** Point-in-time coverage varies by series and archive
start. Some historical component gaps and a producer-price series splice are
explicitly configured. Publication and vintage dates provide a conservative
daily clock, not a verified intraday feed. Fresh Yahoo downloads can change
adjusted price history. These issues limit the common sample and the precision
of any claim about when information was tradable.
[Data policy][data-access] · [Score configuration][score-config]

**Likely causes.** The source archives were designed for economic data access,
not a complete trading replay. Series definitions, release practices, and
provider adjustments change over time. An archive's first available vintage
does not necessarily establish the original economic publication timestamp.

**Approach.** Create a source registry recording reference period, release
timestamp where verifiable, archive availability, retrieval time, source
definition, and transformation. Preserve immutable downloaded snapshots and
explicit reasons for missing rows. Audit the PPI splice and archive boundaries
on common-support samples. Add independent price/action reconciliation before
considering a second market-data provider. Preserve the existing same-vintage
change rule and do not backfill unavailable releases with revised values.

**Acceptance checks.** An as-of query cannot retrieve a future vintage;
replacing future data leaves past features unchanged; supported source changes
have a documented reconciliation; every missing observation has a reason.
Comparisons must separate data-coverage changes from model changes.

### 2. Composite targets, transformations, and scaling

**Current limitations.** M02's target is an equal-weight average of four
standardized components per axis. It is a defined first-release score, not a
directly observed structural economic state. Zero denotes a position relative
to historical standardized values, not an invariant expansion or inflation
threshold. Expanding mean/standard-deviation scaling is unbounded: the published
April 2020 growth score is approximately **-51.83**. Complete scores depend on
all defining inputs becoming available.
[Score definition][score-config] · [Published scores][score-history]

**Likely causes.** Equal weights prioritize interpretation over an estimated
measurement model. Rare shocks can be extreme relative to a long, quieter
history and can influence later scaling. Indicators share information, and
their relationship to economic activity need not be stable.

**Approach.** Specify separately whether M03 predicts a first-release composite,
a fixed-maturity revised quantity, or a latent economic factor. Retain M02's
score as a control. Test one causal robust-scaling alternative and one
regularized measurement model, separately. A mixed-frequency factor model is
a candidate for handling incomplete releases; it must be estimated within each
historical information set. [Bańbura and Modugno's missing-data factor framework](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1189.pdf)
provides a methodological starting point.

**Acceptance checks.** State interpretation, parameter stability, missing-data
behavior, and forecast errors should be documented by episode. Do not claim
better density scores merely because the target was rescaled or redefined;
compare models on a common observable target or a common decision task.
Retain crisis observations in the primary evaluation.

### 3. Macro transition dynamics

**Current limitations.** The promoted model rolls a four-month Gaussian state
using expanding OLS VAR(1) dynamics and plug-in parameters. A fixed transition
relationship cannot directly represent changing persistence, duration effects,
or stochastic volatility. Parameter estimation uncertainty is not a full
posterior distribution over the model coefficients.
[Inference specification][inference]

**Likely causes.** Parsimony is valuable with few independent macro cycles,
but pooling all past observations can slow adaptation. A Gaussian transition
can assign very little density to unprecedented shocks. These are model
assumptions to diagnose, not proof that a larger nonlinear model will help.

**Approach.** Compare the current VAR with a shrinkage VAR and one tightly
limited adaptation or volatility alternative. Inspect innovations, stability,
coverage, and forecast loss before considering switching states or nonlinear
dynamics. Robust VAR variants have already shown gains outside the pandemic
but large losses on the pandemic targets; repeat those comparisons only with
a distinct hypothesis and a preregistered evaluation.
[Prior dynamics experiments][inference-sensitivities]

**Acceptance checks.** Improvement must survive a fixed primary score, calendar
folds, and crisis-inclusive evaluation. Report calibration and computational
cost; do not select a model only because a favorable subperiod improves.

### 4. Release evidence and state updates

**Current limitations.** The reduced core has three non-defining evidence blocks,
fixed seven-degree-of-freedom Student-t likelihoods, and approximate Gaussian
moment updates. Partial defining updates use a fitted component distribution;
the completed first-release score is subsequently imposed exactly. Within-block
covariance and same-day safeguards exist, but the model does not jointly learn
every possible serial and cross-block residual dependency.
[Inference][inference] · [Partial-update design][inference-sensitivities]

**Likely causes.** Releases can repeat the same economic information, overlap
in reference periods, and change their predictive relationships. Partial
defining observations also reveal pieces of the target itself, so their
late-month benefit should not be confused with early forecasting skill.
Exact conditioning is internally correct for the defined first-release score;
it would need reconsideration if M03 instead targets an unobserved economic state.

**Approach.** Measure incremental information conditional on the other blocks,
remaining defining inputs, and publication stage. Recheck serial dependence
and cross-block residual dependence on the promoted graph before changing it.
Compare a joint or whitened observation design with the same inputs, then test
tail or parameter-uncertainty changes separately. Additional indicators need
verified vintage history and conditional contribution, not just plausible
economic relevance.

**Acceptance checks.** Release groups remain atomic for fitting; permuting
simultaneous events cannot change the intended result; duplicate or redundant
evidence does not create unjustified confidence. Evaluate early and later
checkpoints separately on matched months. Prior dependence findings from
broader predecessor graphs must not be attributed directly to the reduced core.

The [six-priority evidence experiment][evidence-experiments] already tested
surveys, continued claims, consumer quantities, housing, import prices, and
capital-goods backlog. None passed the primary Holm-adjusted comparisons, and
the combined addition worsened quadrant losses. A new joint model or new
indicator therefore needs a specific coverage, dependence, or measurement
hypothesis; adding more inputs is not an established improvement.

### 5. Probability mapping and calibration

**Current limitations.** Reporting adds a Gaussian mapping covariance to state
uncertainty. Its baseline combines pooled 12-month revision covariance with
diagonal component-disagreement variance and assumes zero mapping perturbation
mean. This is an uncertainty construction, not an independently observed
probability truth. At the last archived mapping point, growth disagreement
variance was about **1.690**, versus **0.052** for revisions; the construction
can materially affect the probabilities.
[Mapping configuration][mapping-config] · [Uncertainty summary][mapping-summary]

**Likely causes.** Component disagreement may represent economic heterogeneity
as well as uncertainty. Pooled revision errors may hide age, series, and
regime effects. A zero-mean perturbation can miss systematic revision bias.
Those explanations need measurement; the variance comparison alone does not
prove overdispersion.

**Approach.** Define the event whose probability is being assessed and its
verification date. Keep score reconstruction, fixed-maturity revision
prediction, and economic quadrant interpretation separate. Audit interval
coverage, probability-integral transforms, revision bias, and reliability by
release stage. Test age-dependent revision uncertainty or a measurement model
only after a reproducible calibration failure is identified. Calibrating to
the existing soft map evaluates agreement with that map, not independent
economic truth. Use [proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
against the chosen observable outcome; fit recalibration only on past folds.

**Acceptance checks.** Better sharpness must retain calibration and proper-score
performance. Report uncertainty around calibration diagnostics; do not force
more decisive quadrant probabilities simply to move portfolio weights.

### 6. Decision-time representation and return horizon

**Current limitations.** A continuous macro posterior is reduced to four quadrant
weights for a one-week return model. Historical weeks inherit the hard label
of the month containing their Monday. This discards state magnitude, direction
of change, and much of the uncertainty information. The oracle failure is
direct evidence that supplying the exact current label does not repair this
particular payoff model.
[Weekly alignment][weekly-estimator] · [Oracle diagnostic][active-oracle]

**Likely causes.** Monthly conditions and next-week asset returns may have weak
alignment. Markets respond to expectations, surprises, and discount rates as
well as economic levels. A repeated monthly quadrant may be too coarse even
when its classification is correct.

**Approach.** Build a decision-time feature table containing filtered state
means, uncertainty, and separately identified news updates. Distinguish a
mechanical month roll from new information. Compare four small payoff models:
pooled-only, current quadrant means, continuous state means, and continuous
means plus news/uncertainty. Start with the unchanged weekly horizon; examine
two- and four-week horizons in a separate matched experiment. Historical
features must be generated as they would have been known then, not from a
full-sample smoother. Model forecast innovations are not survey-consensus
surprises unless authentic historical consensus data are available.

**Acceptance checks.** Require forward forecast improvement before claiming
allocation value. Match each target to its actual holding horizon and
availability date; purge overlapping target intervals at fold boundaries.
Inspect whether predictive effects persist across months, assets, and episodes.

### 7. Expected returns, shrinkage, and adaptation

**Current limitations.** The return estimator uses expanding raw means, with
quadrant means shrunk toward the pooled mean by 104 pseudo-weeks. Shrinkage
reduces noisy differences but does not correct a stale pooled mean. Weekly
counts also overstate the number of distinct macro episodes. In the expanded
universe, the posterior estimator expected TLT to earn **4.75% annualized**
against **-0.82% realized**, with an average target weight of **17.95%**.
[Estimator][weekly-estimator] · [Asset diagnostics][asset-diagnostics]

**Likely causes.** Historical average returns combine compensation for risk
with conditions that can change. This pattern is consistent with slow
adaptation in duration forecasts; it does not isolate the rate regime as the
only cause. Counting correlated observations as equally informative can also
make the apparent support for a conditional mean misleading.

**Approach.** Forecast excess returns over BIL, comparing expanding estimation
with one limited rolling or forgetting-factor alternative. Keep raw-return and
excess-return control models explicit so a change of target is not mistaken
for new information. Test shrinkage toward zero excess views or economically
motivated carry estimates, with point-in-time inputs for the latter. Estimate
mean uncertainty using dependence-aware methods and transfer that uncertainty
into the allocation decision. Tune window length and shrinkage inside training
folds, not on the reported portfolio history.

**Acceptance checks.** Report forecast bias and loss by asset, coefficient
stability, effective sample support, and sensitivity to the most influential
episode. Evaluate both predictive loss and eventual net portfolio performance;
a less noisy estimate can still be economically wrong.

### 8. Asset universe and risk representation

**Current limitations.** ETF count is a poor measure of independent exposure.
The twelve-ETF experiment mostly added TLT; DBC, UUP, USO, and AGG received
essentially zero optimized weight. Expanding the Ledoit–Wolf estimator also
changed the original assets' estimated diagonal variances by **2.92% on average**,
so the experiment combines admission and covariance effects. Current risk
uses shared within-quadrant covariance plus dispersion of estimated means;
it is not a full predictive distribution including parameter uncertainty.
[Universe and covariance diagnostic][expanded-universe]

**Likely causes.** Nominal ETF weights hide overlapping duration, credit, and
commodity exposure. A historical covariance hedge can attract a large weight
even when its mean estimate is unreliable. Static residual covariance may
adapt slowly during changing volatility or correlation regimes.

**Approach.** Declare economically distinct exposures before selecting
instruments. Compare a common risk model with a preserved original-asset block
against separate covariance re-estimation; the augmented covariance must
remain positive semidefinite. Test one dynamic covariance alternative, then
factor risk budgets or duration limits using verified historical exposure
data. Do not reuse today's duration or holdings as historical metadata.

**Acceptance checks.** Evaluate predicted versus realized variance, tail-loss
coverage, exposure concentration, and stable utilization. M02's realized
10.42% volatility versus its estimated 10% ceiling warrants calibration
analysis, not a claim of an optimization bug. Add assets only when they offer
measurable independent exposure or forecast value on common-support samples.

### 9. Portfolio objective and constraints

**Current limitations.** The optimizer maximizes a plug-in expected return
after modeled trading costs, subject to volatility and concentration limits.
It often produces boundary allocations with few effective holdings. Solver
optimality verifies the numerical problem, not its economic inputs. The
existing benchmark-relative active experiment already tested tighter
expression of the posterior and failed its return-improvement gate.
[Allocation contract][allocation] · [Active diagnostic][active-oracle]

**Likely causes.** Small estimated mean differences can be exaggerated by a
linear objective. Caps and a common risk-premium component can make distinct
posteriors lead to similar portfolios. Conversely, shrinking tracking error
without improving the forecast can preserve an unfavorable view at smaller size.

**Approach.** Hold forecasts fixed while comparing the current objective with
an uncertainty-aware rule or a minimum-variance/risk-budget core plus limited
views. Predeclare whether the goal is excess return, risk-adjusted utility,
drawdown reduction, or another measurable objective. If the core portfolio
changes, retain both old and new cores as controls. Add no-trade bands only
after measuring how much usable signal they suppress.

**Acceptance checks.** Attribute improvements to forecasts, risk estimates, or
allocation separately. Require feasible constraints, stable weights under
small input perturbations, and improved predeclared net utility at comparable
risk. Repeatedly relaxing caps until the backtest improves is not evidence of
a better model.

### 10. Execution and portfolio accounting

**Current limitations.** The backtest uses adjusted opening prices, a fixed
5 bp one-way cost, and idealized fills. It omits variable spreads, impact,
capacity, and taxes. The current weekly update requires an observed common
session before publishing the new week's target, so the operational report
is generated after the opening price it models as execution.
[Allocation timing][allocation] · [Live command][live-command]

**Likely causes.** The pipeline is a research replay rather than an order and
fill system. Daily adjusted bars and a scalar cost model cannot represent all
opening-market conditions. However, lower costs in the failed active test and
small added costs in the universe test make trading cost an unsupported
primary explanation for those historical failures.

**Approach.** Separate pre-execution target generation from subsequent fill
and performance reconciliation. Use a known exchange calendar for planned
sessions and actual observations for realized fills. Stress execution delays,
opening slippage, spreads, and several predeclared costs. Preserve each
strategy's complete prior holdings, initial funding convention, realized drift,
and costs; a latest-week-only optimization is a different strategy.

**Acceptance checks.** Targets are saved before their prospective execution
time, hypothetical and realized timestamps are distinguishable, and accounting
reconciles independently. Benefits must survive reasonable cost and timing
stress. Do not retrospectively label a reconstructed target as a traded result.

### 11. Evaluation, selection, and statistical evidence

**Current limitations.** M02 has causal fitting and paired temporal block
bootstraps, but the graph and portfolio policy were developed after reviewing
the same history. Some experiments apply Holm correction within a defined
family; that does not correct the entire adaptive research process. The oracle
contains only 93 truth months and 58 regime runs, with 79.3% of weeks in
inflation-up quadrants. Thousands of bootstrap replications do not create
additional independent economic observations.
[Selection disclosure][current-config] · [Evidence experiments][evidence-experiments]

**Likely causes.** Limited vintage and ETF histories, persistent states, crisis
concentration, and repeated specification searches constrain power and increase
selection bias. A causal backtest can still be overfit through model selection.
The [backtest-overfitting literature](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
explains why reporting only the selected strategy obscures that process.

**Approach.** Register a small experiment family, primary target, comparator,
metric, economically meaningful threshold, tuning budget, and stopping rule.
Use chronological outer evaluation with all scaling, feature selection,
hyperparameter fitting, and recalibration inside earlier training periods.
Purge overlapping forward targets and apply any embargo required by their
information horizon. Report all attempts and multiplicity handling. Existing
inspected history remains development evidence, even when rearranged into
nested folds; reserve a genuinely prospective period after design lock.

**Acceptance checks.** Report paired effect sizes and intervals, block-length
sensitivity, common-support results, and minimum detectable effects. The
primary result must meet the economic threshold with adequate uncertainty
support and risk controls. A wide interval is inconclusive; a failed mean-alpha
test does not rule out a separately specified risk-forecasting benefit.

### 12. Reproducibility, computation, and live reporting

**Current limitations.** Manifests record hashes and runtime versions, but
dependency declarations use lower bounds rather than a fully locked environment.
Ignored source data and mutable provider histories prevent a clone alone from
reproducing every archived byte. Live runs rebuild history, write a latest
snapshot, and apply broad release-age checks. Those checks can pass while a
specific expected publication is missing. A static README cannot refresh its
displayed freshness label by itself.
[Runtime dependencies][dependencies] · [Live pipeline][live-pipeline]

**Likely causes.** Research provenance, computational reproducibility, and
operational monitoring solve different problems. Hashes identify inputs but
do not ensure access to them; generation timestamps and loose age thresholds
do not verify a full release calendar.

**Approach.** Add a locked, tested runtime and an authorized snapshot
distribution/retrieval contract. Preserve dated run artifacts and distinguish
data-only refreshes from model releases. Profile replay before introducing
incremental checkpoints; cache keys must include data, parameters, code, and
the information cutoff. Compare expected releases with actual arrivals and
track failures, expired outputs, and changed historical inputs. Separate
offline research tests from provider-dependent acquisition checks.

**Acceptance checks.** A clean environment reproduces a pinned fixture and
the recorded research snapshot when its inputs are available. Incremental
replay matches full replay. Missing expected releases and expired results are
visible; failed runs cannot silently replace successful reports with incomplete
or mixed-run artifacts. Publication dates must remain interpretable without
assuming a scheduled job has run.

## Order of work

| Priority | Work package | Reason to do it in this order |
|---|---|---|
| 1 | Lock the evaluation design and reconstruct the M02 control on a declared input snapshot | Every later comparison depends on trustworthy timing, accounting, and an explicit selection boundary |
| 2 | Build decision-time features and test the payoff representation at the existing weekly horizon | The active/oracle evidence makes the state-to-return relationship the first empirical bottleneck to test |
| 3 | Test mean adaptation, excess-return targets, and uncertainty one change at a time | The duration diagnostic exposes a concrete forecasting error that shrinkage toward pooled means did not remove |
| 4 | Audit macro targets and calibration; redesign inference only where the audit identifies a failure | Better classification or more confident probabilities alone did not rescue the existing allocation |
| 5 | Evaluate dynamic risk, factor exposures, and uncertainty-aware allocation under matched controls | These changes can alter portfolio behavior even if the macro forecast is unchanged |
| 6 | Apply implementation stress tests and start a locked prospective record | Historical improvements require an implementable and honestly dated follow-up |

Data integrity and reproducibility checks apply throughout, rather than being
deferred until the final work package. A poor return-forecast result should
stop increasingly complex allocation experiments using that same forecast.
It can instead motivate a separately defined risk-forecasting task.

## First controlled experiment

The first empirical M03 study should test the **payoff representation**, using
the existing M02 inference as a fixed source of information. This isolates a
research question before committing to a new state-space architecture.

| Arm | Decision-time predictors | Purpose |
|---|---|---|
| A | No current macro information; pooled return model | Allocation-relevant null |
| B | Four quadrant probabilities and the existing conditional-mean rule | M02 control |
| A2 | Intercept only, fitted on the same eligible rows as B2/C/D | Matched null for the new learner and training sample |
| B2 | Four quadrant probabilities supplied to the same regularized learner used in C | Separate the feature representation from a change of estimator |
| C | Continuous filtered growth and inflation means | Test information lost by quadrant compression |
| D | Arm C plus a limited, declared set of news changes and uncertainty measures | Test whether new information and confidence matter beyond state levels |

Retain A and B as native policy references, including their original label
availability rules. Compare A2/B2/C/D on identical eligible training rows,
target availability, chronological folds, and scoring dates, using the same
learner family and matched regularization-selection budgets where applicable.
Use unchanged simple weekly return targets for this first study. B2/C/D must
beat A2 before attributing benefits to macro predictors; compare C with B2 when
assessing representation. A difference from native B alone cannot isolate it.

Hold weekly timing, universe, costs, and the optimizer constant. Supply the
same causal covariance estimate to A2/B2/C/D and replay each portfolio's own
holdings path. M02's native posterior and pooled policies use different mixture
covariances, so their native results remain separate references rather than
pure mean-model ablations. Fixing the optimizer without fixing these risk
inputs would leave a confound.

Compare forward predictive loss before portfolio results; report gross and net
incremental performance, risk, turnover, and concentration. Horizon changes,
excess-return targets, new indicators, new assets, and a different optimizer
belong in subsequent experiments so their effects can be identified.

Before running this study, its specification must fix:

1. The observable target, return units, and exact availability timestamps.
2. Training and evaluation boundaries, treatment of overlapping outcomes,
   and the prospective lock date.
3. Primary predictive and economic metrics, the economic threshold, inference
   procedure, and multiplicity treatment.
4. The feature set, model classes, tuning ranges, and allowed number of trials.
5. Failure criteria for data quality, risk, implementation, and numerical stability.

No date already used for design or inspected results should be relabeled an
untouched holdout. If an arm improves macro or risk forecasting but not net
allocation, report that narrower finding explicitly.

## Model 03 version boundary and deliverables

This planning directory uses the neutral name `m03` because its architecture
is undecided. Under the [model-versioning contract][versioning], an allocator
or evidence-allowlist change alone can remain an M02 release. A new M03 model
identity should document a material change in state representation or inference
graph. The first controlled study can inform that decision without itself
claiming to implement M03.

The next deliverables are a versioned experiment specification, data/clock
audit, decision-time feature contract, and comparison report containing every
declared arm. A candidate model card and operational promotion follow only
after those results are reviewed. Existing M01/M02 configurations, manifests,
and published results remain reference artifacts.

[inference]: ../m02_soft_composite/inference.md
[allocation]: ../m02_soft_composite/portfolio_allocation.md
[allocation-results]: ../m02_soft_composite/portfolio_backtest_results.md
[active-oracle]: ../../archive/m02/active_optimizer_oracle_diagnostic.md
[oracle-uncertainty]: ../../../results/published/m02_active_optimizer_diagnostic/oracle_uncertainty.csv
[expanded-universe]: ../../archive/m02/expanded_universe_backtest.md
[asset-diagnostics]: ../../../results/published/m02_expanded_universe_backtest/asset_return_diagnostics.csv
[data-access]: ../../data_access.md
[score-config]: ../../../configs/models/m02_soft_composite.yaml
[score-history]: ../../../results/published/m02_soft_composite/score_history.csv
[inference-sensitivities]: ../../archive/m02/inference_sensitivities.md
[mapping-config]: ../../../configs/models/m02_probability_map.yaml
[mapping-summary]: ../../../results/published/m02_soft_composite/probability_map/uncertainty_summary.json
[weekly-estimator]: ../../../src/regime_allocation/portfolio/m02_estimation.py
[evidence-experiments]: ../../archive/m02/evidence_block_experiments.md
[current-config]: ../../../configs/models/m02_current_baseline.yaml
[live-command]: ../../../src/regime_allocation/cli/update_m02_weekly.py
[live-pipeline]: ../../../src/regime_allocation/portfolio/m02_live.py
[dependencies]: ../../../pyproject.toml
[versioning]: ../../architecture/model_versioning.md
