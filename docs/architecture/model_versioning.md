# Model versioning and comparison contract

The repository treats materially different probabilistic architectures as
separate models, not as flags hidden inside one implementation.

```text
point-in-time data contract
        |
        +--> m01 deterministic composites
        +--> m02 soft deterministic composite scores
        +--> m03 switching state-space model
        |
        +--> shared evaluation --> shared backtest --> published comparison
```

Each model owns:

- a frozen configuration under `configs/models/`;
- a package under `src/regime_allocation/models/`;
- model-specific methodology under `docs/models/`;
- its fitted artifacts and diagnostic outputs;
- a public result directory under `results/published/`.

All models must consume the same point-in-time event schema and use shared
portfolio/backtest code. This prevents a supposedly architectural comparison
from silently changing data, costs, benchmarks, or evaluation dates.

Model IDs are stable. A change to a threshold, data series, transformation,
warm-up, or likelihood is recorded in configuration and provenance. A change
to the state representation or inference graph receives a new model ID.

The fixed first-order transition matrix is part of
`m01_deterministic_composite`: it propagates uncertainty over the existing four
states without changing their definition or adding a new latent state. A
duration-dependent, semi-Markov, covariate-dependent, or time-varying transition
law changes the inference graph and therefore belongs under a distinct model ID
rather than behind a Model 01 configuration flag.

`m02_soft_composite` preserves observed, economically specified composite
scores but removes Model 01's trailing three-month average and hard quadrant
target. Its score-definition stage uses percentage payroll growth and publishes
no categorical regime. Its completed Gaussian map integrates score uncertainty
over the four quadrants using pooled component disagreement and fixed-horizon
revision covariance. The transition and filter evolve the released score center
$\boldsymbol Z_m=(G_m,I_m)^\top$ through an expanding, causal VAR(1). Once
released, $\boldsymbol Z_m$ is exact; only VAR process covariance
$\boldsymbol Q$ is propagated. The separate reporting variable
$\boldsymbol U_m=\boldsymbol Z_m+\boldsymbol\epsilon_m^{\mathrm{map}}$ adds
$\boldsymbol\Omega_{\mathrm{map}}$ when score distributions are integrated over
quadrants. Source-month mapping covariance is never propagated as
$\boldsymbol A\boldsymbol\Omega_{\mathrm{map}}\boldsymbol A^\top$.

Model 02's inference stage is implemented as a rolling four-month joint
Gaussian over score centers. Point-in-time non-defining releases entered the
earlier frozen Gaussian replay through structured ridge linear-Gaussian
observation models with one Ledoit--Wolf residual covariance per jointly fitted
model. A separate historical selection stage adds sequential partial
score-defining releases, Student-$t$ block emissions, Huber and Student-$t$ VAR
fits, and alternative
retail specifications without deleting or rewriting the upstream artifact.
At that stage, `student_t_7_combined` was selected as the inference baseline:
partial defining updates, fixed-$\nu=7$ Student-$t$ block emissions, the
nominal-retail specification, and OLS VAR(1) dynamics. Gaussian alternatives,
annually selected tails, robust VAR fits, and retail alternatives remain named
historical sensitivities. Complete composite scores become exact end-of-day
observations. That stage's baseline preserves actual reference months, so a
release for an already exact target is a
predictive diagnostic rather than being silently retargeted. Quadrant mapping
remains a readout layer; the four-month path approximation adds mapping
covariances block-diagonally and therefore records cross-month mapping-error
independence as an explicit approximation.

The current `current_baseline` publication stage promotes
`student_t_7_reduced_core` as the operational Model 02 baseline. It preserves
the four-month Gaussian state, OLS VAR(1), fixed-$\nu=7$ emissions, partial
defining updates, and exact completed-score conditioning while restricting
non-defining evidence to ICSA, joint real-retail/implicit-price evidence, and
joint capital-goods activity/pipeline evidence. Its public benchmarks are
`transition_only`, `partial_only`, and the frozen predecessor
`student_t_7_combined`; adding inflation expectations is the sole current
sensitivity. The earlier 23 feature variants and all four historical
sensitivity namespaces remain addressable through the current sensitivity
catalog.

The evidence data, frozen Gaussian replay, historical selection stages, current
baseline, and full causal replay through 20 July 2026 are complete and
published without altering Model 01 outputs. The component feed extends through
June even though the last complete score is May, allowing six released June
components to update the June/July joint posterior without inventing the two
missing PCE components. Historically, partial defining releases account for
most of the late-month improvement over transition-only. Heavy-tailed evidence
adds a smaller, metric-dependent increment; robust VAR and retail alternatives
do not dominate across all metrics and remain named sensitivities. Residual
diagnostics reject the conditional-independence approximation in the full sample,
although most cross-model rejections disappear when 2020 is excluded. A future
time-varying transition model or joint cross-block disturbance model must be
recorded as another named Model 02 sensitivity or a new model ID, depending on
whether the inference graph changes materially. Model 02's weekly allocation
is a downstream evaluation stage: it consumes the promoted baseline without
changing that inference graph and publishes under its own manifest and result
namespace.

The subsequent six-priority evidence-block experiment is an additive Model 02
sensitivity stage, not a new model and not a baseline rewrite. It holds the
fixed-$\nu=7$ Student-$t$ emissions, OLS VAR(1), partial defining releases,
score map, and evaluation clock constant while varying explicit evidence-set
allowlists. The frozen `student_t_7_combined` event path is verified before the
experiment can publish. None of the six candidates is promoted: continued
claims has the best isolated mean changes but not Holm-adjusted significance,
whereas the combined candidate worsens quadrant losses and exhibits extensive
conditional-residual dependence. No model role changed at that historical
stage.

The existing-block attribution stage is another non-promoting Model 02
sensitivity. It uses the original evidence table and tests each of the seven
asynchronous observation models both as a singleton addition to `partial_only`
and as a singleton removal from `student_t_7_combined`. It also directly joins
the prior consumer, housing, and backlog replacements to the appropriate
reduced-core references. The stage changes no score, mapping, transition,
emission, partial-release, or evaluation rule. Its conclusions specify a
reduced-core candidate for a later locked replay; they did not change the
then-selected baseline on the feature-selection sample.

That reduced-core replay is recorded as the Model 02 `feature_revision` stage.
It holds the score, quadrant map, OLS VAR(1), partial defining updates,
fixed-$\nu=7$ robust emissions, and evaluation calendar fixed while changing
only explicit evidence allowlists and declared response constructions. It
excludes legacy JOLTS and aggregate housing from every non-control arm, tests
import prices, decomposes consumer demand into real activity and an implicit
price coordinate, rotates business investment into current shipments activity
and an orders/shipments pipeline, and factorizes joint claims evidence so ICSA
and CCSA keep their true reference months. Its frozen controls are compared
semantically with numeric tolerance; `fit_id` is ignored because the expanded
candidate registry changes non-economic identifiers. The stage is
developmental and did not itself promote a baseline on the same history used to
choose features.

The subsequent immutable `current_baseline` stage is a model-governance layer,
not a fresh selection experiment. It replays only the reduced-core baseline,
three public benchmarks, and the inflation-expectations sensitivity; verifies
the three frozen predecessor paths; and writes to a new namespace. Because the
evidence graph was chosen using the same causal history, its promotion is for
prospective use and is not an unbiased out-of-sample validation claim. The
exact hierarchy and results are recorded in
[`baseline_promotion.md`](../models/m02_soft_composite/baseline_promotion.md).
