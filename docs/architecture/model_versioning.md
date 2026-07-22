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
Gaussian over score centers. Point-in-time non-defining releases enter the
frozen baseline through structured ridge linear-Gaussian observation models
with one Ledoit–Wolf residual covariance per jointly fitted model. A separate,
named sensitivity stage adds sequential partial score-defining releases,
Student-$t$ block emissions, Huber and Student-$t$ VAR fits, and alternative
retail specifications without overwriting the Gaussian baseline. Complete
composite scores become exact end-of-day observations. The baseline preserves
actual reference months, so a release for an already exact target is a
predictive diagnostic rather than being silently retargeted. Quadrant mapping
remains a readout layer; the four-month path approximation adds mapping
covariances block-diagonally and therefore records cross-month mapping-error
independence as an explicit approximation.

The evidence data, baseline filter, robust sensitivity implementation, and full
causal replay through 20 July 2026 are complete and published without altering
Model 01 outputs. The component feed extends through June even though the last
complete score is May, allowing six released June components to update the
June/July joint posterior without inventing the two missing PCE components.
Historically, partial defining releases account for most of the late-month
improvement over transition-only. Heavy-tailed evidence adds a smaller,
metric-dependent increment; robust VAR and retail alternatives do not dominate
across all metrics and remain named sensitivities. Residual diagnostics reject
the baseline conditional-independence approximation in the full sample,
although most cross-model rejections disappear when 2020 is excluded. A future
time-varying transition model or joint cross-block disturbance model must be
recorded as another named Model 02 sensitivity or a new model ID, depending on
whether the inference graph changes materially. Model 02 allocation and
backtesting remain pending.
