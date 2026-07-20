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
revision covariance. Later Model 02 stages will add score transition dynamics
and score-conditioned evidence updates without altering Model 01 artifacts.
