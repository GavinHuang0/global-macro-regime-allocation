# Model versioning and comparison contract

The repository treats materially different probabilistic architectures as
separate models, not as flags hidden inside one implementation.

```text
point-in-time data contract
        |
        +--> m01 deterministic composites
        +--> m02 continuous latent state
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

