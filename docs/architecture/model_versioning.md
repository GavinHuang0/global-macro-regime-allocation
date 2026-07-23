# Model versioning

## Current model lines

| Model | Inference baseline | Allocation baseline | Status |
|---|---|---|---|
| `m01_deterministic_composite` | Event-driven fixed-\(\nu=7\) filter over deterministic regimes | Monthly `posterior_optimized` | Frozen benchmark |
| `m02_soft_composite` | `student_t_7_reduced_core` | Weekly `posterior_optimized` | Promoted; current iteration closed; not frozen |

Model 01 is retained as a stable architectural benchmark. Model 02 is the
current promoted model line and may receive a later, explicitly documented
revision.

## Version boundary

A model ID identifies a state representation and inference graph. Changes that
materially alter either receive a new model ID. Examples include:

- replacing Model 01's four hard quadrants with a continuous latent state;
- changing a first-order Markov state into a duration-dependent or
  covariate-dependent process; or
- replacing Model 02's rolling Gaussian score state with another state-space
  family.

A change within an existing graph may remain under the same model ID when it is
published as a named configuration or release. Examples include:

- an evidence allowlist;
- a likelihood family or fixed hyperparameter;
- an allocation frequency or constraint policy; and
- a promoted baseline selected from already named variants.

Such changes must update the relevant configuration, release identifier,
manifest, documentation, and result namespace. Existing content-addressed
artifacts remain reproducible and are not silently rewritten.

## Stage ownership

Each published stage owns:

```text
configuration       configs/models/
implementation      src/regime_allocation/
documentation       docs/models/
manifest            data/manifests/
public outputs       results/published/
tests                tests/
```

Manifests bind the exact configuration, inputs, implementation files, and
outputs used for a publication. A **hash-pinned** artifact is immutable as an
input to that publication. This is separate from declaring an entire model
line frozen.

## Shared comparison contract

Architectural comparisons should hold constant:

- the point-in-time data and release clock;
- the formal evaluation window;
- execution and holding-period definitions;
- portfolio universe, constraints, and cost treatment where applicable; and
- metric and resampling definitions.

When one of these must change, the comparison must state it directly. Model 01
and Model 02 inference scores should not be ranked against each other because
their targets differ: Model 01 scores eventual hard regimes, while Model 02
scores continuous macro states and a soft quadrant reporting distribution.

## Model 02 closeout status

The current Model 02 iteration is summarized and closed for now. Its promoted
inference and allocation baselines remain the repository defaults. This is not
a permanent freeze: later work may revise Model 02 through a new declared
release while preserving the current manifests and outputs as reproducible
predecessors.
