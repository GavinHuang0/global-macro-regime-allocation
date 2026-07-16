# Global Macro Regime Detection and Allocation

An independent, point-in-time rebuild of a team course project on macro-regime
detection and regime-conditioned asset allocation.

## Current status

Model 01 now implements the deterministic regime-definition and transition
layers:

- release-coherent first-release monthly macro features;
- four equal-weight growth and inflation components;
- strictly lagged expanding standardization;
- trailing three-month smoothing;
- deterministic four-quadrant regime labels;
- an expanding, first-order transition matrix with Jeffreys-prior smoothing;
- a 26-year point-in-time component panel from June 2000 through May 2026;
- local processed research files plus publication-safe regime outputs.

The acquired source panel spans 312 reference months from June 2000 through
May 2026. After the 60-observation warm-up, the public history spans July 2005
through May 2026: 247 months are classified and four are explicitly unavailable.
The gap from October 2025 through January 2026 originates with unpublished
October core-CPI and unemployment observations during the federal shutdown and
then propagates through the frozen one-month-change and three-month-smoothing
rules. Model 01 does not impute those observations or redistribute their weights.

The event-level Bayesian posterior, portfolio construction, and backtest remain
later milestones. Published Model 01 outputs include confirmed historical
regime labels and a transition-only next-month prior; they are not yet a
release-conditioned current-month nowcast.

## Model versions

| ID | Architecture | Status |
|---|---|---|
| `m01_deterministic_composite` | Deterministic first-release macro composites | Regime definition and fixed transition layer implemented |
| `m02_continuous_state` | Continuous latent growth/inflation state with quadrant probabilities | Planned |
| `m03_switching_state_space` | Regime-switching continuous state-space model | Planned |

Each later model receives its own configuration and model package. Data
contracts, evaluation, backtesting, and reporting remain shared so comparisons
are apples-to-apples.

## Model 01 in one paragraph

Growth combines nonfarm payrolls, industrial production, real personal
consumption, and the sign-reversed unemployment-rate change. Inflation combines
core CPI, core PCE prices, core finished-goods PPI, and average hourly earnings.
Monthly first-release transformations are standardized against expanding
history ending in the prior month, equally weighted within each axis, and
averaged over the current and preceding two months. The signs of those two
smoothed scores identify one of four regimes.

The complete mathematics, vintage rule, source substitutions, and limitations
are in the Model 01
[`regime definition`](docs/models/m01_deterministic_composite/regime_definition.md)
and [`transition model`](docs/models/m01_deterministic_composite/transition_model.md)
specifications.

## Current published result

The newest fully confirmed reference month is May 2026:

- regime: `growth_up_inflation_up`;
- growth score: `0.017332`;
- inflation score: `0.443875`;
- label available: June 25, 2026.

See the machine-readable
[`latest_confirmed.json`](results/published/m01_deterministic_composite/latest_confirmed.json),
the complete public
[`regime_history.csv`](results/published/m01_deterministic_composite/regime_history.csv),
and the frozen
[`data manifest`](data/manifests/m01_deterministic_composite.json). This is a
confirmed historical target, not a daily posterior or investable recommendation.

The transition layer publishes the auditable
[`transition model`](results/published/m01_deterministic_composite/transition_model.json),
its long-form
[`transition matrix`](results/published/m01_deterministic_composite/transition_matrix.csv),
and a
[`transition-only next-month prior`](results/published/m01_deterministic_composite/latest_transition_prior.json).
The prior is conditional on the latest eligible confirmed regime and contains
no event-level evidence.

At the June 25, 2026 information cutoff, the transition-only prior for the June
2026 reference month is:

| Regime ID | Probability |
|---|---:|
| `growth_up_inflation_up` | 69.05% |
| `growth_down_inflation_up` | 15.71% |
| `growth_up_inflation_down` | 12.86% |
| `growth_down_inflation_down` | 2.38% |

These probabilities are the Jeffreys-smoothed row conditioned on May's
confirmed `growth_up_inflation_up` regime. They are a baseline prior, not a
June posterior or a trading recommendation.

## Repository layout

```text
configs/
  models/                    frozen model definitions
data/
  raw/                       local provider downloads; ignored by Git
  processed/                 local research tables; ignored by Git
  manifests/                 tracked provenance and hashes
docs/
  architecture/              cross-model design decisions
  models/                    model-specific methodology
legacy/                      preserved team paper and surviving script
results/
  published/                 small, reviewable public outputs
src/regime_allocation/
  data/                      point-in-time providers and vintage logic
  features/                  shared feature transformations
  models/                    separately versioned architectures
  cli/                       reproducible commands
tests/                       unit, leakage, and integration tests
```

## Rebuild Model 01

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m regime_allocation.cli.build_m01_dataset --provider auto
python -m regime_allocation.cli.build_m01_transition
pytest
```

`auto` prefers the official FRED API whenever `FRED_API_KEY` is available and
otherwise selects the retained keyless ALFRED client. Both providers produce the
same validated, deterministic observation-by-vintage ZIP contract. Compatible
existing ALFRED matrices are checked before any network request, so adding the
secret does not discard or redownload data already gathered. An explicit FRED
refresh writes under a separate `data/raw/fred_api/` tree and cannot overwrite
the historical `data/raw/alfred/` artifacts.

Use `--provider fred --refresh` to require authenticated acquisition, or
`--provider alfred` to force the keyless path. Provider selection, cache
provenance, and the providers actually used are recorded separately in newly
generated manifests. See [`docs/data_access.md`](docs/data_access.md) for the
exact retrieval and credential contract.

## Legacy project and attribution

The original team manuscript and the only surviving Bayesian orchestration
script are in [`legacy/`](legacy/README.md). They are retained for provenance,
not as working Model 01 code. The team project was produced by Zekai Yao,
Mianchen Zhang, Gavin Huang, and Serin Gleave. This repository documents Gavin
Huang's later independent reimplementation, corrections, and extensions.

## Secrets and broker credentials

The FRED API key is optional locally because the ALFRED fallback remains
available. The authenticated GitHub workflow reads only the repository Actions
secret named `FRED_API_KEY`, scopes it to the retrieval step, and never passes
it as a command-line argument. Schwab tokens and local API keys must remain
outside this repository and outside Codex's readable environment; `.gitignore`
alone prevents commits but does not deny a local process read access. See
[`SECURITY.md`](SECURITY.md) before adding any credentialed provider.

## FRED attribution

This product uses the FRED® API but is not endorsed or certified by the
Federal Reserve Bank of St. Louis. Individual series may also be subject to
third-party usage restrictions; the project preserves public series links and
publishes derived, reviewable outputs rather than credentialed API responses.
