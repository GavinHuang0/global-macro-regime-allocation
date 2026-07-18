# Data access and credential handling

This project supports two official St. Louis Fed data-access paths.
Authenticated FRED API access is the preferred path for new downloads, while
the existing keyless ALFRED web client remains available as a reproducibility
and availability fallback.

## Provider selection

The Model 01 data builder accepts three provider modes:

| Mode | Behavior |
|---|---|
| `auto` | Uses the authenticated FRED API when `FRED_API_KEY` is present and nonempty; otherwise uses the keyless ALFRED client. |
| `fred` | Requires `FRED_API_KEY` and fails with a sanitized error when the variable is unavailable. |
| `alfred` | Uses the retained keyless ALFRED release-calendar and graph-CSV path, regardless of whether an API key exists. |

The normal local command uses automatic selection:

```powershell
python -m regime_allocation.cli.build_m01_dataset --provider auto
python -m regime_allocation.cli.build_m01_evidence --provider auto
python -m regime_allocation.cli.build_m01_transition
python -m regime_allocation.cli.build_m01_inference
```

The two providers can be selected explicitly for diagnostics and
reproducibility:

```powershell
python -m regime_allocation.cli.build_m01_dataset --provider fred
python -m regime_allocation.cli.build_m01_dataset --provider alfred
python -m regime_allocation.cli.build_m01_evidence --provider fred
python -m regime_allocation.cli.build_m01_evidence --provider alfred
```

Provider selection changes how missing data are acquired. It does not change
the mathematical feature or regime definition, the normalized vintage-matrix
contract, or downstream output schemas.

## Authenticated retrieval contract

The FRED provider uses two official `fred/series/observations` output modes:

1. `output_type=4` identifies each observation's initial publication date.
2. `output_type=2`, requested at those dates with `units=lin`, returns every
   level available in each point-in-time snapshot.

This two-step design is deliberate. Initial-release values alone are not enough
to calculate a clean monthly change: the prior month's input must be read from
the same vintage as the current month's first release. The provider therefore
downloads full level snapshots, validates the exact requested vintage columns,
merges bounded batches, and serializes the result to the same deterministic ZIP
format used by the keyless provider. All transformations remain local and
auditable.

The authenticated implementation uses Python's in-process HTTPS client. It does
not use a subprocess transport because the documented FRED v1 authentication
contract supplies the API key as a query parameter. Request failures are
re-raised without the original credential-bearing URL.

Official references:

- [FRED series observations API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
- [FRED API-key contract](https://fred.stlouisfed.org/docs/api/api_key.html)
- [ALFRED output-format definitions](https://alfred.stlouisfed.org/help/downloaddata)

## GitHub Actions secret

The repository-level Actions secret must be named exactly:

```text
FRED_API_KEY
```

The manual workflow at `.github/workflows/reproduce-model-01.yml` passes that
secret only to the two macro-acquisition steps as an environment variable. It
deliberately:

- runs only through `workflow_dispatch`;
- grants only `contents: read` permission;
- checks out the repository with persisted Git credentials disabled;
- never prints or interpolates the key into a shell command;
- never supplies the key as a command-line argument;
- performs no commit, tag, release, pull-request, or push operation;
- uploads only generated manifests, public result files, and derived processed
  audit/reproducibility tables;
- excludes raw downloads and provider caches from the artifact.

The workflow invokes `--provider fred --refresh` explicitly, then rebuilds the
transition, posterior, frozen ETF snapshot, allocation, backtest, and tests. A
missing or invalid secret therefore stops reproduction instead of silently
producing an artifact through a different macro provider.

GitHub Actions secrets are not files in the checked-out repository. A secret
configured on GitHub is consequently unavailable to a normal local process—and
to tools operating only on the local workspace—unless it is separately
exported into that process's environment.

## Local credential rules

For local authenticated retrieval, provide `FRED_API_KEY` through a trusted
process environment or an operating-system secret manager. Do not place the
value in:

- source code, tests, notebooks, configuration YAML, or documentation;
- `.env` or other files inside the repository;
- command-line options or shell scripts;
- request logs, exception messages, cache keys, filenames, or manifests.

The key is authentication material, not part of a dataset's identity. Cache
identities and provenance records therefore exclude it. Errors should identify
the provider and failed operation without reproducing an authenticated request
URL.

If a key may have entered Git history, an Actions log, an artifact, or another
shared location, revoke it at FRED immediately and replace the GitHub Actions
secret with a new key. Removing it only from the current working tree is
insufficient.

## Keyless fallback

The ALFRED provider is intentionally retained. It supports:

- local reproduction when no FRED API key is available;
- comparison of normalized outputs across acquisition methods;
- recovery from an authenticated-provider configuration problem;
- continued use of the historical keyless acquisition path.

Automatic fallback occurs only when `FRED_API_KEY` is absent. When a key is
present but an authenticated request fails, the builder should surface that
failure rather than silently switching providers. This avoids concealing
authentication, rate-limit, or data-contract problems and ensures provenance
remains unambiguous.

## Existing caches and gathered data

Adding the FRED API provider does not delete or invalidate data already gathered
through the keyless client. Normal runs continue to reuse compatible normalized
cache files before making a network request.

Provider-specific intermediate caches remain separate so authenticated and
keyless response formats cannot collide. Downstream code consumes the same
normalized vintage matrix regardless of provider.

The `--refresh` flag is an explicit request to reacquire the configured date
range. In the manual GitHub Actions workflow this happens in an ephemeral
runner: the refreshed files are tested and uploaded as a temporary artifact,
but they do not alter the repository because the workflow has no write
permission and no commit or push step.

The committed historical data and existing normalized caches therefore remain
intact unless a maintainer separately reviews and intentionally commits
replacements.

## Manual refresh workflow

To run the authenticated refresh:

1. Configure `FRED_API_KEY` under the repository's GitHub Actions secrets.
2. Open **Actions** on GitHub.
3. Select **Reproduce frozen Model 01**.
4. Choose **Run workflow**.
5. Review the build and test logs.
6. Download the `frozen-model-01-<run-id>` artifact if the job succeeds.

The artifact contains only:

```text
data/manifests/m01_deterministic_composite.json
data/manifests/m01_non_defining_release_evidence.json
data/manifests/m01_event_driven_bayesian_filter.json
data/manifests/us_cross_asset_etf_universe_v1.json
data/manifests/m01_regime_allocation_backtest.json
data/processed/m01_deterministic_composite/*
data/processed/m01_non_defining_release_evidence/*
data/processed/m01_bayesian_filter/*
data/processed/us_cross_asset_etf_universe_v1/*
data/processed/m01_regime_allocation_backtest/*
results/published/m01_deterministic_composite/
results/published/m01_bayesian_filter/
results/published/m01_regime_allocation_backtest/
```

Artifact creation is a validation and handoff mechanism, not a publication
step. Publishing refreshed results to the repository should be handled later
through an intentionally reviewed commit or pull request.

## Attribution and data rights

This product uses the FRED® API but is not endorsed or certified by the Federal
Reserve Bank of St. Louis.

FRED aggregates series from multiple originating agencies, and individual
series may carry separate third-party restrictions. Public project outputs
therefore retain series links and provenance while excluding credentials and
raw authenticated request URLs. See the official
[FRED API terms of use](https://fred.stlouisfed.org/docs/api/terms_of_use.html).
