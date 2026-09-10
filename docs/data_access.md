# Data access and credentials

## Macro-data providers

The weekly M02 update requires authenticated FRED access through `FRED_API_KEY`:

```bash
python -m regime_allocation.cli.update_m02_weekly
```

It acquires the required historical vintages and writes dated working data under
`outputs/m02_weekly_live/`. See the [weekly update guide](operations/weekly_updates.md)
for credential setup and the generated artifacts.

The Model 01 acquisition commands support two St. Louis Fed data paths:

| Mode | Behavior |
|---|---|
| `auto` | Use the FRED API when `FRED_API_KEY` is available; otherwise use ALFRED |
| `fred` | Require authenticated FRED API access |
| `alfred` | Use the retained keyless ALFRED client |

Normal local use:

```powershell
python -m regime_allocation.cli.build_m01_dataset --provider auto
python -m regime_allocation.cli.build_m01_evidence --provider auto
```

Provider selection changes acquisition, not the normalized vintage-matrix
schema or downstream model definitions.

## Point-in-time rule

For a macro observation in reference month $`m`$, the pipeline identifies its
first eligible publication and reads both:

- the newly released level for $`m`$; and
- the preceding level visible in that same vintage.

Monthly changes are calculated locally from those two levels. This prevents a
later revision of month $`m-1`$ from entering month $`m`$'s first-release
feature.

The authenticated FRED path uses:

1. `output_type=4` to identify initial publication dates; and
2. `output_type=2` at those dates to retrieve the corresponding level
   snapshots.

The ALFRED path produces the same normalized vintage-matrix contract.

## Credential policy

`FRED_API_KEY` must be provided through a trusted process environment,
operating-system secret manager, or GitHub Actions secret. Never place it in:

- source code, tests, notebooks, YAML, or documentation;
- repository `.env` files;
- command-line arguments;
- request logs, cache identifiers, filenames, or manifests; or
- raw authenticated URLs in exceptions.

When the key is present and an authenticated request fails, the command reports
the sanitized failure rather than silently switching providers. Automatic
fallback occurs only when the key is absent.

If a key may have entered Git history, a log, or a shared artifact, revoke it
and replace the secret. Removing it from the current working tree is not
sufficient.

## Cache and provenance policy

Authenticated and keyless intermediate caches use separate directories.
Compatible local caches are validated before a network request. The
`--refresh` flag explicitly reacquires the configured date range.

Raw and processed research data are ignored by Git. Tracked manifests record
their expected paths, byte sizes, and SHA-256 hashes. Published outputs contain
derived research artifacts and provenance, never credentials.

## ETF data

The weekly M02 command downloads adjusted ETF history from Yahoo Finance and
records its coverage and input hash in the
[run manifest](../results/live/m02_weekly/run_manifest.json).
Historical portfolio stages use the local snapshot declared in:

```text
data/manifests/us_cross_asset_etf_universe_v1.json
```

Adjusted history includes provider back-adjustments and may change after future
distributions or corrections. It is therefore a reproducible snapshot, not
point-in-time market data.

## GitHub Actions

The [manual Model 01 reproduction workflow](../.github/workflows/reproduce-model-01.yml)
uses read-only repository permission and uploads generated artifacts without
committing them. The [weekly M02 workflow](../.github/workflows/update-m02-weekly.yml)
additionally publishes validated results and the README snapshot through a
separate job with write permission. Both read the repository secret
`FRED_API_KEY`; credential values are excluded from generated artifacts.

## Official references

- [FRED series observations API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
- [FRED API-key documentation](https://fred.stlouisfed.org/docs/api/api_key.html)
- [FRED API terms](https://fred.stlouisfed.org/docs/api/terms_of_use.html)
- [ALFRED download help](https://alfred.stlouisfed.org/help/downloaddata)

This product uses the FRED API but is not endorsed or certified by the Federal
Reserve Bank of St. Louis. Individual source series may have additional
third-party data restrictions.
