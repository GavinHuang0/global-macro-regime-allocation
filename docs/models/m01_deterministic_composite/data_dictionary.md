# Model 01 data dictionary

## Source components

| Component | FRED/ALFRED series | Release | Earliest vintage used | Axis | Frozen transformation |
|---|---|---:|---|---|---|
| `payrolls` | `PAYEMS` | 50 | 1990-01-01 global floor | Growth | Same-vintage level difference |
| `industrial_production` | `INDPRO` | 13 | 1990-01-01 global floor | Growth | Same-vintage monthly log change |
| `consumer_activity` | `PCEC96` | 54 | 1990-01-01 global floor | Growth | Same-vintage monthly log change |
| `unemployment_rate` | `UNRATE` | 50 | 1990-01-01 global floor | Growth | Negative same-vintage level difference |
| `core_cpi` | `CPILFESL` | 10 | 1996-12-12 | Inflation | Same-vintage monthly log change |
| `core_pce` | `PCEPILFE` | 54 | 2000-08-01 | Inflation | Same-vintage monthly log change |
| `producer_prices` | `PPILFE`, then `WPSFD4131` | 46 | 1996-12-11; 2015-03-13 | Inflation | Same-vintage monthly log change |
| `average_hourly_earnings` | `AHETPI` | 50 | 1999-08-06 | Inflation | Same-vintage monthly log change |

All series are monthly and seasonally adjusted. Candidate vintage dates come
from the associated release family's official ALFRED calendar. Batched as-of
level snapshots are then retrieved from ALFRED's historical graph CSV endpoint.
A release calendar can include a date on which another series in the same
release changed;
this is harmless because the first-release extractor selects the earliest dated
snapshot that contains the target observation.

Responses are requested in bounded vintage batches, validated against the exact
requested vintage set, and merged into a deterministic normalized ZIP cache.
The cache is not a byte-for-byte provider response.

## `first_release_components_long.csv`

| Field | Meaning |
|---|---|
| `reference_month` | Month the observation measures, normalized to month start |
| `component` | Stable semantic component name |
| `series_id` | Provider series identifier used for that row |
| `release_date` | Earliest selected ALFRED vintage containing the current month |
| `current_value` | Current-month level in that first-release vintage |
| `previous_value_as_of_release` | Prior-month level in the same vintage |
| `transform` | `difference`, `negative_difference`, or `log_difference` |
| `transformed_value` | Frozen first-release monthly feature |
| `release_lag_days` | Calendar days from reference month end to release date |
| `source_url` | Public ALFRED series page |

## `composite_features_and_regimes.csv`

For each component `k`:

- `{k}_transformed`: frozen monthly feature;
- `{k}_z`: strictly lagged expanding z-score.

Axis and label fields:

| Field | Meaning |
|---|---|
| `growth_raw` | Equal-weight mean of four growth z-scores |
| `inflation_raw` | Equal-weight mean of four inflation z-scores |
| `growth_smoothed` | Trailing three-month growth composite |
| `inflation_smoothed` | Trailing three-month inflation composite |
| `label_available_at` | Latest first release date among all eight components |
| `regime_id` | Stable quadrant identifier |
| `regime_label` | Human-readable regime name |
| `data_status` | `classified`, `missing_component_feature`, or `unavailable_trailing_window` in the public history |

## Publication outputs

`results/published/m01_deterministic_composite/regime_history.csv` contains only
axis scores, deterministic labels, availability dates, and explicit data-status
flags. It excludes raw provider levels and retains the four unavailable months
from October 2025 through January 2026. `latest_confirmed.json` is a compact
machine-readable snapshot of the newest fully classified reference month.

`data/manifests/m01_deterministic_composite.json` records the configuration
hash, raw ZIP hashes, coverage, and generated-file paths.
