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

All series are monthly and seasonally adjusted. The primary FRED API provider
uses output type 4 to identify series-specific initial-release dates, then uses
output type 2 to retrieve full level snapshots at those dates. The retained
keyless provider instead gets candidate dates from the associated release
family's ALFRED calendar and retrieves snapshots from the historical graph CSV
endpoint. A release-family calendar can include a date on which another series
changed; this is harmless because the extractor selects the earliest dated
snapshot containing the target observation.

Responses are requested in bounded vintage batches, validated against the exact
requested vintage set, and merged into a deterministic normalized ZIP cache.
The cache is not a byte-for-byte provider response.

## `first_release_components_long.csv`

| Field | Meaning |
|---|---|
| `reference_month` | Month the observation measures, normalized to month start |
| `component` | Stable semantic component name |
| `series_id` | Provider series identifier used for that row |
| `release_date` | Earliest selected point-in-time vintage containing the current month |
| `current_value` | Current-month level in that first-release vintage |
| `previous_value_as_of_release` | Prior-month level in the same vintage |
| `transform` | `difference`, `negative_difference`, or `log_difference` |
| `transformed_value` | Frozen first-release monthly feature |
| `release_lag_days` | Calendar days from reference month end to release date |
| `source_url` | Public, credential-free FRED series page |

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

## `transition_pairs.csv`

`data/processed/m01_deterministic_composite/transition_pairs.csv` is the
pair-level audit trail used to learn the transition matrix. It contains every
adjacent row pair considered by the transition build, including exclusions.

| Field | Meaning |
|---|---|
| `source_reference_month` | Earlier reference month in the candidate pair |
| `destination_reference_month` | Later reference month in the candidate pair |
| `source_regime_id` | Deterministic regime ID for the source month, when available |
| `destination_regime_id` | Deterministic regime ID for the destination month, when available |
| `source_label_available_at` | Date on which the source label became knowable |
| `destination_label_available_at` | Date on which the destination label became knowable |
| `pair_available_at` | Later of the two label-availability dates |
| `included` | Whether the pair contributes to the fitted count matrix |
| `exclusion_reason` | Empty for an included pair; otherwise the failed eligibility rule |

A pair is included only if the reference months are exactly one calendar month
apart, both regimes are nonmissing, and both labels are available by the
knowledge cutoff. A missing block is never bridged. The audit table is a local
processed artifact and is ignored by Git.

## `transition_model.json`

`results/published/m01_deterministic_composite/transition_model.json` is the
authoritative machine-readable transition artifact. The independently hashed
stage configuration is
`configs/models/m01_deterministic_composite_transition.yaml`; transition
settings are not hidden inside the deterministic data configuration. Its
schema contains:

| Field or block | Meaning |
|---|---|
| `schema_version` | Version of this artifact's serialization contract |
| `model_id` | Stable owning model ID, `m01_deterministic_composite` |
| `stage_id` | Stable transition-stage identifier, `fixed_first_order_transition` |
| `generated_at_utc` | Build timestamp; distinct from the information cutoff |
| `knowledge_cutoff` | Latest date on which an input label may be known for this fit |
| `state_order` | Frozen row and column order, with stable regime IDs and labels |
| `source` | Regime-history path, SHA-256, and row count |
| `configuration` | Exact transition-configuration path and SHA-256 |
| `transition_specification` | First-order and time-homogeneous flags, expanding-window rule, $\alpha=0.5$, and credible-interval level |
| `diagnostics` | History and adjacent-pair counts, included transitions, exclusions by reason, and latest eligible destination month |
| `counts` | $4\times4$ nested matrix of eligible $N_{ij}$ counts |
| `mle_probabilities` | Nested unsmoothed diagnostic matrix $N_{ij}/N_i$; not the production transition law |
| `posterior_parameters` | Nested matrix of Dirichlet parameters $N_{ij}+0.5$ |
| `posterior_predictive` | Nested Dirichlet-smoothed matrix used by Model 01 |
| `marginal_credible_intervals` | Interval level, lower and upper nested matrices, and interpretation |
| `published_files` | Paths of the matrix CSV and latest transition-only prior published with the model artifact |

Matrices are accompanied by `state_order`; their positional indices must never
be interpreted without it. The raw MLE may contain zero probabilities and is
published only as a diagnostic. The `posterior_predictive` block is the fitted
transition law.

## `transition_matrix.csv`

`results/published/m01_deterministic_composite/transition_matrix.csv` is a
long-form, human-reviewable view with 16 rows, one for every directed state
pair.

| Field | Meaning |
|---|---|
| `from_regime_id` | Stable origin regime ID |
| `from_regime_label` | Human-readable origin regime label |
| `to_regime_id` | Stable destination regime ID |
| `to_regime_label` | Human-readable destination regime label |
| `transition_count` | Eligible raw count $N_{ij}$ |
| `from_row_total` | Total eligible outgoing transitions $N_i$ |
| `mle_probability` | Unsmoothed diagnostic probability |
| `posterior_parameter` | Dirichlet posterior parameter $N_{ij}+0.5$ |
| `posterior_predictive_probability` | Production probability $(N_{ij}+0.5)/(N_i+2)$ |
| `credible_interval_level` | Marginal interval mass, fixed at 0.95 |
| `credible_interval_lower` | Lower endpoint of the marginal 95% credible interval |
| `credible_interval_upper` | Upper endpoint of the marginal 95% credible interval |

Long form avoids unlabeled positional matrices and makes every probability easy
to review or compare across builds.

## `latest_transition_prior.json`

`results/published/m01_deterministic_composite/latest_transition_prior.json`
contains the transition-only prior for the month immediately after the latest
eligible confirmed regime.

| Field | Meaning |
|---|---|
| `schema_version` | Version of the prior serialization contract |
| `model_id` | Owning Model 01 ID |
| `stage_id` | `fixed_first_order_transition` |
| `status` | `transition_only_prior` |
| `knowledge_cutoff` | Information date used to fit the matrix and select the conditioning label |
| `source_reference_month` | Latest eligible confirmed reference month |
| `target_reference_month` | Calendar month immediately after the source month |
| `conditioning_regime_id` | Confirmed deterministic source regime ID |
| `conditioning_regime_label` | Human-readable source label |
| `probabilities` | Four ordered destination records containing regime ID, label, and probability |
| `probability_sum` | Explicit normalization diagnostic |
| `interpretation` | Machine-readable warning that no event likelihood has been applied |

This file is one posterior-predictive matrix row. It is not a four-month joint
distribution and does not incorporate event-level likelihoods, market data, or
current-month release evidence. It must not be described as a Bayesian
posterior or an investable recommendation.

## Publication outputs

`results/published/m01_deterministic_composite/regime_history.csv` contains only
axis scores, deterministic labels, availability dates, and explicit data-status
flags. It excludes raw provider levels and retains the four unavailable months
from October 2025 through January 2026. `latest_confirmed.json` is a compact
machine-readable snapshot of the newest fully classified reference month.

The transition build publishes `transition_model.json`,
`transition_matrix.csv`, and `latest_transition_prior.json` alongside those
regime outputs. The first is the authoritative specification and fit summary,
the second is a reviewable cell-level representation, and the third is an
explicitly transition-only next-month prior.

`data/manifests/m01_deterministic_composite.json` records the configuration
hash, raw ZIP hashes, coverage, generated-file paths, requested and selected
providers, the provider associated with each cache, and whether each matrix was
newly downloaded or reused. Credentials and credential-derived identifiers are
excluded.
