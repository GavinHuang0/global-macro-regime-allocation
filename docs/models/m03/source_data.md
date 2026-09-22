# M03 source-data layer

This implements the first stage of the [M03 research plan](research_plan.md#1-source-data-vintages-and-availability):
source definitions, preserved inputs, causal availability, and data-quality
comparisons. It produces audit artifacts and unscaled component changes. It
does not fit a new model or change the promoted M02 allocation pipeline.

## Data contract

The [source registry](../../../configs/data/m03_sources.yaml) declares the nine
score series and five evidence series used by the operational M02 graph. It
records each definition, transformation, reference frequency, acquisition
window, source URL, and configured active range. The two PPI series remain
separate inputs with an explicit splice in changes.

Acquisition boundaries inherited from M02 are **policies**, not independently
verified archive inception dates. The old M02 publication cutoff is retained
as provenance and does not cap new downloads. Release IDs describe acquisition
configuration; they do not establish an official publication timestamp.

| Clock | Meaning | Treatment |
|---|---|---|
| Reference period | Month or week measured by the observation | Explicit in every ledger row |
| Archive vintage date | Date attached to a FRED historical snapshot | Eligible at midnight in New York on the **next calendar day**, with daylight saving handled |
| Official publication timestamp | Verified intraday public release time | Unknown in this version; remains null |
| Retrieval time | Time the acquisition completed on this computer | UTC timestamp for fresh acquisition; null for imported legacy caches |
| Ingestion time | Time bytes entered the immutable store | Recorded independently in UTC |

For example, a vintage dated 6 March is unavailable to an as-of query at
23:59 New York time on 6 March. It first becomes eligible at 00:00 on 7 March.
This deliberately conservative rule does not prove when a release was tradable.

Two query modes make the distinction between archive history and system
history explicit:

- **`archive`** reconstructs information under the archive-date rule. A newly
  downloaded historical vintage may be used for historical research, but is
  not evidence that the local system possessed those bytes at that time.
- **`captured`** additionally requires a known retrieval time no later than the
  requested cutoff. Imported legacy caches cannot pass this test. A fresh run
  started at its as-of time generally completes retrieval afterward; replay
  its manifest at a later cutoff to test what has actually been captured.

Retained rows expose `information_available_at`: archive eligibility in archive
mode, or the later of archive eligibility and retrieval in captured mode.

Neither mode treats an unverified release date as an intraday timestamp.

## Preserved inputs and replay

Each fresh FRED acquisition saves the returned normalized vintage ZIP, or the
ICSA first-release CSV, under its SHA-256 content hash in
`data/raw/m03_sources/objects/sha256/`. These are the provider adapter's normalized
artifacts, not a claim that every original HTTP response is retained. Yahoo
imports preserve the original chart JSON bytes.

Every capture also creates a separate immutable receipt with the content hash,
query, source definition and version, provider, public URL, and both clocks.
The receipt has its own integrity hash. Reads verify the persisted receipt and
the object before using either. Capturing identical content reuses its object
but records a new receipt; changed content creates a new object. Existing bytes
are never replaced. This is application-enforced integrity, not hardware-backed
write-once storage; preserve the store with normal backups.

Legacy M02 imports first verify the input manifest's recorded hashes. Their
original download times remain unknown: file modification times and run dates
are not substituted. Credentials are neither stored in queries nor embedded
in source URLs.

## Run an audit

Run these commands from the repository root with the project environment
installed. A fresh macro acquisition requires `FRED_API_KEY` in the process
environment. On Windows, an existing user environment setting can be loaded
into the current PowerShell session without displaying the value:

```powershell
$env:FRED_API_KEY = [Environment]::GetEnvironmentVariable('FRED_API_KEY', 'User')
```

Acquire all 14 registered macro inputs with the current timestamp as the cutoff:

```powershell
.venv\Scripts\python.exe -m regime_allocation.cli.build_m03_source_audit --output-dir outputs/m03_sources/run-001
```

Each command requires a **new output directory**. Reusing a directory fails
instead of replacing the prior run. Downloads preserve the configured history
needed to identify first appearances; `--reference-start` and `--reference-end`
select the reported audit window. By default that window begins in January
2000 and ends on the final day of the previous month.

Useful options:

| Option | Purpose |
|---|---|
| `--as-of 2026-09-21T00:00:00-04:00` | Explicit information cutoff; a timezone is required |
| `--series PAYEMS UNRATE` | Audit a registered subset |
| `--reference-start 2014-01-01 --reference-end 2016-12-31` | Limit the reported reference periods |
| `--import-m02-manifest <path>` | Import and verify existing operational M02 inputs without downloading |
| `--replay-manifest <path>` | Rebuild from the immutable M03 objects and their original receipts |
| `--previous-manifest <path>` | Compare two snapshots using the current run's same cutoff, window and availability policy |
| `--prices-dir <directory>` | Import and audit saved Yahoo files named `TICKER.json` |
| `--availability-mode captured` | Require demonstrated retrieval by the cutoff |

For example, replay an earlier M03 run without contacting FRED:

```powershell
.venv\Scripts\python.exe -m regime_allocation.cli.build_m03_source_audit --output-dir outputs/m03_sources/replay-001 --replay-manifest outputs/m03_sources/run-001/manifest.json --as-of 2026-09-21T12:00:00-04:00
```

The same operation is available as `build-m03-source-audit` after reinstalling
the editable package to register the new entry point. Raw objects, receipts,
and run artifacts live in Git-ignored directories. Preserve the run manifest
**and its referenced snapshot store** to reproduce a run on another computer.

## Outputs and interpretation

| Artifact | Contents |
|---|---|
| `summary.md` | Readable coverage and comparison summary |
| `manifest.json` | Cutoff, registry, source receipts, implementation and artifact hashes |
| `feature_ledger.csv` | One row per score source and requested month, including excluded months |
| `observation_ledger.csv` | One row per evidence source and expected native reference period; ICSA uses Saturdays |
| `source_audit.json` | Coverage, exclusion reasons, PPI overlap, optional price checks, and snapshot comparisons |

Score changes use the earliest visible nonmissing appearance of the current
month and the prior month's level **from that same vintage**. Later revisions
cannot repair an invalid first appearance or a missing prior level. There is
no backfill or imputation. The first archive snapshot, negative or excessive
release lag, invalid levels, active source support, and missing references each
have explicit exclusion reasons. A missing observation is not automatically
classified as an unreleased observation: the archive may lack historical
coverage. Its exact economic cause often remains unknown.

Each row also records the source request's observation and vintage boundaries.
Periods outside the acquired reference range and missing values beyond an old
query's vintage horizon have distinct reasons. A payload contradicting its
declared acquisition bounds is rejected before feature construction.

The evidence ledger reports archive first-observed levels. It deliberately
does not apply the M02 evidence innovations, retail decomposition, or revision
features, and does not assert that every archive appearance is a genuine
initial economic publication. Downstream evidence construction is a separate
research stage.

The PPI audit temporarily removes the active splice ranges, then compares
retained first-appearance changes on common reference months. It retains each
leg's availability date and their later joint availability. It reports overlap,
coverage outside that overlap, and change differences; it never joins levels
or silently approves a source substitution. A short matching overlap does
not establish equality of definitions or behavior outside that window.

Comparisons with a previous manifest re-evaluate both snapshots at the same
cutoff and reporting window. Added or lost support is reported separately
from changed values and changed availability on common support. Registry
definition changes require a separate acquisition and explicit reconciliation,
rather than silently comparing differently defined inputs. These comparisons
measure data changes; they are not model-performance comparisons.

## Price and corporate-action checks

The price audit uses an independent implementation to compare adjusted-close
returns with returns reconstructed from close prices and cash dividends.
Yahoo's chart close is already split adjusted, so splits are not multiplied
into that return a second time. It records every eligible adjacent interval,
unmatched actions, invalid prices, and discrepancies above a stated 5 basis
point tolerance. Unknown simultaneous dividend/split units cannot pass
automatically. No comparable intervals means insufficient data, not a pass.

Only calendar dates strictly before the cutoff's New York date enter the
price audit. A single raw history file may have an incomplete latest session;
the audit records that problem instead of borrowing a separate recovery file.
The original downloader can continue its established recovery process.

A previous run with the same ticker enables common-support price and action
revision comparisons. Added/removed dates are reported separately from
changed levels and returns. This is **internal consistency within Yahoo**,
not validation against an independent provider. Adjusted histories can change
after later corporate actions, even for old dates. Preserving successive
snapshots makes those changes observable; it does not reconstruct missing
historical price vintages. There is no independent exchange-calendar check or
second market-data feed in this implementation.

## Acceptance checks and remaining work

The automated tests cover future-vintage perturbations, exact cutoff and
daylight-saving boundaries, same-vintage changes, rejected first appearances,
complete exclusion ledgers, unknown retrieval times, corruption, replay from
preserved bytes, common-support comparisons, and dividend/split arithmetic.

Run the focused checks with:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/data -k "source" -q
```

The remaining source-data research includes independently verified release
timestamps, deeper historical archive-boundary investigation, evidence-feature
construction under the new availability contract, and independent market-data
reconciliation. M01/M02 code, historical publications, and the main README are
unchanged by this experiment. Adoption by an allocation model requires a
separate comparison on common support.
