# Weekly Model 02 updates

The [weekly workflow](../../.github/workflows/update-m02-weekly.yml) computes a
dated Model 02 research allocation, refreshes its performance summary, and
updates the generated section of the repository README. The live run uses
`outputs/m02_weekly_live/` for ignored working files and publishes only:

- `results/live/m02_weekly/latest_allocation.json`;
- `results/live/m02_weekly/performance_summary.csv`;
- `results/live/m02_weekly/run_manifest.json`; and
- `README.md`.

## Enable GitHub updates

The workflow runs from the repository's default branch. For a fork, first
check that GitHub Actions is enabled in the repository.

1. Open the repository on GitHub. Choose **Settings → Secrets and variables →
   Actions → New repository secret**. Use the name `FRED_API_KEY` and paste the
   key into the secret value field. Save it there; do not put it in a tracked
   file, workflow, commit message, or issue. See GitHub's
   [repository secret instructions](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets#creating-secrets-for-a-repository).
2. Open **Actions → Update weekly Model 02 research snapshot → Run workflow**,
   select the default branch, and start a manual run. The workflow deliberately
   skips other branches.
3. Check both the compute and publish jobs. A successful publish creates a
   commit by `github-actions[bot]` only when the four publication files change.
   Repository or organization policies must permit the publish job's
   `contents: write` permission and its push to the default branch. Branch
   protection remains in force; the workflow does not bypass it.

The schedule is Monday and Tuesday at **22:23 UTC**, after the US trading
session: 6:23 p.m. in New York during daylight saving time and 5:23 p.m. during
standard time. Tuesday provides a retry after a Monday market holiday or a
temporary provider failure. It is an update attempt; the dates and freshness
checks in the output determine whether a snapshot is usable. GitHub schedules
run from the default branch and may be delayed or dropped during high load.
Public-repository schedules can also be disabled after 60 days of inactivity.
Use a manual run when needed. See GitHub's
[scheduled workflow behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).

Concurrent updates are serialized. Tests and computation run with read-only
repository permission and no stored push credentials. A separate job receives
only the completed four-file artifact, verifies its digest and file list, and
has `contents: write` permission. It stages those exact paths and refuses to
publish if the default branch moved during computation. A failed test, data
download, calculation, validation, or artifact transfer cannot publish a
partial update. GitHub's
[workflow permissions reference](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#permissions)
describes how job permissions interact with repository policy.

## Run locally

Install the package using the [quick start](../../README.md#quick-start).
The update command reads `FRED_API_KEY` from its process environment.

### macOS and Linux

In Bash or Zsh, enter the key at a hidden prompt and export it for the current
terminal session:

```bash
printf 'FRED API key: '
read -r -s FRED_API_KEY
printf '\n'
export FRED_API_KEY
python -m regime_allocation.cli.update_m02_weekly
```

### Windows

To make the key available to local Python, open Start and search for **Edit
environment variables for your account**. Under **User variables**, choose
**New** (or edit an existing entry), set the variable name to `FRED_API_KEY`,
and paste the key into its value. Save the dialog. Restart the terminal and
any editor or app that launches it so the new process inherits the variable.
The GitHub Actions secret is separate from this local setting.

From the repository folder, verify that the variable is present without
displaying its value:

```powershell
if ([string]::IsNullOrWhiteSpace($env:FRED_API_KEY)) {
    throw "FRED_API_KEY is not available in this terminal."
}
"FRED_API_KEY is available."
```

Use the installed virtual environment to run the same checks and update:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/unit/data/test_etf_history_download.py tests/unit/data/test_m02_live_inputs.py tests/unit/portfolio/test_m02_live.py tests/unit/reporting/test_readme_snapshot.py
.\.venv\Scripts\python.exe -m regime_allocation.cli.update_m02_weekly
```

Local runs write the same publication files without committing or pushing.
Inspect `git diff` and the run manifest after a successful run.

## Read the dates and research scope

The README snapshot distinguishes its generation time, signal week,
information cutoff, macro and price input dates, and expiry. A new rendering
time does not make old inputs current. Check those dates together with the
reported freshness status before interpreting the allocation.

The allocation is a model research output. Its estimated return and volatility
are forecasts, and the performance summary is a historical simulation with
the model's stated assumptions. Fresh provider downloads and the live update
window can change historical inputs or results. This workflow does not claim
exact reproduction of the hash-pinned archival runs under
[`results/published/`](../../results/published/), whose recorded provenance is
preserved separately.

If a run fails, the last published README and its dates remain visible. Inspect
the failed Actions step, correct the missing secret or provider problem, and
rerun. If publishing reports that the default branch moved, start a new run
from its latest commit. A permission or branch-protection rejection requires
a repository policy decision; the workflow never force-pushes.

Yahoo can return an incomplete latest row in multi-day price history even when
its one-day daily response is complete. The downloader recovers missing price
fields only for the requested latest session, at least 30 minutes after that
session closes. It requires matching session dates, provider identity, observed
prices and volume, and corporate actions. Both raw responses and a recovery
record are retained in the working market-data manifest. Historical gaps,
conflicting responses, or missing adjusted closes still stop publication.
