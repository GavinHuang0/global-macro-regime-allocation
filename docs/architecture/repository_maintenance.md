# Repository maintenance

The repository keeps executable model stages, their specifications, and the
evidence behind research decisions together. Cleanup removes unused code while
preserving the ability to inspect and reproduce those decisions.

## What remains part of the research

| Location | Maintenance purpose |
|---|---|
| [`src/regime_allocation/`](../../src/regime_allocation/) | Shared acquisition, features, inference, allocation, and command-line stages |
| [`configs/models/`](../../configs/models/) | Explicit designs for current models, benchmarks, and historical experiments |
| [`data/manifests/`](../../data/manifests/) | Input, output, configuration, and source provenance for recorded runs |
| [`results/published/`](../../results/published/) | Reviewable evidence, including experiments that did not justify promotion |
| [`docs/archive/`](../archive/) | Historical methods and decision records |
| [`legacy/`](../../legacy/) | Original course manuscript and authorship attribution |
| [`tests/`](../../tests/) | Behavioral, causality, configuration, and artifact checks |

Historical names do not imply unused implementations. The current Model 02
baseline imports shared helpers from the evidence-experiment, existing-block
attribution, and inference-sensitivity stages. Removing those modules would
break the promoted pipeline. Model 02 also reuses Model 01 acquisition,
release-evidence, and allocation components.

The ETF download and FRED/ALFRED comparison scripts remain inputs to the
repository's GitHub Actions workflows. Command-line modules can also be run
with `python -m`, even when they have no console-script entry in
[`pyproject.toml`](../../pyproject.toml). Package `__init__.py` files remain part
of package discovery even when there is no direct import of the package name.

The incomplete course-project `legacy/bayesian_v2.py` script was removed: its
local dependencies were missing and no research stage used it. Its source
remains in Git history; the manuscript and team attribution remain in the
[legacy archive](../../legacy/README.md).

## Checking a proposed cleanup

1. Search tracked source, tests, console-script entries, workflows, configs,
   manifests, and documentation for references. Inspect the import graph and
   any string-based dispatch or public exports before deleting a definition.
2. Check both the promoted pipeline and historical reproduction commands.
   A rejected experiment can still supply shared code or explain a published
   research decision.
3. Update links and command documentation alongside any deletion. Keep the
   original manuscript, attribution, and recorded research evidence available.
4. Run the affected unit tests and `git diff --check`. Run relevant artifact
   integration checks when their recorded input data are available; distinguish
   missing data from a code regression.

Frozen manifests describe the code and data of their original runs. Source
edits and fresh provider downloads can change those hashes. Preserve the
recorded provenance; create new run artifacts through the relevant pipeline
instead of rewriting historical hashes to make a check pass.
