# Repository maintenance

Executable model stages, specifications, and results are maintained as linked
research artifacts. Changes must preserve the dependencies and provenance
needed to inspect and reproduce recorded research decisions.

## Research artifacts

| Location | Maintenance purpose |
|---|---|
| [`src/regime_allocation/`](../../src/regime_allocation/) | Shared acquisition, features, inference, allocation, and command-line stages |
| [`configs/models/`](../../configs/models/) | Explicit designs for current models, benchmarks, and historical experiments |
| [`data/manifests/`](../../data/manifests/) | Input, output, configuration, and source provenance for recorded runs |
| [`results/published/`](../../results/published/) | Reviewable evidence, including experiments that did not justify promotion |
| [`results/live/`](../../results/live/) | Dated outputs from the weekly research update |
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

## Changing or removing code

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
