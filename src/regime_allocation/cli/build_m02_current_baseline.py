"""Build and publish the compact current Model 02 inference baseline.

This command is the governance boundary between the historical feature-
revision experiment and the model that the repository currently presents as
its operational baseline.  It does not download data or read credentials.  It
verifies the feature-revision manifest, reuses its point-in-time consolidated
event universe, and performs a fresh causal replay of only five models:

* ``student_t_7_reduced_core`` -- the selected reduced-core baseline;
* ``transition_only`` -- the dynamics-only benchmark;
* ``partial_only`` -- the partial-defining-release benchmark;
* ``student_t_7_combined`` -- the frozen predecessor benchmark; and
* ``student_t_7_reduced_core_with_expectations`` -- one parsimoniousness
  sensitivity.

The selected evidence factorization is deliberately small.  It contains ICSA,
the joint real-retail/implicit-price response, and the joint capital-goods
activity/pipeline response.  JOLTS, aggregate housing, legacy input costs,
import prices, vehicle sales, CCSA, and inflation expectations are absent from
the baseline.  Inflation expectations are isolated in the fifth replay so the
choice to omit their three usable historical updates remains auditable.

Three predecessor models are compared semantically with their frozen artifacts
after replay.  Their publication labels may change, but their probabilities,
scores, event updates, and fit timing may not.  Full feature-search artifacts
remain under the historical ``feature_revision`` stage; this command publishes
only a compact current-model view and a catalog pointing back to those retained
sensitivities.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from regime_allocation.cli.build_m02_existing_block_attribution import (
    METRICS,
    _comparison_id,
    _eligible_frames,
    _invariance_rows,
    _paired_bootstrap,
    _paired_comparisons,
)
from regime_allocation.cli.build_m02_evidence_experiment_inference import (
    _merge_observation_models,
)
from regime_allocation.cli.build_m02_inference_sensitivities import (
    _evaluation_summary,
    _load_yaml,
    _project_path,
    _write_csv,
    _write_json,
)
from regime_allocation.data.dataset_acquisition import sha256
from regime_allocation.models.m02_soft_composite.inference_sensitivities import (
    run_inference_sensitivities,
    variant_registry_from_config,
)
from regime_allocation.models.m02_soft_composite.walkforward import (
    prepare_observation_data,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "current_baseline"
BASELINE_ID = "student_t_7_reduced_core"
EXPECTATIONS_SENSITIVITY_ID = "student_t_7_reduced_core_with_expectations"
LEGACY_BASELINE_ID = "student_t_7_combined"
ENGINE_BENCHMARK_IDS = ("transition_only", "partial_only")
PUBLICATION_BENCHMARK_IDS = (*ENGINE_BENCHMARK_IDS, LEGACY_BASELINE_ID)
FROZEN_CONTROL_IDS = (*ENGINE_BENCHMARK_IDS, LEGACY_BASELINE_ID)
DEFAULT_CONFIG = Path("configs/models/m02_current_baseline.yaml")

SELECTED_CANDIDATE_MODELS = (
    "consumer_real_implicit_joint",
    "business_activity_pipeline_joint",
)
BASELINE_OBSERVATION_MODELS = frozenset(
    {
        "weekly_labor_stress",
        "consumer_real_implicit_joint",
        "business_activity_pipeline_joint",
    }
)
EXPECTATIONS_OBSERVATION_MODELS = frozenset(
    {*BASELINE_OBSERVATION_MODELS, "inflation_expectations"}
)

IMPLEMENTATION_FILES = (
    "src/regime_allocation/cli/build_m02_current_baseline.py",
    "src/regime_allocation/cli/build_m02_evidence_experiment_inference.py",
    "src/regime_allocation/cli/build_m02_existing_block_attribution.py",
    "src/regime_allocation/cli/build_m02_inference_sensitivities.py",
    "src/regime_allocation/models/m02_soft_composite/gaussian_emissions.py",
    "src/regime_allocation/models/m02_soft_composite/inference_sensitivities.py",
    "src/regime_allocation/models/m02_soft_composite/joint_filter.py",
    "src/regime_allocation/models/m02_soft_composite/partial_defining.py",
    "src/regime_allocation/models/m02_soft_composite/robust_emissions.py",
    "src/regime_allocation/models/m02_soft_composite/var_transition.py",
    "src/regime_allocation/models/m02_soft_composite/walkforward.py",
)

REQUIRED_SOURCE_KEYS = frozenset(
    {
        "base_filter_config",
        "frozen_sensitivity_config",
        "feature_revision_config",
        "feature_revision_manifest",
        "score_features",
        "defining_components",
        "mapping_history",
        "consolidated_event_universe",
        "frozen_evaluation_rows",
        "frozen_event_update_audit",
        "frozen_partial_defining_audit",
        "frozen_exact_score_audit",
        "frozen_latest_marginals",
        "feature_revision_registry",
        "feature_revision_summary",
    }
)

REQUIRED_OUTPUT_KEYS = frozenset(
    {
        "processed_dir",
        "published_dir",
        "manifest",
        "evaluation_rows",
        "evaluation_summary",
        "evaluation_subperiod_summary",
        "paired_comparisons",
        "paired_block_bootstrap",
        "event_update_audit",
        "emission_fit_audit",
        "partial_defining_audit",
        "exact_score_audit",
        "transition_fit_audit",
        "latest_marginals",
        "baseline_invariance",
        "engine_registry",
        "current_registry",
        "sensitivity_catalog",
        "baseline_latest_marginals",
        "benchmark_latest_marginals",
        "method_summary",
    }
)


def _manifest_file_hash(manifest: Mapping[str, Any], relative_path: str) -> str:
    """Return the unique hash declared for an input or generated artifact."""

    rows: list[Mapping[str, Any]] = []
    for key in (
        "source_files",
        "generated_files",
        "generated_file_hashes",
        "implementation_files",
        "implementation_file_hashes",
    ):
        declared = manifest.get(key, ())
        if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
            rows.extend(row for row in declared if isinstance(row, Mapping))
    matches = [
        str(row["sha256"])
        for row in rows
        if str(row.get("path", "")) == relative_path and row.get("sha256")
    ]
    if len(set(matches)) != 1:
        raise ValueError(f"feature-revision manifest lacks one hash for {relative_path}")
    return matches[0]


def _verified_feature_bytes(
    root: Path, relative_path: object, manifest: Mapping[str, Any]
) -> bytes:
    """Read a feature-stage artifact only after its declared hash matches."""

    relative = str(relative_path)
    path = _project_path(root, relative)
    content = path.read_bytes()
    if sha256(content) != _manifest_file_hash(manifest, relative):
        raise ValueError(f"feature-revision artifact hash mismatch: {relative}")
    return content


def _flat_comparisons(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize the four current-stage comparisons for shared evaluators."""

    records: list[dict[str, Any]] = []
    for raw in config["comparisons"]:
        record = dict(raw)
        candidate = str(record["candidate_id"])
        reference = str(record["reference_id"])
        expected = _comparison_id(candidate, reference)
        if str(record.get("id", expected)) != expected:
            raise ValueError(f"comparison id must be {expected}")
        record.update(
            {
                "id": expected,
                "candidate_id": candidate,
                "reference_id": reference,
                "block_id": str(record.get("block_id", "current_model")),
                "experiment_arm": str(
                    record.get("experiment_arm", "direct_model_comparison")
                ),
                "primary_atomic": bool(record.get("primary_atomic", False)),
            }
        )
        records.append(record)
    return records


def _model_selection_ids(config: Mapping[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return selected, engine-benchmark, and publication-benchmark IDs."""

    selection = config["model_selection"]
    baseline = str(selection.get("baseline", selection.get("selected_baseline", "")))
    engine = tuple(str(value) for value in selection["engine_major_benchmarks"])
    publication = tuple(str(value) for value in selection["publication_benchmarks"])
    return baseline, engine, publication


def _load_current_config(path: Path) -> tuple[dict[str, Any], bytes]:
    """Load and strictly validate the compact current-baseline declaration."""

    config, raw = _load_yaml(path)
    if (
        config.get("schema_version") != 1
        or config.get("model_id") != MODEL_ID
        or config.get("stage_id") != STAGE_ID
    ):
        raise ValueError("unexpected Model 02 current-baseline identity")
    required_sections = {
        "model_selection",
        "sources",
        "source_integrity",
        "evidence_sets",
        "variants",
        "variant_defaults",
        "comparisons",
        "evaluation",
        "archived_sensitivity_namespaces",
        "outputs",
    }
    missing_sections = required_sections.difference(config)
    if missing_sections:
        raise ValueError(
            "current-baseline config omits: " + ", ".join(sorted(missing_sections))
        )
    missing_sources = REQUIRED_SOURCE_KEYS.difference(config["sources"])
    if missing_sources:
        raise ValueError(
            "current-baseline sources omit: " + ", ".join(sorted(missing_sources))
        )
    missing_outputs = REQUIRED_OUTPUT_KEYS.difference(config["outputs"])
    if missing_outputs:
        raise ValueError(
            "current-baseline outputs omit: " + ", ".join(sorted(missing_outputs))
        )

    baseline, engine_benchmarks, publication_benchmarks = _model_selection_ids(config)
    if baseline != BASELINE_ID:
        raise ValueError(f"current baseline must be {BASELINE_ID}")
    if engine_benchmarks != ENGINE_BENCHMARK_IDS:
        raise ValueError("engine benchmarks must remain transition_only and partial_only")
    if publication_benchmarks != PUBLICATION_BENCHMARK_IDS:
        raise ValueError("publication benchmarks differ from the declared compact set")
    frozen = tuple(str(value) for value in config["model_selection"]["frozen_controls"])
    if frozen != FROZEN_CONTROL_IDS:
        raise ValueError("frozen controls differ from the three predecessor models")

    variants = {str(row["id"]): dict(row) for row in config["variants"]}
    expected_ids = {
        BASELINE_ID,
        EXPECTATIONS_SENSITIVITY_ID,
        *FROZEN_CONTROL_IDS,
    }
    if len(variants) != 5 or set(variants) != expected_ids:
        raise ValueError("current-baseline replay must declare exactly five variants")
    expected_engine_roles = {
        BASELINE_ID: "baseline",
        "transition_only": "major_benchmark",
        "partial_only": "major_benchmark",
        LEGACY_BASELINE_ID: "sensitivity",
        EXPECTATIONS_SENSITIVITY_ID: "sensitivity",
    }
    for variant_id, role in expected_engine_roles.items():
        if str(variants[variant_id].get("model_role", "")) != role:
            raise ValueError(f"wrong engine role for {variant_id}")

    evidence_sets = config["evidence_sets"]
    for variant_id, declaration in variants.items():
        set_id = str(declaration["evidence_set"])
        if set_id not in evidence_sets:
            raise ValueError(f"variant {variant_id} uses an unknown evidence set")
    baseline_set = evidence_sets[str(variants[BASELINE_ID]["evidence_set"])]
    expectation_set = evidence_sets[
        str(variants[EXPECTATIONS_SENSITIVITY_ID]["evidence_set"])
    ]
    if frozenset(str(value) for value in baseline_set["observation_models"]) != (
        BASELINE_OBSERVATION_MODELS
    ):
        raise ValueError("selected baseline evidence allowlist has changed")
    if frozenset(str(value) for value in expectation_set["observation_models"]) != (
        EXPECTATIONS_OBSERVATION_MODELS
    ):
        raise ValueError("expectations sensitivity differs by more than one model")

    comparisons = _flat_comparisons(config)
    expected_pairs = {
        (BASELINE_ID, "transition_only"),
        (BASELINE_ID, "partial_only"),
        (BASELINE_ID, LEGACY_BASELINE_ID),
        (EXPECTATIONS_SENSITIVITY_ID, BASELINE_ID),
    }
    observed_pairs = {
        (str(row["candidate_id"]), str(row["reference_id"])) for row in comparisons
    }
    if len(comparisons) != 4 or observed_pairs != expected_pairs:
        raise ValueError("current stage must contain the four declared direct comparisons")
    atomic_by_pair = {
        (str(row["candidate_id"]), str(row["reference_id"])): bool(
            row["primary_atomic"]
        )
        for row in comparisons
    }
    if any(
        atomic_by_pair[pair]
        for pair in expected_pairs
        if pair != (EXPECTATIONS_SENSITIVITY_ID, BASELINE_ID)
    ) or not atomic_by_pair[(EXPECTATIONS_SENSITIVITY_ID, BASELINE_ID)]:
        raise ValueError(
            "only the one-model inflation-expectations add-one may be primary atomic"
        )

    evaluation = config["evaluation"]
    if set(evaluation["metrics"]) != set(METRICS):
        raise ValueError("current-baseline metrics differ from the frozen scoring set")
    bootstrap = evaluation["paired_block_bootstrap"]
    if tuple(str(value) for value in bootstrap["samples"]) != (
        "full_sample",
        "excluding_2020_03_through_2020_05",
    ):
        raise ValueError("current-baseline bootstrap sample contract has changed")
    return config, raw


def _run_config(
    frozen: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Overlay the five variants without changing any estimation mechanics."""

    result = deepcopy(dict(frozen))
    defaults = dict(current["variant_defaults"])
    variants: list[dict[str, Any]] = []
    for raw in current["variants"]:
        declaration = dict(raw)
        if not bool(declaration.get("frozen_control", False)):
            declaration = {**defaults, **declaration}
        declaration.setdefault("retail", "baseline_nominal")
        declaration.pop("frozen_control", None)
        declaration.pop("publication_role", None)
        variants.append(declaration)
    result["stage_id"] = STAGE_ID
    result["model_selection"] = {
        "baseline": BASELINE_ID,
        "major_benchmarks": list(ENGINE_BENCHMARK_IDS),
        "role_vocabulary": ["baseline", "major_benchmark", "sensitivity"],
        "selection_status": str(current["model_selection"]["selection_status"]),
    }
    result["evidence_sets"] = deepcopy(current["evidence_sets"])
    result["variants"] = variants
    result["evaluation"] = {
        **deepcopy(dict(frozen["evaluation"])),
        "initialization_burn_in_months": int(
            current["evaluation"]["initialization_burn_in_months"]
        ),
        "information_stage_checkpoints": list(
            current["evaluation"]["information_stage_checkpoints"]
        ),
    }
    return result


def _subperiod_summary(
    evaluations: pd.DataFrame, *, initial_date: pd.Timestamp, burn_in_months: int
) -> pd.DataFrame:
    """Summarize the four standard diagnostic calendar samples."""

    month = pd.to_datetime(evaluations["reference_month"], errors="raise")
    masks = {
        "full_sample": pd.Series(True, index=evaluations.index),
        "excluding_2020_03_through_2020_05": ~month.between(
            "2020-03-01", "2020-05-01", inclusive="both"
        ),
        "through_2022_12": month.le("2022-12-01"),
        "from_2023_01": month.ge("2023-01-01"),
    }
    parts: list[pd.DataFrame] = []
    for sample_id, mask in masks.items():
        summary = _evaluation_summary(
            evaluations.loc[mask].copy(),
            initial_date=initial_date,
            burn_in_months=burn_in_months,
        )
        summary.insert(0, "evaluation_sample", sample_id)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True, sort=False)


def _publication_registry(
    engine_registry: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Attach explicit public roles without rewriting engine provenance."""

    declared = {str(row["id"]): dict(row) for row in config["variants"]}
    result = engine_registry.rename(columns={"model_role": "engine_model_role"}).copy()
    result["publication_role"] = result["variant_id"].map(
        lambda variant_id: str(
            declared[str(variant_id)].get(
                "publication_role", declared[str(variant_id)]["model_role"]
            )
        )
    )
    expected = {
        BASELINE_ID: "baseline",
        "transition_only": "major_benchmark",
        "partial_only": "major_benchmark",
        LEGACY_BASELINE_ID: "major_benchmark",
        EXPECTATIONS_SENSITIVITY_ID: "sensitivity",
    }
    observed = result.set_index("variant_id")["publication_role"].to_dict()
    if observed != expected:
        raise ValueError("publication roles differ from the selected model hierarchy")
    result["source_stage"] = STAGE_ID
    return result


def _sensitivity_catalog(
    feature_registry: pd.DataFrame,
    current_registry: pd.DataFrame,
    archived_namespaces: Mapping[str, Any],
) -> pd.DataFrame:
    """Catalog current variants and every retained historical namespace."""

    required = {"variant_id", "model_role"}
    if not required.issubset(feature_registry):
        raise ValueError("feature-revision registry lacks role metadata")
    historical = feature_registry.loc[
        feature_registry["model_role"].astype(str).eq("sensitivity")
    ].copy()
    historical = historical.loc[
        ~historical["variant_id"].astype(str).isin(
            {BASELINE_ID, EXPECTATIONS_SENSITIVITY_ID}
        )
    ]
    historical_catalog = pd.DataFrame(
        {
            "catalog_id": "variant:" + historical["variant_id"].astype(str),
            "entry_type": "model_variant",
            "variant_id": historical["variant_id"].astype(str),
            "publication_role": "sensitivity",
            "source_stage": "feature_revision",
            "status": "retained_historical_sensitivity",
            "results_path": "results/published/m02_soft_composite/feature_revision",
        }
    )
    current = current_registry.loc[
        current_registry["publication_role"].astype(str).eq("sensitivity")
    ]
    current_catalog = pd.DataFrame(
        {
            "catalog_id": "variant:" + current["variant_id"].astype(str),
            "entry_type": "model_variant",
            "variant_id": current["variant_id"].astype(str),
            "publication_role": "sensitivity",
            "source_stage": STAGE_ID,
            "status": "current_baseline_sensitivity",
            "results_path": "results/published/m02_soft_composite/current",
        }
    )
    archive_records: list[dict[str, object]] = []
    for stage_id, declaration in archived_namespaces.items():
        if not isinstance(declaration, Mapping):
            raise ValueError(f"archived sensitivity namespace is invalid: {stage_id}")
        archive_records.append(
            {
                "catalog_id": f"stage:{stage_id}",
                "entry_type": "archived_stage",
                "variant_id": "",
                "publication_role": "sensitivity",
                "source_stage": str(stage_id),
                "status": "retained_historical_sensitivity_namespace",
                "results_path": str(declaration["published_dir"]),
                "config_path": str(declaration["config"]),
                "manifest_path": str(declaration["manifest"]),
            }
        )
    archive_catalog = pd.DataFrame.from_records(archive_records)
    result = pd.concat(
        [current_catalog, historical_catalog, archive_catalog],
        ignore_index=True,
        sort=False,
    ).drop_duplicates("catalog_id", keep="first")
    return result.sort_values(
        ["entry_type", "source_stage", "catalog_id"], kind="stable"
    ).reset_index(drop=True)


def _publication_view(
    frame: pd.DataFrame,
    current_registry: pd.DataFrame,
    *,
    variant_column: str | None = None,
) -> pd.DataFrame:
    """Attach public roles while retaining any distinct engine-role metadata."""

    result = frame.copy()
    roles = current_registry.set_index("variant_id")["publication_role"].astype(str)
    if "model_role" in result:
        result = result.rename(columns={"model_role": "engine_model_role"})
    if variant_column is not None:
        result["publication_role"] = result[variant_column].astype(str).map(roles)
        if result["publication_role"].isna().any():
            raise ValueError("public table contains an undeclared variant role")
    for side in ("candidate", "reference"):
        column = f"{side}_id"
        if column in result:
            result[f"{side}_publication_role"] = result[column].astype(str).map(roles)
            if result[f"{side}_publication_role"].isna().any():
                raise ValueError("comparison contains an undeclared publication role")
    return result


def _file_declaration(root: Path, path: Path) -> dict[str, object]:
    """Describe one generated or implementation file for the manifest."""

    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256(path.read_bytes()),
        "bytes": path.stat().st_size,
    }


def _output_path(namespace: Path, filename: object) -> Path:
    """Resolve an output below its configured namespace."""

    base = namespace.resolve()
    path = (base / str(filename)).resolve()
    if path == base or base not in path.parents:
        raise ValueError(f"configured output leaves namespace: {filename}")
    return path


def build_current_baseline(
    *, project_root: Path, config_path: Path, replay_end: object | None = None
) -> dict[str, Path]:
    """Replay, verify, and publish the selected compact Model 02 hierarchy."""

    root = project_root.resolve()
    config_path = _project_path(root, config_path)
    config, config_bytes = _load_current_config(config_path)
    sources = config["sources"]

    feature_manifest_path = _project_path(root, sources["feature_revision_manifest"])
    feature_manifest_bytes = feature_manifest_path.read_bytes()
    integrity = config["source_integrity"]
    if sha256(feature_manifest_bytes) != str(
        integrity["feature_revision_manifest_sha256"]
    ):
        raise ValueError("configured feature-revision manifest hash mismatch")
    feature_manifest = json.loads(feature_manifest_bytes)
    if (
        feature_manifest.get("model_id") != MODEL_ID
        or feature_manifest.get("stage_id") != "feature_revision"
        or not bool(feature_manifest.get("baseline_invariance_passed"))
    ):
        raise ValueError("feature-revision manifest is not an invariant upstream stage")

    feature_config_path = _project_path(root, sources["feature_revision_config"])
    feature_config, feature_config_bytes = _load_yaml(feature_config_path)
    if sha256(feature_config_bytes) != str(
        integrity["feature_revision_config_sha256"]
    ):
        raise ValueError("configured feature-revision config hash mismatch")
    if feature_manifest.get("configuration") != feature_config_path.relative_to(root).as_posix():
        raise ValueError("feature-revision manifest names a different configuration")
    if feature_manifest.get("configuration_sha256") != sha256(feature_config_bytes):
        raise ValueError("feature-revision configuration hash mismatch")
    if feature_config.get("stage_id") != "feature_revision":
        raise ValueError("feature-revision configuration identity mismatch")

    base_path = _project_path(root, sources["base_filter_config"])
    base, base_bytes = _load_yaml(base_path)
    frozen_path = _project_path(root, sources["frozen_sensitivity_config"])
    frozen, frozen_bytes = _load_yaml(frozen_path)
    if base.get("model_id") != MODEL_ID or frozen.get("model_id") != MODEL_ID:
        raise ValueError("current-baseline upstream configuration identity mismatch")
    if feature_manifest.get("base_configuration_sha256") != sha256(base_bytes):
        raise ValueError("base filter differs from the feature-revision lineage")
    if feature_manifest.get("frozen_sensitivity_configuration_sha256") != sha256(
        frozen_bytes
    ):
        raise ValueError("frozen sensitivity config differs from feature-revision lineage")

    artifact_keys = sorted(
        REQUIRED_SOURCE_KEYS.difference(
            {
                "base_filter_config",
                "frozen_sensitivity_config",
                "feature_revision_config",
                "feature_revision_manifest",
            }
        )
    )
    source_bytes = {
        key: _verified_feature_bytes(root, sources[key], feature_manifest)
        for key in artifact_keys
    }
    for key, content in source_bytes.items():
        declared_key = f"{key}_sha256"
        if declared_key in integrity and sha256(content) != str(integrity[declared_key]):
            raise ValueError(f"configured source hash mismatch: {key}")

    selected_specs = feature_config.get("candidate_observation_models", {})
    missing_specs = set(SELECTED_CANDIDATE_MODELS).difference(selected_specs)
    if missing_specs:
        raise ValueError(
            "feature revision omits selected observation specs: "
            + ", ".join(sorted(missing_specs))
        )
    base_augmented = _merge_observation_models(
        base, {model_id: selected_specs[model_id] for model_id in SELECTED_CANDIDATE_MODELS}
    )
    run_config = _run_config(frozen, config)
    engine_registry = variant_registry_from_config(run_config)
    current_registry = _publication_registry(engine_registry, config)

    requested_end = pd.Timestamp(
        replay_end
        if replay_end is not None
        else feature_manifest["coverage"]["replay_end"]
    ).normalize()
    if requested_end != pd.Timestamp(feature_manifest["coverage"]["replay_end"]).normalize():
        raise ValueError("current baseline must use the locked feature-revision replay end")

    scores = pd.read_csv(BytesIO(source_bytes["score_features"]))
    components = pd.read_csv(BytesIO(source_bytes["defining_components"]))
    mapping = pd.read_csv(BytesIO(source_bytes["mapping_history"]))
    events = pd.read_csv(BytesIO(source_bytes["consolidated_event_universe"]))
    for column in (
        "release_date",
        "reference_date",
        "reference_month",
        "control_source_reference_date",
        "control_source_reference_month",
    ):
        if column in events:
            events[column] = pd.to_datetime(
                events[column], format="mixed", errors="raise"
            )
    prepared = prepare_observation_data(events, scores, base_augmented)
    result = run_inference_sensitivities(
        scores,
        mapping,
        prepared,
        components,
        base_augmented,
        run_config,
        replay_end=requested_end,
    )

    burn = int(config["evaluation"]["initialization_burn_in_months"])
    evaluation_summary = _evaluation_summary(
        result.evaluations, initial_date=result.initial_date, burn_in_months=burn
    )
    evaluation_subperiod = _subperiod_summary(
        result.evaluations, initial_date=result.initial_date, burn_in_months=burn
    )
    frames = {
        str(variant_id): frame.copy()
        for variant_id, frame in result.evaluations.groupby("filter_variant", sort=False)
    }
    eligible = _eligible_frames(
        frames, initial_date=result.initial_date, burn_in_months=burn
    )
    comparisons = _flat_comparisons(config)
    paired = _paired_comparisons(
        eligible, comparisons, event_audit=None, blocks={}
    )
    bootstrap_config = {
        "evaluation": {
            "information_stage_checkpoints": list(
                config["evaluation"]["information_stage_checkpoints"]
            ),
            "metrics": dict(config["evaluation"]["metrics"]),
            "paired_block_bootstrap": deepcopy(
                dict(config["evaluation"]["paired_block_bootstrap"])
            ),
        }
    }
    bootstrap = _paired_bootstrap(eligible, comparisons, bootstrap_config)

    frozen_evaluation = pd.read_csv(
        BytesIO(source_bytes["frozen_evaluation_rows"]), float_precision="round_trip"
    )
    frozen_events = pd.read_csv(
        BytesIO(source_bytes["frozen_event_update_audit"]),
        compression="gzip",
        float_precision="round_trip",
    )
    frozen_partial = pd.read_csv(
        BytesIO(source_bytes["frozen_partial_defining_audit"]),
        compression="gzip",
        float_precision="round_trip",
    )
    frozen_exact = pd.read_csv(
        BytesIO(source_bytes["frozen_exact_score_audit"]),
        float_precision="round_trip",
    )
    frozen_latest = pd.read_csv(
        BytesIO(source_bytes["frozen_latest_marginals"]),
        float_precision="round_trip",
    )
    invariance_records: list[dict[str, object]] = []
    for variant_id in FROZEN_CONTROL_IDS:
        invariance_records.extend(
            _invariance_rows(
                result.evaluations,
                frozen_evaluation,
                artifact="evaluation_rows",
                variant_column="filter_variant",
                variant_id=variant_id,
                key_columns=(
                    "reference_month",
                    "evaluation_checkpoint",
                    "availability_date",
                    "forecast_as_of_date",
                ),
            )
        )
        invariance_records.extend(
            _invariance_rows(
                result.partial_defining_audit,
                frozen_partial,
                artifact="partial_defining_audit",
                variant_column="variant_id",
                variant_id=variant_id,
                key_columns=("release_date", "event_id", "reference_month"),
            )
        )
        invariance_records.extend(
            _invariance_rows(
                result.exact_score_audit,
                frozen_exact,
                artifact="exact_score_audit",
                variant_column="variant_id",
                variant_id=variant_id,
                key_columns=("availability_date", "reference_month"),
            )
        )
        invariance_records.extend(
            _invariance_rows(
                result.latest_marginals,
                frozen_latest,
                artifact="latest_marginals",
                variant_column="variant_id",
                variant_id=variant_id,
                key_columns=("reference_month", "relative_month", "as_of_date"),
            )
        )
    invariance_records.extend(
        _invariance_rows(
            result.event_audit,
            frozen_events,
            artifact="event_update_audit",
            variant_column="variant_id",
            variant_id=LEGACY_BASELINE_ID,
            key_columns=(
                "release_date",
                "observation_model_id",
                "reference_month",
                "event_instance_id",
            ),
            ignored_columns=frozenset({"fit_id"}),
        )
    )
    invariance = pd.DataFrame.from_records(invariance_records)
    if invariance.empty or not bool(invariance["passed"].all()):
        raise ValueError("a frozen predecessor changed in the current-baseline replay")

    feature_registry = pd.read_csv(BytesIO(source_bytes["feature_revision_registry"]))
    feature_summary = pd.read_csv(BytesIO(source_bytes["feature_revision_summary"]))
    if feature_summary.empty or feature_summary["comparison_id"].nunique() != 23:
        raise ValueError("feature-revision summary does not contain the locked experiment")
    sensitivity_catalog = _sensitivity_catalog(
        feature_registry,
        current_registry,
        config["archived_sensitivity_namespaces"],
    )

    baseline_latest = result.latest_marginals.loc[
        result.latest_marginals["variant_id"].astype(str).eq(BASELINE_ID)
    ].copy()
    benchmark_latest = result.latest_marginals.loc[
        result.latest_marginals["variant_id"].astype(str).isin(
            PUBLICATION_BENCHMARK_IDS
        )
    ].copy()
    if baseline_latest.empty or set(benchmark_latest["variant_id"].astype(str)) != set(
        PUBLICATION_BENCHMARK_IDS
    ):
        raise ValueError("compact latest-marginal publication is incomplete")

    output = config["outputs"]
    processed_dir = _project_path(root, output["processed_dir"])
    published_dir = _project_path(root, output["published_dir"])
    processed_keys = (
        "evaluation_rows",
        "event_update_audit",
        "emission_fit_audit",
        "partial_defining_audit",
        "exact_score_audit",
        "transition_fit_audit",
        "latest_marginals",
        "baseline_invariance",
        "engine_registry",
    )
    published_keys = (
        "evaluation_summary",
        "evaluation_subperiod_summary",
        "paired_comparisons",
        "paired_block_bootstrap",
        "current_registry",
        "sensitivity_catalog",
        "baseline_latest_marginals",
        "benchmark_latest_marginals",
        "method_summary",
    )
    processed_paths = {
        key: _output_path(processed_dir, output[key]) for key in processed_keys
    }
    published_paths = {
        key: _output_path(published_dir, output[key]) for key in published_keys
    }
    processed_frames = {
        "evaluation_rows": result.evaluations,
        "event_update_audit": result.event_audit,
        "emission_fit_audit": result.emission_fit_audit,
        "partial_defining_audit": result.partial_defining_audit,
        "exact_score_audit": result.exact_score_audit,
        "transition_fit_audit": result.transition_fit_audit,
        "latest_marginals": result.latest_marginals,
        "baseline_invariance": invariance,
        "engine_registry": engine_registry,
    }
    published_frames = {
        "evaluation_summary": _publication_view(
            evaluation_summary, current_registry, variant_column="filter_variant"
        ),
        "evaluation_subperiod_summary": _publication_view(
            evaluation_subperiod, current_registry, variant_column="filter_variant"
        ),
        "paired_comparisons": _publication_view(paired, current_registry),
        "paired_block_bootstrap": _publication_view(bootstrap, current_registry),
        "current_registry": current_registry,
        "sensitivity_catalog": sensitivity_catalog,
        "baseline_latest_marginals": _publication_view(
            baseline_latest, current_registry, variant_column="variant_id"
        ),
        "benchmark_latest_marginals": _publication_view(
            benchmark_latest, current_registry, variant_column="variant_id"
        ),
    }
    for key, frame in processed_frames.items():
        _write_csv(frame, processed_paths[key])
    for key, frame in published_frames.items():
        _write_csv(frame, published_paths[key])

    method_summary = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "selected_baseline": BASELINE_ID,
        "predecessor_baseline": LEGACY_BASELINE_ID,
        "engine_major_benchmarks": list(ENGINE_BENCHMARK_IDS),
        "publication_benchmarks": list(PUBLICATION_BENCHMARK_IDS),
        "current_sensitivities": [EXPECTATIONS_SENSITIVITY_ID],
        "historical_sensitivity_count": int(
            (
                sensitivity_catalog["entry_type"].eq("model_variant")
                & sensitivity_catalog["source_stage"].eq("feature_revision")
            ).sum()
        ),
        "archived_sensitivity_stage_count": int(
            sensitivity_catalog["entry_type"].eq("archived_stage").sum()
        ),
        "selection_status": str(config["model_selection"]["selection_status"]),
        "selection_rationale": str(config["model_selection"]["rationale"]),
        "validation_scope": (
            "operational promotion after same-history development; not a fresh "
            "out-of-sample validation"
        ),
        "baseline_observation_models": sorted(BASELINE_OBSERVATION_MODELS),
        "baseline_mechanics": {
            "transition": "OLS VAR(1)",
            "non_defining_emission": "fixed Student-t with 7 degrees of freedom",
            "partial_defining_releases": True,
            "joint_state_months": 4,
            "causal_fit_cutoff": "strictly before each scored release",
        },
        "replay_start": result.initial_date.date().isoformat(),
        "replay_end": result.replay_end.date().isoformat(),
        "variant_count": len(engine_registry),
        "comparison_count": len(comparisons),
        "frozen_control_invariance_checks": len(invariance),
        "frozen_control_invariance_passed": True,
        "latest_as_of_date": pd.to_datetime(
            baseline_latest["as_of_date"], errors="raise"
        ).max().date().isoformat(),
        "credential_policy": "offline; no environment credential is read",
    }
    _write_json(method_summary, published_paths["method_summary"])

    generated_paths = [*processed_paths.values(), *published_paths.values()]
    generated = sorted(
        (_file_declaration(root, path) for path in generated_paths),
        key=lambda row: str(row["path"]),
    )
    implementation = sorted(
        (
            _file_declaration(root, _project_path(root, relative))
            for relative in IMPLEMENTATION_FILES
        ),
        key=lambda row: str(row["path"]),
    )
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config_path.relative_to(root).as_posix(),
        "configuration_sha256": sha256(config_bytes),
        "base_configuration_sha256": sha256(base_bytes),
        "frozen_sensitivity_configuration_sha256": sha256(frozen_bytes),
        "upstream_manifest": {
            "path": feature_manifest_path.relative_to(root).as_posix(),
            "sha256": sha256(feature_manifest_bytes),
        },
        "source_files": [
            {
                "path": str(sources[key]),
                "sha256": sha256(content),
                "bytes": len(content),
            }
            for key, content in sorted(source_bytes.items())
        ],
        "implementation_files": implementation,
        "generated_files": generated,
        "model_selection": {
            "baseline": BASELINE_ID,
            "predecessor_baseline": LEGACY_BASELINE_ID,
            "engine_major_benchmarks": list(ENGINE_BENCHMARK_IDS),
            "publication_benchmarks": list(PUBLICATION_BENCHMARK_IDS),
            "current_sensitivities": [EXPECTATIONS_SENSITIVITY_ID],
            "selection_status": str(config["model_selection"]["selection_status"]),
        },
        "coverage": {
            "replay_start": result.initial_date.date().isoformat(),
            "replay_end": result.replay_end.date().isoformat(),
            "variants": len(engine_registry),
            "comparisons": len(comparisons),
            "evaluation_rows": len(result.evaluations),
            "event_audit_rows": len(result.event_audit),
            "frozen_control_invariance_checks": len(invariance),
        },
        "frozen_control_invariance_passed": True,
        "credential_policy": "offline; no environment credential is read",
    }
    manifest_path = _project_path(root, output["manifest"])
    _write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "latest": published_paths["baseline_latest_marginals"],
        "benchmarks": published_paths["benchmark_latest_marginals"],
        "evaluation": published_paths["evaluation_summary"],
        "comparisons": published_paths["paired_block_bootstrap"],
        "registry": published_paths["current_registry"],
        "sensitivities": published_paths["sensitivity_catalog"],
    }


def _parse_args() -> argparse.Namespace:
    """Parse the offline current-baseline build arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replay-end", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the current-baseline build and print its public artifact paths."""

    args = _parse_args()
    outputs = build_current_baseline(
        project_root=args.project_root,
        config_path=args.config,
        replay_end=args.replay_end,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
