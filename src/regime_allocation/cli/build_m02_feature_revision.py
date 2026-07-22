"""Build the causal Model 02 feature-revision experiment.

This offline command makes no network request and never reads credentials. It
reuses verified point-in-time artifacts to test four predeclared changes while
keeping the selected ``student_t_7_combined`` model as a frozen control:

* remove aggregate housing and the lagged JOLTS likelihood from revised arms;
* compare import prices directly with intermediate-material input costs;
* attribute redesigned consumer and business responses with add/remove tests;
* compare independent continued claims with a conditional ICSA--CCSA
  factorization that preserves both releases' actual target months.

Every experimental arm retains fixed-nu=7 Student-t emissions, the OLS VAR(1),
partial defining releases, expanding causal fits, and the four-month joint
Gaussian state. Results are development evidence; this stage cannot promote a
new production baseline on the same history used to choose the features.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from regime_allocation.cli.build_m02_evidence_experiment_inference import (
    _load_manifest,
    _merge_observation_models,
    _profile_coverage,
)
from regime_allocation.cli.build_m02_existing_block_attribution import (
    METRICS,
    _comparison_id,
    _eligible_frames,
    _invariance_rows,
    _paired_bootstrap,
    _paired_comparisons,
)
from regime_allocation.cli.build_m02_inference_sensitivities import (
    _evaluation_summary,
    _latest_joint_state_rows,
    _load_yaml,
    _namespace_path,
    _project_path,
    _verified_bytes,
    _write_csv,
    _write_json,
)
from regime_allocation.data.dataset_acquisition import sha256
from regime_allocation.features.m02_feature_revision import (
    BUSINESS_ACTIVITY_RESPONSE,
    BUSINESS_PIPELINE_RESPONSE,
    BUSINESS_REVISION_BLOCK,
    CONDITIONAL_CLAIMS_BLOCK,
    CONDITIONAL_CLAIMS_CONTROL,
    build_business_investment_revision_events,
    build_conditional_continued_claims_events,
)
from regime_allocation.models.m02_soft_composite.inference_sensitivities import (
    run_inference_sensitivities,
    variant_registry_from_config,
)
from regime_allocation.models.m02_soft_composite.walkforward import (
    prepare_observation_data,
)


MODEL_ID = "m02_soft_composite"
STAGE_ID = "feature_revision"
BASELINE_ID = "student_t_7_combined"
CONTROL_IDS = ("transition_only", "partial_only", BASELINE_ID)
DEFAULT_CONFIG = Path("configs/models/m02_feature_revision.yaml")

IMPLEMENTATION_FILES = (
    "src/regime_allocation/cli/build_m02_feature_revision.py",
    "src/regime_allocation/features/m02_feature_revision.py",
    "src/regime_allocation/cli/build_m02_existing_block_attribution.py",
    "src/regime_allocation/cli/build_m02_evidence_experiment_inference.py",
    "src/regime_allocation/cli/build_m02_inference_sensitivities.py",
    "src/regime_allocation/models/m02_soft_composite/gaussian_emissions.py",
    "src/regime_allocation/models/m02_soft_composite/inference_sensitivities.py",
    "src/regime_allocation/models/m02_soft_composite/joint_filter.py",
    "src/regime_allocation/models/m02_soft_composite/partial_defining.py",
    "src/regime_allocation/models/m02_soft_composite/robust_emissions.py",
    "src/regime_allocation/models/m02_soft_composite/var_transition.py",
    "src/regime_allocation/models/m02_soft_composite/walkforward.py",
)


def _flat_comparisons(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family, declarations in config["comparisons"].items():
        for raw in declarations:
            record = dict(raw)
            expected = _comparison_id(
                str(record["candidate_id"]), str(record["reference_id"])
            )
            if str(record["id"]) != expected:
                raise ValueError(f"comparison id must be {expected}")
            arm = str(record["arm"])
            default_atomic = arm in {
                "add_one",
                "legacy_add_one",
                "independent_add_one",
            }
            record.update(
                {
                    "family": str(family),
                    "block_id": str(record.get("response", family)),
                    "experiment_arm": (
                        "leave_one_out" if arm == "leave_one_out" else "add_one"
                    ),
                    "primary_atomic": bool(
                        record.get("primary_atomic", default_atomic)
                    ),
                }
            )
            records.append(record)
    return records


def _load_feature_config(path: Path) -> tuple[dict[str, Any], bytes]:
    config, raw = _load_yaml(path)
    if (
        config.get("schema_version") != 1
        or config.get("model_id") != MODEL_ID
        or config.get("stage_id") != STAGE_ID
    ):
        raise ValueError("unexpected Model 02 feature-revision identity")
    required = {
        "model_selection",
        "sources",
        "candidate_observation_models",
        "evidence_sets",
        "variants",
        "variant_defaults",
        "comparisons",
        "evaluation",
        "outputs",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError("feature-revision config omits: " + ", ".join(sorted(missing)))
    variants = {str(row["id"]): row for row in config["variants"]}
    if len(variants) != len(config["variants"]) or set(CONTROL_IDS).difference(variants):
        raise ValueError("feature-revision variants are duplicated or omit controls")
    if len(variants) != 26:
        raise ValueError("feature revision must contain the predeclared 26 variants")
    evidence_sets = config["evidence_sets"]
    for variant_id, declaration in variants.items():
        if str(declaration["evidence_set"]) not in evidence_sets:
            raise ValueError(f"variant {variant_id} names an unknown evidence set")
    comparisons = _flat_comparisons(config)
    if len(comparisons) != 23:
        raise ValueError("feature revision must contain 23 predeclared comparisons")
    for declaration in comparisons:
        if (
            declaration["candidate_id"] not in variants
            or declaration["reference_id"] not in variants
            or declaration["candidate_id"] == declaration["reference_id"]
        ):
            raise ValueError("comparison references an unknown or identical variant")
    declared_families = config["evaluation"]["multiplicity"]["families"]
    for family, members in declared_families.items():
        observed = {row["id"] for row in comparisons if row["family"] == family}
        if set(members) != observed:
            raise ValueError(f"Holm family differs from comparisons for {family}")
    conditional = config["candidate_observation_models"].get(
        "weekly_continued_claims_conditional", {}
    )
    if (
        conditional.get("event_block") != CONDITIONAL_CLAIMS_BLOCK
        or list(conditional.get("controls", ())) != [CONDITIONAL_CLAIMS_CONTROL]
    ):
        raise ValueError("conditional claims spec differs from the exact pairing contract")
    return config, raw


def _run_config(
    frozen: Mapping[str, Any], experiment: Mapping[str, Any]
) -> dict[str, Any]:
    """Overlay variants/evidence membership without changing model mechanics."""

    result = deepcopy(dict(frozen))
    defaults = dict(experiment["variant_defaults"])
    variants: list[dict[str, Any]] = []
    for raw in experiment["variants"]:
        declaration = dict(raw)
        if not bool(declaration.get("frozen_control", False)):
            declaration = {**defaults, **declaration}
        declaration.setdefault("model_role", "sensitivity")
        declaration.setdefault("retail", "baseline_nominal")
        declaration.pop("family", None)
        declaration.pop("frozen_control", None)
        variants.append(declaration)
    result["stage_id"] = STAGE_ID
    result["model_selection"] = {
        "baseline": BASELINE_ID,
        "major_benchmarks": ["transition_only", "partial_only"],
        "role_vocabulary": ["baseline", "major_benchmark", "sensitivity"],
        "selection_status": "development_feature_revision_no_automatic_promotion",
    }
    result["evidence_sets"] = deepcopy(experiment["evidence_sets"])
    result["variants"] = variants
    result["evaluation"] = {
        **deepcopy(dict(frozen["evaluation"])),
        "initialization_burn_in_months": int(
            experiment["evaluation"]["initialization_burn_in_months"]
        ),
        "information_stage_checkpoints": list(
            experiment["evaluation"]["information_stage_checkpoints"]
        ),
    }
    return result


def _metric_directions(config: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in ("primary", "confirmatory", "secondary", "descriptive"):
        result.update(
            {str(key): str(value) for key, value in config["evaluation"]["metrics"][group].items()}
        )
    if set(result) != set(METRICS):
        raise ValueError("feature revision metrics differ from the frozen scoring set")
    return result


def _bootstrap_config(config: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = config["evaluation"]["paired_block_bootstrap"]
    return {
        "evaluation": {
            "information_stage_checkpoints": list(
                config["evaluation"]["information_stage_checkpoints"]
            ),
            "metrics": _metric_directions(config),
            "paired_block_bootstrap": {
                "random_seed": int(bootstrap["random_seed"]),
                "replications": int(bootstrap["replications"]),
                "block_length_months": int(bootstrap["block_length_months"]),
                "confidence_level": float(bootstrap["confidence_level"]),
                "samples": [
                    "full_sample",
                    "excluding_2020_03_through_2020_05",
                ],
                "holm_families": deepcopy(
                    config["evaluation"]["multiplicity"]["families"]
                ),
            },
        }
    }


def _subperiod_summary(
    evaluations: pd.DataFrame, *, initial_date: pd.Timestamp, burn_in_months: int
) -> pd.DataFrame:
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
    for sample, mask in masks.items():
        summary = _evaluation_summary(
            evaluations.loc[mask].copy(),
            initial_date=initial_date,
            burn_in_months=burn_in_months,
        )
        summary.insert(0, "evaluation_sample", sample)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True, sort=False)


def _feature_summary(
    bootstrap: pd.DataFrame, comparisons: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> pd.DataFrame:
    primary = str(config["evaluation"]["primary_checkpoint"])
    selected = bootstrap.loc[
        bootstrap["evaluation_sample"].eq("full_sample")
        & bootstrap["evaluation_checkpoint"].eq(primary)
    ].copy()
    metadata = pd.DataFrame.from_records(
        [
            {
                "comparison_id": row["id"],
                "family": row["family"],
                "declared_arm": row["arm"],
                "response": row.get("response", ""),
            }
            for row in comparisons
        ]
    )
    return selected.merge(metadata, on="comparison_id", how="left", validate="many_to_one").sort_values(
        ["family", "comparison_id", "metric"], kind="stable"
    )


def _file_declaration(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "sha256": sha256(path.read_bytes()),
        "bytes": path.stat().st_size,
    }


def build_feature_revision(
    *, project_root: Path, config_path: Path, replay_end: object | None = None
) -> dict[str, Path]:
    """Verify inputs, replay all variants, and publish revision artifacts."""

    root = project_root.resolve()
    config_path = _project_path(root, config_path)
    config, config_bytes = _load_feature_config(config_path)
    sources = config["sources"]

    base_path = _project_path(root, sources["base_filter_config"])
    base, base_bytes = _load_yaml(base_path)
    frozen_config_path = _project_path(root, sources["baseline_sensitivity_config"])
    frozen_config, frozen_config_bytes = _load_yaml(frozen_config_path)
    evidence_config_path = _project_path(root, sources["evidence_experiment_config"])
    evidence_config, evidence_config_bytes = _load_yaml(evidence_config_path)
    if base.get("model_id") != MODEL_ID or frozen_config.get("model_id") != MODEL_ID:
        raise ValueError("feature revision has an upstream model identity mismatch")

    reused_continued = {
        "weekly_continued_claims": evidence_config["candidate_observation_models"][
            "weekly_continued_claims"
        ]
    }
    base_augmented = _merge_observation_models(base, reused_continued)
    base_augmented = _merge_observation_models(
        base_augmented, config["candidate_observation_models"]
    )
    run_config = _run_config(frozen_config, config)
    registry = variant_registry_from_config(run_config)

    manifest_specs = {
        "score": (sources["score_manifest"], None),
        "mapping": (sources["mapping_manifest"], "m02_probability_map"),
        "extended": (
            sources["extended_evidence_manifest"],
            "m02_evidence_block_experiments",
        ),
        "evidence_inference": (
            sources["evidence_experiment_inference_manifest"],
            "m02_evidence_block_inference_experiments",
        ),
        "retail": (
            sources["retail_real_decomposition_manifest"],
            "m02_retail_real_decomposition_sensitivity",
        ),
        "frozen": (
            sources["frozen_baseline_manifest"],
            "partial_defining_and_robustness_sensitivities",
        ),
    }
    manifests: dict[str, dict[str, Any]] = {}
    manifest_bytes: dict[str, bytes] = {}
    manifest_paths: dict[str, Path] = {}
    for name, (relative, stage_id) in manifest_specs.items():
        path = _project_path(root, relative)
        manifest_paths[name] = path
        manifests[name], manifest_bytes[name] = _load_manifest(path, stage_id=stage_id)

    frozen_manifest = manifests["frozen"]
    if frozen_manifest.get("base_configuration_sha256") != sha256(base_bytes):
        raise ValueError("base filter differs from the frozen selected replay")
    if frozen_manifest.get("configuration_sha256") != sha256(frozen_config_bytes):
        raise ValueError("frozen sensitivity configuration hash mismatch")
    if manifests["evidence_inference"].get("configuration_sha256") != sha256(
        evidence_config_bytes
    ):
        raise ValueError("independent CCSA specification config hash mismatch")
    requested_end = pd.Timestamp(
        replay_end if replay_end is not None else base.get("calendar", {}).get("replay_end")
    ).normalize()
    if requested_end != pd.Timestamp(frozen_manifest["coverage"]["replay_end"]).normalize():
        raise ValueError("feature revision must use the frozen replay end")

    source_contract = {
        "score_features": "score",
        "defining_components": "score",
        "mapping_history": "mapping",
        "extended_evidence_events": "extended",
        "retail_real_decomposition_events": "retail",
        "frozen_baseline_evaluation_rows": "frozen",
        "frozen_baseline_event_update_audit": "frozen",
        "frozen_baseline_partial_defining_audit": "frozen",
        "frozen_baseline_exact_score_audit": "frozen",
        "frozen_baseline_latest_marginals": "frozen",
    }
    source_bytes: dict[str, bytes] = {}
    for source_key, manifest_name in source_contract.items():
        source_bytes[source_key] = _verified_bytes(
            root, str(sources[source_key]), manifests[manifest_name]
        )

    scores = pd.read_csv(BytesIO(source_bytes["score_features"]))
    components = pd.read_csv(BytesIO(source_bytes["defining_components"]))
    mapping = pd.read_csv(BytesIO(source_bytes["mapping_history"]))
    extended = pd.read_csv(BytesIO(source_bytes["extended_evidence_events"]))
    retail = pd.read_csv(BytesIO(source_bytes["retail_real_decomposition_events"]))
    conditional_events, claims_audit = build_conditional_continued_claims_events(extended)
    business_events, business_audit = build_business_investment_revision_events(
        extended
    )
    revised_events = pd.concat(
        [extended, retail, conditional_events, business_events],
        ignore_index=True,
        sort=False,
    )
    prepared = prepare_observation_data(revised_events, scores, base_augmented)
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
    comparison_declarations = _flat_comparisons(config)
    paired = _paired_comparisons(
        eligible,
        comparison_declarations,
        event_audit=None,
        blocks={},
    )
    bootstrap = _paired_bootstrap(
        eligible,
        comparison_declarations,
        _bootstrap_config(config),
        family_override=config["evaluation"]["multiplicity"]["families"],
    )
    feature_summary = _feature_summary(bootstrap, comparison_declarations, config)
    coverage = _profile_coverage(
        prepared, result.event_audit, registry, run_config["evidence_sets"]
    )

    frozen_evaluation = pd.read_csv(
        BytesIO(source_bytes["frozen_baseline_evaluation_rows"]),
        float_precision="round_trip",
    )
    frozen_events = pd.read_csv(
        BytesIO(source_bytes["frozen_baseline_event_update_audit"]),
        compression="gzip",
        float_precision="round_trip",
    )
    frozen_partial = pd.read_csv(
        BytesIO(source_bytes["frozen_baseline_partial_defining_audit"]),
        compression="gzip",
        float_precision="round_trip",
    )
    frozen_exact = pd.read_csv(
        BytesIO(source_bytes["frozen_baseline_exact_score_audit"]),
        float_precision="round_trip",
    )
    frozen_latest = pd.read_csv(
        BytesIO(source_bytes["frozen_baseline_latest_marginals"]),
        float_precision="round_trip",
    )
    invariance_records: list[dict[str, object]] = []
    for variant_id in CONTROL_IDS:
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
            variant_id=BASELINE_ID,
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
        raise ValueError("a frozen control changed during feature revision")

    output = config["outputs"]
    processed = _project_path(root, output["processed_dir"])
    published = _project_path(root, output["published_dir"])
    processed_paths = {
        key: _namespace_path(processed, output[key])
        for key in (
            "revised_events",
            "claims_pairing_audit",
            "business_feature_audit",
            "profile_coverage",
            "evaluation_rows",
            "evaluation_summary",
            "evaluation_subperiod_summary",
            "paired_comparisons",
            "paired_block_bootstrap",
            "feature_attribution_summary",
            "event_update_audit",
            "emission_fit_audit",
            "event_weight_audit",
            "partial_defining_audit",
            "exact_score_audit",
            "transition_fit_audit",
            "latest_marginals",
            "model_registry",
            "baseline_invariance",
        )
    }
    public_keys = [key for key in output if key.startswith("public_")]
    public_paths = {
        key: _namespace_path(published, output[key]) for key in public_keys
    }
    claims_frame = pd.DataFrame([asdict(claims_audit)])
    business_frame = pd.DataFrame([asdict(business_audit)])
    event_weight = result.event_audit.loc[
        result.event_audit["update_status"].eq("applied"),
        [
            column
            for column in result.event_audit.columns
            if column
            in {
                "release_date",
                "variant_id",
                "observation_model_id",
                "event_instance_id",
                "reference_month",
                "observed_responses",
                "event_weight",
                "raw_event_weight",
                "mahalanobis_squared",
            }
        ],
    ].copy()
    processed_frames = {
        "revised_events": revised_events,
        "claims_pairing_audit": claims_frame,
        "business_feature_audit": business_frame,
        "profile_coverage": coverage,
        "evaluation_rows": result.evaluations,
        "evaluation_summary": evaluation_summary,
        "evaluation_subperiod_summary": evaluation_subperiod,
        "paired_comparisons": paired,
        "paired_block_bootstrap": bootstrap,
        "feature_attribution_summary": feature_summary,
        "event_update_audit": result.event_audit,
        "emission_fit_audit": result.emission_fit_audit,
        "event_weight_audit": event_weight,
        "partial_defining_audit": result.partial_defining_audit,
        "exact_score_audit": result.exact_score_audit,
        "transition_fit_audit": result.transition_fit_audit,
        "latest_marginals": result.latest_marginals,
        "model_registry": registry,
        "baseline_invariance": invariance,
    }
    for key, frame in processed_frames.items():
        _write_csv(frame, processed_paths[key])
    public_frames = {
        "public_evaluation_summary": evaluation_summary,
        "public_evaluation_subperiod_summary": evaluation_subperiod,
        "public_paired_comparisons": paired,
        "public_paired_block_bootstrap": bootstrap,
        "public_feature_attribution_summary": feature_summary,
        "public_claims_pairing_audit": claims_frame,
        "public_business_feature_audit": business_frame,
        "public_profile_coverage": coverage,
        "public_latest_marginals": result.latest_marginals,
        "public_model_registry": registry,
        "public_baseline_invariance": invariance,
    }
    for key, frame in public_frames.items():
        _write_csv(frame, public_paths[key])

    method_summary = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "stage_id": STAGE_ID,
        "selected_baseline_changed": False,
        "predeclared_combined_candidate": config["model_selection"][
            "predeclared_combined_candidate"
        ],
        "structural_exclusions": config["model_selection"]["structural_exclusions"],
        "claims_factorization": (
            "p(ICSA_t | Z[q_I]) * p(CCSA_t | Z[q_C], ICSA_t)"
        ),
        "claims_pairing": asdict(claims_audit),
        "business_feature_rotation": {
            "event_block": BUSINESS_REVISION_BLOCK,
            "activity_response": BUSINESS_ACTIVITY_RESPONSE,
            "pipeline_response": BUSINESS_PIPELINE_RESPONSE,
            "formula": "100 * Delta log(core orders / core shipments)",
            "audit": asdict(business_audit),
        },
        "variant_count": len(registry),
        "comparison_count": len(comparison_declarations),
        "primary_checkpoint": config["evaluation"]["primary_checkpoint"],
        "bootstrap": config["evaluation"]["paired_block_bootstrap"],
        "multiplicity": config["evaluation"]["multiplicity"],
        "promotion_rules": config["promotion_rules"],
        "baseline_invariance_checks": len(invariance),
        "baseline_invariance_passed": True,
        "latest_joint_gaussian_states": _latest_joint_state_rows(result),
        "credential_policy": "offline; no environment credential is read",
    }
    _write_json(method_summary, public_paths["public_method_summary"])

    generated_paths = [*processed_paths.values(), *public_paths.values()]
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
        "frozen_sensitivity_configuration_sha256": sha256(frozen_config_bytes),
        "upstream_manifests": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(manifest_bytes[name]),
            }
            for name, path in manifest_paths.items()
        ],
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
        "coverage": {
            "replay_start": result.initial_date.date().isoformat(),
            "replay_end": result.replay_end.date().isoformat(),
            "variants": len(registry),
            "comparisons": len(comparison_declarations),
            "evaluation_rows": len(result.evaluations),
            "event_audit_rows": len(result.event_audit),
            "invariance_checks": len(invariance),
        },
        "claims_pairing": asdict(claims_audit),
        "business_feature_rotation": {
            "event_block": BUSINESS_REVISION_BLOCK,
            "activity_response": BUSINESS_ACTIVITY_RESPONSE,
            "pipeline_response": BUSINESS_PIPELINE_RESPONSE,
            "audit": asdict(business_audit),
        },
        "baseline_invariance_passed": True,
        "model_selection": {
            "selected_baseline": BASELINE_ID,
            "selected_baseline_changed": False,
            "predeclared_combined_candidate": config["model_selection"][
                "predeclared_combined_candidate"
            ],
            "status": config["model_selection"]["selection_status"],
        },
        "credential_policy": "offline; no environment credential is read",
    }
    manifest_path = _project_path(root, output["manifest"])
    _write_json(manifest, manifest_path)
    return {
        "manifest": manifest_path,
        "feature_summary": public_paths["public_feature_attribution_summary"],
        "evaluation": public_paths["public_evaluation_summary"],
        "bootstrap": public_paths["public_paired_block_bootstrap"],
        "claims_pairing": public_paths["public_claims_pairing_audit"],
        "invariance": public_paths["public_baseline_invariance"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replay-end", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = build_feature_revision(
        project_root=args.project_root,
        config_path=args.config,
        replay_end=args.replay_end,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
