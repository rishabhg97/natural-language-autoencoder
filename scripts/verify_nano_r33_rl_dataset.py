#!/usr/bin/env python3
"""Strictly verify train-only R33 RL data and its source lineage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_nano_r33_rl_dataset import (  # noqa: E402
    INJECT_PLACEHOLDER,
    SCHEMA_VERSION,
    sha256_file,
    sidecar_path_for,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _split_sections(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = manifest.get("splits")
    sections = value if isinstance(value, dict) else {
        name: manifest.get(name) or {}
        for name in ("train", "validation", "test")
    }
    return {
        name: sections.get(name) if isinstance(sections.get(name), dict) else {}
        for name in ("train", "validation", "test")
    }


def _values(section: dict[str, Any], *names: str) -> set[str]:
    for name in names:
        value = section.get(name)
        if isinstance(value, list):
            return {str(item) for item in value}
    return set()


def _provenance_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    if row.get("sample_uuid") not in {None, ""}:
        return ("uuid", str(row["sample_uuid"]))
    if row.get("doc_id") not in {None, ""} and row.get("token_position") is not None:
        return ("position", str(row["doc_id"]), int(row["token_position"]))
    if row.get("doc_id") not in {None, ""} and row.get("n_raw_tokens") is not None:
        return ("raw_tokens", str(row["doc_id"]), int(row["n_raw_tokens"]))
    return None


def _load_dataset_sidecar(
    dataset_path: Path,
    rows: int,
    expected_d_model: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    path = sidecar_path_for(dataset_path)
    if not path.is_file():
        return None, [f"missing:{path}"]
    try:
        metadata = yaml.safe_load(path.read_text())
    except Exception as exc:
        return None, [f"parse:{type(exc).__name__}:{exc}"]
    errors: list[str] = []
    if not isinstance(metadata, dict):
        return None, ["not_mapping"]
    if metadata.get("kind") != "nla_dataset":
        errors.append("kind")
    if metadata.get("schema_version") != 1:
        errors.append("schema_version")
    if metadata.get("stage") != "rl":
        errors.append("stage")
    if metadata.get("row_count") != rows:
        errors.append("row_count")
    extraction = metadata.get("extraction")
    if not isinstance(extraction, dict) or not isinstance(extraction.get("d_model"), int):
        errors.append("extraction.d_model")
    elif extraction["d_model"] != expected_d_model:
        errors.append("extraction.d_model_mismatch")
    if not isinstance(extraction, dict) or not isinstance(extraction.get("layer_index"), int):
        errors.append("extraction.layer_index")
    tokens = metadata.get("tokens")
    required_tokens = {
        "injection_char",
        "injection_token_id",
        "injection_left_neighbor_id",
        "injection_right_neighbor_id",
    }
    if not isinstance(tokens, dict) or any(tokens.get(key) is None for key in required_tokens):
        errors.append("tokens")
    templates = metadata.get("prompt_templates")
    actor_template = templates.get("actor") if isinstance(templates, dict) else None
    if not isinstance(actor_template, str) or actor_template.count("{injection_char}") != 1:
        errors.append("prompt_templates.actor")
    return metadata, errors


def _canonical_prompt_content(metadata: dict[str, Any] | None) -> str | None:
    if metadata is None:
        return None
    templates = metadata.get("prompt_templates")
    actor_template = templates.get("actor") if isinstance(templates, dict) else None
    if not isinstance(actor_template, str) or actor_template.count("{injection_char}") != 1:
        return None
    return actor_template.format(injection_char=INJECT_PLACEHOLDER)


def _actor_prompt_content(value: Any) -> str | None:
    if not isinstance(value, list) or len(value) != 1:
        return None
    message = value[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    content = message.get("content")
    return content if isinstance(content, str) and content.strip() else None


def verify_dataset(
    *,
    dataset: str | Path,
    split_manifest: str | Path,
    content_family_manifest: str | Path | None = None,
    content_family_coverage: str | Path | None = None,
    expected_rows: int | None = None,
    expected_d_model: int = 2_688,
    batch_size: int = 4_096,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    manifest_path = Path(split_manifest)
    manifest = json.loads(manifest_path.read_text())
    sections = _split_sections(manifest)
    train_docs = _values(sections["train"], "docs", "doc_ids")
    heldout_docs = _values(sections["validation"], "docs", "doc_ids") | _values(
        sections["test"], "docs", "doc_ids"
    )
    train_units = _values(
        sections["train"],
        "split_unit_ids",
        "component_ids",
        "components",
        "split_units",
    )
    heldout_units = _values(
        sections["validation"],
        "split_unit_ids",
        "component_ids",
        "components",
        "split_units",
    ) | _values(
        sections["test"],
        "split_unit_ids",
        "component_ids",
        "components",
        "split_units",
    )
    family_manifest_path = (
        Path(content_family_manifest) if content_family_manifest is not None else None
    )
    family_coverage_path = (
        Path(content_family_coverage) if content_family_coverage is not None else None
    )
    family_contract_errors: list[str] = []
    family_assignments: dict[str, str] = {}
    heldout_family_ids: set[str] = set()
    heldout_family_docs: set[str] = set()
    family_manifest_hash = None
    family_coverage_hash = None
    if family_manifest_path is None or family_coverage_path is None:
        family_contract_errors.append("family_contract_paths_missing")
    if family_manifest_path is not None and family_manifest_path.is_file():
        family_manifest_doc = json.loads(family_manifest_path.read_text())
        if family_manifest_doc.get("schema_version") != "nano_content_family_manifest.v1":
            family_contract_errors.append("family_manifest_schema")
        family_assignments = {
            str(doc_id): str(family_id)
            for doc_id, family_id in (
                family_manifest_doc.get("doc_assignments") or {}
            ).items()
        }
        family_manifest_hash = sha256_file(family_manifest_path)
    elif family_manifest_path is not None:
        family_contract_errors.append("family_manifest_missing")
    if family_coverage_path is not None and family_coverage_path.is_file():
        family_coverage_doc = json.loads(family_coverage_path.read_text())
        if family_coverage_doc.get("schema_version") != (
            "nano_content_family_exposure_report.v1"
        ):
            family_contract_errors.append("family_coverage_schema")
        for split in ("validation", "test"):
            section = (family_coverage_doc.get("splits") or {}).get(split) or {}
            heldout_family_ids.update(
                str(value) for value in section.get("eligible_family_ids") or []
            )
            heldout_family_docs.update(
                str(value) for value in section.get("eligible_doc_ids") or []
            )
        family_coverage_hash = sha256_file(family_coverage_path)
    elif family_coverage_path is not None:
        family_contract_errors.append("family_coverage_missing")
    derived_manifest_units = False
    if not train_units and family_assignments and train_docs:
        missing_train_assignments = train_docs - family_assignments.keys()
        missing_heldout_assignments = heldout_docs - family_assignments.keys()
        if missing_train_assignments or missing_heldout_assignments:
            family_contract_errors.append("family_assignments_incomplete")
        else:
            train_units = {family_assignments[doc_id] for doc_id in train_docs}
            heldout_units = {
                family_assignments[doc_id] for doc_id in heldout_docs
            }
            derived_manifest_units = True
    manifest_errors: list[str] = []
    if not train_docs:
        manifest_errors.append("train_docs_missing")
    if not _values(sections["validation"], "docs", "doc_ids"):
        manifest_errors.append("validation_docs_missing")
    if not _values(sections["test"], "docs", "doc_ids"):
        manifest_errors.append("test_docs_missing")
    if not train_units:
        manifest_errors.append("train_split_units_missing")
    attested_overlap = (manifest.get("content_verification") or {}).get(
        "content_cross_split_overlap_count"
    )
    if attested_overlap != 0:
        manifest_errors.append("content_overlap_attestation_missing_or_nonzero")

    parquet = pq.ParquetFile(dataset_path)
    schema_names = set(parquet.schema_arrow.names)
    metadata = {
        key.decode(errors="replace"): value.decode(errors="replace")
        for key, value in (parquet.schema_arrow.metadata or {}).items()
    }
    required = {
        "doc_id",
        "split_unit_id",
        "content_family_id",
        "activation_layer",
        "prompt",
        "activation_vector",
        "token_ids_prefix",
    }
    missing_columns = sorted(required - schema_names)
    forbidden_teacher_columns = sorted(
        schema_names
        & {"api_explanation", "explanation", "teacher_explanation", "response"}
    )
    read_columns = [
        name
        for name in (
            "sample_uuid",
            "doc_id",
            "split_unit_id",
            "content_family_id",
            "token_position",
            "n_raw_tokens",
            "activation_layer",
            "activation_vector",
            "token_ids_prefix",
            "prompt",
        )
        if name in schema_names
    ]
    rows = 0
    docs: set[str] = set()
    split_units: set[str] = set()
    families: set[str] = set()
    keys: set[tuple[Any, ...]] = set()
    duplicate_keys = 0
    missing_keys = 0
    nonfinite_rows = 0
    empty_prompts = 0
    invalid_prompt_rows = 0
    canonical_prompt_rows = 0
    injection_placeholder_rows = 0
    empty_prefixes = 0
    missing_family_rows = 0
    family_assignment_mismatches = 0
    dimensions: Counter[int] = Counter()
    activation_layers: Counter[int] = Counter()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=read_columns):
        vector_column = batch.column(
            batch.schema.get_field_index("activation_vector")
        )
        vector_lengths = pc.list_value_length(vector_column).to_pylist()
        normalized_lengths = [
            -1 if value is None else int(value) for value in vector_lengths
        ]
        dimensions.update(normalized_lengths)
        if normalized_lengths and all(
            value == expected_d_model for value in normalized_lengths
        ):
            flat_vectors = np.asarray(
                pc.list_flatten(vector_column).to_numpy(zero_copy_only=False),
                dtype=np.float32,
            )
            vectors = flat_vectors.reshape(batch.num_rows, expected_d_model)
            nonfinite_rows += int(
                np.count_nonzero(~np.isfinite(vectors).all(axis=1))
            )
        else:
            for vector_value in vector_column.to_pylist():
                vector = np.asarray(vector_value, dtype=np.float32)
                nonfinite_rows += int(
                    vector.ndim != 1 or not np.isfinite(vector).all()
                )

        scalar_columns = {
            name: batch.column(name).to_pylist()
            for name in read_columns
            if name != "activation_vector"
        }
        for row_index in range(batch.num_rows):
            row = {
                name: values[row_index]
                for name, values in scalar_columns.items()
            }
            rows += 1
            docs.add(str(row.get("doc_id")))
            if row.get("split_unit_id") not in {None, ""}:
                split_units.add(str(row["split_unit_id"]))
            if row.get("content_family_id") not in {None, ""}:
                family_id = str(row["content_family_id"])
                families.add(family_id)
                expected_family_id = family_assignments.get(str(row.get("doc_id")))
                if expected_family_id is not None and family_id != expected_family_id:
                    family_assignment_mismatches += 1
            else:
                missing_family_rows += 1
            key = _provenance_key(row)
            if key is None:
                missing_keys += 1
            elif key in keys:
                duplicate_keys += 1
            else:
                keys.add(key)
            if row.get("activation_layer") is not None:
                activation_layers[int(row["activation_layer"])] += 1
            prompt_content = _actor_prompt_content(row.get("prompt"))
            empty_prompts += int(prompt_content is None)
            invalid_prompt_rows += int(prompt_content is None)
            if prompt_content is not None:
                injection_placeholder_rows += int(
                    prompt_content.count(INJECT_PLACEHOLDER) == 1
                )
            empty_prefixes += int(not row.get("token_ids_prefix"))

    dataset_sidecar, sidecar_errors = _load_dataset_sidecar(
        dataset_path,
        rows,
        expected_d_model,
    )
    canonical_prompt_content = _canonical_prompt_content(dataset_sidecar)
    if canonical_prompt_content is not None and "prompt" in schema_names:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["prompt"]):
            for row in batch.to_pylist():
                canonical_prompt_rows += int(
                    _actor_prompt_content(row.get("prompt")) == canonical_prompt_content
                )

    heldout_overlap = sorted(docs & heldout_docs)
    heldout_family_doc_overlap = sorted(docs & heldout_family_docs)
    heldout_family_overlap = sorted(families & heldout_family_ids)
    nontrain_docs = sorted(docs - train_docs) if train_docs else []
    split_unit_overlap = sorted(split_units & heldout_units)

    manifest_hash = sha256_file(manifest_path)
    lineage_manifest_hash = metadata.get("source_split_manifest_sha256")
    source_base_path = Path(metadata["source_base_parquet"]) if metadata.get("source_base_parquet") else None
    source_base_hash_actual = (
        sha256_file(source_base_path)
        if source_base_path is not None and source_base_path.is_file()
        else None
    )
    source_base_hash_expected = metadata.get("source_base_sha256")
    actor_sidecar_path = (
        Path(metadata["source_actor_sidecar"])
        if metadata.get("source_actor_sidecar")
        else None
    )
    actor_sidecar_hash_actual = (
        sha256_file(actor_sidecar_path)
        if actor_sidecar_path is not None and actor_sidecar_path.is_file()
        else None
    )
    actor_sidecar_hash_expected = metadata.get("source_actor_sidecar_sha256")
    family_manifest_hash_expected = metadata.get("content_family_manifest_sha256")
    family_coverage_hash_expected = metadata.get("content_family_coverage_sha256")
    sidecar_lineage = (
        dataset_sidecar.get("lineage")
        if isinstance(dataset_sidecar, dict)
        and isinstance(dataset_sidecar.get("lineage"), dict)
        else {}
    )
    expected_layer = None
    if dataset_sidecar is not None and isinstance(dataset_sidecar.get("extraction"), dict):
        expected_layer = dataset_sidecar["extraction"].get("layer_index")

    blockers: list[str] = []
    if manifest_errors:
        blockers.append("split_manifest_structure")
    if missing_columns:
        blockers.append("missing_columns")
    if forbidden_teacher_columns:
        blockers.append("teacher_text_columns")
    if rows == 0:
        blockers.append("empty_dataset")
    if nonfinite_rows:
        blockers.append("nonfinite_activations")
    if set(dimensions) != {expected_d_model}:
        blockers.append("d_model")
    if not isinstance(expected_layer, int) or set(activation_layers) != {expected_layer}:
        blockers.append("activation_layer")
    if duplicate_keys:
        blockers.append("duplicate_provenance_keys")
    if missing_keys:
        blockers.append("missing_provenance_keys")
    if heldout_overlap or nontrain_docs:
        blockers.append("heldout_doc_overlap")
    if split_unit_overlap or split_units - train_units:
        blockers.append("split_unit_overlap")
    if empty_prompts:
        blockers.append("empty_prompts")
    if invalid_prompt_rows:
        blockers.append("prompt_schema")
    if injection_placeholder_rows != rows:
        blockers.append("injection_placeholder")
    if canonical_prompt_content is None or canonical_prompt_rows != rows:
        blockers.append("prompt_template")
    if sidecar_errors:
        blockers.append("dataset_sidecar")
    if empty_prefixes:
        blockers.append("empty_token_prefixes")
    if metadata.get("nano_schema_version") != SCHEMA_VERSION:
        blockers.append("schema_version")
    if lineage_manifest_hash != manifest_hash:
        blockers.append("split_manifest_hash")
    if (
        source_base_path is None
        or source_base_hash_expected is None
        or source_base_hash_actual is None
    ):
        blockers.append("source_base_provenance")
    elif source_base_hash_expected != source_base_hash_actual:
        blockers.append("source_base_hash")
    if (
        actor_sidecar_path is None
        or actor_sidecar_hash_expected is None
        or actor_sidecar_hash_actual is None
    ):
        blockers.append("actor_sidecar_provenance")
    elif actor_sidecar_hash_expected != actor_sidecar_hash_actual:
        blockers.append("actor_sidecar_hash")
    if expected_rows is None or rows != expected_rows:
        blockers.append("expected_rows")
    train_membership_mode = metadata.get("train_membership_mode")
    explicit_unit_contract = (
        train_membership_mode == "split_unit"
        and metadata.get("split_unit_filter_applied") == "true"
    )
    derived_family_contract = (
        train_membership_mode == "doc_content_family"
        and metadata.get("doc_filter_applied") == "true"
        and metadata.get("derived_split_unit_ids") == "true"
        and metadata.get("family_filter_applied") == "true"
        and derived_manifest_units
    )
    if not (explicit_unit_contract or derived_family_contract):
        blockers.append("train_membership_filter_required")
    if metadata.get("family_filter_applied") != "true":
        blockers.append("family_filter_required")
    if missing_family_rows or family_assignment_mismatches:
        blockers.append("content_family_assignment")
    if heldout_family_overlap or heldout_family_doc_overlap:
        blockers.append("heldout_family_overlap")
    if family_contract_errors:
        blockers.append("family_contract")
    if family_manifest_hash_expected != family_manifest_hash:
        blockers.append("content_family_manifest_hash")
    if family_coverage_hash_expected != family_coverage_hash:
        blockers.append("content_family_coverage_hash")
    if sidecar_lineage.get("content_family_manifest_sha256") != family_manifest_hash:
        blockers.append("sidecar_content_family_manifest_hash")
    if sidecar_lineage.get("content_family_coverage_sha256") != family_coverage_hash:
        blockers.append("sidecar_content_family_coverage_hash")
    sidecar_explicit_contract = (
        sidecar_lineage.get("train_membership_mode") == "split_unit"
        and sidecar_lineage.get("split_unit_filter_applied") is True
    )
    sidecar_derived_contract = (
        sidecar_lineage.get("train_membership_mode") == "doc_content_family"
        and sidecar_lineage.get("doc_filter_applied") is True
        and sidecar_lineage.get("derived_split_unit_ids") is True
        and sidecar_lineage.get("family_filter_applied") is True
    )
    if not (sidecar_explicit_contract or sidecar_derived_contract):
        blockers.append("sidecar_train_membership_filter_required")
    if sidecar_lineage.get("family_filter_applied") is not True:
        blockers.append("sidecar_family_filter_required")

    return {
        "schema_version": "nano_r33_rl_dataset_verify.v1",
        "passed": not blockers,
        "blockers": blockers,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "rows": rows,
        "unique_documents": len(docs),
        "unique_provenance_keys": len(keys),
        "duplicate_provenance_keys": duplicate_keys,
        "missing_provenance_keys": missing_keys,
        "d_model_counts": {str(key): value for key, value in sorted(dimensions.items())},
        "activation_layer_counts": {
            str(key): value for key, value in sorted(activation_layers.items())
        },
        "expected_d_model": expected_d_model,
        "nonfinite_activation_rows": nonfinite_rows,
        "empty_prompts": empty_prompts,
        "invalid_prompt_rows": invalid_prompt_rows,
        "canonical_prompt_rows": canonical_prompt_rows,
        "injection_placeholder_rows": injection_placeholder_rows,
        "empty_token_prefixes": empty_prefixes,
        "forbidden_teacher_columns": forbidden_teacher_columns,
        "dataset_sidecar": str(sidecar_path_for(dataset_path)),
        "dataset_sidecar_errors": sidecar_errors,
        "heldout_doc_overlap_count": len(heldout_overlap),
        "heldout_doc_overlap_sample": heldout_overlap[:20],
        "nontrain_doc_count": len(nontrain_docs),
        "split_unit_overlap_count": len(split_unit_overlap),
        "unique_split_units": len(split_units),
        "train_membership_mode": train_membership_mode,
        "derived_manifest_units": derived_manifest_units,
        "unique_content_families": len(families),
        "missing_content_family_rows": missing_family_rows,
        "content_family_assignment_mismatches": family_assignment_mismatches,
        "heldout_family_overlap_count": len(heldout_family_overlap),
        "heldout_family_overlap_sample": heldout_family_overlap[:20],
        "heldout_family_doc_overlap_count": len(heldout_family_doc_overlap),
        "expected_rows": expected_rows,
        "family_contract_errors": family_contract_errors,
        "source_hashes": {
            "split_manifest_expected": lineage_manifest_hash,
            "split_manifest_actual": manifest_hash,
            "base_expected": source_base_hash_expected,
            "base_actual": source_base_hash_actual,
            "actor_sidecar_expected": actor_sidecar_hash_expected,
            "actor_sidecar_actual": actor_sidecar_hash_actual,
            "content_family_manifest_expected": family_manifest_hash_expected,
            "content_family_manifest_actual": family_manifest_hash,
            "content_family_coverage_expected": family_coverage_hash_expected,
            "content_family_coverage_actual": family_coverage_hash,
        },
        "split_manifest_errors": manifest_errors,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--content-family-manifest", type=Path, required=True)
    parser.add_argument("--content-family-coverage", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-d-model", type=int, default=2_688)
    parser.add_argument("--batch-size", type=int, default=4_096)
    args = parser.parse_args()
    report = verify_dataset(
        dataset=args.dataset,
        split_manifest=args.split_manifest,
        content_family_manifest=args.content_family_manifest,
        content_family_coverage=args.content_family_coverage,
        expected_rows=args.expected_rows,
        expected_d_model=args.expected_d_model,
        batch_size=args.batch_size,
    )
    _write_json_atomic(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
