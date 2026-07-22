#!/usr/bin/env python3
"""Build a heldout-clean, train-only R33 RL parquet without teacher text."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


SCHEMA_VERSION = "nano_r33_rl_dataset.v3"
DATASET_SIDECAR_SCHEMA_VERSION = 1
MODEL_SIDECAR_SCHEMA_VERSION = 2
INJECT_PLACEHOLDER = "<INJECT>"
REQUIRED_COLUMNS = {
    "doc_id",
    "activation_layer",
    "activation_vector",
    "token_ids_prefix",
}
PROVENANCE_COLUMNS = (
    "sample_uuid",
    "doc_id",
    "token_position",
    "n_raw_tokens",
    "token_id",
)
OUTPUT_COLUMNS = (
    "sample_uuid",
    "doc_id",
    "split_unit_id",
    "content_family_id",
    "token_position",
    "n_raw_tokens",
    "token_id",
    "activation_layer",
    "prompt",
    "activation_vector",
    "token_ids_prefix",
    "source_row_index",
)


class RLDatasetBuildError(ValueError):
    """Raised when train-only data cannot be derived without ambiguity."""


def sidecar_path_for(source: str | Path) -> Path:
    path = Path(str(source).split("@[")[0])
    if path.is_dir():
        return path / "nla_meta.yaml"
    if path.name.endswith((".yaml", ".yml")):
        return path
    return Path(str(path) + ".nla_meta.yaml")


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _split_sections(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sections = manifest.get("splits")
    if isinstance(sections, dict):
        return sections
    output = {
        name: manifest.get(name)
        for name in ("train", "validation", "test")
        if isinstance(manifest.get(name), dict)
    }
    if len(output) != 3:
        raise RLDatasetBuildError("split manifest needs train, validation, and test sections")
    return output


def _values(section: dict[str, Any], *names: str) -> set[str]:
    for name in names:
        value = section.get(name)
        if isinstance(value, list):
            return {str(item) for item in value}
    return set()


def _validate_source_schema(schema_names: set[str]) -> None:
    missing = sorted(REQUIRED_COLUMNS - schema_names)
    if missing:
        raise RLDatasetBuildError(f"base parquet is missing required columns: {missing}")
    has_key = "sample_uuid" in schema_names or (
        "doc_id" in schema_names
        and ({"token_position", "n_raw_tokens"} & schema_names)
    )
    if not has_key:
        raise RLDatasetBuildError(
            "base parquet needs sample_uuid or doc_id plus token_position/n_raw_tokens"
        )


def _load_actor_contract(
    source: str | Path,
    *,
    expected_layer: int | None = None,
) -> tuple[Path, dict[str, Any], str]:
    sidecar_path = sidecar_path_for(source)
    if not sidecar_path.is_file():
        raise RLDatasetBuildError(f"actor sidecar does not exist: {sidecar_path}")
    metadata = yaml.safe_load(sidecar_path.read_text())
    if not isinstance(metadata, dict):
        raise RLDatasetBuildError(f"actor sidecar is not a mapping: {sidecar_path}")
    kind = metadata.get("kind")
    if kind not in {"nla_dataset", "nla_model"}:
        raise RLDatasetBuildError("actor sidecar kind must be nla_dataset or nla_model")
    schema_version = int(metadata.get("schema_version", -1))
    expected_schema = (
        DATASET_SIDECAR_SCHEMA_VERSION
        if kind == "nla_dataset"
        else MODEL_SIDECAR_SCHEMA_VERSION
    )
    if schema_version != expected_schema:
        raise RLDatasetBuildError(
            f"{kind} actor sidecar schema_version must be {expected_schema}"
        )
    if kind == "nla_model" and metadata.get("role") != "actor":
        raise RLDatasetBuildError("nla_model sidecar role must be actor")
    tokens = metadata.get("tokens")
    if not isinstance(tokens, dict):
        raise RLDatasetBuildError("actor sidecar is missing token metadata")
    for key in (
        "injection_char",
        "injection_token_id",
        "injection_left_neighbor_id",
        "injection_right_neighbor_id",
    ):
        if tokens.get(key) is None:
            raise RLDatasetBuildError(f"actor sidecar tokens missing {key}")
    templates = metadata.get("prompt_templates")
    actor_template = templates.get("actor") if isinstance(templates, dict) else None
    if not isinstance(actor_template, str) or not actor_template.strip():
        raise RLDatasetBuildError("actor sidecar is missing prompt_templates.actor")
    if actor_template.count("{injection_char}") != 1:
        raise RLDatasetBuildError(
            "actor prompt template must contain exactly one {injection_char} placeholder"
        )
    actor_prompt_content = actor_template.format(injection_char=INJECT_PLACEHOLDER)
    if actor_prompt_content.count(INJECT_PLACEHOLDER) != 1:
        raise RLDatasetBuildError(
            f"canonical actor prompt must contain exactly one {INJECT_PLACEHOLDER}"
        )
    extraction = dict(metadata.get("extraction") or {})
    if kind == "nla_model":
        d_model = metadata.get("d_model")
        if not isinstance(d_model, int):
            raise RLDatasetBuildError("actor model sidecar is missing integer d_model")
        extraction["d_model"] = d_model

        trained_layers: set[int] = set()
        for trained_source in metadata.get("trained_on") or []:
            trained_sidecar_path = sidecar_path_for(str(trained_source))
            if not trained_sidecar_path.is_file():
                continue
            trained_metadata = yaml.safe_load(trained_sidecar_path.read_text())
            if not isinstance(trained_metadata, dict):
                continue
            if trained_metadata.get("kind") != "nla_dataset":
                continue
            if int(trained_metadata.get("schema_version", -1)) != DATASET_SIDECAR_SCHEMA_VERSION:
                continue
            trained_extraction = trained_metadata.get("extraction") or {}
            trained_layer = trained_extraction.get("layer_index")
            trained_d_model = trained_extraction.get("d_model")
            trained_tokens = trained_metadata.get("tokens") or {}
            trained_templates = trained_metadata.get("prompt_templates") or {}
            if trained_d_model != d_model:
                raise RLDatasetBuildError(
                    "actor model and trained_on dataset d_model differ"
                )
            if any(trained_tokens.get(key) != tokens.get(key) for key in tokens):
                raise RLDatasetBuildError(
                    "actor model and trained_on dataset token contracts differ"
                )
            if trained_templates.get("actor") != actor_template:
                raise RLDatasetBuildError(
                    "actor model and trained_on dataset prompt templates differ"
                )
            if isinstance(trained_layer, int):
                trained_layers.add(trained_layer)
        if len(trained_layers) > 1:
            raise RLDatasetBuildError(
                "actor model trained_on datasets disagree on activation layer"
            )
        trained_layer = next(iter(trained_layers), None)
        if (
            trained_layer is not None
            and expected_layer is not None
            and trained_layer != expected_layer
        ):
            raise RLDatasetBuildError(
                "configured activation layer disagrees with actor trained_on lineage: "
                f"expected={expected_layer} trained_on={trained_layer}"
            )
        resolved_layer = trained_layer if trained_layer is not None else expected_layer
        if resolved_layer is None:
            raise RLDatasetBuildError(
                "actor model sidecar needs a resolvable trained_on dataset layer or "
                "an explicit expected_layer"
            )
        extraction["layer_index"] = int(resolved_layer)

    if not isinstance(extraction.get("d_model"), int):
        raise RLDatasetBuildError("actor sidecar extraction is missing integer d_model")
    if not isinstance(extraction.get("layer_index"), int):
        raise RLDatasetBuildError("actor sidecar extraction is missing integer layer_index")
    normalized_metadata = dict(metadata)
    normalized_metadata["extraction"] = extraction
    return sidecar_path, normalized_metadata, actor_prompt_content


def _write_dataset_sidecar(
    *,
    output_path: Path,
    actor_metadata: dict[str, Any],
    actor_sidecar_path: Path,
    actor_sidecar_hash: str,
    base_path: Path,
    base_hash: str,
    manifest_path: Path,
    manifest_hash: str,
    rows: int,
    content_family_manifest_path: Path | None = None,
    content_family_manifest_hash: str | None = None,
    content_family_coverage_path: Path | None = None,
    content_family_coverage_hash: str | None = None,
    doc_filter_applied: bool,
    split_unit_filter_applied: bool,
    train_membership_mode: str,
    derived_split_unit_ids: bool,
    family_filter_applied: bool,
) -> Path:
    output_sidecar = sidecar_path_for(output_path)
    parent_datasets = list(actor_metadata.get("parent_datasets") or [])
    actor_dataset_id = actor_metadata.get("dataset_id")
    if actor_dataset_id and actor_dataset_id not in parent_datasets:
        parent_datasets.append(actor_dataset_id)
    payload = {
        "kind": "nla_dataset",
        "schema_version": DATASET_SIDECAR_SCHEMA_VERSION,
        "dataset_id": output_path.stem,
        "stage": "rl",
        "row_count": rows,
        "extraction": dict(actor_metadata["extraction"]),
        "keep_debug_metadata": True,
        "tokens": dict(actor_metadata["tokens"]),
        "prompt_templates": dict(actor_metadata["prompt_templates"]),
        "parent_datasets": parent_datasets,
        "created_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "created_by": "scripts.build_nano_r33_rl_dataset",
        "lineage": {
            "source_base_parquet": str(base_path),
            "source_base_sha256": base_hash,
            "source_split_manifest": str(manifest_path),
            "source_split_manifest_sha256": manifest_hash,
            "source_actor_sidecar": str(actor_sidecar_path),
            "source_actor_sidecar_sha256": actor_sidecar_hash,
            "content_family_manifest": (
                None
                if content_family_manifest_path is None
                else str(content_family_manifest_path)
            ),
            "content_family_manifest_sha256": content_family_manifest_hash,
            "content_family_coverage": (
                None
                if content_family_coverage_path is None
                else str(content_family_coverage_path)
            ),
            "content_family_coverage_sha256": content_family_coverage_hash,
            "doc_filter_applied": bool(doc_filter_applied),
            "split_unit_filter_applied": bool(split_unit_filter_applied),
            "train_membership_mode": train_membership_mode,
            "derived_split_unit_ids": bool(derived_split_unit_ids),
            "family_filter_applied": bool(family_filter_applied),
        },
    }
    output_sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_sidecar.with_name(output_sidecar.name + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    os.replace(temporary, output_sidecar)
    return output_sidecar


def build_dataset(
    *,
    base_parquet: str | Path,
    actor_sidecar_source: str | Path,
    split_manifest: str | Path,
    output: str | Path,
    report_json: str | Path,
    content_family_manifest: str | Path | None = None,
    content_family_coverage: str | Path | None = None,
    expected_rows: int | None = None,
    expected_layer: int | None = None,
    batch_size: int = 4_096,
    overwrite: bool = False,
) -> dict[str, Any]:
    base_path = Path(base_parquet)
    manifest_path = Path(split_manifest)
    output_path = Path(output)
    report_path = Path(report_json)
    if batch_size <= 0:
        raise RLDatasetBuildError("batch_size must be positive")
    if expected_rows is not None and expected_rows <= 0:
        raise RLDatasetBuildError("expected_rows must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass overwrite=True to replace it")

    actor_sidecar_path, actor_metadata, actor_prompt_content = _load_actor_contract(
        actor_sidecar_source,
        expected_layer=expected_layer,
    )
    actor_sidecar_hash = sha256_file(actor_sidecar_path)
    expected_d_model = int(actor_metadata["extraction"]["d_model"])
    expected_layer = int(actor_metadata["extraction"]["layer_index"])

    manifest = json.loads(manifest_path.read_text())
    sections = _split_sections(manifest)
    train_docs = _values(sections["train"], "docs", "doc_ids")
    train_units = _values(
        sections["train"],
        "split_unit_ids",
        "component_ids",
        "components",
        "split_units",
    )
    if not train_docs:
        raise RLDatasetBuildError("split manifest train section has no document IDs")

    family_manifest_path = (
        Path(content_family_manifest) if content_family_manifest is not None else None
    )
    family_coverage_path = (
        Path(content_family_coverage) if content_family_coverage is not None else None
    )
    family_manifest_hash = None
    family_coverage_hash = None
    family_assignments: dict[str, str] = {}
    heldout_family_ids: set[str] = set()
    heldout_family_docs: set[str] = set()
    if family_manifest_path is None or family_coverage_path is None:
        raise RLDatasetBuildError(
            "RL dataset builds require content family manifest and coverage"
        )
    if family_manifest_path is not None:
        family_manifest_doc = json.loads(family_manifest_path.read_text())
        if family_manifest_doc.get("schema_version") != "nano_content_family_manifest.v1":
            raise RLDatasetBuildError("invalid content family manifest schema")
        family_assignments = {
            str(doc_id): str(family_id)
            for doc_id, family_id in (
                family_manifest_doc.get("doc_assignments") or {}
            ).items()
        }
        family_manifest_hash = sha256_file(family_manifest_path)
    if family_coverage_path is not None:
        family_coverage_doc = json.loads(family_coverage_path.read_text())
        if family_coverage_doc.get("schema_version") != (
            "nano_content_family_exposure_report.v1"
        ):
            raise RLDatasetBuildError("invalid content family coverage schema")
        for split in ("validation", "test"):
            section = (family_coverage_doc.get("splits") or {}).get(split) or {}
            heldout_family_ids.update(
                str(value) for value in section.get("eligible_family_ids") or []
            )
            heldout_family_docs.update(
                str(value) for value in section.get("eligible_doc_ids") or []
            )
        family_coverage_hash = sha256_file(family_coverage_path)
    missing_family_docs = sorted(train_docs - set(family_assignments))
    if missing_family_docs:
        raise RLDatasetBuildError(
            f"content family manifest is missing train docs: {missing_family_docs[:10]}"
        )
    train_family_ids = {family_assignments[doc_id] for doc_id in train_docs}
    family_overlap = sorted(train_family_ids & heldout_family_ids)
    doc_overlap = sorted(train_docs & heldout_family_docs)
    if family_overlap or doc_overlap:
        raise RLDatasetBuildError(
            "train rows overlap heldout families: "
            f"families={family_overlap[:10]} docs={doc_overlap[:10]}"
        )

    source_hash = sha256_file(base_path)
    manifest_hash = sha256_file(manifest_path)
    parquet = pq.ParquetFile(base_path)
    source_names = set(parquet.schema_arrow.names)
    _validate_source_schema(source_names)
    columns = [
        name
        for name in OUTPUT_COLUMNS
        if name == "prompt" or name == "source_row_index" or name in source_names
    ]
    read_columns = [
        name for name in columns if name not in {"prompt", "source_row_index"}
    ]
    family_filter_applied = bool(family_assignments)
    doc_filter_applied = bool(train_docs and "doc_id" in source_names)
    split_unit_filter_applied = bool(
        train_units and "split_unit_id" in source_names
    )
    derived_split_unit_ids = False
    train_membership_mode = "split_unit"
    if not split_unit_filter_applied:
        if not family_filter_applied:
            raise RLDatasetBuildError(
                "source without explicit split units requires a content family manifest"
            )
        missing_train_assignments = sorted(train_docs - family_assignments.keys())
        if missing_train_assignments:
            raise RLDatasetBuildError(
                "content family manifest is missing train documents: "
                f"{missing_train_assignments[:10]}"
            )
        derived_train_units = {family_assignments[doc_id] for doc_id in train_docs}
        if train_units and train_units != derived_train_units:
            raise RLDatasetBuildError(
                "manifest split units cannot be reconciled with content families"
            )
        train_units = derived_train_units
        train_membership_mode = "doc_content_family"
        derived_split_unit_ids = True
    if expected_rows is None:
        raise RLDatasetBuildError("RL dataset builds require expected_rows")
    if family_filter_applied and "content_family_id" not in columns:
        columns.insert(columns.index("token_position"), "content_family_id")
    if derived_split_unit_ids and "split_unit_id" not in columns:
        columns.insert(columns.index("content_family_id"), "split_unit_id")
    metadata = dict(parquet.schema_arrow.metadata or {})
    metadata.update(
        {
            b"nano_schema_version": SCHEMA_VERSION.encode(),
            b"source_base_parquet": str(base_path).encode(),
            b"source_base_sha256": source_hash.encode(),
            b"source_split_manifest": str(manifest_path).encode(),
            b"source_split_manifest_sha256": manifest_hash.encode(),
            b"source_split": b"train",
            b"source_actor_sidecar": str(actor_sidecar_path).encode(),
            b"source_actor_sidecar_sha256": actor_sidecar_hash.encode(),
            b"doc_filter_applied": str(doc_filter_applied).lower().encode(),
            b"split_unit_filter_applied": str(split_unit_filter_applied).lower().encode(),
            b"train_membership_mode": train_membership_mode.encode(),
            b"derived_split_unit_ids": str(derived_split_unit_ids).lower().encode(),
            b"family_filter_applied": str(family_filter_applied).lower().encode(),
        }
    )
    if family_manifest_path is not None and family_manifest_hash is not None:
        metadata[b"content_family_manifest"] = str(family_manifest_path).encode()
        metadata[b"content_family_manifest_sha256"] = family_manifest_hash.encode()
    if family_coverage_path is not None and family_coverage_hash is not None:
        metadata[b"content_family_coverage"] = str(family_coverage_path).encode()
        metadata[b"content_family_coverage_sha256"] = family_coverage_hash.encode()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    source_offset = 0
    rows_written = 0
    unique_docs: set[str] = set()
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=read_columns):
            row_indices = pa.array(
                range(source_offset, source_offset + batch.num_rows),
                type=pa.int64(),
            )
            source_offset += batch.num_rows
            batch = batch.append_column("source_row_index", row_indices)
            doc_mask = pc.is_in(
                batch.column(batch.schema.get_field_index("doc_id")),
                value_set=pa.array(sorted(train_docs)),
            )
            mask = doc_mask
            if split_unit_filter_applied:
                unit_mask = pc.is_in(
                    batch.column(batch.schema.get_field_index("split_unit_id")),
                    value_set=pa.array(sorted(train_units)),
                )
                mask = pc.and_(mask, unit_mask)
            filtered = batch.filter(mask)
            if filtered.num_rows == 0:
                continue
            vector_lengths = pc.list_value_length(
                filtered.column(filtered.schema.get_field_index("activation_vector"))
            )
            dimensions_match = pc.all(
                pc.fill_null(pc.equal(vector_lengths, expected_d_model), False)
            ).as_py()
            if not dimensions_match:
                observed = sorted(set(vector_lengths.to_pylist()), key=lambda value: (value is None, value))
                raise RLDatasetBuildError(
                    "activation dimension does not match actor sidecar: "
                    f"expected={expected_d_model} observed={observed}"
                )
            layer_values = filtered.column(
                filtered.schema.get_field_index("activation_layer")
            )
            layers_match = pc.all(
                pc.fill_null(pc.equal(layer_values, expected_layer), False)
            ).as_py()
            if not layers_match:
                raise RLDatasetBuildError(
                    "activation layer does not match actor sidecar: "
                    f"expected={expected_layer} observed={sorted(set(layer_values.to_pylist()))}"
                )
            derived_columns = {"split_unit_id"} if derived_split_unit_ids else set()
            built = {
                name: filtered.column(name)
                for name in columns
                if name not in {
                    "prompt",
                    "source_row_index",
                    "content_family_id",
                } | derived_columns
            }
            if family_filter_applied:
                doc_ids = [str(value) for value in filtered.column("doc_id").to_pylist()]
                missing_batch_docs = sorted(
                    {doc_id for doc_id in doc_ids if doc_id not in family_assignments}
                )
                if missing_batch_docs:
                    raise RLDatasetBuildError(
                        "content family manifest is missing filtered docs: "
                        f"{missing_batch_docs[:10]}"
                    )
                family_ids = [family_assignments[doc_id] for doc_id in doc_ids]
                overlap = sorted(set(family_ids) & heldout_family_ids)
                if overlap:
                    raise RLDatasetBuildError(
                        f"filtered rows overlap heldout content families: {overlap[:10]}"
                    )
                built["content_family_id"] = pa.array(family_ids, type=pa.string())
                if derived_split_unit_ids:
                    built["split_unit_id"] = pa.array(family_ids, type=pa.string())
            built["prompt"] = pa.array(
                [[{"role": "user", "content": actor_prompt_content}]]
                * filtered.num_rows,
                type=pa.list_(
                    pa.struct([("role", pa.string()), ("content", pa.string())])
                ),
            )
            built["source_row_index"] = filtered.column("source_row_index")
            table = pa.table({name: built[name] for name in columns})
            if writer is None:
                output_schema = table.schema.with_metadata(metadata)
                writer = pq.ParquetWriter(temporary, output_schema, compression="zstd")
                table = table.cast(output_schema)
            writer.write_table(table)
            rows_written += table.num_rows
            unique_docs.update(str(value) for value in table.column("doc_id").to_pylist())
    finally:
        if writer is not None:
            writer.close()
    if rows_written == 0 or writer is None:
        temporary.unlink(missing_ok=True)
        raise RLDatasetBuildError("train filter produced zero rows")
    if expected_rows is not None and rows_written != expected_rows:
        temporary.unlink(missing_ok=True)
        raise RLDatasetBuildError(
            f"expected {expected_rows} RL rows, built {rows_written}"
        )
    os.replace(temporary, output_path)
    output_sidecar = _write_dataset_sidecar(
        output_path=output_path,
        actor_metadata=actor_metadata,
        actor_sidecar_path=actor_sidecar_path,
        actor_sidecar_hash=actor_sidecar_hash,
        base_path=base_path,
        base_hash=source_hash,
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        rows=rows_written,
        content_family_manifest_path=family_manifest_path,
        content_family_manifest_hash=family_manifest_hash,
        content_family_coverage_path=family_coverage_path,
        content_family_coverage_hash=family_coverage_hash,
        doc_filter_applied=doc_filter_applied,
        split_unit_filter_applied=split_unit_filter_applied,
        train_membership_mode=train_membership_mode,
        derived_split_unit_ids=derived_split_unit_ids,
        family_filter_applied=family_filter_applied,
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "base_parquet": str(base_path),
        "base_sha256": source_hash,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": manifest_hash,
        "actor_sidecar": str(actor_sidecar_path),
        "actor_sidecar_sha256": actor_sidecar_hash,
        "actor_sidecar_kind": actor_metadata.get("kind"),
        "actor_sidecar_schema_version": actor_metadata.get("schema_version"),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_sidecar": str(output_sidecar),
        "output_sidecar_sha256": sha256_file(output_sidecar),
        "rows": rows_written,
        "unique_documents": len(unique_docs),
        "train_manifest_documents": len(train_docs),
        "train_manifest_split_units": len(train_units),
        "doc_filter_applied": doc_filter_applied,
        "split_unit_filter_applied": split_unit_filter_applied,
        "train_membership_mode": train_membership_mode,
        "derived_split_unit_ids": derived_split_unit_ids,
        "family_filter_applied": family_filter_applied,
        "content_family_manifest": (
            None if family_manifest_path is None else str(family_manifest_path)
        ),
        "content_family_manifest_sha256": family_manifest_hash,
        "content_family_coverage": (
            None if family_coverage_path is None else str(family_coverage_path)
        ),
        "content_family_coverage_sha256": family_coverage_hash,
        "expected_rows": expected_rows,
        "columns": columns,
        "teacher_text_copied": False,
        "canonical_prompt_placeholder": INJECT_PLACEHOLDER,
    }
    _write_json_atomic(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-parquet", type=Path, required=True)
    parser.add_argument("--actor-sidecar-source", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--content-family-manifest", type=Path, required=True)
    parser.add_argument("--content-family-coverage", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-layer", type=int)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_dataset(
        base_parquet=args.base_parquet,
        actor_sidecar_source=args.actor_sidecar_source,
        split_manifest=args.split_manifest,
        output=args.output,
        report_json=args.report_json,
        content_family_manifest=args.content_family_manifest,
        content_family_coverage=args.content_family_coverage,
        expected_rows=args.expected_rows,
        expected_layer=args.expected_layer,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
