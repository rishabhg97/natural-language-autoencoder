#!/usr/bin/env python3
"""Freeze a confirmatory family split that excludes prior eval families from test."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_functional_eval_data import (  # noqa: E402
    FunctionalEvaluationError,
    assign_family_splits,
    build_content_families,
    normalized_content_tokens,
)


SCHEMA_VERSION = "nano_publication_family_split.v1"
REPORT_SCHEMA = "nano_publication_family_split_report.v1"
EXPOSURE_INVENTORY_SCHEMA = "nano_publication_exposure_inventory.v1"


class PublicationFamilySplitError(ValueError):
    """Raised when a confirmatory family split cannot be frozen safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, *, config_path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else config_path.parent / candidate


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise PublicationFamilySplitError(
            f"config must use schema_version {SCHEMA_VERSION}"
        )
    if not config.get("base_manifest"):
        raise PublicationFamilySplitError("base_manifest is required")
    globs = config.get("prior_exposure_globs") or config.get(
        "prior_evaluation_globs"
    )
    if not isinstance(globs, list) or not globs:
        raise PublicationFamilySplitError(
            "prior_exposure_globs must be non-empty"
        )
    assignment = config.get("split_assignment") or {}
    if assignment.get("seed") is None or not assignment.get("weights"):
        raise PublicationFamilySplitError(
            "split_assignment.seed and split_assignment.weights are required"
        )
    outputs = config.get("outputs") or {}
    for key in ("manifest_json", "report_json"):
        if not outputs.get(key):
            raise PublicationFamilySplitError(f"outputs.{key} is required")
    if (config.get("joint_family_resolution") or {}).get("enabled", False):
        if not outputs.get("joint_family_manifest_json"):
            raise PublicationFamilySplitError(
                "outputs.joint_family_manifest_json is required when joint family "
                "resolution is enabled"
            )
    return config


def _expand_sources(patterns: list[str], *, config_path: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        candidate = Path(pattern)
        resolved_pattern = str(
            candidate if candidate.is_absolute() else config_path.parent / candidate
        )
        paths.update(Path(match).resolve() for match in glob.glob(resolved_pattern, recursive=True))
    sources = sorted(path for path in paths if path.is_file())
    if not sources:
        raise PublicationFamilySplitError(
            "prior exposure globs resolved to no source parquets"
        )
    return sources


def _source_doc_ids_and_hashes(
    path: Path,
    *,
    doc_id_column: str = "doc_id",
    text_column: str | None = None,
    require_text: bool = False,
    skip_text_for_doc_ids: set[str] | None = None,
) -> tuple[set[str], dict[str, str], dict[str, str], str | None]:
    schema = pq.ParquetFile(path).schema_arrow.names
    if doc_id_column not in schema:
        raise PublicationFamilySplitError(
            f"source has no {doc_id_column}: {path}"
        )
    if text_column is not None and text_column not in schema:
        raise PublicationFamilySplitError(
            f"source has no configured text column {text_column}: {path}"
        )
    if text_column is None:
        text_column = next(
            (
                name
                for name in (
                    "detokenized_text_truncated",
                    "source_text",
                    "text",
                )
                if name in schema
            ),
            None,
        )
    if require_text and text_column is None:
        raise PublicationFamilySplitError(f"source has no usable text column: {path}")
    doc_ids: set[str] = set()
    representative_text: dict[str, str] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=[doc_id_column]):
        for row in batch.to_pylist():
            doc_id = str(row.get(doc_id_column) or "").strip()
            if not doc_id:
                continue
            doc_ids.add(doc_id)
    if text_column is not None:
        text_doc_ids = (
            doc_ids
            if skip_text_for_doc_ids is None
            else doc_ids - skip_text_for_doc_ids
        )
        for batch in parquet.iter_batches(
            batch_size=8192,
            columns=[doc_id_column, text_column],
        ):
            for row in batch.to_pylist():
                doc_id = str(row.get(doc_id_column) or "").strip()
                if doc_id not in text_doc_ids:
                    continue
                tokens = normalized_content_tokens(str(row.get(text_column) or ""))
                normalized = " ".join(tokens)
                previous = representative_text.get(doc_id, "")
                if (len(tokens), normalized) > (len(previous.split()), previous):
                    representative_text[doc_id] = normalized
    hashes = {
        doc_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for doc_id, text in representative_text.items()
        if text
    }
    return doc_ids, hashes, representative_text, text_column


def _joint_family_resolution(
    config: dict[str, Any],
    *,
    config_path: Path,
    base_manifest: dict[str, Any],
    prior_documents: dict[str, dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, Any] | None]:
    resolution = config.get("joint_family_resolution") or {}
    if not resolution.get("enabled", False):
        return {}, {}, None
    base_sources = resolution.get("base_content_sources")
    if not isinstance(base_sources, list) or not base_sources:
        raise PublicationFamilySplitError(
            "joint_family_resolution.base_content_sources must be non-empty"
        )

    base_text_by_doc: dict[str, str] = {}
    source_reports: list[dict[str, Any]] = []
    for raw_source in base_sources:
        source = raw_source if isinstance(raw_source, dict) else {"path": raw_source}
        if not source.get("path"):
            raise PublicationFamilySplitError(
                "joint family base sources require path"
            )
        source_path = _resolve(source["path"], config_path=config_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"joint family base source not found: {source_path}")
        doc_ids, _, texts, selected_text_column = _source_doc_ids_and_hashes(
            source_path,
            doc_id_column=str(source.get("doc_id_column") or "doc_id"),
            text_column=source.get("text_column"),
            require_text=True,
        )
        for doc_id, text in texts.items():
            previous = base_text_by_doc.get(doc_id, "")
            if (len(text.split()), text) > (len(previous.split()), previous):
                base_text_by_doc[doc_id] = text
        source_reports.append(
            {
                "path": str(source_path),
                "sha256": _sha256_file(source_path),
                "row_count": int(pq.ParquetFile(source_path).metadata.num_rows),
                "document_count": len(doc_ids),
                "text_column": selected_text_column,
            }
        )

    base_assignments = {
        str(doc_id): str(family_id)
        for doc_id, family_id in (base_manifest.get("doc_assignments") or {}).items()
    }
    expected_base_docs = set(base_assignments)
    observed_base_docs = set(base_text_by_doc)
    missing_base_docs = sorted(expected_base_docs - observed_base_docs)
    extra_base_docs = sorted(observed_base_docs - expected_base_docs)
    if bool(resolution.get("require_exact_base_document_coverage", True)) and (
        missing_base_docs or extra_base_docs
    ):
        raise PublicationFamilySplitError(
            "joint family base document coverage mismatch: "
            f"missing={len(missing_base_docs)} sample={missing_base_docs[:5]}, "
            f"extra={len(extra_base_docs)} sample={extra_base_docs[:5]}"
        )

    rows = [
        {"doc_id": doc_id, "source_text": text}
        for doc_id, text in sorted(base_text_by_doc.items())
        if doc_id in expected_base_docs
    ]
    prior_documents_with_text = 0
    for doc_id, record in sorted(prior_documents.items()):
        text = str(record.get("representative_text") or "")
        if not text:
            continue
        prior_documents_with_text += 1
        rows.append({"doc_id": doc_id, "source_text": text})

    algorithm = resolution.get("algorithm") or base_manifest.get("algorithm") or {}
    combined_manifest = build_content_families(
        rows,
        text_field="source_text",
        shingle_width=int(algorithm.get("shingle_width", 5)),
        similarity_threshold=float(algorithm.get("similarity_threshold", 0.8)),
        signature_size=int(algorithm.get("signature_size", 32)),
        candidate_min_shared=int(algorithm.get("candidate_min_shared", 4)),
        max_signature_bucket_size=int(
            algorithm.get("max_signature_bucket_size", 256)
        ),
    )
    combined_assignments = combined_manifest["doc_assignments"]
    base_families_by_joint_family: dict[str, set[str]] = {}
    for doc_id, base_family_id in base_assignments.items():
        joint_family_id = combined_assignments.get(doc_id)
        if joint_family_id is None:
            raise PublicationFamilySplitError(
                f"joint family manifest omitted base document: {doc_id}"
            )
        base_families_by_joint_family.setdefault(joint_family_id, set()).add(
            base_family_id
        )

    candidate_families_by_prior_doc: dict[str, set[str]] = {}
    joint_family_by_prior_doc: dict[str, str] = {}
    for doc_id in prior_documents:
        joint_family_id = combined_assignments.get(doc_id)
        if joint_family_id is None:
            continue
        joint_family_by_prior_doc[doc_id] = joint_family_id
        candidate_families_by_prior_doc[doc_id] = set(
            base_families_by_joint_family.get(joint_family_id, set())
        )

    combined_manifest["publication_joint_family_resolution"] = {
        "base_content_sources": source_reports,
        "base_document_count": len(expected_base_docs),
        "prior_document_count": len(prior_documents),
        "prior_documents_with_text": prior_documents_with_text,
    }
    return (
        candidate_families_by_prior_doc,
        joint_family_by_prior_doc,
        combined_manifest,
    )


def run_build(config_path: str | Path) -> dict[str, Any]:
    resolved_config = Path(config_path).resolve()
    config = _load_config(resolved_config)
    outputs = config["outputs"]
    manifest_out = _resolve(outputs["manifest_json"], config_path=resolved_config)
    report_out = _resolve(outputs["report_json"], config_path=resolved_config)
    inventory_out = (
        _resolve(outputs["exposure_inventory_json"], config_path=resolved_config)
        if outputs.get("exposure_inventory_json")
        else None
    )
    joint_manifest_out = (
        _resolve(outputs["joint_family_manifest_json"], config_path=resolved_config)
        if outputs.get("joint_family_manifest_json")
        else None
    )
    existing_outputs = [
        path
        for path in (manifest_out, report_out, inventory_out, joint_manifest_out)
        if path and path.exists()
    ]
    if existing_outputs:
        raise PublicationFamilySplitError(
            "frozen publication split output already exists: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    base_path = _resolve(config["base_manifest"], config_path=resolved_config)
    base_manifest = json.loads(base_path.read_text())
    if base_manifest.get("schema_version") != "nano_content_family_manifest.v1":
        raise PublicationFamilySplitError(
            "base manifest must use schema_version nano_content_family_manifest.v1"
        )
    doc_assignments = base_manifest.get("doc_assignments") or {}
    if not doc_assignments:
        raise PublicationFamilySplitError("base manifest has no doc assignments")
    known_docs = set(doc_assignments)

    exposure_globs = config.get("prior_exposure_globs") or config.get(
        "prior_evaluation_globs"
    )
    source_paths = _expand_sources(
        [str(pattern) for pattern in exposure_globs],
        config_path=resolved_config,
    )
    forbidden_families: set[str] = set()
    source_reports: list[dict[str, Any]] = []
    document_exposure: dict[str, dict[str, Any]] = {}
    content_hash_to_families: dict[str, set[str]] = {}
    for family in base_manifest.get("families") or []:
        family_id = str(family.get("content_family_id") or "")
        values = family.get("normalized_text_sha256") or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            content_hash_to_families.setdefault(str(value), set()).add(family_id)
    for source_path in source_paths:
        doc_ids, source_hashes, source_texts, text_column = (
            _source_doc_ids_and_hashes(
                source_path,
                skip_text_for_doc_ids=known_docs,
            )
        )
        source_families = {
            str(doc_assignments[doc_id])
            for doc_id in doc_ids
            if doc_id in doc_assignments
        }
        direct_mapped_docs = doc_ids & known_docs
        source_missing_docs = doc_ids - direct_mapped_docs
        hash_mapped_docs: set[str] = set()
        hash_mapped_families: set[str] = set()
        ambiguous_hash_docs: set[str] = set()
        source_hash_families: dict[str, set[str]] = {}
        for doc_id in source_missing_docs:
            families = content_hash_to_families.get(source_hashes.get(doc_id, ""), set())
            if not families:
                continue
            hash_mapped_docs.add(doc_id)
            hash_mapped_families.update(families)
            source_hash_families[doc_id] = set(families)
            if len(families) > 1:
                ambiguous_hash_docs.add(doc_id)
        source_missing_docs -= hash_mapped_docs
        forbidden_families.update(source_families | hash_mapped_families)
        for doc_id in sorted(doc_ids):
            record = document_exposure.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "seen_in_sources": [],
                    "normalized_text_sha256s": set(),
                    "representative_text": "",
                    "direct_family_id": None,
                    "content_hash_family_ids": set(),
                },
            )
            record["seen_in_sources"].append(str(source_path))
            if source_hashes.get(doc_id):
                record["normalized_text_sha256s"].add(source_hashes[doc_id])
            if doc_id in direct_mapped_docs:
                record["direct_family_id"] = str(doc_assignments[doc_id])
            source_text = source_texts.get(doc_id, "")
            previous_text = str(record["representative_text"])
            if (len(source_text.split()), source_text) > (
                len(previous_text.split()),
                previous_text,
            ):
                record["representative_text"] = source_text
            record["content_hash_family_ids"].update(
                source_hash_families.get(doc_id, set())
            )
        source_reports.append(
            {
                "path": str(source_path),
                "sha256": _sha256_file(source_path),
                "row_count": int(pq.ParquetFile(source_path).metadata.num_rows),
                "document_count": len(doc_ids),
                "text_column": text_column,
                "direct_mapped_document_count": len(direct_mapped_docs),
                "family_count": len(source_families),
                "content_hash_mapped_document_count": len(hash_mapped_docs),
                "content_hash_mapped_family_count": len(hash_mapped_families),
                "ambiguous_content_hash_document_count": len(ambiguous_hash_docs),
                "unmapped_document_count": len(source_missing_docs),
            }
        )

    (
        joint_candidate_families,
        joint_family_by_doc,
        joint_manifest,
    ) = _joint_family_resolution(
        config,
        config_path=resolved_config,
        base_manifest=base_manifest,
        prior_documents=document_exposure,
    )
    joint_manifest_sha256 = None
    if joint_manifest is not None and joint_manifest_out is not None:
        joint_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        joint_manifest_out.write_text(
            json.dumps(joint_manifest, indent=2, sort_keys=True) + "\n"
        )
        joint_manifest_sha256 = _sha256_file(joint_manifest_out)

    exposure_documents: list[dict[str, Any]] = []
    missing_docs: set[str] = set()
    exposure_status_counts: dict[str, int] = {}
    source_global_status_counts: dict[str, dict[str, int]] = {}
    for doc_id, raw_record in sorted(document_exposure.items()):
        direct_family_id = raw_record["direct_family_id"]
        hash_family_ids = sorted(raw_record["content_hash_family_ids"])
        near_duplicate_family_ids = sorted(
            joint_candidate_families.get(doc_id, set()) - set(hash_family_ids)
        )
        joint_family_id = joint_family_by_doc.get(doc_id)
        if direct_family_id:
            status = "direct_doc_id"
        elif len(hash_family_ids) == 1:
            status = "content_hash"
        elif hash_family_ids:
            status = "ambiguous_content_hash"
        elif near_duplicate_family_ids:
            status = "near_duplicate"
        elif joint_manifest is not None and joint_family_id:
            status = "outside_candidate_universe"
        else:
            status = "unmapped"
            missing_docs.add(doc_id)
        forbidden_families.update(hash_family_ids)
        forbidden_families.update(near_duplicate_family_ids)
        exposure_status_counts[status] = exposure_status_counts.get(status, 0) + 1
        for source in set(raw_record["seen_in_sources"]):
            counts = source_global_status_counts.setdefault(source, {})
            counts[status] = counts.get(status, 0) + 1
        exposure_documents.append(
            {
                "doc_id": doc_id,
                "status": status,
                "direct_family_id": direct_family_id,
                "content_hash_family_ids": hash_family_ids,
                "near_duplicate_family_ids": near_duplicate_family_ids,
                "joint_content_family_id": joint_family_id,
                "normalized_text_sha256s": sorted(
                    raw_record["normalized_text_sha256s"]
                ),
                "seen_in_sources": sorted(set(raw_record["seen_in_sources"])),
            }
        )
    for source_report in source_reports:
        source_report["global_status_counts"] = dict(
            sorted(source_global_status_counts.get(source_report["path"], {}).items())
        )
    if not forbidden_families:
        raise PublicationFamilySplitError(
            "prior evaluation sources map to no families in the R33 base manifest"
        )

    assignment = config["split_assignment"]
    constraints = {family_id: {"test"} for family_id in forbidden_families}
    assignment_error: str | None = None
    try:
        frozen_manifest = assign_family_splits(
            base_manifest,
            split_weights={
                str(name): float(weight)
                for name, weight in assignment["weights"].items()
            },
            seed=int(assignment["seed"]),
            forbidden_splits_by_family=constraints,
        )
    except FunctionalEvaluationError as exc:
        frozen_manifest = None
        assignment_error = str(exc)

    test_summary = (
        ((frozen_manifest.get("split_summary") or {}).get("test") or {})
        if frozen_manifest is not None
        else {}
    )
    requirements = config.get("requirements") or {}
    min_test_rows = int(requirements.get("min_test_rows", 512))
    min_test_families = int(requirements.get("min_test_families", 100))
    max_unmapped_prior_documents = int(
        requirements.get("max_unmapped_prior_documents", 0)
    )
    errors: list[str] = []
    if assignment_error is not None:
        errors.append(f"split assignment is infeasible: {assignment_error}")
    else:
        if int(test_summary.get("row_count") or 0) < min_test_rows:
            errors.append(f"test row count is below {min_test_rows}")
        if int(test_summary.get("family_count") or 0) < min_test_families:
            errors.append(f"test family count is below {min_test_families}")
    if len(missing_docs) > max_unmapped_prior_documents:
        errors.append(
            "unmapped prior document count exceeds "
            f"{max_unmapped_prior_documents}: {len(missing_docs)}"
        )

    provenance = {
        "config": str(resolved_config),
        "config_sha256": _sha256_file(resolved_config),
        "base_manifest": str(base_path),
        "base_manifest_sha256": _sha256_file(base_path),
        "prior_evaluation_sources": source_reports,
        "prior_exposure_sources": source_reports,
        "prior_evaluation_source_count": len(source_reports),
        "prior_exposure_source_count": len(source_reports),
        "test_forbidden_family_count": len(forbidden_families),
        "unmapped_prior_document_count": len(missing_docs),
    }
    if joint_manifest_out is not None:
        provenance["joint_family_manifest_json"] = str(joint_manifest_out)
        provenance["joint_family_manifest_sha256"] = joint_manifest_sha256
    exposure_inventory = {
        "schema_version": EXPOSURE_INVENTORY_SCHEMA,
        "base_manifest": str(base_path),
        "base_manifest_sha256": provenance["base_manifest_sha256"],
        "joint_family_manifest_json": (
            str(joint_manifest_out) if joint_manifest_out is not None else None
        ),
        "joint_family_manifest_sha256": joint_manifest_sha256,
        "prior_evaluation_sources": source_reports,
        "prior_exposure_sources": source_reports,
        "summary": {
            "prior_evaluation_source_count": len(source_reports),
            "prior_exposure_source_count": len(source_reports),
            "unique_prior_document_count": len(exposure_documents),
            "test_forbidden_family_count": len(forbidden_families),
            "status_counts": dict(sorted(exposure_status_counts.items())),
            "unmapped_prior_document_count": len(missing_docs),
        },
        "documents": exposure_documents,
    }
    if inventory_out is not None:
        inventory_out.parent.mkdir(parents=True, exist_ok=True)
        inventory_out.write_text(
            json.dumps(exposure_inventory, indent=2, sort_keys=True) + "\n"
        )
        provenance["exposure_inventory_json"] = str(inventory_out)
        provenance["exposure_inventory_sha256"] = _sha256_file(inventory_out)
    if frozen_manifest is not None:
        frozen_manifest["publication_split_provenance"] = provenance
    report = {
        "schema_version": REPORT_SCHEMA,
        "passed": not errors,
        "errors": errors,
        "manifest_json": str(manifest_out),
        "base_manifest_sha256": provenance["base_manifest_sha256"],
        "prior_evaluation_source_count": len(source_reports),
        "prior_exposure_source_count": len(source_reports),
        "test_forbidden_family_count": len(forbidden_families),
        "unmapped_prior_document_count": len(missing_docs),
        "prior_evaluation_sources": source_reports,
        "prior_exposure_sources": source_reports,
        "exposure_inventory_json": str(inventory_out) if inventory_out else None,
        "exposure_inventory_sha256": provenance.get("exposure_inventory_sha256"),
        "joint_family_manifest_json": (
            str(joint_manifest_out) if joint_manifest_out is not None else None
        ),
        "joint_family_manifest_sha256": joint_manifest_sha256,
        "split_summary": (
            frozen_manifest["split_summary"] if frozen_manifest is not None else {}
        ),
        "requirements": {
            "min_test_rows": min_test_rows,
            "min_test_families": min_test_families,
            "max_unmapped_prior_documents": max_unmapped_prior_documents,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise PublicationFamilySplitError("; ".join(errors))
    if frozen_manifest is None:
        raise AssertionError("passing publication split has no frozen manifest")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    report = run_build(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
