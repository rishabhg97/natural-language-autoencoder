#!/usr/bin/env python3
"""Merge content families connected by exact row-level content prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_nano_miles_av_dataset import (  # noqa: E402
    _content_column_order,
    _row_content_key,
)


SCHEMA_VERSION = "nano_content_family_refinement.v1"
REPORT_SCHEMA_VERSION = "nano_content_family_refinement_report.v1"


class ContentFamilyRefinementError(ValueError):
    """Raised when exact-content family refinement cannot be proven safe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path, *, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ContentFamilyRefinementError(
            f"config must use schema_version {SCHEMA_VERSION}"
        )
    if not config.get("base_manifest"):
        raise ContentFamilyRefinementError("base_manifest is required")
    sources = config.get("exact_content_sources")
    if not isinstance(sources, list) or not sources:
        raise ContentFamilyRefinementError(
            "exact_content_sources must be a non-empty list"
        )
    outputs = config.get("outputs") or {}
    for key in ("manifest_json", "report_json"):
        if not outputs.get(key):
            raise ContentFamilyRefinementError(f"outputs.{key} is required")
    return config


def _source_paths(values: Iterable[Any], *, config_path: Path) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        source = value.get("path") if isinstance(value, dict) else value
        if not source:
            raise ContentFamilyRefinementError(
                "exact_content_sources entries must be paths or mappings with path"
            )
        path = _resolve(source, config_path=config_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"exact content source not found: {path}")
        paths.append(path)
    return paths


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep, drop = sorted((left_root, right_root))
        self.parent[drop] = keep


def _merged_family_id(source_family_ids: list[str]) -> str:
    if len(source_family_ids) == 1:
        return source_family_ids[0]
    material = json.dumps(source_family_ids, separators=(",", ":"))
    return "cf_exact_" + hashlib.sha256(material.encode()).hexdigest()[:20]


def refine_manifest(
    base_manifest: dict[str, Any],
    source_paths: list[Path],
    *,
    require_exact_document_coverage: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if base_manifest.get("schema_version") != "nano_content_family_manifest.v1":
        raise ContentFamilyRefinementError(
            "base manifest must use schema_version nano_content_family_manifest.v1"
        )
    doc_assignments = {
        str(doc_id): str(family_id)
        for doc_id, family_id in (base_manifest.get("doc_assignments") or {}).items()
    }
    families = base_manifest.get("families") or []
    family_by_id = {
        str(family["content_family_id"]): family for family in families
    }
    family_ids = set(family_by_id)
    if not doc_assignments or not family_ids:
        raise ContentFamilyRefinementError(
            "base manifest must contain doc_assignments and families"
        )
    unknown_assignment_families = sorted(set(doc_assignments.values()) - family_ids)
    if unknown_assignment_families:
        raise ContentFamilyRefinementError(
            "doc assignments reference missing families: "
            f"{unknown_assignment_families[:5]}"
        )

    union_find = _UnionFind(family_ids)
    key_first_family: dict[str, str] = {}
    key_first_doc: dict[str, str] = {}
    duplicate_keys: set[str] = set()
    cross_family_keys: set[str] = set()
    cross_family_samples: list[dict[str, Any]] = []
    observed_docs: set[str] = set()
    observed_rows = 0
    source_reports: list[dict[str, Any]] = []

    for source_path in source_paths:
        parquet = pq.ParquetFile(source_path)
        names = parquet.schema_arrow.names
        if "doc_id" not in names:
            raise ContentFamilyRefinementError(
                f"exact content source has no doc_id: {source_path}"
            )
        content_columns = _content_column_order(names)
        if not content_columns:
            raise ContentFamilyRefinementError(
                f"exact content source has no supported content columns: {source_path}"
            )
        selected_columns = ["doc_id", *content_columns]
        source_rows = 0
        source_docs: set[str] = set()
        for batch in parquet.iter_batches(batch_size=4096, columns=selected_columns):
            batch_names = batch.schema.names
            docs = batch.column(batch_names.index("doc_id")).to_pylist()
            values = {
                name: batch.column(batch_names.index(name)).to_pylist()
                for name in content_columns
            }
            for offset, doc_value in enumerate(docs):
                doc_id = str(doc_value or "")
                family_id = doc_assignments.get(doc_id)
                if family_id is None:
                    raise ContentFamilyRefinementError(
                        f"exact content source document is absent from base manifest: {doc_id!r}"
                    )
                content_key = _row_content_key(values, offset)
                first_doc = key_first_doc.setdefault(content_key, doc_id)
                first_family = key_first_family.setdefault(content_key, family_id)
                if first_doc != doc_id:
                    duplicate_keys.add(content_key)
                if first_family != family_id:
                    if content_key not in cross_family_keys and len(cross_family_samples) < 20:
                        cross_family_samples.append(
                            {
                                "content_hash": content_key,
                                "doc_ids": [first_doc, doc_id],
                                "family_ids": [first_family, family_id],
                            }
                        )
                    cross_family_keys.add(content_key)
                    union_find.union(first_family, family_id)
                source_docs.add(doc_id)
                observed_docs.add(doc_id)
                source_rows += 1
                observed_rows += 1
        source_reports.append(
            {
                "path": str(source_path),
                "sha256": _sha256_file(source_path),
                "row_count": source_rows,
                "document_count": len(source_docs),
                "content_columns": content_columns,
            }
        )

    expected_docs = set(doc_assignments)
    missing_docs = sorted(expected_docs - observed_docs)
    extra_docs = sorted(observed_docs - expected_docs)
    if require_exact_document_coverage and (missing_docs or extra_docs):
        raise ContentFamilyRefinementError(
            "exact content source document coverage mismatch: "
            f"missing={len(missing_docs)} sample={missing_docs[:5]}, "
            f"extra={len(extra_docs)} sample={extra_docs[:5]}"
        )

    component_families: dict[str, list[str]] = defaultdict(list)
    for family_id in sorted(family_ids):
        component_families[union_find.find(family_id)].append(family_id)
    source_to_refined: dict[str, str] = {}
    for source_family_ids in component_families.values():
        refined_id = _merged_family_id(sorted(source_family_ids))
        for source_family_id in source_family_ids:
            source_to_refined[source_family_id] = refined_id

    refined_assignments = {
        doc_id: source_to_refined[family_id]
        for doc_id, family_id in sorted(doc_assignments.items())
    }
    docs_by_refined: dict[str, list[str]] = defaultdict(list)
    for doc_id, refined_id in refined_assignments.items():
        docs_by_refined[refined_id].append(doc_id)

    refined_families: list[dict[str, Any]] = []
    for source_family_ids in sorted(
        (sorted(values) for values in component_families.values()),
        key=lambda values: _merged_family_id(values),
    ):
        refined_id = _merged_family_id(source_family_ids)
        source_families = [family_by_id[family_id] for family_id in source_family_ids]
        refined_family = {
            "content_family_id": refined_id,
            "doc_ids": sorted(docs_by_refined[refined_id]),
            "document_count": len(docs_by_refined[refined_id]),
            "row_count": sum(int(family.get("row_count") or 0) for family in source_families),
            "normalized_text_sha256": sorted(
                {
                    str(value)
                    for family in source_families
                    for value in (family.get("normalized_text_sha256") or [])
                }
            ),
        }
        if len(source_family_ids) > 1:
            refined_family["source_family_ids"] = source_family_ids
            refined_family["exact_content_refined"] = True
        refined_families.append(refined_family)

    base_row_count = int((base_manifest.get("stats") or {}).get("row_count") or 0)
    refined_row_count = sum(int(family["row_count"]) for family in refined_families)
    if refined_row_count != base_row_count:
        raise ContentFamilyRefinementError(
            f"refined family row count {refined_row_count} != base {base_row_count}"
        )

    stale_split_fields = {
        "family_splits",
        "split_assignment",
        "split_summary",
        "overlap",
    }
    refined = {
        key: value
        for key, value in base_manifest.items()
        if key not in stale_split_fields
    }
    refinement = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_reports": source_reports,
        "require_exact_document_coverage": bool(require_exact_document_coverage),
        "exact_document_coverage": not missing_docs and not extra_docs,
        "observed_row_count": observed_rows,
        "observed_document_count": len(observed_docs),
        "duplicate_key_count": len(duplicate_keys),
        "cross_family_duplicate_key_count": len(cross_family_keys),
        "cross_family_duplicate_keys_sample": cross_family_samples,
        "family_count_before": len(family_ids),
        "family_count_after": len(refined_families),
        "merged_family_component_count": sum(
            1 for values in component_families.values() if len(values) > 1
        ),
    }
    refined.update(
        {
            "doc_assignments": refined_assignments,
            "families": refined_families,
            "stats": {
                **(base_manifest.get("stats") or {}),
                "document_count": len(refined_assignments),
                "family_count": len(refined_families),
                "row_count": refined_row_count,
            },
            "exact_content_refinement": refinement,
        }
    )
    return refined, refinement


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_refinement(
    config_path: str | Path,
) -> dict[str, Any]:
    resolved_config = Path(config_path).resolve()
    config = _load_config(resolved_config)
    base_path = _resolve(config["base_manifest"], config_path=resolved_config).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"base manifest not found: {base_path}")
    outputs = config["outputs"]
    manifest_out = _resolve(outputs["manifest_json"], config_path=resolved_config)
    report_out = _resolve(outputs["report_json"], config_path=resolved_config)
    if manifest_out.exists() or report_out.exists():
        raise ContentFamilyRefinementError(
            f"refinement output already exists: {manifest_out} or {report_out}"
        )
    sources = _source_paths(
        config["exact_content_sources"],
        config_path=resolved_config,
    )
    requirements = config.get("requirements") or {}
    refined, refinement = refine_manifest(
        json.loads(base_path.read_text()),
        sources,
        require_exact_document_coverage=bool(
            requirements.get("exact_document_coverage", True)
        ),
    )
    _write_json(manifest_out, refined)
    report = {
        **refinement,
        "passed": True,
        "config": str(resolved_config),
        "config_sha256": _sha256_file(resolved_config),
        "base_manifest": str(base_path),
        "base_manifest_sha256": _sha256_file(base_path),
        "manifest_json": str(manifest_out),
        "manifest_sha256": _sha256_file(manifest_out),
    }
    _write_json(report_out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    report = run_refinement(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
