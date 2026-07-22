#!/usr/bin/env python3
"""Materialize Nano AV/AR-SFT train/validation/test parquets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_nano_miles_av_dataset import (  # noqa: E402
    _content_column_order,
    _row_content_key,
    _split_items_three_way,
    sidecar_path_for,
)


def _take(table: pa.Table, indexes: list[int]) -> pa.Table:
    if not indexes:
        return table.slice(0, 0)
    return table.take(pa.array(indexes, type=pa.int64()))


def _write_split_sidecar(
    source_sidecar: dict[str, Any],
    output_parquet: Path,
    *,
    split_name: str,
    row_count: int,
    doc_count: int,
    source_parquet: Path,
    split_unit_kind: str,
    split_unit_count: int,
    padding_duplicate_count: int = 0,
) -> None:
    sidecar = dict(source_sidecar)
    sidecar["row_count"] = row_count
    sidecar["split"] = {
        "name": split_name,
        "source_parquet": str(source_parquet),
        "doc_count": doc_count,
        "row_count": row_count,
        "split_unit_column": "split_unit_id",
        "split_unit_kind": split_unit_kind,
        "split_unit_count": split_unit_count,
        "padding_duplicate_count": padding_duplicate_count,
    }
    sidecar_path_for(output_parquet).write_text(yaml.safe_dump(sidecar, sort_keys=False))


def _doc_to_rows(table: pa.Table) -> dict[str, list[int]]:
    names = table.schema.names
    if "doc_id" in names:
        docs = table.column("doc_id").to_pylist()
    else:
        docs = [f"__row_{idx}" for idx in range(table.num_rows)]
    out: dict[str, list[int]] = defaultdict(list)
    for idx, doc_id in enumerate(docs):
        out[str(doc_id or f"__row_{idx}")].append(idx)
    return dict(out)


def _with_string_column(
    table: pa.Table,
    name: str,
    values: list[str],
) -> pa.Table:
    if len(values) != table.num_rows:
        raise ValueError(
            f"{name} row count mismatch: {len(values)} != {table.num_rows}"
        )
    array = pa.array(values, type=pa.string())
    if name not in table.schema.names:
        return table.append_column(name, array)
    existing = [str(value) for value in table.column(name).to_pylist()]
    if existing != values:
        raise ValueError(f"existing {name} column disagrees with split assignment")
    return table


def _lowest_doc_key(doc_id: str) -> tuple[int, str]:
    import re

    match = re.search(r":(\d+)$", str(doc_id))
    if match is None:
        return (10**18, str(doc_id))
    return (int(match.group(1)), str(doc_id))


def _content_components(table: pa.Table) -> tuple[dict[str, list[str]], dict[str, Any]]:
    doc_to_rows = _doc_to_rows(table)
    parents = {doc_id: doc_id for doc_id in doc_to_rows}
    names = table.schema.names
    content_columns = _content_column_order(names)
    content_values = {name: table.column(name).to_pylist() for name in content_columns}
    key_to_docs: dict[str, set[str]] = defaultdict(set)

    def find(doc_id: str) -> str:
        parent = parents[doc_id]
        if parent != doc_id:
            parents[doc_id] = find(parent)
        return parents[doc_id]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        keep, drop = sorted([left_root, right_root], key=_lowest_doc_key)
        parents[drop] = keep

    docs = table.column("doc_id").to_pylist() if "doc_id" in names else [f"__row_{idx}" for idx in range(table.num_rows)]
    for row_index, doc_id_value in enumerate(docs):
        doc_id = str(doc_id_value or f"__row_{row_index}")
        fallback = None
        if "prompt" in names:
            fallback = table.column("prompt")[row_index].as_py()
        content_key = _row_content_key(content_values, row_index, fallback=fallback)
        key_to_docs[content_key].add(doc_id)

    duplicate_groups = []
    for content_key, docs_for_key in sorted(key_to_docs.items()):
        doc_ids = sorted(docs_for_key, key=_lowest_doc_key)
        if len(doc_ids) <= 1:
            continue
        for doc_id in doc_ids[1:]:
            union(doc_ids[0], doc_id)
        duplicate_groups.append(
            {
                "content_hash": content_key,
                "doc_count": len(doc_ids),
                "doc_ids_sample": doc_ids[:10],
            }
        )

    components: dict[str, list[str]] = defaultdict(list)
    for doc_id in doc_to_rows:
        components[find(doc_id)].append(doc_id)
    normalized = {
        root: sorted(component_docs, key=_lowest_doc_key)
        for root, component_docs in components.items()
    }
    duplicate_components = [
        {
            "component_id": root,
            "doc_count": len(component_docs),
            "doc_ids_sample": component_docs[:10],
        }
        for root, component_docs in sorted(normalized.items())
        if len(component_docs) > 1
    ]
    report = {
        "content_columns": content_columns,
        "component_count": len(normalized),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_component_count": len(duplicate_components),
        "duplicate_doc_count": sum(item["doc_count"] for item in duplicate_components),
        "duplicate_groups_sample": duplicate_groups[:20],
        "duplicate_components_sample": duplicate_components[:20],
    }
    return normalized, report


def _pad_indexes(indexes: list[int], multiple: int | None) -> tuple[list[int], int]:
    if multiple is None or multiple <= 0:
        return list(indexes), 0
    if not indexes:
        raise ValueError("cannot pad an empty train split")
    remainder = len(indexes) % multiple
    if remainder == 0:
        return list(indexes), 0
    duplicate_count = multiple - remainder
    return list(indexes) + [indexes[i % len(indexes)] for i in range(duplicate_count)], duplicate_count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _splits_from_family_manifest(
    manifest_path: Path,
    *,
    docs: list[str],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[
    dict[str, list[str]],
    dict[str, list[str]],
    list[str],
    dict[str, Any],
]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "nano_content_family_manifest.v1":
        raise ValueError(
            "content family manifest must use schema_version nano_content_family_manifest.v1"
        )
    assignment = manifest.get("split_assignment") or {}
    expected_weights = {
        "train": float(train_fraction),
        "validation": float(validation_fraction),
        "test": float(test_fraction),
    }
    actual_weights = assignment.get("weights") or {}
    if int(assignment.get("seed", -1)) != int(seed):
        raise ValueError("content family manifest seed does not match dataset seed")
    if set(actual_weights) != set(expected_weights) or any(
        abs(float(actual_weights[name]) - expected_weights[name]) > 1e-12
        for name in expected_weights
    ):
        raise ValueError("content family manifest weights do not match dataset fractions")

    doc_assignments = manifest.get("doc_assignments") or {}
    family_splits = manifest.get("family_splits") or {}
    missing_docs = [doc_id for doc_id in docs if not doc_assignments.get(doc_id)]
    if missing_docs:
        raise ValueError(
            f"content family manifest is missing family assignments for {len(missing_docs)} documents: "
            f"{missing_docs[:5]}"
        )

    split_docs = {"train": [], "validation": [], "test": []}
    split_families = {"train": set(), "validation": set(), "test": set()}
    split_units: dict[str, list[str]] = defaultdict(list)
    for doc_id in docs:
        family_id = str(doc_assignments[doc_id])
        split = family_splits.get(family_id)
        if split not in split_docs:
            raise ValueError(
                f"content family {family_id!r} has invalid or missing split {split!r}"
            )
        split_docs[split].append(doc_id)
        split_families[split].add(family_id)
        split_units[family_id].append(doc_id)

    overlap = (
        (split_families["train"] & split_families["validation"])
        | (split_families["train"] & split_families["test"])
        | (split_families["validation"] & split_families["test"])
    )
    if overlap:
        raise ValueError(f"content family split overlap: {sorted(overlap)[:5]}")
    split_units_by_name = {
        split: sorted(families) for split, families in split_families.items()
    }
    unit_ids = sorted(split_units)
    metadata = {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "schema_version": manifest["schema_version"],
        "seed": int(seed),
        "family_count": len(unit_ids),
        "split_family_counts": {
            split: len(families) for split, families in split_families.items()
        },
        "overlap_count": 0,
    }
    return split_docs, split_units_by_name, unit_ids, metadata


def materialize_splits(
    parquet_path: str | Path,
    output_dir: str | Path,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int = 42,
    row_limit: int | None = None,
    pad_train_to_multiple: int | None = None,
    split_mode: str = "doc",
    content_family_manifest: str | Path | None = None,
) -> dict[str, Any]:
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(parquet_path)
    if row_limit is not None:
        table = table.slice(0, row_limit)
    source_sidecar = yaml.safe_load(sidecar_path_for(parquet_path).read_text())
    doc_to_rows = _doc_to_rows(table)
    docs = sorted(doc_to_rows)
    family_manifest_metadata: dict[str, Any] | None = None
    doc_to_unit: dict[str, str]
    if split_mode == "content_family_manifest":
        if content_family_manifest is None:
            raise ValueError(
                "content_family_manifest is required for content_family_manifest split mode"
            )
        split_docs, split_units_by_name, unit_ids, family_manifest_metadata = (
            _splits_from_family_manifest(
                Path(content_family_manifest),
                docs=docs,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                seed=seed,
            )
        )
        component_report = {
            "content_columns": [],
            "component_count": len(unit_ids),
            "duplicate_group_count": 0,
            "duplicate_component_count": 0,
            "duplicate_doc_count": 0,
            "duplicate_groups_sample": [],
            "duplicate_components_sample": [],
        }
        family_manifest_doc = json.loads(Path(content_family_manifest).read_text())
        doc_to_unit = {
            doc_id: str(family_manifest_doc["doc_assignments"][doc_id])
            for doc_id in docs
        }
        split_unit_kind = "content_family"
    elif split_mode == "doc":
        split_units = {doc_id: [doc_id] for doc_id in docs}
        doc_to_unit = {doc_id: doc_id for doc_id in docs}
        split_unit_kind = "document"
        component_report = {
            "content_columns": [],
            "component_count": len(split_units),
            "duplicate_group_count": 0,
            "duplicate_component_count": 0,
            "duplicate_doc_count": 0,
            "duplicate_groups_sample": [],
            "duplicate_components_sample": [],
        }
    elif split_mode == "content_component":
        split_units, component_report = _content_components(table)
        doc_to_unit = {
            doc_id: str(unit_id)
            for unit_id, unit_docs in split_units.items()
            for doc_id in unit_docs
        }
        split_unit_kind = "content_component"
    else:
        raise ValueError(
            f"unsupported split_mode {split_mode!r}; expected doc, content_component, "
            "or content_family_manifest"
        )
    if split_mode != "content_family_manifest":
        unit_ids = sorted(split_units, key=_lowest_doc_key)
        train_units, validation_units, test_units = _split_items_three_way(
            unit_ids, train_fraction, validation_fraction, test_fraction, seed
        )
        split_docs = {
            "train": [doc for unit in train_units for doc in split_units[unit]],
            "validation": [doc for unit in validation_units for doc in split_units[unit]],
            "test": [doc for unit in test_units for doc in split_units[unit]],
        }
        split_units_by_name = {
            "train": train_units,
            "validation": validation_units,
            "test": test_units,
        }
    overlap = (
        (set(split_docs["train"]) & set(split_docs["validation"]))
        | (set(split_docs["train"]) & set(split_docs["test"]))
        | (set(split_docs["validation"]) & set(split_docs["test"]))
    )
    if overlap:
        raise ValueError(f"doc split overlap: {sorted(overlap)[:5]}")
    missing_units = sorted(set(docs) - set(doc_to_unit))
    if missing_units:
        raise ValueError(f"documents are missing split units: {missing_units[:5]}")

    row_units = [""] * table.num_rows
    for doc_id, indexes in doc_to_rows.items():
        for index in indexes:
            row_units[index] = doc_to_unit[doc_id]
    if any(not value for value in row_units):
        raise ValueError("split-unit materialization produced empty row assignments")
    table = _with_string_column(table, "split_unit_id", row_units)
    if split_unit_kind == "content_family":
        table = _with_string_column(table, "content_family_id", row_units)
    elif split_unit_kind == "content_component":
        table = _with_string_column(table, "content_component_id", row_units)

    manifest: dict[str, Any] = {
        "schema_version": "nano_split_manifest.v2",
        "source_parquet": str(parquet_path),
        "output_dir": str(output_dir),
        "split_mode": split_mode,
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "source_row_count": table.num_rows,
        "source_doc_count": len(docs),
        "source_split_unit_count": len(unit_ids),
        "split_unit_column": "split_unit_id",
        "split_unit_kind": split_unit_kind,
        "doc_overlap_count": 0,
        "content_components": component_report,
        "family_overlap_count": 0,
        "splits": {},
    }
    if family_manifest_metadata is not None:
        manifest["content_family_manifest"] = family_manifest_metadata
        manifest["content_family_manifest_sha256"] = family_manifest_metadata[
            "sha256"
        ]

    for split_name, docs_for_split in split_docs.items():
        indexes = [row for doc in docs_for_split for row in doc_to_rows[doc]]
        split_table = _take(table, indexes)
        out_path = output_dir / f"{split_name}.parquet"
        pq.write_table(split_table, out_path)
        _write_split_sidecar(
            source_sidecar,
            out_path,
            split_name=split_name,
            row_count=split_table.num_rows,
            doc_count=len(docs_for_split),
            source_parquet=parquet_path,
            split_unit_kind=split_unit_kind,
            split_unit_count=len(split_units_by_name[split_name]),
        )
        manifest["splits"][split_name] = {
            "path": str(out_path),
            "row_count": split_table.num_rows,
            "doc_count": len(docs_for_split),
            "split_unit_count": len(split_units_by_name[split_name]),
            "split_unit_ids": list(split_units_by_name[split_name]),
            "docs": list(docs_for_split),
        }

    train_docs = split_docs["train"]
    train_indexes = [row for doc in train_docs for row in doc_to_rows[doc]]
    padded_indexes, duplicate_count = _pad_indexes(train_indexes, pad_train_to_multiple)
    train_padded_path = output_dir / "train_padded.parquet"
    train_padded = _take(table, padded_indexes)
    pq.write_table(train_padded, train_padded_path)
    _write_split_sidecar(
        source_sidecar,
        train_padded_path,
        split_name="train_padded",
        row_count=train_padded.num_rows,
        doc_count=len(train_docs),
        source_parquet=parquet_path,
        split_unit_kind=split_unit_kind,
        split_unit_count=len(split_units_by_name["train"]),
        padding_duplicate_count=duplicate_count,
    )
    manifest["train"] = {
        "path": manifest["splits"]["train"]["path"],
        "padded_path": str(train_padded_path),
        "row_count": manifest["splits"]["train"]["row_count"],
        "padded_row_count": train_padded.num_rows,
        "padding_duplicate_count": duplicate_count,
        "pad_train_to_multiple": pad_train_to_multiple,
        "split_unit_ids": list(split_units_by_name["train"]),
    }
    manifest["validation"] = manifest["splits"]["validation"]
    manifest["test"] = manifest["splits"]["test"]

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, required=True)
    parser.add_argument("--validation-fraction", type=float, required=True)
    parser.add_argument("--test-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--pad-train-to-multiple", type=int)
    parser.add_argument(
        "--split-mode",
        choices=["doc", "content_component", "content_family_manifest"],
        default="doc",
    )
    parser.add_argument("--content-family-manifest", type=Path)
    args = parser.parse_args()

    manifest = materialize_splits(
        args.parquet,
        args.output_dir,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        row_limit=args.row_limit,
        pad_train_to_multiple=args.pad_train_to_multiple,
        split_mode=args.split_mode,
        content_family_manifest=args.content_family_manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
