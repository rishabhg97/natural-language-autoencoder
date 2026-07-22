#!/usr/bin/env python3
"""Deduplicate teacher-key rows by source-text prefix before fresh layer extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


TEXT_OPEN = "<text>"
TEXT_CLOSE = "</text>"
DEFAULT_KEEP_COLUMNS = (
    "doc_id",
    "token_ids_prefix",
    "n_raw_tokens",
    "token_position",
    "token_id",
    "token_text",
    "detokenized_text_truncated",
)
DEFAULT_CONTENT_COLUMNS = ("token_ids_prefix", "detokenized_text_truncated")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extract_explanation_from_prompt(prompt: str) -> str:
    start = prompt.find(TEXT_OPEN)
    if start < 0:
        return ""
    start += len(TEXT_OPEN)
    end = prompt.find(TEXT_CLOSE, start)
    if end < 0:
        return ""
    return prompt[start:end].strip()


def doc_numeric_suffix(doc_id: str) -> int | None:
    match = re.search(r":(\d+)$", str(doc_id))
    if match is None:
        return None
    return int(match.group(1))


def _content_payload(value: Any, *, max_tokens: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.strip().split()
        if not parts:
            return None
        return json.dumps(parts[:max_tokens], ensure_ascii=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            return json.dumps([str(item) for item in value[:max_tokens]], ensure_ascii=True, separators=(",", ":"))
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=True, separators=(",", ":"))
    text = str(value).strip()
    return text or None


def _content_key(value: Any, *, column: str, max_tokens: int) -> str | None:
    payload = _content_payload(value, max_tokens=max_tokens)
    if payload is None:
        return None
    digest = hashlib.sha256(f"{column}\0{payload}".encode("utf-8")).hexdigest()
    return f"{column}:{digest}"


def _lowest_doc_key(doc_id: str) -> tuple[int, str]:
    suffix = doc_numeric_suffix(doc_id)
    return (suffix if suffix is not None else 10**18, doc_id)


def _doc_content_hashes(table: pa.Table, *, max_tokens: int) -> tuple[dict[str, set[str]], list[str]]:
    docs = table.column("doc_id").to_pylist()
    content_columns = [name for name in DEFAULT_CONTENT_COLUMNS if name in table.column_names]
    content_values = {name: table.column(name).to_pylist() for name in content_columns}
    hashes_by_doc: dict[str, set[str]] = defaultdict(set)
    for name, values in content_values.items():
        if len(docs) != len(values):
            raise ValueError(f"doc_id/{name} length mismatch: {len(docs)} != {len(values)}")
    for row_index, doc_id_value in enumerate(docs):
        doc_id = str(doc_id_value)
        for name, values in content_values.items():
            key = _content_key(values[row_index], column=name, max_tokens=max_tokens)
            if key is not None:
                hashes_by_doc[doc_id].add(key)
        if not hashes_by_doc[doc_id]:
            raise ValueError(f"row {row_index}: no usable content key in columns {content_columns}")
    return dict(hashes_by_doc), content_columns


def _selected_docs(doc_to_hashes: dict[str, set[str]]) -> tuple[set[str], dict[str, Any]]:
    hash_to_docs: dict[str, set[str]] = defaultdict(set)
    for doc_id, content_hashes in doc_to_hashes.items():
        for content_hash in content_hashes:
            hash_to_docs[content_hash].add(doc_id)

    parents = {doc_id: doc_id for doc_id in doc_to_hashes}

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

    duplicate_groups = []
    for content_hash, doc_ids_set in sorted(hash_to_docs.items()):
        doc_ids = sorted(doc_ids_set, key=_lowest_doc_key)
        if len(doc_ids) <= 1:
            continue
        for doc_id in doc_ids[1:]:
            union(doc_ids[0], doc_id)
        duplicate_groups.append(
            {
                "content_hash": content_hash,
                "doc_ids_sample": doc_ids[:11],
                "doc_count": len(doc_ids),
            }
        )

    component_to_docs: dict[str, list[str]] = defaultdict(list)
    for doc_id in doc_to_hashes:
        component_to_docs[find(doc_id)].append(doc_id)

    selected: set[str] = set()
    duplicate_components = []
    for docs in component_to_docs.values():
        docs = sorted(docs, key=_lowest_doc_key)
        selected.add(docs[0])
        if len(docs) > 1:
            duplicate_components.append(
                {
                    "kept_doc_id": docs[0],
                    "dropped_doc_ids_sample": docs[1:11],
                    "doc_count": len(docs),
                }
            )

    duplicate_docs = {doc_id for docs in component_to_docs.values() if len(docs) > 1 for doc_id in docs}
    report = {
        "source_doc_count": len(doc_to_hashes),
        "kept_doc_count": len(selected),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_doc_count": len(duplicate_docs),
        "duplicate_component_count": len(duplicate_components),
        "dropped_doc_count": len(doc_to_hashes) - len(selected),
        "duplicate_groups_sample": duplicate_groups[:20],
        "duplicate_components_sample": duplicate_components[:20],
    }
    return selected, report


def _suffix_report(doc_ids: list[str]) -> dict[str, Any]:
    suffixes = sorted(value for doc_id in doc_ids if (value := doc_numeric_suffix(doc_id)) is not None)
    if not suffixes:
        return {
            "numeric_doc_suffix_count": 0,
            "numeric_doc_suffix_min": None,
            "numeric_doc_suffix_max": None,
            "numeric_doc_suffix_contiguous": None,
        }
    expected = suffixes[-1] - suffixes[0] + 1
    return {
        "numeric_doc_suffix_count": len(suffixes),
        "numeric_doc_suffix_min": suffixes[0],
        "numeric_doc_suffix_max": suffixes[-1],
        "numeric_doc_suffix_contiguous": expected == len(set(suffixes)),
    }


def build_dedup_teacher_keys(
    *,
    source: str | Path,
    output: str | Path,
    max_prefix_tokens: int = 300,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    pf = pq.ParquetFile(source)
    names = set(pf.schema_arrow.names)
    required = {"doc_id", "token_ids_prefix"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")

    read_columns = [name for name in [*DEFAULT_KEEP_COLUMNS, "api_explanation", "prompt"] if name in names]
    table = pq.read_table(source, columns=read_columns)
    doc_to_hashes, content_columns = _doc_content_hashes(table, max_tokens=max_prefix_tokens)
    selected_docs, dedup_report = _selected_docs(doc_to_hashes)

    rows = table.to_pylist()
    kept_rows: dict[str, list[Any]] = {name: [] for name in DEFAULT_KEEP_COLUMNS if name in table.column_names}
    kept_rows["api_explanation"] = []
    empty_explanations = 0
    kept_doc_ids: list[str] = []
    for row in rows:
        doc_id = str(row["doc_id"])
        if doc_id not in selected_docs:
            continue
        kept_doc_ids.append(doc_id)
        for name in kept_rows:
            if name == "api_explanation":
                explanation = str(row.get("api_explanation") or "").strip()
                if not explanation:
                    prompt = row.get("prompt")
                    explanation = extract_explanation_from_prompt(prompt) if isinstance(prompt, str) else ""
                if not explanation:
                    empty_explanations += 1
                kept_rows[name].append(explanation)
            else:
                kept_rows[name].append(row.get(name))

    if empty_explanations:
        raise ValueError(f"{empty_explanations} kept rows have empty api_explanation")

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(kept_rows), output)
    kept_unique_docs = sorted(set(kept_doc_ids), key=_lowest_doc_key)
    report = {
        "schema_version": "nano_dedup_teacher_keys.v1",
        "created_at": utc_now(),
        "source": str(source),
        "output": str(output),
        "content_hash_mode": f"row_content_first_{max_prefix_tokens}_components",
        "content_columns": content_columns,
        "source_rows": pf.metadata.num_rows,
        "output_rows": len(kept_doc_ids),
        "dropped_rows": pf.metadata.num_rows - len(kept_doc_ids),
        "empty_api_explanation": empty_explanations,
        **dedup_report,
        "source_suffixes": _suffix_report(list(doc_to_hashes)),
        "kept_suffixes": _suffix_report(kept_unique_docs),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prefix-tokens", type=int, default=300)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = build_dedup_teacher_keys(
        source=args.source,
        output=args.output,
        max_prefix_tokens=args.max_prefix_tokens,
        overwrite=args.overwrite,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
