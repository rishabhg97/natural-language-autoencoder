#!/usr/bin/env python3
"""Inventory teacher-backed Nano datasets and external document coverage."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SCHEMA_VERSION = "nano_teacher_corpus_inventory.v1"
REPORT_SCHEMA_VERSION = "nano_teacher_corpus_inventory_report.v1"
DIRECT_TEXT_COLUMNS = ("api_explanation", "explanation", "teacher_explanation")
JOIN_KEY_COLUMNS = ("token_position", "n_raw_tokens", "token_id")
TEXT_OPEN = "<text>"
TEXT_CLOSE = "</text>"


def _extract_prompt_text(prompt: Any) -> str:
    text = str(prompt or "")
    start = text.find(TEXT_OPEN)
    if start < 0:
        return ""
    start += len(TEXT_OPEN)
    end = text.find(TEXT_CLOSE, start)
    return (text[start:] if end < 0 else text[start:end]).strip()


def _doc_suffix(doc_id: str) -> int | None:
    match = re.search(r"(\d+)$", doc_id)
    return int(match.group(1)) if match else None


def _resolve(value: Any, *, config_path: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else config_path.parent / path


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config must use schema_version {SCHEMA_VERSION}")
    if not isinstance(config.get("roots"), list) or not config["roots"]:
        raise ValueError("roots must be a non-empty list")
    if not isinstance(config.get("patterns"), list) or not config["patterns"]:
        raise ValueError("patterns must be a non-empty list")
    if not (config.get("outputs") or {}).get("report_json"):
        raise ValueError("outputs.report_json is required")
    return config


def _candidate_paths(config: dict[str, Any], *, config_path: Path) -> list[Path]:
    candidates: set[Path] = set()
    for root_value in config["roots"]:
        root = _resolve(root_value, config_path=config_path)
        if not root.is_dir():
            continue
        for pattern in config["patterns"]:
            candidates.update(
                path.resolve()
                for path in root.glob(str(pattern))
                if path.is_file() and not path.name.startswith("._")
            )
    return sorted(candidates)


def _d_model(schema: pa.Schema) -> int | None:
    if "activation_vector" not in schema.names:
        return None
    value_type = schema.field("activation_vector").type
    return int(value_type.list_size) if pa.types.is_fixed_size_list(value_type) else None


def _outside_ranges(value: int, ranges: list[tuple[int, int]]) -> bool:
    return not any(start <= value <= end for start, end in ranges)


def audit_table(
    path: Path,
    *,
    known_exposed_ranges: list[tuple[int, int]],
    batch_size: int,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    names = set(schema.names)
    direct_text = next((name for name in DIRECT_TEXT_COLUMNS if name in names), None)
    text_column = direct_text or ("prompt" if "prompt" in names else None)
    text_mode = (
        "direct"
        if direct_text is not None
        else "prompt_text_tags"
        if text_column == "prompt"
        else None
    )
    report: dict[str, Any] = {
        "path": str(path),
        "row_count": int(parquet.metadata.num_rows),
        "columns": sorted(schema.names),
        "text_column": text_column,
        "text_mode": text_mode,
        "join_keys": [name for name in JOIN_KEY_COLUMNS if name in names],
        "has_doc_id": "doc_id" in names,
        "d_model": _d_model(schema),
    }
    if "doc_id" not in names:
        return report

    columns = ["doc_id"] + ([text_column] if text_column else [])
    unique_docs: set[str] = set()
    suffixes: set[int] = set()
    external_docs: set[str] = set()
    nonnumeric_docs: set[str] = set()
    empty_explanations = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            doc_id = str(row.get("doc_id") or "").strip()
            if not doc_id:
                continue
            unique_docs.add(doc_id)
            suffix = _doc_suffix(doc_id)
            if suffix is None:
                nonnumeric_docs.add(doc_id)
            else:
                suffixes.add(suffix)
                if _outside_ranges(suffix, known_exposed_ranges):
                    external_docs.add(doc_id)
            if text_column:
                explanation = (
                    _extract_prompt_text(row.get(text_column))
                    if text_mode == "prompt_text_tags"
                    else str(row.get(text_column) or "").strip()
                )
                if not explanation:
                    empty_explanations += 1

    contiguous = None
    if suffixes:
        contiguous = len(suffixes) == max(suffixes) - min(suffixes) + 1
    report.update(
        {
            "unique_doc_count": len(unique_docs),
            "numeric_doc_suffix_min": min(suffixes) if suffixes else None,
            "numeric_doc_suffix_max": max(suffixes) if suffixes else None,
            "numeric_doc_suffix_unique_count": len(suffixes),
            "numeric_doc_suffix_contiguous": contiguous,
            "nonnumeric_doc_count": len(nonnumeric_docs),
            "external_numeric_doc_count": len(external_docs),
            "external_numeric_doc_sample": sorted(external_docs)[:20],
            "empty_explanation_count": (
                empty_explanations if text_column is not None else None
            ),
            "usable_teacher_text": bool(
                text_column is not None and empty_explanations == 0
            ),
            "usable_join_keys": bool(
                "doc_id" in names and any(name in names for name in JOIN_KEY_COLUMNS[:2])
            ),
        }
    )
    return report


def run_audit(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    output = _resolve(config["outputs"]["report_json"], config_path=config_path)
    if output.exists():
        raise FileExistsError(f"frozen teacher inventory exists: {output}")
    ranges = [
        (int(values[0]), int(values[1]))
        for values in config.get("known_exposed_numeric_doc_ranges", [])
    ]
    tables = [
        audit_table(
            path,
            known_exposed_ranges=ranges,
            batch_size=int(config.get("batch_size", 8192)),
        )
        for path in _candidate_paths(config, config_path=config_path)
    ]
    usable = [table for table in tables if table.get("usable_teacher_text")]
    external = [
        table for table in usable if int(table.get("external_numeric_doc_count") or 0) > 0
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "config": str(config_path),
        "known_exposed_numeric_doc_ranges": [list(values) for values in ranges],
        "summary": {
            "candidate_table_count": len(tables),
            "usable_teacher_table_count": len(usable),
            "external_teacher_table_count": len(external),
            "max_numeric_doc_suffix": max(
                (
                    int(table["numeric_doc_suffix_max"])
                    for table in usable
                    if table.get("numeric_doc_suffix_max") is not None
                ),
                default=None,
            ),
        },
        "tables": tables,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    report = run_audit(args.config)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
