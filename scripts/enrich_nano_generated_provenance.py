#!/usr/bin/env python3
"""Add stable split-row provenance to historical generated explanation JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SCHEMA_VERSION = "nano_generated_provenance_enrichment.v1"
PROVENANCE_COLUMNS = (
    "sample_uuid",
    "doc_id",
    "token_position",
    "n_raw_tokens",
    "token_id",
)


class ProvenanceEnrichmentError(ValueError):
    """Raised when historical generated rows cannot be joined exactly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProvenanceEnrichmentError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            rows.append(value)
    if not rows:
        raise ProvenanceEnrichmentError(f"generated JSONL is empty: {path}")
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _split_layout(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    offset = 0
    layout: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        path = paths[split]
        parquet = pq.ParquetFile(path)
        row_count = parquet.metadata.num_rows
        layout[split] = {
            "path": path,
            "parquet": parquet,
            "offset": offset,
            "row_count": row_count,
        }
        offset += row_count
    return layout


def enrich_generated_jsonl(
    *,
    generated_jsonl: str | Path,
    train_parquet: str | Path,
    validation_parquet: str | Path,
    test_parquet: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path | None = None,
    batch_size: int = 4_096,
    overwrite: bool = False,
) -> dict[str, Any]:
    generated_path = Path(generated_jsonl)
    output_path = Path(output_jsonl)
    if batch_size <= 0:
        raise ProvenanceEnrichmentError("batch_size must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass overwrite=True to replace it")
    split_paths = {
        "train": Path(train_parquet),
        "validation": Path(validation_parquet),
        "test": Path(test_parquet),
    }
    layout = _split_layout(split_paths)
    rows = _read_jsonl(generated_path)
    requested_by_split: dict[str, dict[int, int]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    seen_global: set[int] = set()
    for output_index, row in enumerate(rows):
        split = str(row.get("split"))
        if split not in layout:
            raise ProvenanceEnrichmentError(f"unknown generated split: {split!r}")
        global_index = int(row.get("row_index", -1))
        split_start = int(layout[split]["offset"])
        split_end = split_start + int(layout[split]["row_count"])
        if not split_start <= global_index < split_end:
            raise ProvenanceEnrichmentError(
                f"row_index={global_index} is out of range for split={split} "
                f"[{split_start}, {split_end})"
            )
        if global_index in seen_global:
            raise ProvenanceEnrichmentError(
                f"generated JSONL has duplicate row_index={global_index}"
            )
        seen_global.add(global_index)
        requested_by_split[split][global_index - split_start] = output_index

    found = 0
    copied_fields: set[str] = set()
    for split, requested in requested_by_split.items():
        if not requested:
            continue
        parquet = layout[split]["parquet"]
        schema_names = set(parquet.schema_arrow.names)
        columns = [name for name in PROVENANCE_COLUMNS if name in schema_names]
        if "doc_id" not in columns or not ({"token_position", "n_raw_tokens"} & set(columns)):
            raise ProvenanceEnrichmentError(
                f"{split} parquet lacks doc_id plus token_position/n_raw_tokens"
            )
        local_offset = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            batch_end = local_offset + batch.num_rows
            relevant = sorted(
                index for index in requested if local_offset <= index < batch_end
            )
            if relevant:
                values = batch.to_pylist()
                for local_index in relevant:
                    source = values[local_index - local_offset]
                    target = rows[requested[local_index]]
                    for name in columns:
                        value = source.get(name)
                        if value is None:
                            continue
                        if target.get(name) is not None and target[name] != value:
                            raise ProvenanceEnrichmentError(
                                f"conflicting {name} for row_index={target['row_index']}"
                            )
                        target[name] = value
                        copied_fields.add(name)
                    found += 1
            local_offset = batch_end
            if found == len(rows):
                break
    missing = len(rows) - found
    if missing:
        raise ProvenanceEnrichmentError(f"failed to resolve {missing} generated rows")
    _write_jsonl_atomic(output_path, rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_jsonl": str(generated_path),
        "generated_sha256": _sha256(generated_path),
        "output_jsonl": str(output_path),
        "output_sha256": _sha256(output_path),
        "rows": len(rows),
        "missing_rows": missing,
        "copied_fields": sorted(copied_fields),
        "split_rows": {
            split: len(requested) for split, requested in requested_by_split.items()
        },
        "split_parquets": {split: str(path) for split, path in split_paths.items()},
    }
    if report_json is not None:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(report_path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = enrich_generated_jsonl(
        generated_jsonl=args.generated_jsonl,
        train_parquet=args.train_parquet,
        validation_parquet=args.validation_parquet,
        test_parquet=args.test_parquet,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
