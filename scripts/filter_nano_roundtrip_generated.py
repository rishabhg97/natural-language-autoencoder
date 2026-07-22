#!/usr/bin/env python3
"""Select cached round-trip generations by canonical reference row identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROW_KEY_FIELDS = (
    "doc_id",
    "token_position",
    "n_raw_tokens",
    "token_id",
    "sample_uuid",
)


class GeneratedFilterError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GeneratedFilterError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            records.append(value)
    return records


def canonical_identity(record: dict[str, Any]) -> str:
    row_key = {
        field: record[field]
        for field in ROW_KEY_FIELDS
        if record.get(field) is not None
    }
    if not row_key:
        raise GeneratedFilterError("generated record has no canonical row key")
    return json.dumps(
        {"split": str(record.get("split") or ""), "row_key": row_key},
        sort_keys=True,
        separators=(",", ":"),
    )


def unique_by_identity(
    records: Iterable[dict[str, Any]], *, source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        identity = canonical_identity(record)
        if identity in indexed:
            raise GeneratedFilterError(f"duplicate {source} identity: {identity}")
        indexed[identity] = record
    return indexed


def select_records(
    source_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_identity = unique_by_identity(source_records, source="source")
    reference_by_identity = unique_by_identity(reference_records, source="reference")
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for identity in reference_by_identity:
        record = source_by_identity.get(identity)
        if record is None:
            missing.append(identity)
        else:
            selected.append(record)
    if missing:
        raise GeneratedFilterError(
            f"source is missing {len(missing)} reference rows; first={missing[0]}"
        )
    return selected


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    source_records = read_jsonl(args.source_jsonl)
    reference_records = read_jsonl(args.reference_jsonl)
    selected = select_records(source_records, reference_records)
    if args.expected_rows is not None and len(selected) != args.expected_rows:
        raise GeneratedFilterError(
            f"selected {len(selected)} rows, expected {args.expected_rows}"
        )
    write_jsonl_atomic(args.output_jsonl, selected)
    report = {
        "schema_version": "nano_roundtrip_generated_filter.v1",
        "source_jsonl": str(args.source_jsonl),
        "source_sha256": file_sha256(args.source_jsonl),
        "source_rows": len(source_records),
        "reference_jsonl": str(args.reference_jsonl),
        "reference_sha256": file_sha256(args.reference_jsonl),
        "reference_rows": len(reference_records),
        "output_jsonl": str(args.output_jsonl),
        "output_sha256": file_sha256(args.output_jsonl),
        "output_rows": len(selected),
        "order": "reference",
        "row_key_fields": list(ROW_KEY_FIELDS),
        "passed": True,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
