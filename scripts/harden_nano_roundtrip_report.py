#!/usr/bin/env python3
"""Bind an existing round-trip report to dataset bytes and stable row provenance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_nano_av_ar_roundtrip_gate import file_provenance, write_json  # noqa: E402


ROW_KEY_FIELDS = ("doc_id", "token_position", "n_raw_tokens", "token_id", "sample_uuid")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def harden_report(
    *,
    report_json: str | Path,
    generated_jsonl: str | Path,
    train_parquet: str | Path,
    validation_parquet: str | Path,
    test_parquet: str | Path,
) -> dict[str, Any]:
    report_path = Path(report_json)
    generated_path = Path(generated_jsonl)
    report = json.loads(report_path.read_text())
    records = read_jsonl(generated_path)
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("split")), int(record.get("row_index", -1)))
        if key in by_key:
            raise ValueError(f"duplicate generated row provenance: {key}")
        by_key[key] = record

    for split_name, split in (report.get("splits") or {}).items():
        row_indices = split.get("row_indices")
        if not isinstance(row_indices, list):
            raise ValueError(f"report split {split_name!r} has no row_indices")
        split_records = []
        for row_index in row_indices:
            key = (str(split_name), int(row_index))
            if key not in by_key:
                raise ValueError(f"generated provenance missing report row: {key}")
            split_records.append(by_key[key])
        doc_ids = [str(record.get("doc_id") or "") for record in split_records]
        if not all(doc_ids):
            raise ValueError(f"generated provenance has empty doc_id in split {split_name!r}")
        split["doc_ids"] = doc_ids
        split["row_keys"] = [
            {
                field: record[field]
                for field in ROW_KEY_FIELDS
                if record.get(field) is not None
            }
            for record in split_records
        ]

    report["dataset_provenance"] = {
        "train": file_provenance(train_parquet),
        "validation": file_provenance(validation_parquet),
        "test": file_provenance(test_parquet),
    }
    report["provenance_hardening"] = {
        "schema_version": "nano_roundtrip_report_hardening.v1",
        "source_report": str(report_path),
        "source_generated_jsonl": str(generated_path),
        "generated_row_count": len(records),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = harden_report(
        report_json=args.report_json,
        generated_jsonl=args.generated_jsonl,
        train_parquet=args.train_parquet,
        validation_parquet=args.validation_parquet,
        test_parquet=args.test_parquet,
    )
    write_json(args.output_json, report)
    print(json.dumps({"output_json": str(args.output_json), "dataset_provenance": report["dataset_provenance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
