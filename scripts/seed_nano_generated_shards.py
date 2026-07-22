#!/usr/bin/env python3
"""Seed round-trip generation worker shards from an existing partial JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class ShardSeedError(ValueError):
    """Raised when existing generated records cannot seed the target layout."""


def shard_generated_records(
    records: list[dict[str, Any]],
    *,
    split_starts: dict[str, int],
    validation_limit: int,
    test_limit: int,
    shard_count: int,
) -> list[list[dict[str, Any]]]:
    if validation_limit <= 0 or test_limit <= 0 or shard_count <= 0:
        raise ShardSeedError("limits and shard_count must be positive")
    limits = {"validation": validation_limit, "test": test_limit}
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    seen: set[tuple[str, int]] = set()
    for record in records:
        split = str(record.get("split"))
        if split not in limits or split not in split_starts:
            raise ShardSeedError(f"unsupported generated split: {split!r}")
        row_index = int(record.get("row_index", -1))
        key = (split, row_index)
        if key in seen:
            raise ShardSeedError(f"duplicate generated row: {key}")
        seen.add(key)
        local_index = row_index - int(split_starts[split])
        if local_index < 0 or local_index >= limits[split]:
            raise ShardSeedError(
                f"generated row {key} is outside target {split} limit"
            )
        combined_position = local_index
        if split == "test":
            combined_position += validation_limit
        shards[combined_position % shard_count].append(record)

    split_order = {"validation": 0, "test": 1}
    for shard in shards:
        shard.sort(
            key=lambda record: (
                split_order[str(record["split"])],
                int(record["row_index"]),
            )
        )
    return shards


def worker_shard_paths(output_jsonl: Path, shard_count: int) -> list[Path]:
    stem = output_jsonl.with_suffix("")
    return [
        stem.with_name(f"{stem.name}_worker{index:02d}of{shard_count:02d}.jsonl")
        for index in range(shard_count)
    ]


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as pq

    records = [
        json.loads(line)
        for line in args.generated_jsonl.read_text().splitlines()
        if line.strip()
    ]
    train_rows = int(pq.ParquetFile(args.train_parquet).metadata.num_rows)
    validation_rows = int(pq.ParquetFile(args.validation_parquet).metadata.num_rows)
    test_rows = int(pq.ParquetFile(args.test_parquet).metadata.num_rows)
    if args.validation_limit > validation_rows or args.test_limit > test_rows:
        raise ShardSeedError("target limits exceed available split rows")
    split_starts = {
        "validation": train_rows,
        "test": train_rows + validation_rows,
    }
    shards = shard_generated_records(
        records,
        split_starts=split_starts,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        shard_count=args.shard_count,
    )
    paths = worker_shard_paths(args.output_generated_jsonl, args.shard_count)
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise ShardSeedError(f"worker shard already exists: {existing[0]}")
    for path, shard in zip(paths, shards, strict=True):
        _write_jsonl_atomic(path, shard)
    report = {
        "schema_version": "nano_generated_shard_seed.v1",
        "source_jsonl": str(args.generated_jsonl),
        "source_sha256": _sha256(args.generated_jsonl),
        "source_rows": len(records),
        "output_generated_jsonl": str(args.output_generated_jsonl),
        "worker_shards": [
            {"path": str(path), "rows": len(shard)}
            for path, shard in zip(paths, shards, strict=True)
        ],
        "split_starts": split_starts,
        "validation_limit": args.validation_limit,
        "test_limit": args.test_limit,
        "shard_count": args.shard_count,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report_json.with_name(args.report_json.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.report_json)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--output-generated-jsonl", type=Path, required=True)
    parser.add_argument("--validation-limit", type=int, required=True)
    parser.add_argument("--test-limit", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
