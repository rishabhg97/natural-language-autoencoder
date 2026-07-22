#!/usr/bin/env python3
"""Resolve exact R33 source rows from stable generated-record provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


class SourceRowError(ValueError):
    """Raised when source provenance is ambiguous, incomplete, or inconsistent."""


def provenance_keys(record: dict[str, Any]) -> list[tuple[Any, ...]]:
    keys: list[tuple[Any, ...]] = []
    sample_uuid = record.get("sample_uuid")
    if sample_uuid not in {None, ""}:
        keys.append(("uuid", str(sample_uuid)))
    doc_id = record.get("doc_id")
    token_position = record.get("token_position")
    if doc_id not in {None, ""} and token_position is not None:
        keys.append(("position", str(doc_id), int(token_position)))
    n_raw_tokens = record.get("n_raw_tokens")
    if doc_id not in {None, ""} and n_raw_tokens is not None:
        keys.append(("raw_tokens", str(doc_id), int(n_raw_tokens)))
    return keys


def provenance_key(record: dict[str, Any]) -> tuple[Any, ...]:
    keys = provenance_keys(record)
    if not keys:
        raise SourceRowError(
            "row needs sample_uuid, doc_id plus token_position, or doc_id plus n_raw_tokens"
        )
    return keys[0]


def matching_batch_rows(
    batch: Any,
    wanted: set[tuple[Any, ...]],
) -> list[tuple[int, list[tuple[Any, ...]]]]:
    """Scan only lightweight provenance columns and return matching offsets."""
    key_columns = [
        name
        for name in ("sample_uuid", "doc_id", "token_position", "n_raw_tokens")
        if name in batch.schema.names
    ]
    values = {
        name: batch.column(batch.schema.get_field_index(name)).to_pylist()
        for name in key_columns
    }
    matches: list[tuple[int, list[tuple[Any, ...]]]] = []
    for row_index in range(batch.num_rows):
        key_row = {name: column[row_index] for name, column in values.items()}
        matching_keys = [key for key in provenance_keys(key_row) if key in wanted]
        if matching_keys:
            matches.append((row_index, matching_keys))
    return matches


def resolve_source_rows(
    parquet_path: str | Path,
    requested: list[dict[str, Any]],
    *,
    batch_size: int = 4_096,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not requested:
        raise SourceRowError("requested source rows are empty")
    if batch_size <= 0:
        raise SourceRowError("batch_size must be positive")

    requested_keys = [provenance_key(record) for record in requested]
    if len(set(requested_keys)) != len(requested_keys):
        raise SourceRowError("requested records contain duplicate provenance keys")
    wanted = set(requested_keys)

    parquet = pq.ParquetFile(Path(parquet_path))
    schema_names = set(parquet.schema_arrow.names)
    required_columns = {"token_ids_prefix", "activation_vector"}
    missing_required = sorted(required_columns - schema_names)
    if missing_required:
        raise SourceRowError(
            f"source parquet is missing required columns: {missing_required}"
        )
    key_columns = {"sample_uuid", "doc_id", "token_position", "n_raw_tokens"}
    if not schema_names.intersection(key_columns):
        raise SourceRowError("source parquet has no supported provenance key columns")

    preferred_columns = (
        "sample_uuid",
        "row_index",
        "source_row_index",
        "doc_id",
        "token_position",
        "n_raw_tokens",
        "token_id",
        "token_text",
        "detokenized_text_truncated",
        "token_ids_prefix",
        "activation_vector",
        "activation_layer",
        "split",
        "api_explanation",
        "explanation",
        "teacher_explanation",
        "prompt",
        "response",
    )
    columns = [column for column in preferred_columns if column in schema_names]
    found: dict[tuple[Any, ...], dict[str, Any]] = {}
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row_index, matching_keys in matching_batch_rows(batch, wanted):
            for key in matching_keys:
                if key in found:
                    raise SourceRowError(f"duplicate source provenance key: {key}")
            row = batch.slice(row_index, 1).to_pylist()[0]
            for key in matching_keys:
                found[key] = row
        if len(found) == len(wanted):
            break

    missing = sorted(wanted - found.keys(), key=repr)
    if missing:
        raise SourceRowError(f"missing source rows: {missing[:10]}")
    return found
