#!/usr/bin/env python3
"""Verify that a Nano AV-SFT parquet matches the Miles/NLA dataset contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


EXPLANATION_OPEN = "<explanation>"
EXPLANATION_CLOSE = "</explanation>"
INJECT_PLACEHOLDER = "<INJECT>"
DEFAULT_SPLITS = ((0.8, 0.1, 0.1), (0.9, 0.05, 0.05))
TOKEN_CONTENT_COLUMNS = ("token_ids_prefix", "source_token_ids", "source_tokens", "input_ids")
TEXT_CONTENT_COLUMNS = (
    "detokenized_text_truncated",
    "source_text",
    "text",
    "api_explanation",
    "teacher_explanation",
    "explanation",
    "response",
    "prompt",
)


def sidecar_path_for(parquet_path: Path) -> Path:
    return parquet_path.with_name(parquet_path.name + ".nla_meta.yaml")


def _split_label(spec: tuple[float, float, float]) -> str:
    return "/".join(str(int(round(frac * 100))) for frac in spec)


def _split_items_three_way(
    items: list[Any],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Any], list[Any], list[Any]]:
    if not items:
        return [], [], []
    if validation_fraction < 0 or test_fraction < 0 or train_fraction <= 0:
        raise ValueError("split fractions must be non-negative with train_fraction > 0")
    total_fraction = train_fraction + validation_fraction + test_fraction
    if total_fraction > 1.000001:
        raise ValueError("train + validation + test fractions must be <= 1.0")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n_items = len(shuffled)
    if validation_fraction == 0 and test_fraction == 0:
        n_train = max(1, min(n_items - 1, round(n_items * train_fraction))) if n_items > 1 else n_items
        return shuffled[:n_train], shuffled[n_train:], []

    n_train = round(n_items * train_fraction)
    n_validation = round(n_items * validation_fraction)
    if validation_fraction > 0:
        n_validation = max(1, n_validation)
    if test_fraction > 0:
        if total_fraction >= 0.999999:
            n_test = n_items - n_train - n_validation
        else:
            n_test = round(n_items * test_fraction)
        n_test = max(1, n_test)
    else:
        n_test = 0
    if n_train < 1 and n_items:
        n_train = 1
    while n_train + n_validation + n_test > n_items and n_train > 1:
        n_train -= 1
    while n_train + n_validation + n_test > n_items and n_validation > 0:
        n_validation -= 1
    while n_train + n_validation + n_test > n_items and n_test > 0:
        n_test -= 1

    train = shuffled[:n_train]
    validation = shuffled[n_train : n_train + n_validation]
    test = shuffled[n_train + n_validation : n_train + n_validation + n_test]
    return train, validation, test


def _doc_split_report(
    doc_to_rows: dict[str, list[int]],
    spec: tuple[float, float, float],
    seed: int,
    doc_to_content_keys: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    train_fraction, validation_fraction, test_fraction = spec
    docs = sorted(doc_to_rows)
    train_docs, validation_docs, test_docs = _split_items_three_way(
        docs, train_fraction, validation_fraction, test_fraction, seed
    )
    train_set = set(train_docs)
    validation_set = set(validation_docs)
    test_set = set(test_docs)
    overlap = (train_set & validation_set) | (train_set & test_set) | (validation_set & test_set)
    train_count = sum(len(doc_to_rows[doc]) for doc in train_set)
    validation_count = sum(len(doc_to_rows[doc]) for doc in validation_set)
    test_count = sum(len(doc_to_rows[doc]) for doc in test_set)
    report = {
        "split_mode": "doc",
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "doc_count": len(docs),
        "train_doc_count": len(train_set),
        "validation_doc_count": len(validation_set),
        "test_doc_count": len(test_set),
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "row_count": train_count + validation_count + test_count,
        "doc_overlap_count": len(overlap),
    }
    if doc_to_content_keys is not None:
        report.update(
            _content_overlap_report(
                doc_to_content_keys,
                {
                    "train": train_set,
                    "validation": validation_set,
                    "test": test_set,
                },
            )
        )
    return report


def _content_payload(value: Any, *, max_tokens: int = 300) -> str | None:
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
    return str(value).strip() or None


def _content_key(value: Any, *, column: str) -> str | None:
    payload = _content_payload(value)
    if payload is None:
        return None
    digest = hashlib.sha256(f"{column}\0{payload}".encode("utf-8")).hexdigest()
    return f"{column}:{digest}"


def _content_column_order(names: list[str]) -> list[str]:
    available = set(names)
    ordered = [name for name in TOKEN_CONTENT_COLUMNS if name in available]
    ordered.extend(name for name in TEXT_CONTENT_COLUMNS if name in available and name not in ordered)
    return ordered


def _row_content_key(content_values: dict[str, list[Any]], offset: int, *, fallback: Any = None) -> str:
    for column, values in content_values.items():
        key = _content_key(values[offset], column=column)
        if key is not None:
            return key
    fallback_key = _content_key(fallback, column="fallback")
    return fallback_key or f"row:{offset}"


def _content_overlap_report(
    doc_to_content_keys: dict[str, set[str]],
    split_docs: dict[str, set[str]],
    *,
    sample_limit: int = 10,
) -> dict[str, Any]:
    key_to_docs: dict[str, set[str]] = defaultdict(set)
    for doc_id, content_keys in doc_to_content_keys.items():
        for content_key in content_keys:
            key_to_docs[content_key].add(doc_id)

    duplicate_groups = {
        key: docs
        for key, docs in key_to_docs.items()
        if len(docs) > 1
    }
    doc_to_split: dict[str, str] = {}
    for split_name, docs in split_docs.items():
        for doc_id in docs:
            doc_to_split[doc_id] = split_name

    cross_split_groups = []
    for content_key, docs in sorted(duplicate_groups.items()):
        splits = sorted({doc_to_split[doc_id] for doc_id in docs if doc_id in doc_to_split})
        if len(splits) > 1:
            cross_split_groups.append(
                {
                    "content_hash": content_key,
                    "split_names": splits,
                    "doc_count": len(docs),
                    "doc_ids_sample": sorted(docs)[:sample_limit],
                }
            )

    duplicate_docs = set().union(*duplicate_groups.values()) if duplicate_groups else set()
    cross_docs = set()
    for group in cross_split_groups:
        content_docs = duplicate_groups.get(group["content_hash"], set())
        cross_docs.update(content_docs)
    return {
        "content_hash_mode": "first_300_token_prefix_preferred",
        "content_duplicate_group_count": len(duplicate_groups),
        "content_duplicate_doc_count": len(duplicate_docs),
        "content_duplicate_groups_sample": [
            {
                "content_hash": key,
                "doc_count": len(docs),
                "doc_ids_sample": sorted(docs)[:sample_limit],
            }
            for key, docs in list(sorted(duplicate_groups.items()))[:sample_limit]
        ],
        "content_cross_split_overlap_count": len(cross_split_groups),
        "content_cross_split_doc_sample_count": len(cross_docs),
        "content_cross_split_groups_sample": cross_split_groups[:sample_limit],
    }


def _assert_no_content_cross_split_overlap(splits: dict[str, dict[str, Any]]) -> None:
    for label, report in splits.items():
        overlap_count = int(report.get("content_cross_split_overlap_count") or 0)
        if overlap_count:
            sample = report.get("content_cross_split_groups_sample") or []
            raise ValueError(
                f"content-hash cross-split overlap in split {label}: {overlap_count} duplicate groups; "
                f"sample={sample[:1]}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_family_manifest_split_report(
    doc_to_rows: dict[str, list[int]],
    doc_to_content_keys: dict[str, set[str]],
    manifest_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate exact document coverage and split isolation from a frozen family manifest."""

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"content family manifest not found: {path}")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            "content family manifest SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "nano_content_family_manifest.v1":
        raise ValueError(
            "content family manifest must use schema_version "
            "nano_content_family_manifest.v1"
        )

    doc_assignments = manifest.get("doc_assignments") or {}
    family_splits = manifest.get("family_splits") or {}
    dataset_docs = set(doc_to_rows)
    manifest_docs = set(doc_assignments)
    missing_docs = sorted(dataset_docs - manifest_docs)
    extra_docs = sorted(manifest_docs - dataset_docs)
    if missing_docs or extra_docs:
        raise ValueError(
            "content family manifest document coverage mismatch: "
            f"missing={len(missing_docs)} sample={missing_docs[:5]}, "
            f"extra={len(extra_docs)} sample={extra_docs[:5]}"
        )

    split_docs: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    split_families: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for doc_id in sorted(dataset_docs):
        family_id = str(doc_assignments[doc_id])
        split_name = family_splits.get(family_id)
        if split_name not in split_docs:
            raise ValueError(
                f"content family {family_id!r} for document {doc_id!r} "
                f"has invalid split {split_name!r}"
            )
        split_docs[split_name].add(doc_id)
        split_families[split_name].add(family_id)

    doc_overlap = (
        (split_docs["train"] & split_docs["validation"])
        | (split_docs["train"] & split_docs["test"])
        | (split_docs["validation"] & split_docs["test"])
    )
    family_overlap = (
        (split_families["train"] & split_families["validation"])
        | (split_families["train"] & split_families["test"])
        | (split_families["validation"] & split_families["test"])
    )
    declared_overlap = manifest.get("overlap") or {}
    nonempty_declared_overlap = {
        name: values for name, values in declared_overlap.items() if values
    }
    if doc_overlap or family_overlap or nonempty_declared_overlap:
        raise ValueError(
            "content family manifest declares overlapping splits: "
            f"doc_overlap={len(doc_overlap)}, family_overlap={len(family_overlap)}, "
            f"declared={sorted(nonempty_declared_overlap)}"
        )

    report = _content_overlap_report(doc_to_content_keys, split_docs)
    report.update(
        {
            "split_mode": "content_family_manifest",
            "manifest_path": str(path),
            "manifest_sha256": actual_sha256,
            "manifest_schema_version": manifest["schema_version"],
            "exact_document_coverage": True,
            "doc_count": len(dataset_docs),
            "doc_overlap_count": 0,
            "family_count": len(set(doc_assignments.values())),
            "family_overlap_count": 0,
            "split_doc_counts": {
                split: len(docs) for split, docs in split_docs.items()
            },
            "split_family_counts": {
                split: len(families) for split, families in split_families.items()
            },
            "split_row_counts": {
                split: sum(len(doc_to_rows[doc_id]) for doc_id in docs)
                for split, docs in split_docs.items()
            },
        }
    )
    manifest_summary = manifest.get("split_summary") or {}
    for split_name, row_count in report["split_row_counts"].items():
        expected = manifest_summary.get(split_name) or {}
        if expected.get("row_count") is not None and int(expected["row_count"]) != row_count:
            raise ValueError(
                f"content family manifest {split_name} row count "
                f"{expected['row_count']} != dataset {row_count}"
            )
        family_count = report["split_family_counts"][split_name]
        if expected.get("family_count") is not None and int(expected["family_count"]) != family_count:
            raise ValueError(
                f"content family manifest {split_name} family count "
                f"{expected['family_count']} != dataset {family_count}"
            )
    _assert_no_content_cross_split_overlap({"content_family_manifest": report})
    return report


def materialized_split_content_report(
    split_paths: dict[str, str | Path],
    *,
    batch_size: int = 4096,
) -> dict[str, Any]:
    doc_to_split: dict[str, str] = {}
    doc_to_content_keys: dict[str, set[str]] = defaultdict(set)
    row_counts: dict[str, int] = {}
    content_columns_by_split: dict[str, list[str]] = {}
    for split_name, path_value in split_paths.items():
        path = Path(path_value)
        pf = pq.ParquetFile(path)
        row_counts[split_name] = pf.metadata.num_rows
        split_row_index = 0
        for batch in pf.iter_batches(batch_size=batch_size):
            names = batch.schema.names
            doc_values = batch.column(names.index("doc_id")).to_pylist() if "doc_id" in names else [None] * len(batch)
            content_columns = _content_column_order(names)
            content_columns_by_split[split_name] = content_columns
            content_values = {name: batch.column(names.index(name)).to_pylist() for name in content_columns}
            fallback_values = (
                batch.column(names.index("response")).to_pylist()
                if "response" in names
                else batch.column(names.index("prompt")).to_pylist()
                if "prompt" in names
                else [None] * len(batch)
            )
            for offset, (doc_id, fallback) in enumerate(zip(doc_values, fallback_values)):
                doc_key = str(doc_id or f"__{split_name}_row_{split_row_index + offset}")
                existing = doc_to_split.get(doc_key)
                if existing is not None and existing != split_name:
                    doc_to_split[doc_key] = "__overlap__"
                else:
                    doc_to_split[doc_key] = split_name
                doc_to_content_keys[doc_key].add(_row_content_key(content_values, offset, fallback=fallback))
            split_row_index += len(batch)

    split_docs: dict[str, set[str]] = {name: set() for name in split_paths}
    doc_overlap = []
    for doc_id, split_name in doc_to_split.items():
        if split_name == "__overlap__":
            doc_overlap.append(doc_id)
            continue
        split_docs.setdefault(split_name, set()).add(doc_id)

    report = _content_overlap_report(doc_to_content_keys, split_docs)
    report.update(
        {
            "split_mode": "materialized_doc",
            "row_counts": row_counts,
            "doc_overlap_count": len(doc_overlap),
            "doc_overlap_sample": sorted(doc_overlap)[:10],
            "content_columns_by_split": content_columns_by_split,
        }
    )
    _assert_no_content_cross_split_overlap({"materialized": report})
    if doc_overlap:
        raise ValueError(f"materialized split doc overlap: {len(doc_overlap)} docs; sample={doc_overlap[:10]}")
    return report


def _activation_lengths(array: pa.Array) -> np.ndarray:
    if pa.types.is_fixed_size_list(array.type):
        return np.full(len(array), array.type.list_size, dtype=np.int64)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        return np.diff(offsets).astype(np.int64, copy=False)
    raise TypeError(f"activation_vector must be a list type, got {array.type}")


def _finite_counts(array: pa.Array) -> tuple[int, int]:
    flat = array.flatten().to_numpy(zero_copy_only=False)
    finite = np.isfinite(flat)
    return int(finite.sum()), int(flat.size - finite.sum())


def _prompt_messages_with_injection(prompt: Any, injection_char: str) -> list[dict[str, str]]:
    if not isinstance(prompt, list):
        raise ValueError(f"prompt must be list[dict], got {type(prompt).__name__}")
    out = []
    marker_count = 0
    for msg in prompt:
        if not isinstance(msg, dict):
            raise ValueError(f"prompt message must be dict, got {type(msg).__name__}")
        content = str(msg.get("content", ""))
        marker_count += content.count(INJECT_PLACEHOLDER)
        out.append({**msg, "content": content.replace(INJECT_PLACEHOLDER, injection_char)})
    if marker_count != 1:
        raise ValueError(f"prompt must contain exactly one {INJECT_PLACEHOLDER}, found {marker_count}")
    return out


def _token_ids_from_chat(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError(f"expected one tokenized prompt, got batch of {len(ids)}")
        ids = ids[0]
    return [int(tok) for tok in ids]


def _valid_marker_positions(ids: list[int], inj_id: int, left_id: int, right_id: int) -> list[int]:
    positions = []
    for idx, tok in enumerate(ids):
        if tok != inj_id or idx == 0 or idx == len(ids) - 1:
            continue
        if ids[idx - 1] == left_id and ids[idx + 1] == right_id:
            positions.append(idx)
    return positions


def _load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def verify_dataset(
    parquet_path: str | Path,
    *,
    tokenizer: Any | None = None,
    tokenizer_model: str | None = None,
    expected_rows: int | None = None,
    expected_d_model: int | None = None,
    row_limit: int | None = None,
    prompt_check_limit: int | None = None,
    split_specs: tuple[tuple[float, float, float], ...] = DEFAULT_SPLITS,
    split_seed: int = 42,
    content_family_manifest: str | Path | None = None,
    content_family_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    parquet_path = Path(parquet_path)
    sidecar_path = sidecar_path_for(parquet_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet not found: {parquet_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"sidecar not found: {sidecar_path}")

    sidecar = yaml.safe_load(sidecar_path.read_text())
    sidecar_rows = sidecar.get("row_count")
    sidecar_d_model = ((sidecar.get("extraction") or {}).get("d_model"))
    tokens = sidecar.get("tokens") or {}
    expected_d_model = int(expected_d_model or sidecar_d_model)
    if tokenizer is None and tokenizer_model is not None:
        tokenizer = _load_tokenizer(tokenizer_model)

    pf = pq.ParquetFile(parquet_path)
    columns = set(pf.schema_arrow.names)
    required = {"prompt", "response", "activation_vector"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"parquet missing required columns: {missing}")
    if expected_rows is not None and pf.metadata.num_rows != expected_rows:
        raise ValueError(f"row count {pf.metadata.num_rows} != expected {expected_rows}")
    if sidecar_rows is not None and int(sidecar_rows) != pf.metadata.num_rows:
        raise ValueError(f"sidecar row_count {sidecar_rows} != parquet rows {pf.metadata.num_rows}")

    length_counts: Counter[int] = Counter()
    response_bad_rows = []
    prompt_bad_rows = []
    doc_to_rows: dict[str, list[int]] = defaultdict(list)
    doc_to_content_keys: dict[str, set[str]] = defaultdict(set)
    content_columns_seen: set[str] = set()
    finite_count = 0
    nonfinite_count = 0
    inspected = 0
    prompt_checked = 0
    prompt_cache: dict[str, tuple[bool, str | None]] = {}

    if tokenizer is not None:
        for key in ("injection_char", "injection_token_id", "injection_left_neighbor_id", "injection_right_neighbor_id"):
            if key not in tokens:
                raise ValueError(f"sidecar tokens missing {key!r}")
        injection_char = str(tokens["injection_char"])
        inj_id = int(tokens["injection_token_id"])
        left_id = int(tokens["injection_left_neighbor_id"])
        right_id = int(tokens["injection_right_neighbor_id"])
    else:
        injection_char = ""
        inj_id = left_id = right_id = -1

    for batch in pf.iter_batches(batch_size=4096):
        if row_limit is not None and inspected >= row_limit:
            break
        if row_limit is not None:
            batch = batch.slice(0, min(len(batch), row_limit - inspected))

        names = batch.schema.names
        av = batch.column(names.index("activation_vector"))
        lengths = _activation_lengths(av)
        length_counts.update(int(length) for length in lengths)
        finite, nonfinite = _finite_counts(av)
        finite_count += finite
        nonfinite_count += nonfinite

        response_values = batch.column(names.index("response")).to_pylist()
        prompt_values = batch.column(names.index("prompt")).to_pylist()
        doc_values = batch.column(names.index("doc_id")).to_pylist() if "doc_id" in names else [None] * len(batch)
        content_columns = _content_column_order(names)
        content_columns_seen.update(content_columns)
        content_values = {name: batch.column(names.index(name)).to_pylist() for name in content_columns}

        for offset, (response, prompt, doc_id) in enumerate(zip(response_values, prompt_values, doc_values)):
            row_index = inspected + offset
            doc_key = str(doc_id or f"__row_{row_index}")
            doc_to_rows[doc_key].append(row_index)
            doc_to_content_keys[doc_key].add(_row_content_key(content_values, offset, fallback=response))
            response_text = str(response or "")
            if not (response_text.startswith(EXPLANATION_OPEN) and response_text.rstrip().endswith(EXPLANATION_CLOSE)):
                response_bad_rows.append(row_index)

            if tokenizer is not None and (prompt_check_limit is None or prompt_checked < prompt_check_limit):
                try:
                    messages = _prompt_messages_with_injection(prompt, injection_char)
                    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    cached = prompt_cache.get(rendered)
                    if cached is None:
                        ids = _token_ids_from_chat(tokenizer, messages)
                        positions = _valid_marker_positions(ids, inj_id, left_id, right_id)
                        if len(positions) != 1:
                            cached = (False, f"found {len(positions)} valid injection marker positions")
                        else:
                            cached = (True, None)
                        prompt_cache[rendered] = cached
                    ok, error = cached
                    if not ok:
                        prompt_bad_rows.append({"row_index": row_index, "error": error})
                except Exception as exc:  # noqa: BLE001 - report row-level verifier failures.
                    prompt_bad_rows.append({"row_index": row_index, "error": str(exc)})
                prompt_checked += 1

        inspected += len(batch)

    if set(length_counts) != {expected_d_model}:
        raise ValueError(f"activation_vector lengths {dict(length_counts)} != expected d_model {expected_d_model}")
    if nonfinite_count:
        raise ValueError(f"activation_vector contains {nonfinite_count} non-finite values")
    if response_bad_rows:
        raise ValueError(f"{len(response_bad_rows)} responses do not have explanation tags")
    if prompt_bad_rows:
        first = prompt_bad_rows[0]
        raise ValueError(
            f"{len(prompt_bad_rows)} prompts failed valid injection marker check; "
            f"first row {first['row_index']}: {first['error']}"
        )

    if content_family_manifest is not None and split_specs:
        raise ValueError(
            "content_family_manifest and synthetic split_specs are mutually exclusive"
        )
    splits = {
        _split_label(spec): _doc_split_report(doc_to_rows, spec, split_seed, doc_to_content_keys)
        for spec in split_specs
    }
    _assert_no_content_cross_split_overlap(splits)
    report = {
        "parquet": str(parquet_path),
        "sidecar": str(sidecar_path),
        "row_count": pf.metadata.num_rows,
        "inspected_count": inspected,
        "drop_count": 0,
        "sidecar_row_count": sidecar_rows,
        "stage": sidecar.get("stage"),
        "activation": {
            "d_model": expected_d_model,
            "length_counts": dict(length_counts),
            "finite_count": finite_count,
            "nonfinite_count": nonfinite_count,
        },
        "responses": {
            "malformed_count": len(response_bad_rows),
            "malformed_rows_sample": response_bad_rows[:10],
        },
        "prompt_markers": {
            "checked_count": prompt_checked,
            "bad_count": len(prompt_bad_rows),
            "bad_rows_sample": prompt_bad_rows[:10],
            "tokenizer_check_skipped": tokenizer is None,
        },
        "content_hash": {
            "mode": "first_300_token_prefix_preferred",
            "columns_seen": sorted(content_columns_seen),
        },
        "splits": splits,
    }
    if content_family_manifest is not None:
        report["content_family_manifest_split"] = content_family_manifest_split_report(
            doc_to_rows,
            doc_to_content_keys,
            content_family_manifest,
            expected_sha256=content_family_manifest_sha256,
        )
    return report


def parse_split(value: str) -> tuple[float, float, float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split must be train,validation,test")
    if not math.isclose(sum(parts), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise argparse.ArgumentTypeError("split fractions must sum to 1.0")
    return parts[0], parts[1], parts[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-d-model", type=int)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--prompt-check-limit", type=int)
    parser.add_argument("--tokenizer-model", help="HF tokenizer/model id or local model path for marker checks.")
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    parser.add_argument("--split", dest="splits", action="append", type=parse_split)
    parser.add_argument(
        "--skip-synthetic-split-checks",
        action="store_true",
        help=(
            "Skip verifier-generated doc splits. Use this when validating an explicit "
            "materialized split layout such as content_component."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--content-family-manifest", type=Path)
    parser.add_argument("--content-family-manifest-sha256")
    parser.add_argument("--materialized-train", type=Path)
    parser.add_argument("--materialized-validation", type=Path)
    parser.add_argument("--materialized-test", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    tokenizer_model = None if args.skip_tokenizer_check else args.tokenizer_model
    if tokenizer_model is None and not args.skip_tokenizer_check:
        raise SystemExit("--tokenizer-model is required unless --skip-tokenizer-check is set")

    if args.content_family_manifest is not None and args.splits:
        parser.error("--content-family-manifest and --split are mutually exclusive")
    split_specs = (
        ()
        if args.skip_synthetic_split_checks or args.content_family_manifest is not None
        else tuple(args.splits or DEFAULT_SPLITS)
    )
    report = verify_dataset(
        args.parquet,
        tokenizer_model=tokenizer_model,
        expected_rows=args.expected_rows,
        expected_d_model=args.expected_d_model,
        row_limit=args.row_limit,
        prompt_check_limit=args.prompt_check_limit,
        split_specs=split_specs,
        split_seed=args.split_seed,
        content_family_manifest=args.content_family_manifest,
        content_family_manifest_sha256=args.content_family_manifest_sha256,
    )
    if args.skip_synthetic_split_checks or args.content_family_manifest is not None:
        report["synthetic_splits_skipped"] = True
    materialized_paths = {
        key: path
        for key, path in {
            "train": args.materialized_train,
            "validation": args.materialized_validation,
            "test": args.materialized_test,
        }.items()
        if path is not None
    }
    if materialized_paths:
        if set(materialized_paths) != {"train", "validation", "test"}:
            raise SystemExit("--materialized-train, --materialized-validation, and --materialized-test are all required")
        report["materialized_splits"] = materialized_split_content_report(materialized_paths)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
