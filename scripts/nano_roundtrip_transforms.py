#!/usr/bin/env python3
"""Versioned meaning-preserving transforms for generated NLA explanations."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


TRANSFORM_SCHEMA_VERSION = "nano_roundtrip_transform.v1"
DETERMINISTIC_TRANSFORMS = (
    "format_normalized",
    "surface_canonicalized",
    "unit_reordered",
    "truncate_words_25",
    "truncate_words_50",
    "truncate_words_75",
    "drop_first_unit",
    "drop_last_unit",
)
_EXPLANATION_RE = re.compile(
    r"(?P<open><explanation\b[^>]*>)(?P<body>.*?)(?P<close></explanation>)",
    flags=re.IGNORECASE | re.DOTALL,
)


class TransformError(ValueError):
    """Raised when transformed evidence is missing, stale, or malformed."""


def text_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def stable_row_key(record: dict[str, Any]) -> str:
    sample_uuid = record.get("sample_uuid")
    if sample_uuid not in {None, ""}:
        return f"uuid:{sample_uuid}"
    if record.get("split") is not None and record.get("row_index") is not None:
        return f"{record['split']}:{int(record['row_index'])}"
    if record.get("doc_id") is not None and record.get("token_position") is not None:
        return f"position:{record['doc_id']}:{int(record['token_position'])}"
    if record.get("doc_id") is not None and record.get("n_raw_tokens") is not None:
        return f"raw_tokens:{record['doc_id']}:{int(record['n_raw_tokens'])}"
    raise TransformError("record has no stable row identity")


def normalize_formatting(text: str) -> str:
    lines = [" ".join(line.split()) for line in str(text).strip().splitlines()]
    return "\n".join(line for line in lines if line)


def split_semantic_units(text: str) -> list[str]:
    normalized = normalize_formatting(text)
    lines = [line for line in normalized.splitlines() if line]
    if len(lines) > 1:
        return lines
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", normalized)
        if part.strip()
    ]


def reorder_units(text: str, *, seed: int) -> str:
    units = split_semantic_units(text)
    shuffled = list(units)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == units:
        shuffled = shuffled[1:] + shuffled[:1]
    separator = "\n" if "\n" in str(text) else " "
    return separator.join(shuffled)


def canonicalize_surface(text: str) -> str:
    """Remove list/heading markup while retaining the lexical content and order."""

    units = split_semantic_units(text)
    canonical = []
    for unit in units:
        value = re.sub(r"^\s*(?:[-*+] |\d+[.)]\s+)", "", unit)
        value = re.sub(r"^\s*#{1,6}\s+", "", value)
        value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
        canonical.append(" ".join(value.split()))
    return " ".join(value for value in canonical if value)


def truncate_words(text: str, fraction: float) -> str:
    words = normalize_formatting(text).split()
    if not words:
        return ""
    keep = max(1, min(len(words), int(round(len(words) * fraction))))
    return " ".join(words[:keep])


def drop_unit(text: str, *, first: bool) -> str:
    units = split_semantic_units(text)
    if len(units) <= 1:
        return units[0] if units else ""
    retained = units[1:] if first else units[:-1]
    return "\n".join(retained) if "\n" in str(text) else " ".join(retained)


def _transform_body(body: str, *, transform: str, seed: int) -> str:
    if transform == "format_normalized":
        return normalize_formatting(body)
    if transform == "surface_canonicalized":
        return canonicalize_surface(body)
    if transform == "unit_reordered":
        return reorder_units(body, seed=seed)
    if transform.startswith("truncate_words_"):
        fraction = int(transform.rsplit("_", 1)[1]) / 100.0
        return truncate_words(body, fraction)
    if transform == "drop_first_unit":
        return drop_unit(body, first=True)
    if transform == "drop_last_unit":
        return drop_unit(body, first=False)
    raise TransformError(f"unsupported deterministic transform: {transform}")


def transform_generated_text(text: str, *, transform: str, seed: int) -> str:
    source = str(text)
    match = _EXPLANATION_RE.search(source)
    if match is None:
        transformed = _transform_body(source, transform=transform, seed=seed)
        if not transformed.strip():
            raise TransformError("transformed generated text is empty")
        return transformed
    transformed_body = _transform_body(
        match.group("body"),
        transform=transform,
        seed=seed,
    )
    if not transformed_body.strip():
        raise TransformError("transformed explanation body is empty")
    return (
        source[: match.start()]
        + match.group("open")
        + transformed_body
        + match.group("close")
        + source[match.end() :]
    )


def build_transform_record(
    *,
    row_key: str,
    source: str,
    transform: str,
    transformed: str,
    seed: int,
    model: str | None = None,
    prompt_sha256: str | None = None,
) -> dict[str, Any]:
    if not str(row_key).strip():
        raise TransformError("row_key is empty")
    if not str(transformed).strip():
        raise TransformError("transformed text is empty")
    return {
        "schema_version": TRANSFORM_SCHEMA_VERSION,
        "row_key": str(row_key),
        "transform": str(transform),
        "source_sha256": text_sha256(source),
        "transformed_text": str(transformed),
        "seed": int(seed),
        "model": model,
        "prompt_sha256": prompt_sha256,
    }


def deterministic_transform_records(
    generated_records: list[dict[str, Any]],
    *,
    transform: str,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for record in generated_records:
        row_key = stable_row_key(record)
        source = str(record.get("controls", {}).get("real", {}).get("generated") or "")
        if not source.strip():
            raise TransformError(f"real generated text is empty for {row_key}")
        row_seed = seed + int(record.get("row_index", 0)) * 1009
        output.append(
            build_transform_record(
                row_key=row_key,
                source=source,
                transform=transform,
                transformed=transform_generated_text(
                    source,
                    transform=transform,
                    seed=row_seed,
                ),
                seed=row_seed,
            )
        )
    return output


def index_transform_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("schema_version") != TRANSFORM_SCHEMA_VERSION:
            raise TransformError(f"invalid transform schema: {record.get('schema_version')!r}")
        key = (str(record.get("row_key")), str(record.get("transform")))
        if key in indexed:
            raise TransformError(f"duplicate transform record: {key}")
        indexed[key] = record
    return indexed


def apply_transform_records(
    generated_records: list[dict[str, Any]],
    transforms_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    transform: str,
) -> list[dict[str, Any]]:
    output = copy.deepcopy(generated_records)
    for record in output:
        row_key = stable_row_key(record)
        item = transforms_by_key.get((row_key, transform))
        if item is None:
            raise TransformError(f"missing {transform} transform for {row_key}")
        source = str(record.get("controls", {}).get("real", {}).get("generated") or "")
        if item.get("source_sha256") != text_sha256(source):
            raise TransformError(f"source hash mismatch for {transform} transform {row_key}")
        transformed = str(item.get("transformed_text") or "")
        if not transformed.strip():
            raise TransformError(f"empty {transform} transform for {row_key}")
        record["controls"]["real"]["generated"] = transformed
        record["controls"]["real"]["semantic_transform"] = {
            "schema_version": item["schema_version"],
            "transform": transform,
            "source_sha256": item["source_sha256"],
            "seed": item.get("seed"),
            "model": item.get("model"),
            "prompt_sha256": item.get("prompt_sha256"),
        }
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise TransformError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
