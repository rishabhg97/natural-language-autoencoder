#!/usr/bin/env python3
"""Plan, run, and score Nano AR layer-sweep probes.

This is intentionally lightweight on import: local tests and queue operations do
not need Torch. The GPU-only extraction path imports Nano model helpers lazily.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_queue_status import ACTIVE_LAYER_STATUSES, VALID_LAYER_STATUSES, status_counts  # noqa: E402


SCHEMA_VERSION = "nano_ar_layer_sweep_queue.v1"
SCORE_SCHEMA_VERSION = "nano_ar_layer_sweep_score.v1"
EXTRACT_SCHEMA_VERSION = "nano_ar_layer_sweep_extract.v1"
DEFAULT_MODEL_ID = "/workspace/interp/models/nano-30b-a3b-bf16-hf"
VALID_STATUSES = VALID_LAYER_STATUSES
ACTIVE_STATUSES = ACTIVE_LAYER_STATUSES
JOIN_KEY_CANDIDATES = ("doc_id", "token_position", "token_id", "n_raw_tokens")
TEXT_COLUMNS = ("api_explanation", "explanation", "teacher_explanation")
LAYER_RE = re.compile(r"^\s*R?_?(\d+)\s*$", re.IGNORECASE)
LAYER_RANGE_RE = re.compile(r"^\s*R?_?(\d+)\s*-\s*R?_?(\d+)\s*$", re.IGNORECASE)
MIN_POSITION = 50


class QueueError(ValueError):
    """Raised when a layer-sweep queue manifest is malformed."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def parse_boundary_spec(spec: str | list[int] | tuple[int, ...]) -> list[int]:
    """Parse specs like R25-R27,R_30,34 into ordered unique boundary ids."""
    if isinstance(spec, (list, tuple)):
        tokens: list[int] = [int(value) for value in spec]
    else:
        tokens = []
        for raw_part in str(spec).split(","):
            part = raw_part.strip()
            if not part:
                continue
            range_match = LAYER_RANGE_RE.match(part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                step = 1 if end >= start else -1
                tokens.extend(range(start, end + step, step))
                continue
            single_match = LAYER_RE.match(part)
            if not single_match:
                raise ValueError(f"invalid layer boundary spec {part!r}")
            tokens.append(int(single_match.group(1)))

    seen: set[int] = set()
    layers: list[int] = []
    for layer in tokens:
        if layer < 0:
            raise ValueError(f"layer boundary must be non-negative: {layer}")
        if layer not in seen:
            seen.add(layer)
            layers.append(layer)
    if not layers:
        raise ValueError("at least one layer boundary is required")
    return layers


def format_layer_spec(layers: list[int] | tuple[int, ...]) -> str:
    ordered = parse_boundary_spec(list(layers))
    ranges: list[str] = []
    start = prev = ordered[0]
    for layer in ordered[1:]:
        if layer == prev + 1:
            prev = layer
            continue
        ranges.append(f"R{start}" if start == prev else f"R{start}-R{prev}")
        start = prev = layer
    ranges.append(f"R{start}" if start == prev else f"R{start}-R{prev}")
    return ",".join(ranges)


def format_layer_slug(layers: list[int] | tuple[int, ...]) -> str:
    return format_layer_spec(layers).replace(",", "_").lower()


def build_queue_doc(
    *,
    code_root: Path,
    python_bin: str,
    layers: list[int],
    output_root: Path,
    teacher_table: Path,
    model_id: str,
    corpus_start: int,
    corpus_length: int,
    positions_per_doc: int,
    score_top_k: int = 5,
    validation_limit: int = 512,
    test_limit: int = 512,
    hash_dim: int = 512,
    batch_size: int = 2,
    chunk_size: int = 8,
    max_length: int = 1024,
    seed: int = 42,
    local_files_only: bool = True,
    overwrite: bool = True,
) -> dict[str, Any]:
    layer_spec = format_layer_spec(layers)
    slug = format_layer_slug(layers)
    script = code_root / "scripts" / "nano_ar_layer_sweep.py"
    logs_dir = output_root / "queue_logs"
    extract_log = logs_dir / f"extract-{slug}.log"
    score_log = logs_dir / f"score-{slug}.log"
    score_report = output_root / f"score_{slug}_teacher_knn_report.json"

    extract_command = [
        python_bin,
        str(script),
        "extract",
        "--layers",
        layer_spec,
        "--output-root",
        str(output_root),
        "--model-id",
        model_id,
        "--corpus-start",
        str(corpus_start),
        "--corpus-length",
        str(corpus_length),
        "--positions-per-doc",
        str(positions_per_doc),
        "--chunk-size",
        str(chunk_size),
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
        "--seed",
        str(seed),
    ]
    if local_files_only:
        extract_command.append("--local-files-only")
    if overwrite:
        extract_command.append("--overwrite")

    score_command = [
        python_bin,
        str(script),
        "score",
        "--layers",
        layer_spec,
        "--output-root",
        str(output_root),
        "--teacher-table",
        str(teacher_table),
        "--report-json",
        str(score_report),
        "--validation-limit",
        str(validation_limit),
        "--test-limit",
        str(test_limit),
        "--hash-dim",
        str(hash_dim),
        "--knn-k",
        str(score_top_k),
        "--seed",
        str(seed),
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "defaults": {
            "code_root": str(code_root),
            "python": python_bin,
            "output_root": str(output_root),
            "teacher_table": str(teacher_table),
            "logs_dir": str(logs_dir),
        },
        "items": [
            {
                "name": f"extract-{slug}",
                "status": "pending",
                "kind": "extract",
                "layers": layer_spec,
                "log": str(extract_log),
                "command": extract_command,
            },
            {
                "name": f"score-{slug}",
                "status": "pending",
                "kind": "score",
                "layers": layer_spec,
                "log": str(score_log),
                "report_json": str(score_report),
                "command": score_command,
            },
        ],
    }


def validate_queue(data: dict[str, Any], *, source: Path | None = None) -> dict[str, Any]:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise QueueError(f"schema_version must be {SCHEMA_VERSION}")
    defaults = data.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        raise QueueError("defaults must be a mapping")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise QueueError("items must be a non-empty list")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise QueueError(f"item {index} must be a mapping")
        item.setdefault("status", "pending")
        if item["status"] not in VALID_STATUSES:
            raise QueueError(f"item {index} has invalid status {item['status']!r}")
        if not item.get("name") or not isinstance(item.get("command"), list) or not item["command"]:
            label = f" in {source}" if source else ""
            raise QueueError(f"item {index}{label} requires name and non-empty command")
    return data


def load_queue(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict):
        raise QueueError(f"queue YAML must contain a mapping: {source}")
    return validate_queue(data, source=source)


def write_queue(path: str | Path, queue_doc: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(queue_doc, sort_keys=False))
    tmp.replace(destination)


def next_pending_index(queue_doc: dict[str, Any]) -> int | None:
    for index, item in enumerate(queue_doc["items"]):
        if item.get("status") == "pending":
            return index
    return None


def update_item(path: str | Path, index: int, **fields: Any) -> dict[str, Any]:
    queue_doc = load_queue(path)
    item = queue_doc["items"][index]
    item.update({key: value for key, value in fields.items() if value is not None})
    write_queue(path, queue_doc)
    return item


def reset_active_items(path: str | Path, *, reason: str = "manual active reset") -> dict[str, Any]:
    queue_doc = load_queue(path)
    reset_items: list[str] = []
    reset_at = utc_now()
    for item in queue_doc["items"]:
        status = item.get("status")
        if status not in ACTIVE_STATUSES:
            continue
        item["previous_status"] = status
        item["status"] = "pending"
        item["reset_at"] = reset_at
        item["reset_reason"] = reason
        reset_items.append(str(item["name"]))
    if reset_items:
        write_queue(path, queue_doc)
    return {"queue": str(path), "reset_count": len(reset_items), "items": reset_items}


@contextlib.contextmanager
def queue_lock(queue_path: str | Path):
    lock_path = Path(queue_path).with_suffix(Path(queue_path).suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QueueError(f"queue watcher already active: {lock_path}") from exc
        handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _env_for_item(queue_doc: dict[str, Any], item: dict[str, Any] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    defaults = queue_doc.get("defaults", {})
    env.update({str(key): str(value) for key, value in (defaults.get("environment") or {}).items()})
    if item is not None:
        env.update({str(key): str(value) for key, value in (item.get("environment") or {}).items()})
    code_root = defaults.get("code_root")
    if code_root:
        pythonpath_parts = [
            str(Path(code_root)),
            str(Path(code_root) / "scripts"),
            str(Path(code_root) / "external" / "natural_language_autoencoders"),
            str(Path(code_root) / "external" / "natural_language_autoencoders" / "Miles"),
        ]
        existing = env.get("PYTHONPATH")
        if existing:
            pythonpath_parts.append(existing)
        env["PYTHONPATH"] = ":".join(pythonpath_parts)
    env.setdefault("WANDB_MODE", "offline")
    return env


def _run_logged(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n# started_utc={utc_now()}\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"# completed_utc={utc_now()}\n")


def process_next_item(queue_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    queue_path = Path(queue_path)
    queue_doc = load_queue(queue_path)
    index = next_pending_index(queue_doc)
    if index is None:
        return {"status": "idle"}

    item = queue_doc["items"][index]
    defaults = queue_doc.get("defaults", {})
    cwd = Path(defaults.get("code_root") or queue_path.parent)
    log_path = Path(item.get("log") or Path(defaults.get("logs_dir") or queue_path.parent / "logs") / f"{item['name']}.log")
    command = [str(part) for part in item["command"]]
    if dry_run:
        return {
            "status": "dry_run",
            "item_index": index,
            "item_name": item["name"],
            "cwd": str(cwd),
            "log": str(log_path),
            "command": command,
        }

    try:
        update_item(queue_path, index, status="running", started_at=utc_now(), log=str(log_path))
        _run_logged(command, cwd=cwd, env=_env_for_item(queue_doc, item), log_path=log_path)
        update_item(queue_path, index, status="complete", completed_at=utc_now(), log=str(log_path))
        return {"status": "complete", "item_index": index, "item_name": item["name"], "log": log_path}
    except Exception as exc:
        update_item(queue_path, index, status="failed", failed_at=utc_now(), failure=str(exc), log=str(log_path))
        return {"status": "failed", "item_index": index, "item_name": item["name"], "failure": str(exc), "log": log_path}


def queue_status(queue_path: str | Path) -> dict[str, Any]:
    queue_doc = load_queue(queue_path)
    counts = status_counts(queue_doc["items"], VALID_STATUSES)
    return {"queue": str(queue_path), "counts": counts, "items": queue_doc["items"]}


def watch_queue(
    queue_path: Path,
    *,
    poll_seconds: int,
    dry_run: bool,
    once: bool,
    stop_when_idle: bool = False,
    stop_on_failure: bool = True,
) -> int:
    with queue_lock(queue_path):
        while True:
            result = process_next_item(queue_path, dry_run=dry_run)
            print(json.dumps(json_safe(result), sort_keys=True), flush=True)
            if once or dry_run:
                return 0
            if result["status"] == "failed" and stop_on_failure:
                return 1
            if result["status"] == "idle":
                if stop_when_idle:
                    return 0
                time.sleep(poll_seconds)


def _read_table(path: Path) -> pa.Table:
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path)


def _column_to_list(table: pa.Table, name: str) -> list[Any]:
    return table.column(name).combine_chunks().to_pylist()


def _activation_matrix(table: pa.Table) -> np.ndarray:
    column = table.column("activation_vector").combine_chunks()
    if not pa.types.is_fixed_size_list(column.type):
        values = column.to_pylist()
        return np.asarray(values, dtype=np.float32)
    list_size = int(column.type.list_size)
    values = column.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(column), list_size)


def _available_text_column(table: pa.Table) -> str:
    names = set(table.column_names)
    for column in TEXT_COLUMNS:
        if column in names:
            return column
    raise ValueError(f"teacher table needs one of {TEXT_COLUMNS}")


def _join_keys(base: pa.Table, teacher: pa.Table) -> list[str]:
    base_names = set(base.column_names)
    teacher_names = set(teacher.column_names)
    keys = [name for name in JOIN_KEY_CANDIDATES if name in base_names and name in teacher_names]
    if not keys:
        raise ValueError("base and teacher tables need at least one shared key column")
    return keys


def _teacher_lookup(teacher: pa.Table, keys: list[str], text_column: str) -> dict[tuple[Any, ...], str]:
    key_columns = [_column_to_list(teacher, key) for key in keys]
    texts = _column_to_list(teacher, text_column)
    lookup: dict[tuple[Any, ...], str] = {}
    for row_index, text in enumerate(texts):
        if text is None:
            continue
        lookup[tuple(columns[row_index] for columns in key_columns)] = str(text)
    return lookup


def _match_base_to_teacher(base: pa.Table, teacher_lookup: dict[tuple[Any, ...], str], keys: list[str]) -> tuple[list[int], list[str]]:
    key_columns = [_column_to_list(base, key) for key in keys]
    matched_indices: list[int] = []
    texts: list[str] = []
    for row_index in range(base.num_rows):
        key = tuple(columns[row_index] for columns in key_columns)
        text = teacher_lookup.get(key)
        if text is None:
            continue
        matched_indices.append(row_index)
        texts.append(text)
    return matched_indices, texts


def _doc_suffix(value: Any) -> int:
    try:
        return int(str(value).rsplit(":", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"could not parse numeric doc suffix from {value!r}") from exc


def _load_teacher_position_requests(
    teacher_table: Path,
    *,
    corpus_start: int,
    corpus_length: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    table = pq.read_table(teacher_table)
    names = set(table.column_names)
    if "doc_id" not in names:
        raise ValueError("teacher keys table requires doc_id")
    if "token_position" not in names and "n_raw_tokens" not in names:
        raise ValueError("teacher keys table requires token_position or n_raw_tokens")

    available = ["doc_id"]
    for name in ("token_position", "n_raw_tokens", "token_id"):
        if name in names:
            available.append(name)
    text_column = next((name for name in TEXT_COLUMNS if name in names), None)
    if text_column is not None:
        available.append(text_column)

    columns = {name: _column_to_list(table, name) for name in available}
    start = int(corpus_start)
    end = start + int(corpus_length)
    grouped: dict[str, list[dict[str, Any]]] = {}
    outside_rows = 0
    invalid_rows = 0
    empty_text_rows = 0
    suffixes: list[int] = []

    for row_index in range(table.num_rows):
        doc_id = str(columns["doc_id"][row_index])
        suffix = _doc_suffix(doc_id)
        if suffix < start or suffix >= end:
            outside_rows += 1
            continue
        suffixes.append(suffix)
        raw_position = columns.get("token_position", [None] * table.num_rows)[row_index]
        raw_n_tokens = columns.get("n_raw_tokens", [None] * table.num_rows)[row_index]
        if raw_position is None:
            if raw_n_tokens is None:
                invalid_rows += 1
                continue
            position = int(raw_n_tokens) - 1
        else:
            position = int(raw_position)
        if position < 0:
            invalid_rows += 1
            continue

        request: dict[str, Any] = {"position": position, "source_row": row_index}
        if raw_n_tokens is not None:
            request["n_raw_tokens"] = int(raw_n_tokens)
        raw_token_id = columns.get("token_id", [None] * table.num_rows)[row_index]
        if raw_token_id is not None:
            request["token_id"] = int(raw_token_id)
        if text_column is not None:
            text = columns[text_column][row_index]
            if text is None or not str(text).strip():
                empty_text_rows += 1
            request["text"] = "" if text is None else str(text)
        grouped.setdefault(doc_id, []).append(request)

    for requests in grouped.values():
        requests.sort(key=lambda item: int(item["source_row"]))

    missing_suffixes: list[int] = []
    if suffixes:
        seen_suffixes = set(suffixes)
        missing_suffixes = [value for value in range(min(seen_suffixes), max(seen_suffixes) + 1) if value not in seen_suffixes]

    summary = {
        "table": str(teacher_table),
        "rows": table.num_rows,
        "requested_rows": sum(len(requests) for requests in grouped.values()),
        "requested_docs": len(grouped),
        "outside_slice_rows": outside_rows,
        "invalid_rows": invalid_rows,
        "empty_text_rows": empty_text_rows,
        "has_text_column": text_column is not None,
        "text_column": text_column,
        "numeric_doc_suffix_min": min(suffixes) if suffixes else None,
        "numeric_doc_suffix_max": max(suffixes) if suffixes else None,
        "numeric_doc_suffix_unique_count": len(set(suffixes)),
        "numeric_doc_suffix_contiguous": len(missing_suffixes) == 0,
        "missing_numeric_suffix_count": len(missing_suffixes),
        "missing_numeric_suffixes_preview": missing_suffixes[:50],
    }
    return grouped, summary


def _positions_from_teacher_requests(
    token_ids: list[int],
    requests: list[dict[str, Any]],
    payload: dict[str, Any],
) -> list[int]:
    positions: list[int] = []
    skipped = payload.setdefault("skipped", {})
    for key in ("teacher_position_oob", "teacher_token_mismatch", "teacher_n_raw_mismatch"):
        skipped.setdefault(key, 0)
    for request in requests:
        position = int(request["position"])
        if position >= len(token_ids):
            skipped["teacher_position_oob"] += 1
            continue
        expected_n_raw = request.get("n_raw_tokens")
        if expected_n_raw is not None and int(expected_n_raw) != position + 1:
            skipped["teacher_n_raw_mismatch"] += 1
            continue
        expected_token_id = request.get("token_id")
        if expected_token_id is not None and int(expected_token_id) != int(token_ids[position]):
            skipped["teacher_token_mismatch"] += 1
            continue
        positions.append(position)
    return positions


def _split_indices(doc_ids: list[str], *, validation_limit: int, test_limit: int, seed: int) -> dict[str, np.ndarray]:
    docs = sorted(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(docs)
    doc_to_rows: dict[str, list[int]] = {}
    for index, doc_id in enumerate(doc_ids):
        doc_to_rows.setdefault(doc_id, []).append(index)

    validation_docs: list[str] = []
    test_docs: list[str] = []
    train_docs: list[str] = []
    val_rows = 0
    test_rows = 0
    for doc_id in docs:
        rows = doc_to_rows[doc_id]
        if val_rows < validation_limit:
            validation_docs.append(doc_id)
            val_rows += len(rows)
        elif test_rows < test_limit:
            test_docs.append(doc_id)
            test_rows += len(rows)
        else:
            train_docs.append(doc_id)

    if not train_docs and len(docs) >= 3:
        train_docs.append(test_docs.pop())
    if not validation_docs and train_docs:
        validation_docs.append(train_docs.pop())
    if not test_docs and train_docs:
        test_docs.append(train_docs.pop())

    def rows_for(split_docs: list[str], limit: int | None = None) -> np.ndarray:
        rows = [row for doc_id in split_docs for row in doc_to_rows[doc_id]]
        if limit is not None:
            rows = rows[:limit]
        return np.asarray(rows, dtype=np.int64)

    return {
        "train": rows_for(train_docs),
        "validation": rows_for(validation_docs, validation_limit),
        "test": rows_for(test_docs, test_limit),
    }


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def _tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _hash_token(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "little") % dim
    sign = 1.0 if digest[8] & 1 else -1.0
    return index, sign


def _text_features(texts: list[str], dim: int) -> np.ndarray:
    features = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in _tokenize_text(text):
            index, sign = _hash_token(token, dim)
            features[row, index] += sign
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    np.divide(features, np.maximum(norms, 1e-6), out=features)
    return features


def _predict_knn(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    eval_features: np.ndarray,
    *,
    k: int,
    chunk_size: int = 256,
) -> np.ndarray:
    if len(train_targets) == 0 or len(eval_features) == 0:
        return np.zeros((len(eval_features), train_targets.shape[1] if train_targets.ndim == 2 else 0), dtype=np.float32)
    k = min(max(1, int(k)), len(train_targets))
    predictions: list[np.ndarray] = []
    for start in range(0, len(eval_features), chunk_size):
        chunk = eval_features[start : start + chunk_size]
        scores = chunk @ train_features.T
        if k == len(train_targets):
            neighbor_indices = np.tile(np.arange(len(train_targets)), (len(chunk), 1))
        else:
            neighbor_indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        pred = train_targets[neighbor_indices].mean(axis=1)
        predictions.append(pred.astype(np.float32, copy=False))
    return np.concatenate(predictions, axis=0) if predictions else np.zeros_like(train_targets[:0])


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if len(target) == 0:
        return {"normalized_mse": float("nan"), "fve": float("nan"), "n": 0.0}
    diff = prediction.astype(np.float64) - target.astype(np.float64)
    sse = float(np.square(diff).sum())
    target_centered = target.astype(np.float64) - target.astype(np.float64).mean(axis=0, keepdims=True)
    sst = float(np.square(target_centered).sum())
    energy = float(np.square(target.astype(np.float64)).sum())
    denom = max(energy, 1e-12)
    return {
        "normalized_mse": sse / denom,
        "fve": 1.0 - (sse / max(sst, 1e-12)),
        "n": float(len(target)),
    }


def _score_split(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    eval_features: np.ndarray,
    eval_targets: np.ndarray,
    mean_vector: np.ndarray,
    knn_k: int,
) -> dict[str, float]:
    mean_prediction = np.repeat(mean_vector[None, :], len(eval_targets), axis=0)
    knn_prediction = _predict_knn(train_features, train_targets, eval_features, k=knn_k)
    mean_metrics = _metrics(mean_prediction, eval_targets)
    knn_metrics = _metrics(knn_prediction, eval_targets)
    return {
        "n": int(mean_metrics["n"]),
        "mean_normalized_mse": mean_metrics["normalized_mse"],
        "mean_fve": mean_metrics["fve"],
        "teacher_knn_normalized_mse": knn_metrics["normalized_mse"],
        "teacher_knn_fve": knn_metrics["fve"],
    }


def score_layers(
    *,
    layers: list[int],
    output_root: Path,
    teacher_table: Path,
    report_json: Path,
    validation_limit: int = 512,
    test_limit: int = 512,
    hash_dim: int = 512,
    knn_k: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    teacher = _read_table(Path(teacher_table))
    text_column = _available_text_column(teacher)
    report: dict[str, Any] = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "output_root": str(output_root),
        "teacher_table": str(teacher_table),
        "text_column": text_column,
        "layers_requested": list(layers),
        "validation_limit": validation_limit,
        "test_limit": test_limit,
        "hash_dim": hash_dim,
        "knn_k": knn_k,
        "layers": [],
    }

    for layer in layers:
        base_path = Path(output_root) / f"R_{layer}" / "base.parquet"
        layer_payload: dict[str, Any] = {"layer": layer, "base_path": str(base_path)}
        try:
            base = _read_table(base_path)
            keys = _join_keys(base, teacher)
            lookup = _teacher_lookup(teacher, keys, text_column)
            matched_indices, texts = _match_base_to_teacher(base, lookup, keys)
            if not matched_indices:
                raise ValueError("no base rows matched the teacher table")
            activations = _activation_matrix(base)[np.asarray(matched_indices, dtype=np.int64)]
            doc_ids_all = _column_to_list(base, "doc_id")
            doc_ids = [str(doc_ids_all[index]) for index in matched_indices]
            splits = _split_indices(doc_ids, validation_limit=validation_limit, test_limit=test_limit, seed=seed)
            if len(splits["train"]) == 0:
                raise ValueError("split left no train rows for controls")
            features = _text_features(texts, hash_dim)
            mean_vector = activations[splits["train"]].mean(axis=0)
            layer_payload.update(
                {
                    "matched_rows": len(matched_indices),
                    "join_keys": keys,
                    "split_counts": {name: int(len(indexes)) for name, indexes in splits.items()},
                    "validation": _score_split(
                        train_features=features[splits["train"]],
                        train_targets=activations[splits["train"]],
                        eval_features=features[splits["validation"]],
                        eval_targets=activations[splits["validation"]],
                        mean_vector=mean_vector,
                        knn_k=knn_k,
                    ),
                    "test": _score_split(
                        train_features=features[splits["train"]],
                        train_targets=activations[splits["train"]],
                        eval_features=features[splits["test"]],
                        eval_targets=activations[splits["test"]],
                        mean_vector=mean_vector,
                        knn_k=knn_k,
                    ),
                }
            )
        except Exception as exc:
            layer_payload.update(
                {
                    "matched_rows": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=6),
                }
            )
        report["layers"].append(layer_payload)

    successful = [item for item in report["layers"] if "validation" in item]
    if successful:
        report["best_validation_teacher_knn"] = min(
            successful,
            key=lambda item: item["validation"]["teacher_knn_normalized_mse"],
        )
        report["best_validation_teacher_vs_mean_delta"] = max(
            successful,
            key=lambda item: item["validation"]["mean_normalized_mse"]
            - item["validation"]["teacher_knn_normalized_mse"],
        )
    write_json(Path(report_json), report)
    return report


def _import_extraction_helpers() -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    repo_root = script_dir.parent
    nla_root = repo_root / "external" / "natural_language_autoencoders"
    if nla_root.exists() and str(nla_root) not in sys.path:
        sys.path.insert(0, str(nla_root))

    import torch
    from nano_extraction_identity import _layer_mask_for_block, _module_execution_device, _move_optional_tensor
    from nano_introspection import (
        DEFAULT_MODEL_ID,
        add_bool_optional_arg,
        classify_blocker,
        load_config_from_args,
        load_model_from_args,
        load_tokenizer_from_args,
        resolve_nano_module_paths,
    )
    from nano_realdata_stage0_extract import _dataset_id, _encode_batch, _load_corpus, _model_start_device, _sample_positions, _schema

    return {
        "torch": torch,
        "_layer_mask_for_block": _layer_mask_for_block,
        "_module_execution_device": _module_execution_device,
        "_move_optional_tensor": _move_optional_tensor,
        "DEFAULT_MODEL_ID": DEFAULT_MODEL_ID,
        "add_bool_optional_arg": add_bool_optional_arg,
        "classify_blocker": classify_blocker,
        "load_config_from_args": load_config_from_args,
        "load_model_from_args": load_model_from_args,
        "load_tokenizer_from_args": load_tokenizer_from_args,
        "resolve_nano_module_paths": resolve_nano_module_paths,
        "_dataset_id": _dataset_id,
        "_encode_batch": _encode_batch,
        "_load_corpus": _load_corpus,
        "_model_start_device": _model_start_device,
        "_sample_positions": _sample_positions,
        "_schema": _schema,
    }


def _capture_selected(hidden_states: Any, selected_positions: list[tuple[int, int]]) -> dict[tuple[int, int], list[float]]:
    captures: dict[tuple[int, int], list[float]] = {}
    for batch_idx, pos in selected_positions:
        captures[(batch_idx, pos)] = hidden_states[batch_idx, pos].detach().float().cpu().tolist()
    return captures


def _forward_selected_boundaries(
    *,
    helpers: dict[str, Any],
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    layers: list[int],
    selected_positions: list[tuple[int, int]],
) -> dict[int, dict[tuple[int, int], list[float]]]:
    torch = helpers["torch"]
    resolve_nano_module_paths = helpers["resolve_nano_module_paths"]
    _module_execution_device = helpers["_module_execution_device"]
    _move_optional_tensor = helpers["_move_optional_tensor"]
    _layer_mask_for_block = helpers["_layer_mask_for_block"]

    resolved = resolve_nano_module_paths(model)
    backbone = resolved["backbone"].obj
    blocks = resolved["layers"].obj
    embeddings = resolved["embeddings"].obj
    if backbone is None or blocks is None or embeddings is None:
        raise RuntimeError(f"could not resolve Nano modules: {json_safe(resolved)}")
    max_boundary = max(layers)
    if not 0 <= max_boundary <= len(blocks):
        raise ValueError(f"requested max boundary R{max_boundary}, but model has {len(blocks)} blocks")

    layer_set = set(layers)
    captured: dict[int, dict[tuple[int, int], list[float]]] = {}
    hidden_states = embeddings(input_ids)
    if 0 in layer_set:
        captured[0] = _capture_selected(hidden_states, selected_positions)

    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    for layer_idx in range(max_boundary):
        block = blocks[layer_idx]
        block_device = _module_execution_device(block, hidden_states.device)
        if hidden_states.device != block_device:
            hidden_states = hidden_states.to(block_device)
        block_attention_mask = _move_optional_tensor(attention_mask, block_device)
        block_cache_position = cache_position.to(block_device) if cache_position.device != block_device else cache_position
        layer_mask = _layer_mask_for_block(backbone, block, block_attention_mask, hidden_states, block_cache_position)
        output = block(
            hidden_states,
            cache_params=None,
            cache_position=block_cache_position,
            attention_mask=layer_mask,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
        boundary = layer_idx + 1
        if boundary in layer_set:
            captured[boundary] = _capture_selected(hidden_states, selected_positions)
    return captured


def _write_layer_sidecar(args: argparse.Namespace, helpers: dict[str, Any], *, output: Path, row_count: int, d_model: int, layer: int) -> None:
    try:
        from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar_local

        corpus_slice = {"start": args.corpus_start, "length": args.corpus_length}
        meta = NLADatasetMeta(
            dataset_id=helpers["_dataset_id"](args.model_id, args.model_revision, layer, args.corpus, corpus_slice),
            stage="base",
            row_count=row_count,
            extraction=NLAExtractionMeta(
                base_model=args.model_id,
                d_model=d_model,
                layer_index=layer,
                norm="none",
                corpus=args.corpus,
                corpus_slice=corpus_slice,
                positions_per_doc=args.positions_per_doc,
            ),
            created_by="scripts.nano_ar_layer_sweep",
        )
        write_sidecar_local(output, meta)
    except Exception as exc:
        fallback = {
            "dataset_id": f"base_nano_layer_sweep_R{layer}",
            "stage": "base",
            "row_count": row_count,
            "extraction": {
                "base_model": args.model_id,
                "d_model": d_model,
                "layer_index": layer,
                "norm": "none",
                "corpus": args.corpus,
                "corpus_slice": {"start": args.corpus_start, "length": args.corpus_length},
                "positions_per_doc": args.positions_per_doc,
            },
            "created_by": "scripts.nano_ar_layer_sweep",
            "sidecar_fallback_reason": f"{type(exc).__name__}: {exc}",
        }
        Path(str(output) + ".nla_meta.yaml").write_text(yaml.safe_dump(fallback, sort_keys=False))


def extract_layers(args: argparse.Namespace) -> dict[str, Any]:
    helpers = _import_extraction_helpers()
    torch = helpers["torch"]
    layers = parse_boundary_spec(args.layers)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_output or output_root / "extract_metadata.json")
    payload: dict[str, Any] = {
        "schema_version": EXTRACT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "layers": layers,
        "output_root": str(output_root),
        "row_counts": {f"R_{layer}": 0 for layer in layers},
        "skipped": {"missing_text": 0, "too_short": 0, "short_sampled": 0},
        "blockers": [],
    }
    teacher_requests: dict[str, list[dict[str, Any]]] | None = None
    if args.teacher_keys_table is not None:
        teacher_requests, teacher_summary = _load_teacher_position_requests(
            Path(args.teacher_keys_table),
            corpus_start=args.corpus_start,
            corpus_length=args.corpus_length,
        )
        payload["teacher_keys"] = teacher_summary
        payload["skipped"].update(
            {
                "teacher_docs_not_requested": 0,
                "teacher_position_oob": 0,
                "teacher_token_mismatch": 0,
                "teacher_n_raw_mismatch": 0,
            }
        )

    layer_paths = {layer: output_root / f"R_{layer}" / "base.parquet" for layer in layers}
    for path in layer_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    writers: dict[int, Any] = {}
    try:
        tokenizer = helpers["load_tokenizer_from_args"](args)
        config, config_error = helpers["load_config_from_args"](args)
        if config_error is not None:
            payload["blockers"].append(helpers["classify_blocker"]("remote-code load", config_error))
        model = helpers["load_model_from_args"](args, config)
        model.eval()
        d_model = int(getattr(config, "hidden_size"))
        schema = helpers["_schema"](d_model, args.keep_token_metadata)
        examples = helpers["_load_corpus"](args)
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        for layer, path in layer_paths.items():
            writers[layer] = pq.ParquetWriter(path, schema)
        write_json(metadata_path, payload)
    except Exception as exc:
        payload["blockers"].append({"kind": "setup", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)})
        write_json(metadata_path, payload)
        raise

    try:
        with torch.no_grad():
            for chunk_start in range(0, len(examples), args.chunk_size):
                chunk = examples[chunk_start : chunk_start + args.chunk_size]
                texts: list[str] = []
                doc_indices: list[int] = []
                for offset, row in enumerate(chunk):
                    text = row.get(args.text_column)
                    if not isinstance(text, str) or not text:
                        payload["skipped"]["missing_text"] += 1
                        continue
                    texts.append(text)
                    doc_indices.append(args.corpus_start + chunk_start + offset)

                for batch_start in range(0, len(texts), args.batch_size):
                    batch_texts = texts[batch_start : batch_start + args.batch_size]
                    batch_doc_indices = doc_indices[batch_start : batch_start + args.batch_size]
                    encoded = helpers["_encode_batch"](tokenizer, batch_texts, args.max_length)
                    input_ids = encoded["input_ids"].to(helpers["_model_start_device"](model))
                    attention_mask = encoded["attention_mask"].to(input_ids.device)

                    doc_records: list[dict[str, Any]] = []
                    selected_positions: list[tuple[int, int]] = []
                    for batch_idx, doc_idx in enumerate(batch_doc_indices):
                        valid_len = int(attention_mask[batch_idx].sum().item())
                        token_ids = input_ids[batch_idx, :valid_len].detach().cpu().tolist()
                        doc_id = f"{args.corpus}:{args.corpus_split}:{doc_idx}"
                        if teacher_requests is not None:
                            requests = teacher_requests.get(doc_id)
                            if not requests:
                                payload["skipped"]["teacher_docs_not_requested"] += 1
                                continue
                            positions = _positions_from_teacher_requests(token_ids, requests, payload)
                        else:
                            positions = helpers["_sample_positions"](token_ids, args.positions_per_doc, special_ids, doc_id, args.seed)
                        if not positions:
                            payload["skipped"]["too_short"] += 1
                            continue
                        if len(positions) < args.positions_per_doc:
                            payload["skipped"]["short_sampled"] += 1
                        doc_records.append({"batch_idx": batch_idx, "doc_id": doc_id, "token_ids": token_ids, "positions": positions})
                        selected_positions.extend((batch_idx, pos) for pos in positions)

                    if not selected_positions:
                        continue
                    captures = _forward_selected_boundaries(
                        helpers=helpers,
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        layers=layers,
                        selected_positions=selected_positions,
                    )

                    rows_by_layer: dict[int, dict[str, list[Any]]] = {
                        layer: {name: [] for name in schema.names} for layer in layers
                    }
                    for record in doc_records:
                        batch_idx = int(record["batch_idx"])
                        token_ids = record["token_ids"]
                        doc_id = record["doc_id"]
                        for pos in record["positions"]:
                            truncated_ids = token_ids[: pos + 1]
                            token_id = int(token_ids[pos])
                            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                            truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
                            for layer in layers:
                                rows = rows_by_layer[layer]
                                rows["n_raw_tokens"].append(pos + 1)
                                rows["detokenized_text_truncated"].append(truncated_text)
                                rows["activation_vector"].append(captures[layer][(batch_idx, pos)])
                                rows["activation_layer"].append(layer)
                                rows["doc_id"].append(doc_id)
                                if args.keep_token_metadata:
                                    rows["token_position"].append(pos)
                                    rows["token_id"].append(token_id)
                                    rows["token_text"].append(token_text)
                                    rows["token_ids_prefix"].append([int(value) for value in truncated_ids])

                    for layer, rows in rows_by_layer.items():
                        if not rows["doc_id"]:
                            continue
                        writers[layer].write_table(pa.Table.from_pydict(rows, schema=schema))
                        payload["row_counts"][f"R_{layer}"] += len(rows["doc_id"])
                    write_json(metadata_path, payload)

        for layer, writer in writers.items():
            writer.close()
            writers[layer] = None
            row_count = int(payload["row_counts"][f"R_{layer}"])
            _write_layer_sidecar(args, helpers, output=layer_paths[layer], row_count=row_count, d_model=d_model, layer=layer)
            write_json(layer_paths[layer].with_suffix(layer_paths[layer].suffix + ".metadata.json"), {
                "schema_version": EXTRACT_SCHEMA_VERSION,
                "layer": layer,
                "row_count": row_count,
                "output": str(layer_paths[layer]),
            })
        payload["completed_at"] = utc_now()
        payload["sidecars"] = {f"R_{layer}": str(layer_paths[layer]) + ".nla_meta.yaml" for layer in layers}
        write_json(metadata_path, payload)
        if teacher_requests is not None and args.strict_teacher_keys:
            requested_rows = int(payload["teacher_keys"]["requested_rows"])
            bad_layers = [
                f"R_{layer}={payload['row_counts'][f'R_{layer}']}"
                for layer in layers
                if int(payload["row_counts"][f"R_{layer}"]) != requested_rows
            ]
            if bad_layers:
                raise RuntimeError(
                    "teacher-key extraction did not cover every requested row: "
                    f"requested={requested_rows}, extracted={', '.join(bad_layers)}, skipped={payload['skipped']}"
                )
        return payload
    except Exception as exc:
        payload["blockers"].append({"kind": "extraction", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)})
        write_json(metadata_path, payload)
        raise
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()


def _add_extract_args(parser: argparse.ArgumentParser) -> None:
    def add_bool_optional_arg(local_parser: argparse.ArgumentParser, name: str, *, default: bool) -> None:
        local_parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)

    parser.add_argument("--layers", required=True, help="Layer boundaries, e.g. R25-R51.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--model-id", default=os.environ.get("NANO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--model-revision", default=os.environ.get("NANO_MODEL_REVISION"))
    parser.add_argument("--tokenizer-revision", default=os.environ.get("NANO_TOKENIZER_REVISION"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    add_bool_optional_arg(parser, "--trust-remote-code", default=True)
    parser.add_argument("--corpus", default="HuggingFaceFW/fineweb")
    parser.add_argument("--corpus-config", default="sample-10BT")
    parser.add_argument("--corpus-revision", default=None)
    parser.add_argument("--corpus-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--corpus-start", type=int, default=0)
    parser.add_argument("--corpus-length", type=int, default=64)
    add_bool_optional_arg(parser, "--streaming", default=True)
    parser.add_argument("--positions-per-doc", type=int, default=2)
    parser.add_argument("--teacher-keys-table", type=Path, default=None)
    parser.add_argument("--strict-teacher-keys", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-token-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(load_mode="full")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    plan = subparsers.add_parser("plan", help="Write a layer-sweep queue YAML.")
    plan.add_argument("--queue", type=Path, required=True)
    plan.add_argument("--layers", required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--teacher-table", type=Path, required=True)
    plan.add_argument("--model-id", required=True)
    plan.add_argument("--code-root", type=Path, default=Path.cwd())
    plan.add_argument("--python", dest="python_bin", default=sys.executable)
    plan.add_argument("--corpus-start", type=int, default=10500)
    plan.add_argument("--corpus-length", type=int, default=2048)
    plan.add_argument("--positions-per-doc", type=int, default=10)
    plan.add_argument("--score-top-k", type=int, default=5)
    plan.add_argument("--validation-limit", type=int, default=512)
    plan.add_argument("--test-limit", type=int, default=512)
    plan.add_argument("--hash-dim", type=int, default=512)
    plan.add_argument("--batch-size", type=int, default=2)
    plan.add_argument("--chunk-size", type=int, default=8)
    plan.add_argument("--max-length", type=int, default=1024)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--no-local-files-only", action="store_true")
    plan.add_argument("--no-overwrite", action="store_true")

    watch = subparsers.add_parser("watch", help="Process queue items sequentially.")
    watch.add_argument("queue", type=Path)
    watch.add_argument("--status", action="store_true")
    watch.add_argument("--dry-run", action="store_true")
    watch.add_argument("--once", action="store_true")
    watch.add_argument("--run-until-empty", action="store_true")
    watch.add_argument("--reset-active", action="store_true")
    watch.add_argument("--continue-on-failure", action="store_true")
    watch.add_argument("--poll-seconds", type=int, default=60)

    score = subparsers.add_parser("score", help="Score each layer with teacher-text kNN against mean-h control.")
    score.add_argument("--layers", required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument("--teacher-table", type=Path, required=True)
    score.add_argument("--report-json", type=Path, required=True)
    score.add_argument("--validation-limit", type=int, default=512)
    score.add_argument("--test-limit", type=int, default=512)
    score.add_argument("--hash-dim", type=int, default=512)
    score.add_argument("--knn-k", type=int, default=5)
    score.add_argument("--seed", type=int, default=42)

    extract = subparsers.add_parser("extract", help="Extract selected residual boundaries in one Nano forward pass.")
    _add_extract_args(extract)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command_name == "plan":
        queue_doc = build_queue_doc(
            code_root=args.code_root,
            python_bin=args.python_bin,
            layers=parse_boundary_spec(args.layers),
            output_root=args.output_root,
            teacher_table=args.teacher_table,
            model_id=args.model_id,
            corpus_start=args.corpus_start,
            corpus_length=args.corpus_length,
            positions_per_doc=args.positions_per_doc,
            score_top_k=args.score_top_k,
            validation_limit=args.validation_limit,
            test_limit=args.test_limit,
            hash_dim=args.hash_dim,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            max_length=args.max_length,
            seed=args.seed,
            local_files_only=not args.no_local_files_only,
            overwrite=not args.no_overwrite,
        )
        write_queue(args.queue, queue_doc)
        print(json.dumps(json_safe({"queue": args.queue, "items": [item["name"] for item in queue_doc["items"]]}), indent=2))
        return 0
    if args.command_name == "watch":
        if args.reset_active:
            print(json.dumps(reset_active_items(args.queue), indent=2, sort_keys=True))
        if args.status:
            print(json.dumps(queue_status(args.queue), indent=2, sort_keys=True))
            return 0
        return watch_queue(
            args.queue,
            poll_seconds=args.poll_seconds,
            dry_run=args.dry_run,
            once=args.once,
            stop_when_idle=args.run_until_empty,
            stop_on_failure=not args.continue_on_failure,
        )
    if args.command_name == "score":
        report = score_layers(
            layers=parse_boundary_spec(args.layers),
            output_root=args.output_root,
            teacher_table=args.teacher_table,
            report_json=args.report_json,
            validation_limit=args.validation_limit,
            test_limit=args.test_limit,
            hash_dim=args.hash_dim,
            knn_k=args.knn_k,
            seed=args.seed,
        )
        print(json.dumps(json_safe(report), indent=2, sort_keys=True))
        return 0
    if args.command_name == "extract":
        payload = extract_layers(args)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command_name)


if __name__ == "__main__":
    raise SystemExit(main())
