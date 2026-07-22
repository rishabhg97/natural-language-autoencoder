#!/usr/bin/env python3
"""Information-ceiling diagnostics for Nano AR teacher explanations."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
KNN_TEXT_MODES = (
    "explanation",
    "explanation_target_token",
    "explanation_target_token_id",
    "explanation_position_bucket",
    "explanation_all_token_hints",
)


def canonicalize_explanation(text: str | None) -> str:
    if text is None:
        return ""
    text = TAG_RE.sub(" ", str(text))
    words = WORD_RE.findall(text.lower())
    return " ".join(words)


def l2_normalize_rows(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), eps)
    return arr / norms


def normalized_mean_vector(vectors: np.ndarray | list[list[float]]) -> np.ndarray:
    normalized = l2_normalize_rows(np.asarray(vectors, dtype=np.float32))
    mean = normalized.mean(axis=0, keepdims=True)
    return l2_normalize_rows(mean)[0]


def normalized_mse_rows(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pred = l2_normalize_rows(predictions)
    gold = l2_normalize_rows(targets)
    return np.mean(np.square(pred - gold), axis=-1)


def _row_explanation(row: dict[str, Any]) -> str:
    for key in ("explanation", "api_explanation", "teacher_explanation"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    prompt = row.get("prompt")
    return str(prompt or "")


def _row_vector(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(row["activation_vector"], dtype=np.float32)


def duplicate_group_floor(rows: list[dict[str, Any]], *, min_group_size: int = 2) -> dict[str, Any]:
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = canonicalize_explanation(_row_explanation(row))
        if key:
            groups[key].append(_row_vector(row))

    losses: list[float] = []
    group_reports: list[dict[str, Any]] = []
    covered = 0
    for key, vectors in sorted(groups.items()):
        if len(vectors) < min_group_size:
            continue
        targets = np.stack(vectors)
        pred = np.repeat(normalized_mean_vector(targets)[None, :], len(vectors), axis=0)
        row_losses = normalized_mse_rows(pred, targets)
        losses.extend(float(item) for item in row_losses)
        covered += len(vectors)
        group_reports.append(
            {
                "key": key,
                "row_count": len(vectors),
                "mean_nmse": float(row_losses.mean()),
            }
        )

    return {
        "group_count": len(group_reports),
        "covered_row_count": covered,
        "mean_nmse": float(np.mean(losses)) if losses else None,
        "groups_sample": group_reports[:20],
    }


def knn_floor(
    *,
    train_features: np.ndarray,
    train_vectors: np.ndarray,
    eval_features: np.ndarray,
    eval_vectors: np.ndarray,
    k: int,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")
    train_features = np.asarray(train_features, dtype=np.float32)
    train_vectors = np.asarray(train_vectors, dtype=np.float32)
    eval_features = np.asarray(eval_features, dtype=np.float32)
    eval_vectors = np.asarray(eval_vectors, dtype=np.float32)
    if train_features.ndim != 2 or eval_features.ndim != 2:
        raise ValueError("features must be 2D")
    if train_features.shape[0] != train_vectors.shape[0]:
        raise ValueError("train features/vectors row mismatch")
    if eval_features.shape[0] != eval_vectors.shape[0]:
        raise ValueError("eval features/vectors row mismatch")

    predictions: list[np.ndarray] = []
    k_eff = min(k, train_features.shape[0])
    for feature in eval_features:
        dist = np.sum(np.square(train_features - feature[None, :]), axis=1)
        nearest = np.argpartition(dist, kth=k_eff - 1)[:k_eff]
        predictions.append(normalized_mean_vector(train_vectors[nearest]))
    pred = np.stack(predictions) if predictions else np.zeros_like(eval_vectors)
    row_losses = normalized_mse_rows(pred, eval_vectors)
    return {
        "row_count": int(eval_vectors.shape[0]),
        "k": int(k_eff),
        "mean_nmse": float(row_losses.mean()) if len(row_losses) else None,
        "median_nmse": float(np.median(row_losses)) if len(row_losses) else None,
    }


def baseline_keys_for_row(row: dict[str, Any], *, position_bucket_size: int = 50) -> dict[str, str | None]:
    target_token = row.get("target_token", row.get("token_text"))
    target_token_id = row.get("target_token_id", row.get("token_id"))
    position = row.get("target_position", row.get("token_position"))
    before = row.get("tokens_before") or []
    after = row.get("tokens_after") or []

    out: dict[str, str | None] = {
        "target_token": f"tok:{target_token}" if target_token not in (None, "") else None,
        "target_token_id": f"tok_id:{target_token_id}" if target_token_id is not None else None,
        "position_bucket": None,
        "local_window": None,
    }
    if position is not None:
        start = int(position) // position_bucket_size * position_bucket_size
        out["position_bucket"] = f"pos:{start}-{start + position_bucket_size - 1}"
    if before or after:
        before_text = " ".join(str(item) for item in before)
        after_text = " ".join(str(item) for item in after)
        out["local_window"] = f"local:{before_text} <T> {after_text}".strip()
    return out


def feature_text_for_row(row: dict[str, Any], mode: str, *, position_bucket_size: int = 50) -> str:
    if mode not in KNN_TEXT_MODES:
        raise ValueError(f"unknown feature text mode {mode!r}")
    base = _row_explanation(row)
    keys = baseline_keys_for_row(row, position_bucket_size=position_bucket_size)
    hints: list[str] = []
    if mode in ("explanation_target_token", "explanation_all_token_hints") and keys["target_token"]:
        hints.append(f"target_token={keys['target_token'].removeprefix('tok:')}")
    if mode in ("explanation_target_token_id", "explanation_all_token_hints") and keys["target_token_id"]:
        hints.append(f"target_token_id={keys['target_token_id'].removeprefix('tok_id:')}")
    if mode in ("explanation_position_bucket", "explanation_all_token_hints") and keys["position_bucket"]:
        hints.append(f"position_bucket={keys['position_bucket']}")
    if not hints:
        return base
    return f"{base} {' '.join(hints)}"


def group_key_floor(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    key_name: str,
    position_bucket_size: int = 50,
) -> dict[str, Any]:
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    train_vectors = np.stack([_row_vector(row) for row in train_rows])
    global_pred = normalized_mean_vector(train_vectors)
    for row in train_rows:
        key = baseline_keys_for_row(row, position_bucket_size=position_bucket_size).get(key_name)
        if key:
            groups[key].append(_row_vector(row))

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    matched = 0
    for row in eval_rows:
        key = baseline_keys_for_row(row, position_bucket_size=position_bucket_size).get(key_name)
        if key and key in groups:
            preds.append(normalized_mean_vector(np.stack(groups[key])))
            matched += 1
        else:
            preds.append(global_pred)
        targets.append(_row_vector(row))
    row_losses = normalized_mse_rows(np.stack(preds), np.stack(targets)) if targets else np.asarray([])
    return {
        "key_name": key_name,
        "row_count": len(eval_rows),
        "matched_count": matched,
        "matched_fraction": float(matched / len(eval_rows)) if eval_rows else 0.0,
        "mean_nmse": float(row_losses.mean()) if len(row_losses) else None,
    }


def bucket_summary(values: Iterable[float], buckets: Iterable[str]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, bucket in zip(values, buckets):
        grouped[str(bucket)].append(float(value))
    return {
        bucket: {
            "row_count": len(items),
            "mean": float(np.mean(items)),
            "median": float(np.median(items)),
        }
        for bucket, items in sorted(grouped.items())
    }


def text_hash_features(texts: list[str], *, dim: int = 256) -> np.ndarray:
    features = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        words = WORD_RE.findall(text.lower())
        for word in words:
            features[row, hash(word) % dim] += 1.0
    return l2_normalize_rows(features)


def _read_rows(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    columns = {name: table.column(name).to_pylist() for name in table.schema.names}
    rows = []
    for idx in range(table.num_rows):
        row = {name: values[idx] for name, values in columns.items()}
        if "explanation" not in row and "api_explanation" in row:
            row["explanation"] = row["api_explanation"]
        rows.append(row)
    return rows


def _vectors(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([_row_vector(row) for row in rows]) if rows else np.zeros((0, 0), dtype=np.float32)


def _explanations(rows: list[dict[str, Any]]) -> list[str]:
    return [canonicalize_explanation(_row_explanation(row)) for row in rows]


def _feature_texts(rows: list[dict[str, Any]], mode: str) -> list[str]:
    return [feature_text_for_row(row, mode) for row in rows]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = _read_rows(args.train_parquet, limit=args.train_limit)
    validation = _read_rows(args.validation_parquet, limit=args.validation_limit)
    test = _read_rows(args.test_parquet, limit=args.test_limit)
    train_features_by_mode = {
        mode: text_hash_features(_feature_texts(train, mode), dim=args.feature_dim)
        for mode in KNN_TEXT_MODES
    }
    report: dict[str, Any] = {
        "schema_version": "nano_ar_information_ceiling.v1",
        "train_row_count": len(train),
        "validation_row_count": len(validation),
        "test_row_count": len(test),
        "duplicate_train_floor": duplicate_group_floor(train),
        "splits": {},
    }
    for split_name, rows in (("validation", validation), ("test", test)):
        eval_features_by_mode = {
            mode: text_hash_features(_feature_texts(rows, mode), dim=args.feature_dim)
            for mode in KNN_TEXT_MODES
        }
        knn_text_floors = {
            mode: knn_floor(
                train_features=train_features_by_mode[mode],
                train_vectors=_vectors(train),
                eval_features=eval_features_by_mode[mode],
                eval_vectors=_vectors(rows),
                k=args.knn_k,
            )
            for mode in KNN_TEXT_MODES
        }
        split_report = {
            "knn_explanation_floor": knn_text_floors["explanation"],
            "knn_text_floors": knn_text_floors,
            "baseline_floors": {
                key: group_key_floor(train, rows, key_name=key)
                for key in ("target_token", "target_token_id", "position_bucket", "local_window")
            },
        }
        lengths = [len(text.split()) for text in _explanations(rows)]
        split_report["explanation_length_buckets"] = bucket_summary(
            lengths,
            [
                "len:0-24" if length < 25 else "len:25-74" if length < 75 else "len:75+"
                for length in lengths
            ],
        )
        report["splits"][split_name] = split_report
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--validation-parquet", required=True, type=Path)
    parser.add_argument("--test-parquet", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int, default=512)
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--knn-k", type=int, default=8)
    args = parser.parse_args()

    report = build_report(args)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
