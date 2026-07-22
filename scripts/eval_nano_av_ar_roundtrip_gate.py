#!/usr/bin/env python3
"""Evaluate the AV -> generated text -> AR reconstruction round-trip gate.

This gate is intentionally separate from the AR and AV HPO queues. It reuses
the AV checkpoint evaluator's generation semantics and the AR checkpoint
evaluator's critic prediction semantics, but loads them in two phases so a
single RunAI workspace does not need to keep both Nano models resident at once.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
for candidate in (SCRIPT_DIR, NLA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


ROUNDTRIP_SCHEMA_VERSION = "nano_av_ar_roundtrip_gate.v1"
ACTIVATION_METRIC_SCHEMA_VERSION = "nano_activation_reconstruction_metrics.v2"
GENERATION_PROTOCOL_SCHEMA_VERSION = "nano_generation_protocol.v1"
PARSE_QUALITY_SCHEMA_VERSION = "nano_generation_parse_quality.v2"
LENGTH_CONTROL_SCHEMA_VERSION = "nano_roundtrip_length_control.v1"
PREDICTION_CACHE_SCHEMA_VERSION = "nano_roundtrip_prediction_cache.v1"
DEFAULT_GENERATION_CONTROLS = ("real", "shuffled", "zero", "mean", "none")
CONTROL_NAMES = DEFAULT_GENERATION_CONTROLS
PRIMARY_VARIANT = "av_real"
EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)
EXPLANATION_OPEN_RE = re.compile(r"<explanation>\s*(.*)", re.DOTALL)
GENERATED_TEXT_FALLBACKS = ("empty", "raw")
QUALITY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)*")
NUMBER_TOKEN_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
MODEL_FINGERPRINT_RE = re.compile(r"^(?:dcp_model|hf_model)_sha256:[0-9a-f]{64}$")
TOKENIZER_FINGERPRINT_RE = re.compile(r"^tokenizer_files_sha256:[0-9a-f]{64}$")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_prediction_cache(
    path: Path,
    *,
    split_payloads: dict[str, dict[str, Any]],
    train_mean: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Persist AR predictions needed for lightweight, generation-free audits."""

    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            json.dumps(json_safe(metadata), sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "train_mean": np.asarray(train_mean, dtype=np.float32),
    }
    for split_name, payload in split_payloads.items():
        if not re.fullmatch(r"[a-z0-9_]+", split_name):
            raise ValueError(f"unsafe prediction-cache split name: {split_name!r}")
        prefix = f"{split_name}__"
        arrays[f"{prefix}row_indices"] = np.asarray(payload["row_indices"], dtype=np.int64)
        arrays[f"{prefix}doc_ids"] = np.asarray(payload["doc_ids"], dtype=np.str_)
        arrays[f"{prefix}content_family_ids"] = np.asarray(
            payload["content_family_ids"], dtype=np.str_
        )
        arrays[f"{prefix}targets"] = np.asarray(payload["targets"], dtype=np.float32)
        for variant, prediction in payload["predictions"].items():
            if not re.fullmatch(r"[a-z0-9_]+", variant):
                raise ValueError(f"unsafe prediction-cache variant name: {variant!r}")
            arrays[f"{prefix}prediction__{variant}"] = np.asarray(
                prediction, dtype=np.float32
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def generation_protocol_sha256(protocol: dict[str, Any]) -> str:
    serialized = json.dumps(json_safe(protocol), sort_keys=True, separators=(",", ":"))
    return sha256_text(serialized)


def build_generation_protocol(args: argparse.Namespace) -> dict[str, Any]:
    prefix = str(args.generation_prefix or "")
    parent_worker_count = getattr(args, "generation_parent_worker_count", None)
    worker_count = parent_worker_count or args.generation_workers
    return {
        "schema_version": GENERATION_PROTOCOL_SCHEMA_VERSION,
        "backend": str(args.generation_backend),
        "prefix": prefix,
        "prefix_sha256": sha256_text(prefix),
        "stop_text": None if args.stop_text is None else str(args.stop_text),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": False,
        "temperature": 0.0,
        "seed": int(args.seed),
        "injection_scale": str(args.injection_scale),
        "torch_dtype": str(args.torch_dtype),
        "attention_implementation": args.av_attn_implementation,
        "tokenizer_fingerprint": args.av_tokenizer_fingerprint,
        "worker_count": int(worker_count),
        "controls": [str(control) for control in args.generation_controls],
        "generated_text_fallback": str(args.generated_text_fallback),
    }


def build_generation_provenance(args: argparse.Namespace) -> dict[str, Any]:
    dataset_paths = {
        "train": getattr(args, "train_parquet", None),
        "validation": getattr(args, "validation_parquet", None),
    }
    if "test" in tuple(getattr(args, "eval_splits", ())):
        dataset_paths["test"] = getattr(args, "test_parquet", None)
    datasets: dict[str, dict[str, Any]] = {}
    for split, value in dataset_paths.items():
        if value is None:
            continue
        path = Path(value)
        if path.is_file():
            datasets[split] = file_provenance(path)
        else:
            datasets[split] = {
                "path": str(path),
                "size_bytes": None,
                "sha256": None,
            }
    provenance = {
        "model_fingerprint": args.av_model_fingerprint,
        "checkpoint": None if args.av_hf_checkpoint is None else str(args.av_hf_checkpoint),
        "model_revision": args.av_model_revision,
        "tokenizer_revision": args.av_tokenizer_revision,
        "tokenizer_fingerprint": args.av_tokenizer_fingerprint,
        "datasets": datasets,
    }
    provenance["dataset_bundle_sha256"] = hashlib.sha256(
        json.dumps(datasets, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return provenance


def generation_provenance_sha256(provenance: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_generated_record_provenance(
    records: list[dict[str, Any]],
    *,
    expected_provenance: dict[str, Any],
    require: bool,
) -> dict[str, Any] | None:
    if not records:
        if require:
            raise ValueError("missing generation provenance: no generated records")
        return None
    provenances: dict[str, dict[str, Any]] = {}
    missing_rows: list[int] = []
    invalid_rows: list[int] = []
    for record in records:
        row_index = int(record.get("row_index", -1))
        provenance = record.get("generation_provenance")
        recorded_hash = record.get("generation_provenance_sha256")
        if not isinstance(provenance, dict) or not isinstance(recorded_hash, str):
            missing_rows.append(row_index)
            continue
        actual_hash = generation_provenance_sha256(provenance)
        if recorded_hash != actual_hash:
            invalid_rows.append(row_index)
            continue
        provenances[actual_hash] = provenance
    if missing_rows or invalid_rows:
        if require:
            raise ValueError(
                "invalid generation provenance: "
                f"missing_rows={missing_rows[:10]} invalid_rows={invalid_rows[:10]}"
            )
        return None
    if len(provenances) != 1:
        raise ValueError(f"mixed generation provenances: hashes={sorted(provenances)}")
    resolved = next(iter(provenances.values()))
    if generation_provenance_sha256(resolved) != generation_provenance_sha256(
        expected_provenance
    ):
        raise ValueError("generated provenance does not match model or dataset identity")
    if require:
        datasets = resolved.get("datasets") or {}
        missing_hashes = sorted(
            split
            for split, metadata in datasets.items()
            if not isinstance(metadata, dict) or not metadata.get("sha256")
        )
        if missing_hashes:
            raise ValueError(
                f"generation provenance is missing dataset hashes: {missing_hashes}"
            )
    return resolved


def compare_generation_protocols(
    current: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    publication_errors: list[str] = []
    if any(
        protocol and str(protocol.get("prefix") or "")
        for protocol in (current, baseline)
    ):
        publication_errors.append("nonempty_generation_prefix")
    if not current or not baseline:
        return {
            "matched": False,
            "current_present": bool(current),
            "baseline_present": bool(baseline),
            "current_sha256": None if not current else generation_protocol_sha256(current),
            "baseline_sha256": None if not baseline else generation_protocol_sha256(baseline),
            "mismatched_fields": ["generation_protocol"],
            "publication_compatible": False,
            "publication_errors": publication_errors or ["missing_generation_protocol"],
        }
    fields = sorted(set(current) | set(baseline))
    mismatched = [field for field in fields if current.get(field) != baseline.get(field)]
    return {
        "matched": not mismatched,
        "current_present": True,
        "baseline_present": True,
        "current_sha256": generation_protocol_sha256(current),
        "baseline_sha256": generation_protocol_sha256(baseline),
        "mismatched_fields": mismatched,
        "publication_compatible": not publication_errors,
        "publication_errors": publication_errors,
    }


def validate_generated_record_protocols(
    records: list[dict[str, Any]],
    *,
    expected_protocol: dict[str, Any] | None,
    require: bool,
) -> dict[str, Any] | None:
    if not records:
        if require:
            raise ValueError("missing generation protocol: no generated records")
        return None
    protocols: dict[str, dict[str, Any]] = {}
    missing_rows: list[int] = []
    invalid_rows: list[int] = []
    for record in records:
        row_index = int(record.get("row_index", -1))
        protocol = record.get("generation_protocol")
        recorded_hash = record.get("generation_protocol_sha256")
        if not isinstance(protocol, dict) or not isinstance(recorded_hash, str):
            missing_rows.append(row_index)
            continue
        actual_hash = generation_protocol_sha256(protocol)
        if recorded_hash != actual_hash:
            invalid_rows.append(row_index)
            continue
        protocols[actual_hash] = protocol
    if missing_rows or invalid_rows:
        if require:
            details = []
            if missing_rows:
                details.append(f"missing generation protocol rows={missing_rows[:10]}")
            if invalid_rows:
                details.append(f"invalid generation protocol hash rows={invalid_rows[:10]}")
            raise ValueError("; ".join(details))
        return None
    if len(protocols) != 1:
        raise ValueError(f"mixed generation protocols: hashes={sorted(protocols)}")
    resolved = next(iter(protocols.values()))
    if expected_protocol is not None and generation_protocol_sha256(resolved) != generation_protocol_sha256(
        expected_protocol
    ):
        comparison = compare_generation_protocols(resolved, expected_protocol)
        raise ValueError(
            "generated protocol does not match requested protocol: "
            f"fields={comparison['mismatched_fields']}"
        )
    return resolved


def file_provenance(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> dict[str, Any]:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    stat = source.stat()
    return {"path": str(source), "size_bytes": stat.st_size, "sha256": digest.hexdigest()}


def add_bool_optional_arg(parser: argparse.ArgumentParser, name: str, *, default: bool) -> None:
    dest = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=dest, action="store_true")
    group.add_argument("--no-" + name.lstrip("-"), dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def extract_explanation(response: str) -> str | None:
    match = EXPLANATION_RE.search(str(response or ""))
    return match.group(1).strip() if match else None


def l2_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), eps)


def rowwise_normalized_mse(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    from nano_eval_core import activation_reconstruction_metrics

    targets_array = np.asarray(targets)
    metrics = activation_reconstruction_metrics(
        predictions,
        targets_array,
        train_mean=np.zeros(targets_array.shape[-1], dtype=np.float64),
    )
    return np.asarray(metrics["rowwise_directional_mse"], dtype=np.float64)


def metric_summary(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    train_targets_for_mean: np.ndarray,
    eps: float = 1e-12,
    include_rowwise: bool = False,
) -> dict[str, Any]:
    from nano_eval_core import activation_reconstruction_metrics

    pred = np.asarray(predictions, dtype=np.float64)
    gold = np.asarray(targets, dtype=np.float64)
    train_targets = np.asarray(train_targets_for_mean, dtype=np.float64)
    train_mean = train_targets.mean(axis=0)
    metrics = activation_reconstruction_metrics(
        pred,
        gold,
        train_mean=train_mean,
        eps=eps,
    )
    rowwise_directional = np.asarray(metrics.pop("rowwise_directional_mse"))
    rowwise_unit_feature = np.asarray(
        metrics.pop("rowwise_unit_vector_feature_mse")
    )
    rowwise_raw = np.asarray(metrics.pop("rowwise_raw_mse"))
    mean_prediction = np.repeat(train_mean[None, :], gold.shape[0], axis=0)
    mean_metrics = activation_reconstruction_metrics(
        mean_prediction,
        gold,
        train_mean=train_mean,
        eps=eps,
    )
    mean_directional_mse = float(mean_metrics["directional_mse"])
    directional_mse = float(metrics["directional_mse"])
    raw_sse = float(np.square(pred - gold).sum())
    centered_sst = float(np.square(gold - train_mean[None, :]).sum())
    summary: dict[str, Any] = {
        "row_count": int(gold.shape[0]),
        **metrics,
        "mean_control_directional_mse": mean_directional_mse,
        "mean_control_normalized_mse": mean_directional_mse,
        "fve_directional": (
            None
            if mean_directional_mse <= eps
            else float(1.0 - directional_mse / mean_directional_mse)
        ),
        "fve_nrm": (
            None
            if mean_directional_mse <= eps
            else float(1.0 - directional_mse / mean_directional_mse)
        ),
        "centered_raw_r2": metrics["centered_r2"],
        "centered_raw_sse": raw_sse,
        "centered_raw_sst": centered_sst,
    }
    if include_rowwise:
        summary["rowwise_directional_mse"] = rowwise_directional
        summary["rowwise_unit_vector_feature_mse"] = rowwise_unit_feature
        summary["rowwise_raw_mse"] = rowwise_raw
    return summary


def text_overlap_metrics(generated: str, target: str) -> dict[str, float | int]:
    generated_tokens = set(str(generated or "").lower().split())
    target_tokens = set(str(target or "").lower().split())
    if not generated_tokens or not target_tokens:
        return {"jaccard": 0.0, "generated_token_count": len(generated_tokens), "target_token_count": len(target_tokens)}
    return {
        "jaccard": len(generated_tokens & target_tokens) / len(generated_tokens | target_tokens),
        "generated_token_count": len(generated_tokens),
        "target_token_count": len(target_tokens),
    }


def _read_rows(path: Path, split: str, offset: int) -> list[dict[str, Any]]:
    from nano_av_warmstart_smoke import load_av_rows

    rows = load_av_rows(path)
    for i, row in enumerate(rows):
        row["row_index"] = offset + i
        row["source_row_index"] = i
        row["split"] = split
    return rows


def load_control_rows_by_index(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    selected_by_split: dict[str, list[int]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Load exact source controls separately from generation-provenance inputs."""

    from nano_av_warmstart_smoke import load_av_rows

    controls_by_index: dict[int, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for split_name in args.eval_splits:
        path = getattr(args, f"{split_name}_control_parquet", None)
        if path is None:
            for row_index in selected_by_split[split_name]:
                controls_by_index[row_index] = rows[row_index]
            continue
        source_rows = load_av_rows(path)
        split_rows = [row for row in rows if row.get("split") == split_name]
        if len(source_rows) != len(split_rows):
            raise ValueError(
                f"{split_name} control parquet row count differs from eval parquet: "
                f"control={len(source_rows)} eval={len(split_rows)}"
            )
        for row_index in selected_by_split[split_name]:
            eval_row = rows[row_index]
            source_index = int(eval_row["source_row_index"])
            control_row = source_rows[source_index]
            for field in (
                "doc_id",
                "n_raw_tokens",
                "activation_layer",
                "detokenized_text_truncated",
            ):
                if eval_row.get(field) != control_row.get(field):
                    raise ValueError(
                        f"{split_name} control identity mismatch at row {row_index}: "
                        f"field={field}"
                    )
            if not np.array_equal(
                np.asarray(eval_row["activation_vector"], dtype=np.float32),
                np.asarray(control_row["activation_vector"], dtype=np.float32),
            ):
                raise ValueError(
                    f"{split_name} control activation mismatch at row {row_index}"
                )
            controls_by_index[row_index] = control_row
        provenance[split_name] = file_provenance(path)
    return controls_by_index, provenance


def load_eval_rows(
    train_parquet: Path,
    validation_parquet: Path,
    test_parquet: Path | None,
    *,
    eval_splits: list[str] | tuple[str, ...] = ("validation",),
    content_family_manifest: Path | None = None,
    content_family_coverage: Path | None = None,
    require_family_disjoint_splits: bool = False,
) -> tuple[list[dict[str, Any]], list[int], list[int], list[int]]:
    train = _read_rows(train_parquet, "train", 0)
    requested = tuple(str(split) for split in eval_splits)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("eval_splits must be non-empty and unique")
    unknown = sorted(set(requested) - {"validation", "test"})
    if unknown:
        raise ValueError(f"unsupported eval splits: {unknown}")
    validation = (
        _read_rows(validation_parquet, "validation", len(train))
        if "validation" in requested
        else []
    )
    if "test" in requested and test_parquet is None:
        raise ValueError("test_parquet is required when evaluating test")
    test = (
        _read_rows(test_parquet, "test", len(train) + len(validation))
        if "test" in requested and test_parquet is not None
        else []
    )
    rows = train + validation + test
    if content_family_manifest is not None:
        from nano_functional_eval_data import (
            attach_content_family_ids,
            load_content_family_manifest,
        )

        manifest = load_content_family_manifest(content_family_manifest)
        attach_content_family_ids(
            rows,
            manifest,
            require_disjoint_splits=False,
        )
    if content_family_coverage is not None:
        from nano_functional_eval_data import (
            apply_family_holdout_coverage,
            load_content_family_coverage,
        )

        coverage = load_content_family_coverage(content_family_coverage)
        coverage_report = apply_family_holdout_coverage(
            rows,
            coverage,
            required_splits=requested,
        )
        if require_family_disjoint_splits and not coverage_report["passed"]:
            raise ValueError(
                "publication holdout coverage is not disjoint and complete: "
                f"{coverage_report}"
            )
    elif require_family_disjoint_splits:
        raise ValueError("family-disjoint evaluation requires content-family coverage")
    require_holdout_eligible = content_family_coverage is not None
    return (
        rows,
        _split_indices(rows, "train"),
        _split_indices(
            rows,
            "validation",
            require_holdout_eligible=require_holdout_eligible,
        ),
        _split_indices(
            rows,
            "test",
            require_holdout_eligible=require_holdout_eligible,
        ),
    )


def _split_indices(
    rows: list[dict[str, Any]],
    split: str,
    *,
    require_holdout_eligible: bool = False,
) -> list[int]:
    return [
        int(row["row_index"])
        for row in rows
        if row.get("split") == split
        and (
            not require_holdout_eligible
            or bool(row.get("publication_holdout_eligible"))
        )
    ]


def sample_indices(indices: list[int], limit: int) -> list[int]:
    if limit <= 0 or limit >= len(indices):
        return list(indices)
    return list(indices[:limit])


def sample_eval_indices(
    rows: list[dict[str, Any]],
    indices: list[int],
    limit: int,
    *,
    split: str,
    strategy: str,
    seed: int,
) -> list[int]:
    if strategy == "row_order":
        return sample_indices(indices, limit)
    if strategy != "family_stratified":
        raise ValueError(f"unknown eval selection strategy: {strategy}")
    from nano_functional_eval_data import select_family_stratified_rows

    selected = select_family_stratified_rows(
        [rows[index] for index in indices],
        split=split,
        limit=min(limit, len(indices)) if limit > 0 else len(indices),
        seed=seed,
    )
    return [int(row["row_index"]) for row in selected]


def select_eval_indices_by_split(
    rows: list[dict[str, Any]],
    *,
    validation_indices: list[int],
    test_indices: list[int],
    validation_limit: int,
    test_limit: int,
    eval_splits: list[str] | tuple[str, ...],
    strategy: str,
    seed: int,
) -> dict[str, list[int]]:
    split_sources = {
        "validation": (validation_indices, validation_limit),
        "test": (test_indices, test_limit),
    }
    selected: dict[str, list[int]] = {}
    for split in eval_splits:
        if split not in split_sources:
            raise ValueError(f"unsupported eval split: {split}")
        indices, limit = split_sources[split]
        selected[split] = sample_eval_indices(
            rows,
            indices,
            limit,
            split=split,
            strategy=strategy,
            seed=seed,
        )
    return selected


def target_explanation(row: dict[str, Any]) -> str:
    response = str(row.get("response") or "")
    return (extract_explanation(response) or response).strip()


def generated_record_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "row_index": int(row["row_index"]),
        "source_row_index": int(row.get("source_row_index", row["row_index"])),
        "split": str(row.get("split")),
    }
    for key in (
        "doc_id",
        "content_family_id",
        "n_raw_tokens",
        "token_position",
        "token_id",
        "sample_uuid",
    ):
        if row.get(key) is not None:
            metadata[key] = row[key]
    return metadata


def parse_generated_explanation(generated: str, *, fallback: str = "empty") -> dict[str, Any]:
    if fallback not in GENERATED_TEXT_FALLBACKS:
        raise ValueError(f"unknown generated text fallback {fallback!r}; choices={GENERATED_TEXT_FALLBACKS}")
    text = str(generated or "")
    extracted = extract_explanation(text)
    if extracted is None:
        open_match = EXPLANATION_OPEN_RE.search(text)
        if open_match:
            extracted = open_match.group(1).strip()
            mode = "open_tag"
        elif fallback == "raw":
            extracted = text.strip()
            mode = "raw"
        else:
            extracted = ""
            mode = "empty"
    else:
        extracted = extracted.strip()
        mode = "closed_tag"
    tokens = [token.lower() for token in QUALITY_TOKEN_RE.findall(extracted)]
    repetition_loop = has_repetition_loop(tokens)
    factual_number_count = sum(1 for token in tokens if NUMBER_TOKEN_RE.fullmatch(token))
    content_usable = bool(extracted.strip())
    fallback_only = content_usable and mode != "closed_tag"
    usable = content_usable and not fallback_only and not repetition_loop
    return {
        "explanation": extracted,
        "closed": mode == "closed_tag",
        "empty": not content_usable,
        "content_usable": content_usable,
        "fallback_only": fallback_only,
        "repetition_loop": repetition_loop,
        "usable": usable,
        "token_count": len(tokens),
        "factual_number_count": factual_number_count,
        "factual_number_density": (
            float(factual_number_count / len(tokens)) if tokens else 0.0
        ),
        "extraction_mode": mode,
    }


def has_repetition_loop(tokens: list[str], *, min_repeats: int = 3) -> bool:
    """Detect a consecutively repeated phrase covering at least six tokens."""

    if min_repeats < 2:
        raise ValueError("min_repeats must be at least 2")
    token_count = len(tokens)
    for width in range(1, token_count // min_repeats + 1):
        repeated_token_count = width * min_repeats
        if repeated_token_count < 6:
            continue
        for start in range(0, token_count - repeated_token_count + 1):
            phrase = tokens[start : start + width]
            if all(
                tokens[start + repeat * width : start + (repeat + 1) * width]
                == phrase
                for repeat in range(1, min_repeats)
            ):
                return True
    return False


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = _average_ranks(left) if rank else left.astype(np.float64, copy=False)
    y = _average_ranks(right) if rank else right.astype(np.float64, copy=False)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _relative_improvement(baseline: np.ndarray, candidate: np.ndarray) -> float | None:
    baseline_mean = float(np.mean(baseline))
    if abs(baseline_mean) <= 1e-18:
        return None
    return float((baseline_mean - float(np.mean(candidate))) / baseline_mean)


def summarize_length_control_analysis(
    *,
    candidate_token_counts: list[int],
    sft_token_counts: list[int],
    teacher_token_counts: list[int],
    candidate_losses: list[float],
    baseline_losses: list[float],
    sft_length_matched_losses: list[float],
    teacher_length_matched_losses: list[float],
) -> dict[str, Any]:
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (
            candidate_token_counts,
            sft_token_counts,
            teacher_token_counts,
            candidate_losses,
            baseline_losses,
            sft_length_matched_losses,
            teacher_length_matched_losses,
        )
    ]
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1 or not arrays or len(arrays[0]) == 0:
        raise ValueError("length-control inputs must be non-empty and row-aligned")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("length-control inputs must be finite")
    (
        candidate_tokens,
        sft_tokens,
        teacher_tokens,
        candidate,
        baseline,
        sft_matched,
        teacher_matched,
    ) = arrays
    if np.any(candidate_tokens <= 0) or np.any(sft_tokens < 0) or np.any(teacher_tokens < 0):
        raise ValueError("token counts must be non-negative and candidate counts positive")

    gain = baseline - candidate
    sft_length_delta = candidate_tokens - sft_tokens
    teacher_length_delta = candidate_tokens - teacher_tokens
    sft_relative = _relative_improvement(baseline, sft_matched)
    teacher_relative = _relative_improvement(baseline, teacher_matched)
    available_length_gains = [
        value for value in (sft_relative, teacher_relative) if value is not None
    ]
    return {
        "schema_version": LENGTH_CONTROL_SCHEMA_VERSION,
        "row_count": int(len(candidate)),
        "candidate_token_count_mean": float(np.mean(candidate_tokens)),
        "sft_token_count_mean": float(np.mean(sft_tokens)),
        "teacher_token_count_mean": float(np.mean(teacher_tokens)),
        "candidate_minus_sft_tokens_mean": float(np.mean(sft_length_delta)),
        "candidate_minus_teacher_tokens_mean": float(np.mean(teacher_length_delta)),
        "relative_improvement": _relative_improvement(baseline, candidate),
        "sft_length_matched_relative_improvement": sft_relative,
        "teacher_length_matched_relative_improvement": teacher_relative,
        "best_length_matched_relative_improvement": (
            max(available_length_gains) if available_length_gains else None
        ),
        "gain_per_generated_token_mean": float(np.mean(gain / candidate_tokens)),
        "sft_length_delta_vs_gain_pearson": _correlation(sft_length_delta, gain),
        "sft_length_delta_vs_gain_spearman": _correlation(
            sft_length_delta,
            gain,
            rank=True,
        ),
        "teacher_length_delta_vs_gain_pearson": _correlation(
            teacher_length_delta,
            gain,
        ),
        "teacher_length_delta_vs_gain_spearman": _correlation(
            teacher_length_delta,
            gain,
            rank=True,
        ),
    }


def format_critic_prompt(template: str, explanation: str) -> str:
    return template.format(explanation=explanation)


def teacher_prompt_for_row(row: dict[str, Any], critic_template: str) -> str:
    return format_critic_prompt(critic_template, target_explanation(row))


def _tokenize_explanation(tokenizer: Any, explanation: str) -> list[Any]:
    encoded = tokenizer(explanation, add_special_tokens=False)
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if token_ids is None and hasattr(encoded, "input_ids"):
        token_ids = encoded.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if (
        isinstance(token_ids, list)
        and len(token_ids) == 1
        and isinstance(token_ids[0], (list, tuple))
    ):
        token_ids = list(token_ids[0])
    if not isinstance(token_ids, list):
        raise ValueError("tokenizer must return a list of input_ids")
    if any(isinstance(token_id, (list, tuple)) for token_id in token_ids):
        raise ValueError("tokenizer returned batched input_ids for one explanation")
    return token_ids


def _decode_explanation_tokens(tokenizer: Any, token_ids: list[Any]) -> str:
    return str(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ).strip()


def build_length_matched_explanations(
    tokenizer: Any,
    *,
    candidate_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    rows_by_index: dict[int, dict[str, Any]],
    fallback: str,
) -> dict[str, Any]:
    baseline_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in baseline_records:
        key = (str(record.get("split") or ""), int(record.get("row_index", -1)))
        if key in baseline_by_key:
            raise ValueError(f"duplicate length-baseline row: {key}")
        baseline_by_key[key] = record

    candidate_token_counts: list[int] = []
    sft_token_counts: list[int] = []
    teacher_token_counts: list[int] = []
    sft_matched: list[str] = []
    teacher_matched: list[str] = []
    for candidate in candidate_records:
        row_index = int(candidate.get("row_index", -1))
        key = (str(candidate.get("split") or ""), row_index)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            raise ValueError(f"missing length-baseline row: {key}")
        for identity_field in ("doc_id", "content_family_id"):
            candidate_value = str(candidate.get(identity_field) or "")
            baseline_value = str(baseline.get(identity_field) or "")
            if candidate_value != baseline_value:
                raise ValueError(
                    "length-baseline identity mismatch: "
                    f"row={key} field={identity_field} "
                    f"candidate={candidate_value!r} baseline={baseline_value!r}"
                )
        row = rows_by_index.get(row_index)
        if row is None:
            raise ValueError(f"missing source row for length control: {row_index}")
        candidate_generated = str(
            ((candidate.get("controls") or {}).get("real") or {}).get("generated")
            or ""
        )
        baseline_generated = str(
            ((baseline.get("controls") or {}).get("real") or {}).get("generated")
            or ""
        )
        candidate_explanation = parse_generated_explanation(
            candidate_generated,
            fallback=fallback,
        )["explanation"]
        baseline_explanation = parse_generated_explanation(
            baseline_generated,
            fallback=fallback,
        )["explanation"]
        teacher_explanation = target_explanation(row)
        candidate_ids = _tokenize_explanation(tokenizer, candidate_explanation)
        baseline_ids = _tokenize_explanation(tokenizer, baseline_explanation)
        teacher_ids = _tokenize_explanation(tokenizer, teacher_explanation)
        candidate_token_counts.append(len(candidate_ids))
        sft_token_counts.append(len(baseline_ids))
        teacher_token_counts.append(len(teacher_ids))
        sft_matched.append(
            _decode_explanation_tokens(tokenizer, candidate_ids[: len(baseline_ids)])
        )
        teacher_matched.append(
            _decode_explanation_tokens(tokenizer, candidate_ids[: len(teacher_ids)])
        )
    return {
        "candidate_token_counts": candidate_token_counts,
        "sft_token_counts": sft_token_counts,
        "teacher_token_counts": teacher_token_counts,
        "sft_length_matched_explanations": sft_matched,
        "teacher_length_matched_explanations": teacher_matched,
    }


def default_generated_jsonl(report_json: Path) -> Path:
    return report_json.with_suffix("").with_name(report_json.with_suffix("").name + "_generated.jsonl")


def _vectors_for_rows(rows: list[dict[str, Any]]) -> "Any":
    import torch

    return torch.tensor([row["activation_vector"] for row in rows], dtype=torch.float32)


def _collect_and_empty_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def collect_ar_device_profile(model: Any) -> dict[str, Any]:
    """Summarize AR placement and allocator state without changing scoring."""

    import torch

    parameter_tensors = Counter()
    parameter_numel = Counter()
    for parameter in model.parameters():
        device = str(parameter.device)
        parameter_tensors[device] += 1
        parameter_numel[device] += int(parameter.numel())
    hf_device_map = getattr(model, "hf_device_map", None)
    mapped_devices = Counter()
    if isinstance(hf_device_map, Mapping):
        mapped_devices.update(str(device) for device in hf_device_map.values())
    cuda_memory: dict[str, dict[str, int]] = {}
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            cuda_memory[str(device_index)] = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
                "max_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device_index)
                ),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
            }
    return {
        "parameter_tensor_counts_by_device": dict(sorted(parameter_tensors.items())),
        "parameter_numel_by_device": dict(sorted(parameter_numel.items())),
        "hf_device_map_entry_count": (
            len(hf_device_map) if isinstance(hf_device_map, Mapping) else 0
        ),
        "hf_device_map_entries_by_device": dict(sorted(mapped_devices.items())),
        "cuda_memory_by_device": cuda_memory,
    }


def _av_model_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        model_id=str(args.av_hf_checkpoint),
        model_revision=args.av_model_revision,
        tokenizer_revision=args.av_tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        attn_implementation=args.av_attn_implementation,
        load_mode=args.av_load_mode,
        device_map=args.av_device_map,
        low_cpu_mem_usage=args.av_low_cpu_mem_usage,
        torch_dtype=args.torch_dtype,
    )


def generate_roundtrip_records(args: argparse.Namespace, *, stream_jsonl: Path | None = None) -> list[dict[str, Any]]:
    import torch
    from eval_nano_av_miles_checkpoint import (
        _check_hf_checkpoint,
        build_checkpoint_eval_control_vectors,
    )
    from nano_eval_core import shuffled_control_candidates
    from nano_av_warmstart_smoke import generate_controls_for_row, load_av_config, resolve_injection_scale
    from nano_introspection import load_model_from_args, load_tokenizer_from_args

    hf_checkpoint = Path(args.av_hf_checkpoint)
    if hf_checkpoint.exists():
        _check_hf_checkpoint(hf_checkpoint)
    controls_requested = tuple(args.generation_controls)
    generation_protocol = build_generation_protocol(args)
    generation_protocol_hash = generation_protocol_sha256(generation_protocol)
    generation_provenance = build_generation_provenance(args)
    generation_provenance_hash = generation_provenance_sha256(
        generation_provenance
    )
    unknown_controls = sorted(set(controls_requested) - set(CONTROL_NAMES))
    if unknown_controls:
        raise ValueError(f"unknown generation controls: {unknown_controls}; choices={CONTROL_NAMES}")

    rows, train_indices, validation_indices, test_indices = load_eval_rows(
        args.train_parquet,
        args.validation_parquet,
        args.test_parquet,
        eval_splits=args.eval_splits,
        content_family_manifest=args.content_family_manifest,
        content_family_coverage=args.content_family_coverage,
        require_family_disjoint_splits=args.require_family_level_inference,
    )
    selected_by_split = select_eval_indices_by_split(
        rows,
        validation_indices=validation_indices,
        test_indices=test_indices,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        eval_splits=args.eval_splits,
        strategy=args.selection_strategy,
        seed=args.selection_seed,
    )
    eval_indices = [
        row_index
        for split in args.eval_splits
        for row_index in selected_by_split[split]
    ]
    if args.generation_shard_count > 1:
        eval_indices = shard_eval_indices(
            eval_indices,
            shard_index=args.generation_shard_index,
            shard_count=args.generation_shard_count,
        )
    expected_by_split = {
        split: [
            row_index
            for row_index in eval_indices
            if rows[row_index].get("split") == split
        ]
        for split in args.eval_splits
    }

    records: list[dict[str, Any]] = []
    reusable_keys: set[tuple[str, int]] = set()
    if stream_jsonl is not None:
        stream_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if args.resume_generated and stream_jsonl.exists():
            cached_records = read_generated_jsonl(stream_jsonl)
            validate_generated_record_protocols(
                cached_records,
                expected_protocol=generation_protocol,
                require=True,
            )
            validate_generated_record_provenance(
                cached_records,
                expected_provenance=generation_provenance,
                require=True,
            )
            records.extend(
                select_reusable_generated_records(
                    cached_records,
                    expected_by_split=expected_by_split,
                    controls_requested=controls_requested,
                )
            )
            reusable_keys = {_generated_record_pair(record) for record in records}
        stream_jsonl.write_text("")
        for record in records:
            with stream_jsonl.open("a") as f:
                f.write(json.dumps(json_safe(record), sort_keys=True) + "\n")
    total = len(eval_indices)
    if len(reusable_keys) == total:
        return records

    av_args = _av_model_args(args)
    tokenizer = load_tokenizer_from_args(av_args)
    model = load_model_from_args(av_args)
    model.eval()
    cfg = load_av_config(args.validation_parquet, tokenizer)
    injection_scale = resolve_injection_scale(args.injection_scale, cfg.d_model)
    vectors = _vectors_for_rows(rows)
    if not train_indices:
        raise ValueError("training rows are required to construct the mean control")
    mean_vector = vectors[train_indices].mean(dim=0)

    for ordinal, row_index in enumerate(eval_indices, start=1):
        row = rows[row_index]
        if (str(row.get("split")), int(row_index)) in reusable_keys:
            continue
        if args.progress_every and (ordinal == 1 or ordinal % args.progress_every == 0 or ordinal == total):
            print(f"[roundtrip] generating row {ordinal}/{total} row_index={row_index}", flush=True)
        controls = build_checkpoint_eval_control_vectors(
            vectors,
            row_index=row_index,
            mean_vector=mean_vector,
            seed=args.seed,
            shuffle_candidate_indices=shuffled_control_candidates(
                rows,
                row_index=row_index,
            ),
        )
        item = {
            "schema_version": ROUNDTRIP_SCHEMA_VERSION,
            **generated_record_metadata(row),
            "generation_protocol": generation_protocol,
            "generation_protocol_sha256": generation_protocol_hash,
            "generation_provenance": generation_provenance,
            "generation_provenance_sha256": generation_provenance_hash,
            "target_explanation": target_explanation(row),
            "controls": {},
        }
        generated_by_control = generate_controls_for_row(
            model,
            tokenizer,
            cfg,
            row,
            {name: controls[name] for name in controls_requested},
            list(controls_requested),
            injection_scale=injection_scale,
            max_new_tokens=args.max_new_tokens,
            generation_prefix=args.generation_prefix,
            stop_text=args.stop_text,
            use_cache=args.generation_backend == "cache",
            batch_full_prefix=args.generation_backend == "legacy_batch",
        )
        for name in controls_requested:
            generated = generated_by_control[name]
            item["controls"][name] = {
                "generated": generated,
                "parsed": parse_generated_explanation(generated, fallback=args.generated_text_fallback),
                "text_overlap": text_overlap_metrics(generated, item["target_explanation"]),
            }
        records.append(item)
        if stream_jsonl is not None:
            with stream_jsonl.open("a") as f:
                f.write(json.dumps(json_safe(item), sort_keys=True) + "\n")

    del model, tokenizer, vectors, cfg, mean_vector
    _collect_and_empty_cuda_cache()
    return records


def _optional_path_arg(command: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _optional_str_arg(command: list[str], flag: str, value: str | None) -> None:
    if value not in {None, ""}:
        command.extend([flag, str(value)])


def build_generation_worker_command(
    args: argparse.Namespace,
    *,
    shard_index: int,
    shard_count: int,
    shard_jsonl: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--generation-only",
        "--generation-shard-index",
        str(shard_index),
        "--generation-shard-count",
        str(shard_count),
        "--generation-workers",
        "1",
        "--generation-parent-worker-count",
        str(args.generation_workers),
        "--av-hf-checkpoint",
        str(args.av_hf_checkpoint),
        "--ar-checkpoint-dir",
        str(args.ar_checkpoint_dir),
        "--train-parquet",
        str(args.train_parquet),
        "--validation-parquet",
        str(args.validation_parquet),
        "--report-json",
        str(args.report_json),
        "--generated-jsonl",
        str(shard_jsonl),
        "--validation-limit",
        str(args.validation_limit),
        "--test-limit",
        str(args.test_limit),
        "--eval-splits",
        *[str(split) for split in args.eval_splits],
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--generation-backend",
        str(args.generation_backend),
        "--generated-text-fallback",
        str(args.generated_text_fallback),
        "--seed",
        str(args.seed),
        "--injection-scale",
        str(args.injection_scale),
        "--torch-dtype",
        str(args.torch_dtype),
        "--av-device-map",
        str(args.av_device_map),
        "--av-low-cpu-mem-usage"
        if args.av_low_cpu_mem_usage
        else "--no-av-low-cpu-mem-usage",
        "--ar-device-map",
        str(args.ar_device_map),
        "--ar-batch-size",
        str(args.ar_batch_size),
        "--control-margin",
        str(args.control_margin),
        "--baseline-margin",
        str(args.baseline_margin),
        "--progress-every",
        str(args.progress_every),
        "--selection-strategy",
        str(args.selection_strategy),
        "--selection-seed",
        str(args.selection_seed),
        "--min-closed-fraction",
        str(args.min_closed_fraction),
        "--min-usable-fraction",
        str(args.min_usable_fraction),
    ]
    if args.resume_generated:
        command.append("--resume-generated")
    command.append("--generation-controls")
    command.extend([str(control) for control in args.generation_controls])
    _optional_str_arg(command, "--generation-prefix", args.generation_prefix)
    _optional_str_arg(command, "--stop-text", args.stop_text)
    _optional_path_arg(command, "--critic-template-source", args.critic_template_source)
    if "test" in args.eval_splits:
        _optional_path_arg(command, "--test-parquet", args.test_parquet)
    _optional_path_arg(command, "--baseline-report-json", args.baseline_report_json)
    _optional_path_arg(command, "--content-family-manifest", args.content_family_manifest)
    _optional_path_arg(command, "--content-family-coverage", args.content_family_coverage)
    _optional_str_arg(command, "--critic-template", args.critic_template)
    _optional_str_arg(command, "--av-model-revision", args.av_model_revision)
    _optional_str_arg(command, "--av-tokenizer-revision", args.av_tokenizer_revision)
    _optional_str_arg(command, "--av-attn-implementation", args.av_attn_implementation)
    _optional_str_arg(command, "--av-model-fingerprint", args.av_model_fingerprint)
    _optional_str_arg(command, "--av-tokenizer-fingerprint", args.av_tokenizer_fingerprint)
    if args.ar_max_length is not None:
        command.extend(["--ar-max-length", str(args.ar_max_length)])
    command.extend(["--av-load-mode", str(args.av_load_mode)])
    if args.local_files_only:
        command.append("--local-files-only")
    command.append("--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code")
    if args.require_generation_protocol_match:
        command.append("--require-generation-protocol-match")
    if args.require_family_level_inference:
        command.append("--require-family-level-inference")
    command.append("--stream-generated")
    return command


def _worker_env(args: argparse.Namespace, worker_index: int) -> dict[str, str]:
    env = os.environ.copy()
    devices = list(args.generation_worker_devices or [])
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = str(devices[worker_index % len(devices)])
    return env


def generate_roundtrip_records_with_workers(args: argparse.Namespace, generated_jsonl: Path) -> list[dict[str, Any]]:
    worker_count = int(args.generation_workers)
    if worker_count <= 1:
        return generate_roundtrip_records(args, stream_jsonl=generated_jsonl if args.stream_generated else None)
    if args.resume_generated and generated_jsonl.exists():
        resume_args = argparse.Namespace(**vars(args))
        resume_args.generation_workers = 1
        resume_args.generation_parent_worker_count = worker_count
        return generate_roundtrip_records(resume_args, stream_jsonl=generated_jsonl)
    generated_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stem = generated_jsonl.with_suffix("")
    shard_paths = [
        stem.with_name(f"{stem.name}_worker{index:02d}of{worker_count:02d}.jsonl")
        for index in range(worker_count)
    ]
    if not args.resume_generated:
        for shard_path in shard_paths:
            shard_path.unlink(missing_ok=True)
    requested_parallelism = getattr(args, "generation_max_parallel_workers", None)
    max_parallel_workers = (
        worker_count if requested_parallelism is None else int(requested_parallelism)
    )
    if max_parallel_workers <= 0 or max_parallel_workers > worker_count:
        raise ValueError(
            "generation_max_parallel_workers must be in [1, generation_workers]"
        )
    failures: list[tuple[int, int]] = []
    for group_start in range(0, worker_count, max_parallel_workers):
        group_stop = min(group_start + max_parallel_workers, worker_count)
        print(
            "[roundtrip] launching generation shards "
            f"{group_start + 1}-{group_stop}/{worker_count} "
            f"(max_parallel={max_parallel_workers})",
            flush=True,
        )
        processes: list[tuple[int, subprocess.Popen[Any]]] = []
        for index in range(group_start, group_stop):
            command = build_generation_worker_command(
                args,
                shard_index=index,
                shard_count=worker_count,
                shard_jsonl=shard_paths[index],
            )
            processes.append((index, subprocess.Popen(command, env=_worker_env(args, index))))
        for index, process in processes:
            returncode = process.wait()
            if returncode != 0:
                failures.append((index, returncode))
        if failures:
            break
    if failures:
        raise RuntimeError(f"generation workers failed: {failures}")
    return merge_generated_shards(shard_paths, generated_jsonl)


def write_generated_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(json_safe(record), sort_keys=True) + "\n")


def read_generated_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def shard_eval_indices(indices: list[int], *, shard_index: int, shard_count: int) -> list[int]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [int(value) for position, value in enumerate(indices) if position % shard_count == shard_index]


def _generated_record_order_key(record: dict[str, Any]) -> tuple[int, int]:
    split_order = {"validation": 0, "test": 1}
    return (split_order.get(str(record.get("split")), 99), int(record.get("row_index", 0)))


def merge_generated_shards(shard_paths: list[Path], output_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        if shard_path.exists():
            records.extend(read_generated_jsonl(shard_path))
    records.sort(key=_generated_record_order_key)
    write_generated_jsonl(output_path, records)
    return records


def _ordered_expected_pairs(expected_by_split: dict[str, list[int]]) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    seen_splits: set[str] = set()
    for split in ("validation", "test"):
        seen_splits.add(split)
        pairs.extend((split, int(row_index)) for row_index in expected_by_split.get(split, []))
    for split in sorted(set(expected_by_split) - seen_splits):
        pairs.extend((str(split), int(row_index)) for row_index in expected_by_split.get(split, []))
    return pairs


def _generated_record_pair(record: dict[str, Any]) -> tuple[str, int]:
    return (str(record.get("split")), int(record.get("row_index", -1)))


def _has_requested_controls(record: dict[str, Any], controls_requested: list[str] | tuple[str, ...]) -> bool:
    control_map = record.get("controls") or {}
    return all(str(control) in control_map for control in controls_requested)


def select_reusable_generated_records(
    records: list[dict[str, Any]],
    *,
    expected_by_split: dict[str, list[int]],
    controls_requested: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    ordered_pairs = _ordered_expected_pairs(expected_by_split)
    expected_pairs = set(ordered_pairs)
    reusable_by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        pair = _generated_record_pair(record)
        if pair in expected_pairs and _has_requested_controls(record, controls_requested):
            reusable_by_pair[pair] = record
    return [reusable_by_pair[pair] for pair in ordered_pairs if pair in reusable_by_pair]


def validate_generated_record_coverage(
    records: list[dict[str, Any]],
    *,
    expected_by_split: dict[str, list[int]],
    controls_requested: list[str] | tuple[str, ...],
) -> None:
    expected_pairs = set(_ordered_expected_pairs(expected_by_split))
    seen_pairs: set[tuple[str, int]] = set()
    duplicate_pairs: list[tuple[str, int]] = []
    extra_pairs: list[tuple[str, int]] = []
    missing_controls: list[dict[str, Any]] = []
    controls = [str(control) for control in controls_requested]

    for record in records:
        pair = _generated_record_pair(record)
        if pair in seen_pairs:
            duplicate_pairs.append(pair)
        seen_pairs.add(pair)
        if pair not in expected_pairs:
            extra_pairs.append(pair)
        control_map = record.get("controls") or {}
        absent = [control for control in controls if control not in control_map]
        if absent:
            missing_controls.append({"split": pair[0], "row_index": pair[1], "missing": absent})

    missing_pairs = sorted(expected_pairs - seen_pairs, key=lambda item: (item[0], item[1]))
    if missing_pairs:
        raise ValueError(f"missing generated rows: {missing_pairs[:10]}")
    if extra_pairs:
        raise ValueError(f"unexpected generated rows: {extra_pairs[:10]}")
    if duplicate_pairs:
        raise ValueError(f"duplicate generated rows: {duplicate_pairs[:10]}")
    if missing_controls:
        raise ValueError(f"missing generated controls: {missing_controls[:10]}")


def pairwise_win_summary(
    candidate_losses: np.ndarray,
    baseline_losses: np.ndarray,
    *,
    candidate_name: str,
    baseline_name: str,
    eps: float = 1e-12,
) -> dict[str, float | int | str]:
    candidate = np.asarray(candidate_losses, dtype=np.float64)
    baseline = np.asarray(baseline_losses, dtype=np.float64)
    if candidate.shape != baseline.shape:
        raise ValueError(f"rowwise losses must have same shape, got {candidate.shape} vs {baseline.shape}")
    candidate_better = candidate < (baseline - eps)
    baseline_better = baseline < (candidate - eps)
    ties = ~(candidate_better | baseline_better)
    row_count = int(candidate.shape[0])
    return {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "row_count": row_count,
        "candidate_better_count": int(np.count_nonzero(candidate_better)),
        "candidate_better_fraction": float(np.mean(candidate_better)) if row_count else 0.0,
        "tie_count": int(np.count_nonzero(ties)),
        "tie_fraction": float(np.mean(ties)) if row_count else 0.0,
        "baseline_better_count": int(np.count_nonzero(baseline_better)),
        "baseline_better_fraction": float(np.mean(baseline_better)) if row_count else 0.0,
        "mean_loss_delta_baseline_minus_candidate": float(np.mean(baseline - candidate)) if row_count else 0.0,
    }


def paired_improvement_summary(
    candidate_losses: np.ndarray,
    baseline_losses: np.ndarray,
    *,
    doc_ids: list[str] | None = None,
    content_family_ids: list[str] | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    permutation_samples: int = 100_000,
    permutation_seed: int = 0,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_losses, dtype=np.float64)
    baseline = np.asarray(baseline_losses, dtype=np.float64)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("paired improvement requires same-shape one-dimensional losses")
    if candidate.size == 0:
        raise ValueError("paired improvement requires at least one row")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if permutation_samples <= 0:
        raise ValueError("permutation_samples must be positive")
    delta = baseline - candidate
    groups: list[np.ndarray]
    unit = "row"
    group_ids = content_family_ids if content_family_ids is not None else doc_ids
    if group_ids is not None:
        if len(group_ids) != len(delta):
            raise ValueError("group IDs must align with paired losses")
        grouped: dict[str, list[int]] = {}
        for index, group_id in enumerate(group_ids):
            value = str(group_id)
            if not value:
                raise ValueError("group IDs must be non-empty")
            grouped.setdefault(value, []).append(index)
        groups = [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]
        unit = "content_family_id" if content_family_ids is not None else "doc_id"
    else:
        groups = [np.asarray([index], dtype=np.int64) for index in range(len(delta))]

    unit_effects = np.asarray(
        [float(delta[group].mean()) for group in groups],
        dtype=np.float64,
    )
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    bootstrap_chunk_size = max(
        1,
        min(65_536, 4_000_000 // len(unit_effects)),
    )
    for start in range(0, bootstrap_samples, bootstrap_chunk_size):
        stop = min(start + bootstrap_chunk_size, bootstrap_samples)
        selected = rng.integers(
            0,
            len(unit_effects),
            size=(stop - start, len(unit_effects)),
        )
        bootstrap_means[start:stop] = unit_effects[selected].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    observed_unit_mean = float(unit_effects.mean())
    tolerance = 1e-15
    if len(unit_effects) <= 20:
        total_permutations = 1 << len(unit_effects)
        extreme_count = 0
        bit_positions = np.arange(len(unit_effects), dtype=np.uint64)
        for start in range(0, total_permutations, 65_536):
            stop = min(start + 65_536, total_permutations)
            masks = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = (((masks >> bit_positions) & 1).astype(np.float64) * 2.0) - 1.0
            permuted_means = (signs * unit_effects[None, :]).mean(axis=1)
            extreme_count += int(np.count_nonzero(permuted_means >= observed_unit_mean - tolerance))
        sign_flip_p_value = extreme_count / total_permutations
        sign_flip_method = "exact"
        sign_flip_samples = total_permutations
    else:
        permutation_rng = np.random.default_rng(permutation_seed)
        extreme_count = 0
        completed = 0
        while completed < permutation_samples:
            chunk_size = min(65_536, permutation_samples - completed)
            signs = permutation_rng.integers(
                0,
                2,
                size=(chunk_size, len(unit_effects)),
                dtype=np.int8,
            ).astype(np.float64)
            signs = signs * 2.0 - 1.0
            permuted_means = (signs * unit_effects[None, :]).mean(axis=1)
            extreme_count += int(np.count_nonzero(permuted_means >= observed_unit_mean - tolerance))
            completed += chunk_size
        sign_flip_p_value = (extreme_count + 1) / (permutation_samples + 1)
        sign_flip_method = "monte_carlo"
        sign_flip_samples = permutation_samples
    mean_delta = float(delta.mean())
    baseline_mean = float(baseline.mean())
    positive = np.sort(delta[delta > 0.0])[::-1]
    top_count = min(5, len(positive))
    top_share = None
    if top_count and abs(float(delta.sum())) > 1e-18:
        top_share = float(positive[:top_count].sum() / delta.sum())
    return {
        "row_count": int(len(delta)),
        "independent_unit": unit,
        "independent_unit_count": len(groups),
        "independent_unit_mean_delta": observed_unit_mean,
        "mean_delta_baseline_minus_candidate": mean_delta,
        "median_delta_baseline_minus_candidate": float(np.median(delta)),
        "relative_improvement": None if baseline_mean == 0.0 else float(mean_delta / baseline_mean),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_ci95_low": float(ci_low),
        "bootstrap_ci95_high": float(ci_high),
        "sign_flip_alternative": "candidate_loss_lower",
        "sign_flip_method": sign_flip_method,
        "sign_flip_samples": int(sign_flip_samples),
        "sign_flip_seed": int(permutation_seed),
        "sign_flip_p_value": float(sign_flip_p_value),
        "top5_positive_share_of_net_improvement": top_share,
    }


def summarize_text_overlap(records: list[dict[str, Any]], control_name: str) -> dict[str, float | int]:
    metric_rows = [
        record.get("controls", {}).get(control_name, {}).get("text_overlap", {})
        for record in records
        if isinstance(record.get("controls", {}).get(control_name), dict)
    ]
    keys = sorted({key for metrics in metric_rows if isinstance(metrics, dict) for key in metrics})
    summary: dict[str, float | int] = {"row_count": len(metric_rows)}
    for key in keys:
        values = [metrics.get(key) for metrics in metric_rows if isinstance(metrics.get(key), (int, float))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
    return summary


def summarize_generation_parse(
    records: list[dict[str, Any]],
    control_name: str,
    *,
    fallback: str = "empty",
) -> dict[str, Any]:
    parsed_rows = []
    for record in records:
        control = (record.get("controls") or {}).get(control_name) or {}
        if not isinstance(control, dict):
            continue
        stored = control.get("parsed") or {}
        row_fallback = (
            "raw"
            if isinstance(stored, dict) and stored.get("extraction_mode") == "raw"
            else fallback
        )
        parsed = parse_generated_explanation(
            str(control.get("generated") or ""),
            fallback=row_fallback,
        )
        parsed_rows.append(parsed)
    row_count = len(parsed_rows)
    closed_count = sum(1 for item in parsed_rows if bool(item.get("closed")))
    empty_count = sum(1 for item in parsed_rows if bool(item.get("empty")))
    content_usable_count = sum(1 for item in parsed_rows if bool(item.get("content_usable")))
    fallback_only_count = sum(1 for item in parsed_rows if bool(item.get("fallback_only")))
    repetition_loop_count = sum(1 for item in parsed_rows if bool(item.get("repetition_loop")))
    usable_count = sum(1 for item in parsed_rows if bool(item.get("usable")))
    factual_number_densities = [
        float(item.get("factual_number_density") or 0.0) for item in parsed_rows
    ]
    rows_with_factual_numbers = sum(
        1 for item in parsed_rows if int(item.get("factual_number_count") or 0) > 0
    )
    mode_counts: dict[str, int] = {}
    for item in parsed_rows:
        mode = str(item.get("extraction_mode") or "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    return {
        "schema_version": PARSE_QUALITY_SCHEMA_VERSION,
        "row_count": row_count,
        "closed_count": closed_count,
        "closed_fraction": float(closed_count / row_count) if row_count else 0.0,
        "empty_count": empty_count,
        "empty_fraction": float(empty_count / row_count) if row_count else 0.0,
        "content_usable_count": content_usable_count,
        "content_usable_fraction": (
            float(content_usable_count / row_count) if row_count else 0.0
        ),
        "fallback_only_count": fallback_only_count,
        "fallback_only_fraction": (
            float(fallback_only_count / row_count) if row_count else 0.0
        ),
        "repetition_loop_count": repetition_loop_count,
        "repetition_loop_fraction": (
            float(repetition_loop_count / row_count) if row_count else 0.0
        ),
        "true_usable_count": usable_count,
        "true_usable_fraction": (
            float(usable_count / row_count) if row_count else 0.0
        ),
        "usable_count": usable_count,
        "usable_fraction": float(usable_count / row_count) if row_count else 0.0,
        "factual_number_density_mean": (
            float(np.mean(factual_number_densities))
            if factual_number_densities
            else 0.0
        ),
        "rows_with_factual_numbers_count": rows_with_factual_numbers,
        "rows_with_factual_numbers_fraction": (
            float(rows_with_factual_numbers / row_count) if row_count else 0.0
        ),
        "extraction_mode_counts": mode_counts,
    }


def summarize_variant_predictions(
    predictions: dict[str, np.ndarray],
    *,
    targets: np.ndarray,
    train_targets: np.ndarray,
    primary_variant: str = PRIMARY_VARIANT,
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    rowwise_directional: dict[str, np.ndarray] = {}
    rowwise_raw: dict[str, np.ndarray] = {}
    for name, pred in predictions.items():
        summary = metric_summary(
            predictions=pred,
            targets=targets,
            train_targets_for_mean=train_targets,
            include_rowwise=True,
        )
        rowwise_directional[name] = np.asarray(summary.pop("rowwise_directional_mse"))
        rowwise_raw[name] = np.asarray(summary.pop("rowwise_raw_mse"))
        variants[name] = summary

    rowwise_win_rates: dict[str, Any] = {}
    primary_losses = rowwise_directional.get(primary_variant)
    if primary_losses is not None:
        for name, losses in rowwise_directional.items():
            if name == primary_variant:
                continue
            rowwise_win_rates[f"{primary_variant}_vs_{name}"] = pairwise_win_summary(
                primary_losses,
                losses,
                candidate_name=primary_variant,
                baseline_name=name,
            )
    if "teacher" in rowwise_directional and primary_losses is not None:
        rowwise_win_rates["teacher_vs_av_real"] = pairwise_win_summary(
            rowwise_directional["teacher"],
            primary_losses,
            candidate_name="teacher",
            baseline_name=primary_variant,
        )

    return {
        "row_count": int(targets.shape[0]),
        "variants": variants,
        "rowwise_directional_mse": {
            name: losses.astype(float).tolist()
            for name, losses in rowwise_directional.items()
        },
        "rowwise_normalized_mse": {
            name: losses.astype(float).tolist()
            for name, losses in rowwise_directional.items()
        },
        "rowwise_raw_mse": {
            name: losses.astype(float).tolist()
            for name, losses in rowwise_raw.items()
        },
        "rowwise_win_rates": rowwise_win_rates,
    }


def _mean_prediction(train_targets: np.ndarray, row_count: int) -> np.ndarray:
    return np.repeat(train_targets.mean(axis=0, keepdims=True), row_count, axis=0)


def _rows_by_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["row_index"]): row for row in rows}


def score_generated_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    from eval_nano_ar_miles_checkpoint import (
        _load_model_and_tokenizer,
        _predict_token_batches,
        _resolve_hf_dir,
        _sidecar_template,
        predict_prompts,
    )

    hf_dir = _resolve_hf_dir(args.ar_checkpoint_dir)
    template_sources = [hf_dir]
    if args.critic_template_source:
        template_sources.append(args.critic_template_source)
    template_sources.extend([args.train_parquet, args.validation_parquet])
    critic_template = args.critic_template or _sidecar_template(*template_sources)

    rows, train_indices, validation_indices, test_indices = load_eval_rows(
        args.train_parquet,
        args.validation_parquet,
        args.test_parquet,
        eval_splits=args.eval_splits,
        content_family_manifest=args.content_family_manifest,
        content_family_coverage=args.content_family_coverage,
        require_family_disjoint_splits=args.require_family_level_inference,
    )
    selected_by_split = select_eval_indices_by_split(
        rows,
        validation_indices=validation_indices,
        test_indices=test_indices,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        eval_splits=args.eval_splits,
        strategy=args.selection_strategy,
        seed=args.selection_seed,
    )
    dataset_controls = tuple(getattr(args, "dataset_controls", ()) or ())
    control_rows_by_index: dict[int, dict[str, Any]] = {}
    control_dataset_provenance: dict[str, Any] = {}
    if dataset_controls:
        control_rows_by_index, control_dataset_provenance = load_control_rows_by_index(
            args,
            rows,
            selected_by_split,
        )
    validate_generated_record_coverage(
        records,
        expected_by_split=selected_by_split,
        controls_requested=args.generation_controls,
    )
    rows_by_index = _rows_by_index(rows)
    vectors = np.asarray([row["activation_vector"] for row in rows], dtype=np.float32)
    if not train_indices:
        raise ValueError("training rows are required to construct raw-space controls")
    train_targets = vectors[train_indices]
    length_baseline_records = (
        read_generated_jsonl(args.length_baseline_generated_jsonl)
        if args.length_baseline_generated_jsonl is not None
        else None
    )

    model, tokenizer = _load_model_and_tokenizer(
        hf_dir,
        torch_dtype=args.torch_dtype,
        device_map=args.ar_device_map,
        low_cpu_mem_usage=args.ar_low_cpu_mem_usage,
    )
    ar_device_profile_before = (
        collect_ar_device_profile(model) if args.collect_ar_device_profile else None
    )

    report_splits: dict[str, Any] = {}
    prediction_cache_splits: dict[str, dict[str, Any]] = {}
    for split_name in args.eval_splits:
        split_indices = selected_by_split[split_name]
        split_records = [
            record
            for record in records
            if str(record.get("split")) == split_name and int(record.get("row_index", -1)) in split_indices
        ]
        split_records.sort(key=lambda record: int(record["row_index"]))
        row_indices = [int(record["row_index"]) for record in split_records]
        targets = vectors[row_indices]
        prompts_by_variant: dict[str, list[str]] = {
            "teacher": [teacher_prompt_for_row(rows_by_index[row_index], critic_template) for row_index in row_indices]
        }
        controls_present = sorted(
            {
                control
                for record in split_records
                for control in (record.get("controls") or {})
                if isinstance((record.get("controls") or {}).get(control), dict)
            }
        )
        for control in controls_present:
            prompts_by_variant[f"av_{control}"] = [
                format_critic_prompt(
                    critic_template,
                    parse_generated_explanation(
                        str(record.get("controls", {}).get(control, {}).get("generated") or ""),
                        fallback=args.generated_text_fallback,
                    )["explanation"],
                )
                for record in split_records
            ]
        unknown_dataset_controls = sorted(
            set(dataset_controls) - {"source_context", "source_raw"}
        )
        if unknown_dataset_controls:
            raise ValueError(
                "unknown dataset controls: "
                f"{unknown_dataset_controls}; choices=('source_context', 'source_raw')"
            )
        if "source_context" in dataset_controls:
            source_contexts = [
                str(
                    control_rows_by_index[row_index].get(
                        "detokenized_text_truncated"
                    )
                    or ""
                )
                for row_index in row_indices
            ]
            if not all(text.strip() for text in source_contexts):
                raise ValueError(
                    "source_context was requested but detokenized_text_truncated is "
                    f"missing for {sum(not text.strip() for text in source_contexts)} rows"
                )
            prompts_by_variant["source_context"] = [
                format_critic_prompt(critic_template, text) for text in source_contexts
            ]
        length_control_inputs = None
        if length_baseline_records is not None:
            length_control_inputs = build_length_matched_explanations(
                tokenizer,
                candidate_records=split_records,
                baseline_records=length_baseline_records,
                rows_by_index=rows_by_index,
                fallback=args.generated_text_fallback,
            )
            prompts_by_variant["av_real_sft_length_matched"] = [
                format_critic_prompt(critic_template, explanation)
                for explanation in length_control_inputs[
                    "sft_length_matched_explanations"
                ]
            ]
            prompts_by_variant["av_real_teacher_length_matched"] = [
                format_critic_prompt(critic_template, explanation)
                for explanation in length_control_inputs[
                    "teacher_length_matched_explanations"
                ]
            ]

        predictions: dict[str, np.ndarray] = {}
        for name, prompts in prompts_by_variant.items():
            predictions[name] = predict_prompts(
                model,
                tokenizer,
                prompts,
                batch_size=args.ar_batch_size,
                max_length=args.ar_max_length,
            )
        if "source_raw" in dataset_controls:
            source_raw_ids = [
                control_rows_by_index[row_index].get("token_ids_prefix")
                for row_index in row_indices
            ]
            if not all(isinstance(ids, (list, tuple)) and ids for ids in source_raw_ids):
                missing = sum(
                    not isinstance(ids, (list, tuple)) or not ids
                    for ids in source_raw_ids
                )
                raise ValueError(
                    "source_raw was requested but token_ids_prefix is missing or empty "
                    f"for {missing} rows"
                )
            predictions["source_raw"] = _predict_token_batches(
                model,
                [[int(token) for token in ids] for ids in source_raw_ids],
                pad_token_id=int(tokenizer.pad_token_id),
                batch_size=args.ar_batch_size,
            )
        predictions["mean"] = _mean_prediction(train_targets, len(row_indices))

        split_report = summarize_variant_predictions(
            predictions,
            targets=targets,
            train_targets=train_targets,
            primary_variant=PRIMARY_VARIANT,
        )
        split_report["row_indices"] = row_indices
        split_report["doc_ids"] = [str(record.get("doc_id") or "") for record in split_records]
        split_report["content_family_ids"] = [
            str(record.get("content_family_id") or "") for record in split_records
        ]
        split_report["independent_family_count"] = len(
            {family_id for family_id in split_report["content_family_ids"] if family_id}
        )
        split_report["row_keys"] = [
            {
                key: record.get(key)
                for key in ("doc_id", "token_position", "n_raw_tokens", "token_id", "sample_uuid")
                if record.get(key) is not None
            }
            for record in split_records
        ]
        split_report["text_overlap"] = {
            control: summarize_text_overlap(split_records, control) for control in controls_present
        }
        split_report["generation_parse"] = {
            control: summarize_generation_parse(
                split_records,
                control,
                fallback=args.generated_text_fallback,
            )
            for control in controls_present
        }
        prediction_cache_splits[split_name] = {
            "row_indices": row_indices,
            "doc_ids": split_report["doc_ids"],
            "content_family_ids": split_report["content_family_ids"],
            "targets": targets,
            "predictions": predictions,
        }
        if length_control_inputs is not None:
            split_report["length_control_token_counts"] = {
                key: length_control_inputs[key]
                for key in (
                    "candidate_token_counts",
                    "sft_token_counts",
                    "teacher_token_counts",
                )
            }
        report_splits[split_name] = split_report

    prediction_cache_provenance = None
    prediction_cache_path = getattr(args, "prediction_cache_npz", None)
    if prediction_cache_path is not None:
        prediction_cache_path = Path(prediction_cache_path)
        write_prediction_cache(
            prediction_cache_path,
            split_payloads=prediction_cache_splits,
            train_mean=np.asarray(train_targets, dtype=np.float64).mean(axis=0),
            metadata={
                "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
                "ar_checkpoint_dir": str(args.ar_checkpoint_dir),
                "ar_hf_dir": str(hf_dir),
                "critic_template_sha256": sha256_text(critic_template),
                "dataset_provenance": {
                    "train": file_provenance(args.train_parquet),
                    **(
                        {"validation": file_provenance(args.validation_parquet)}
                        if "validation" in args.eval_splits
                        else {}
                    ),
                    **(
                        {"test": file_provenance(args.test_parquet)}
                        if "test" in args.eval_splits and args.test_parquet is not None
                        else {}
                    ),
                },
                "splits": {
                    split_name: {
                        "row_count": len(payload["row_indices"]),
                        "variants": sorted(payload["predictions"]),
                    }
                    for split_name, payload in prediction_cache_splits.items()
                },
            },
        )
        prediction_cache_provenance = file_provenance(prediction_cache_path)

    ar_device_profile_after = (
        collect_ar_device_profile(model) if args.collect_ar_device_profile else None
    )
    del model, tokenizer
    _collect_and_empty_cuda_cache()
    return {
        "ar_hf_dir": str(hf_dir),
        "critic_template": critic_template,
        "prediction_cache": prediction_cache_provenance,
        "ar_device_profile": (
            None
            if ar_device_profile_before is None
            else {
                "before_scoring": ar_device_profile_before,
                "after_scoring": ar_device_profile_after,
            }
        ),
        "dataset_provenance": {
            "train": file_provenance(args.train_parquet),
            **(
                {"validation": file_provenance(args.validation_parquet)}
                if "validation" in args.eval_splits
                else {}
            ),
            **(
                {"test": file_provenance(args.test_parquet)}
                if "test" in args.eval_splits and args.test_parquet is not None
                else {}
            ),
            "source_controls": control_dataset_provenance,
        },
        "splits": report_splits,
    }


def attach_length_control_analysis(
    splits: dict[str, Any],
    baseline_report: dict[str, Any],
) -> None:
    baseline_splits = baseline_report.get("splits") or {}
    for split_name, split in splits.items():
        token_counts = split.get("length_control_token_counts")
        if not isinstance(token_counts, dict):
            continue
        baseline_split = baseline_splits.get(split_name) or {}
        if split.get("row_indices") != baseline_split.get("row_indices"):
            raise ValueError(
                f"length-control baseline row identity mismatch for {split_name}"
            )
        rowwise = split.get("rowwise_directional_mse") or {}
        baseline_rowwise = baseline_split.get("rowwise_directional_mse") or {}
        required = {
            "candidate_losses": rowwise.get("av_real"),
            "baseline_losses": baseline_rowwise.get("av_real"),
            "sft_length_matched_losses": rowwise.get(
                "av_real_sft_length_matched"
            ),
            "teacher_length_matched_losses": rowwise.get(
                "av_real_teacher_length_matched"
            ),
        }
        missing = [name for name, values in required.items() if not isinstance(values, list)]
        if missing:
            raise ValueError(
                f"missing length-control rowwise metrics for {split_name}: {missing}"
            )
        split["length_analysis"] = summarize_length_control_analysis(
            candidate_token_counts=token_counts["candidate_token_counts"],
            sft_token_counts=token_counts["sft_token_counts"],
            teacher_token_counts=token_counts["teacher_token_counts"],
            **required,
        )


def _variant_nmse(split: dict[str, Any], variant: str) -> float | None:
    value = _variant_metric(split, variant, "directional_mse")
    return float(value) if isinstance(value, (int, float)) else None


def _variant_metric(
    split: dict[str, Any],
    variant: str,
    metric_name: str,
) -> float | None:
    variant_report = (split.get("variants") or {}).get(variant) or {}
    value = variant_report.get(metric_name)
    if value is None and metric_name == "directional_mse":
        value = variant_report.get("normalized_mse")
    return float(value) if isinstance(value, (int, float)) else None


def validate_activation_metric_reports(splits: dict[str, Any]) -> None:
    if not splits:
        raise ValueError("activation metric report has no evaluated splits")
    required = (
        "directional_mse",
        "raw_mse",
        "centered_r2",
        "norm_ratio_mean",
    )
    for split_name, split in splits.items():
        variants = split.get("variants") or {}
        if not variants:
            raise ValueError(f"{split_name} has no activation metric variants")
        for variant_name, report in variants.items():
            missing = [
                metric
                for metric in required
                if not isinstance((report or {}).get(metric), (int, float))
            ]
            if missing:
                raise ValueError(
                    f"{split_name}/{variant_name} is missing required metrics: {missing}"
                )


def _rowwise_metric_values(
    split: dict[str, Any],
    variant: str,
    metric_name: str,
) -> list[float] | None:
    field_names = [f"rowwise_{metric_name}"]
    if metric_name == "directional_mse":
        field_names.append("rowwise_normalized_mse")
    for field_name in field_names:
        values = (split.get(field_name) or {}).get(variant)
        if isinstance(values, list):
            return [float(value) for value in values]
    return None


def paired_metric_effect(
    split: dict[str, Any],
    baseline_split: dict[str, Any],
    *,
    metric_name: str,
    variant: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    permutation_samples: int,
    permutation_seed: int,
) -> dict[str, Any] | None:
    row_indices = split.get("row_indices")
    baseline_row_indices = baseline_split.get("row_indices")
    candidate_losses = _rowwise_metric_values(split, variant, metric_name)
    baseline_losses = _rowwise_metric_values(baseline_split, variant, metric_name)
    if not (
        isinstance(row_indices, list)
        and isinstance(baseline_row_indices, list)
        and isinstance(candidate_losses, list)
        and isinstance(baseline_losses, list)
        and len(row_indices) == len(candidate_losses)
        and len(baseline_row_indices) == len(baseline_losses)
    ):
        return None

    def canonical_row_keys(payload: dict[str, Any], expected: int) -> list[str] | None:
        row_keys = payload.get("row_keys")
        if not isinstance(row_keys, list) or len(row_keys) != expected:
            return None
        canonical: list[str] = []
        for row_key in row_keys:
            if not isinstance(row_key, dict):
                return None
            normalized = {
                str(key): json_safe(value)
                for key, value in row_key.items()
                if value is not None
            }
            if not normalized:
                return None
            canonical.append(
                json.dumps(normalized, sort_keys=True, separators=(",", ":"))
            )
        return canonical

    candidate_identities = canonical_row_keys(split, len(candidate_losses))
    baseline_identities = canonical_row_keys(baseline_split, len(baseline_losses))
    identity_kind = "row_key"
    if candidate_identities is None or baseline_identities is None:
        identity_kind = "row_index"
        candidate_identities = [str(int(value)) for value in row_indices]
        baseline_identities = [str(int(value)) for value in baseline_row_indices]

    if len(set(candidate_identities)) != len(candidate_identities):
        raise ValueError(f"candidate {identity_kind} identities are not unique")
    if len(set(baseline_identities)) != len(baseline_identities):
        raise ValueError(f"baseline {identity_kind} identities are not unique")

    candidate_by_row = dict(zip(candidate_identities, map(float, candidate_losses)))
    baseline_by_row = dict(zip(baseline_identities, map(float, baseline_losses)))
    overlap = [
        identity
        for identity in candidate_identities
        if identity in candidate_by_row and identity in baseline_by_row
    ]
    if not overlap:
        return None
    candidate_overlap = np.asarray(
        [candidate_by_row[row_index] for row_index in overlap],
        dtype=np.float64,
    )
    baseline_overlap = np.asarray(
        [baseline_by_row[row_index] for row_index in overlap],
        dtype=np.float64,
    )
    current_doc_ids = split.get("doc_ids")
    doc_by_row = (
        {
            identity: str(doc_id)
            for identity, doc_id in zip(candidate_identities, current_doc_ids)
        }
        if isinstance(current_doc_ids, list) and len(current_doc_ids) == len(row_indices)
        else {}
    )
    overlap_doc_ids = [doc_by_row.get(row_index, "") for row_index in overlap]
    if not overlap_doc_ids or not all(overlap_doc_ids):
        overlap_doc_ids = None
    current_family_ids = split.get("content_family_ids")
    family_by_row = (
        {
            identity: str(family_id)
            for identity, family_id in zip(
                candidate_identities,
                current_family_ids,
            )
        }
        if isinstance(current_family_ids, list)
        and len(current_family_ids) == len(row_indices)
        else {}
    )
    overlap_family_ids = [family_by_row.get(row_index, "") for row_index in overlap]
    if not overlap_family_ids or not all(overlap_family_ids):
        overlap_family_ids = None
    metric_variant = variant if metric_name == "directional_mse" else f"{variant}_{metric_name}"
    return {
        "metric": metric_name,
        "row_identity_kind": identity_kind,
        "row_identity_match": (
            candidate_identities == baseline_identities
        ),
        "row_overlap_count": len(overlap),
        "candidate_matched_mean": float(candidate_overlap.mean()),
        "baseline_matched_mean": float(baseline_overlap.mean()),
        "rowwise_win_rate": pairwise_win_summary(
            candidate_overlap,
            baseline_overlap,
            candidate_name=metric_variant,
            baseline_name=f"baseline_{metric_variant}",
        ),
        "paired_improvement": paired_improvement_summary(
            candidate_overlap,
            baseline_overlap,
            doc_ids=overlap_doc_ids,
            content_family_ids=overlap_family_ids,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            permutation_samples=permutation_samples,
            permutation_seed=permutation_seed,
        ),
    }


def build_gate_summary(
    splits: dict[str, Any],
    *,
    control_margin: float,
    baseline_report: dict[str, Any] | None = None,
    dataset_provenance: dict[str, Any] | None = None,
    baseline_margin: float = 0.0,
    min_control_win_fraction: float = 0.0,
    min_baseline_win_fraction: float = 0.0,
    min_baseline_relative_improvement: float = 0.0,
    require_baseline_ci_positive: bool = False,
    require_clustered_baseline_ci: bool = False,
    require_baseline_dataset_match: bool = False,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    permutation_samples: int = 100_000,
    permutation_seed: int = 0,
    min_closed_fraction: float = 0.0,
    min_usable_fraction: float = 0.0,
    generation_protocol: dict[str, Any] | None = None,
    require_generation_protocol_match: bool = False,
    require_family_level_inference: bool = False,
    min_independent_families: int = 100,
) -> dict[str, Any]:
    if not splits:
        raise ValueError("at least one evaluated split is required")
    unknown_splits = sorted(set(splits) - {"validation", "test"})
    if unknown_splits:
        raise ValueError(f"unsupported gate splits: {unknown_splits}")
    if control_margin <= 0:
        raise ValueError("control_margin must be positive")
    if baseline_margin < 0:
        raise ValueError("baseline_margin must be non-negative")
    if min_control_win_fraction < 0 or min_control_win_fraction > 1:
        raise ValueError("min_control_win_fraction must be in [0, 1]")
    if min_baseline_win_fraction < 0 or min_baseline_win_fraction > 1:
        raise ValueError("min_baseline_win_fraction must be in [0, 1]")
    if min_baseline_relative_improvement < 0:
        raise ValueError("min_baseline_relative_improvement must be non-negative")
    if permutation_samples <= 0:
        raise ValueError("permutation_samples must be positive")
    if min_independent_families <= 0:
        raise ValueError("min_independent_families must be positive")
    baseline_dataset_provenance = (baseline_report or {}).get("dataset_provenance") or {}
    generation_protocol_parity = compare_generation_protocols(
        generation_protocol,
        (baseline_report or {}).get("generation_protocol"),
    )
    current_dataset_provenance = dataset_provenance or {}
    dataset_hash_match = None
    if baseline_report is not None:
        dataset_hash_match = bool(current_dataset_provenance and baseline_dataset_provenance)
        if dataset_hash_match:
            for split_name in ("train", *splits):
                current_hash = (current_dataset_provenance.get(split_name) or {}).get("sha256")
                baseline_hash = (baseline_dataset_provenance.get(split_name) or {}).get("sha256")
                if not current_hash or current_hash != baseline_hash:
                    dataset_hash_match = False
                    break
    control_variants = ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
    split_gates: dict[str, Any] = {}
    for split_name, split in splits.items():
        primary = _variant_nmse(split, PRIMARY_VARIANT)
        controls: dict[str, Any] = {}
        for control in control_variants:
            value = _variant_nmse(split, control)
            rowwise = (split.get("rowwise_win_rates") or {}).get(f"{PRIMARY_VARIANT}_vs_{control}") or {}
            rowwise_win_fraction = rowwise.get("candidate_better_fraction")
            rowwise_pass = (
                min_control_win_fraction <= 0
                or (
                    isinstance(rowwise_win_fraction, (int, float))
                    and float(rowwise_win_fraction) >= min_control_win_fraction
                )
            )
            margin_pass = primary is not None and value is not None and primary <= value - control_margin
            controls[control] = {
                "directional_mse": value,
                "normalized_mse": value,
                "raw_mse": _variant_metric(split, control, "raw_mse"),
                "primary_beats": margin_pass and rowwise_pass,
                "delta_control_minus_primary": None if primary is None or value is None else value - primary,
                "rowwise_win_fraction": (
                    float(rowwise_win_fraction) if isinstance(rowwise_win_fraction, (int, float)) else None
                ),
                "rowwise_win_fraction_passed": bool(rowwise_pass),
            }
        missing_controls = [name for name, item in controls.items() if item["normalized_mse"] is None]
        teacher_nmse = _variant_nmse(split, "teacher")
        primary_raw_mse = _variant_metric(split, PRIMARY_VARIANT, "raw_mse")
        primary_centered_r2 = _variant_metric(split, PRIMARY_VARIANT, "centered_r2")
        primary_norm_ratio = _variant_metric(split, PRIMARY_VARIANT, "norm_ratio_mean")
        teacher_raw_mse = _variant_metric(split, "teacher", "raw_mse")
        baseline_nmse = None
        baseline_raw_mse = None
        baseline_row_identity_match = None
        baseline_row_identity_kind = None
        baseline_row_overlap_count = 0
        primary_matched_nmse = None
        baseline_matched_nmse = None
        primary_matched_raw_mse = None
        baseline_matched_raw_mse = None
        baseline_rowwise_win_rate = None
        baseline_rowwise_win_rate_by_metric: dict[str, Any] = {}
        baseline_paired_improvement = None
        baseline_paired_improvement_by_metric: dict[str, Any] = {}
        split_family_ids = {
            str(value)
            for value in (split.get("content_family_ids") or [])
            if str(value)
        }
        independent_family_count = (
            len(split_family_ids)
            if split_family_ids
            else int(split.get("independent_family_count") or 0)
        )
        family_inference_passed = (
            independent_family_count >= min_independent_families
        )
        baseline_beaten = None
        if baseline_report is not None:
            baseline_split = (baseline_report.get("splits") or {}).get(split_name) or {}
            baseline_nmse = _variant_nmse(baseline_split, PRIMARY_VARIANT)
            baseline_raw_mse = _variant_metric(baseline_split, PRIMARY_VARIANT, "raw_mse")
            row_indices = split.get("row_indices")
            baseline_row_indices = baseline_split.get("row_indices")
            metric_effects = {
                metric_name: paired_metric_effect(
                    split,
                    baseline_split,
                    metric_name=metric_name,
                    variant=PRIMARY_VARIANT,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                    permutation_samples=permutation_samples,
                    permutation_seed=permutation_seed,
                )
                for metric_name in ("directional_mse", "raw_mse")
            }
            for metric_name, effect in metric_effects.items():
                if effect is None:
                    continue
                baseline_rowwise_win_rate_by_metric[metric_name] = effect[
                    "rowwise_win_rate"
                ]
                baseline_paired_improvement_by_metric[metric_name] = effect[
                    "paired_improvement"
                ]
            directional_effect = metric_effects["directional_mse"]
            if directional_effect is not None:
                baseline_row_identity_match = directional_effect["row_identity_match"]
                baseline_row_identity_kind = directional_effect["row_identity_kind"]
                baseline_row_overlap_count = directional_effect["row_overlap_count"]
                primary_matched_nmse = directional_effect["candidate_matched_mean"]
                baseline_matched_nmse = directional_effect["baseline_matched_mean"]
                baseline_rowwise_win_rate = directional_effect["rowwise_win_rate"]
                baseline_paired_improvement = directional_effect["paired_improvement"]
                if baseline_paired_improvement["independent_unit"] == "content_family_id":
                    independent_family_count = int(
                        baseline_paired_improvement["independent_unit_count"]
                    )
                    family_inference_passed = (
                        independent_family_count >= min_independent_families
                    )
            raw_effect = metric_effects["raw_mse"]
            if raw_effect is not None:
                primary_matched_raw_mse = raw_effect["candidate_matched_mean"]
                baseline_matched_raw_mse = raw_effect["baseline_matched_mean"]
            baseline_for_comparison = baseline_matched_nmse if baseline_matched_nmse is not None else baseline_nmse
            primary_for_comparison = primary_matched_nmse if primary_matched_nmse is not None else primary
            aggregate_beaten = (
                primary_for_comparison is not None
                and baseline_for_comparison is not None
                and primary_for_comparison <= baseline_for_comparison - baseline_margin
            )
            complete_overlap = bool(
                isinstance(row_indices, list)
                and baseline_row_overlap_count == len(row_indices)
            )
            win_fraction = (
                baseline_rowwise_win_rate.get("candidate_better_fraction")
                if baseline_rowwise_win_rate is not None
                else None
            )
            win_rate_passed = bool(
                isinstance(win_fraction, (int, float))
                and float(win_fraction) >= min_baseline_win_fraction
            )
            relative_improvement = (
                baseline_paired_improvement.get("relative_improvement")
                if baseline_paired_improvement is not None
                else None
            )
            relative_improvement_passed = bool(
                isinstance(relative_improvement, (int, float))
                and float(relative_improvement) >= min_baseline_relative_improvement
            )
            ci_positive = bool(
                baseline_paired_improvement is not None
                and baseline_paired_improvement["bootstrap_ci95_low"] > 0.0
            )
            clustered_ci = bool(
                baseline_paired_improvement is not None
                and baseline_paired_improvement["independent_unit"]
                in {"doc_id", "content_family_id"}
            )
            dataset_match_passed = not require_baseline_dataset_match or dataset_hash_match is True
            baseline_beaten = bool(
                aggregate_beaten
                and baseline_rowwise_win_rate is not None
                and complete_overlap
                and win_rate_passed
                and relative_improvement_passed
                and (not require_baseline_ci_positive or ci_positive)
                and (not require_clustered_baseline_ci or clustered_ci)
                and dataset_match_passed
                and (not require_family_level_inference or family_inference_passed)
            )
        parse_report = (split.get("generation_parse") or {}).get("real") or {}
        closed_fraction = float(parse_report.get("closed_fraction") or 0.0)
        usable_fraction = float(parse_report.get("usable_fraction", 1.0 - float(parse_report.get("empty_fraction") or 0.0)))
        parse_health = {
            "control": "real",
            "closed_fraction": closed_fraction,
            "usable_fraction": usable_fraction,
            "min_closed_fraction": float(min_closed_fraction),
            "min_usable_fraction": float(min_usable_fraction),
            "passed": closed_fraction >= min_closed_fraction and usable_fraction >= min_usable_fraction,
        }
        split_gates[split_name] = {
            "primary_variant": PRIMARY_VARIANT,
            "gate_metric": "directional_mse",
            "primary_directional_mse": primary,
            "primary_normalized_mse": primary,
            "primary_raw_mse": primary_raw_mse,
            "primary_centered_r2": primary_centered_r2,
            "primary_norm_ratio_mean": primary_norm_ratio,
            "raw_metrics_complete": all(
                value is not None
                for value in (
                    primary_raw_mse,
                    primary_centered_r2,
                    primary_norm_ratio,
                )
            ),
            "teacher_directional_mse": teacher_nmse,
            "teacher_normalized_mse": teacher_nmse,
            "teacher_raw_mse": teacher_raw_mse,
            "controls": controls,
            "missing_controls": missing_controls,
            "beats_all_controls": not missing_controls and all(item["primary_beats"] for item in controls.values()),
            "baseline_primary_directional_mse": baseline_nmse,
            "baseline_primary_normalized_mse": baseline_nmse,
            "baseline_primary_raw_mse": baseline_raw_mse,
            "baseline_row_identity_match": baseline_row_identity_match,
            "baseline_row_identity_kind": baseline_row_identity_kind,
            "baseline_row_overlap_count": baseline_row_overlap_count,
            "primary_matched_directional_mse": primary_matched_nmse,
            "primary_matched_normalized_mse": primary_matched_nmse,
            "primary_matched_raw_mse": primary_matched_raw_mse,
            "baseline_primary_matched_directional_mse": baseline_matched_nmse,
            "baseline_primary_matched_normalized_mse": baseline_matched_nmse,
            "baseline_primary_matched_raw_mse": baseline_matched_raw_mse,
            "baseline_rowwise_win_rate": baseline_rowwise_win_rate,
            "baseline_rowwise_win_rate_by_metric": baseline_rowwise_win_rate_by_metric,
            "baseline_paired_improvement": baseline_paired_improvement,
            "baseline_paired_improvement_by_metric": baseline_paired_improvement_by_metric,
            "independent_family_count": independent_family_count,
            "min_independent_families": int(min_independent_families),
            "family_inference_passed": bool(family_inference_passed),
            "baseline_dataset_hash_match": dataset_hash_match,
            "baseline_beaten": baseline_beaten,
            "parse_health": parse_health,
        }

    baseline_required = baseline_report is not None
    passed = all(
        item["beats_all_controls"]
        and item["parse_health"]["passed"]
        and (not require_family_level_inference or item["family_inference_passed"])
        for item in split_gates.values()
    )
    if baseline_required:
        passed = passed and all(item["baseline_beaten"] for item in split_gates.values())
    current_protocol_compatible = bool(
        generation_protocol
        and not str(generation_protocol.get("prefix") or "")
    )
    if require_generation_protocol_match:
        passed = passed and current_protocol_compatible
        if baseline_required:
            passed = (
                passed
                and generation_protocol_parity["matched"]
                and generation_protocol_parity["publication_compatible"]
            )
    if require_family_level_inference:
        enough_families = all(
            item["family_inference_passed"] for item in split_gates.values()
        )
        if baseline_required and enough_families:
            publication_status = "confirmatory"
        elif enough_families:
            publication_status = "family_controlled_validation"
        else:
            publication_status = "small_sample_pilot"
    else:
        publication_status = "exploratory"
    return {
        "passed": bool(passed),
        "activation_metric_schema_version": ACTIVATION_METRIC_SCHEMA_VERSION,
        "metric_definitions": {
            "directional_mse": "2 * (1 - cosine_similarity)",
            "raw_mse": "mean((prediction - target)^2) over rows and features",
            "centered_r2": "1 - raw_mse / train_mean_predictor_raw_mse",
            "norm_ratio_mean": "mean(||prediction|| / ||target||)",
        },
        "publication_status": publication_status,
        "generation_protocol_parity": generation_protocol_parity,
        "current_generation_protocol_compatible": current_protocol_compatible,
        "require_generation_protocol_match": bool(require_generation_protocol_match),
        "control_margin": control_margin,
        "baseline_margin": baseline_margin,
        "min_control_win_fraction": min_control_win_fraction,
        "min_baseline_win_fraction": min_baseline_win_fraction,
        "min_baseline_relative_improvement": min_baseline_relative_improvement,
        "require_baseline_ci_positive": bool(require_baseline_ci_positive),
        "require_clustered_baseline_ci": bool(require_clustered_baseline_ci),
        "require_baseline_dataset_match": bool(require_baseline_dataset_match),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "permutation_samples": int(permutation_samples),
        "permutation_seed": int(permutation_seed),
        "require_family_level_inference": bool(require_family_level_inference),
        "min_independent_families": int(min_independent_families),
        "min_closed_fraction": min_closed_fraction,
        "min_usable_fraction": min_usable_fraction,
        "baseline_required": baseline_required,
        "splits": split_gates,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    generated_jsonl = args.generated_jsonl or default_generated_jsonl(args.report_json)
    if args.reuse_generated:
        records = read_generated_jsonl(generated_jsonl)
    else:
        if args.av_hf_checkpoint is None:
            raise ValueError("--av-hf-checkpoint is required unless --reuse-generated is set")
        if args.generation_workers > 1:
            records = generate_roundtrip_records_with_workers(args, generated_jsonl)
        else:
            records = generate_roundtrip_records(
                args,
                stream_jsonl=generated_jsonl if args.stream_generated or args.resume_generated else None,
            )
        if not args.stream_generated and args.generation_workers <= 1:
            write_generated_jsonl(generated_jsonl, records)

    requested_generation_protocol = (
        load_json_object(args.expected_generation_protocol_json)
        if args.expected_generation_protocol_json is not None
        else build_generation_protocol(args)
    )
    identity_required = args.require_generation_protocol_match or args.reuse_generated
    generation_protocol = validate_generated_record_protocols(
        records,
        expected_protocol=requested_generation_protocol,
        require=identity_required,
    )
    generation_provenance = validate_generated_record_provenance(
        records,
        expected_provenance=build_generation_provenance(args),
        require=identity_required,
    )

    score = score_generated_records(args, records)
    validate_activation_metric_reports(score["splits"])
    baseline_report = None
    if args.baseline_report_json:
        baseline_report = json.loads(args.baseline_report_json.read_text())
        validate_activation_metric_reports(baseline_report.get("splits") or {})
    if args.length_baseline_generated_jsonl is not None:
        if baseline_report is None:
            raise ValueError("length controls require a baseline report")
        attach_length_control_analysis(score["splits"], baseline_report)
    gate = build_gate_summary(
        score["splits"],
        control_margin=args.control_margin,
        baseline_report=baseline_report,
        dataset_provenance=score["dataset_provenance"],
        baseline_margin=args.baseline_margin,
        min_control_win_fraction=args.min_control_win_fraction,
        min_baseline_win_fraction=args.min_baseline_win_fraction,
        min_baseline_relative_improvement=args.min_baseline_relative_improvement,
        require_baseline_ci_positive=args.require_baseline_ci_positive,
        require_clustered_baseline_ci=args.require_clustered_baseline_ci,
        require_baseline_dataset_match=args.require_baseline_dataset_match,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        permutation_samples=args.permutation_samples,
        permutation_seed=args.permutation_seed,
        min_closed_fraction=args.min_closed_fraction,
        min_usable_fraction=args.min_usable_fraction,
        generation_protocol=generation_protocol,
        require_generation_protocol_match=args.require_generation_protocol_match,
        require_family_level_inference=args.require_family_level_inference,
        min_independent_families=args.min_independent_families,
    )
    return {
        "schema_version": ROUNDTRIP_SCHEMA_VERSION,
        "parse_quality_schema_version": PARSE_QUALITY_SCHEMA_VERSION,
        "length_control_schema_version": LENGTH_CONTROL_SCHEMA_VERSION,
        "av_hf_checkpoint": None if args.av_hf_checkpoint is None else str(args.av_hf_checkpoint),
        "ar_checkpoint_dir": str(args.ar_checkpoint_dir),
        "ar_hf_dir": score["ar_hf_dir"],
        "train_parquet": str(args.train_parquet),
        "validation_parquet": str(args.validation_parquet),
        "test_parquet": (
            str(args.test_parquet)
            if "test" in args.eval_splits and args.test_parquet is not None
            else None
        ),
        "content_family_manifest": (
            None
            if args.content_family_manifest is None
            else str(args.content_family_manifest)
        ),
        "content_family_coverage": (
            None
            if args.content_family_coverage is None
            else str(args.content_family_coverage)
        ),
        "selection_strategy": args.selection_strategy,
        "selection_seed": args.selection_seed,
        "generated_jsonl": str(generated_jsonl),
        "generated_jsonl_provenance": file_provenance(generated_jsonl),
        "length_baseline_generated_jsonl": (
            None
            if args.length_baseline_generated_jsonl is None
            else str(args.length_baseline_generated_jsonl)
        ),
        "length_baseline_generated_provenance": (
            None
            if args.length_baseline_generated_jsonl is None
            else file_provenance(args.length_baseline_generated_jsonl)
        ),
        "generation_controls": list(args.generation_controls),
        "dataset_controls": list(getattr(args, "dataset_controls", ()) or ()),
        "eval_splits": list(args.eval_splits),
        "validation_limit": args.validation_limit,
        "test_limit": args.test_limit,
        "max_new_tokens": args.max_new_tokens,
        "injection_scale": args.injection_scale,
        "generation_prefix": args.generation_prefix,
        "stop_text": args.stop_text,
        "generated_text_fallback": args.generated_text_fallback,
        "generation_backend": args.generation_backend,
        "generation_workers": args.generation_workers,
        "generation_worker_devices": list(args.generation_worker_devices or []),
        "generation_protocol": generation_protocol,
        "generation_protocol_sha256": (
            None if generation_protocol is None else generation_protocol_sha256(generation_protocol)
        ),
        "expected_generation_protocol_json": (
            None
            if args.expected_generation_protocol_json is None
            else file_provenance(args.expected_generation_protocol_json)
        ),
        "generation_provenance": build_generation_provenance(args),
        "validated_generation_provenance": generation_provenance,
        "critic_template": score["critic_template"],
        "prediction_cache": score.get("prediction_cache"),
        "ar_device_profile": score.get("ar_device_profile"),
        "dataset_provenance": score["dataset_provenance"],
        "splits": score["splits"],
        "gate": gate,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av-hf-checkpoint", type=Path)
    parser.add_argument("--ar-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--validation-control-parquet", type=Path)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--test-control-parquet", type=Path)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--generated-jsonl", type=Path)
    parser.add_argument("--expected-generation-protocol-json", type=Path)
    parser.add_argument("--prediction-cache-npz", type=Path)
    parser.add_argument("--reuse-generated", action="store_true")
    parser.add_argument("--critic-template")
    parser.add_argument("--critic-template-source", type=Path)
    parser.add_argument("--generation-controls", nargs="+", default=list(DEFAULT_GENERATION_CONTROLS))
    parser.add_argument(
        "--dataset-controls",
        nargs="+",
        choices=("source_context", "source_raw"),
        default=[],
        help="Score exact row-derived text/token controls without AV generation.",
    )
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=64)
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("validation", "test"),
        default=["validation"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--generation-prefix", default="")
    parser.add_argument("--stop-text")
    parser.add_argument("--generated-text-fallback", choices=GENERATED_TEXT_FALLBACKS, default="empty")
    parser.add_argument(
        "--generation-backend",
        choices=("legacy", "legacy_batch", "cache"),
        default="legacy",
    )
    parser.add_argument(
        "--allow-unsafe-cache-backend",
        action="store_true",
        help="Permit the known-untrusted incremental cache backend for diagnostics only.",
    )
    parser.add_argument("--generation-workers", type=int, default=1)
    parser.add_argument("--generation-max-parallel-workers", type=int)
    parser.add_argument("--generation-parent-worker-count", type=int)
    parser.add_argument("--generation-worker-devices", nargs="*")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--generation-shard-index", type=int, default=0)
    parser.add_argument("--generation-shard-count", type=int, default=1)
    parser.add_argument("--stream-generated", action="store_true")
    parser.add_argument("--resume-generated", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--injection-scale", default="75")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--av-device-map", default="auto")
    add_bool_optional_arg(parser, "--av-low-cpu-mem-usage", default=True)
    parser.add_argument("--ar-device-map", default="auto")
    add_bool_optional_arg(parser, "--ar-low-cpu-mem-usage", default=False)
    add_bool_optional_arg(parser, "--collect-ar-device-profile", default=False)
    parser.add_argument("--ar-batch-size", type=int, default=4)
    parser.add_argument("--ar-max-length", type=int)
    parser.add_argument("--baseline-report-json", type=Path)
    parser.add_argument("--length-baseline-generated-jsonl", type=Path)
    parser.add_argument("--content-family-manifest", type=Path)
    parser.add_argument("--content-family-coverage", type=Path)
    parser.add_argument(
        "--selection-strategy",
        choices=("row_order", "family_stratified"),
        default="row_order",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--control-margin", type=float, default=0.1)
    parser.add_argument("--baseline-margin", type=float, default=0.0)
    parser.add_argument("--min-control-win-fraction", type=float, default=0.0)
    parser.add_argument("--min-baseline-win-fraction", type=float, default=0.0)
    parser.add_argument("--min-baseline-relative-improvement", type=float, default=0.0)
    add_bool_optional_arg(parser, "--require-baseline-ci-positive", default=False)
    add_bool_optional_arg(parser, "--require-clustered-baseline-ci", default=False)
    add_bool_optional_arg(parser, "--require-baseline-dataset-match", default=False)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--permutation-samples", type=int, default=100_000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--min-independent-families", type=int, default=100)
    parser.add_argument("--min-closed-fraction", type=float, default=0.0)
    parser.add_argument("--min-usable-fraction", type=float, default=0.0)
    parser.add_argument("--av-model-revision", default=None)
    parser.add_argument("--av-tokenizer-revision", default=None)
    parser.add_argument("--av-model-fingerprint", default=None)
    parser.add_argument("--av-tokenizer-fingerprint", default=None)
    parser.add_argument("--av-load-mode", choices=("full", "meta", "config"), default="full")
    parser.add_argument("--av-attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    add_bool_optional_arg(parser, "--trust-remote-code", default=True)
    add_bool_optional_arg(parser, "--require-generation-protocol-match", default=False)
    add_bool_optional_arg(parser, "--require-family-level-inference", default=False)
    args = parser.parse_args(argv)
    if args.generation_backend == "cache" and not args.allow_unsafe_cache_backend:
        parser.error(
            "generation backend 'cache' is quarantined after token-1 equivalence failures; "
            "use legacy_batch, or opt in explicitly for diagnostics"
        )
    if args.generation_parent_worker_count is not None and args.generation_parent_worker_count <= 0:
        parser.error("--generation-parent-worker-count must be positive")
    if len(set(args.eval_splits)) != len(args.eval_splits):
        parser.error("--eval-splits must not contain duplicates")
    if "test" in args.eval_splits and args.test_parquet is None:
        parser.error("--test-parquet is required when --eval-splits includes test")
    identity_required = (
        args.require_generation_protocol_match
        or args.reuse_generated
        or args.resume_generated
    )
    if identity_required and not (
        args.av_model_fingerprint and args.av_tokenizer_fingerprint
    ):
        parser.error(
            "cache reuse and protocol matching require explicit model and tokenizer fingerprints"
        )
    if identity_required and not MODEL_FINGERPRINT_RE.fullmatch(
        str(args.av_model_fingerprint or "")
    ):
        parser.error("--av-model-fingerprint must be a content SHA-256 fingerprint")
    if identity_required and not TOKENIZER_FINGERPRINT_RE.fullmatch(
        str(args.av_tokenizer_fingerprint or "")
    ):
        parser.error("--av-tokenizer-fingerprint must be a content SHA-256 fingerprint")
    if identity_required and str(args.generation_prefix or ""):
        parser.error("cache reuse and protocol matching require an empty generation prefix")
    if args.require_family_level_inference and args.content_family_manifest is None:
        parser.error("--require-family-level-inference requires --content-family-manifest")
    if args.require_family_level_inference and args.content_family_coverage is None:
        parser.error("--require-family-level-inference requires --content-family-coverage")
    if args.require_family_level_inference and args.selection_strategy != "family_stratified":
        parser.error(
            "--require-family-level-inference requires --selection-strategy family_stratified"
        )
    if (
        args.length_baseline_generated_jsonl is not None
        and args.baseline_report_json is None
    ):
        parser.error(
            "--length-baseline-generated-jsonl requires --baseline-report-json"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.generation_only:
        generated_jsonl = args.generated_jsonl or default_generated_jsonl(args.report_json)
        if args.generation_workers > 1:
            records = generate_roundtrip_records_with_workers(args, generated_jsonl)
        else:
            records = generate_roundtrip_records(
                args,
                stream_jsonl=generated_jsonl if args.stream_generated or args.resume_generated else None,
            )
        if not args.stream_generated and args.generation_workers <= 1:
            write_generated_jsonl(generated_jsonl, records)
        generation_protocol = validate_generated_record_protocols(
            records,
            expected_protocol=build_generation_protocol(args),
            require=args.require_generation_protocol_match,
        )
        generation_provenance = validate_generated_record_provenance(
            records,
            expected_provenance=build_generation_provenance(args),
            require=args.require_generation_protocol_match,
        )
        if args.generation_shard_count > 1:
            # Parent-owned reporting prevents concurrent shard workers from racing
            # to overwrite the canonical generation report.
            print(
                json.dumps(
                    {
                        "status": "generation_shard_complete",
                        "generated_jsonl": str(generated_jsonl),
                        "row_count": len(records),
                        "shard_index": int(args.generation_shard_index),
                        "shard_count": int(args.generation_shard_count),
                        "generation_protocol_sha256": (
                            None
                            if generation_protocol is None
                            else generation_protocol_sha256(generation_protocol)
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        generation_report = {
            "schema_version": "nano_roundtrip_generation_report.v1",
            "generated_jsonl": str(generated_jsonl),
            "row_count": len(records),
            "eval_splits": list(args.eval_splits),
            "generation_protocol": generation_protocol,
            "generation_protocol_sha256": (
                None
                if generation_protocol is None
                else generation_protocol_sha256(generation_protocol)
            ),
            "generation_provenance": generation_provenance,
        }
        write_json(args.report_json, generation_report)
        print(json.dumps(json_safe(generation_report), indent=2))
        return 0
    report = evaluate(args)
    write_json(args.report_json, json_safe(report))
    print(json.dumps(json_safe(report), indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
