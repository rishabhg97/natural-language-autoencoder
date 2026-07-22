#!/usr/bin/env python3
"""Nano source-prefix replay probe for activation geometry debugging.

This diagnostic checks whether a stored target activation can be recovered by
feeding the stored source prefix text back through Nano and extracting the same
residual boundary at the last replay token.

It does not train Nano, run PEFT/LoRA, serve, run RL, or call a teacher model.
It is intended to separate target/vector provenance problems from AR text
geometry problems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, NamedTuple

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pq = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_ar_frozen_baseline import (  # noqa: E402
    l2_normalize_rows,
    mean_target_metrics,
    random_matched_norm_targets,
    select_token_vectors_by_lengths,
    vector_metrics,
)
from nano_extraction_identity import parse_boundaries, prefix_forward_to_R_b  # noqa: E402
from nano_introspection import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_ROOT,
    add_bool_optional_arg,
    block_pattern_from_config,
    build_metadata_record,
    classify_blocker,
    get_config_value,
    json_safe,
    load_config_from_args,
    load_model_from_args,
    load_tokenizer_from_args,
    make_run_dir,
    resolve_nano_module_paths,
    write_json,
)


class SourceReplaySpec(NamedTuple):
    record_id: str
    row_index: int
    boundary_b: int
    source_text: str
    source_token_ids: list[int] | None
    activation_vector: list[float]
    metadata: dict[str, Any]


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _pstdev(values: list[float]) -> float | None:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0 if values else None


def summarize_token_count_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [
        int(row["replay_token_count"]) - int(row["stored_n_raw_tokens"])
        for row in rows
        if row.get("stored_n_raw_tokens") is not None and row.get("replay_token_count") is not None
    ]
    exact_count = sum(1 for delta in deltas if delta == 0)
    return {
        "count": len(deltas),
        "exact_count": exact_count,
        "exact_fraction": float(exact_count / len(deltas)) if deltas else None,
        "delta_min": min(deltas) if deltas else None,
        "delta_max": max(deltas) if deltas else None,
        "delta_mean": _mean([float(delta) for delta in deltas]),
        "delta_std": _pstdev([float(delta) for delta in deltas]),
    }


def _subset_rows(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return tensor.new_zeros((0, tensor.shape[-1]))
    return tensor[torch.tensor(indices, dtype=torch.long)]


def _direct_control_metrics(features: torch.Tensor, targets: torch.Tensor, *, seed: int) -> dict[str, Any]:
    if features.shape[0] == 0:
        return {
            "count": 0,
            "correct": None,
            "mean_target": None,
            "shuffled_target": None,
            "random_matched_norm_target": None,
        }
    correct = vector_metrics(features, targets)
    mean_target = mean_target_metrics(targets, targets)
    shuffled = vector_metrics(features, targets.roll(shifts=1, dims=0)) if targets.shape[0] > 1 else None
    random_target = random_matched_norm_targets(targets, seed=seed)
    random_metrics = vector_metrics(features, random_target)
    return {
        "count": int(features.shape[0]),
        "correct": correct,
        "mean_target": mean_target,
        "shuffled_target": shuffled,
        "random_matched_norm_target": random_metrics,
    }


def replay_metric_bundle(
    features: torch.Tensor,
    targets: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if features.shape != targets.shape:
        raise ValueError(f"features shape {tuple(features.shape)} != targets shape {tuple(targets.shape)}")
    if features.shape[0] != len(rows):
        raise ValueError(f"features rows {features.shape[0]} != metadata rows {len(rows)}")
    exact_indices = [
        idx
        for idx, row in enumerate(rows)
        if row.get("stored_n_raw_tokens") is not None
        and row.get("replay_token_count") is not None
        and int(row["stored_n_raw_tokens"]) == int(row["replay_token_count"])
    ]
    return {
        "token_count_deltas": summarize_token_count_deltas(rows),
        "all": _direct_control_metrics(features, targets, seed=seed),
        "exact_token_count": _direct_control_metrics(
            _subset_rows(features, exact_indices),
            _subset_rows(targets, exact_indices),
            seed=seed + 1,
        ),
    }


def target_geometry_summary(targets: torch.Tensor) -> dict[str, Any]:
    targets_f = targets.float()
    norms = targets_f.norm(dim=-1)
    normalized = l2_normalize_rows(targets_f, target_scale=float(targets.shape[-1]) ** 0.5)
    mean_raw = targets_f.mean(dim=0, keepdim=True)
    mean_normed = normalized.mean(dim=0, keepdim=True)
    centered = targets_f - mean_raw
    # SVD on [n, d] is cheap for our current <=2k-row smoke sets and gives a
    # direct read of anisotropy without forming a d x d covariance matrix.
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values.pow(2)
    total_variance = float(variance.sum().item())
    if total_variance > 0:
        fractions = (variance / total_variance).detach().cpu()
        participation = float((variance.sum().pow(2) / variance.pow(2).sum().clamp_min(1e-12)).item())
    else:
        fractions = torch.zeros_like(variance)
        participation = 0.0

    def topk(k: int) -> float:
        if fractions.numel() == 0:
            return 0.0
        return float(fractions[: min(k, fractions.numel())].sum().item())

    return {
        "count": int(targets.shape[0]),
        "hidden_size": int(targets.shape[-1]),
        "target_l2_mean": float(norms.mean().item()),
        "target_l2_std": float(norms.std(unbiased=False).item()) if norms.numel() > 1 else 0.0,
        "target_l2_min": float(norms.min().item()),
        "target_l2_max": float(norms.max().item()),
        "raw_mean_l2": float(mean_raw.norm().item()),
        "normalized_mean_l2": float(mean_normed.norm().item()),
        "centered_effective_rank": participation,
        "centered_variance_top1": topk(1),
        "centered_variance_top5": topk(5),
        "centered_variance_top10": topk(10),
        "centered_variance_top50": topk(50),
    }


def _pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def encode_batch_no_truncate(
    tokenizer: Any,
    texts: list[str],
    max_length: int | None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    encoded_ids: list[list[int]] = []
    lengths: list[int] = []
    for row_idx, text in enumerate(texts):
        encoded = tokenizer(text, add_special_tokens=True)
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        token_count = len(ids)
        if max_length is not None and token_count > max_length:
            raise ValueError(
                f"source replay row={row_idx} token_count={token_count} exceeds --source-max-length={max_length}; "
                "do not truncate source replay because that changes the target token"
            )
        encoded_ids.append([int(item) for item in ids])
        lengths.append(token_count)

    max_batch_length = max(lengths, default=0)
    if max_batch_length <= 0:
        raise ValueError("cannot encode an empty source replay batch")
    input_ids = torch.full((len(encoded_ids), max_batch_length), _pad_token_id(tokenizer), dtype=torch.long)
    attention_mask = torch.zeros((len(encoded_ids), max_batch_length), dtype=torch.long)
    for row_idx, ids in enumerate(encoded_ids):
        row_length = lengths[row_idx]
        input_ids[row_idx, :row_length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row_idx, :row_length] = 1
    return input_ids, attention_mask, lengths


def encode_token_id_batch_no_truncate(
    tokenizer: Any,
    token_id_rows: list[list[int]],
    max_length: int | None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths: list[int] = []
    for row_idx, ids in enumerate(token_id_rows):
        token_count = len(ids)
        if token_count <= 0:
            raise ValueError(f"source replay row={row_idx} has empty token_ids_prefix")
        if max_length is not None and token_count > max_length:
            raise ValueError(
                f"source replay row={row_idx} token_count={token_count} exceeds --source-max-length={max_length}"
            )
        lengths.append(token_count)

    max_batch_length = max(lengths, default=0)
    input_ids = torch.full((len(token_id_rows), max_batch_length), _pad_token_id(tokenizer), dtype=torch.long)
    attention_mask = torch.zeros((len(token_id_rows), max_batch_length), dtype=torch.long)
    for row_idx, ids in enumerate(token_id_rows):
        row_length = lengths[row_idx]
        input_ids[row_idx, :row_length] = torch.tensor([int(item) for item in ids], dtype=torch.long)
        attention_mask[row_idx, :row_length] = 1
    return input_ids, attention_mask, lengths


def _model_start_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _move_to_model(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = _model_start_device(model)
    return input_ids.to(device), attention_mask.to(device)


def load_source_replay_specs(
    parquet_path: Path,
    *,
    boundaries: list[int],
    max_records: int,
    source_column: str,
    source_token_ids_column: str,
    source_mode: str,
) -> list[SourceReplaySpec]:
    if pq is None:
        raise RuntimeError("pyarrow is required for --ar-sft-parquet")
    table = pq.read_table(parquet_path)
    names = set(table.column_names)
    has_token_ids = source_token_ids_column in names
    use_token_ids = source_mode == "token_ids" or (source_mode == "auto" and has_token_ids)
    required = {"activation_vector", "activation_layer", "doc_id", "n_raw_tokens"}
    if use_token_ids:
        required.add(source_token_ids_column)
    else:
        required.add(source_column)
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns for source replay: {missing}")

    requested = set(boundaries)
    specs: list[SourceReplaySpec] = []
    rows = table.to_pylist()
    for row_idx, row in enumerate(rows):
        boundary_b = int(row["activation_layer"])
        if boundary_b not in requested:
            continue
        source_text = row.get(source_column) if source_column in names else ""
        source_token_ids = row.get(source_token_ids_column) if use_token_ids else None
        if use_token_ids:
            if not isinstance(source_token_ids, list) or not source_token_ids:
                continue
            source_token_ids = [int(item) for item in source_token_ids]
            if not isinstance(source_text, str):
                source_text = ""
        elif not isinstance(source_text, str) or not source_text:
            continue
        specs.append(
            SourceReplaySpec(
                record_id=str(row.get("doc_id") or f"row_{row_idx}"),
                row_index=row_idx,
                boundary_b=boundary_b,
                source_text=source_text,
                source_token_ids=source_token_ids,
                activation_vector=[float(x) for x in row["activation_vector"]],
                metadata={
                    "doc_id": row.get("doc_id"),
                    "n_raw_tokens": row.get("n_raw_tokens"),
                    "source_column": source_column,
                    "source_token_ids_column": source_token_ids_column if use_token_ids else None,
                    "source_mode": "token_ids" if use_token_ids else "text",
                },
            )
        )
        if len(specs) >= max_records:
            break
    if not specs:
        raise ValueError(f"no rows matched boundaries {boundaries} with source column {source_column!r}")
    return specs


def materialize_source_replay_examples(
    *,
    model: Any,
    tokenizer: Any,
    specs: list[SourceReplaySpec],
    source_max_length: int | None,
    source_feature_batch_size: int,
    hidden_size: int,
    tau: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if source_feature_batch_size <= 0:
        raise ValueError("source_feature_batch_size must be positive")
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        start = 0
        while start < len(specs):
            boundary_b = specs[start].boundary_b
            end = start
            while (
                end < len(specs)
                and end - start < source_feature_batch_size
                and specs[end].boundary_b == boundary_b
            ):
                end += 1
            batch_specs = specs[start:end]
            if all(spec.source_token_ids is not None for spec in batch_specs):
                input_ids, attention_mask, lengths = encode_token_id_batch_no_truncate(
                    tokenizer,
                    [spec.source_token_ids or [] for spec in batch_specs],
                    source_max_length,
                )
            else:
                input_ids, attention_mask, lengths = encode_batch_no_truncate(
                    tokenizer,
                    [spec.source_text for spec in batch_specs],
                    source_max_length,
                )
            input_ids, attention_mask = _move_to_model(model, input_ids, attention_mask)
            replay_tensor = prefix_forward_to_R_b(
                model,
                input_ids,
                attention_mask,
                boundary_b=boundary_b,
            )
            batch_features = select_token_vectors_by_lengths(replay_tensor, lengths, tau).detach().float().cpu()
            for spec, feature, token_count in zip(batch_specs, batch_features, lengths, strict=True):
                target = torch.tensor(spec.activation_vector, dtype=torch.float32)
                if int(target.numel()) != hidden_size:
                    raise ValueError(
                        f"{spec.record_id} activation_vector has length {int(target.numel())}, expected hidden_size={hidden_size}"
                    )
                stored_tokens = spec.metadata.get("n_raw_tokens")
                features.append(feature)
                targets.append(target)
                records.append(
                    {
                        "record_id": spec.record_id,
                        "row_index": spec.row_index,
                        "boundary_b": spec.boundary_b,
                        "doc_id": spec.metadata.get("doc_id"),
                        "source_column": spec.metadata.get("source_column"),
                        "source_token_ids_column": spec.metadata.get("source_token_ids_column"),
                        "source_mode": spec.metadata.get("source_mode"),
                        "source_sha256": hashlib.sha256(spec.source_text.encode()).hexdigest(),
                        "source_preview": spec.source_text[:160],
                        "source_token_ids_sha256": (
                            hashlib.sha256(json.dumps(spec.source_token_ids).encode()).hexdigest()
                            if spec.source_token_ids is not None
                            else None
                        ),
                        "stored_n_raw_tokens": int(stored_tokens) if stored_tokens is not None else None,
                        "replay_token_count": int(token_count),
                        "token_count_delta": (int(token_count) - int(stored_tokens)) if stored_tokens is not None else None,
                        "target_l2": float(target.norm().item()),
                        "feature_l2": float(feature.norm().item()),
                    }
                )
            start = end
    return torch.stack(features, dim=0), torch.stack(targets, dim=0), records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.environ.get("NANO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--model-revision", default=os.environ.get("NANO_MODEL_REVISION"))
    parser.add_argument("--tokenizer-revision", default=os.environ.get("NANO_TOKENIZER_REVISION"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    add_bool_optional_arg(parser, "--trust-remote-code", default=True)
    parser.add_argument("--ar-sft-parquet", type=Path, required=True)
    parser.add_argument("--boundaries", type=parse_boundaries, default=[34])
    parser.add_argument("--source-column", default="detokenized_text_truncated")
    parser.add_argument("--source-token-ids-column", default="token_ids_prefix")
    parser.add_argument(
        "--source-mode",
        choices=("auto", "text", "token_ids"),
        default="auto",
        help="Use exact token IDs when present, otherwise replay by re-tokenizing source text.",
    )
    parser.add_argument("--source-max-length", type=int, default=2048)
    parser.add_argument("--source-feature-batch-size", type=int, default=4)
    parser.add_argument("--tau", type=int, default=-1)
    parser.add_argument("--max-records", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    parser.set_defaults(load_mode="full")
    return parser.parse_args(argv)


def payload_base(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "nano_source_replay_probe.v1",
        "run_dir": str(run_dir),
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        },
        "data_source": {
            "path": str(args.ar_sft_parquet),
            "source_column": args.source_column,
            "source_token_ids_column": args.source_token_ids_column,
            "source_mode": args.source_mode,
            "source_max_length": args.source_max_length,
            "source_feature_batch_size": args.source_feature_batch_size,
        },
        "boundary_order": args.boundaries,
        "tau": args.tau,
        "max_records": args.max_records,
        "examples": [],
        "geometry": {},
        "replay_eval": {},
        "passed": False,
        "scientific_passed": False,
        "blockers": [],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = make_run_dir(args.output_root, args.timestamp)
    payload = payload_base(args, run_dir)
    output_path = run_dir / "source_replay_probe.json"

    if torch is None or pq is None:
        missing = [name for name, module in {"torch": torch, "pyarrow": pq}.items() if module is None]
        payload["blockers"] = [{"kind": "environment", "label": "imports", "error": f"missing modules: {missing}"}]
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2

    blockers: list[dict[str, str]] = []
    try:
        torch.manual_seed(args.random_seed)
        tokenizer = load_tokenizer_from_args(args)
        config, config_error = load_config_from_args(args)
        if config_error is not None:
            blockers.append(classify_blocker("remote-code load", config_error))
        specs = load_source_replay_specs(
            args.ar_sft_parquet,
            boundaries=args.boundaries,
            max_records=args.max_records,
            source_column=args.source_column,
            source_token_ids_column=args.source_token_ids_column,
            source_mode=args.source_mode,
        )
    except Exception as exc:
        blockers.append(classify_blocker("source replay setup", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))
        payload["blockers"] = blockers
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2

    try:
        model = load_model_from_args(args, config)
    except Exception as exc:
        blockers.append(classify_blocker("model load", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        payload["blockers"] = blockers
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2

    metadata = build_metadata_record(
        args,
        tokenizer=tokenizer,
        config=config,
        model=model,
        blockers=blockers,
        run_dir=run_dir,
    )
    write_json(run_dir / "metadata.json", metadata)
    resolved = resolve_nano_module_paths(model)
    hidden_size = int(get_config_value(config, "hidden_size"))
    payload["model"].update(
        {
            "hidden_size": hidden_size,
            "block_count": get_config_value(config, "num_hidden_layers"),
            "block_pattern": block_pattern_from_config(config, resolved["layers"].obj),
        }
    )

    try:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        features, targets, records = materialize_source_replay_examples(
            model=model,
            tokenizer=tokenizer,
            specs=specs,
            source_max_length=args.source_max_length,
            source_feature_batch_size=args.source_feature_batch_size,
            hidden_size=hidden_size,
            tau=args.tau,
        )
        payload["examples"] = records
        payload["geometry"] = target_geometry_summary(targets)
        payload["replay_eval"] = replay_metric_bundle(features, targets, records, seed=args.random_seed)
        exact = payload["replay_eval"]["exact_token_count"]
        exact_correct = exact["correct"]
        payload["scientific_passed"] = bool(
            exact["count"] > 0
            and exact_correct is not None
            and exact_correct["cosine_mean"] >= 0.98
            and exact_correct["normalized_mse"] <= 0.05
        )
        payload["passed"] = not blockers
        if exact["count"] == 0:
            blockers.append(
                {
                    "kind": "data",
                    "label": "source-token replay",
                    "error": "no rows had matching stored n_raw_tokens and replay token count; exact source-token identity is not testable from this parquet",
                }
            )
    except Exception as exc:
        blockers.append(classify_blocker("source replay", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))

    payload["blockers"] = blockers
    if blockers:
        payload["passed"] = False
    write_json(output_path, payload)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
    print(f"\nwrote {output_path}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
