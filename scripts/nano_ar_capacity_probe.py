#!/usr/bin/env python3
"""Nano reference-style AR critic capacity probe.

This is a bounded diagnostic between the frozen-linear AR smoke and full AR
SFT/PEFT. It keeps the NLA critic contract:

    critic_prompt(z) -> Nano prefix to R_b -> value head -> h_hat_b

but allows a small trainable tail of the Nano prefix before R_b. The lower
prefix is cached once with frozen weights, then only the last N blocks plus the
value head are optimized. This tests whether the head-only failure is a capacity
limit without starting LoRA, PEFT, serving, RL, or full-scale training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, NamedTuple

try:
    import torch
except ModuleNotFoundError:
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_ar_frozen_baseline import (  # noqa: E402
    DEFAULT_CRITIC_TEMPLATE,
    ValueHead,
    centered_raw_diagnostics,
    load_parquet_ar_specs,
    mean_target_metrics,
    normalized_vector_mse,
    parquet_spec_payload,
    random_matched_norm_targets,
    relative_reconstruction_improvement,
    select_token_vectors_by_lengths,
    split_indices,
    split_metadata,
    take_rows,
    vector_metrics,
)
from nano_extraction_identity import _layer_mask_for_block, parse_boundaries, prefix_forward_to_R_b  # noqa: E402
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
from nano_wandb import add_wandb_args, init_wandb  # noqa: E402


class CachedARBatch(NamedTuple):
    prefix_states: list[torch.Tensor]
    targets: torch.Tensor
    records: list[dict[str, Any]]


def reference_layer_metadata(boundary_b: int, block_count: int) -> dict[str, int | None]:
    if not 0 <= boundary_b <= block_count:
        raise ValueError(f"boundary_b={boundary_b} out of range for block_count={block_count}")
    return {
        "boundary_b": int(boundary_b),
        "zero_based_last_block_index": int(boundary_b - 1) if boundary_b > 0 else None,
        "reference_nla_layer_index_K": int(boundary_b - 1) if boundary_b > 0 else None,
        "reference_critic_num_hidden_layers": int(boundary_b),
        "output_hidden_states_index_for_R_b": int(boundary_b),
    }


def tail_start_boundary(boundary_b: int, train_tail_blocks: int) -> int:
    if train_tail_blocks < 0:
        raise ValueError("train_tail_blocks must be non-negative")
    if train_tail_blocks > boundary_b:
        raise ValueError(f"train_tail_blocks={train_tail_blocks} exceeds boundary_b={boundary_b}")
    return int(boundary_b - train_tail_blocks)


def freeze_for_tail_training(
    *,
    model: Any,
    head: Any,
    prefix_start_b: int,
    boundary_b: int,
) -> dict[str, Any]:
    resolved = resolve_nano_module_paths(model)
    layers = resolved["layers"].obj
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(True)

    trainable_layer_indices: list[int] = []
    trainable_layer_paths: list[str] = []
    for layer_idx in range(prefix_start_b, boundary_b):
        trainable_layer_indices.append(layer_idx)
        trainable_layer_paths.append(f"{resolved['layers'].path}.{layer_idx}")
        for parameter in layers[layer_idx].parameters():
            parameter.requires_grad_(True)

    return {
        "prefix_start_b": int(prefix_start_b),
        "trainable_tail_blocks": int(boundary_b - prefix_start_b),
        "trainable_layer_indices": trainable_layer_indices,
        "trainable_layer_paths": trainable_layer_paths,
        "nano_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "nano_total_parameters": int(sum(p.numel() for p in model.parameters())),
        "head_trainable_parameters": int(sum(p.numel() for p in head.parameters() if p.requires_grad)),
        "head_total_parameters": int(sum(p.numel() for p in head.parameters())),
    }


def _pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def encode_prompt_ids_no_truncate(tokenizer: Any, text: str, max_length: int | None) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(item) for item in ids]
    if max_length is not None and len(ids) > max_length:
        raise ValueError(
            f"AR prompt token_count={len(ids)} exceeds --ar-prompt-max-length={max_length}; "
            "do not truncate because the critic extraction point is the final real token"
        )
    if not ids:
        raise ValueError("AR prompt tokenized to zero tokens")
    return ids


def pad_token_ids(tokenizer: Any, token_rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [len(row) for row in token_rows]
    max_len = max(lengths)
    input_ids = torch.full((len(token_rows), max_len), _pad_token_id(tokenizer), dtype=torch.long)
    attention_mask = torch.zeros((len(token_rows), max_len), dtype=torch.long)
    for row_idx, ids in enumerate(token_rows):
        row_len = lengths[row_idx]
        input_ids[row_idx, :row_len] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row_idx, :row_len] = 1
    return input_ids, attention_mask, lengths


def _model_start_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _module_device_dtype(module: Any) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


def materialize_prefix_cache(
    *,
    model: Any,
    tokenizer: Any,
    specs: list[Any],
    prefix_start_b: int,
    ar_prompt_max_length: int | None,
    cache_batch_size: int,
    hidden_size: int,
) -> CachedARBatch:
    if cache_batch_size <= 0:
        raise ValueError("cache_batch_size must be positive")
    resolved = resolve_nano_module_paths(model)
    embeddings = resolved["embeddings"].obj
    prefix_states: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []

    token_rows = [encode_prompt_ids_no_truncate(tokenizer, spec.prompt, ar_prompt_max_length) for spec in specs]
    with torch.no_grad():
        for start in range(0, len(specs), cache_batch_size):
            batch_specs = specs[start : start + cache_batch_size]
            batch_token_rows = token_rows[start : start + cache_batch_size]
            input_ids, attention_mask, lengths = pad_token_ids(tokenizer, batch_token_rows)
            input_ids = input_ids.to(_model_start_device(model))
            attention_mask = attention_mask.to(input_ids.device)
            if prefix_start_b == 0:
                cached = embeddings(input_ids)
            else:
                cached = prefix_forward_to_R_b(
                    model,
                    input_ids,
                    attention_mask,
                    boundary_b=prefix_start_b,
                )
            for spec, row_hidden, token_ids, token_count in zip(batch_specs, cached, batch_token_rows, lengths, strict=True):
                target = torch.tensor(spec.activation_vector, dtype=torch.float32)
                if int(target.numel()) != hidden_size:
                    raise ValueError(
                        f"{spec.record_id} activation_vector has length {int(target.numel())}, expected hidden_size={hidden_size}"
                    )
                prefix_states.append(row_hidden[:token_count].detach().float().cpu().clone())
                targets.append(target)
                records.append(
                    {
                        **parquet_spec_payload(spec),
                        "prompt_token_count": int(token_count),
                        "prompt_token_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                        "prefix_start_b": int(prefix_start_b),
                        "cached_prefix_shape": [int(token_count), int(hidden_size)],
                    }
                )
    return CachedARBatch(prefix_states=prefix_states, targets=torch.stack(targets, dim=0), records=records)


def pad_prefix_states(
    prefix_states: list[torch.Tensor],
    indices: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lengths = [int(prefix_states[idx].shape[0]) for idx in indices]
    max_len = max(lengths)
    hidden_size = int(prefix_states[indices[0]].shape[-1])
    hidden = torch.zeros((len(indices), max_len, hidden_size), device=device, dtype=dtype)
    attention_mask = torch.zeros((len(indices), max_len), device=device, dtype=torch.long)
    for row_idx, source_idx in enumerate(indices):
        row = prefix_states[source_idx].to(device=device, dtype=dtype)
        row_len = lengths[row_idx]
        hidden[row_idx, :row_len] = row
        attention_mask[row_idx, :row_len] = 1
    return hidden, attention_mask, lengths


def forward_tail_from_cache(
    *,
    model: Any,
    head: Any,
    prefix_states: list[torch.Tensor],
    indices: list[int],
    prefix_start_b: int,
    boundary_b: int,
    tau: int,
) -> torch.Tensor:
    resolved = resolve_nano_module_paths(model)
    backbone = resolved["backbone"].obj
    layers = resolved["layers"].obj
    if prefix_start_b < boundary_b:
        device, dtype = _module_device_dtype(layers[prefix_start_b])
    else:
        device, dtype = _module_device_dtype(head)
    hidden_states, attention_mask, lengths = pad_prefix_states(prefix_states, indices, device=device, dtype=dtype)
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

    for layer_idx in range(prefix_start_b, boundary_b):
        block = layers[layer_idx]
        layer_mask = _layer_mask_for_block(backbone, block, attention_mask, hidden_states, cache_position)
        output = block(
            hidden_states,
            cache_params=None,
            cache_position=cache_position,
            attention_mask=layer_mask,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output

    selected = select_token_vectors_by_lengths(hidden_states, lengths, tau).float()
    return head(selected)


def predict_all(
    *,
    model: Any,
    head: Any,
    cache: CachedARBatch,
    indices: list[int],
    prefix_start_b: int,
    boundary_b: int,
    batch_size: int,
    tau: int,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    model.eval()
    head.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            pred = forward_tail_from_cache(
                model=model,
                head=head,
                prefix_states=cache.prefix_states,
                indices=batch_indices,
                prefix_start_b=prefix_start_b,
                boundary_b=boundary_b,
                tau=tau,
            )
            predictions.append(pred.detach().float().cpu())
    return torch.cat(predictions, dim=0) if predictions else torch.zeros((0, cache.targets.shape[-1]))


def train_capacity_probe(
    *,
    model: Any,
    head: Any,
    cache: CachedARBatch,
    train_indices: list[int],
    prefix_start_b: int,
    boundary_b: int,
    batch_size: int,
    tau: int,
    max_steps: int,
    head_lr: float,
    tail_lr: float,
    weight_decay: float,
    seed: int,
    log_every: int,
) -> list[dict[str, float | int]]:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = [{"params": list(head.parameters()), "lr": head_lr, "weight_decay": weight_decay}]
    if trainable_params:
        param_groups.append({"params": trainable_params, "lr": tail_lr, "weight_decay": weight_decay})
    optimizer = torch.optim.AdamW(param_groups)
    rng = random.Random(seed)
    history: list[dict[str, float | int]] = []

    def eval_train_loss(step: int) -> None:
        pred = predict_all(
            model=model,
            head=head,
            cache=cache,
            indices=train_indices,
            prefix_start_b=prefix_start_b,
            boundary_b=boundary_b,
            batch_size=batch_size,
            tau=tau,
        )
        target = take_rows(cache.targets, train_indices)
        loss = normalized_vector_mse(pred, target)
        history.append({"step": int(step), "train_normalized_mse": float(loss.item())})

    eval_train_loss(0)
    if max_steps == 0:
        return history

    model.eval()
    head.train()
    for step in range(1, max_steps + 1):
        batch_indices = [rng.choice(train_indices) for _ in range(min(batch_size, len(train_indices)))]
        pred = forward_tail_from_cache(
            model=model,
            head=head,
            prefix_states=cache.prefix_states,
            indices=batch_indices,
            prefix_start_b=prefix_start_b,
            boundary_b=boundary_b,
            tau=tau,
        )
        target = take_rows(cache.targets, batch_indices).to(pred.device)
        loss = normalized_vector_mse(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == max_steps or step % max(1, log_every) == 0:
            eval_train_loss(step)
            model.eval()
            head.train()
    return history


def prediction_control_eval(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    train_targets_for_mean: torch.Tensor,
    seed: int,
    mse_margin: float,
    cosine_margin: float,
    min_rri: float,
) -> dict[str, Any]:
    if predictions.shape[0] == 0:
        return {
            "count": 0,
            "correct": None,
            "shuffled": None,
            "random_matched_norm": None,
            "mean_train": None,
            "correct_beats_mean": None,
            "correct_beats_controls": None,
        }
    correct = vector_metrics(predictions, targets)
    correct.update(centered_raw_diagnostics(predictions, targets, train_targets_for_mean))
    shuffled = None
    if targets.shape[0] > 1:
        shuffled_targets = targets.roll(shifts=1, dims=0)
        shuffled = vector_metrics(predictions, shuffled_targets)
        shuffled.update(centered_raw_diagnostics(predictions, shuffled_targets, train_targets_for_mean))
    random_targets = random_matched_norm_targets(targets, seed=seed)
    random_metrics = vector_metrics(predictions, random_targets)
    random_metrics.update(centered_raw_diagnostics(predictions, random_targets, train_targets_for_mean))
    mean_train = mean_target_metrics(targets, train_targets_for_mean)
    correct["rri_vs_train_mean"] = relative_reconstruction_improvement(
        correct["normalized_mse"],
        mean_train["normalized_mse"],
    )
    correct_beats_mean = (
        correct["rri_vs_train_mean"] is not None
        and correct["normalized_mse"] <= mean_train["normalized_mse"] - mse_margin
        and correct["rri_vs_train_mean"] >= min_rri
    )
    beats_shuffled = (
        shuffled is not None
        and correct["normalized_mse"] <= shuffled["normalized_mse"] - mse_margin
        and correct["cosine_mean"] >= shuffled["cosine_mean"] + cosine_margin
    )
    beats_random = (
        correct["normalized_mse"] <= random_metrics["normalized_mse"] - mse_margin
        and correct["cosine_mean"] >= random_metrics["cosine_mean"] + cosine_margin
    )
    return {
        "count": int(predictions.shape[0]),
        "correct": correct,
        "shuffled": shuffled,
        "random_matched_norm": random_metrics,
        "mean_train": mean_train,
        "correct_beats_mean": correct_beats_mean,
        "correct_beats_controls": bool(beats_shuffled and beats_random and correct_beats_mean),
    }


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
    parser.add_argument("--max-records", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--split-strategy", choices=("sequential", "alternating", "random", "doc_random"), default="doc_random")
    parser.add_argument("--ar-prompt-max-length", type=int, default=512)
    parser.add_argument("--cache-prefix-batch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ar-tau", type=int, default=-1)
    parser.add_argument("--train-tail-blocks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--tail-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--mse-margin", type=float, default=0.05)
    parser.add_argument("--cosine-margin", type=float, default=0.02)
    parser.add_argument("--min-rri", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    add_wandb_args(parser)
    parser.set_defaults(load_mode="full")
    return parser.parse_args(argv)


def payload_base(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "nano_ar_capacity_probe.v1",
        "run_dir": str(run_dir),
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        },
        "data_source": {
            "kind": "ar_sft_parquet",
            "path": str(args.ar_sft_parquet),
            "ar_prompt_max_length": args.ar_prompt_max_length,
        },
        "boundary_order": args.boundaries,
        "max_records": args.max_records,
        "train_fraction": args.train_fraction,
        "split_strategy": args.split_strategy,
        "train_tail_blocks": args.train_tail_blocks,
        "training": {},
        "eval": {},
        "examples": [],
        "passed": False,
        "scientific_passed": False,
        "blockers": [],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = make_run_dir(args.output_root, args.timestamp)
    payload = payload_base(args, run_dir)
    output_path = run_dir / "ar_capacity_probe.json"
    tracker = init_wandb(
        args,
        run_dir=run_dir,
        job_type="ar_capacity_probe",
        config=json_safe({"args": vars(args), "run_dir": run_dir}),
    )
    payload["wandb"] = tracker.metadata

    if torch is None:
        payload["blockers"] = [{"kind": "environment", "label": "torch import", "error": "PyTorch is required"}]
        tracker.log_summary(payload)
        tracker.finish({"status/passed": False, "status/blockers": len(payload["blockers"])})
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2
    if len(args.boundaries) != 1:
        payload["blockers"] = [{"kind": "configuration", "label": "boundaries", "error": "capacity probe currently accepts exactly one boundary"}]
        tracker.log_summary(payload)
        tracker.finish({"status/passed": False, "status/blockers": len(payload["blockers"])})
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2
    if args.max_records <= 0 or args.max_records > 2048:
        payload["blockers"] = [{"kind": "configuration", "label": "max_records", "error": "max_records must be in [1, 2048]"}]
        tracker.log_summary(payload)
        tracker.finish({"status/passed": False, "status/blockers": len(payload["blockers"])})
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2

    blockers: list[dict[str, str]] = []
    boundary_b = int(args.boundaries[0])
    try:
        torch.manual_seed(args.random_seed)
        tokenizer = load_tokenizer_from_args(args)
        config, config_error = load_config_from_args(args)
        if config_error is not None:
            blockers.append(classify_blocker("remote-code load", config_error))
        specs = load_parquet_ar_specs(
            args.ar_sft_parquet,
            boundaries=[boundary_b],
            max_records=args.max_records,
        )
    except Exception as exc:
        blockers.append(classify_blocker("setup", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))
        payload["blockers"] = blockers
        tracker.log_summary(payload)
        tracker.finish({"status/passed": False, "status/blockers": len(payload["blockers"])})
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        return 2

    try:
        model = load_model_from_args(args, config)
    except Exception as exc:
        blockers.append(classify_blocker("model load", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        payload["blockers"] = blockers
        tracker.log_summary(payload)
        tracker.finish({"status/passed": False, "status/blockers": len(payload["blockers"])})
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

    try:
        resolved = resolve_nano_module_paths(model)
        layers = resolved["layers"].obj
        hidden_size = int(get_config_value(config, "hidden_size"))
        block_count = len(layers)
        prefix_start_b = tail_start_boundary(boundary_b, args.train_tail_blocks)
        layer_meta = reference_layer_metadata(boundary_b, block_count)
        payload["model"].update(
            {
                "hidden_size": hidden_size,
                "block_count": get_config_value(config, "num_hidden_layers"),
                "block_pattern": block_pattern_from_config(config, layers),
            }
        )
        payload["reference_layer_metadata"] = layer_meta
        payload["specs"] = [parquet_spec_payload(spec) for spec in specs]

        model.eval()
        head = ValueHead(hidden_size=hidden_size)
        if prefix_start_b < boundary_b:
            head_device = next(layers[boundary_b - 1].parameters()).device
        else:
            head_device = _model_start_device(model)
        head.to(device=head_device, dtype=torch.float32)
        trainability = freeze_for_tail_training(
            model=model,
            head=head,
            prefix_start_b=prefix_start_b,
            boundary_b=boundary_b,
        )
        payload["trainability"] = trainability

        cache = materialize_prefix_cache(
            model=model,
            tokenizer=tokenizer,
            specs=specs,
            prefix_start_b=prefix_start_b,
            ar_prompt_max_length=args.ar_prompt_max_length,
            cache_batch_size=args.cache_prefix_batch_size,
            hidden_size=hidden_size,
        )
        payload["examples"] = cache.records
        payload["prefix_cache"] = {
            "count": len(cache.records),
            "prefix_start_b": prefix_start_b,
            "target_l2_mean": float(cache.targets.float().norm(dim=-1).mean().item()),
            "prompt_token_count_min": min(record["prompt_token_count"] for record in cache.records),
            "prompt_token_count_max": max(record["prompt_token_count"] for record in cache.records),
        }

        train_indices, eval_indices = split_indices(
            len(cache.records),
            args.train_fraction,
            strategy=args.split_strategy,
            seed=args.random_seed,
            records=cache.records,
        )
        train_targets = take_rows(cache.targets, train_indices)
        eval_targets = take_rows(cache.targets, eval_indices)

        train_before_pred = predict_all(
            model=model,
            head=head,
            cache=cache,
            indices=train_indices,
            prefix_start_b=prefix_start_b,
            boundary_b=boundary_b,
            batch_size=args.batch_size,
            tau=args.ar_tau,
        )
        train_before = vector_metrics(train_before_pred, train_targets)
        history = train_capacity_probe(
            model=model,
            head=head,
            cache=cache,
            train_indices=train_indices,
            prefix_start_b=prefix_start_b,
            boundary_b=boundary_b,
            batch_size=args.batch_size,
            tau=args.ar_tau,
            max_steps=args.max_steps,
            head_lr=args.head_lr,
            tail_lr=args.tail_lr,
            weight_decay=args.weight_decay,
            seed=args.random_seed,
            log_every=args.log_every,
        )
        tracker.log_history(history, prefix="train")
        train_after_pred = predict_all(
            model=model,
            head=head,
            cache=cache,
            indices=train_indices,
            prefix_start_b=prefix_start_b,
            boundary_b=boundary_b,
            batch_size=args.batch_size,
            tau=args.ar_tau,
        )
        eval_pred = predict_all(
            model=model,
            head=head,
            cache=cache,
            indices=eval_indices,
            prefix_start_b=prefix_start_b,
            boundary_b=boundary_b,
            batch_size=args.batch_size,
            tau=args.ar_tau,
        )
        train_after = vector_metrics(train_after_pred, train_targets)
        train_loss_decreased = train_after["normalized_mse"] < train_before["normalized_mse"]
        payload["training"] = {
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "head_lr": args.head_lr,
            "tail_lr": args.tail_lr,
            "weight_decay": args.weight_decay,
            "history": history,
            "train_before": train_before,
            "train_after": train_after,
            "train_loss_decreased": train_loss_decreased,
            "split_metadata": split_metadata(cache.records, train_indices, eval_indices),
        }
        payload["eval"] = {
            "train_controls": prediction_control_eval(
                train_after_pred,
                train_targets,
                train_targets_for_mean=train_targets,
                seed=args.random_seed + 1,
                mse_margin=args.mse_margin,
                cosine_margin=args.cosine_margin,
                min_rri=args.min_rri,
            ),
            "heldout_controls": prediction_control_eval(
                eval_pred,
                eval_targets,
                train_targets_for_mean=train_targets,
                seed=args.random_seed + 2,
                mse_margin=args.mse_margin,
                cosine_margin=args.cosine_margin,
                min_rri=args.min_rri,
            ),
        }
        heldout = payload["eval"]["heldout_controls"]
        payload["scientific_passed"] = bool(heldout["correct_beats_controls"]) if heldout["correct_beats_controls"] is not None else False
        payload["passed"] = not blockers and bool(train_loss_decreased) and len(cache.records) > 0
    except Exception as exc:
        blockers.append(classify_blocker("capacity probe", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))

    payload["blockers"] = blockers
    if blockers:
        payload["passed"] = False
    tracker.log_summary(payload)
    tracker.finish({"status/passed": bool(payload["passed"]), "status/blockers": len(payload["blockers"])})
    write_json(output_path, payload)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
    print(f"\nwrote {output_path}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
