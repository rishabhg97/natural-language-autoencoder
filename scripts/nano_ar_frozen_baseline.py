#!/usr/bin/env python3
"""Nano30B frozen-prefix AR value-head baseline.

This is the first constrained training harness after the Nano extraction and
injection probes. It freezes Nano, builds a tiny deterministic AR smoke set,
runs explanation text z through the exact Nano prefix to R_b, and trains only a
small value head to predict the extracted target residual h_b.

It does not train Nano, run PEFT/LoRA, train AV, serve, run RL, or generate a
large dataset. The explanation text is a candidate AR input, not ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from nano_extraction_identity import build_prompt_inputs, parse_boundaries, prefix_forward_to_R_b  # noqa: E402
from nano_extraction_serialize_probe import DEFAULT_PROMPT_NAMES, parse_prompt_names  # noqa: E402
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
from nano_track_a_probe import select_token_vector  # noqa: E402
from nano_wandb import add_wandb_args, init_wandb  # noqa: E402


DEFAULT_CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
EXPLANATION_TEMPLATE_CHOICES = ("generic", "prompt_label", "boundary_prompt_key")
SPLIT_STRATEGY_CHOICES = ("sequential", "alternating", "random", "doc_random")


class TinyARSpec(NamedTuple):
    record_id: str
    source_prompt_name: str
    boundary_b: int
    explanation_text: str


class ParquetARSpec(NamedTuple):
    record_id: str
    boundary_b: int
    prompt: str
    activation_vector: list[float]
    metadata: dict[str, Any]


class ValueHead(torch.nn.Module if torch is not None else object):
    """Small AR head trained on frozen Nano prefix features."""

    def __init__(self, hidden_size: int):
        if torch is None:
            raise RuntimeError("ValueHead requires PyTorch")
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


def freeze_module(module: Any) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def trainable_parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def total_parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def l2_normalize_rows(tensor: torch.Tensor, target_scale: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    tensor_f = tensor.float()
    return target_scale * tensor_f / tensor_f.norm(dim=-1, keepdim=True).clamp_min(eps)


def normalized_vector_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_scale = math.sqrt(float(prediction.shape[-1]))
    return torch.nn.functional.mse_loss(
        l2_normalize_rows(prediction, target_scale=target_scale),
        l2_normalize_rows(target, target_scale=target_scale),
    )


def cosine_mean(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_n = l2_normalize_rows(prediction)
    target_n = l2_normalize_rows(target)
    return float((pred_n * target_n).sum(dim=-1).mean().item())


def vector_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    normalized_mse = normalized_vector_mse(predictions.float(), targets.float())
    raw_mse = torch.nn.functional.mse_loss(predictions.float(), targets.float())
    return {
        "normalized_mse": float(normalized_mse.item()),
        "raw_mse": float(raw_mse.item()),
        "cosine_mean": cosine_mean(predictions, targets.float()),
    }


def centered_raw_diagnostics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    train_targets_for_mean: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, float | None]:
    train_mean = train_targets_for_mean.float().mean(dim=0, keepdim=True)
    targets_f = targets.float()
    predictions_f = predictions.float()
    sse = float((predictions_f - targets_f).pow(2).sum().item())
    sst = float((targets_f - train_mean).pow(2).sum().item())
    r2 = None if sst <= eps else float(1.0 - sse / sst)
    return {
        "centered_raw_r2": r2,
        "centered_raw_sse": sse,
        "centered_raw_sst": sst,
        "train_mean_l2": float(train_mean.norm().item()),
    }


def evaluate_head(
    head: ValueHead,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    train_targets_for_center: torch.Tensor | None = None,
    batch_size: int | None = None,
) -> dict[str, float | None]:
    head.eval()
    predictions = predict_head(head, features, batch_size=batch_size)
    metrics = vector_metrics(predictions, targets)
    if train_targets_for_center is not None:
        metrics.update(centered_raw_diagnostics(predictions, targets, train_targets_for_center))
    return metrics


def _head_device(head: ValueHead) -> torch.device:
    return next(head.parameters()).device


def resolve_train_device(train_device: str) -> torch.device:
    if train_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(train_device)


def predict_head(head: ValueHead, features: torch.Tensor, *, batch_size: int | None = None) -> torch.Tensor:
    head.eval()
    device = _head_device(head)
    if batch_size is None or batch_size <= 0 or batch_size >= int(features.shape[0]):
        with torch.no_grad():
            return head(features.float().to(device)).detach().float().cpu()

    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), batch_size):
            batch = features[start : start + batch_size].float().to(device)
            predictions.append(head(batch).detach().float().cpu())
    return torch.cat(predictions, dim=0) if predictions else features.new_zeros((0, features.shape[-1]))


def fit_ridge_map(features: torch.Tensor, targets: torch.Tensor, *, ridge_alpha: float) -> torch.Tensor:
    """Closed-form row-vector ridge map W for predictions = features @ W."""
    if torch is None:
        raise RuntimeError("fit_ridge_map requires PyTorch")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    x = features.float()
    y = targets.float()
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("features and targets must be 2D")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row mismatch: features={tuple(x.shape)} targets={tuple(y.shape)}")
    xtx = x.T @ x
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
    return torch.linalg.solve(xtx + float(ridge_alpha) * eye, x.T @ y)


def fit_procrustes_map(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Orthogonal row-vector map W minimizing ||features @ W - targets||."""
    if torch is None:
        raise RuntimeError("fit_procrustes_map requires PyTorch")
    x = features.float()
    y = targets.float()
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("features and targets must be 2D")
    if x.shape != y.shape:
        raise ValueError(f"Procrustes requires matching shapes, got {tuple(x.shape)} and {tuple(y.shape)}")
    u, _, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
    return u @ vh


def predict_linear_map(
    features: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("predict_linear_map requires PyTorch")
    predictions = features.float() @ weight.float()
    if bias is not None:
        predictions = predictions + bias.float()
    return predictions


def closed_form_report_row(
    *,
    method: str,
    readout_mode: str,
    split: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": method,
        "readout_mode": readout_mode,
        "split": split,
        "metrics": metrics,
    }


def train_value_head(
    head: ValueHead,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_steps: int,
    lr: float,
    weight_decay: float = 0.0,
    log_every: int = 10,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    seed: int = 0,
    eval_batch_size: int | None = None,
) -> list[dict[str, float | int]]:
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if device is not None:
        head.to(torch.device(device))
    train_device = _head_device(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float | int]] = []
    features_f = features.float()
    targets_f = targets.float()
    row_count = int(features_f.shape[0])
    use_minibatch = batch_size is not None and batch_size > 0 and batch_size < row_count
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def record(step: int, loss: torch.Tensor) -> None:
        history.append({"step": int(step), "loss": float(loss.detach().item())})

    def train_loss() -> torch.Tensor:
        predictions = predict_head(head, features_f, batch_size=eval_batch_size)
        return normalized_vector_mse(predictions, targets_f)

    head.train()
    with torch.no_grad():
        initial = train_loss()
    record(0, initial)

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if use_minibatch:
            indices = torch.randint(row_count, (int(batch_size),), generator=generator)
            batch_features = features_f[indices].to(train_device)
            batch_targets = targets_f[indices].to(train_device)
        else:
            batch_features = features_f.to(train_device)
            batch_targets = targets_f.to(train_device)
        predictions = head(batch_features)
        loss = normalized_vector_mse(predictions, batch_targets)
        loss.backward()
        optimizer.step()
        if step == max_steps or step % max(1, log_every) == 0:
            with torch.no_grad():
                record(step, train_loss())
            head.train()
    return history


def build_explanation_body(prompt_name: str, boundary_b: int, explanation_template: str) -> str:
    if explanation_template == "generic":
        return (
            f"Candidate Nano AR smoke description for residual boundary R_{boundary_b}. "
            f"It identifies prompt mode {prompt_name} as a boundary diagnostic, not source context."
        )
    if explanation_template == "prompt_label":
        return f"Nano AR smoke summary for prompt mode {prompt_name} at residual boundary R_{boundary_b}."
    if explanation_template == "boundary_prompt_key":
        return f"R_{boundary_b} prompt_mode_{prompt_name} Nano residual summary key."
    raise ValueError(f"unknown explanation_template={explanation_template!r}")


def format_critic_prompt(explanation: str, critic_template: str) -> str:
    if "{explanation}" not in critic_template:
        raise ValueError("critic_template must contain {explanation}")
    return critic_template.format(explanation=explanation)


def build_tiny_ar_specs(
    prompts: list[Any],
    *,
    boundaries: list[int],
    max_records: int,
    explanation_template: str = "generic",
    critic_template: str = DEFAULT_CRITIC_TEMPLATE,
) -> list[TinyARSpec]:
    specs: list[TinyARSpec] = []
    for boundary_b in boundaries:
        for prompt in prompts:
            explanation_body = build_explanation_body(
                prompt_name=prompt.name,
                boundary_b=boundary_b,
                explanation_template=explanation_template,
            )
            explanation = format_critic_prompt(explanation_body, critic_template)
            specs.append(
                TinyARSpec(
                    record_id=f"R{boundary_b}_{prompt.name}",
                    source_prompt_name=prompt.name,
                    boundary_b=int(boundary_b),
                    explanation_text=explanation,
                )
            )
            if len(specs) >= max_records:
                return specs
    return specs


def _model_start_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _move_to_model(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = _model_start_device(model)
    return input_ids.to(device), attention_mask.to(device)


def _encode_text(tokenizer: Any, text: str, max_length: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs: dict[str, Any] = {"return_tensors": "pt", "add_special_tokens": True}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})
    encoded = tokenizer(text, **kwargs)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    return input_ids, attention_mask


def _encode_text_no_truncate(tokenizer: Any, text: str, max_length: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"]
    if max_length is not None and int(input_ids.shape[-1]) > max_length:
        raise ValueError(
            f"AR prompt token_count={int(input_ids.shape[-1])} exceeds --ar-prompt-max-length={max_length}; "
            "do not right-truncate AR prompts because the critic suffix anchors extraction at tokens[-1]"
        )
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    return input_ids, attention_mask


def _pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def _encode_text_batch_no_truncate(
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
                f"AR prompt row={row_idx} token_count={token_count} exceeds --ar-prompt-max-length={max_length}; "
                "do not right-truncate AR prompts because the critic suffix anchors extraction at tokens[-1]"
            )
        encoded_ids.append([int(item) for item in ids])
        lengths.append(token_count)

    max_batch_length = max(lengths, default=0)
    if max_batch_length <= 0:
        raise ValueError("cannot encode an empty AR prompt batch")
    input_ids = torch.full((len(encoded_ids), max_batch_length), _pad_token_id(tokenizer), dtype=torch.long)
    attention_mask = torch.zeros((len(encoded_ids), max_batch_length), dtype=torch.long)
    for row_idx, ids in enumerate(encoded_ids):
        row_length = lengths[row_idx]
        input_ids[row_idx, :row_length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row_idx, :row_length] = 1
    return input_ids, attention_mask, lengths


def select_token_vectors_by_lengths(tensor: torch.Tensor, lengths: list[int], tau: int) -> torch.Tensor:
    if tensor.shape[0] != len(lengths):
        raise ValueError(f"tensor batch={tensor.shape[0]} does not match lengths={len(lengths)}")
    rows: list[torch.Tensor] = []
    for row_idx, length in enumerate(lengths):
        if length <= 0:
            raise ValueError(f"row {row_idx} has non-positive token length {length}")
        token_idx = length + tau if tau < 0 else tau
        if token_idx < 0 or token_idx >= length:
            raise ValueError(f"row {row_idx} tau={tau} selects token_idx={token_idx} outside length={length}")
        rows.append(tensor[row_idx, token_idx])
    return torch.stack(rows, dim=0)


def materialize_ar_examples(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[Any],
    specs: list[TinyARSpec],
    source_tau: int,
    ar_tau: int,
    explanation_max_length: int | None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    prompt_by_name = {prompt.name: prompt for prompt in prompts}
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for spec in specs:
            source_prompt = prompt_by_name[spec.source_prompt_name]
            source_ids, source_mask = _move_to_model(model, source_prompt.input_ids, source_prompt.attention_mask)
            source_tensor = prefix_forward_to_R_b(
                model,
                source_ids,
                source_mask,
                boundary_b=spec.boundary_b,
            )
            target = select_token_vector(source_tensor, source_tau).detach().float().cpu()

            ar_ids, ar_mask = _encode_text(tokenizer, spec.explanation_text, explanation_max_length)
            ar_ids, ar_mask = _move_to_model(model, ar_ids, ar_mask)
            ar_tensor = prefix_forward_to_R_b(
                model,
                ar_ids,
                ar_mask,
                boundary_b=spec.boundary_b,
            )
            feature = select_token_vector(ar_tensor, ar_tau).detach().float().cpu()

            features.append(feature)
            targets.append(target)
            records.append(
                {
                    **spec._asdict(),
                    "source_tau": source_tau,
                    "ar_tau": ar_tau,
                    "source_token_count": int(source_tensor.shape[1]),
                    "explanation_token_count": int(ar_tensor.shape[1]),
                    "target_l2": float(target.norm().item()),
                    "feature_l2": float(feature.norm().item()),
                }
            )

    return torch.stack(features, dim=0), torch.stack(targets, dim=0), records


def load_parquet_ar_specs(
    parquet_path: Path,
    *,
    boundaries: list[int],
    max_records: int,
) -> list[ParquetARSpec]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for --ar-sft-parquet") from exc

    table = pq.read_table(parquet_path)
    names = set(table.column_names)
    required = {"prompt", "activation_vector"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {missing}")

    prompts = table.column("prompt").to_pylist()
    vectors = table.column("activation_vector").to_pylist()
    activation_layers = table.column("activation_layer").to_pylist() if "activation_layer" in names else [None] * table.num_rows
    doc_ids = table.column("doc_id").to_pylist() if "doc_id" in names else [None] * table.num_rows
    n_raw_tokens = table.column("n_raw_tokens").to_pylist() if "n_raw_tokens" in names else [None] * table.num_rows

    specs: list[ParquetARSpec] = []
    requested = set(boundaries)
    if "activation_layer" not in names and len(requested) != 1:
        raise ValueError("parquet has no activation_layer column; pass exactly one --boundaries value")

    for row_idx, (prompt, vector, activation_layer, doc_id, raw_tokens) in enumerate(
        zip(prompts, vectors, activation_layers, doc_ids, n_raw_tokens, strict=True)
    ):
        boundary_b = int(activation_layer) if activation_layer is not None else boundaries[0]
        if boundary_b not in requested:
            continue
        if not isinstance(prompt, str):
            raise ValueError(f"row {row_idx} prompt must be a string for AR-SFT, got {type(prompt).__name__}")
        specs.append(
            ParquetARSpec(
                record_id=str(doc_id) if doc_id is not None else f"row_{row_idx}",
                boundary_b=boundary_b,
                prompt=prompt,
                activation_vector=[float(x) for x in vector],
                metadata={
                    "row_index": row_idx,
                    "doc_id": doc_id,
                    "n_raw_tokens": raw_tokens,
                    "source": "ar_sft_parquet",
                },
            )
        )
        if len(specs) >= max_records:
            break

    if not specs:
        raise ValueError(f"no AR-SFT rows matched requested boundaries {boundaries} in {parquet_path}")
    return specs


def parquet_spec_payload(spec: ParquetARSpec) -> dict[str, Any]:
    target = torch.tensor(spec.activation_vector, dtype=torch.float32) if torch is not None else None
    return {
        "record_id": spec.record_id,
        "boundary_b": spec.boundary_b,
        "prompt_sha256": hashlib.sha256(spec.prompt.encode()).hexdigest(),
        "prompt_preview": spec.prompt[:160],
        "target_l2": float(target.norm().item()) if target is not None else None,
        **spec.metadata,
    }


def materialize_parquet_ar_examples(
    *,
    model: Any,
    tokenizer: Any,
    specs: list[ParquetARSpec],
    ar_tau: int,
    ar_prompt_max_length: int | None,
    hidden_size: int,
    ar_feature_batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if ar_feature_batch_size <= 0:
        raise ValueError("ar_feature_batch_size must be positive")
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        start = 0
        while start < len(specs):
            batch_boundary = specs[start].boundary_b
            end = start
            while (
                end < len(specs)
                and end - start < ar_feature_batch_size
                and specs[end].boundary_b == batch_boundary
            ):
                end += 1
            batch_specs = specs[start:end]
            batch_targets: list[torch.Tensor] = []
            for spec in batch_specs:
                target = torch.tensor(spec.activation_vector, dtype=torch.float32)
                if int(target.numel()) != hidden_size:
                    raise ValueError(
                        f"{spec.record_id} activation_vector has length {int(target.numel())}, expected hidden_size={hidden_size}"
                    )
                batch_targets.append(target)

            ar_ids, ar_mask, lengths = _encode_text_batch_no_truncate(
                tokenizer,
                [spec.prompt for spec in batch_specs],
                ar_prompt_max_length,
            )
            ar_ids, ar_mask = _move_to_model(model, ar_ids, ar_mask)
            ar_tensor = prefix_forward_to_R_b(
                model,
                ar_ids,
                ar_mask,
                boundary_b=batch_boundary,
            )
            batch_features = select_token_vectors_by_lengths(ar_tensor, lengths, ar_tau).detach().float().cpu()

            for spec, target, feature, token_count in zip(batch_specs, batch_targets, batch_features, lengths, strict=True):
                features.append(feature)
                targets.append(target)
                records.append(
                    {
                        **parquet_spec_payload(spec),
                        "ar_tau": ar_tau,
                        "explanation_token_count": int(token_count),
                        "feature_l2": float(feature.norm().item()),
                    }
                )
            start = end

    return torch.stack(features, dim=0), torch.stack(targets, dim=0), records


def split_indices(
    count: int,
    train_fraction: float,
    *,
    strategy: str = "sequential",
    seed: int = 0,
    records: list[dict[str, Any]] | None = None,
) -> tuple[list[int], list[int]]:
    if count <= 0:
        raise ValueError("cannot split empty example set")
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    if strategy not in SPLIT_STRATEGY_CHOICES:
        raise ValueError(f"split strategy must be one of {SPLIT_STRATEGY_CHOICES}, got {strategy!r}")
    train_count = max(1, int(math.ceil(count * train_fraction)))
    train_count = min(train_count, count)
    if strategy == "doc_random":
        return split_doc_random_indices(count, train_fraction, seed=seed, records=records)
    if strategy == "sequential":
        ordered = list(range(count))
    elif strategy == "alternating":
        ordered = list(range(0, count, 2)) + list(range(1, count, 2))
    else:
        ordered = list(range(count))
        random.Random(seed).shuffle(ordered)
    train = ordered[:train_count]
    eval_indices = ordered[train_count:]
    return train, eval_indices


def split_doc_random_indices(
    count: int,
    train_fraction: float,
    *,
    seed: int,
    records: list[dict[str, Any]] | None,
) -> tuple[list[int], list[int]]:
    if records is None or len(records) != count:
        raise ValueError("doc_random split requires one record per example")
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        doc_id = record.get("doc_id")
        if doc_id is None:
            raise ValueError(f"doc_random split requires records[{idx}]['doc_id']")
        groups.setdefault(str(doc_id), []).append(idx)
    if not groups:
        raise ValueError("doc_random split received no documents")

    doc_ids = list(groups)
    random.Random(seed).shuffle(doc_ids)
    train_target = max(1, min(count, int(math.ceil(count * train_fraction))))
    train: list[int] = []
    eval_indices: list[int] = []
    train_docs: list[str] = []
    for doc_id in doc_ids:
        if len(train) < train_target or not train:
            train.extend(groups[doc_id])
            train_docs.append(doc_id)
        else:
            eval_indices.extend(groups[doc_id])

    if not eval_indices and len(train_docs) > 1 and train_fraction < 1:
        moved_doc = train_docs.pop()
        moved = groups[moved_doc]
        moved_set = set(moved)
        train = [idx for idx in train if idx not in moved_set]
        eval_indices.extend(moved)

    return train, eval_indices


def take_rows(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return tensor.new_zeros((0, tensor.shape[-1]))
    return tensor[torch.tensor(indices, dtype=torch.long)]


def random_matched_norm_targets(targets: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(targets.shape, generator=generator, dtype=torch.float32)
    norms = targets.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return l2_normalize_rows(random) * norms


def mean_target_metrics(targets: torch.Tensor, train_targets_for_mean: torch.Tensor) -> dict[str, float | None]:
    mean_target = train_targets_for_mean.float().mean(dim=0, keepdim=True)
    predictions = mean_target.expand_as(targets.float())
    metrics = vector_metrics(predictions, targets.float())
    metrics.update(centered_raw_diagnostics(predictions, targets.float(), train_targets_for_mean))
    return metrics


def split_metadata(records: list[dict[str, Any]], train_indices: list[int], eval_indices: list[int]) -> dict[str, Any]:
    def doc_ids(indices: list[int]) -> set[str]:
        return {
            str(records[idx]["doc_id"])
            for idx in indices
            if idx < len(records) and records[idx].get("doc_id") is not None
        }

    train_docs = doc_ids(train_indices)
    eval_docs = doc_ids(eval_indices)
    overlap = sorted(train_docs & eval_docs)
    return {
        "train_count": len(train_indices),
        "eval_count": len(eval_indices),
        "train_doc_count": len(train_docs),
        "eval_doc_count": len(eval_docs),
        "doc_overlap_count": len(overlap),
        "doc_overlap_sample": overlap[:10],
    }


def relative_reconstruction_improvement(loss: float, baseline_loss: float | None) -> float | None:
    if baseline_loss is None or baseline_loss <= 0:
        return None
    return float(1.0 - loss / baseline_loss)


def control_eval(
    head: ValueHead,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    seed: int,
    train_targets_for_mean: torch.Tensor | None = None,
    mse_margin: float = 0.0,
    cosine_margin: float = 0.0,
    min_rri: float = 0.0,
    eval_batch_size: int | None = None,
) -> dict[str, Any]:
    if features.shape[0] == 0:
        return {
            "count": 0,
            "correct": None,
            "shuffled": None,
            "random_matched_norm": None,
            "mean_train": None,
            "correct_beats_mean": None,
            "correct_beats_controls": None,
        }
    correct = evaluate_head(
        head,
        features,
        targets,
        train_targets_for_center=train_targets_for_mean,
        batch_size=eval_batch_size,
    )
    if targets.shape[0] > 1:
        shuffled_targets = targets.roll(shifts=1, dims=0)
        shuffled = evaluate_head(
            head,
            features,
            shuffled_targets,
            train_targets_for_center=train_targets_for_mean,
            batch_size=eval_batch_size,
        )
    else:
        shuffled = None
    random_targets = random_matched_norm_targets(targets, seed=seed)
    random_metrics = evaluate_head(
        head,
        features,
        random_targets,
        train_targets_for_center=train_targets_for_mean,
        batch_size=eval_batch_size,
    )
    mean_train = None
    correct_beats_mean = None
    if train_targets_for_mean is not None:
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
    beats_mean = True if correct_beats_mean is None else correct_beats_mean
    beats = beats_shuffled and beats_random and beats_mean
    return {
        "count": int(features.shape[0]),
        "correct": correct,
        "shuffled": shuffled,
        "random_matched_norm": random_metrics,
        "mean_train": mean_train,
        "correct_beats_mean": correct_beats_mean,
        "correct_beats_controls": beats,
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
    parser.add_argument("--boundaries", type=parse_boundaries, default=[34, 27])
    parser.add_argument("--prompt-names", type=parse_prompt_names, default=list(DEFAULT_PROMPT_NAMES))
    parser.add_argument("--prompt-max-length", type=int, default=256)
    parser.add_argument("--explanation-max-length", type=int, default=128)
    parser.add_argument("--source-tau", type=int, default=-1)
    parser.add_argument("--ar-tau", type=int, default=-1)
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument(
        "--ar-sft-parquet",
        type=Path,
        default=None,
        help="Training-ready real-data AR-SFT parquet with prompt and activation_vector columns.",
    )
    parser.add_argument(
        "--ar-prompt-max-length",
        type=int,
        default=256,
        help="Maximum tokenized AR prompt length for --ar-sft-parquet rows; rows are not truncated.",
    )
    parser.add_argument(
        "--ar-feature-batch-size",
        type=int,
        default=4,
        help="Batch size for frozen Nano prefix forwards over --ar-sft-parquet rows.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--split-strategy", choices=SPLIT_STRATEGY_CHOICES, default="sequential")
    parser.add_argument("--explanation-template", choices=EXPLANATION_TEMPLATE_CHOICES, default="generic")
    parser.add_argument("--critic-template", default=DEFAULT_CRITIC_TEMPLATE)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=0,
        help="Mini-batch size for value-head training; 0 keeps the historical full-batch path.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Batch size for value-head evaluation; 0 evaluates all rows at once.",
    )
    parser.add_argument(
        "--train-device",
        default="auto",
        help="Device for value-head training/eval: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--mse-margin", type=float, default=0.05)
    parser.add_argument("--cosine-margin", type=float, default=0.02)
    parser.add_argument("--min-rri", type=float, default=0.05)
    parser.add_argument("--save-head", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    add_wandb_args(parser)
    parser.set_defaults(load_mode="full")
    return parser.parse_args(argv)


def payload_base(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "nano_ar_frozen_baseline.v2",
        "run_dir": str(run_dir),
        "model": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        },
        "boundary_order": args.boundaries,
        "data_source": {
            "kind": "ar_sft_parquet" if args.ar_sft_parquet else "synthetic_smoke",
            "path": str(args.ar_sft_parquet) if args.ar_sft_parquet else None,
            "ar_prompt_max_length": getattr(args, "ar_prompt_max_length", None),
            "ar_feature_batch_size": getattr(args, "ar_feature_batch_size", None),
        },
        "requested_prompt_names": args.prompt_names,
        "source_tau": args.source_tau,
        "ar_tau": args.ar_tau,
        "max_records": args.max_records,
        "train_fraction": args.train_fraction,
        "split_strategy": args.split_strategy,
        "explanation_template": args.explanation_template,
        "critic_template": args.critic_template,
        "benchmark_thresholds": {
            "mse_margin": args.mse_margin,
            "cosine_margin": args.cosine_margin,
            "min_rri": args.min_rri,
        },
        "prompt_modes": [],
        "specs": [],
        "examples": [],
        "frozen_check": {},
        "value_head": {},
        "training": {},
        "eval": {},
        "passed": False,
        "scientific_passed": False,
        "blockers": [],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = make_run_dir(args.output_root, args.timestamp)
    payload = payload_base(args, run_dir)
    output_path = run_dir / "ar_frozen_baseline.json"
    tracker = init_wandb(
        args,
        run_dir=run_dir,
        job_type="ar_frozen_baseline",
        config=json_safe({"args": vars(args), "run_dir": run_dir}),
    )
    payload["wandb"] = tracker.metadata

    if torch is None:
        payload["blockers"] = [{"kind": "environment", "label": "torch import", "error": "PyTorch is required for frozen AR baseline."}]
        tracker.log_summary(payload)
        tracker.finish({"status/passed": False, "status/blockers": len(payload["blockers"])})
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        print(f"\nwrote {output_path}")
        return 2
    max_allowed_records = 200000 if args.ar_sft_parquet else 64
    if args.max_records <= 0 or args.max_records > max_allowed_records:
        payload["blockers"] = [
            {
                "kind": "configuration",
                "label": "max_records",
                "error": f"max_records must be between 1 and {max_allowed_records} for this harness mode.",
            }
        ]
        tracker.log_summary(payload)
        tracker.finish({"status/passed": False, "status/blockers": len(payload["blockers"])})
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        print(f"\nwrote {output_path}")
        return 2

    blockers: list[dict[str, str]] = []
    parquet_specs: list[ParquetARSpec] | None = None
    prompts: list[Any] = []
    specs: list[TinyARSpec] = []
    try:
        torch.manual_seed(args.random_seed)
        tokenizer = load_tokenizer_from_args(args)
        config, config_error = load_config_from_args(args)
        if config_error is not None:
            blockers.append(classify_blocker("remote-code load", config_error))
        if args.ar_sft_parquet:
            parquet_specs = load_parquet_ar_specs(
                args.ar_sft_parquet,
                boundaries=args.boundaries,
                max_records=args.max_records,
            )
            payload["specs"] = [parquet_spec_payload(spec) for spec in parquet_specs]
        else:
            prompts = build_prompt_inputs(tokenizer, args.prompt_max_length)
            prompts = [item for item in prompts if item.name in args.prompt_names]
            missing = sorted(set(args.prompt_names) - {item.name for item in prompts})
            if missing:
                raise ValueError(f"requested prompt names were not built: {missing}")
            payload["prompt_modes"] = [{"name": item.name, **item.metadata} for item in prompts]
            specs = build_tiny_ar_specs(
                prompts,
                boundaries=args.boundaries,
                max_records=args.max_records,
                explanation_template=args.explanation_template,
                critic_template=args.critic_template,
            )
            payload["specs"] = [spec._asdict() for spec in specs]
    except Exception as exc:
        blockers.append(classify_blocker("template ambiguity", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        payload["blockers"] = blockers
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        print(f"\nwrote {output_path}")
        return 2

    try:
        model = load_model_from_args(args, config)
    except Exception as exc:
        blockers.append(classify_blocker("model load", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"))
        payload["blockers"] = blockers
        write_json(output_path, payload)
        print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
        print(f"\nwrote {output_path}")
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
        freeze_module(model)
        payload["frozen_check"] = {
            "nano_trainable_parameters": trainable_parameter_count(model),
            "nano_total_parameters": total_parameter_count(model),
        }
        if parquet_specs is not None:
            features, targets, example_records = materialize_parquet_ar_examples(
                model=model,
                tokenizer=tokenizer,
                specs=parquet_specs,
                ar_tau=args.ar_tau,
                ar_prompt_max_length=args.ar_prompt_max_length,
                hidden_size=hidden_size,
                ar_feature_batch_size=args.ar_feature_batch_size,
            )
        else:
            features, targets, example_records = materialize_ar_examples(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                specs=specs,
                source_tau=args.source_tau,
                ar_tau=args.ar_tau,
                explanation_max_length=args.explanation_max_length,
            )
        payload["examples"] = example_records

        head = ValueHead(hidden_size=hidden_size)
        payload["value_head"] = {
            "class": type(head).__name__,
            "init": "identity",
            "bias": False,
            "trainable_parameters": trainable_parameter_count(head),
            "total_parameters": total_parameter_count(head),
        }

        train_indices, eval_indices = split_indices(
            features.shape[0],
            args.train_fraction,
            strategy=args.split_strategy,
            seed=args.random_seed,
            records=example_records,
        )
        train_features = take_rows(features, train_indices)
        train_targets = take_rows(targets, train_indices)
        eval_features = take_rows(features, eval_indices)
        eval_targets = take_rows(targets, eval_indices)
        train_device = resolve_train_device(args.train_device)
        eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else None
        train_batch_size = args.train_batch_size if args.train_batch_size > 0 else None

        head.to(train_device)
        train_before = evaluate_head(head, train_features, train_targets, batch_size=eval_batch_size)
        history = train_value_head(
            head,
            train_features,
            train_targets,
            max_steps=args.max_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=train_batch_size,
            device=train_device,
            seed=args.random_seed,
            eval_batch_size=eval_batch_size,
        )
        tracker.log_history(history, prefix="train")
        train_after = evaluate_head(head, train_features, train_targets, batch_size=eval_batch_size)
        train_loss_decreased = train_after["normalized_mse"] < train_before["normalized_mse"]
        payload["training"] = {
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "train_device": str(train_device),
            "history": history,
            "train_before": train_before,
            "train_after": train_after,
            "train_loss_decreased": train_loss_decreased,
            "split_metadata": split_metadata(example_records, train_indices, eval_indices),
        }

        payload["eval"] = {
            "train_controls": control_eval(
                head,
                train_features,
                train_targets,
                seed=args.random_seed + 1,
                train_targets_for_mean=train_targets,
                mse_margin=args.mse_margin,
                cosine_margin=args.cosine_margin,
                min_rri=args.min_rri,
                eval_batch_size=eval_batch_size,
            ),
            "heldout_controls": control_eval(
                head,
                eval_features,
                eval_targets,
                seed=args.random_seed + 2,
                train_targets_for_mean=train_targets,
                mse_margin=args.mse_margin,
                cosine_margin=args.cosine_margin,
                min_rri=args.min_rri,
                eval_batch_size=eval_batch_size,
            ),
        }
        heldout = payload["eval"]["heldout_controls"]
        payload["scientific_passed"] = bool(heldout["correct_beats_controls"]) if heldout["correct_beats_controls"] is not None else False

        if args.save_head:
            head_path = run_dir / "value_head.pt"
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "hidden_size": hidden_size,
                    "schema_version": payload["schema_version"],
                },
                head_path,
            )
            payload["value_head"]["checkpoint_path"] = str(head_path)

        payload["passed"] = (
            not blockers
            and payload["frozen_check"]["nano_trainable_parameters"] == 0
            and payload["value_head"]["trainable_parameters"] > 0
            and bool(train_loss_decreased)
        )
    except Exception as exc:
        blockers.append(classify_blocker("frozen AR baseline", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"))

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
