#!/usr/bin/env python3
"""Qwen-faithful Nano AV warm-start smoke.

This is intentionally small: build an AV-SFT parquet from existing teacher
labels, then run the same sidecar marker contract and pure NLA injection helper
against Nano with real/shuffled/zero/mean/no-injection controls.

It does not run RL, serve a model, or generate new teacher labels. Larger
adapter-capacity runs can enable optional PEFT LoRA while preserving the same
AV dataset and control evaluations.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import math
import os
import random
import re
import sys
import traceback
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pa = None
    pq = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if NLA_ROOT.exists() and str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))

from nano_introspection import (  # noqa: E402
    DEFAULT_MODEL_ID,
    add_bool_optional_arg,
    classify_blocker,
    json_safe,
    load_config_from_args,
    load_model_from_args,
    load_tokenizer_from_args,
    make_run_dir,
    utc_timestamp,
    write_json,
)
from nano_realdata_stage3_build import (  # noqa: E402
    DEFAULT_ACTOR_TEMPLATE,
    DEFAULT_CRITIC_TEMPLATE,
    build_stage3,
)
from nano_wandb import add_wandb_args, init_wandb  # noqa: E402
from nla.config import load_nla_config  # noqa: E402
from nla.datagen.sidecar import read_sidecar_local, write_sidecar_local  # noqa: E402
from nla.injection import inject_at_marked_positions  # noqa: E402
from nla.schema import INJECT_PLACEHOLDER, extract_explanation, normalize_activation  # noqa: E402


CONTROL_NAMES = ("real", "shuffled", "zero", "mean", "none")
EXAMPLE_CONTROLS = ("real", "shuffled", "zero", "none")
TRAINABLE_SUBSETS = ("none", "lm_head", "embeddings", "lm_head+embeddings", "all")
PEFT_METHODS = ("none", "lora")
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
)
SPLIT_MODES = ("row", "doc")
TRAIN_SAMPLING_MODES = ("random", "epoch")
EXPERIMENT_CLASSES = ("small-smoke", "medium-small", "complete-performance", "legacy")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")
STOPWORDS = {
    "about", "after", "also", "and", "are", "because", "been", "being",
    "between", "from", "has", "have", "into", "its", "model", "that",
    "the", "their", "there", "these", "this", "through", "with", "within",
    "would", "text", "passage", "following", "summary", "explanation",
}


def _require_pyarrow() -> None:
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required for AV warm-start smoke")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for AV warm-start smoke")


def _ids_to_list(ids: Any) -> list[int]:
    if isinstance(ids, dict) and "input_ids" in ids:
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError(f"expected one prompt, got batch of {len(ids)}")
        ids = ids[0]
    return [int(x) for x in ids]


def _tokenize(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    return _ids_to_list(ids)


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
) -> str | list[int]:
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        result = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        result = tokenizer.apply_chat_template(messages, **kwargs)
    if tokenize:
        return _ids_to_list(result)
    return result if isinstance(result, str) else str(result)


def _with_injection_char(messages: list[dict[str, str]], injection_char: str) -> list[dict[str, str]]:
    out = []
    for msg in messages:
        content = msg.get("content", "")
        out.append({**msg, "content": content.replace(INJECT_PLACEHOLDER, injection_char)})
    return out


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def resolve_injection_scale(value: str | float | None, d_model: int) -> float | None:
    if value is None:
        return math.sqrt(d_model)
    if isinstance(value, (int, float)):
        return float(value)
    lowered = str(value).strip().lower()
    if lowered in {"raw", "none", "null"}:
        return None
    if lowered in {"sqrt_d_model", "sqrt_d", "sqrt"}:
        return math.sqrt(d_model)
    return float(value)


def content_terms(text: str) -> set[str]:
    return {
        match.group(0).lower().strip("'_-")
        for match in WORD_RE.finditer(text or "")
        if match.group(0).lower().strip("'_-") not in STOPWORDS
    }


def text_overlap_metrics(generated: str, target: str) -> dict[str, float | int]:
    generated_terms = content_terms(generated)
    target_terms = content_terms(target)
    overlap = generated_terms & target_terms
    precision = len(overlap) / len(generated_terms) if generated_terms else 0.0
    recall = len(overlap) / len(target_terms) if target_terms else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "generated_terms": len(generated_terms),
        "target_terms": len(target_terms),
        "overlap_terms": len(overlap),
        "content_precision": precision,
        "content_recall": recall,
        "content_f1": f1,
    }


def split_indices(n_rows: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n_rows))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = max(1, min(n_rows - 1, round(n_rows * train_fraction))) if n_rows > 1 else n_rows
    train = sorted(indices[:n_train])
    heldout = sorted(indices[n_train:])
    return train, heldout


def split_items_three_way(
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
        raise ValueError(
            "train_fraction + validation_fraction + test_fraction must be <= 1.0"
        )

    shuffled = list(items)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_items = len(shuffled)
    if validation_fraction == 0 and test_fraction == 0:
        n_train = max(1, min(n_items - 1, round(n_items * train_fraction))) if n_items > 1 else n_items
        return shuffled[:n_train], shuffled[n_train:], []

    n_train = round(n_items * train_fraction)
    n_validation = round(n_items * validation_fraction)
    if validation_fraction > 0:
        n_validation = max(1, n_validation)
    if test_fraction > 0:
        n_test = n_items - n_train - n_validation if total_fraction >= 0.999999 else round(n_items * test_fraction)
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


def split_indices_for_rows(
    rows: list[dict[str, Any]],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    *,
    split_mode: str,
) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"unknown split_mode={split_mode!r}; choices={SPLIT_MODES}")
    if split_mode == "row":
        train, validation, test = split_items_three_way(
            list(range(len(rows))),
            train_fraction,
            validation_fraction,
            test_fraction,
            seed,
        )
        return sorted(train), sorted(validation), sorted(test), {
            "split_mode": "row",
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
        }

    doc_to_indices: dict[str, list[int]] = {}
    for row in rows:
        doc = str(row.get("doc_id") or f"__row_{row['row_index']}")
        doc_to_indices.setdefault(doc, []).append(int(row["row_index"]))
    docs = sorted(doc_to_indices)
    train_doc_list, validation_doc_list, test_doc_list = split_items_three_way(
        docs,
        train_fraction,
        validation_fraction,
        test_fraction,
        seed,
    )
    train_docs = set(train_doc_list)
    validation_docs = set(validation_doc_list)
    test_docs = set(test_doc_list)
    train = sorted(i for doc in train_docs for i in doc_to_indices[doc])
    validation = sorted(i for doc in validation_docs for i in doc_to_indices[doc])
    test = sorted(i for doc in test_docs for i in doc_to_indices[doc])
    return train, validation, test, {
        "split_mode": "doc",
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "doc_count": len(docs),
        "train_doc_count": len(train_docs),
        "validation_doc_count": len(validation_docs),
        "test_doc_count": len(test_docs),
        "heldout_doc_count": len(validation_docs) + len(test_docs),
        "doc_overlap_count": 0,
    }


def sample_eval_indices(
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
    *,
    train_limit: int,
    validation_limit: int,
    test_limit: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)

    def sample(indices: list[int], limit: int) -> list[int]:
        if limit <= 0 or limit >= len(indices):
            return sorted(indices)
        chosen = list(indices)
        rng.shuffle(chosen)
        return sorted(chosen[:limit])

    return (
        sample(train_indices, train_limit),
        sample(validation_indices, validation_limit),
        sample(test_indices, test_limit),
    )


def build_control_vectors(
    vectors: "torch.Tensor",
    *,
    row_index: int,
    train_indices: list[int],
    seed: int,
) -> dict[str, "torch.Tensor | None"]:
    _require_torch()
    n_rows, d_model = vectors.shape
    rng = random.Random(seed + row_index * 1009)
    candidates = [i for i in range(n_rows) if i != row_index]
    shuffled_index = rng.choice(candidates) if candidates else row_index
    if not train_indices:
        raise ValueError("train_indices must not be empty when building mean controls")
    mean_source = train_indices
    return {
        "real": vectors[row_index],
        "shuffled": vectors[shuffled_index],
        "zero": torch.zeros(d_model, dtype=vectors.dtype),
        "mean": vectors[mean_source].mean(dim=0),
        "none": None,
    }


def slice_stage2_input(
    input_path: Path,
    output_path: Path,
    *,
    row_limit: int,
    row_offset: int,
) -> dict[str, Any]:
    _require_pyarrow()
    table = pq.read_table(input_path)
    if "api_explanation" not in table.column_names:
        raise ValueError(f"{input_path} has no api_explanation column")

    explanations = table.column("api_explanation").to_pylist()
    usable = [i for i, text in enumerate(explanations) if isinstance(text, str) and text.strip()]
    selected = usable[row_offset : row_offset + row_limit]
    if not selected:
        raise ValueError(
            f"no usable api_explanation rows selected from {input_path} "
            f"(usable={len(usable)}, row_offset={row_offset}, row_limit={row_limit})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sliced = table.take(pa.array(selected, type=pa.int64()))
    pq.write_table(sliced, output_path)

    in_meta = read_sidecar_local(input_path)
    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__avwarm_{row_offset}_{len(selected)}",
        row_count=len(selected),
        parent_datasets=[in_meta.dataset_id],
        created_by="scripts.nano_av_warmstart_smoke.slice_stage2_input",
        created_at="",
    )
    write_sidecar_local(output_path, out_meta)
    return {
        "input_rows": table.num_rows,
        "usable_rows": len(usable),
        "selected_rows": len(selected),
        "dropped_missing_explanation": table.num_rows - len(usable),
        "selected_source_indices": selected,
    }


def load_av_rows(av_sft_path: Path) -> list[dict[str, Any]]:
    _require_pyarrow()
    table = pq.read_table(av_sft_path)
    rows = table.to_pylist()
    for i, row in enumerate(rows):
        row["row_index"] = i
    return rows


def av_dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _require_torch()
    if not rows:
        return {"row_count": 0}

    dim_counts: dict[int, int] = {}
    nonfinite_rows = 0
    norm_sum = 0.0
    norm_min: float | None = None
    norm_max: float | None = None
    doc_counts: dict[str, int] = {}
    layer_counts: dict[str, int] = {}
    empty_response_rows = 0
    malformed_response_rows = 0

    for row in rows:
        vector = torch.tensor(row["activation_vector"], dtype=torch.float32)
        dim = int(vector.numel())
        dim_counts[dim] = dim_counts.get(dim, 0) + 1
        finite = bool(torch.isfinite(vector).all().item())
        if not finite:
            nonfinite_rows += 1
        norm = float(vector.norm().item()) if finite else float("nan")
        if finite:
            norm_sum += norm
            norm_min = norm if norm_min is None else min(norm_min, norm)
            norm_max = norm if norm_max is None else max(norm_max, norm)
        doc = str(row.get("doc_id") or "")
        doc_counts[doc] = doc_counts.get(doc, 0) + 1
        layer = str(row.get("activation_layer"))
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        response = str(row.get("response") or "")
        if not response.strip():
            empty_response_rows += 1
        if not (response.startswith("<explanation>") and response.rstrip().endswith("</explanation>")):
            malformed_response_rows += 1

    finite_count = len(rows) - nonfinite_rows
    duplicate_docs = sum(1 for count in doc_counts.values() if count > 1)
    return {
        "row_count": len(rows),
        "activation_dim_counts": dim_counts,
        "nonfinite_activation_rows": nonfinite_rows,
        "activation_norm_mean": norm_sum / finite_count if finite_count else None,
        "activation_norm_min": norm_min,
        "activation_norm_max": norm_max,
        "unique_doc_count": len(doc_counts),
        "duplicate_doc_count": duplicate_docs,
        "max_rows_per_doc": max(doc_counts.values()) if doc_counts else 0,
        "activation_layer_counts": layer_counts,
        "empty_response_rows": empty_response_rows,
        "malformed_response_rows": malformed_response_rows,
    }


def load_av_config(av_sft_path: Path, tokenizer: Any) -> Any:
    """Load NLA config from the parquet source, letting NLA resolve its sidecar."""
    return load_nla_config(av_sft_path, tokenizer)


def build_actor_sft_example(
    tokenizer: Any,
    cfg: Any,
    row: dict[str, Any],
    *,
    max_target_tokens: int,
) -> dict[str, Any]:
    """Build the Miles-free equivalent of nla.rollout.sft_actor output.

    Reference path:
      NLADataSource substitutes <INJECT> with the sidecar injection char.
      sft_actor appends {"role": "assistant", "content": response}.
      MultiTurnLossMaskGenerator masks prompt tokens and trains on response.

    Here we preserve those semantics without Miles' Sample/DataIterator types.
    """
    _require_torch()
    messages = _with_injection_char(row["prompt"], cfg.injection_char)
    input_ids, label_start, tokenization_mode = build_loss_sequence(
        tokenizer,
        messages,
        row["response"],
        max_target_tokens=max_target_tokens,
    )
    if label_start >= len(input_ids):
        raise ValueError(f"row {row['row_index']} produced no target tokens")
    ids = torch.tensor(input_ids, dtype=torch.long)
    labels = ids.clone()
    labels[:label_start] = -100
    return {
        "row_index": int(row["row_index"]),
        "messages": messages,
        "input_ids": ids,
        "labels": labels,
        "label_start": label_start,
        "target_tokens": int((labels != -100).sum().item()),
        "activation_vector": torch.tensor(row["activation_vector"], dtype=torch.float32),
        "tokenization_mode": tokenization_mode,
    }


def configure_trainable_parameters(model: Any, *, trainable_subset: str) -> dict[str, Any]:
    """Freeze Nano and unfreeze only the requested HF-native SFT subset.

    This replaces Miles/FSDP full-actor training for tiny single-GPU overfit
    probes. `all` exists for parity experiments when a proper distributed setup
    is available; the safe local default is `lm_head`.
    """
    if trainable_subset not in TRAINABLE_SUBSETS:
        raise ValueError(f"unknown trainable_subset={trainable_subset!r}; choices={TRAINABLE_SUBSETS}")

    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        if trainable_subset == "all":
            requires_grad = True
        elif trainable_subset == "none":
            requires_grad = False
        elif trainable_subset == "lm_head":
            requires_grad = name.startswith("lm_head.")
        elif trainable_subset == "embeddings":
            requires_grad = ".embeddings." in name or name.startswith("embeddings.")
        elif trainable_subset == "lm_head+embeddings":
            requires_grad = (
                name.startswith("lm_head.")
                or ".embeddings." in name
                or name.startswith("embeddings.")
            )
        else:
            raise AssertionError(f"unreachable subset: {trainable_subset}")
        parameter.requires_grad = requires_grad
        if requires_grad:
            trainable_names.append(name)

    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return {
        "trainable_subset": trainable_subset,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "trainable_names": trainable_names[:50],
        "trainable_name_count": len(trainable_names),
    }


def summarize_trainable_parameters(model: Any, *, trainable_subset: str) -> dict[str, Any]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return {
        "trainable_subset": trainable_subset,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "trainable_names": trainable_names[:50],
        "trainable_name_count": len(trainable_names),
    }


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def estimate_lora_parameters(
    model: Any,
    *,
    target_modules: list[str],
    rank: int,
) -> dict[str, Any]:
    target_set = set(target_modules)
    matched: list[dict[str, Any]] = []
    estimated = 0
    for name, parameter in model.named_parameters():
        if not name.endswith(".weight") or parameter.ndim != 2:
            continue
        module_name = name.removesuffix(".weight")
        suffix = module_name.split(".")[-1]
        if suffix not in target_set:
            continue
        out_features, in_features = (int(parameter.shape[0]), int(parameter.shape[1]))
        lora_params = rank * (out_features + in_features)
        estimated += lora_params
        matched.append(
            {
                "module": module_name,
                "suffix": suffix,
                "weight_shape": [out_features, in_features],
                "lora_parameters": lora_params,
            }
        )
    total = sum(int(p.numel()) for p in model.parameters())
    return {
        "target_modules": target_modules,
        "rank": rank,
        "matched_module_count": len(matched),
        "estimated_lora_parameters": estimated,
        "estimated_lora_fraction_of_base": estimated / total if total else 0.0,
        "matched_modules_sample": matched[:50],
    }


def disable_transformer_engine_for_peft_if_requested() -> None:
    """Avoid broken TransformerEngine imports when PEFT only needs torch Linear."""
    if os.environ.get("NANO_PEFT_DISABLE_TE", "1") == "0":
        return
    if "transformer_engine" in sys.modules:
        return
    module = types.ModuleType("transformer_engine")
    module.__spec__ = importlib.machinery.ModuleSpec("transformer_engine", loader=None)
    sys.modules["transformer_engine"] = module


def apply_lora_adapters(
    model: Any,
    *,
    target_modules: list[str],
    rank: int,
    alpha: int,
    dropout: float,
    bias: str,
    modules_to_save: list[str],
    use_rslora: bool = False,
    use_dora: bool = False,
) -> tuple[Any, dict[str, Any]]:
    disable_transformer_engine_for_peft_if_requested()
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ModuleNotFoundError as exc:
        raise RuntimeError("peft is required for --peft-method lora") from exc

    estimate = estimate_lora_parameters(model, target_modules=target_modules, rank=rank)
    config_kwargs = {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
        "lora_dropout": dropout,
        "bias": bias,
        "task_type": TaskType.CAUSAL_LM,
        "modules_to_save": modules_to_save or None,
    }
    if use_rslora:
        config_kwargs["use_rslora"] = True
    if use_dora:
        config_kwargs["use_dora"] = True
    try:
        config = LoraConfig(**config_kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "Installed peft does not support the requested LoRA options; "
            "check --lora-use-rslora/--lora-use-dora against the peft version."
        ) from exc
    model = get_peft_model(model, config)
    summary = summarize_trainable_parameters(model, trainable_subset="peft:lora")
    summary.update(
        {
            "peft_method": "lora",
            "lora_rank": rank,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "lora_bias": bias,
            "lora_target_modules": target_modules,
            "lora_modules_to_save": modules_to_save,
            "lora_use_rslora": use_rslora,
            "lora_use_dora": use_dora,
            "lora_estimate_before_wrap": estimate,
        }
    )
    return model, summary


def build_loss_sequence(
    tokenizer: Any,
    messages: list[dict[str, str]],
    response: str,
    *,
    max_target_tokens: int,
) -> tuple[list[int], int, str]:
    prompt_ids = _apply_chat_template(
        tokenizer, messages, tokenize=True, add_generation_prompt=True
    )
    full_messages = messages + [{"role": "assistant", "content": response}]
    full_ids = _apply_chat_template(
        tokenizer, full_messages, tokenize=True, add_generation_prompt=False
    )
    assert isinstance(prompt_ids, list) and isinstance(full_ids, list)
    if common_prefix_len(prompt_ids, full_ids) == len(prompt_ids):
        label_start = len(prompt_ids)
        mode = "chat_template"
    else:
        response_ids = _tokenize(tokenizer, response, add_special_tokens=False)
        full_ids = prompt_ids + response_ids
        label_start = len(prompt_ids)
        mode = "prompt_plus_response_fallback"
    label_end = min(len(full_ids), label_start + max_target_tokens)
    return full_ids[:label_end], label_start, mode


def build_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> list[int]:
    prompt_ids = _apply_chat_template(
        tokenizer, messages, tokenize=True, add_generation_prompt=True
    )
    assert isinstance(prompt_ids, list)
    return prompt_ids


def _embedding_device(model: Any) -> Any:
    return model.get_input_embeddings().weight.device


def should_train_embeddings(trainable_subset: str) -> bool:
    return trainable_subset in {"embeddings", "lm_head+embeddings", "all"}


def injected_embeddings(
    model: Any,
    input_ids: list[int],
    vector: "torch.Tensor | None",
    *,
    cfg: Any,
    injection_scale: float | None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    _require_torch()
    device = _embedding_device(model)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        embeds = model.get_input_embeddings()(ids)
        if vector is None:
            return ids, embeds
        raw = vector.detach().float().view(1, -1)
        scaled = normalize_activation(raw, injection_scale)
        embeds = inject_at_marked_positions(
            ids.detach().cpu(),
            embeds,
            scaled,
            cfg.injection_token_id,
            cfg.injection_left_neighbor_id,
            cfg.injection_right_neighbor_id,
        )
    return ids, embeds


def injected_embeddings_for_training(
    model: Any,
    input_ids: "torch.Tensor",
    vector: "torch.Tensor | None",
    *,
    cfg: Any,
    injection_scale: float | None,
    train_embeddings: bool,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    _require_torch()
    device = _embedding_device(model)
    ids = input_ids.to(device=device, dtype=torch.long).unsqueeze(0)
    if train_embeddings:
        embeds = model.get_input_embeddings()(ids)
    else:
        with torch.no_grad():
            embeds = model.get_input_embeddings()(ids)
        embeds = embeds.detach()
    if vector is None:
        return ids, embeds
    scaled = normalize_activation(vector.detach().float().view(1, -1), injection_scale)
    embeds = inject_at_marked_positions(
        ids.detach().cpu(),
        embeds,
        scaled,
        cfg.injection_token_id,
        cfg.injection_left_neighbor_id,
        cfg.injection_right_neighbor_id,
    )
    return ids, embeds


def actor_sft_loss(
    model: Any,
    cfg: Any,
    example: dict[str, Any],
    *,
    injection_scale: float | None,
    train_embeddings: bool,
) -> "torch.Tensor":
    _require_torch()
    ids, embeds = injected_embeddings_for_training(
        model,
        example["input_ids"],
        example["activation_vector"],
        cfg=cfg,
        injection_scale=injection_scale,
        train_embeddings=train_embeddings,
    )
    labels = example["labels"].to(device=ids.device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(ids)
    out = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    return out.loss


def teacher_forced_loss(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    row: dict[str, Any],
    vector: "torch.Tensor | None",
    *,
    injection_scale: float | None,
    max_target_tokens: int,
) -> dict[str, Any]:
    _require_torch()
    messages = _with_injection_char(row["prompt"], cfg.injection_char)
    input_ids, label_start, tokenization_mode = build_loss_sequence(
        tokenizer, messages, row["response"], max_target_tokens=max_target_tokens
    )
    if label_start >= len(input_ids):
        raise ValueError(f"row {row['row_index']} produced no target tokens")
    ids, embeds = injected_embeddings(
        model, input_ids, vector, cfg=cfg, injection_scale=injection_scale
    )
    labels = ids.clone()
    labels[:, :label_start] = -100
    attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
    return {
        "loss": float(out.loss.detach().float().cpu()),
        "target_tokens": int((labels != -100).sum().item()),
        "total_tokens": len(input_ids),
        "tokenization_mode": tokenization_mode,
    }


def generate_with_control(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    row: dict[str, Any],
    vector: "torch.Tensor | None",
    *,
    injection_scale: float | None,
    max_new_tokens: int,
    generation_prefix: str = "",
    stop_text: str | None = None,
    use_cache: bool = False,
) -> str:
    _require_torch()
    messages = _with_injection_char(row["prompt"], cfg.injection_char)
    input_ids = build_prompt_ids(tokenizer, messages)
    ids, embeds = injected_embeddings(
        model, input_ids, vector, cfg=cfg, injection_scale=injection_scale
    )
    attention_mask = torch.ones_like(ids)
    generated_ids: list[int] = []
    prefix_ids = _tokenize(tokenizer, generation_prefix, add_special_tokens=False) if generation_prefix else []
    if prefix_ids:
        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=ids.device)
        prefix_embeds = model.get_input_embeddings()(prefix_tensor).to(dtype=embeds.dtype)
        embeds = torch.cat([embeds, prefix_embeds], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], len(prefix_ids)), device=attention_mask.device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )
        generated_ids.extend(prefix_ids)
    if use_cache:
        from nano_av_generation import greedy_generate_with_cache

        result = greedy_generate_with_cache(
            model,
            tokenizer,
            initial_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            generated_ids=generated_ids,
            stop_text=stop_text,
        )
        return result.text
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    with torch.no_grad():
        cur_embeds = embeds
        for _ in range(max_new_tokens):
            out = model(inputs_embeds=cur_embeds, attention_mask=attention_mask, use_cache=False)
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1)
            token_id = int(next_id[0].detach().cpu())
            generated_ids.append(token_id)
            if token_id in eos_ids:
                break
            if stop_text and stop_text in tokenizer.decode(generated_ids, skip_special_tokens=True):
                break
            next_embed = model.get_input_embeddings()(next_id[:, None]).to(dtype=cur_embeds.dtype)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_controls_for_row(
    model: Any,
    tokenizer: Any,
    cfg: Any,
    row: dict[str, Any],
    controls: dict[str, "torch.Tensor | None"],
    control_names: list[str] | tuple[str, ...],
    *,
    injection_scale: float | None,
    max_new_tokens: int,
    generation_prefix: str = "",
    stop_text: str | None = None,
    use_cache: bool = False,
    batch_full_prefix: bool = False,
) -> dict[str, str]:
    """Generate AV text for same-prompt controls with the requested batching mode."""

    _require_torch()
    requested = [str(name) for name in control_names]
    if (not use_cache and not batch_full_prefix) or len(requested) <= 1:
        return {
            name: generate_with_control(
                model,
                tokenizer,
                cfg,
                row,
                controls[name],
                injection_scale=injection_scale,
                max_new_tokens=max_new_tokens,
                generation_prefix=generation_prefix,
                stop_text=stop_text,
                use_cache=use_cache,
            )
            for name in requested
        }

    from nano_av_generation import greedy_generate_batch_full_prefix, greedy_generate_batch_with_cache

    messages = _with_injection_char(row["prompt"], cfg.injection_char)
    input_ids = build_prompt_ids(tokenizer, messages)
    batched_embeds = []
    batched_ids = []
    for name in requested:
        ids, embeds = injected_embeddings(
            model,
            input_ids,
            controls[name],
            cfg=cfg,
            injection_scale=injection_scale,
        )
        batched_ids.append(ids)
        batched_embeds.append(embeds)
    ids_batch = torch.cat(batched_ids, dim=0)
    embeds_batch = torch.cat(batched_embeds, dim=0)
    attention_mask = torch.ones_like(ids_batch)
    prefix_ids = _tokenize(tokenizer, generation_prefix, add_special_tokens=False) if generation_prefix else []
    generated_ids = [list(prefix_ids) for _ in requested]
    if prefix_ids:
        prefix_tensor = torch.tensor([prefix_ids] * len(requested), dtype=torch.long, device=ids_batch.device)
        prefix_embeds = model.get_input_embeddings()(prefix_tensor).to(dtype=embeds_batch.dtype)
        embeds_batch = torch.cat([embeds_batch, prefix_embeds], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], len(prefix_ids)),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )
    generate_batch = greedy_generate_batch_with_cache if use_cache else greedy_generate_batch_full_prefix
    results = generate_batch(
        model,
        tokenizer,
        initial_embeds=embeds_batch,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        generated_ids=generated_ids,
        stop_text=stop_text,
    )
    return {name: result.text for name, result in zip(requested, results)}


def mean_float(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize_losses(
    per_row: list[dict[str, Any]],
    validation_indices: set[int],
    test_indices: set[int] | None = None,
) -> dict[str, Any]:
    test_indices = test_indices or set()
    summary: dict[str, Any] = {}
    for split_name, predicate in {
        "train": lambda idx: idx not in validation_indices and idx not in test_indices,
        "validation": lambda idx: idx in validation_indices,
        "heldout": lambda idx: idx in validation_indices,
        "test": lambda idx: idx in test_indices,
        "all": lambda idx: True,
    }.items():
        split_rows = [r for r in per_row if predicate(int(r["row_index"]))]
        summary[split_name] = {
            "count": len(split_rows),
            "loss": mean_float([float(r["loss"]) for r in split_rows]),
            "target_tokens": mean_float([float(r["target_tokens"]) for r in split_rows]),
        }
    return summary


def evaluate_smoke(
    *,
    av_sft_path: Path,
    model: Any,
    tokenizer: Any,
    cfg: Any,
    injection_scale: float | None,
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
    eval_train_limit: int,
    eval_validation_limit: int,
    eval_test_limit: int,
    seed: int,
    max_target_tokens: int,
    generate_examples: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    _require_torch()
    rows = load_av_rows(av_sft_path)
    vectors = torch.tensor([row["activation_vector"] for row in rows], dtype=torch.float32)
    validation_set = set(validation_indices)
    test_set = set(test_indices)
    eval_train_indices, eval_validation_indices, eval_test_indices = sample_eval_indices(
        train_indices,
        validation_indices,
        test_indices,
        train_limit=eval_train_limit,
        validation_limit=eval_validation_limit,
        test_limit=eval_test_limit,
        seed=seed + 91_337,
    )
    eval_indices = sorted(set(eval_train_indices) | set(eval_validation_indices) | set(eval_test_indices))

    losses: dict[str, list[dict[str, Any]]] = {name: [] for name in CONTROL_NAMES}
    for row_index in eval_indices:
        row = rows[row_index]
        controls = build_control_vectors(
            vectors,
            row_index=int(row["row_index"]),
            train_indices=train_indices,
            seed=seed,
        )
        for name in CONTROL_NAMES:
            item = teacher_forced_loss(
                model,
                tokenizer,
                cfg,
                row,
                controls[name],
                injection_scale=injection_scale,
                max_target_tokens=max_target_tokens,
            )
            item.update({"row_index": int(row["row_index"]), "control": name})
            losses[name].append(item)

    loss_summary = {
        name: summarize_losses(items, validation_set, test_set) for name, items in losses.items()
    }
    real_heldout = loss_summary["real"]["validation"]["loss"]
    for name in ("shuffled", "zero", "mean", "none"):
        control_loss = loss_summary[name]["validation"]["loss"]
        loss_summary["real"][f"validation_loss_gap_vs_{name}"] = (
            None if real_heldout is None or control_loss is None else control_loss - real_heldout
        )
        loss_summary["real"][f"heldout_loss_gap_vs_{name}"] = loss_summary["real"][
            f"validation_loss_gap_vs_{name}"
        ]
        real_test = loss_summary["real"]["test"]["loss"]
        control_test = loss_summary[name]["test"]["loss"]
        loss_summary["real"][f"test_loss_gap_vs_{name}"] = (
            None if real_test is None or control_test is None else control_test - real_test
        )

    examples = []
    example_source = validation_indices if validation_indices else list(range(len(rows)))
    example_rows = [rows[i] for i in example_source[: max(0, generate_examples)]]
    for row in example_rows:
        controls = build_control_vectors(
            vectors,
            row_index=int(row["row_index"]),
            train_indices=train_indices,
            seed=seed,
        )
        target = extract_explanation(row["response"]) or row["response"]
        item = {
            "row_index": int(row["row_index"]),
            "doc_id": row.get("doc_id"),
            "target_excerpt": target[:500],
            "controls": {},
        }
        for name in EXAMPLE_CONTROLS:
            try:
                generated = generate_with_control(
                    model,
                    tokenizer,
                    cfg,
                    row,
                    controls[name],
                    injection_scale=injection_scale,
                    max_new_tokens=max_new_tokens,
                )
                metrics = text_overlap_metrics(generated, target)
                item["controls"][name] = {
                    "generated": generated,
                    "metrics": metrics,
                }
            except Exception as exc:
                item["controls"][name] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=4),
                }
        examples.append(item)

    return {
        "row_count": len(rows),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "heldout_indices": validation_indices,
        "eval_train_indices": eval_train_indices,
        "eval_validation_indices": eval_validation_indices,
        "eval_test_indices": eval_test_indices,
        "eval_heldout_indices": eval_validation_indices,
        "injection_scale": injection_scale,
        "loss_summary": loss_summary,
        "examples": examples,
    }


def save_trainable_state(
    *,
    model: Any,
    tokenizer: Any,
    output_path: Path,
    trainable_subset: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    _require_torch()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "format": "nano_av_trainable_state.v1",
        "trainable_subset": trainable_subset,
        "metadata": metadata,
        "state_dict": state,
    }
    torch.save(payload, output_path)
    tokenizer_dir = output_path.parent / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "tensor_count": len(state),
        "tensor_names": list(state)[:50],
        "tokenizer_dir": str(tokenizer_dir),
    }


def train_actor_sft_smoke(
    *,
    av_sft_path: Path,
    model: Any,
    tokenizer: Any,
    cfg: Any,
    injection_scale: float | None,
    train_indices: list[int],
    trainable_subset: str,
    train_steps: int,
    train_batch_size: int,
    train_sampling: str,
    train_lr: float,
    max_target_tokens: int,
    log_every: int,
    max_grad_norm: float,
    gradient_checkpointing: bool,
    respect_existing_trainable: bool,
) -> dict[str, Any]:
    _require_torch()
    rows = load_av_rows(av_sft_path)
    if not train_indices:
        raise ValueError("train_actor_sft_smoke requires at least one train row")

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if respect_existing_trainable:
        parameter_summary = summarize_trainable_parameters(
            model,
            trainable_subset=trainable_subset,
        )
    else:
        parameter_summary = configure_trainable_parameters(
            model,
            trainable_subset=trainable_subset,
        )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if train_steps > 0 and not trainable_params:
        raise ValueError("train_steps > 0 but no trainable parameters were selected")

    examples = {
        idx: build_actor_sft_example(
            tokenizer,
            cfg,
            rows[idx],
            max_target_tokens=max_target_tokens,
        )
        for idx in train_indices
    }
    optimizer = torch.optim.AdamW(trainable_params, lr=train_lr) if trainable_params else None
    rng = random.Random(17_171 + len(train_indices) + train_steps)
    history: list[dict[str, Any]] = []
    train_embeddings = should_train_embeddings(trainable_subset)
    epoch_order = list(train_indices)
    epoch_cursor = 0
    epoch_passes_started = 0
    if train_sampling == "epoch":
        rng.shuffle(epoch_order)
        epoch_passes_started = 1

    def next_train_index() -> int:
        nonlocal epoch_cursor, epoch_passes_started
        if train_sampling == "random":
            return rng.choice(train_indices)
        if epoch_cursor >= len(epoch_order):
            rng.shuffle(epoch_order)
            epoch_cursor = 0
            epoch_passes_started += 1
        idx = epoch_order[epoch_cursor]
        epoch_cursor += 1
        return idx

    model.train()
    for step in range(1, train_steps + 1):
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        batch_indices = [next_train_index() for _ in range(train_batch_size)]
        losses = []
        for idx in batch_indices:
            loss = actor_sft_loss(
                model,
                cfg,
                examples[idx],
                injection_scale=injection_scale,
                train_embeddings=train_embeddings,
            )
            (loss / train_batch_size).backward()
            losses.append(float(loss.detach().float().cpu()))
        grad_norm = None
        if max_grad_norm > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm))
        optimizer.step()
        if step == 1 or step == train_steps or step % max(1, log_every) == 0:
            history.append(
                {
                    "step": step,
                    "loss": mean_float(losses),
                    "batch_indices": batch_indices,
                    "grad_norm": grad_norm,
                }
            )
    model.eval()
    return {
        "train_steps": train_steps,
        "train_batch_size": train_batch_size,
        "train_sampling": train_sampling,
        "train_examples_requested": train_steps * train_batch_size,
        "train_unique_indices_seen": (
            min(len(train_indices), train_steps * train_batch_size)
            if train_sampling == "epoch"
            else None
        ),
        "train_epoch_passes_started": epoch_passes_started if train_sampling == "epoch" else None,
        "train_lr": train_lr,
        "max_target_tokens": max_target_tokens,
        "parameter_summary": parameter_summary,
        "history": history,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-explained", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/introspection"))
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--row-limit", type=int, default=32)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument(
        "--experiment-class",
        choices=EXPERIMENT_CLASSES,
        default="legacy",
        help="Experiment scale bucket used for split/row-limit validation.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--split-mode", choices=SPLIT_MODES, default="row")
    parser.add_argument("--eval-train-limit", type=int, default=0)
    parser.add_argument("--eval-heldout-limit", type=int, default=0)
    parser.add_argument("--eval-validation-limit", type=int, default=None)
    parser.add_argument("--eval-test-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--injection-scale", default="150")
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--generate-examples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument(
        "--train-epochs",
        type=float,
        default=0.0,
        help=(
            "When >0, override --train-steps with ceil(train_count * epochs / "
            "train_batch_size). Use with --train-sampling epoch for full-split passes."
        ),
    )
    parser.add_argument("--train-sampling", choices=TRAIN_SAMPLING_MODES, default="random")
    parser.add_argument("--train-lr", type=float, default=1e-4)
    parser.add_argument("--trainable-subset", choices=TRAINABLE_SUBSETS, default="lm_head")
    parser.add_argument("--peft-method", choices=PEFT_METHODS, default="none")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-bias", choices=("none", "all", "lora_only"), default="none")
    parser.add_argument("--lora-use-rslora", action="store_true")
    parser.add_argument("--lora-use-dora", action="store_true")
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
        help="Comma-separated module-name suffixes for PEFT LoRA.",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        default="",
        help="Comma-separated full modules to keep trainable/saved under PEFT.",
    )
    parser.add_argument("--train-log-every", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-trainable-state", action="store_true")
    parser.add_argument("--model-id", default=os.environ.get("NANO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--model-revision", default=os.environ.get("NANO_MODEL_REVISION"))
    parser.add_argument("--tokenizer-revision", default=os.environ.get("NANO_TOKENIZER_REVISION"))
    parser.add_argument("--load-mode", choices=("full", "meta", "config"), default="full")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    add_bool_optional_arg(parser, "--trust-remote-code", default=True)
    add_wandb_args(parser)
    return parser.parse_args(argv)


def validate_experiment_class(args: argparse.Namespace) -> None:
    if args.experiment_class == "legacy":
        return
    if args.experiment_class == "small-smoke":
        if args.row_limit >= 100:
            raise ValueError("small-smoke requires --row-limit < 100")
    elif args.experiment_class == "medium-small":
        if args.row_limit >= 1000:
            raise ValueError("medium-small requires --row-limit < 1000")
    elif args.experiment_class == "complete-performance":
        if args.row_limit < 90000:
            raise ValueError("complete-performance requires --row-limit >= 90000")
        if args.split_mode != "doc":
            raise ValueError("complete-performance requires --split-mode doc")
        expected = {
            "train_fraction": (args.train_fraction, 0.9),
            "validation_fraction": (args.validation_fraction, 0.05),
            "test_fraction": (args.test_fraction, 0.05),
        }
        for name, (actual, target) in expected.items():
            if abs(actual - target) > 1e-9:
                raise ValueError(f"complete-performance requires {name}={target}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_experiment_class(args)
    run_dir = make_run_dir(args.output_root, args.timestamp or f"av-warmstart-smoke-{utc_timestamp()}")
    base_slice = run_dir / "av_warmstart_base_slice.parquet"
    av_sft = run_dir / "av_sft.parquet"
    report_path = run_dir / "av_warmstart_smoke.json"
    manifest: dict[str, Any] = {
        "schema_version": "nano_av_warmstart_smoke.v1",
        "run_dir": str(run_dir),
        "input_explained": str(args.input_explained),
        "base_slice": str(base_slice),
        "av_sft": str(av_sft),
        "report": str(report_path),
        "row_limit": args.row_limit,
        "row_offset": args.row_offset,
        "experiment_class": args.experiment_class,
        "split_mode": args.split_mode,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "eval_train_limit": args.eval_train_limit,
        "eval_heldout_limit": args.eval_heldout_limit,
        "seed": args.seed,
        "requested_trainable_subset": args.trainable_subset,
        "effective_trainable_subset": args.trainable_subset,
        "peft_method": args.peft_method,
        "blockers": [],
    }
    tracker = init_wandb(
        args,
        run_dir=run_dir,
        job_type="av_warmstart",
        config=json_safe({"args": vars(args), "run_dir": run_dir}),
    )
    manifest["wandb"] = tracker.metadata

    try:
        tokenizer = load_tokenizer_from_args(args)
        slice_result = slice_stage2_input(
            args.input_explained,
            base_slice,
            row_limit=args.row_limit,
            row_offset=args.row_offset,
        )
        manifest["slice"] = slice_result
        stage3_result = build_stage3(
            input_path=base_slice,
            output_path=av_sft,
            stage="av_sft",
            tokenizer=tokenizer,
            actor_template=DEFAULT_ACTOR_TEMPLATE,
            critic_template=DEFAULT_CRITIC_TEMPLATE,
            keep_debug_metadata=True,
        )
        manifest["stage3"] = stage3_result

        cfg = load_av_config(av_sft, tokenizer)
        injection_scale = resolve_injection_scale(args.injection_scale, cfg.d_model)
        manifest["injection_scale"] = injection_scale
        av_rows = load_av_rows(av_sft)
        manifest["validation"] = av_dataset_summary(av_rows)

        if not args.prepare_only:
            config, config_error = load_config_from_args(args)
            if config_error is not None:
                manifest["blockers"].append(classify_blocker("config load", config_error))
            model = load_model_from_args(args, config)
            effective_trainable_subset = args.trainable_subset
            if args.peft_method == "lora":
                lora_target_modules = parse_csv_list(args.lora_target_modules)
                lora_modules_to_save = parse_csv_list(args.lora_modules_to_save)
                model, peft_summary = apply_lora_adapters(
                    model,
                    target_modules=lora_target_modules,
                    rank=args.lora_r,
                    alpha=args.lora_alpha,
                    dropout=args.lora_dropout,
                    bias=args.lora_bias,
                    modules_to_save=lora_modules_to_save,
                    use_rslora=args.lora_use_rslora,
                    use_dora=args.lora_use_dora,
                )
                manifest["peft"] = peft_summary
                effective_trainable_subset = "peft:lora"
            manifest["effective_trainable_subset"] = effective_trainable_subset
            model.eval()
            validation_fraction = args.validation_fraction
            test_fraction = args.test_fraction
            if validation_fraction == 0 and test_fraction == 0:
                eval_validation_limit = args.eval_heldout_limit
            else:
                eval_validation_limit = (
                    args.eval_validation_limit
                    if args.eval_validation_limit is not None
                    else args.eval_heldout_limit
                )
            train_indices, validation_indices, test_indices, split_meta = split_indices_for_rows(
                av_rows,
                args.train_fraction,
                validation_fraction,
                test_fraction,
                args.seed,
                split_mode=args.split_mode,
            )
            heldout_indices = validation_indices + test_indices
            manifest["split"] = {
                **split_meta,
                "train_indices": train_indices,
                "heldout_indices": heldout_indices,
                "validation_indices": validation_indices,
                "test_indices": test_indices,
                "train_count": len(train_indices),
                "validation_count": len(validation_indices),
                "test_count": len(test_indices),
                "heldout_count": len(heldout_indices),
            }
            effective_train_steps = args.train_steps
            if args.train_epochs > 0:
                if args.train_batch_size <= 0:
                    raise ValueError("--train-batch-size must be positive")
                effective_train_steps = math.ceil(
                    len(train_indices) * args.train_epochs / args.train_batch_size
                )
            manifest["training_request"] = {
                "train_steps_requested": args.train_steps,
                "train_epochs": args.train_epochs,
                "effective_train_steps": effective_train_steps,
                "train_batch_size": args.train_batch_size,
                "train_sampling": args.train_sampling,
                "effective_train_examples": effective_train_steps * args.train_batch_size,
            }
            if effective_train_steps > 0:
                manifest["training"] = train_actor_sft_smoke(
                    av_sft_path=av_sft,
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    injection_scale=injection_scale,
                    train_indices=train_indices,
                    trainable_subset=effective_trainable_subset,
                    train_steps=effective_train_steps,
                    train_batch_size=args.train_batch_size,
                    train_sampling=args.train_sampling,
                    train_lr=args.train_lr,
                    max_target_tokens=args.max_target_tokens,
                    log_every=args.train_log_every,
                    max_grad_norm=args.max_grad_norm,
                    gradient_checkpointing=args.gradient_checkpointing,
                    respect_existing_trainable=args.peft_method != "none",
                )
                tracker.log_history(manifest["training"].get("history"), prefix="train")
                if args.save_trainable_state:
                    manifest["trainable_state"] = save_trainable_state(
                        model=model,
                        tokenizer=tokenizer,
                        output_path=run_dir / "trainable_state.pt",
                        trainable_subset=effective_trainable_subset,
                        metadata={
                            "model_id": args.model_id,
                            "model_revision": args.model_revision,
                            "input_explained": str(args.input_explained),
                            "av_sft": str(av_sft),
                            "injection_scale": injection_scale,
                            "max_target_tokens": args.max_target_tokens,
                            "train_steps": effective_train_steps,
                            "train_epochs": args.train_epochs,
                            "train_sampling": args.train_sampling,
                            "train_lr": args.train_lr,
                            "peft_method": args.peft_method,
                        },
                    )
            manifest["evaluation"] = evaluate_smoke(
                av_sft_path=av_sft,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                injection_scale=injection_scale,
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
                eval_train_limit=args.eval_train_limit,
                eval_validation_limit=eval_validation_limit,
                eval_test_limit=args.eval_test_limit,
                seed=args.seed,
                max_target_tokens=args.max_target_tokens,
                generate_examples=args.generate_examples,
                max_new_tokens=args.max_new_tokens,
            )
            tracker.log_summary(manifest)
    except Exception as exc:
        manifest["blockers"].append(
            classify_blocker("av warm-start smoke", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}")
        )
        tracker.log_summary(manifest)
        tracker.finish({"status/passed": False, "status/blockers": len(manifest["blockers"])})
        write_json(report_path, manifest)
        print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
        return 1

    tracker.finish({"status/passed": not bool(manifest["blockers"]), "status/blockers": len(manifest["blockers"])})
    write_json(report_path, manifest)
    print(json.dumps(json_safe(manifest), indent=2, sort_keys=True))
    return 0 if not manifest["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
