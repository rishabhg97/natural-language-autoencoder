#!/usr/bin/env python3
"""Evaluate Nano AR-SFT critic checkpoints against heldout reconstruction controls."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
for candidate in (SCRIPT_DIR, NLA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from verify_nano_miles_av_dataset import sidecar_path_for  # noqa: E402


DEFAULT_GENERIC_EXPLANATION = "This text describes a generic activation pattern."
DEFAULT_CONTROLS = ("teacher", "teacher_shuffled", "blank", "generic", "mean", "source_context", "source_raw")


def l2_normalize_rows(array: np.ndarray, target_scale: float | None = None, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if target_scale is None:
        target_scale = math.sqrt(float(arr.shape[-1]))
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return target_scale * arr / np.maximum(norms, eps)


def normalized_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred = l2_normalize_rows(predictions)
    gold = l2_normalize_rows(targets)
    return float(np.mean(np.square(pred - gold)))


def rowwise_normalized_mse(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pred = l2_normalize_rows(predictions)
    gold = l2_normalize_rows(targets)
    return np.mean(np.square(pred - gold), axis=-1)


def rowwise_cosine(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pred = np.asarray(predictions, dtype=np.float32)
    gold = np.asarray(targets, dtype=np.float32)
    pred_n = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), eps)
    gold_n = gold / np.maximum(np.linalg.norm(gold, axis=-1, keepdims=True), eps)
    return np.sum(pred_n * gold_n, axis=-1)


def bootstrap_ci(values: np.ndarray, *, samples: int, seed: int, ci: float = 0.95) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    n = int(arr.shape[0])
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = float(arr.mean())
    if samples <= 0:
        return {"n": n, "mean": mean, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(samples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return {
        "n": n,
        "mean": mean,
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
    }


def build_prediction_dump_rows(
    *,
    split_name: str,
    control_name: str,
    row_indices: list[int],
    doc_ids: list[str | None],
    predictions: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, Any]]:
    pred = np.asarray(predictions, dtype=np.float32)
    gold = np.asarray(targets, dtype=np.float32)
    losses = rowwise_normalized_mse(pred, gold)
    cosines = rowwise_cosine(pred, gold)
    pred_norms = np.linalg.norm(pred, axis=-1)
    gold_norms = np.linalg.norm(gold, axis=-1)
    rows: list[dict[str, Any]] = []
    for idx in range(pred.shape[0]):
        rows.append(
            {
                "split": split_name,
                "control": control_name,
                "row_index": int(row_indices[idx]),
                "doc_id": None if doc_ids[idx] is None else str(doc_ids[idx]),
                "normalized_mse": float(losses[idx]),
                "cosine": float(cosines[idx]),
                "pred_norm": float(pred_norms[idx]),
                "gold_norm": float(gold_norms[idx]),
            }
        )
    return rows


def rowwise_win_summary(
    teacher_losses: np.ndarray,
    control_losses: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    teacher = np.asarray(teacher_losses, dtype=np.float32)
    control = np.asarray(control_losses, dtype=np.float32)
    if teacher.shape != control.shape:
        raise ValueError(f"teacher/control rowwise losses must have same shape, got {teacher.shape} vs {control.shape}")
    teacher_better = teacher < (control - eps)
    control_better = control < (teacher - eps)
    ties = ~(teacher_better | control_better)
    row_count = int(teacher.shape[0])
    return {
        "row_count": row_count,
        "teacher_better_count": int(np.count_nonzero(teacher_better)),
        "teacher_better_fraction": float(np.mean(teacher_better)) if row_count else 0.0,
        "tie_count": int(np.count_nonzero(ties)),
        "tie_fraction": float(np.mean(ties)) if row_count else 0.0,
        "control_better_count": int(np.count_nonzero(control_better)),
        "control_better_fraction": float(np.mean(control_better)) if row_count else 0.0,
        "mean_loss_delta_control_minus_teacher": float(np.mean(control - teacher)) if row_count else 0.0,
    }


def cosine_mean(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.asarray(predictions, dtype=np.float32)
    gold = np.asarray(targets, dtype=np.float32)
    pred_n = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), eps)
    gold_n = gold / np.maximum(np.linalg.norm(gold, axis=-1, keepdims=True), eps)
    return float(np.mean(np.sum(pred_n * gold_n, axis=-1)))


def metric_summary(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    train_targets_for_mean: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, float | None]:
    pred = np.asarray(predictions, dtype=np.float32)
    gold = np.asarray(targets, dtype=np.float32)
    train_targets = np.asarray(train_targets_for_mean, dtype=np.float32)
    train_mean = train_targets.mean(axis=0, keepdims=True)
    mean_prediction = np.repeat(train_mean, gold.shape[0], axis=0)
    norm_mse = normalized_mse(pred, gold)
    mean_norm_mse = normalized_mse(mean_prediction, gold)
    raw_sse = float(np.square(pred - gold).sum())
    centered_sst = float(np.square(gold - train_mean).sum())
    return {
        "row_count": int(gold.shape[0]),
        "normalized_mse": norm_mse,
        "raw_mse": float(np.mean(np.square(pred - gold))),
        "cosine_mean": cosine_mean(pred, gold),
        "mean_control_normalized_mse": mean_norm_mse,
        "fve_nrm": None if mean_norm_mse <= eps else float(1.0 - norm_mse / mean_norm_mse),
        "centered_raw_r2": None if centered_sst <= eps else float(1.0 - raw_sse / centered_sst),
        "centered_raw_sse": raw_sse,
        "centered_raw_sst": centered_sst,
    }


def deranged_indices(count: int, seed: int) -> list[int]:
    indices = list(range(count))
    if count <= 1:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    for idx, source_idx in enumerate(indices):
        if source_idx == idx:
            swap_idx = (idx + 1) % count
            indices[idx], indices[swap_idx] = indices[swap_idx], indices[idx]
    return indices


def _resolve_hf_dir(checkpoint_dir: str | Path) -> Path:
    path = Path(checkpoint_dir)
    if (path / "hf" / "config.json").is_file():
        return path / "hf"
    return path


def _sidecar_template(*paths: str | Path | None) -> str:
    for path in paths:
        if path is None:
            continue
        candidate = Path(path)
        sidecar = candidate / "nla_meta.yaml" if candidate.is_dir() else sidecar_path_for(candidate)
        if sidecar.is_file():
            meta = yaml.safe_load(sidecar.read_text())
            template = (meta.get("prompt_templates") or {}).get("critic")
            if isinstance(template, str) and "{explanation}" in template:
                return template
    raise ValueError("could not resolve critic prompt template from checkpoint or dataset sidecar")


def _read_rows(path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    if "prompt" not in columns or "activation_vector" not in columns:
        raise ValueError(f"{path} must contain prompt and activation_vector")
    targets = np.asarray(columns["activation_vector"], dtype=np.float32)
    return {
        "path": str(path),
        "row_count": int(table.num_rows),
        "columns": table.column_names,
        "row_indices": [
            int(item) if item is not None else idx
            for idx, item in enumerate(columns.get("row_index", list(range(table.num_rows))))
        ],
        "doc_ids": [
            str(item) if item is not None else None
            for item in columns.get("doc_id", [None] * table.num_rows)
        ],
        "prompts": [str(item) for item in columns["prompt"]],
        "targets": targets,
        "source_contexts": [
            str(item) if item is not None else ""
            for item in columns.get("detokenized_text_truncated", [""] * table.num_rows)
        ],
        "source_raw_ids": columns.get("token_ids_prefix"),
    }


def _torch_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    return getattr(torch, name)


def _patch_remote_code_for_eval(hf_dir: Path) -> None:
    from nla.remote_code_patches import prepare_nemotron_h_checkpoint_for_load

    prepare_nemotron_h_checkpoint_for_load(hf_dir)


def _load_tokenizer_for_eval(hf_dir: Path):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend does not exist" not in str(exc):
            raise
        tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_dir, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _direct_device_map(device_map: str) -> str | dict[str, int]:
    """Resolve a config-friendly CUDA device string to a Hugging Face device map."""

    normalized = str(device_map).strip()
    if normalized == "single_gpu":
        return {"": 0}
    if normalized.startswith("cuda:"):
        device_index = normalized.removeprefix("cuda:")
        if not device_index.isdigit():
            raise ValueError(
                "AR device map must use cuda:<non-negative-integer>, "
                f"got {device_map!r}"
            )
        return {"": int(device_index)}
    return normalized


def _load_model_and_tokenizer(
    hf_dir: Path,
    *,
    torch_dtype: str,
    device_map: str,
    low_cpu_mem_usage: bool = False,
):
    import torch
    from nla.models import NLACriticModel

    _patch_remote_code_for_eval(hf_dir)
    tokenizer = _load_tokenizer_for_eval(hf_dir)
    kwargs: dict[str, Any] = {"torch_dtype": _torch_dtype(torch_dtype)}
    if device_map != "none":
        resolved_device_map = _direct_device_map(device_map)
        kwargs["device_map"] = resolved_device_map
        # A direct one-GPU map must stream tensors into its destination. Without
        # this, the fallback materializes the whole model on CPU then moves each
        # of thousands of tensors individually during model.cuda().
        if low_cpu_mem_usage or isinstance(resolved_device_map, dict):
            kwargs["low_cpu_mem_usage"] = True
    model = NLACriticModel.from_pretrained(hf_dir, **kwargs)
    if device_map == "none" and torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model, tokenizer


def _input_device(model: Any):
    return next(model.parameters()).device


def _predict_token_batches(
    model: Any,
    token_batches: list[list[int]],
    *,
    pad_token_id: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    device = _input_device(model)
    outputs: list[np.ndarray] = []
    for start in range(0, len(token_batches), batch_size):
        chunk = token_batches[start : start + batch_size]
        max_len = max(len(ids) for ids in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for row, ids in enumerate(chunk):
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            input_ids[row, : len(ids)] = ids_t
            attention_mask[row, : len(ids)] = 1
        with torch.no_grad():
            values = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).values.float()
            value_device = values.device
            last_idx = (attention_mask.sum(dim=1) - 1).to(value_device)
            pred = values[torch.arange(len(chunk), device=value_device), last_idx]
        outputs.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def predict_prompts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    batch_size: int,
    max_length: int | None,
) -> np.ndarray:
    token_batches: list[list[int]] = []
    tokenizer.truncation_side = "left"
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=max_length is not None,
            max_length=max_length,
        )["input_ids"]
        token_batches.append([int(tok) for tok in encoded])
    return _predict_token_batches(model, token_batches, pad_token_id=tokenizer.pad_token_id, batch_size=batch_size)


def _prompt_from_template(template: str, text: str) -> str:
    return template.format(explanation=text)


def _evaluate_split(
    model: Any,
    tokenizer: Any,
    split: dict[str, Any],
    *,
    split_name: str,
    train_targets: np.ndarray,
    critic_template: str,
    controls: tuple[str, ...],
    batch_size: int,
    max_length: int | None,
    generic_explanation: str,
    seed: int,
    prediction_dump_rows: list[dict[str, Any]] | None = None,
    bootstrap_samples: int = 0,
) -> dict[str, Any]:
    targets = split["targets"]
    prompts = split["prompts"]
    row_indices = split["row_indices"]
    doc_ids = split["doc_ids"]
    shuffled = deranged_indices(len(prompts), seed)
    source_context_prompts = [
        _prompt_from_template(critic_template, text)
        for text in split["source_contexts"]
        if text
    ]
    source_raw_ids = [
        [int(tok) for tok in ids]
        for ids in (split.get("source_raw_ids") or [])
        if isinstance(ids, list) and ids
    ]

    out: dict[str, Any] = {
        "row_count": int(targets.shape[0]),
        "controls": {},
        "skipped_controls": [],
        "rowwise_win_rates": {},
    }
    rowwise_losses: dict[str, np.ndarray] = {}

    def record_prediction(control_name: str, pred: np.ndarray) -> None:
        if prediction_dump_rows is None:
            return
        prediction_dump_rows.extend(
            build_prediction_dump_rows(
                split_name=split_name,
                control_name=control_name,
                row_indices=row_indices,
                doc_ids=doc_ids,
                predictions=pred,
                targets=targets,
            )
        )

    prompt_controls = {
        "teacher": prompts,
        "teacher_shuffled": [prompts[i] for i in shuffled],
        "blank": [_prompt_from_template(critic_template, "") for _ in prompts],
        "generic": [_prompt_from_template(critic_template, generic_explanation) for _ in prompts],
    }
    for name in controls:
        if name == "mean":
            mean_pred = np.repeat(train_targets.mean(axis=0, keepdims=True), targets.shape[0], axis=0)
            out["controls"][name] = metric_summary(
                predictions=mean_pred,
                targets=targets,
                train_targets_for_mean=train_targets,
            )
            rowwise_losses[name] = rowwise_normalized_mse(mean_pred, targets)
            record_prediction(name, mean_pred)
        elif name in prompt_controls:
            pred = predict_prompts(
                model,
                tokenizer,
                prompt_controls[name],
                batch_size=batch_size,
                max_length=max_length,
            )
            out["controls"][name] = metric_summary(
                predictions=pred,
                targets=targets,
                train_targets_for_mean=train_targets,
            )
            rowwise_losses[name] = rowwise_normalized_mse(pred, targets)
            record_prediction(name, pred)
        elif name == "source_context" and len(source_context_prompts) == len(prompts):
            pred = predict_prompts(
                model,
                tokenizer,
                source_context_prompts,
                batch_size=batch_size,
                max_length=max_length,
            )
            out["controls"][name] = metric_summary(
                predictions=pred,
                targets=targets,
                train_targets_for_mean=train_targets,
            )
            rowwise_losses[name] = rowwise_normalized_mse(pred, targets)
            record_prediction(name, pred)
        elif name == "source_raw" and len(source_raw_ids) == len(prompts):
            pred = _predict_token_batches(
                model,
                source_raw_ids,
                pad_token_id=tokenizer.pad_token_id,
                batch_size=batch_size,
            )
            out["controls"][name] = metric_summary(
                predictions=pred,
                targets=targets,
                train_targets_for_mean=train_targets,
            )
            rowwise_losses[name] = rowwise_normalized_mse(pred, targets)
            record_prediction(name, pred)
        else:
            out["skipped_controls"].append(name)
    if "teacher" in rowwise_losses:
        teacher_losses = rowwise_losses["teacher"]
        for name, losses in rowwise_losses.items():
            if name == "teacher":
                continue
            out["rowwise_win_rates"][f"teacher_vs_{name}"] = rowwise_win_summary(teacher_losses, losses)
        if bootstrap_samples > 0:
            out["bootstrap_ci"] = {
                "teacher": bootstrap_ci(teacher_losses, samples=bootstrap_samples, seed=seed),
                "teacher_vs_controls": {
                    name: bootstrap_ci(losses - teacher_losses, samples=bootstrap_samples, seed=seed + idx + 1)
                    for idx, (name, losses) in enumerate(rowwise_losses.items())
                    if name != "teacher"
                },
            }
    return out


def main() -> int:
    from nano_eval_core import select_requested_eval_splits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="Critic HF dir or Miles iter dir containing hf/.")
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--validation-parquet", required=True, type=Path)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=64)
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("validation", "test"),
        default=["validation"],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="Pass 'none' to load on the default torch device.")
    parser.add_argument("--generic-explanation", default=DEFAULT_GENERIC_EXPLANATION)
    parser.add_argument("--critic-template")
    parser.add_argument("--controls", nargs="+", default=list(DEFAULT_CONTROLS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-dump-jsonl", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int)
    args = parser.parse_args()
    if "test" in args.eval_splits and args.test_parquet is None:
        parser.error("--test-parquet is required when --eval-splits includes test")

    hf_dir = _resolve_hf_dir(args.checkpoint_dir)
    critic_template = args.critic_template or _sidecar_template(hf_dir, args.validation_parquet, args.train_parquet)
    train = _read_rows(args.train_parquet)
    split_specs = select_requested_eval_splits(
        args.eval_splits,
        validation=(args.validation_parquet, args.validation_limit),
        test=(args.test_parquet, args.test_limit),
    )
    split_data = {
        split: _read_rows(path, limit=limit)
        for split, (path, limit) in split_specs.items()
    }
    model, tokenizer = _load_model_and_tokenizer(hf_dir, torch_dtype=args.torch_dtype, device_map=args.device_map)
    controls = tuple(args.controls)
    prediction_rows: list[dict[str, Any]] | None = [] if args.prediction_dump_jsonl else None
    bootstrap_seed = args.seed if args.bootstrap_seed is None else args.bootstrap_seed
    evaluated_splits = {}
    for split_name, split in split_data.items():
        evaluated_splits[split_name] = _evaluate_split(
            model,
            tokenizer,
            split,
            split_name=split_name,
            train_targets=train["targets"],
            critic_template=critic_template,
            controls=controls,
            batch_size=args.batch_size,
            max_length=args.max_length,
            generic_explanation=args.generic_explanation,
            seed=bootstrap_seed + (1 if split_name == "test" else 0),
            prediction_dump_rows=prediction_rows,
            bootstrap_samples=args.bootstrap_samples,
        )
    report = {
        "schema_version": "nano_ar_checkpoint_eval.v1",
        "checkpoint_dir": str(args.checkpoint_dir),
        "hf_dir": str(hf_dir),
        "critic_template": critic_template,
        "controls_requested": list(controls),
        "eval_splits": list(args.eval_splits),
        "splits": evaluated_splits,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.prediction_dump_jsonl and prediction_rows is not None:
        args.prediction_dump_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.prediction_dump_jsonl.open("w") as f:
            for row in prediction_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
