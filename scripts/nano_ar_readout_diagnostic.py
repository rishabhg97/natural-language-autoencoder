#!/usr/bin/env python3
"""Diagnose whether trained Nano AR hidden states contain a better readout than the current value head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
for candidate in (SCRIPT_DIR, NLA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from eval_nano_ar_miles_checkpoint import (  # noqa: E402
    _load_model_and_tokenizer,
    _read_rows,
    _resolve_hf_dir,
    _sidecar_template,
    metric_summary,
    rowwise_normalized_mse,
)


class RidgeReadout(NamedTuple):
    weights: np.ndarray
    bias: bool
    ridge: float
    transform: str


def row_rms_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    rms = np.sqrt(np.mean(np.square(arr), axis=1, keepdims=True))
    return arr / np.maximum(rms, eps)


def transform_features(x: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return np.asarray(x, dtype=np.float32)
    if transform == "rmsnorm":
        return row_rms_normalize(x)
    raise ValueError(f"unknown feature transform {transform!r}")


def _augment_bias(x: np.ndarray) -> np.ndarray:
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.concatenate([x, ones], axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, *, ridge: float, bias: bool, transform: str = "raw") -> RidgeReadout:
    features = transform_features(x, transform).astype(np.float64)
    targets = np.asarray(y, dtype=np.float64)
    if bias:
        features = _augment_bias(features)
    xtx = features.T @ features
    regularizer = float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
    if bias:
        regularizer[-1, -1] = 0.0
    xty = features.T @ targets
    weights = np.linalg.solve(xtx + regularizer, xty)
    return RidgeReadout(weights=weights.astype(np.float32), bias=bias, ridge=float(ridge), transform=transform)


def predict_ridge(x: np.ndarray, model: RidgeReadout) -> np.ndarray:
    features = transform_features(x, model.transform).astype(np.float32)
    if model.bias:
        features = _augment_bias(features)
    return (features @ model.weights).astype(np.float32)


def _token_batches(tokenizer: Any, prompts: list[str], *, max_length: int | None) -> list[list[int]]:
    tokenizer.truncation_side = "left"
    batches: list[list[int]] = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=max_length is not None,
            max_length=max_length,
        )["input_ids"]
        batches.append([int(tok) for tok in encoded])
    return batches


def collect_last_token_features(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    batch_size: int,
    max_length: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    token_batches = _token_batches(tokenizer, prompts, max_length=max_length)
    device = next(model.parameters()).device
    hidden_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    for start in range(0, len(token_batches), batch_size):
        chunk = token_batches[start : start + batch_size]
        max_len = max(len(ids) for ids in chunk)
        input_ids = torch.full((len(chunk), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for row, ids in enumerate(chunk):
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            input_ids[row, : len(ids)] = ids_t
            attention_mask[row, : len(ids)] = 1
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden = out.backbone_last_hidden.float()
            values = out.values.float()
            value_device = values.device
            last_idx = (attention_mask.sum(dim=1) - 1).to(value_device)
            batch_idx = torch.arange(len(chunk), device=value_device)
            hidden_last = hidden[batch_idx, last_idx]
            pred_last = values[batch_idx, last_idx]
        hidden_rows.append(hidden_last.detach().cpu().numpy().astype(np.float32))
        pred_rows.append(pred_last.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(hidden_rows, axis=0), np.concatenate(pred_rows, axis=0)


def summarize_readout(
    *,
    name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    train_targets: np.ndarray,
) -> dict[str, Any]:
    summary = metric_summary(predictions=predictions, targets=targets, train_targets_for_mean=train_targets)
    losses = rowwise_normalized_mse(predictions, targets)
    summary["rowwise_nmse_p50"] = float(np.quantile(losses, 0.50))
    summary["rowwise_nmse_p90"] = float(np.quantile(losses, 0.90))
    summary["name"] = name
    return summary


def parse_ridges(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--train-parquet", required=True, type=Path)
    parser.add_argument("--validation-parquet", required=True, type=Path)
    parser.add_argument("--test-parquet", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument("--validation-limit", type=int, default=512)
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--critic-template")
    parser.add_argument("--ridges", default="0.001,0.01,0.1,1,10,100")
    args = parser.parse_args()

    hf_dir = _resolve_hf_dir(args.checkpoint_dir)
    critic_template = args.critic_template or _sidecar_template(hf_dir, args.validation_parquet, args.train_parquet)
    train = _read_rows(args.train_parquet, limit=args.train_limit)
    validation = _read_rows(args.validation_parquet, limit=args.validation_limit)
    test = _read_rows(args.test_parquet, limit=args.test_limit)
    model, tokenizer = _load_model_and_tokenizer(hf_dir, torch_dtype=args.torch_dtype, device_map=args.device_map)
    model.eval()

    train_hidden, train_current = collect_last_token_features(
        model, tokenizer, train["prompts"], batch_size=args.batch_size, max_length=args.max_length
    )
    val_hidden, val_current = collect_last_token_features(
        model, tokenizer, validation["prompts"], batch_size=args.batch_size, max_length=args.max_length
    )
    test_hidden, test_current = collect_last_token_features(
        model, tokenizer, test["prompts"], batch_size=args.batch_size, max_length=args.max_length
    )

    ridges = parse_ridges(args.ridges)
    validation_results: list[dict[str, Any]] = [
        summarize_readout(
            name="current_value_head",
            predictions=val_current,
            targets=validation["targets"],
            train_targets=train["targets"],
        )
    ]
    test_results: dict[str, dict[str, Any]] = {
        "current_value_head": summarize_readout(
            name="current_value_head",
            predictions=test_current,
            targets=test["targets"],
            train_targets=train["targets"],
        )
    }

    fitted: dict[str, RidgeReadout] = {}
    for transform in ("raw", "rmsnorm"):
        for bias in (False, True):
            for ridge in ridges:
                readout = fit_ridge(train_hidden, train["targets"], ridge=ridge, bias=bias, transform=transform)
                key = f"ridge_{transform}_{'bias' if bias else 'nobias'}_{ridge:g}"
                val_pred = predict_ridge(val_hidden, readout)
                validation_results.append(
                    summarize_readout(
                        name=key,
                        predictions=val_pred,
                        targets=validation["targets"],
                        train_targets=train["targets"],
                    )
                )
                fitted[key] = readout

    best = min(validation_results, key=lambda item: float(item["normalized_mse"]))
    best_name = str(best["name"])
    if best_name in fitted:
        test_pred = predict_ridge(test_hidden, fitted[best_name])
        test_results[best_name] = summarize_readout(
            name=best_name,
            predictions=test_pred,
            targets=test["targets"],
            train_targets=train["targets"],
        )

    report = {
        "schema_version": "nano_ar_readout_diagnostic.v1",
        "checkpoint_dir": str(args.checkpoint_dir),
        "hf_dir": str(hf_dir),
        "critic_template": critic_template,
        "limits": {
            "train": int(train["targets"].shape[0]),
            "validation": int(validation["targets"].shape[0]),
            "test": int(test["targets"].shape[0]),
        },
        "feature_shape": {
            "train_hidden": list(train_hidden.shape),
            "validation_hidden": list(val_hidden.shape),
            "test_hidden": list(test_hidden.shape),
        },
        "validation": sorted(validation_results, key=lambda item: float(item["normalized_mse"])),
        "test": test_results,
        "best_validation_name": best_name,
        "decision": {
            "head_bottleneck_likely": float(best["normalized_mse"]) <= 0.40,
            "strong_head_bottleneck_signal": float(best["normalized_mse"]) <= 0.35,
            "best_validation_nmse": float(best["normalized_mse"]),
            "current_validation_nmse": float(validation_results[0]["normalized_mse"]),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
