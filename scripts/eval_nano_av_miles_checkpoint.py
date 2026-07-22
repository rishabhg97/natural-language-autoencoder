#!/usr/bin/env python3
"""Evaluate a Nano AV Miles checkpoint after conversion to HF format.

The Miles actor checkpoints are FSDP DCP directories. This script intentionally
evaluates HF-format checkpoints so the teacher-forced real-vs-control metrics
reuse the same Nano AV semantics as the legacy smoke harness. Convert a DCP
checkpoint first when needed, then point --hf-checkpoint at the converted dir.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if NLA_ROOT.exists() and str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))

from nano_av_warmstart_smoke import (  # noqa: E402
    CONTROL_NAMES,
    generate_with_control,
    load_av_config,
    load_av_rows,
    resolve_injection_scale,
    summarize_losses,
    teacher_forced_loss,
    text_overlap_metrics,
)
from nano_introspection import (  # noqa: E402
    add_bool_optional_arg,
    json_safe,
    load_model_from_args,
    load_tokenizer_from_args,
    write_json,
)
from nano_eval_core import (  # noqa: E402
    select_requested_eval_splits,
    shuffled_control_candidates,
)
from nano_wandb import add_wandb_args, init_wandb  # noqa: E402
from nla.schema import extract_explanation  # noqa: E402


WANDB_METRIC_SCHEMA = (
    "eval/validation/real_nll",
    "eval/test/real_nll",
    "eval/validation/gap_vs_shuffled",
    "eval/test/gap_vs_none",
)


def _read_rows(path: Path, split: str, offset: int) -> list[dict[str, Any]]:
    rows = load_av_rows(path)
    for i, row in enumerate(rows):
        row["row_index"] = offset + i
        row["source_row_index"] = i
        row["split"] = split
    return rows


def _split_indexes(rows: list[dict[str, Any]], split: str) -> list[int]:
    return [int(row["row_index"]) for row in rows if row.get("split") == split]


def _load_eval_rows(
    train_parquet: Path,
    validation_parquet: Path,
    test_parquet: Path | None,
    *,
    eval_splits: list[str],
) -> tuple[list[dict[str, Any]], list[int], list[int], list[int]]:
    train = _read_rows(train_parquet, "train", 0)
    validation = (
        _read_rows(validation_parquet, "validation", len(train))
        if "validation" in eval_splits
        else []
    )
    test = (
        _read_rows(test_parquet, "test", len(train) + len(validation))
        if "test" in eval_splits and test_parquet is not None
        else []
    )
    rows = train + validation + test
    return rows, _split_indexes(rows, "train"), _split_indexes(rows, "validation"), _split_indexes(rows, "test")


def _sample(indices: list[int], limit: int) -> list[int]:
    if limit <= 0 or limit >= len(indices):
        return list(indices)
    return list(indices[:limit])


def _check_hf_checkpoint(path: Path) -> None:
    if (path / "latest_checkpointed_iteration.txt").exists():
        raise SystemExit(
            f"{path} looks like a Miles/FSDP DCP checkpoint root. Convert an iter_*/model "
            "checkpoint to HF format first, then pass the converted directory as --hf-checkpoint."
        )
    if not ((path / "config.json").exists() or not path.exists()):
        raise SystemExit(f"{path} is not a local HF checkpoint directory with config.json")


def build_checkpoint_eval_control_vectors(
    vectors: "torch.Tensor",
    *,
    row_index: int,
    mean_vector: "torch.Tensor",
    seed: int,
    shuffle_candidate_indices: list[int],
) -> dict[str, "torch.Tensor | None"]:
    n_rows, d_model = vectors.shape
    rng = random.Random(seed + row_index * 1009)
    candidates = sorted(
        {
            int(index)
            for index in shuffle_candidate_indices
            if 0 <= int(index) < n_rows and int(index) != row_index
        }
    )
    if not candidates:
        raise ValueError(
            f"row {row_index} has no same-split, cross-family shuffled-control candidate"
        )
    shuffled_index = rng.choice(candidates)
    return {
        "real": vectors[row_index],
        "shuffled": vectors[shuffled_index],
        "zero": torch.zeros(d_model, dtype=vectors.dtype),
        "mean": mean_vector,
        "none": None,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    hf_checkpoint = Path(args.hf_checkpoint)
    if hf_checkpoint.exists():
        _check_hf_checkpoint(hf_checkpoint)

    tokenizer = load_tokenizer_from_args(args)
    model = load_model_from_args(args)
    model.eval()
    cfg = load_av_config(args.validation_parquet, tokenizer)
    injection_scale = resolve_injection_scale(args.injection_scale, cfg.d_model)

    rows, train_indices, validation_indices, test_indices = _load_eval_rows(
        args.train_parquet,
        args.validation_parquet,
        args.test_parquet,
        eval_splits=args.eval_splits,
    )
    vectors = torch.tensor([row["activation_vector"] for row in rows], dtype=torch.float32)
    if not train_indices:
        raise ValueError("training rows are required to construct the mean control")
    mean_source = train_indices
    mean_vector = vectors[mean_source].mean(dim=0)
    selected_indices = select_requested_eval_splits(
        args.eval_splits,
        validation=_sample(validation_indices, args.validation_limit),
        test=_sample(test_indices, args.test_limit),
    )
    eval_indices = [
        row_index
        for split in args.eval_splits
        for row_index in selected_indices[split]
    ]
    validation_set = set(validation_indices)
    test_set = set(test_indices)

    losses: dict[str, list[dict[str, Any]]] = {name: [] for name in CONTROL_NAMES}
    for row_index in eval_indices:
        row = rows[row_index]
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
        for name in CONTROL_NAMES:
            item = teacher_forced_loss(
                model,
                tokenizer,
                cfg,
                row,
                controls[name],
                injection_scale=injection_scale,
                max_target_tokens=args.max_target_tokens,
            )
            item.update({"row_index": row_index, "split": row["split"], "control": name})
            losses[name].append(item)

    loss_summary = {
        name: summarize_losses(items, validation_set, test_set) for name, items in losses.items()
    }
    for name in ("shuffled", "zero", "mean", "none"):
        for split in args.eval_splits:
            real_loss = loss_summary["real"][split]["loss"]
            control_loss = loss_summary[name][split]["loss"]
            loss_summary["real"][f"{split}_loss_gap_vs_{name}"] = (
                None if real_loss is None or control_loss is None else control_loss - real_loss
            )

    examples = []
    example_indices = (
        validation_indices[: max(0, args.generation_examples)]
        if "validation" in args.eval_splits
        else []
    )
    for row_index in example_indices:
        row = rows[row_index]
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
        target = extract_explanation(row["response"]) or row["response"]
        item = {
            "row_index": row_index,
            "split": row["split"],
            "doc_id": row.get("doc_id"),
            "target_excerpt": target[:500],
            "controls": {},
        }
        for name in ("real", "shuffled", "zero", "none"):
            generated = generate_with_control(
                model,
                tokenizer,
                cfg,
                row,
                controls[name],
                injection_scale=injection_scale,
                max_new_tokens=args.max_new_tokens,
            )
            item["controls"][name] = {
                "generated": generated,
                "metrics": text_overlap_metrics(generated, target),
            }
        examples.append(item)

    return {
        "format": "nano_av_miles_checkpoint_eval.v1",
        "hf_checkpoint": str(args.hf_checkpoint),
        "train_parquet": str(args.train_parquet),
        "validation_parquet": str(args.validation_parquet),
        "test_parquet": None if args.test_parquet is None else str(args.test_parquet),
        "row_count": len(rows),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "test_count": len(test_indices),
        "eval_splits": list(args.eval_splits),
        "eval_validation_count": len(selected_indices.get("validation", [])),
        "eval_test_count": len(selected_indices.get("test", [])),
        "injection_scale": injection_scale,
        "loss_summary": loss_summary,
        "examples": examples,
    }


def build_wandb_eval_metrics(report: dict[str, Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "eval/row_count": int(report["row_count"]),
        "eval/train_count": int(report["train_count"]),
        "eval/validation_count": int(report["validation_count"]),
        "eval/test_count": int(report["test_count"]),
        "eval/eval_validation_count": int(report["eval_validation_count"]),
        "eval/eval_test_count": int(report["eval_test_count"]),
    }
    injection_scale = report.get("injection_scale")
    if isinstance(injection_scale, (int, float)):
        metrics["eval/injection_scale"] = float(injection_scale)

    loss_summary = report.get("loss_summary") or {}
    for control_name, control_summary in loss_summary.items():
        if not isinstance(control_summary, dict):
            continue
        for split in ("validation", "test"):
            split_summary = control_summary.get(split)
            if not isinstance(split_summary, dict):
                continue
            loss = split_summary.get("loss")
            count = split_summary.get("count")
            if isinstance(loss, (int, float)):
                metrics[f"eval/{split}/{control_name}_nll"] = float(loss)
            if isinstance(count, int):
                metrics[f"eval/{split}/{control_name}_count"] = count

    real_summary = loss_summary.get("real") if isinstance(loss_summary, dict) else None
    if isinstance(real_summary, dict):
        for split in ("validation", "test"):
            real_key = f"eval/{split}/real_nll"
            real_loss = metrics.get(real_key)
            if isinstance(real_loss, (int, float)):
                metrics[f"eval/{split}/loss"] = float(real_loss)
            for control_name in ("shuffled", "zero", "mean", "none"):
                gap = real_summary.get(f"{split}_loss_gap_vs_{control_name}")
                if isinstance(gap, (int, float)):
                    metrics[f"eval/{split}/gap_vs_{control_name}"] = float(gap)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--validation-limit", type=int, default=32)
    parser.add_argument("--test-limit", type=int, default=32)
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=("validation", "test"),
        default=["validation"],
    )
    parser.add_argument("--generation-examples", type=int, default=0)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--injection-scale", default="75")
    parser.add_argument("--model-id", dest="model_id", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--load-mode", choices=("full", "meta", "config"), default="full")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    add_bool_optional_arg(parser, "--trust-remote-code", default=True)
    parser.add_argument("--wandb-step", type=int, default=None)
    add_wandb_args(parser)
    args = parser.parse_args(argv)
    if "test" in args.eval_splits and args.test_parquet is None:
        parser.error("--test-parquet is required when --eval-splits includes test")
    args.model_id = args.hf_checkpoint
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(args)
    write_json(args.report_json, json_safe(report))
    tracker = init_wandb(
        args,
        run_dir=args.report_json.parent,
        job_type="checkpoint_eval",
        config={
            "hf_checkpoint": str(args.hf_checkpoint),
            "train_parquet": str(args.train_parquet),
            "validation_parquet": str(args.validation_parquet),
            "test_parquet": None if args.test_parquet is None else str(args.test_parquet),
            "validation_limit": args.validation_limit,
            "test_limit": args.test_limit,
            "eval_splits": list(args.eval_splits),
            "generation_examples": args.generation_examples,
            "injection_scale": args.injection_scale,
        },
    )
    metrics = build_wandb_eval_metrics(json_safe(report))
    tracker.log(metrics, step=args.wandb_step)
    tracker.finish(metrics)
    print(json.dumps(json_safe(report), indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
