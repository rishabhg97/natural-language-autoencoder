#!/usr/bin/env python3
"""Summarize Nano AV warm-start reports.

The AV smoke report is the source of truth for scientific gates. W&B is useful
for curves, but this script keeps the end-of-run comparison reproducible from a
single `av_warmstart_smoke.json` file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTROL_ORDER = ("real", "shuffled", "zero", "mean", "none")


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _loss(loss_summary: dict[str, Any], control: str, split: str = "heldout") -> float | None:
    item = loss_summary.get(control)
    if not isinstance(item, dict):
        return None
    split_item = item.get(split)
    if not isinstance(split_item, dict):
        return None
    value = split_item.get("loss")
    return float(value) if isinstance(value, (int, float)) else None


def _mean_example_metric(examples: list[dict[str, Any]], control: str, metric: str) -> float | None:
    values: list[float] = []
    for example in examples:
        item = _example_control(example, control)
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else item
        value = metrics.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _parsed_count(examples: list[dict[str, Any]], control: str) -> int:
    count = 0
    for example in examples:
        item = _example_control(example, control)
        if not isinstance(item, dict):
            continue
        generated = item.get("generated")
        parsed = item.get("parsed_explanation")
        if parsed or (isinstance(generated, str) and "</explanation>" in generated):
            count += 1
    return count


def _example_control(example: dict[str, Any], control: str) -> dict[str, Any] | None:
    controls = example.get("controls")
    if isinstance(controls, dict):
        aliases = {
            "no_injection": "none",
            "none": "none",
        }
        item = controls.get(aliases.get(control, control))
        return item if isinstance(item, dict) else None
    item = example.get(control)
    return item if isinstance(item, dict) else None


def _state_summary(report: dict[str, Any], run_dir: Path | None) -> dict[str, Any]:
    state = report.get("trainable_state")
    if not isinstance(state, dict):
        return {}
    result = {
        "path": state.get("path"),
        "bytes": state.get("bytes"),
        "tensor_count": state.get("tensor_count"),
    }
    path_text = state.get("path")
    if isinstance(path_text, str):
        path = Path(path_text)
        if not path.exists() and run_dir is not None:
            path = run_dir / Path(path_text).name
        if path.exists():
            result["size_gib"] = path.stat().st_size / (1024**3)
    return result


def _wandb_offline_dirs(run_dir: Path | None) -> list[str]:
    if run_dir is None:
        return []
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return []
    return [str(path) for path in sorted(wandb_dir.glob("offline-run-*"))]


def summarize_report(report: dict[str, Any], run_dir: Path | None = None) -> dict[str, Any]:
    evaluation = report.get("evaluation") if isinstance(report.get("evaluation"), dict) else {}
    loss_summary = evaluation.get("loss_summary") if isinstance(evaluation.get("loss_summary"), dict) else {}
    examples = evaluation.get("examples") if isinstance(evaluation.get("examples"), list) else []
    real = loss_summary.get("real") if isinstance(loss_summary.get("real"), dict) else {}
    split = report.get("split") if isinstance(report.get("split"), dict) else {}
    peft = report.get("peft") if isinstance(report.get("peft"), dict) else {}
    training = report.get("training") if isinstance(report.get("training"), dict) else {}
    training_request = (
        report.get("training_request") if isinstance(report.get("training_request"), dict) else {}
    )
    history = training.get("history") if isinstance(training.get("history"), list) else []

    requested_trainable = report.get("requested_trainable_subset", report.get("trainable_subset"))
    effective_trainable = report.get("effective_trainable_subset", report.get("trainable_subset"))
    warnings = []
    if requested_trainable != effective_trainable:
        warnings.append(
            f"requested_trainable_subset={requested_trainable} effective_trainable_subset={effective_trainable}"
        )

    return {
        "blockers": report.get("blockers", []),
        "warnings": warnings,
        "split": {
            "train_count": split.get("train_count"),
            "validation_count": split.get("validation_count"),
            "test_count": split.get("test_count"),
            "heldout_count": split.get("heldout_count"),
            "doc_overlap_count": split.get("doc_overlap_count"),
        },
        "peft": {
            "requested_trainable_subset": requested_trainable,
            "effective_trainable_subset": effective_trainable,
            "peft_method": report.get("peft_method"),
            "lora_rank": peft.get("lora_rank"),
            "lora_alpha": peft.get("lora_alpha"),
            "lora_use_rslora": peft.get("lora_use_rslora"),
            "lora_use_dora": peft.get("lora_use_dora"),
            "trainable_parameters": peft.get("trainable_parameters"),
            "trainable_fraction": peft.get("trainable_fraction"),
        },
        "training": {
            "train_steps": training.get("train_steps"),
            "train_batch_size": training.get("train_batch_size"),
            "train_sampling": training.get("train_sampling"),
            "train_epochs": training_request.get("train_epochs"),
            "effective_train_examples": training_request.get("effective_train_examples"),
            "train_unique_indices_seen": training.get("train_unique_indices_seen"),
            "history_tail": history[-5:],
            "final_logged_loss": history[-1].get("loss") if history and isinstance(history[-1], dict) else None,
        },
        "heldout_losses": {
            control: _loss(loss_summary, control, split="heldout") for control in CONTROL_ORDER
        },
        "validation_losses": {
            control: _loss(loss_summary, control, split="validation") for control in CONTROL_ORDER
        },
        "test_losses": {
            control: _loss(loss_summary, control, split="test") for control in CONTROL_ORDER
        },
        "real_heldout_gaps": {
            key: value for key, value in real.items() if isinstance(key, str) and "heldout_loss_gap" in key
        },
        "real_validation_gaps": {
            key: value for key, value in real.items() if isinstance(key, str) and "validation_loss_gap" in key
        },
        "real_test_gaps": {
            key: value for key, value in real.items() if isinstance(key, str) and "test_loss_gap" in key
        },
        "examples": {
            "count": len(examples),
            "parsed_real": _parsed_count(examples, "real"),
            "mean_content_f1_real": _mean_example_metric(examples, "real", "content_f1"),
            "mean_content_f1_shuffled": _mean_example_metric(examples, "shuffled", "content_f1"),
            "mean_content_f1_zero": _mean_example_metric(examples, "zero", "content_f1"),
            "mean_content_f1_none": _mean_example_metric(examples, "no_injection", "content_f1"),
        },
        "wandb": {
            **(report.get("wandb") if isinstance(report.get("wandb"), dict) else {}),
            "offline_dirs": _wandb_offline_dirs(run_dir),
        },
        "trainable_state": _state_summary(report, run_dir),
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"blockers: {summary['blockers']}",
        f"warnings: {summary['warnings']}",
        "split: train={train_count} validation={validation_count} test={test_count} heldout={heldout_count} doc_overlap={doc_overlap_count}".format(
            **summary["split"]
        ),
    ]
    peft = summary["peft"]
    trainable_fraction = peft.get("trainable_fraction")
    fraction_text = f"{100.0 * trainable_fraction:.2f}%" if isinstance(trainable_fraction, float) else "n/a"
    lines.append(
        "peft: requested={requested} effective={effective} rank={rank} alpha={alpha} rslora={rslora} dora={dora} trainable={params} ({fraction})".format(
            requested=peft.get("requested_trainable_subset"),
            effective=peft.get("effective_trainable_subset"),
            rank=peft.get("lora_rank"),
            alpha=peft.get("lora_alpha"),
            rslora=peft.get("lora_use_rslora"),
            dora=peft.get("lora_use_dora"),
            params=peft.get("trainable_parameters"),
            fraction=fraction_text,
        )
    )
    training = summary["training"]
    lines.append(
        "training: steps={steps} batch={batch} sampling={sampling} epochs={epochs} examples={examples} unique_seen={unique} final_logged_loss={loss}".format(
            steps=training.get("train_steps"),
            batch=training.get("train_batch_size"),
            sampling=training.get("train_sampling"),
            epochs=training.get("train_epochs"),
            examples=training.get("effective_train_examples"),
            unique=training.get("train_unique_indices_seen"),
            loss=_round(training.get("final_logged_loss")),
        )
    )
    lines.append(
        "validation_nll: "
        + ", ".join(
            f"{control}={_round(value)}" for control, value in summary["validation_losses"].items()
        )
    )
    lines.append(
        "test_nll: "
        + ", ".join(
            f"{control}={_round(value)}" for control, value in summary["test_losses"].items()
        )
    )
    lines.append(
        "heldout_nll: "
        + ", ".join(
            f"{control}={_round(value)}" for control, value in summary["heldout_losses"].items()
        )
    )
    lines.append(
        "validation_gaps: "
        + ", ".join(
            f"{key}={_round(value)}" for key, value in summary["real_validation_gaps"].items()
        )
    )
    lines.append(
        "test_gaps: "
        + ", ".join(
            f"{key}={_round(value)}" for key, value in summary["real_test_gaps"].items()
        )
    )
    lines.append(
        "heldout_gaps: "
        + ", ".join(
            f"{key}={_round(value)}" for key, value in summary["real_heldout_gaps"].items()
        )
    )
    examples = summary["examples"]
    lines.append(
        "examples: count={count} parsed_real={parsed_real} f1_real={real} f1_shuffled={shuffled} f1_zero={zero} f1_none={none}".format(
            count=examples.get("count"),
            parsed_real=examples.get("parsed_real"),
            real=_round(examples.get("mean_content_f1_real")),
            shuffled=_round(examples.get("mean_content_f1_shuffled")),
            zero=_round(examples.get("mean_content_f1_zero")),
            none=_round(examples.get("mean_content_f1_none")),
        )
    )
    wandb = summary["wandb"]
    lines.append(
        "wandb: status={status} mode={mode} name={name} offline_dirs={offline_dirs}".format(
            status=wandb.get("status"),
            mode=wandb.get("mode"),
            name=wandb.get("name"),
            offline_dirs=len(wandb.get("offline_dirs") or []),
        )
    )
    state = summary["trainable_state"]
    if state:
        size_gib = state.get("size_gib")
        size_text = f"{size_gib:.2f} GiB" if isinstance(size_gib, float) else state.get("bytes")
        lines.append(
            "trainable_state: tensors={tensor_count} size={size} path={path}".format(
                tensor_count=state.get("tensor_count"),
                size=size_text,
                path=state.get("path"),
            )
        )
    return "\n".join(lines)


def resolve_report_path(path: Path) -> tuple[Path, Path | None]:
    if path.is_dir():
        return path / "av_warmstart_smoke.json", path
    return path, path.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_or_run_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON.")
    args = parser.parse_args()

    report_path, run_dir = resolve_report_path(args.report_or_run_dir)
    report = json.loads(report_path.read_text())
    summary = summarize_report(report, run_dir=run_dir)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_summary(summary))


if __name__ == "__main__":
    main()
