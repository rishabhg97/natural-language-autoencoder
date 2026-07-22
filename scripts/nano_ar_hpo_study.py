#!/usr/bin/env python3
"""Offline Nano AR-SFT HPO study helper.

This utility does not launch training. It records completed AR eval reports in a
small JSONL study file and ranks bounded next-trial suggestions from those
heldout metrics.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "nano_ar_hpo_trial.v1"


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _min_lr_ratio(training: dict[str, Any]) -> float | None:
    lr = _as_float(training.get("lr"))
    min_lr = _as_float(training.get("min_lr"))
    if lr is None or min_lr is None or lr == 0:
        return None
    return min_lr / lr


def params_from_config(config_path: Path) -> dict[str, Any]:
    spec = _load_yaml(config_path)
    training = spec.get("training") or {}
    if not isinstance(training, dict):
        raise ValueError(f"training section must be a mapping: {config_path}")
    params = {
        "lr": _as_float(training.get("lr")),
        "min_lr": _as_float(training.get("min_lr")),
        "min_lr_ratio": _min_lr_ratio(training),
        "lr_decay_style": str(training.get("lr_decay_style", "constant")),
        "lr_warmup_iters": _as_int(training.get("lr_warmup_iters"), 0),
        "resume_steps": _as_int(training.get("resume_steps") or training.get("num_rollout")),
        "global_batch_size": _as_int(training.get("global_batch_size")),
        "micro_batch_size": _as_int(training.get("micro_batch_size")),
        "rollout_batch_size": _as_int(training.get("rollout_batch_size")),
        "injection_scale": _as_float(training.get("injection_scale")),
        "epochs": _as_int(training.get("epochs")),
    }
    return {key: value for key, value in params.items() if value is not None}


def task_from_config(config_path: Path) -> str:
    spec = _load_yaml(config_path)
    training = spec.get("training") or {}
    objective = training.get("objective")
    if objective is None:
        return "ar"
    objective = str(objective)
    if objective == "ar_sft":
        return "ar"
    if objective == "av_sft":
        return "av"
    raise ValueError(f"cannot infer HPO task from objective {objective!r}: {config_path}")


def objective_key_for_task(task: str) -> str:
    if task == "av_roundtrip":
        return "objective_roundtrip_nmse"
    if task == "av":
        return "objective_nll"
    if task == "ar":
        return "objective_nmse"
    raise ValueError(f"unknown task: {task!r}")


def _is_av_task(task: str) -> bool:
    return task in {"av", "av_roundtrip"}


def _teacher_control(split: dict[str, Any]) -> dict[str, Any]:
    controls = split.get("controls") or {}
    teacher = controls.get("teacher")
    if not isinstance(teacher, dict):
        raise ValueError("eval report split is missing controls.teacher")
    return teacher


def metrics_from_ar_eval(eval_report_path: Path) -> dict[str, Any]:
    report = json.loads(eval_report_path.read_text())
    splits = report.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"eval report has no splits mapping: {eval_report_path}")
    metrics: dict[str, Any] = {}
    validation_teacher_nmse: float | None = None
    for split_name in ("validation", "test"):
        split = splits.get(split_name)
        if not isinstance(split, dict):
            continue
        teacher = _teacher_control(split)
        prefix = f"{split_name}_teacher"
        nmse = _as_float(teacher.get("normalized_mse"))
        if nmse is not None:
            metrics[f"{prefix}_nmse"] = nmse
            if split_name == "validation":
                validation_teacher_nmse = nmse
        for source_key, metric_key in (
            ("cosine_mean", "cosine"),
            ("fve_nrm", "fve"),
            ("raw_mse", "raw_mse"),
        ):
            value = _as_float(teacher.get(source_key))
            if value is not None:
                metrics[f"{prefix}_{metric_key}"] = value
        wins = split.get("rowwise_win_rates") or {}
        if isinstance(wins, dict):
            for win_key in ("teacher_vs_mean", "teacher_vs_source_context", "teacher_vs_teacher_shuffled"):
                win = wins.get(win_key)
                if isinstance(win, dict):
                    value = _as_float(win.get("teacher_better_fraction"))
                    if value is not None:
                        metrics[f"{split_name}_{win_key.replace('teacher_vs_', 'teacher_beats_')}"] = value
    if validation_teacher_nmse is not None:
        metrics["objective_nmse"] = validation_teacher_nmse
        metrics["objective_key"] = "objective_nmse"
        metrics["objective_split"] = "validation"
    return metrics


def _split_loss(control: dict[str, Any], split_name: str) -> float | None:
    split = control.get(split_name)
    if isinstance(split, dict):
        return _as_float(split.get("loss"))
    return None


def _weighted_mean_loss(control: dict[str, Any], split_names: tuple[str, ...]) -> float | None:
    total = 0.0
    count = 0
    for split_name in split_names:
        split = control.get(split_name)
        if not isinstance(split, dict):
            continue
        loss = _as_float(split.get("loss"))
        split_count = _as_int(split.get("count"), 1)
        if loss is None or split_count is None:
            continue
        total += loss * split_count
        count += split_count
    if count == 0:
        return None
    return total / count


def metrics_from_av_eval(eval_report_path: Path) -> dict[str, Any]:
    report = json.loads(eval_report_path.read_text())
    loss_summary = report.get("loss_summary")
    if not isinstance(loss_summary, dict):
        raise ValueError(f"AV eval report has no loss_summary mapping: {eval_report_path}")
    real = loss_summary.get("real")
    if not isinstance(real, dict):
        raise ValueError(f"AV eval report has no real loss summary: {eval_report_path}")
    metrics: dict[str, Any] = {"objective_key": "objective_nll"}
    for split_name in ("validation", "test", "heldout", "all"):
        value = _split_loss(real, split_name)
        if value is not None:
            metrics[f"{split_name}_real_nll"] = value
    objective = _split_loss(real, "validation")
    if objective is not None:
        metrics["objective_nll"] = objective
        metrics["objective_split"] = "validation"
    for control_name in ("shuffled", "zero", "mean", "none"):
        control = loss_summary.get(control_name)
        if not isinstance(control, dict):
            continue
        for split_name in ("validation", "test", "heldout", "all"):
            control_loss = _split_loss(control, split_name)
            if control_loss is not None:
                metrics[f"{split_name}_{control_name}_nll"] = control_loss
            gap_key = f"{split_name}_loss_gap_vs_{control_name}"
            gap = _as_float(real.get(gap_key))
            if gap is None:
                real_loss = _split_loss(real, split_name)
                if real_loss is not None and control_loss is not None:
                    gap = control_loss - real_loss
            if gap is not None:
                metrics[f"{split_name}_gap_vs_{control_name}"] = gap
    return metrics


def _roundtrip_variant_nmse(split: dict[str, Any], variant: str) -> float | None:
    variants = split.get("variants") or {}
    if not isinstance(variants, dict):
        return None
    metrics = variants.get(variant)
    if not isinstance(metrics, dict):
        return None
    return _as_float(metrics.get("normalized_mse"))


def metrics_from_roundtrip_report(roundtrip_report_path: Path) -> dict[str, Any]:
    report = json.loads(roundtrip_report_path.read_text())
    splits = report.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"round-trip report has no splits mapping: {roundtrip_report_path}")
    metrics: dict[str, Any] = {"objective_key": "objective_roundtrip_nmse"}
    validation_roundtrip_nmse: float | None = None
    for split_name in ("validation", "test"):
        split = splits.get(split_name)
        if not isinstance(split, dict):
            continue
        primary = _roundtrip_variant_nmse(split, "av_real")
        teacher = _roundtrip_variant_nmse(split, "teacher")
        mean = _roundtrip_variant_nmse(split, "mean")
        if primary is not None:
            metrics[f"{split_name}_roundtrip_av_real_nmse"] = primary
            if split_name == "validation":
                validation_roundtrip_nmse = primary
        if teacher is not None:
            metrics[f"{split_name}_roundtrip_teacher_nmse"] = teacher
        if mean is not None:
            metrics[f"{split_name}_roundtrip_mean_nmse"] = mean
        parse = (split.get("generation_parse") or {}).get("real")
        if isinstance(parse, dict):
            closed_fraction = _as_float(parse.get("closed_fraction"))
            empty_fraction = _as_float(parse.get("empty_fraction"))
            if closed_fraction is not None:
                metrics[f"{split_name}_roundtrip_parse_closed_fraction"] = closed_fraction
            if empty_fraction is not None:
                metrics[f"{split_name}_roundtrip_parse_empty_fraction"] = empty_fraction
        gate_split = ((report.get("gate") or {}).get("splits") or {}).get(split_name)
        if isinstance(gate_split, dict):
            metrics[f"{split_name}_roundtrip_beats_all_controls"] = bool(gate_split.get("beats_all_controls"))
            if gate_split.get("baseline_beaten") is not None:
                metrics[f"{split_name}_roundtrip_baseline_beaten"] = bool(gate_split.get("baseline_beaten"))
    if validation_roundtrip_nmse is not None:
        metrics["objective_roundtrip_nmse"] = validation_roundtrip_nmse
        metrics["objective_split"] = "validation"
    gate = report.get("gate")
    if isinstance(gate, dict):
        metrics["roundtrip_gate_passed"] = bool(gate.get("passed"))
    return metrics


def metrics_from_eval(eval_report_path: Path, *, task: str | None = None) -> dict[str, Any]:
    report = json.loads(eval_report_path.read_text())
    inferred = task
    if inferred is None:
        if "loss_summary" in report:
            inferred = "av"
        elif "splits" in report:
            inferred = "ar"
    if inferred in {"av", "av_roundtrip"}:
        return metrics_from_av_eval(eval_report_path)
    if inferred == "ar":
        return metrics_from_ar_eval(eval_report_path)
    raise ValueError(f"could not infer eval report task: {eval_report_path}")


def metrics_from_train_log(train_log_path: Path | None) -> dict[str, Any]:
    if train_log_path is None or not train_log_path.exists():
        return {}
    latest: dict[str, Any] | None = None
    step_pattern = re.compile(r"step\s+\d+:\s+(\{.*\})")
    for line in train_log_path.read_text(errors="replace").splitlines():
        match = step_pattern.search(line)
        if not match:
            continue
        try:
            parsed = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            latest = parsed
    if latest is None:
        return {}
    metrics: dict[str, Any] = {}
    mapping = {
        "train/loss": "final_train_loss",
        "train/fve_nrm": "final_train_fve",
        "train/grad_norm": "final_train_grad_norm",
        "train/step": "final_train_step",
        "train/lr-pg_0": "final_train_lr",
    }
    for source, target in mapping.items():
        value = latest.get(source)
        if isinstance(value, int):
            metrics[target] = value
        else:
            numeric = _as_float(value)
            if numeric is not None:
                metrics[target] = numeric
    return metrics


def lr_decay_canary(params: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Check whether a nominal decay schedule actually decayed in observed logs."""

    schedule = str(params.get("lr_decay_style", "constant")).lower()
    lr = _as_float(params.get("lr"))
    min_lr = _as_float(params.get("min_lr"))
    final_lr = _as_float(metrics.get("final_train_lr"))
    if schedule in {"", "constant", "none"} or lr is None:
        return {"applicable": False, "passed": True, "message": "no decay schedule to check"}
    if min_lr is not None and min_lr >= lr:
        return {"applicable": True, "passed": False, "message": "min_lr is not below lr"}
    if final_lr is None:
        return {"applicable": True, "passed": False, "message": "missing final_train_lr for decay schedule"}
    threshold = 0.9 * lr
    passed = final_lr < threshold
    message = (
        f"observed decay final_lr={final_lr:g} < {threshold:g}"
        if passed
        else f"flat or under-decayed LR: final_lr={final_lr:g} >= {threshold:g}"
    )
    return {"applicable": True, "passed": passed, "message": message}


def assert_lr_decay_canary_passed(params: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    canary = lr_decay_canary(params, metrics)
    if canary["applicable"] and not canary["passed"]:
        raise ValueError(f"LR decay canary failed: {canary['message']}")
    return canary


def assert_lr_decay_canary_for_run(config_path: Path, train_log_path: Path | None) -> dict[str, Any]:
    params = params_from_config(config_path)
    metrics = metrics_from_train_log(train_log_path)
    return assert_lr_decay_canary_passed(params, metrics)


def build_trial_record(
    *,
    trial_name: str,
    config_path: Path,
    eval_report_path: Path | None = None,
    roundtrip_report_path: Path | None = None,
    train_log_path: Path | None = None,
    run_dir: Path | None = None,
    status: str = "complete",
    notes: str | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    task = task or task_from_config(config_path)
    metrics: dict[str, Any] = {}
    checkpoint_dir = None
    if eval_report_path is not None:
        report = json.loads(eval_report_path.read_text())
        checkpoint_dir = report.get("checkpoint_dir") or report.get("hf_checkpoint")
        metrics.update(metrics_from_eval(eval_report_path, task=task))
    if roundtrip_report_path is not None:
        metrics.update(metrics_from_roundtrip_report(roundtrip_report_path))
    metrics.update(metrics_from_train_log(train_log_path))
    params = params_from_config(config_path)
    canary = lr_decay_canary(params, metrics)
    if canary["applicable"]:
        metrics["lr_decay_canary_passed"] = bool(canary["passed"])
    record = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "trial_name": trial_name,
        "status": status,
        "params": params,
        "metrics": metrics,
        "artifacts": {
            "config": str(config_path),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "eval_report": str(eval_report_path) if eval_report_path is not None else None,
            "roundtrip_report": str(roundtrip_report_path) if roundtrip_report_path is not None else None,
            "train_log": str(train_log_path) if train_log_path is not None else None,
            "checkpoint_dir": checkpoint_dir,
        },
    }
    if notes:
        record["notes"] = notes
    return record


def load_trials(study_jsonl: Path) -> list[dict[str, Any]]:
    if not study_jsonl.exists():
        return []
    trials: list[dict[str, Any]] = []
    for line in study_jsonl.read_text().splitlines():
        if line.strip():
            trials.append(json.loads(line))
    return trials


def upsert_trial(study_jsonl: Path, record: dict[str, Any]) -> None:
    trials = load_trials(study_jsonl)
    key = record.get("trial_name")
    kept = [trial for trial in trials if trial.get("trial_name") != key]
    kept.append(record)
    study_jsonl.parent.mkdir(parents=True, exist_ok=True)
    study_jsonl.write_text("\n".join(json.dumps(trial, sort_keys=True) for trial in kept) + "\n")


def param_signature(params: dict[str, Any], *, task: str = "ar") -> tuple[Any, ...]:
    common = (
        round(float(params.get("lr", 0.0)), 12),
        str(params.get("lr_decay_style", "constant")),
        int(params.get("lr_warmup_iters") or 0),
        int(params.get("resume_steps") or 0),
        int(params.get("global_batch_size") or 0),
        None if params.get("min_lr_ratio") is None else round(float(params["min_lr_ratio"]), 6),
    )
    if _is_av_task(task):
        return common + (None if params.get("injection_scale") is None else round(float(params["injection_scale"]), 6),)
    return common


def _trial_task(trial: dict[str, Any], default: str) -> str:
    return str(trial.get("task") or default)


def _completed_trials(trials: list[dict[str, Any]], *, task: str = "ar") -> list[dict[str, Any]]:
    objective_key = objective_key_for_task(task)
    return [
        trial
        for trial in trials
        if _trial_task(trial, task) == task
        and trial.get("status") == "complete"
        and _as_float((trial.get("metrics") or {}).get(objective_key)) is not None
    ]


def _best_trial(trials: list[dict[str, Any]], *, task: str = "ar") -> dict[str, Any] | None:
    completed = _completed_trials(trials, task=task)
    if not completed:
        return None
    objective_key = objective_key_for_task(task)
    return min(completed, key=lambda trial: float(trial["metrics"][objective_key]))


def _candidate_grid(best_params: dict[str, Any], *, task: str = "ar") -> list[dict[str, Any]]:
    best_lr = float(best_params.get("lr") or 1e-5)
    if _is_av_task(task):
        lr_values = sorted({5e-6, 7.5e-6, 1e-5, 1.5e-5, 2e-5, 5e-5, 1e-4, best_lr})
    else:
        lr_values = sorted({2e-6, 3e-6, 5e-6, 7.5e-6, 1e-5, 1.5e-5, 2e-5, best_lr})
    schedules = ["cosine", "constant"]
    warmups = [10, 25, 50]
    steps = [467, 934] if _is_av_task(task) else [128, 256, 512]
    batches = [192, 256]
    ratios = [0.05, 0.1, 0.2]
    injection_scales = (
        [50.0, 75.0, 100.0, 150.0]
        if _is_av_task(task)
        else [float(best_params.get("injection_scale") or 75.0)]
    )
    grid: list[dict[str, Any]] = []
    for lr in lr_values:
        for schedule in schedules:
            schedule_ratios: list[float | None]
            if _is_av_task(task) and schedule == "constant":
                schedule_ratios = [None]
            else:
                schedule_ratios = ratios
            for warmup in warmups:
                for resume_steps in steps:
                    for batch in batches:
                        for ratio in schedule_ratios:
                            for injection_scale in injection_scales:
                                candidate = {
                                    "lr": lr,
                                    "lr_decay_style": schedule,
                                    "lr_warmup_iters": warmup,
                                    "resume_steps": resume_steps,
                                    "global_batch_size": batch,
                                }
                                if ratio is not None:
                                    candidate["min_lr_ratio"] = ratio
                                    candidate["min_lr"] = lr * ratio
                                if _is_av_task(task):
                                    candidate["injection_scale"] = injection_scale
                                grid.append(candidate)
    return grid


def _candidate_score(candidate: dict[str, Any], best_params: dict[str, Any], *, task: str = "ar") -> tuple[float, str]:
    best_lr = float(best_params.get("lr") or 1e-5)
    lr = float(candidate["lr"])
    distance = abs(math.log(lr / best_lr))
    score = distance
    reasons = [f"near best lr {best_lr:g}"]
    if lr < best_lr:
        score -= 0.12
        reasons.append("cooldown below best")
    if candidate["lr_decay_style"] == best_params.get("lr_decay_style", "cosine"):
        score -= 0.08
        reasons.append("keeps best schedule")
    steps = int(candidate["resume_steps"])
    if _is_av_task(task):
        if steps == 467:
            score -= 0.08
            reasons.append("single-epoch hero replay length")
        elif steps == 934:
            score += 0.05
            reasons.append("longer two-epoch probe")
    elif steps == 256:
        score -= 0.08
        reasons.append("bounded 256-step probe")
    elif steps == 128:
        score += 0.05
    else:
        score += 0.18
    if int(candidate["global_batch_size"]) == int(best_params.get("global_batch_size") or 192):
        score -= 0.04
    else:
        score += 0.08
        reasons.append("batch-size exploration")
    if int(candidate["lr_warmup_iters"]) == int(best_params.get("lr_warmup_iters") or 25):
        score -= 0.03
    if float(candidate.get("min_lr_ratio") or 0.1) == float(best_params.get("min_lr_ratio") or 0.1):
        score -= 0.02
    if _is_av_task(task):
        best_scale = float(best_params.get("injection_scale") or 75.0)
        scale = float(candidate.get("injection_scale") or 75.0)
        if scale == best_scale:
            score -= 0.16
            reasons.append(f"keeps best injection scale {best_scale:g}")
        else:
            score += abs(math.log(scale / best_scale)) * 0.4
            reasons.append("injection-scale exploration")
    return score, "; ".join(reasons)


def suggest_next_trials(trials: list[dict[str, Any]], *, top_n: int = 5, task: str = "ar") -> list[dict[str, Any]]:
    best = _best_trial(trials, task=task)
    if best is None:
        best_params = {
            "lr": 1e-5,
            "min_lr_ratio": 0.1,
            "lr_decay_style": "cosine",
            "lr_warmup_iters": 25,
            "resume_steps": 256,
            "global_batch_size": 192,
        }
        if _is_av_task(task):
            best_params.update({"resume_steps": 467, "injection_scale": 75.0})
        best_objective = None
    else:
        best_params = best["params"]
        best_objective = float(best["metrics"][objective_key_for_task(task)])
    seen = {param_signature(trial.get("params") or {}, task=task) for trial in trials if _trial_task(trial, task) == task}
    ranked: list[tuple[float, dict[str, Any], str]] = []
    for candidate in _candidate_grid(best_params, task=task):
        if param_signature(candidate, task=task) in seen:
            continue
        score, reason = _candidate_score(candidate, best_params, task=task)
        ranked.append((score, candidate, reason))
    ranked.sort(key=lambda item: item[0])
    suggestions: list[dict[str, Any]] = []
    for rank, (score, params, reason) in enumerate(ranked[:top_n], start=1):
        suggestions.append(
            {
                "rank": rank,
                "params": params,
                "objective_hint": best_objective,
                "score": score,
                "reason": reason,
            }
        )
    return suggestions


def render_suggestions_markdown(suggestions: list[dict[str, Any]], trials: list[dict[str, Any]], *, task: str = "ar") -> str:
    best = _best_trial(trials, task=task)
    title = "Nano AV HPO Suggestions" if _is_av_task(task) else "Nano AR HPO Suggestions"
    if task == "av_roundtrip":
        objective_label = "round-trip objective NMSE"
    else:
        objective_label = "objective NLL" if task == "av" else "objective NMSE"
    lines = [f"# {title}", ""]
    if best is not None:
        lines.append(
            "Best completed trial: "
            f"`{best.get('trial_name')}` with {objective_label} "
            f"`{float(best['metrics'][objective_key_for_task(task)]):.6f}`."
        )
    else:
        lines.append("Best completed trial: none yet.")
    lines.extend(
        [
            "",
            "| Rank | lr | min_lr_ratio | schedule | warmup | steps | batch | reason |",
            "|---:|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    if _is_av_task(task):
        lines[-2] = "| Rank | lr | min_lr_ratio | scale | schedule | warmup | steps | batch | reason |"
        lines[-1] = "|---:|---:|---:|---:|---|---:|---:|---:|---|"
    for suggestion in suggestions:
        params = suggestion["params"]
        if _is_av_task(task):
            ratio = params.get("min_lr_ratio")
            ratio_text = "-" if ratio is None else f"{float(ratio):g}"
            lines.append(
                "| {rank} | `{lr:g}` | `{ratio}` | `{scale:g}` | `{schedule}` | `{warmup}` | `{steps}` | `{batch}` | {reason} |".format(
                    rank=suggestion["rank"],
                    lr=float(params["lr"]),
                    ratio=ratio_text,
                    scale=float(params.get("injection_scale") or 0.0),
                    schedule=params["lr_decay_style"],
                    warmup=int(params["lr_warmup_iters"]),
                    steps=int(params["resume_steps"]),
                    batch=int(params["global_batch_size"]),
                    reason=suggestion["reason"],
                )
            )
        else:
            lines.append(
                "| {rank} | `{lr:g}` | `{ratio:g}` | `{schedule}` | `{warmup}` | `{steps}` | `{batch}` | {reason} |".format(
                    rank=suggestion["rank"],
                    lr=float(params["lr"]),
                    ratio=float(params.get("min_lr_ratio") or 0.0),
                    schedule=params["lr_decay_style"],
                    warmup=int(params["lr_warmup_iters"]),
                    steps=int(params["resume_steps"]),
                    batch=int(params["global_batch_size"]),
                    reason=suggestion["reason"],
                )
            )
    lines.extend(
        [
            "",
            f"Use heldout {task.upper()} eval {objective_label} as the selection metric. Do not select from train loss alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def export_optuna_payload(trials: list[dict[str, Any]], *, task: str = "ar") -> dict[str, Any]:
    objective_key = objective_key_for_task(task)
    payload_trials: list[dict[str, Any]] = []
    state_map = {
        "complete": "COMPLETE",
        "running": "RUNNING",
        "failed": "FAIL",
        "pruned": "PRUNED",
    }
    for number, trial in enumerate(trials):
        metrics = trial.get("metrics") or {}
        value = _as_float(metrics.get(objective_key))
        payload_trials.append(
            {
                "number": number,
                "trial_name": trial.get("trial_name"),
                "state": state_map.get(str(trial.get("status", "")).lower(), "RUNNING"),
                "value": value,
                "params": trial.get("params") or {},
                "user_attrs": {
                    "artifacts": trial.get("artifacts") or {},
                    "metrics": metrics,
                    "notes": trial.get("notes"),
                },
            }
        )
    return {
        "schema_version": "nano_nla_optuna_export.v1",
        "task": task,
        "direction": "minimize",
        "objective": objective_key,
        "trials": payload_trials,
    }


def _record_cmd(args: argparse.Namespace) -> int:
    record = build_trial_record(
        trial_name=args.trial_name,
        config_path=args.config,
        eval_report_path=args.eval_report,
        roundtrip_report_path=args.roundtrip_report,
        train_log_path=args.train_log,
        run_dir=args.run_dir,
        status=args.status,
        notes=args.notes,
        task=args.task,
    )
    upsert_trial(args.study_jsonl, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def _suggest_cmd(args: argparse.Namespace) -> int:
    trials = load_trials(args.study_jsonl)
    suggestions = suggest_next_trials(trials, top_n=args.top_n, task=args.task)
    payload = {"schema_version": "nano_nla_hpo_suggestions.v1", "task": args.task, "suggestions": suggestions}
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = render_suggestions_markdown(suggestions, trials, task=args.task)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown)
    print(markdown)
    return 0


def _export_optuna_cmd(args: argparse.Namespace) -> int:
    payload = export_optuna_payload(load_trials(args.study_jsonl), task=args.task)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser(*, default_task: str = "ar") -> argparse.ArgumentParser:
    if default_task not in {"ar", "av", "av_roundtrip"}:
        raise ValueError(f"unsupported default HPO task: {default_task}")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Record or update one completed/running NLA HPO trial.")
    record.add_argument("--study-jsonl", type=Path, required=True)
    record.add_argument("--task", choices=["ar", "av", "av_roundtrip"], default=None)
    record.add_argument("--trial-name", required=True)
    record.add_argument("--config", type=Path, required=True)
    record.add_argument("--eval-report", type=Path)
    record.add_argument("--roundtrip-report", type=Path)
    record.add_argument("--train-log", type=Path)
    record.add_argument("--run-dir", type=Path)
    record.add_argument("--status", choices=["complete", "running", "failed", "pruned"], default="complete")
    record.add_argument("--notes")
    record.set_defaults(func=_record_cmd)

    suggest = subparsers.add_parser("suggest", help="Rank next bounded NLA HPO parameter suggestions.")
    suggest.add_argument("--study-jsonl", type=Path, required=True)
    suggest.add_argument(
        "--task",
        choices=["ar", "av", "av_roundtrip"],
        default=default_task,
    )
    suggest.add_argument("--top-n", type=int, default=5)
    suggest.add_argument("--out-json", type=Path)
    suggest.add_argument("--out-md", type=Path)
    suggest.set_defaults(func=_suggest_cmd)

    export = subparsers.add_parser("export-optuna", help="Write Optuna-style completed/running trial payload.")
    export.add_argument("--study-jsonl", type=Path, required=True)
    export.add_argument(
        "--task",
        choices=["ar", "av", "av_roundtrip"],
        default=default_task,
    )
    export.add_argument("--out-json", type=Path, required=True)
    export.set_defaults(func=_export_optuna_cmd)
    return parser


def main(argv: list[str] | None = None, *, default_task: str = "ar") -> int:
    parser = build_parser(default_task=default_task)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
