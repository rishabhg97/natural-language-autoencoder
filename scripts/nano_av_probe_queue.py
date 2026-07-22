#!/usr/bin/env python3
"""Sequential queue for Nano AV layer probes."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nano_av_runner  # noqa: E402
import nano_ar_hpo_study  # noqa: E402
from nano_queue_status import ACTIVE_HPO_STATUSES, VALID_HPO_STATUSES, status_counts  # noqa: E402


VALID_STATUSES = VALID_HPO_STATUSES
ACTIVE_STATUSES = ACTIVE_HPO_STATUSES
MAX_AUTOMATED_EVAL_LIMIT = 512


class QueueError(ValueError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_queue(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != "nano_av_probe_queue.v1":
        raise QueueError("queue YAML must use schema_version nano_av_probe_queue.v1")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise QueueError("queue YAML must contain non-empty items")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise QueueError(f"item {index} must be a mapping")
        item.setdefault("status", "pending")
        if item["status"] not in VALID_STATUSES:
            raise QueueError(f"item {index} has invalid status {item['status']!r}")
        if not item.get("name") or not item.get("config"):
            raise QueueError(f"item {index} requires name and config")
    return data


def write_queue(path: str | Path, queue_doc: dict[str, Any]) -> None:
    destination = Path(path)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(queue_doc, sort_keys=False))
    tmp.replace(destination)


def update_item(path: str | Path, index: int, **fields: Any) -> None:
    queue_doc = load_queue(path)
    queue_doc["items"][index].update({key: value for key, value in fields.items() if value is not None})
    write_queue(path, queue_doc)


def reset_active_items(path: str | Path, *, reason: str = "manual active reset") -> dict[str, Any]:
    queue_doc = load_queue(path)
    reset_items: list[str] = []
    reset_at = utc_now()
    for item in queue_doc["items"]:
        status = item.get("status")
        if status not in ACTIVE_STATUSES:
            continue
        item["previous_status"] = status
        item["status"] = "pending"
        item["reset_at"] = reset_at
        item["reset_reason"] = reason
        reset_items.append(str(item["name"]))
    if reset_items:
        write_queue(path, queue_doc)
    return {"queue": str(path), "reset_count": len(reset_items), "items": reset_items}


def queue_status(queue_path: str | Path) -> dict[str, Any]:
    queue_doc = load_queue(queue_path)
    return {"queue": str(queue_path), "counts": status_counts(queue_doc["items"], VALID_STATUSES), "items": queue_doc["items"]}


def resolve_path(value: str | Path, queue_path: Path, queue_doc: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    code_root = queue_doc.get("defaults", {}).get("code_root")
    if code_root:
        candidate = Path(code_root) / path
        if candidate.exists():
            return candidate
    candidate = queue_path.parent / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def expected_checkpoint_for_plan(plan: dict[str, Any]) -> Path:
    return Path(plan["run_dir"]) / "checkpoints" / f"iter_{int(plan['num_rollout']):07d}"


def eval_paths_for_plan(plan: dict[str, Any]) -> tuple[Path, Path, Path]:
    manifest = plan.get("split_manifest")
    if isinstance(manifest, dict):
        return (
            Path(plan["train_parquet"]),
            Path(manifest["validation"]["path"]),
            Path(manifest["test"]["path"]),
        )
    train = Path(plan["train_parquet"])
    return train, train.parent / "validation.parquet", train.parent / "test.parquet"


def env_for_run(plan: dict[str, Any], code_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (plan.get("environment") or {}).items()})
    env["WANDB_MODE"] = "offline"
    pythonpath = [
        str(code_root / "external" / "natural_language_autoencoders"),
        str(code_root / "external" / "natural_language_autoencoders" / "Miles"),
        str(code_root),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    return env


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n# started_utc={utc_now()}\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"# completed_utc={utc_now()}\n")


def converted_hf_checkpoint_for_dcp(checkpoint_dir: Path) -> Path:
    return checkpoint_dir.parent.parent / f"hf_{checkpoint_dir.name}"


def build_convert_command(
    *,
    python_bin: str,
    code_root: Path,
    checkpoint_dir: Path,
    origin_hf_dir: Path,
    output_dir: Path,
    torch_dtype: str | None = None,
) -> list[str]:
    command = [
        python_bin,
        str(code_root / "external" / "natural_language_autoencoders" / "tools" / "convert_fsdp_to_hf.py"),
        "--input-dir",
        str(checkpoint_dir),
        "--origin-hf-dir",
        str(origin_hf_dir),
        "--output-dir",
        str(output_dir),
    ]
    if torch_dtype:
        command.extend(["--torch-dtype", str(torch_dtype)])
    return command


def build_eval_command(
    *,
    python_bin: str,
    hf_checkpoint: Path,
    train_parquet: Path,
    validation_parquet: Path,
    test_parquet: Path | None,
    report_json: Path,
    validation_limit: int,
    test_limit: int,
    eval_splits: list[str],
    generation_examples: int,
    injection_scale: str,
    wandb_project: str,
    wandb_group: str,
) -> list[str]:
    if validation_limit > MAX_AUTOMATED_EVAL_LIMIT or test_limit > MAX_AUTOMATED_EVAL_LIMIT:
        raise QueueError("automated AV eval limits must be <= 512")
    if not eval_splits or len(set(eval_splits)) != len(eval_splits) or not set(
        eval_splits
    ).issubset({"validation", "test"}):
        raise QueueError("eval_splits must be a non-empty unique validation/test list")
    command = [
        python_bin,
        "scripts/eval_nano_av_miles_checkpoint.py",
        "--hf-checkpoint",
        str(hf_checkpoint),
        "--train-parquet",
        str(train_parquet),
        "--validation-parquet",
        str(validation_parquet),
        "--report-json",
        str(report_json),
        "--validation-limit",
        str(validation_limit),
        "--test-limit",
        str(test_limit),
        "--eval-splits",
        *[str(split) for split in eval_splits],
        "--generation-examples",
        str(generation_examples),
        "--injection-scale",
        str(injection_scale),
        "--wandb-mode",
        "offline",
        "--wandb-project",
        wandb_project,
        "--wandb-group",
        wandb_group,
    ]
    if "test" in eval_splits:
        if test_parquet is None:
            raise QueueError("test_parquet is required when eval_splits includes test")
        command.extend(["--test-parquet", str(test_parquet)])
    return command


def build_roundtrip_command(
    *,
    python_bin: str,
    av_hf_checkpoint: Path,
    ar_checkpoint_dir: Path,
    train_parquet: Path,
    validation_parquet: Path,
    test_parquet: Path | None,
    validation_control_parquet: Path | None = None,
    test_control_parquet: Path | None = None,
    report_json: Path,
    generated_jsonl: Path | None = None,
    expected_generation_protocol_json: Path | None = None,
    prediction_cache_npz: Path | None = None,
    generation_controls: list[str] | tuple[str, ...] | None = None,
    dataset_controls: list[str] | tuple[str, ...] | None = None,
    validation_limit: int = 128,
    test_limit: int = 128,
    eval_splits: list[str] | tuple[str, ...] | None = None,
    max_new_tokens: int = 200,
    seed: int = 1234,
    injection_scale: str = "75",
    ar_batch_size: int = 4,
    ar_max_length: int | None = None,
    torch_dtype: str = "bfloat16",
    control_margin: float = 0.1,
    baseline_report_json: Path | None = None,
    length_baseline_generated_jsonl: Path | None = None,
    baseline_margin: float = 0.0,
    critic_template: str | None = None,
    critic_template_source: Path | None = None,
    av_device_map: str | None = None,
    av_low_cpu_mem_usage: bool = True,
    ar_device_map: str | None = None,
    ar_low_cpu_mem_usage: bool = False,
    collect_ar_device_profile: bool = False,
    generation_prefix: str | None = None,
    stop_text: str | None = None,
    generated_text_fallback: str | None = None,
    generation_backend: str | None = None,
    generation_workers: int | None = None,
    generation_max_parallel_workers: int | None = None,
    generation_worker_devices: list[str] | tuple[str, ...] | None = None,
    stream_generated: bool = False,
    resume_generated: bool = False,
    generation_only: bool = False,
    reuse_generated: bool = False,
    progress_every: int | None = None,
    min_control_win_fraction: float | None = None,
    min_baseline_win_fraction: float | None = None,
    min_baseline_relative_improvement: float | None = None,
    require_baseline_ci_positive: bool = False,
    require_clustered_baseline_ci: bool = False,
    require_baseline_dataset_match: bool = False,
    bootstrap_samples: int | None = None,
    bootstrap_seed: int | None = None,
    permutation_samples: int | None = None,
    permutation_seed: int | None = None,
    min_closed_fraction: float | None = None,
    min_usable_fraction: float | None = None,
    content_family_manifest: Path | None = None,
    content_family_coverage: Path | None = None,
    selection_strategy: str | None = None,
    selection_seed: int | None = None,
    require_family_level_inference: bool = False,
    min_independent_families: int | None = None,
    av_model_fingerprint: str | None = None,
    av_tokenizer_fingerprint: str | None = None,
    require_generation_protocol_match: bool = False,
    max_automated_limit: int = MAX_AUTOMATED_EVAL_LIMIT,
) -> list[str]:
    if validation_limit > max_automated_limit or test_limit > max_automated_limit:
        raise QueueError("automated round-trip eval limits must be <= 512")
    if generation_only and reuse_generated:
        raise QueueError("generation_only and reuse_generated are mutually exclusive")
    if reuse_generated and generated_jsonl is None:
        raise QueueError("reuse_generated requires generated_jsonl")
    requested_splits = tuple(str(split) for split in (eval_splits or ("validation",)))
    if len(set(requested_splits)) != len(requested_splits) or not set(
        requested_splits
    ).issubset({"validation", "test"}):
        raise QueueError("eval_splits must be a non-empty unique validation/test list")
    if "test" in requested_splits and test_parquet is None:
        raise QueueError("test_parquet is required when eval_splits includes test")
    command = [
        python_bin,
        "scripts/eval_nano_av_ar_roundtrip_gate.py",
        "--av-hf-checkpoint",
        str(av_hf_checkpoint),
        "--ar-checkpoint-dir",
        str(ar_checkpoint_dir),
        "--train-parquet",
        str(train_parquet),
        "--validation-parquet",
        str(validation_parquet),
        "--report-json",
        str(report_json),
        "--validation-limit",
        str(validation_limit),
        "--test-limit",
        str(test_limit),
        "--max-new-tokens",
        str(max_new_tokens),
        "--seed",
        str(seed),
        "--injection-scale",
        str(injection_scale),
        "--ar-batch-size",
        str(ar_batch_size),
        "--torch-dtype",
        str(torch_dtype),
        "--control-margin",
        str(control_margin),
        "--baseline-margin",
        str(baseline_margin),
    ]
    if test_parquet is not None:
        command.extend(["--test-parquet", str(test_parquet)])
    if validation_control_parquet is not None:
        command.extend(
            ["--validation-control-parquet", str(validation_control_parquet)]
        )
    if test_control_parquet is not None:
        command.extend(["--test-control-parquet", str(test_control_parquet)])
    command.append("--eval-splits")
    command.extend(requested_splits)
    if generated_jsonl is not None:
        command.extend(["--generated-jsonl", str(generated_jsonl)])
    if expected_generation_protocol_json is not None:
        command.extend(
            [
                "--expected-generation-protocol-json",
                str(expected_generation_protocol_json),
            ]
        )
    if prediction_cache_npz is not None:
        command.extend(["--prediction-cache-npz", str(prediction_cache_npz)])
    if generation_controls:
        command.append("--generation-controls")
        command.extend([str(control) for control in generation_controls])
    if dataset_controls:
        command.append("--dataset-controls")
        command.extend([str(control) for control in dataset_controls])
    if ar_max_length is not None:
        command.extend(["--ar-max-length", str(ar_max_length)])
    if baseline_report_json is not None:
        command.extend(["--baseline-report-json", str(baseline_report_json)])
    if length_baseline_generated_jsonl is not None:
        command.extend(
            [
                "--length-baseline-generated-jsonl",
                str(length_baseline_generated_jsonl),
            ]
        )
    if critic_template is not None:
        command.extend(["--critic-template", str(critic_template)])
    if critic_template_source is not None:
        command.extend(["--critic-template-source", str(critic_template_source)])
    if av_device_map is not None:
        command.extend(["--av-device-map", str(av_device_map)])
    command.append(
        "--av-low-cpu-mem-usage"
        if av_low_cpu_mem_usage
        else "--no-av-low-cpu-mem-usage"
    )
    if ar_device_map is not None:
        command.extend(["--ar-device-map", str(ar_device_map)])
    command.append(
        "--ar-low-cpu-mem-usage"
        if ar_low_cpu_mem_usage
        else "--no-ar-low-cpu-mem-usage"
    )
    if collect_ar_device_profile:
        command.append("--collect-ar-device-profile")
    if generation_prefix is not None:
        command.extend(["--generation-prefix", str(generation_prefix)])
    if stop_text is not None:
        command.extend(["--stop-text", str(stop_text)])
    if generated_text_fallback is not None:
        command.extend(["--generated-text-fallback", str(generated_text_fallback)])
    if generation_backend is not None:
        command.extend(["--generation-backend", str(generation_backend)])
    if generation_workers is not None:
        command.extend(["--generation-workers", str(generation_workers)])
    if generation_max_parallel_workers is not None:
        command.extend(
            ["--generation-max-parallel-workers", str(generation_max_parallel_workers)]
        )
    if generation_worker_devices:
        command.append("--generation-worker-devices")
        command.extend([str(device) for device in generation_worker_devices])
    if stream_generated:
        command.append("--stream-generated")
    if resume_generated:
        command.append("--resume-generated")
    if generation_only:
        command.append("--generation-only")
    if reuse_generated:
        command.append("--reuse-generated")
    if progress_every is not None:
        command.extend(["--progress-every", str(progress_every)])
    if min_control_win_fraction is not None:
        command.extend(["--min-control-win-fraction", str(min_control_win_fraction)])
    if min_baseline_win_fraction is not None:
        command.extend(["--min-baseline-win-fraction", str(min_baseline_win_fraction)])
    if min_baseline_relative_improvement is not None:
        command.extend(
            [
                "--min-baseline-relative-improvement",
                str(min_baseline_relative_improvement),
            ]
        )
    if require_baseline_ci_positive:
        command.append("--require-baseline-ci-positive")
    if require_clustered_baseline_ci:
        command.append("--require-clustered-baseline-ci")
    if require_baseline_dataset_match:
        command.append("--require-baseline-dataset-match")
    if bootstrap_samples is not None:
        command.extend(["--bootstrap-samples", str(bootstrap_samples)])
    if bootstrap_seed is not None:
        command.extend(["--bootstrap-seed", str(bootstrap_seed)])
    if permutation_samples is not None:
        command.extend(["--permutation-samples", str(permutation_samples)])
    if permutation_seed is not None:
        command.extend(["--permutation-seed", str(permutation_seed)])
    if min_closed_fraction is not None:
        command.extend(["--min-closed-fraction", str(min_closed_fraction)])
    if min_usable_fraction is not None:
        command.extend(["--min-usable-fraction", str(min_usable_fraction)])
    if content_family_manifest is not None:
        command.extend(["--content-family-manifest", str(content_family_manifest)])
    if content_family_coverage is not None:
        command.extend(["--content-family-coverage", str(content_family_coverage)])
    if selection_strategy is not None:
        command.extend(["--selection-strategy", str(selection_strategy)])
    if selection_seed is not None:
        command.extend(["--selection-seed", str(selection_seed)])
    if require_family_level_inference:
        command.append("--require-family-level-inference")
    if min_independent_families is not None:
        command.extend(
            ["--min-independent-families", str(min_independent_families)]
        )
    if av_model_fingerprint is not None:
        command.extend(["--av-model-fingerprint", str(av_model_fingerprint)])
    if av_tokenizer_fingerprint is not None:
        command.extend(["--av-tokenizer-fingerprint", str(av_tokenizer_fingerprint)])
    if require_generation_protocol_match:
        command.append("--require-generation-protocol-match")
    return command


def next_pending_index(queue_doc: dict[str, Any]) -> int | None:
    for index, item in enumerate(queue_doc["items"]):
        if item.get("status") == "pending":
            return index
    return None


def should_skip_training(item: dict[str, Any], expected_checkpoint: Path) -> bool:
    if not (item.get("eval_only") or item.get("skip_training_if_checkpoint_exists")):
        return False
    return expected_checkpoint.exists()


def should_cleanup_converted_hf(defaults: dict[str, Any], item: dict[str, Any]) -> bool:
    value = item.get("cleanup_converted_hf_after_eval")
    if value is None:
        value = defaults.get("cleanup_converted_hf_after_eval", True)
    return bool(value)


def conversion_dtype_for_item(defaults: dict[str, Any], item: dict[str, Any]) -> str | None:
    value = item.get("converted_hf_dtype")
    if value is None:
        value = item.get("conversion_dtype")
    if value is None:
        value = defaults.get("converted_hf_dtype")
    if value is None:
        value = defaults.get("conversion_dtype")
    if value in {None, "", "preserve", "auto"}:
        return None
    return str(value)


def roundtrip_config_for_item(
    *,
    defaults: dict[str, Any],
    item: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for source in (
        defaults.get("roundtrip"),
        (spec.get("eval") or {}).get("roundtrip"),
        item.get("roundtrip"),
    ):
        if isinstance(source, dict):
            merged.update(source)
    enabled = item.get("roundtrip_enabled")
    if enabled is None:
        enabled = defaults.get("roundtrip_enabled")
    if enabled is None:
        enabled = merged.get("enabled", False)
    if not bool(enabled):
        return None
    merged["enabled"] = True
    return merged


def _roundtrip_path(value: Any, *, queue_path: Path, queue_doc: dict[str, Any]) -> Path | None:
    if value in {None, ""}:
        return None
    return resolve_path(value, queue_path, queue_doc)


def prepare_roundtrip_plan(
    *,
    queue_path: Path,
    queue_doc: dict[str, Any],
    defaults: dict[str, Any],
    item: dict[str, Any],
    spec: dict[str, Any],
    run_dir: Path,
    expected_checkpoint: Path,
    hf_checkpoint: Path,
    train_parquet: Path,
    validation_parquet: Path,
    test_parquet: Path,
    python_bin: str,
) -> dict[str, Any] | None:
    roundtrip_cfg = roundtrip_config_for_item(
        defaults=defaults,
        item=item,
        spec=spec,
    )
    if roundtrip_cfg is None:
        return None
    ar_checkpoint_dir = _roundtrip_path(
        roundtrip_cfg.get("ar_checkpoint_dir"),
        queue_path=queue_path,
        queue_doc=queue_doc,
    )
    if ar_checkpoint_dir is None:
        raise QueueError("roundtrip.ar_checkpoint_dir is required when roundtrip is enabled")
    validation_limit = int(
        item.get("roundtrip_validation_limit")
        or defaults.get("roundtrip_validation_limit")
        or roundtrip_cfg.get("validation_limit")
        or 128
    )
    test_limit = int(
        item.get("roundtrip_test_limit")
        or defaults.get("roundtrip_test_limit")
        or roundtrip_cfg.get("test_limit")
        or 128
    )
    report = run_dir / (
        f"roundtrip_{expected_checkpoint.name}_v{validation_limit}_t{test_limit}_report.json"
    )
    generated_jsonl = report.with_suffix("").with_name(
        report.with_suffix("").name + "_generated.jsonl"
    )
    log = report.with_suffix(".log")
    command = build_roundtrip_command(
        python_bin=python_bin,
        av_hf_checkpoint=hf_checkpoint,
        ar_checkpoint_dir=ar_checkpoint_dir,
        train_parquet=train_parquet,
        validation_parquet=validation_parquet,
        test_parquet=test_parquet,
        report_json=report,
        generated_jsonl=generated_jsonl,
        generation_controls=roundtrip_cfg.get("generation_controls"),
        validation_limit=validation_limit,
        test_limit=test_limit,
        max_new_tokens=int(roundtrip_cfg.get("max_new_tokens", 200)),
        injection_scale=str(
            roundtrip_cfg.get("injection_scale", spec["training"]["injection_scale"])
        ),
        ar_batch_size=int(roundtrip_cfg.get("ar_batch_size", 4)),
        ar_max_length=roundtrip_cfg.get("ar_max_length"),
        torch_dtype=str(roundtrip_cfg.get("torch_dtype", "bfloat16")),
        control_margin=float(roundtrip_cfg.get("control_margin", 5e-5)),
        baseline_report_json=_roundtrip_path(
            roundtrip_cfg.get("baseline_report_json"),
            queue_path=queue_path,
            queue_doc=queue_doc,
        ),
        baseline_margin=float(roundtrip_cfg.get("baseline_margin", 0.0)),
        critic_template=roundtrip_cfg.get("critic_template"),
        critic_template_source=_roundtrip_path(
            roundtrip_cfg.get("critic_template_source"),
            queue_path=queue_path,
            queue_doc=queue_doc,
        ),
        av_device_map=roundtrip_cfg.get("av_device_map"),
        ar_low_cpu_mem_usage=bool(
            roundtrip_cfg.get("ar_low_cpu_mem_usage", False)
        ),
        ar_device_map=roundtrip_cfg.get("ar_device_map"),
        collect_ar_device_profile=bool(
            roundtrip_cfg.get("collect_ar_device_profile", False)
        ),
        generation_prefix=roundtrip_cfg.get("generation_prefix"),
        stop_text=roundtrip_cfg.get("stop_text"),
        generated_text_fallback=roundtrip_cfg.get("generated_text_fallback"),
        generation_backend=roundtrip_cfg.get("generation_backend"),
        generation_workers=roundtrip_cfg.get("generation_workers"),
        generation_worker_devices=roundtrip_cfg.get("generation_worker_devices"),
        stream_generated=bool(roundtrip_cfg.get("stream_generated", False)),
        resume_generated=bool(roundtrip_cfg.get("resume_generated", False)),
        progress_every=roundtrip_cfg.get("progress_every"),
        min_control_win_fraction=roundtrip_cfg.get("min_control_win_fraction"),
        min_baseline_win_fraction=roundtrip_cfg.get("min_baseline_win_fraction"),
        min_baseline_relative_improvement=roundtrip_cfg.get(
            "min_baseline_relative_improvement"
        ),
        require_baseline_ci_positive=bool(
            roundtrip_cfg.get("require_baseline_ci_positive", False)
        ),
        require_clustered_baseline_ci=bool(
            roundtrip_cfg.get("require_clustered_baseline_ci", False)
        ),
        require_baseline_dataset_match=bool(
            roundtrip_cfg.get("require_baseline_dataset_match", False)
        ),
        bootstrap_samples=roundtrip_cfg.get("bootstrap_samples"),
        bootstrap_seed=roundtrip_cfg.get("bootstrap_seed"),
        permutation_samples=roundtrip_cfg.get("permutation_samples"),
        permutation_seed=roundtrip_cfg.get("permutation_seed"),
        min_closed_fraction=roundtrip_cfg.get("min_closed_fraction"),
        min_usable_fraction=roundtrip_cfg.get("min_usable_fraction"),
        content_family_manifest=_roundtrip_path(
            roundtrip_cfg.get("content_family_manifest"),
            queue_path=queue_path,
            queue_doc=queue_doc,
        ),
        content_family_coverage=_roundtrip_path(
            roundtrip_cfg.get("content_family_coverage"),
            queue_path=queue_path,
            queue_doc=queue_doc,
        ),
        selection_strategy=roundtrip_cfg.get("selection_strategy"),
        selection_seed=roundtrip_cfg.get("selection_seed"),
        require_family_level_inference=bool(
            roundtrip_cfg.get("require_family_level_inference", False)
        ),
        min_independent_families=roundtrip_cfg.get("min_independent_families"),
        av_model_fingerprint=roundtrip_cfg.get("av_model_fingerprint"),
        av_tokenizer_fingerprint=roundtrip_cfg.get("av_tokenizer_fingerprint"),
        require_generation_protocol_match=bool(
            roundtrip_cfg.get("require_generation_protocol_match", False)
        ),
    )
    return {
        "config": roundtrip_cfg,
        "report": report,
        "generated_jsonl": generated_jsonl,
        "log": log,
        "command": command,
    }


def record_trial(
    *,
    queue_doc: dict[str, Any],
    item: dict[str, Any],
    config_path: Path,
    eval_report: Path,
    roundtrip_report: Path | None = None,
    train_log: Path,
    run_dir: Path,
) -> None:
    study_jsonl = queue_doc.get("defaults", {}).get("study_jsonl")
    if not study_jsonl:
        return
    record = nano_ar_hpo_study.build_trial_record(
        trial_name=str(item["name"]),
        config_path=config_path,
        eval_report_path=eval_report,
        roundtrip_report_path=roundtrip_report,
        train_log_path=train_log,
        run_dir=run_dir,
        status="complete",
        notes=item.get("notes"),
        task=str(item.get("study_task") or queue_doc.get("defaults", {}).get("study_task") or ("av_roundtrip" if roundtrip_report else "av")),
    )
    nano_ar_hpo_study.upsert_trial(Path(study_jsonl), record)


def process_next(queue_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    queue_doc = load_queue(queue_path)
    index = next_pending_index(queue_doc)
    if index is None:
        return {"status": "idle"}
    defaults = queue_doc.get("defaults") or {}
    item = queue_doc["items"][index]
    python_bin = str(defaults.get("python") or sys.executable)
    config_path = resolve_path(item["config"], queue_path, queue_doc)
    spec = nano_av_runner.load_and_validate_spec(config_path)
    plan = nano_av_runner.prepare_run(spec, run_id=item.get("run_id") or item["name"])
    run_dir = Path(plan["run_dir"])
    code_root = Path(spec["paths"]["code_root"])
    expected_checkpoint = expected_checkpoint_for_plan(plan)
    hf_checkpoint = converted_hf_checkpoint_for_dcp(expected_checkpoint)
    train_parquet, validation_parquet, test_parquet = eval_paths_for_plan(plan)
    eval_spec = spec.get("eval") or {}
    validation_limit = int(item.get("validation_limit") or defaults.get("validation_limit") or eval_spec.get("validation_limit") or 512)
    test_limit = int(item.get("test_limit") or defaults.get("test_limit") or eval_spec.get("test_limit") or 512)
    eval_splits = list(
        item.get("eval_splits")
        or defaults.get("eval_splits")
        or eval_spec.get("eval_splits")
        or ["validation"]
    )
    generation_examples = int(
        item.get("generation_examples")
        or defaults.get("generation_examples")
        or eval_spec.get("generation_examples")
        or 0
    )
    report_json = run_dir / f"eval_{expected_checkpoint.name}_v{validation_limit}_t{test_limit}_gen{generation_examples}_report.json"
    train_command = [python_bin, *[str(part) for part in plan["command"][1:]]]
    eval_command = build_eval_command(
        python_bin=python_bin,
        hf_checkpoint=hf_checkpoint,
        train_parquet=train_parquet,
        validation_parquet=validation_parquet,
        test_parquet=test_parquet,
        report_json=report_json,
        validation_limit=validation_limit,
        test_limit=test_limit,
        eval_splits=eval_splits,
        generation_examples=generation_examples,
        injection_scale=str(spec["training"]["injection_scale"]),
        wandb_project=str(spec["run"].get("wandb_project") or "nano30b-nla-pilot"),
        wandb_group=str(spec["run"].get("wandb_group") or "nano-av-layer-probes"),
    )
    train_log = run_dir / "train.log"
    convert_log = hf_checkpoint.with_suffix(".convert.log")
    eval_log = report_json.with_suffix(".log")
    env = env_for_run(plan, code_root)
    convert_command = build_convert_command(
        python_bin=python_bin,
        code_root=code_root,
        checkpoint_dir=expected_checkpoint,
        origin_hf_dir=Path(spec["paths"]["model_id"]),
        output_dir=hf_checkpoint,
        torch_dtype=conversion_dtype_for_item(defaults, item),
    )
    roundtrip_plan = prepare_roundtrip_plan(
        queue_path=queue_path,
        queue_doc=queue_doc,
        defaults=defaults,
        item=item,
        spec=spec,
        run_dir=run_dir,
        expected_checkpoint=expected_checkpoint,
        hf_checkpoint=hf_checkpoint,
        train_parquet=train_parquet,
        validation_parquet=validation_parquet,
        test_parquet=test_parquet,
        python_bin=python_bin,
    )
    roundtrip_report = roundtrip_plan["report"] if roundtrip_plan else None
    roundtrip_log = roundtrip_plan["log"] if roundtrip_plan else None
    roundtrip_generated_jsonl = (
        roundtrip_plan["generated_jsonl"] if roundtrip_plan else None
    )
    roundtrip_command = roundtrip_plan["command"] if roundtrip_plan else None
    if dry_run:
        return {
            "status": "dry_run",
            "item_index": index,
            "item_name": item["name"],
            "run_dir": run_dir,
            "expected_checkpoint": expected_checkpoint,
            "hf_checkpoint": hf_checkpoint,
            "train_command": train_command,
            "convert_command": convert_command,
            "eval_command": eval_command,
            "roundtrip_command": roundtrip_command,
        }
    try:
        skip_training = should_skip_training(item, expected_checkpoint)
        update_item(
            queue_path,
            index,
            status="eval_running" if skip_training else "training",
            started_at=utc_now(),
            run_dir=str(run_dir),
            expected_checkpoint=str(expected_checkpoint),
            train_log=str(train_log),
        )
        if not skip_training:
            run_logged(train_command, cwd=code_root, env=env, log_path=train_log)
        if not expected_checkpoint.exists():
            raise QueueError(f"training completed without expected checkpoint: {expected_checkpoint}")
        if not skip_training or train_log.exists():
            nano_ar_hpo_study.assert_lr_decay_canary_for_run(config_path, train_log)
        update_item(queue_path, index, hf_checkpoint=str(hf_checkpoint), convert_log=str(convert_log))
        try:
            if not (hf_checkpoint / "config.json").exists():
                run_logged(convert_command, cwd=code_root, env=env, log_path=convert_log)
            if not (hf_checkpoint / "config.json").exists():
                raise QueueError(f"conversion completed without HF config.json: {hf_checkpoint}")
            update_item(queue_path, index, status="eval_running", eval_started_at=utc_now(), eval_report=str(report_json), eval_log=str(eval_log))
            run_logged(eval_command, cwd=code_root, env=env, log_path=eval_log)
            if roundtrip_plan is not None:
                update_item(
                    queue_path,
                    index,
                    roundtrip_status="running",
                    roundtrip_started_at=utc_now(),
                    roundtrip_report=str(roundtrip_report),
                    roundtrip_log=str(roundtrip_log),
                )
                run_logged(roundtrip_command, cwd=code_root, env=env, log_path=roundtrip_log)
                update_item(
                    queue_path,
                    index,
                    roundtrip_status="complete",
                    roundtrip_completed_at=utc_now(),
                    roundtrip_generated_jsonl=str(roundtrip_generated_jsonl),
                )
            record_trial(
                queue_doc=queue_doc,
                item=item,
                config_path=config_path,
                eval_report=report_json,
                roundtrip_report=roundtrip_report,
                train_log=train_log,
                run_dir=run_dir,
            )
        finally:
            if should_cleanup_converted_hf(defaults, item):
                shutil.rmtree(hf_checkpoint, ignore_errors=True)
        if (defaults.get("cleanup_dcp_model_after_eval") is True) or (item.get("cleanup_dcp_model_after_eval") is True):
            shutil.rmtree(expected_checkpoint / "model", ignore_errors=True)
            shutil.rmtree(expected_checkpoint / "optimizer", ignore_errors=True)
        update_item(
            queue_path,
            index,
            status="complete",
            completed_at=utc_now(),
            eval_report=str(report_json),
            eval_log=str(eval_log),
            roundtrip_report=str(roundtrip_report) if roundtrip_report else None,
            roundtrip_log=str(roundtrip_log) if roundtrip_log else None,
        )
        return {
            "status": "complete",
            "item": item["name"],
            "eval_report": str(report_json),
            "roundtrip_report": str(roundtrip_report) if roundtrip_report else None,
        }
    except Exception as exc:
        update_item(queue_path, index, status="failed", failed_at=utc_now(), failure=str(exc))
        return {"status": "failed", "item": item["name"], "failure": str(exc)}


@contextlib.contextmanager
def queue_lock(queue_path: Path):
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--reset-active", action="store_true", help="Reset stale training/eval_running items to pending.")
    parser.add_argument("--run-until-empty", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the next item and print commands without running them.",
    )
    args = parser.parse_args()
    if args.reset_active:
        print(json.dumps(reset_active_items(args.queue), indent=2, sort_keys=True))
    if args.status:
        print(json.dumps(queue_status(args.queue), indent=2, sort_keys=True))
        return 0
    with queue_lock(args.queue):
        while True:
            result = process_next(args.queue, dry_run=args.dry_run)
            print(json.dumps(result, default=str, sort_keys=True), flush=True)
            if result["status"] == "idle":
                return 0
            if args.dry_run:
                return 0
            if result["status"] == "failed" and not args.continue_on_failure:
                return 1
            if not args.run_until_empty:
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
