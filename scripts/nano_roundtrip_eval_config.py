#!/usr/bin/env python3
"""Render or run config-driven Nano AV->AR round-trip evals."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_av_probe_queue import build_roundtrip_command  # noqa: E402


class RoundtripConfigError(ValueError):
    pass


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise RoundtripConfigError(f"{key!r} section is required")
    return value


def _path(value: Any, *, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def load_raw_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != "nano_roundtrip_eval.v1":
        raise RoundtripConfigError("round-trip config must use schema_version nano_roundtrip_eval.v1")
    return config


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    paths = _required_mapping(config, "paths")
    eval_cfg = _required_mapping(config, "eval")
    eval_splits = tuple(str(split) for split in eval_cfg.get("eval_splits", ["validation"]))
    dataset_controls = tuple(
        str(control) for control in eval_cfg.get("dataset_controls", [])
    )
    if len(set(dataset_controls)) != len(dataset_controls) or not set(
        dataset_controls
    ).issubset({"source_context", "source_raw"}):
        raise RoundtripConfigError(
            "eval.dataset_controls must be a unique subset of "
            "[source_context, source_raw]"
        )
    if not eval_splits or len(set(eval_splits)) != len(eval_splits) or not set(
        eval_splits
    ).issubset({"validation", "test"}):
        raise RoundtripConfigError(
            "eval.eval_splits must be a non-empty unique validation/test list"
        )
    required_paths = (
        "av_hf_checkpoint",
        "ar_checkpoint_dir",
        "train_parquet",
        "validation_parquet",
        "report_json",
    )
    missing = [key for key in required_paths if not paths.get(key)]
    if missing:
        raise RoundtripConfigError(f"paths is missing required keys: {missing}")
    if "test" in eval_splits and not paths.get("test_parquet"):
        raise RoundtripConfigError(
            "paths.test_parquet is required when eval.eval_splits includes test"
        )
    if "validation" in eval_splits and int(eval_cfg.get("validation_limit", 0)) <= 0:
        raise RoundtripConfigError("eval.validation_limit must be positive")
    if "test" in eval_splits and int(eval_cfg.get("test_limit", 0)) <= 0:
        raise RoundtripConfigError("eval.test_limit must be positive")
    if bool(eval_cfg.get("generation_only")) and bool(eval_cfg.get("reuse_generated")):
        raise RoundtripConfigError(
            "eval.generation_only and eval.reuse_generated are mutually exclusive"
        )
    if bool(eval_cfg.get("reuse_generated")) and not paths.get("generated_jsonl"):
        raise RoundtripConfigError("eval.reuse_generated requires paths.generated_jsonl")
    if bool(eval_cfg.get("require_generation_protocol_match")) and not (
        eval_cfg.get("av_model_fingerprint") and eval_cfg.get("av_tokenizer_fingerprint")
    ):
        raise RoundtripConfigError(
            "eval.require_generation_protocol_match requires explicit model and tokenizer fingerprints"
        )
    if bool(eval_cfg.get("require_generation_protocol_match")) and str(
        eval_cfg.get("generation_prefix") or ""
    ):
        raise RoundtripConfigError(
            "eval.require_generation_protocol_match requires an empty generation_prefix"
        )
    if bool(eval_cfg.get("require_family_level_inference")) and not paths.get(
        "content_family_manifest"
    ):
        raise RoundtripConfigError(
            "eval.require_family_level_inference requires paths.content_family_manifest"
        )
    if bool(eval_cfg.get("require_family_level_inference")) and not paths.get(
        "content_family_coverage"
    ):
        raise RoundtripConfigError(
            "eval.require_family_level_inference requires paths.content_family_coverage"
        )
    if bool(eval_cfg.get("require_family_level_inference")) and eval_cfg.get(
        "selection_strategy"
    ) != "family_stratified":
        raise RoundtripConfigError(
            "eval.require_family_level_inference requires selection_strategy family_stratified"
        )
    if paths.get("length_baseline_generated_jsonl") and not paths.get(
        "baseline_report_json"
    ):
        raise RoundtripConfigError(
            "paths.length_baseline_generated_jsonl requires paths.baseline_report_json"
        )
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(load_raw_config(path))


def build_command(config: dict[str, Any], *, config_path: Path, python_bin: str | None = None) -> list[str]:
    paths = config["paths"]
    eval_cfg = config["eval"]
    eval_splits = tuple(
        str(split) for split in eval_cfg.get("eval_splits", ["validation"])
    )
    base = config_path.parent
    return build_roundtrip_command(
        python_bin=python_bin or str(config.get("python") or sys.executable),
        av_hf_checkpoint=_path(paths["av_hf_checkpoint"], base=base),
        ar_checkpoint_dir=_path(paths["ar_checkpoint_dir"], base=base),
        train_parquet=_path(paths["train_parquet"], base=base),
        validation_parquet=_path(paths["validation_parquet"], base=base),
        validation_control_parquet=(
            _path(paths["validation_control_parquet"], base=base)
            if paths.get("validation_control_parquet")
            else None
        ),
        test_parquet=(
            _path(paths["test_parquet"], base=base)
            if "test" in eval_splits and paths.get("test_parquet")
            else None
        ),
        test_control_parquet=(
            _path(paths["test_control_parquet"], base=base)
            if paths.get("test_control_parquet")
            else None
        ),
        report_json=_path(paths["report_json"], base=base),
        generated_jsonl=_path(paths["generated_jsonl"], base=base) if paths.get("generated_jsonl") else None,
        expected_generation_protocol_json=(
            _path(paths["expected_generation_protocol_json"], base=base)
            if paths.get("expected_generation_protocol_json")
            else None
        ),
        prediction_cache_npz=(
            _path(paths["prediction_cache_npz"], base=base)
            if paths.get("prediction_cache_npz")
            else None
        ),
        generation_controls=eval_cfg.get("generation_controls"),
        dataset_controls=eval_cfg.get("dataset_controls"),
        validation_limit=int(eval_cfg.get("validation_limit", 64)),
        test_limit=int(eval_cfg.get("test_limit", 64)),
        eval_splits=eval_splits,
        max_new_tokens=int(eval_cfg.get("max_new_tokens", 96)),
        seed=int(eval_cfg.get("seed", 1234)),
        injection_scale=str(eval_cfg.get("injection_scale", "75")),
        ar_batch_size=int(eval_cfg.get("ar_batch_size", 4)),
        ar_max_length=eval_cfg.get("ar_max_length"),
        torch_dtype=str(eval_cfg.get("torch_dtype", "bfloat16")),
        control_margin=float(eval_cfg.get("control_margin", 0.1)),
        baseline_report_json=_path(paths["baseline_report_json"], base=base)
        if paths.get("baseline_report_json")
        else None,
        length_baseline_generated_jsonl=_path(
            paths["length_baseline_generated_jsonl"],
            base=base,
        )
        if paths.get("length_baseline_generated_jsonl")
        else None,
        baseline_margin=float(eval_cfg.get("baseline_margin", 0.0)),
        critic_template=eval_cfg.get("critic_template"),
        critic_template_source=_path(paths["critic_template_source"], base=base)
        if paths.get("critic_template_source")
        else None,
        av_device_map=eval_cfg.get("av_device_map"),
        av_low_cpu_mem_usage=bool(eval_cfg.get("av_low_cpu_mem_usage", True)),
        ar_device_map=eval_cfg.get("ar_device_map"),
        ar_low_cpu_mem_usage=bool(eval_cfg.get("ar_low_cpu_mem_usage", False)),
        collect_ar_device_profile=bool(
            eval_cfg.get("collect_ar_device_profile", False)
        ),
        generation_prefix=eval_cfg.get("generation_prefix"),
        stop_text=eval_cfg.get("stop_text"),
        generated_text_fallback=eval_cfg.get("generated_text_fallback"),
        generation_backend=eval_cfg.get("generation_backend"),
        generation_workers=eval_cfg.get("generation_workers"),
        generation_max_parallel_workers=eval_cfg.get("generation_max_parallel_workers"),
        generation_worker_devices=eval_cfg.get("generation_worker_devices"),
        stream_generated=bool(eval_cfg.get("stream_generated", False)),
        resume_generated=bool(eval_cfg.get("resume_generated", False)),
        generation_only=bool(eval_cfg.get("generation_only", False)),
        reuse_generated=bool(eval_cfg.get("reuse_generated", False)),
        progress_every=eval_cfg.get("progress_every"),
        min_control_win_fraction=eval_cfg.get("min_control_win_fraction"),
        min_baseline_win_fraction=eval_cfg.get("min_baseline_win_fraction"),
        min_baseline_relative_improvement=eval_cfg.get(
            "min_baseline_relative_improvement"
        ),
        require_baseline_ci_positive=bool(
            eval_cfg.get("require_baseline_ci_positive", False)
        ),
        require_clustered_baseline_ci=bool(
            eval_cfg.get("require_clustered_baseline_ci", False)
        ),
        require_baseline_dataset_match=bool(
            eval_cfg.get("require_baseline_dataset_match", False)
        ),
        bootstrap_samples=eval_cfg.get("bootstrap_samples"),
        bootstrap_seed=eval_cfg.get("bootstrap_seed"),
        permutation_samples=eval_cfg.get("permutation_samples"),
        permutation_seed=eval_cfg.get("permutation_seed"),
        min_closed_fraction=eval_cfg.get("min_closed_fraction"),
        min_usable_fraction=eval_cfg.get("min_usable_fraction"),
        content_family_manifest=_path(paths["content_family_manifest"], base=base)
        if paths.get("content_family_manifest")
        else None,
        content_family_coverage=_path(paths["content_family_coverage"], base=base)
        if paths.get("content_family_coverage")
        else None,
        selection_strategy=eval_cfg.get("selection_strategy"),
        selection_seed=eval_cfg.get("selection_seed"),
        require_family_level_inference=bool(
            eval_cfg.get("require_family_level_inference", False)
        ),
        min_independent_families=eval_cfg.get("min_independent_families"),
        av_model_fingerprint=eval_cfg.get("av_model_fingerprint"),
        av_tokenizer_fingerprint=eval_cfg.get("av_tokenizer_fingerprint"),
        require_generation_protocol_match=bool(
            eval_cfg.get("require_generation_protocol_match", False)
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--python", dest="python_bin", help="Override Python executable.")
    parser.add_argument("--run", action="store_true", help="Run the rendered command.")
    parser.add_argument("--cwd", type=Path, help="Working directory for --run. Defaults to config paths.code_root or cwd.")
    parser.add_argument("--report-json", type=Path, help="Write the rendered command/config summary.")
    args = parser.parse_args()

    config = load_config(args.config)
    command = build_command(config, config_path=args.config, python_bin=args.python_bin)
    payload = {"config": str(args.config), "command": command}
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.run:
        cwd = args.cwd or Path((config.get("paths") or {}).get("code_root") or os.getcwd())
        subprocess.run(command, cwd=cwd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
