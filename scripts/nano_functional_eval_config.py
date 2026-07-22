#!/usr/bin/env python3
"""Render or run config-driven Nano target-model functional evaluations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "nano_functional_eval.v1"


class FunctionalEvalConfigError(ValueError):
    """Raised when a functional-evaluation config is incomplete or invalid."""


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise FunctionalEvalConfigError(f"{key!r} section is required")
    return value


def _path(value: Any, *, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise FunctionalEvalConfigError(
            f"functional eval config must use schema_version {SCHEMA_VERSION}"
        )

    paths = _mapping(config, "paths")
    eval_cfg = _mapping(config, "eval")
    required_paths = (
        "generated_jsonl",
        "ar_checkpoint_dir",
        "source_base_parquet",
        "target_model",
        "report_json",
    )
    missing_paths = [key for key in required_paths if not paths.get(key)]
    if missing_paths:
        raise FunctionalEvalConfigError(
            f"paths is missing required keys: {missing_paths}"
        )

    required_eval = (
        "boundary",
        "validation_limit",
        "test_limit",
        "batch_size",
        "identity_relative_l2",
        "identity_max_abs",
        "identity_one_minus_cos",
    )
    missing_eval = [key for key in required_eval if eval_cfg.get(key) is None]
    if missing_eval:
        raise FunctionalEvalConfigError(
            f"eval is missing required keys: {missing_eval}"
        )
    positive = ("boundary", "validation_limit", "test_limit", "batch_size")
    if any(int(eval_cfg[key]) <= 0 for key in positive):
        raise FunctionalEvalConfigError(
            "eval boundary, limits, and batch_size must be positive"
        )
    tolerances = (
        "identity_relative_l2",
        "identity_max_abs",
        "identity_one_minus_cos",
    )
    if any(float(eval_cfg[key]) < 0.0 for key in tolerances):
        raise FunctionalEvalConfigError("identity tolerances must be non-negative")
    if int(eval_cfg.get("bootstrap_resamples", 10_000)) <= 0:
        raise FunctionalEvalConfigError("eval.bootstrap_resamples must be positive")
    if int(eval_cfg.get("min_independent_families", 1)) <= 0:
        raise FunctionalEvalConfigError(
            "eval.min_independent_families must be positive"
        )
    eval_splits = eval_cfg.get("eval_splits", ["validation", "test"])
    if not isinstance(eval_splits, list) or not eval_splits:
        raise FunctionalEvalConfigError("eval.eval_splits must be a non-empty list")
    if len(eval_splits) != len(set(eval_splits)):
        raise FunctionalEvalConfigError("eval.eval_splits must be unique")
    unknown_splits = sorted(set(eval_splits) - {"validation", "test"})
    if unknown_splits:
        raise FunctionalEvalConfigError(
            f"eval.eval_splits contains unsupported values: {unknown_splits}"
        )
    return config


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_command(
    config: dict[str, Any],
    *,
    config_path: Path,
    python_bin: str | None = None,
) -> list[str]:
    paths = config["paths"]
    eval_cfg = config["eval"]
    base = config_path.parent
    command = [
        python_bin or str(config.get("python") or sys.executable),
        "scripts/eval_nano_r33_functional_recovery.py",
        "--generated-jsonl",
        str(_path(paths["generated_jsonl"], base=base)),
        "--ar-checkpoint-dir",
        str(_path(paths["ar_checkpoint_dir"], base=base)),
        "--source-base-parquet",
        str(_path(paths["source_base_parquet"], base=base)),
        "--target-model",
        str(paths["target_model"]),
        "--report-json",
        str(_path(paths["report_json"], base=base)),
        "--config-path",
        str(config_path.resolve()),
        "--boundary",
        str(int(eval_cfg["boundary"])),
        "--validation-limit",
        str(int(eval_cfg["validation_limit"])),
        "--test-limit",
        str(int(eval_cfg["test_limit"])),
        "--eval-splits",
        *[str(split) for split in eval_cfg.get("eval_splits", ["validation", "test"])],
        "--batch-size",
        str(int(eval_cfg["batch_size"])),
        "--identity-relative-l2",
        str(float(eval_cfg["identity_relative_l2"])),
        "--identity-max-abs",
        str(float(eval_cfg["identity_max_abs"])),
        "--identity-one-minus-cos",
        str(float(eval_cfg["identity_one_minus_cos"])),
        "--bootstrap-resamples",
        str(int(eval_cfg.get("bootstrap_resamples", 10_000))),
        "--control",
        str(eval_cfg.get("control", "real")),
    ]

    path_options = {
        "sft_generated_jsonl": "--sft-generated-jsonl",
        "critic_template_source": "--critic-template-source",
        "mean_activation_parquet": "--mean-activation-parquet",
    }
    for key, flag in path_options.items():
        if paths.get(key):
            _append_option(command, flag, _path(paths[key], base=base))

    scalar_options = {
        "ar_batch_size": "--ar-batch-size",
        "ar_max_length": "--ar-max-length",
        "ar_torch_dtype": "--ar-torch-dtype",
        "ar_device_map": "--ar-device-map",
        "target_torch_dtype": "--target-torch-dtype",
        "target_device_map": "--target-device-map",
        "target_revision": "--target-revision",
        "critic_template": "--critic-template",
        "generated_text_fallback": "--generated-text-fallback",
        "seed": "--seed",
        "source_batch_size": "--source-batch-size",
        "selection_strategy": "--selection-strategy",
        "min_independent_families": "--min-independent-families",
    }
    for key, flag in scalar_options.items():
        _append_option(command, flag, eval_cfg.get(key))

    boolean_options = {
        "target_local_files_only": "--target-local-files-only",
        "target_trust_remote_code": "--target-trust-remote-code",
        "require_generation_identity": "--require-generation-identity",
    }
    for key, flag in boolean_options.items():
        if key in eval_cfg:
            command.append(flag if bool(eval_cfg[key]) else "--no-" + flag[2:])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--python", dest="python_bin")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    command = build_command(config, config_path=args.config, python_bin=args.python_bin)
    payload = {"schema_version": SCHEMA_VERSION, "config": str(args.config), "command": command}
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.run:
        cwd = args.cwd or Path((config.get("paths") or {}).get("code_root") or os.getcwd())
        return subprocess.run(command, cwd=cwd, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
