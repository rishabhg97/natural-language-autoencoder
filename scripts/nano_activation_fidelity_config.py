#!/usr/bin/env python3
"""Build or run a config-driven Nano activation-fidelity diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "nano_activation_fidelity_eval.v1"


def _path(value: Any, *, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config must use schema_version {SCHEMA_VERSION}")
    paths = config.get("paths") or {}
    for key in (
        "generated_jsonl",
        "source_base_parquet",
        "target_model",
        "report_json",
    ):
        if not paths.get(key):
            raise ValueError(f"paths.{key} is required")
    evaluation = config.get("eval") or {}
    for key in ("boundary", "validation_limit", "test_limit"):
        if evaluation.get(key) is None:
            raise ValueError(f"eval.{key} is required")
    return config


def _append(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_command(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    python_bin: str | None = None,
) -> list[str]:
    config_path = Path(config_path)
    base = config_path.parent
    paths = config["paths"]
    evaluation = config["eval"]
    command = [
        python_bin or str(config.get("python") or sys.executable),
        "scripts/diagnose_nano_r33_activation_identity.py",
        "--generated-jsonl",
        str(_path(paths["generated_jsonl"], base=base)),
        "--source-base-parquet",
        str(_path(paths["source_base_parquet"], base=base)),
        "--target-model",
        str(paths["target_model"]),
        "--report-json",
        str(_path(paths["report_json"], base=base)),
        "--boundary",
        str(int(evaluation["boundary"])),
        "--validation-limit",
        str(int(evaluation["validation_limit"])),
        "--test-limit",
        str(int(evaluation["test_limit"])),
        "--eval-splits",
        *[str(split) for split in evaluation.get("eval_splits", ["validation", "test"])],
    ]

    for key, flag in {
        "mean_activation_parquet": "--mean-activation-parquet",
        "content_family_manifest": "--content-family-manifest",
        "extraction_source_parquet": "--extraction-source-parquet",
    }.items():
        if paths.get(key):
            _append(command, flag, _path(paths[key], base=base))

    for key, flag in {
        "target_model_fingerprint": "--target-model-fingerprint",
        "source_batch_size": "--source-batch-size",
        "batch_size": "--batch-size",
        "extraction_batch_size": "--extraction-batch-size",
        "selection_strategy": "--selection-strategy",
        "selection_seed": "--selection-seed",
        "fidelity_max_relative_l2": "--fidelity-max-relative-l2",
        "fidelity_max_abs": "--fidelity-max-abs",
        "fidelity_max_one_minus_cos": "--fidelity-max-one-minus-cos",
        "target_torch_dtype": "--target-torch-dtype",
        "target_device_map": "--target-device-map",
        "target_revision": "--target-revision",
        "model_loader": "--model-loader",
        "float32_matmul_precision": "--float32-matmul-precision",
        "cublas_workspace_config": "--cublas-workspace-config",
        "seed": "--seed",
    }.items():
        _append(command, flag, evaluation.get(key))

    for key, flag in {
        "publication_mode": "--publication-mode",
        "repeat_full_forward": "--repeat-full-forward",
        "target_local_files_only": "--target-local-files-only",
        "target_trust_remote_code": "--target-trust-remote-code",
        "deterministic_algorithms": "--deterministic-algorithms",
        "allow_tf32": "--allow-tf32",
        "cudnn_benchmark": "--cudnn-benchmark",
    }.items():
        if key in evaluation:
            command.append(flag if bool(evaluation[key]) else "--no-" + flag[2:])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--python", dest="python_bin")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--command-report-json", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    command = build_command(
        config,
        config_path=args.config,
        python_bin=args.python_bin,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": str(args.config.resolve()),
        "command": command,
    }
    if args.command_report_json:
        args.command_report_json.parent.mkdir(parents=True, exist_ok=True)
        args.command_report_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.run:
        cwd = args.cwd or Path((config.get("paths") or {}).get("code_root") or os.getcwd())
        paths = config.get("paths") or {}
        log_path = (
            _path(paths["log_file"], base=args.config.parent)
            if paths.get("log_file")
            else None
        )
        started = time.time()
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as handle:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    check=False,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        else:
            completed = subprocess.run(command, cwd=cwd, check=False)
        run_report = {
            **payload,
            "cwd": str(cwd),
            "log_file": str(log_path) if log_path is not None else None,
            "returncode": int(completed.returncode),
            "elapsed_seconds": time.time() - started,
        }
        if paths.get("runner_report_json"):
            report_path = _path(
                paths["runner_report_json"],
                base=args.config.parent,
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(run_report, indent=2, sort_keys=True) + "\n"
            )
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
