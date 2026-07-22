#!/usr/bin/env python3
"""Orchestrate the isolated Observatory GPU pipeline with fail-closed gates."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .launch_after_prepare import process_exists
from .model_runtime import hf_checkpoint_complete


SCHEMA_VERSION = "nano_viz_pipeline_supervisor.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def wait_for_process(pid: int, *, poll_seconds: float) -> None:
    while process_exists(pid):
        time.sleep(poll_seconds)


def validate_critic_preparation(report_path: Path, required_hf: list[Path]) -> dict[str, Any]:
    if not report_path.is_file():
        raise ValueError(f"critic preparation report is missing: {report_path}")
    report = json.loads(report_path.read_text())
    if not report.get("passed"):
        raise ValueError("critic preparation report did not pass")
    stages = report.get("critic_stages") or {}
    if len(stages) < 2 or any(
        stage.get("status") not in {"copied", "reused"} for stage in stages.values()
    ):
        raise ValueError("critic preparation did not stage both critic checkpoints")
    incomplete = [str(path) for path in required_hf if not hf_checkpoint_complete(path)]
    if incomplete:
        raise ValueError(f"required staged checkpoints are incomplete: {incomplete}")
    return report


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "WANDB_MODE": "offline",
                "TOKENIZERS_PARALLELISM": "false",
                "NLA_TRAIN_MAMBA_KERNEL_MODE": "unfused_torch_conv",
                "NLA_CRITIC_FWD_DISABLE_MAMBA_FAST_PATH": "1",
                "PYTHONPATH": f"{cwd / 'scripts'}:{cwd / 'external' / 'natural_language_autoencoders'}",
            },
        )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-pid", type=int, required=True)
    parser.add_argument("--prepare-report", type=Path, required=True)
    parser.add_argument("--required-hf", type=Path, action="append", default=[])
    parser.add_argument("--base-fetch-pid", type=int)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-queue", type=Path, required=True)
    parser.add_argument("--base-queue", type=Path, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "waiting_for_critics",
        "created_at": utc_now(),
        "prepare_pid": args.prepare_pid,
        "base_fetch_pid": args.base_fetch_pid,
    }
    write_state(args.state_json, state)
    wait_for_process(args.prepare_pid, poll_seconds=args.poll_seconds)
    try:
        preparation = validate_critic_preparation(
            args.prepare_report, list(args.required_hf)
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        state.update(status="failed", stage="critic_preparation", error=str(exc), updated_at=utc_now())
        write_state(args.state_json, state)
        return 2
    state["critic_preparation"] = preparation.get("critic_stages")

    if args.base_fetch_pid and process_exists(args.base_fetch_pid):
        os.kill(args.base_fetch_pid, signal.SIGCONT)
        state["base_fetch_resumed_at"] = utc_now()
    state.update(status="running", stage="gpu_queue", updated_at=utc_now())
    write_state(args.state_json, state)
    gpu_returncode = run_logged(
        [
            sys.executable,
            "-m",
            "observatory.queue",
            "--queue",
            str(args.gpu_queue),
        ],
        cwd=args.code_root,
        log_path=args.log_dir / "gpu_queue.log",
    )
    state["gpu_queue_returncode"] = gpu_returncode
    if gpu_returncode != 0:
        state.update(status="failed", stage="gpu_queue", updated_at=utc_now())
        write_state(args.state_json, state)
        return gpu_returncode

    if args.base_fetch_pid:
        state.update(stage="waiting_for_base_download", updated_at=utc_now())
        write_state(args.state_json, state)
        wait_for_process(args.base_fetch_pid, poll_seconds=args.poll_seconds)
    state.update(stage="base_stage", updated_at=utc_now())
    write_state(args.state_json, state)
    base_returncode = run_logged(
        [
            sys.executable,
            "-m",
            "observatory.fetch_base",
            "--config",
            str(args.config),
        ],
        cwd=args.code_root,
        log_path=args.log_dir / "base_stage.log",
    )
    state["base_stage_returncode"] = base_returncode
    if base_returncode != 0:
        state.update(status="failed", stage="base_stage", updated_at=utc_now())
        write_state(args.state_json, state)
        return base_returncode

    state.update(stage="base_queue", updated_at=utc_now())
    write_state(args.state_json, state)
    base_queue_returncode = run_logged(
        [
            sys.executable,
            "-m",
            "observatory.queue",
            "--queue",
            str(args.base_queue),
        ],
        cwd=args.code_root,
        log_path=args.log_dir / "base_queue.log",
    )
    state["base_queue_returncode"] = base_queue_returncode
    state.update(
        status="complete" if base_queue_returncode == 0 else "failed",
        stage="complete" if base_queue_returncode == 0 else "base_queue",
        updated_at=utc_now(),
    )
    write_state(args.state_json, state)
    return base_queue_returncode


if __name__ == "__main__":
    raise SystemExit(main())
