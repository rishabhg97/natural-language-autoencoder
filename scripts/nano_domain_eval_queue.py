#!/usr/bin/env python3
"""Run a resumable, config-driven Nano safety-domain evaluation chain."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

import nano_domain_eval
import nano_roundtrip_queue


STATE_SCHEMA_VERSION = "nano_domain_eval_queue_state.v1"
PHASES = (
    "build-manifest",
    "prepare-av",
    "extract",
    "describe",
    "behavior",
    "analyze",
)
SHARDED_PHASE_CONFIG_PREFIX = {
    "extract": "extraction",
    "describe": "description",
    "behavior": "behavior",
}


class DomainQueueError(ValueError):
    """Raised when the domain queue cannot be launched reproducibly."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["output_root"]) / "queue_state.json"


def _log_path(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["output_root"]) / "logs" / "domain_eval_chain.log"


def _new_state(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "config": str(config_path),
        "config_sha256": nano_domain_eval.config_sha256(config),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "pending",
        "phases": {phase: {"status": "pending"} for phase in PHASES},
    }


def _load_state(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = _state_path(config)
    if not path.is_file():
        return _new_state(config_path, config)
    state = json.loads(path.read_text())
    expected = nano_domain_eval.config_sha256(config)
    if state.get("config_sha256") != expected:
        raise DomainQueueError(
            "existing queue state belongs to a different config; use a new output_root"
        )
    return state


def _write_state(config: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = utc_now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _record_phase(
    config: Mapping[str, Any],
    state: dict[str, Any],
    phase: str,
    status: str,
    **details: Any,
) -> None:
    phase_state = state["phases"][phase]
    if status in {"running", "complete"}:
        phase_state.pop("error", None)
    phase_state.update({"status": status, **details})
    if status == "running":
        phase_state["started_at"] = utc_now()
    elif status in {"complete", "failed"}:
        phase_state["finished_at"] = utc_now()
    state["status"] = "failed" if status == "failed" else "running"
    _write_state(config, state)


def _run_command(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(f"\n[{utc_now()}] $ {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(
            list(command), cwd=cwd, env=dict(env), stdout=handle, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise DomainQueueError(
            f"command failed with exit {result.returncode}: {' '.join(command)}"
        )


def _run_sharded_gpu_phase(
    *,
    phase: str,
    workers: int,
    gpus_per_worker: int,
    python_bin: str,
    config_path: Path,
    code_root: Path,
    env: Mapping[str, str],
    log_path: Path,
) -> None:
    if workers < 1 or gpus_per_worker < 1:
        raise DomainQueueError(f"{phase} worker and GPU counts must be positive")
    commands: list[tuple[list[str], dict[str, str], Path]] = []
    for worker in range(workers):
        worker_env = dict(env)
        first_gpu = worker * gpus_per_worker
        worker_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in range(first_gpu, first_gpu + gpus_per_worker)
        )
        commands.append(
            (
                [
                    python_bin,
                    "scripts/nano_domain_eval.py",
                    phase,
                    str(config_path),
                    "--shard-index",
                    str(worker),
                    "--shard-count",
                    str(workers),
                ],
                worker_env,
                log_path.with_name(f"{phase}_shard_{worker:02d}.log"),
            )
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _run_command,
                command,
                cwd=code_root,
                env=worker_env,
                log_path=worker_log,
            )
            for command, worker_env, worker_log in commands
        ]
        for future in futures:
            future.result()
    _run_command(
        [
            python_bin,
            "scripts/nano_domain_eval.py",
            f"merge-{phase}",
            str(config_path),
            "--shard-count",
            str(workers),
        ],
        cwd=code_root,
        env=env,
        log_path=log_path,
    )


def _reuse_manifest(config: Mapping[str, Any], state: dict[str, Any]) -> None:
    paths = config["paths"]
    manifest_path = Path(paths["manifest_jsonl"])
    manifest_report = json.loads(Path(paths["manifest_report_json"]).read_text())
    if not manifest_report.get("passed"):
        raise DomainQueueError("cannot reuse a failed manifest report")
    if manifest_report.get("manifest", {}).get("sha256") != nano_domain_eval.sha256_file(
        manifest_path
    ):
        raise DomainQueueError("manifest hash changed; extraction cannot be reused")
    state["phases"]["build-manifest"] = {
        "status": "complete",
        "reused_at": utc_now(),
        "reuse_validation": "passed",
    }


def _reuse_through_prepare_av(
    config: Mapping[str, Any], state: dict[str, Any]
) -> None:
    _reuse_manifest(config, state)

    av_hf = Path(config["models"]["av_hf"])
    if not (av_hf / "config.json").is_file():
        raise DomainQueueError("prepared AV checkpoint is unavailable for reuse")
    state["phases"]["prepare-av"] = {
        "status": "complete",
        "reused_at": utc_now(),
        "reuse_validation": "passed",
    }


def _reuse_through_extract(config: Mapping[str, Any], state: dict[str, Any]) -> None:
    _reuse_manifest(config, state)
    paths = config["paths"]
    manifest_path = Path(paths["manifest_jsonl"])

    activation_path = Path(paths["activations_jsonl"])
    activation_report = json.loads(Path(paths["activation_report_json"]).read_text())
    evaluation = config["evaluation"]
    checks = {
        "passed": activation_report.get("passed") is True,
        "manifest_sha256": activation_report.get("manifest_sha256")
        == nano_domain_eval.sha256_file(manifest_path),
        "activation_sha256": activation_report.get("activations", {}).get("sha256")
        == nano_domain_eval.sha256_file(activation_path),
        "boundary": int(activation_report.get("boundary", -1))
        == int(evaluation.get("boundary", 33)),
        "capture_backend": activation_report.get("capture_backend")
        == "truncated_causal_prefix_per_anchor",
        "pre_condition_invariance": activation_report.get(
            "pre_condition_invariance", {}
        ).get("passed")
        is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise DomainQueueError(
            f"activation reuse validation failed: {', '.join(failures)}"
        )
    state["phases"]["extract"] = {
        "status": "complete",
        "reused_at": utc_now(),
        "reuse_validation": "passed",
    }

    # Extraction depends only on the base model, so checkpoint-comparison runs may
    # reuse it while preparing a different AV checkpoint. Reuse AV preparation only
    # when the configured HF artifact is actually present.
    av_hf = Path(config["models"]["av_hf"])
    if (av_hf / "config.json").is_file():
        state["phases"]["prepare-av"] = {
            "status": "complete",
            "reused_at": utc_now(),
            "reuse_validation": "passed",
        }


def _reuse_through_describe(config: Mapping[str, Any], state: dict[str, Any]) -> None:
    _reuse_through_extract(config, state)
    paths = config["paths"]
    activation_path = Path(paths["activations_jsonl"])
    description_path = Path(paths["descriptions_jsonl"])
    report = json.loads(Path(paths["description_report_json"]).read_text())
    checks = {
        "passed": report.get("passed") is True,
        "activation_sha256": report.get("activation_sha256")
        == nano_domain_eval.sha256_file(activation_path),
        "description_sha256": report.get("descriptions", {}).get("sha256")
        == nano_domain_eval.sha256_file(description_path),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise DomainQueueError(
            f"description reuse validation failed: {', '.join(failures)}"
        )
    state["phases"]["describe"] = {
        "status": "complete",
        "reused_at": utc_now(),
        "reuse_validation": "passed",
    }


def run_queue(
    config_path: Path,
    *,
    resume: bool = True,
    reuse_through: str | None = None,
) -> dict[str, Any]:
    config = nano_domain_eval.load_config(config_path)
    code_root = Path(config["paths"]["code_root"])
    if not code_root.is_dir():
        raise DomainQueueError(f"code_root does not exist: {code_root}")
    python_bin = str((config.get("execution") or {}).get("python") or sys.executable)
    state = _load_state(config_path, config) if resume else _new_state(config_path, config)
    if reuse_through == "prepare-av":
        _reuse_through_prepare_av(config, state)
        _write_state(config, state)
    elif reuse_through == "extract":
        _reuse_through_extract(config, state)
        _write_state(config, state)
    elif reuse_through == "describe":
        _reuse_through_describe(config, state)
        _write_state(config, state)
    log_path = _log_path(config)
    env = os.environ.copy()
    env.update(
        {
            "WANDB_MODE": "offline",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )

    try:
        for phase in PHASES:
            # A fresh state may still contain explicitly validated reusable phases.
            # Skip every completed phase; --no-resume already resets ordinary phases
            # to pending before any requested reuse boundary is applied.
            if state["phases"][phase]["status"] == "complete":
                continue
            _record_phase(config, state, phase, "running")
            if phase == "prepare-av":
                queue_doc = {
                    "defaults": {"av_checkpoint_prepare": config["checkpoint_prepare"]}
                }
                report = nano_roundtrip_queue.prepare_av_checkpoint(
                    queue_doc, code_root=code_root, env=env
                )
                _record_phase(
                    config,
                    state,
                    phase,
                    "complete",
                    fingerprint_report=(report or {}).get("av_model_fingerprint"),
                )
                continue
            if phase in SHARDED_PHASE_CONFIG_PREFIX:
                execution = config.get("execution") or {}
                prefix = SHARDED_PHASE_CONFIG_PREFIX[phase]
                workers = int(execution.get(f"{prefix}_workers", 1))
                gpus_per_worker = int(
                    execution.get(f"{prefix}_gpus_per_worker", 1)
                )
                if workers > 1:
                    _run_sharded_gpu_phase(
                        phase=phase,
                        workers=workers,
                        gpus_per_worker=gpus_per_worker,
                        python_bin=python_bin,
                        config_path=config_path,
                        code_root=code_root,
                        env=env,
                        log_path=log_path,
                    )
                    _record_phase(config, state, phase, "complete")
                    continue
            command = [
                python_bin,
                "scripts/nano_domain_eval.py",
                phase,
                str(config_path),
            ]
            _run_command(command, cwd=code_root, env=env, log_path=log_path)
            _record_phase(config, state, phase, "complete")
    except Exception as exc:
        active = next(
            (name for name in PHASES if state["phases"][name]["status"] == "running"),
            "unknown",
        )
        if active in state["phases"]:
            _record_phase(
                config,
                state,
                active,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        raise

    cleanup = bool(config["checkpoint_prepare"].get("cleanup_after_queue", False))
    av_hf = Path(config["models"]["av_hf"])
    if cleanup and av_hf.exists():
        shutil.rmtree(av_hf)
        state["av_hf_cleaned_at"] = utc_now()
    state["status"] = "complete"
    state["finished_at"] = utc_now()
    _write_state(config, state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--reuse-through", choices=("prepare-av", "extract", "describe")
    )
    args = parser.parse_args(argv)
    try:
        result = run_queue(
            args.config,
            resume=not args.no_resume,
            reuse_through=args.reuse_through,
        )
    except (OSError, ValueError, DomainQueueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
