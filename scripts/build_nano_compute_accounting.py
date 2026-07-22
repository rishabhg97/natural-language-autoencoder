#!/usr/bin/env python3
"""Build hash-bound compute accounting from Nano Miles training logs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "nano_compute_accounting.v1"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
START_RE = re.compile(r"^# started_utc=(\S+)$")
COMPLETE_RE = re.compile(r"^# completed_utc=(\S+)$")
STEP_RE = re.compile(r"\bstep\s+(\d+):\s+(\{.*\})\s*$")

SYSTEM_METRICS = (
    "train/nla/system/cuda_max_memory_allocated_gib",
    "train/nla/system/cuda_max_memory_reserved_gib",
    "train/nla/system/nvidia_smi_memory_used_mib",
    "train/nla/system/nvidia_smi_gpu_util_pct",
    "train/nla/system/nvidia_smi_memory_util_pct",
    "train/nla/system/nvidia_smi_power_w",
)

COMMAND_FLAGS = (
    "--global-batch-size",
    "--micro-batch-size",
    "--lr",
    "--min-lr",
    "--lr-warmup-iters",
    "--num-rollout",
    "--actor-num-gpus-per-node",
)


class AccountingError(ValueError):
    """Raised when a compute record cannot be proven from its inputs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise AccountingError(f"timestamp lacks timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _command_values(command: str | None) -> dict[str, str]:
    if not command:
        return {}
    tokens = shlex.split(command)
    result: dict[str, str] = {}
    for flag in COMMAND_FLAGS:
        if flag not in tokens:
            continue
        index = tokens.index(flag)
        if index + 1 >= len(tokens):
            raise AccountingError(f"missing value after {flag}")
        result[flag.removeprefix("--").replace("-", "_")] = tokens[index + 1]
    return result


def parse_train_log(path: Path, *, expected_updates: int) -> dict[str, Any]:
    if not path.is_file():
        raise AccountingError(f"train log does not exist: {path}")

    started: datetime | None = None
    completed: datetime | None = None
    command: str | None = None
    step_metrics: list[tuple[int, dict[str, Any]]] = []

    with path.open(errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line).rstrip("\n")
            if match := START_RE.match(line):
                started = _parse_utc(match.group(1))
            elif match := COMPLETE_RE.match(line):
                completed = _parse_utc(match.group(1))
            elif command is None and "train.py" in line and "--num-rollout" in line:
                command = line

            match = STEP_RE.search(line)
            if not match:
                continue
            try:
                payload = ast.literal_eval(match.group(2))
            except (SyntaxError, ValueError) as exc:
                raise AccountingError(
                    f"could not parse metrics at logged step {match.group(1)} in {path}"
                ) from exc
            if not isinstance(payload, dict):
                raise AccountingError(f"step payload is not a mapping in {path}")
            step_metrics.append((int(match.group(1)), payload))

    if started is None or completed is None:
        raise AccountingError(f"missing start/completion timestamp in {path}")
    if completed <= started:
        raise AccountingError(f"non-positive wall time in {path}")
    if not step_metrics:
        raise AccountingError(f"no optimizer-step metrics found in {path}")

    unique_steps = sorted({step for step, _ in step_metrics})
    expected_steps = list(range(expected_updates))
    if unique_steps != expected_steps:
        raise AccountingError(
            f"optimizer steps do not match 0..{expected_updates - 1} in {path}: "
            f"observed {len(unique_steps)} unique, final={unique_steps[-1]}"
        )

    by_step: dict[int, dict[str, Any]] = {}
    for step, payload in step_metrics:
        by_step[step] = payload
    final = by_step[unique_steps[-1]]

    envelope: dict[str, Any] = {}
    for key in SYSTEM_METRICS:
        values = [
            float(payload[key])
            for payload in by_step.values()
            if key in payload and isinstance(payload[key], (int, float))
        ]
        if not values:
            continue
        if not all(math.isfinite(value) for value in values):
            raise AccountingError(f"nonfinite system metric {key} in {path}")
        short_key = key.removeprefix("train/nla/system/")
        envelope[f"{short_key}_max"] = max(values)
        envelope[f"{short_key}_mean"] = sum(values) / len(values)
        envelope[f"{short_key}_logged_steps"] = len(values)

    wall_seconds = (completed - started).total_seconds()
    return {
        "train_log": str(path),
        "train_log_sha256": _sha256(path),
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "completed_utc": completed.isoformat().replace("+00:00", "Z"),
        "wall_seconds": wall_seconds,
        "optimizer_updates": expected_updates,
        "logged_optimizer_steps": len(unique_steps),
        "final_step": unique_steps[-1],
        "final_metrics": {
            key: final[key]
            for key in (
                "train/loss",
                "train/fve_nrm",
                "train/grad_norm",
                "train/lr-pg_0",
            )
            if key in final
        },
        "resolved_command": _command_values(command),
        "system_metric_semantics": (
            "Values are the rank-aggregated metrics emitted by the training "
            "logger; they are not a per-device profiler trace."
        ),
        "system_envelope": envelope,
    }


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise AccountingError(
            f"config schema_version must be {SCHEMA_VERSION!r}"
        )
    raw_runs = config.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise AccountingError("config runs must be a non-empty list")

    runs: list[dict[str, Any]] = []
    total_gpu_hours = 0.0
    for raw in raw_runs:
        if not isinstance(raw, dict):
            raise AccountingError("each run entry must be a mapping")
        name = str(raw["name"])
        gpu_count = int(raw["gpu_count"])
        expected_updates = int(raw["expected_updates"])
        if gpu_count <= 0 or expected_updates <= 0:
            raise AccountingError(f"invalid GPU/update count for {name}")
        parsed = parse_train_log(
            Path(raw["train_log"]), expected_updates=expected_updates
        )
        gpu_hours = parsed["wall_seconds"] * gpu_count / 3600.0
        total_gpu_hours += gpu_hours
        runs.append(
            {
                "name": name,
                "component": str(raw["component"]),
                "gpu_type": str(raw["gpu_type"]),
                "gpu_count": gpu_count,
                "gpu_hours": gpu_hours,
                "checkpoint_identity": raw.get("checkpoint_identity"),
                **parsed,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Accounting covers the listed successful training processes only. "
            "It excludes extraction, conversion, evaluation, queue idle time, "
            "and failed attempts unless separately enumerated."
        ),
        "runs": runs,
        "successful_training_gpu_hours": total_gpu_hours,
        "known_exclusions": list(config.get("known_exclusions", [])),
        "failed_attempts": list(config.get("failed_attempts", [])),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise AccountingError("config root must be a mapping")
    report = build_report(config)
    output = Path(config["output_json"])
    _write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
