#!/usr/bin/env python3
"""Start one Nano queue only after another queue completes successfully."""

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


TERMINAL_FAILURES = {"failed", "cancelled", "blocked", "blocked_missing_dataset"}
KNOWN_STATUSES = {"pending", "running", "training", "eval_running", "complete", *TERMINAL_FAILURES}


class QueueChainError(ValueError):
    """Raised when queue state cannot safely drive a chained launch."""


def chain_environment(code_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["NANO_QUEUE_CODE_ROOT"] = str(code_root.resolve())
    return env


def queue_state(
    queue: dict[str, Any],
    *,
    item_name: str | None = None,
    require_gate_pass: bool = False,
) -> str:
    items = queue.get("items")
    if not isinstance(items, list) or not items:
        raise QueueChainError("queue requires a non-empty items list")
    if item_name is not None:
        matches = [item for item in items if isinstance(item, dict) and item.get("name") == item_name]
        if len(matches) != 1:
            detail = "not found" if not matches else "is duplicated"
            raise QueueChainError(f"prerequisite item {item_name!r} {detail}")
        items = matches
    statuses = [str(item.get("status")) for item in items if isinstance(item, dict)]
    if len(statuses) != len(items):
        raise QueueChainError("every queue item must be a mapping with status")
    unknown = sorted(set(statuses) - KNOWN_STATUSES)
    if unknown:
        raise QueueChainError(f"queue contains unknown statuses: {unknown}")
    if any(status in TERMINAL_FAILURES for status in statuses):
        return "failed"
    if all(status == "complete" for status in statuses):
        if require_gate_pass and not all(item.get("gate_passed") is True for item in items):
            return "failed"
        return "complete"
    return "waiting"


def load_queue(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise QueueChainError(f"queue must be a mapping: {path}")
    return payload


def build_watch_command(
    *, python_bin: str, next_queue: Path, poll_seconds: float, queue_type: str = "layer"
) -> list[str]:
    if not float(poll_seconds).is_integer():
        raise QueueChainError(
            "--poll-seconds must be a whole number for the queue watcher"
        )
    if queue_type == "rl":
        return [
            python_bin,
            "scripts/nano_rl_queue.py",
            str(next_queue),
            "--run-until-empty",
            "--poll-seconds",
            str(int(poll_seconds)),
        ]
    if queue_type == "ar_hpo":
        return [
            python_bin,
            "scripts/nano_ar_hpo_queue.py",
            str(next_queue),
            "--run-until-empty",
            "--poll-seconds",
            str(int(poll_seconds)),
        ]
    if queue_type == "av_probe":
        return [
            python_bin,
            "scripts/nano_av_probe_queue.py",
            str(next_queue),
            "--run-until-empty",
        ]
    if queue_type == "roundtrip":
        return [
            python_bin,
            "scripts/nano_roundtrip_queue.py",
            "run-loop",
            str(next_queue),
            "--sleep-seconds",
            str(int(poll_seconds)),
        ]
    if queue_type == "prefix_dataset":
        return [
            python_bin,
            "scripts/nano_prefix_dataset_queue.py",
            str(next_queue),
            "--run-until-empty",
            "--poll-seconds",
            str(int(poll_seconds)),
        ]
    if queue_type != "layer":
        raise QueueChainError(f"unknown queue type: {queue_type}")
    return [python_bin, "scripts/nano_ar_layer_sweep.py", "watch", str(next_queue), "--run-until-empty", "--poll-seconds", str(int(poll_seconds))]


def queue_type(queue: dict[str, Any]) -> str:
    schema = str(queue.get("schema_version") or "")
    if schema == "nano_rl_queue.v1":
        return "rl"
    if schema == "nano_ar_hpo_queue.v1":
        return "ar_hpo"
    if schema == "nano_av_probe_queue.v1":
        return "av_probe"
    if schema == "nano_roundtrip_queue.v1":
        return "roundtrip"
    if schema == "nano_prefix_dataset_queue.v1":
        return "prefix_dataset"
    return "layer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prerequisite-queue", type=Path, required=True)
    parser.add_argument(
        "--prerequisite-item",
        help="Wait for one named item instead of requiring every historical item to complete.",
    )
    parser.add_argument(
        "--require-gate-pass",
        action="store_true",
        help="Require each selected completed item to record gate_passed: true.",
    )
    parser.add_argument("--next-queue", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    previous: str | None = None
    while True:
        state = queue_state(
            load_queue(args.prerequisite_queue),
            item_name=args.prerequisite_item,
            require_gate_pass=args.require_gate_pass,
        )
        if state != previous:
            print(json.dumps({
                "prerequisite_queue": str(args.prerequisite_queue),
                "state": state,
            }, sort_keys=True), flush=True)
            previous = state
        if state == "failed":
            return 2
        if state == "complete":
            break
        time.sleep(args.poll_seconds)

    next_queue_doc = load_queue(args.next_queue)
    command = build_watch_command(
        python_bin=args.python_bin,
        next_queue=args.next_queue,
        poll_seconds=args.poll_seconds,
        queue_type=queue_type(next_queue_doc),
    )
    print(json.dumps({"launching": command}, sort_keys=True), flush=True)
    return subprocess.run(
        command,
        cwd=args.code_root,
        env=chain_environment(args.code_root),
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
