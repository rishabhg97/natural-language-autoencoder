#!/usr/bin/env python3
"""Start a durable Observatory queue only after an upstream state passes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "nano_viz_queue_watcher.v1"
TERMINAL_FAILURES = {"failed", "blocked", "cancelled", "canceled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def upstream_outcome(value: dict[str, Any]) -> str:
    status = str(value.get("status") or "").lower()
    if status == "complete":
        return "ready"
    if status in TERMINAL_FAILURES:
        return "failed"
    return "waiting"


def run(
    *,
    upstream_state: Path,
    queue: Path,
    code_root: Path,
    python_bin: str,
    state_path: Path,
    log_path: Path,
    poll_seconds: float,
) -> int:
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "waiting",
        "upstream_state": str(upstream_state),
        "queue": str(queue),
        "created_at": utc_now(),
    }
    write_state(state_path, state)
    while True:
        if upstream_state.is_file():
            try:
                upstream = json.loads(upstream_state.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                upstream = {}
            outcome = upstream_outcome(upstream)
            state.update(
                upstream_status=upstream.get("status"),
                upstream_stage=upstream.get("stage"),
                updated_at=utc_now(),
            )
            write_state(state_path, state)
            if outcome == "failed":
                state.update(status="blocked", error="upstream queue failed", updated_at=utc_now())
                write_state(state_path, state)
                return 1
            if outcome == "ready":
                break
        time.sleep(poll_seconds)

    command = [
        python_bin,
        "-m",
        "observatory.queue",
        "--queue",
        str(queue),
    ]
    state.update(status="running", command=command, started_at=utc_now(), updated_at=utc_now())
    write_state(state_path, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        completed = subprocess.run(
            command,
            cwd=code_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(code_root)},
        )
    state.update(
        status="complete" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        finished_at=utc_now(),
        updated_at=utc_now(),
    )
    write_state(state_path, state)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-state", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    return run(
        upstream_state=args.upstream_state,
        queue=args.queue,
        code_root=args.code_root,
        python_bin=args.python,
        state_path=args.state_json,
        log_path=args.log,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
