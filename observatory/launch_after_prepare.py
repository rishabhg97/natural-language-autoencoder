#!/usr/bin/env python3
"""Wait for AV preparation, validate its artifact, then start the GPU queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .model_runtime import hf_checkpoint_complete


def process_exists(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        try:
            state = stat_path.read_text().split()[2]
        except (OSError, IndexError):
            state = ""
        if state == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def validate_preparation(report_path: Path, av_hf: Path) -> dict:
    if not report_path.is_file():
        raise ValueError(f"runtime preparation report is missing: {report_path}")
    report = json.loads(report_path.read_text())
    if not report.get("passed"):
        raise ValueError("runtime preparation report did not pass")
    if (report.get("av_stage") or {}).get("status") not in {"converted", "reused"}:
        raise ValueError("runtime preparation did not complete AV conversion")
    if not hf_checkpoint_complete(av_hf):
        raise ValueError(f"temporary AV HF checkpoint is missing: {av_hf}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-pid", type=int, required=True)
    parser.add_argument("--prepare-report", type=Path, required=True)
    parser.add_argument("--av-hf", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    while process_exists(args.prepare_pid):
        time.sleep(args.poll_seconds)
    try:
        validate_preparation(args.prepare_report, args.av_hf)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    command = [
        sys.executable,
        "-m",
        "observatory.queue",
        "--queue",
        str(args.queue),
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
