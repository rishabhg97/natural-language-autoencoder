#!/usr/bin/env python3
"""Prune Nano Miles/FSDP checkpoint roots to the minimum resume set."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


ITER_RE = re.compile(r"^iter_(\d{7})$")
ROLLOUT_RE = re.compile(r"^global_dataset_state_dict_(\d+)\.pt$")


@dataclass(frozen=True)
class PrunePlan:
    latest_iteration: int
    keep_iterations: tuple[int, ...]
    remove_iter_dirs: tuple[Path, ...]
    remove_rollout_files: tuple[Path, ...]

    def as_jsonable(self) -> dict[str, object]:
        return {
            "latest_iteration": self.latest_iteration,
            "keep_iterations": list(self.keep_iterations),
            "remove_iter_dirs": [str(path) for path in self.remove_iter_dirs],
            "remove_rollout_files": [str(path) for path in self.remove_rollout_files],
        }


def iter_number(path: Path) -> int | None:
    match = ITER_RE.match(path.name)
    return int(match.group(1)) if match else None


def rollout_number(path: Path) -> int | None:
    match = ROLLOUT_RE.match(path.name)
    return int(match.group(1)) if match else None


def read_latest_iteration(checkpoint_root: Path) -> int:
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise FileNotFoundError(f"latest checkpoint tracker not found: {tracker}")
    return int(tracker.read_text().strip())


def build_prune_plan(checkpoint_root: Path, *, keep_full: int = 1) -> PrunePlan:
    if keep_full < 1:
        raise ValueError("keep_full must be >= 1")
    latest = read_latest_iteration(checkpoint_root)
    completed = sorted(
        n
        for path in checkpoint_root.glob("iter_*")
        if path.is_dir() and (n := iter_number(path)) is not None and n <= latest
    )
    keep_iterations = tuple(completed[-keep_full:])
    keep_set = set(keep_iterations)
    remove_iter_dirs = tuple(
        path
        for path in sorted(checkpoint_root.glob("iter_*"))
        if path.is_dir()
        and (n := iter_number(path)) is not None
        and n <= latest
        and n not in keep_set
    )

    min_kept_rollout = (min(keep_iterations) - 1) if keep_iterations else latest - 1
    rollout_dir = checkpoint_root / "rollout"
    remove_rollout_files: tuple[Path, ...] = ()
    if rollout_dir.is_dir():
        remove_rollout_files = tuple(
            path
            for path in sorted(rollout_dir.glob("global_dataset_state_dict_*.pt"))
            if (n := rollout_number(path)) is not None and n < min_kept_rollout
        )

    return PrunePlan(
        latest_iteration=latest,
        keep_iterations=keep_iterations,
        remove_iter_dirs=remove_iter_dirs,
        remove_rollout_files=remove_rollout_files,
    )


def apply_prune_plan(plan: PrunePlan, *, dry_run: bool) -> None:
    if dry_run:
        return
    for path in plan.remove_iter_dirs:
        shutil.rmtree(path)
    for path in plan.remove_rollout_files:
        path.unlink(missing_ok=True)


def run_once(checkpoint_root: Path, *, keep_full: int, dry_run: bool) -> PrunePlan:
    plan = build_prune_plan(checkpoint_root, keep_full=keep_full)
    apply_prune_plan(plan, dry_run=dry_run)
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--keep-full", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    while True:
        plan = run_once(args.checkpoint_root, keep_full=args.keep_full, dry_run=args.dry_run)
        print(json.dumps(plan.as_jsonable(), indent=2, sort_keys=True), flush=True)
        if not args.monitor:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
