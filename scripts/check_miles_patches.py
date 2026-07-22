#!/usr/bin/env python3
"""Validate and optionally apply Nano's Miles patch stack."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATCH_DIR = ROOT / "external" / "natural_language_autoencoders" / "nla" / "miles_patches"
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _count_hunk(lines: list[str], start_index: int) -> tuple[int, int, int]:
    old_count = 0
    new_count = 0
    index = start_index + 1
    while index < len(lines) and not lines[index].startswith("@@") and not lines[index].startswith("diff --git "):
        line = lines[index]
        if line.startswith("\\ No newline"):
            index += 1
            continue
        if line.startswith((" ", "-")):
            old_count += 1
        if line.startswith((" ", "+")):
            new_count += 1
        index += 1
    return old_count, new_count, index


def check_hunk_counts(patch_dir: str | Path = DEFAULT_PATCH_DIR) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for patch_path in sorted(Path(patch_dir).glob("*.patch")):
        lines = patch_path.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            if not line.startswith("@@"):
                continue
            match = HUNK_RE.match(line)
            if match is None:
                failures.append({"patch": str(patch_path), "line": index + 1, "error": "malformed hunk header"})
                continue
            expected_old = int(match.group(2) or "1")
            expected_new = int(match.group(4) or "1")
            actual_old, actual_new, _ = _count_hunk(lines, index)
            if (actual_old, actual_new) != (expected_old, expected_new):
                failures.append(
                    {
                        "patch": str(patch_path),
                        "line": index + 1,
                        "error": "hunk line count mismatch",
                        "expected_old": expected_old,
                        "expected_new": expected_new,
                        "actual_old": actual_old,
                        "actual_new": actual_new,
                    }
                )
    return failures


def apply_patches_to_checkout(
    *,
    miles_root: str | Path,
    patch_dir: str | Path = DEFAULT_PATCH_DIR,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(miles_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Miles checkout not found: {source}")
    parent = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="nano_miles_patch_check_"))
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / "miles"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))

    applied: list[str] = []
    for patch_path in sorted(Path(patch_dir).glob("*.patch")):
        subprocess.run(
            ["git", "-C", str(destination), "apply", "--recount", "--whitespace=nowarn", str(patch_path.resolve())],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        applied.append(str(patch_path))
    return {"worktree": str(destination), "applied": applied}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-dir", type=Path, default=DEFAULT_PATCH_DIR)
    parser.add_argument("--miles-root", type=Path, help="Optional pinned Miles checkout to copy and patch.")
    parser.add_argument("--work-dir", type=Path, help="Optional parent for the copied Miles checkout.")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "patch_dir": str(args.patch_dir),
        "hunk_failures": check_hunk_counts(args.patch_dir),
    }
    if report["hunk_failures"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    if args.miles_root is not None:
        report["apply"] = apply_patches_to_checkout(
            miles_root=args.miles_root,
            patch_dir=args.patch_dir,
            work_dir=args.work_dir,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
