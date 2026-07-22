#!/usr/bin/env python3
"""Plan and optionally apply explicit, manifest-first checkpoint retention."""

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union


SCHEMA_VERSION = "nano_checkpoint_retention.v1"


class RetentionError(ValueError):
    """Raised when a retention policy could escape its declared output root."""


@dataclass
class RetentionPolicy:
    output_root: Path
    protected: set[Path] = field(default_factory=set)
    current_best: Optional[Path] = None
    keep_challenger: Optional[Path] = None


@dataclass
class CleanupPlan:
    output_root: Path
    candidates: list[Path]
    keep: list[Path]
    delete: list[Path]


def _validate_root(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise RetentionError(f"output_root must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise RetentionError(f"output_root must be an existing directory: {path}")
    return resolved


def _validate_member(path: Path, *, root: Path, label: str) -> Path:
    original = Path(path)
    if original.is_symlink():
        raise RetentionError(f"{label} must not be a symlink: {original}")
    resolved = original.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise RetentionError(f"{label} is outside output_root: {original}")
    if not resolved.is_dir():
        raise RetentionError(f"{label} must be an existing directory: {original}")
    return resolved


def build_cleanup_plan(
    policy: RetentionPolicy,
    *,
    candidates: list[Path],
) -> CleanupPlan:
    root = _validate_root(policy.output_root)
    protected = {
        _validate_member(path, root=root, label="protected path")
        for path in policy.protected
    }
    if policy.current_best is not None:
        protected.add(
            _validate_member(policy.current_best, root=root, label="current_best")
        )
    if policy.keep_challenger is not None:
        protected.add(
            _validate_member(
                policy.keep_challenger,
                root=root,
                label="challenger",
            )
        )
    resolved_candidates = sorted(
        {
            _validate_member(path, root=root, label="candidate")
            for path in candidates
        },
        key=str,
    )
    keep = sorted(protected, key=str)
    delete = [path for path in resolved_candidates if path not in protected]
    return CleanupPlan(
        output_root=root,
        candidates=resolved_candidates,
        keep=keep,
        delete=delete,
    )


def _plan_payload(plan: CleanupPlan, *, apply: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "apply_requested": apply,
        "output_root": str(plan.output_root),
        "candidates": [str(path) for path in plan.candidates],
        "keep": [str(path) for path in plan.keep],
        "delete": [str(path) for path in plan.delete],
    }


def _write_manifest_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def execute_cleanup(
    plan: CleanupPlan,
    *,
    manifest_path: Union[str, Path],
    apply: bool = False,
    delete_function: Callable[[Path], None] = shutil.rmtree,
) -> dict[str, Any]:
    payload = _plan_payload(plan, apply=apply)
    manifest = Path(manifest_path)
    _write_manifest_atomic(manifest, payload)
    if not apply:
        return payload

    deleted: list[str] = []
    for path in plan.delete:
        if path.is_symlink():
            raise RetentionError(f"candidate became a symlink before deletion: {path}")
        current = path.resolve()
        if current == plan.output_root or not current.is_relative_to(plan.output_root):
            raise RetentionError(f"candidate escaped output_root before deletion: {path}")
        delete_function(path)
        deleted.append(str(path))
    payload["deleted"] = deleted
    _write_manifest_atomic(manifest, payload)
    return payload


def load_policy(path: Union[str, Path]) -> tuple[RetentionPolicy, list[Path]]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RetentionError("retention policy JSON must be an object")
    required = ("output_root", "protected", "candidates")
    missing = [name for name in required if name not in value]
    if missing:
        raise RetentionError(f"retention policy is missing keys: {missing}")
    policy = RetentionPolicy(
        output_root=Path(value["output_root"]),
        protected={Path(item) for item in value.get("protected", [])},
        current_best=Path(value["current_best"]) if value.get("current_best") else None,
        keep_challenger=Path(value["challenger"]) if value.get("challenger") else None,
    )
    return policy, [Path(item) for item in value["candidates"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_json", type=Path)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    policy, candidates = load_policy(args.policy_json)
    plan = build_cleanup_plan(policy, candidates=candidates)
    payload = execute_cleanup(
        plan,
        manifest_path=args.manifest_json,
        apply=args.apply,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
