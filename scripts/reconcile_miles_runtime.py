#!/usr/bin/env python3
"""Reconcile an existing patched Miles runtime with Nano's required hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_source_provenance import fingerprint_directory  # noqa: E402


ACTOR_RELATIVE = Path("miles/backends/fsdp_utils/actor.py")
ARGUMENTS_RELATIVE = Path("miles/backends/fsdp_utils/arguments.py")
LEGACY_LOCAL_NORM_IMPORT = "from nla.audit_runtime import clip_grad_norm_local_shards"
AUDIT_RUNTIME_IMPORT = (
    "from nla.audit_runtime import aggregate_train_losses_by_key, clip_grad_norm_local_shards"
)
LOCAL_NORM_FIELD = "nla_local_grad_norm: bool = True"
LEGACY_LOSS_AGGREGATION = "aggregate_train_losses(losses_reduced, self.parallel_state)"
KEYED_LOSS_AGGREGATION = "aggregate_train_losses_by_key(losses_reduced, self.parallel_state)"


class ReconcileError(RuntimeError):
    """Raised when a runtime does not match a supported reconciliation state."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ReconcileError(f"expected exactly one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def _reconcile_actor(text: str) -> str:
    updated = text
    if AUDIT_RUNTIME_IMPORT not in updated:
        if LEGACY_LOCAL_NORM_IMPORT in updated:
            updated = updated.replace(LEGACY_LOCAL_NORM_IMPORT, AUDIT_RUNTIME_IMPORT, 1)
        else:
            updated = _replace_once(
                updated,
                "import torch.distributed as dist\n",
                "import torch.distributed as dist\n" + AUDIT_RUNTIME_IMPORT + "\n",
                label="torch.distributed import",
            )

    if "clip_grad_norm_local_shards(" not in updated:
        start_marker = '                else:\n                    nla_timing_clip_start = self._nla_timing_start()\n'
        end_marker = "\n                nla_timing_optimizer_start = self._nla_timing_start()"
        start = updated.find(start_marker)
        end = updated.find(end_marker, start + len(start_marker)) if start >= 0 else -1
        if start < 0 or end < 0:
            start_marker = '        else:\n            nla_timing_clip_start = self._nla_timing_start()\n'
            end_marker = "\n        nla_timing_optimizer_start = self._nla_timing_start()"
            start = updated.find(start_marker)
            end = updated.find(end_marker, start + len(start_marker)) if start >= 0 else -1
        if start < 0 or end < 0:
            raise ReconcileError("could not locate the supported FSDP gradient-norm block")

        indent = start_marker.split("else:", 1)[0]
        inner = indent + "    "
        nested = inner + "    "
        replacement = (
            f"{indent}else:\n"
            f"{inner}nla_timing_clip_start = self._nla_timing_start()\n"
            f"{inner}if getattr(self.args, \"nla_local_grad_norm\", True):\n"
            f"{nested}grad_norm = clip_grad_norm_local_shards(\n"
            f"{nested}    self.model.parameters(),\n"
            f"{nested}    self.args.clip_grad,\n"
            f"{nested}    process_group=self.parallel_state.dp_group,\n"
            f"{nested})\n"
            f"{nested}grad_norm = float(grad_norm.item())\n"
            f"{nested}self._nla_timing_log(\n"
            f"{nested}    \"nla_timing_clip_grad_norm_local_shards\",\n"
            f"{nested}    nla_timing_clip_start,\n"
            f"{nested}    rollout_id=rollout_id,\n"
            f"{nested}    step_id=step_id,\n"
            f"{nested}    grad_norm=grad_norm,\n"
            f"{nested})\n"
            f"{inner}else:\n"
            f"{nested}grad_norm = torch.nn.utils.clip_grad_norm_(\n"
            f"{nested}    self.model.parameters(), self.args.clip_grad\n"
            f"{nested})\n"
            f"{nested}grad_norm = grad_norm.full_tensor().item()\n"
        )
        updated = updated[:start] + replacement + updated[end:]

    if KEYED_LOSS_AGGREGATION not in updated:
        updated = _replace_once(
            updated,
            LEGACY_LOSS_AGGREGATION,
            KEYED_LOSS_AGGREGATION,
            label="train-loss aggregation call",
        )
    return updated


def _reconcile_arguments(text: str) -> str:
    if LOCAL_NORM_FIELD in text:
        return text
    return _replace_once(
        text,
        "    nla_skip_grad_norm: bool = False\n",
        "    nla_skip_grad_norm: bool = False\n    " + LOCAL_NORM_FIELD + "\n",
        label="nla_skip_grad_norm field",
    )


def reconcile_miles_runtime(
    miles_root: str | Path,
    *,
    apply: bool,
    backup_dir: str | Path | None = None,
    miles_patches_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(miles_root).resolve()
    paths = {
        "actor": root / ACTOR_RELATIVE,
        "arguments": root / ARGUMENTS_RELATIVE,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Miles {name} runtime file not found: {path}")

    original = {name: path.read_text() for name, path in paths.items()}
    reconciled = {
        "actor": _reconcile_actor(original["actor"]),
        "arguments": _reconcile_arguments(original["arguments"]),
    }
    changed = [name for name in paths if original[name] != reconciled[name]]
    if apply and changed:
        backup_root = Path(backup_dir).resolve() if backup_dir is not None else None
        for name in changed:
            path = paths[name]
            if backup_root is not None:
                backup_path = backup_root / path.relative_to(root)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_path)
            path.write_text(reconciled[name])

    effective = {
        name: (reconciled[name] if apply else original[name])
        for name in paths
    }
    passed = (
        AUDIT_RUNTIME_IMPORT in effective["actor"]
        and "clip_grad_norm_local_shards(" in effective["actor"]
        and KEYED_LOSS_AGGREGATION in effective["actor"]
        and LOCAL_NORM_FIELD in effective["arguments"]
    )
    miles_tree = fingerprint_directory(root, label="miles")
    patches_tree = (
        fingerprint_directory(miles_patches_root, label="miles_patches")
        if miles_patches_root is not None
        else None
    )
    runtime_material = {
        "miles": miles_tree,
        "miles_patches": patches_tree,
    }
    return {
        "schema_version": "nano_miles_runtime_reconcile.v1",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "miles_root": str(root),
        "apply_requested": bool(apply),
        "passed": passed,
        "miles_tree": miles_tree,
        "miles_patches_tree": patches_tree,
        "runtime_sha256": _sha256(
            json.dumps(runtime_material, sort_keys=True, separators=(",", ":")).encode()
        ),
        "changed_files": [str(paths[name]) for name in changed] if apply else [],
        "planned_files": [str(paths[name]) for name in changed],
        "files": {
            name: {
                "path": str(paths[name]),
                "sha256": _sha256(effective[name].encode()),
                "bytes": len(effective[name].encode()),
            }
            for name in paths
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miles-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--miles-patches-root", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    report = reconcile_miles_runtime(
        args.miles_root,
        apply=args.apply,
        backup_dir=args.backup_dir,
        miles_patches_root=args.miles_patches_root,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
