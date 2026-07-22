#!/usr/bin/env python3
"""Apply the offline W&B ownership change to a Miles source checkout.

The checked-in unified patch documents the desired code review diff. This
installer is intentionally used for the live Miles checkout because it has a
minor, harmless source-context difference from the revision used to author
that patch. Each replacement is exact, idempotent, and fails before writing
anything when the checkout no longer matches the supported shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class PatchShapeError(RuntimeError):
    """Raised when a Miles source file is not a supported pre/postimage."""


def _replace_once_or_keep(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    matches = text.count(old)
    if matches != 1:
        raise PatchShapeError(f"{label}: expected one preimage, found {matches}")
    return text.replace(old, new, 1), True


def _secondary_body(text: str) -> tuple[str, str]:
    marker = "def init_wandb_secondary("
    index = text.find(marker)
    if index < 0:
        raise PatchShapeError("wandb_utils: init_wandb_secondary not found")
    return text[:index], text[index:]


def apply_offline_wandb_role_patch(miles_root: Path) -> dict[str, object]:
    """Apply the four offline role-ownership edits and return a summary."""

    paths = {
        "tracking": miles_root / "miles/utils/tracking_utils.py",
        "rollout": miles_root / "miles/ray/rollout.py",
        "actor": miles_root / "miles/backends/fsdp_utils/actor.py",
        "wandb": miles_root / "miles/utils/wandb_utils.py",
    }
    originals = {name: path.read_text() for name, path in paths.items()}
    updated = dict(originals)

    tracking, _ = _replace_once_or_keep(
        updated["tracking"],
        "def init_tracking(args, primary: bool = True, **kwargs):",
        "def init_tracking(args, primary: bool = True, role: str | None = None, **kwargs):",
        "tracking_utils signature",
    )
    updated["tracking"], _ = _replace_once_or_keep(
        tracking,
        "wandb_utils.init_wandb_secondary(args, **kwargs)",
        "wandb_utils.init_wandb_secondary(args, role=role, **kwargs)",
        "tracking_utils secondary call",
    )

    updated["rollout"], _ = _replace_once_or_keep(
        updated["rollout"],
        "init_tracking(args, primary=False, router_addr=",
        'init_tracking(args, primary=False, role="rollout", router_addr=',
        "rollout tracking role",
    )
    updated["actor"], _ = _replace_once_or_keep(
        updated["actor"],
        "init_tracking(args, primary=False)",
        "init_tracking(args, primary=False, role=role)",
        "actor tracking role",
    )

    wandb, _ = _replace_once_or_keep(
        updated["wandb"],
        '        "config": _compute_config_for_logging(args),\n    }',
        '        "config": _compute_config_for_logging(args),\n    }\n\n'
        '    if args.wandb_run_id:\n'
        '        init_kwargs["id"] = args.wandb_run_id\n'
        '        init_kwargs["resume"] = "allow"',
        "wandb primary identity",
    )
    prefix, secondary = _secondary_body(wandb)
    secondary, _ = _replace_once_or_keep(
        secondary,
        "def init_wandb_secondary(args, router_addr=None):",
        "def init_wandb_secondary(args, router_addr=None, role=None):",
        "wandb secondary signature",
    )
    secondary, _ = _replace_once_or_keep(
        secondary,
        "    offline = _is_offline_mode(args)",
        "    offline = _is_offline_mode(args)\n"
        '    offline_role = role or "secondary"\n'
        '    offline_run_id = f"{wandb_run_id}-{offline_role}" if offline else wandb_run_id',
        "wandb secondary offline identity",
    )
    secondary, _ = _replace_once_or_keep(
        secondary,
        '        "id": wandb_run_id,',
        '        "id": offline_run_id,',
        "wandb secondary id",
    )
    secondary, _ = _replace_once_or_keep(
        secondary,
        "    # Add custom directory if specified",
        "    if offline:\n"
        '        init_kwargs["group"] = args.wandb_group\n'
        '        init_kwargs["name"] = f"{args.wandb_group}-{offline_role}"\n\n'
        "    # Add custom directory if specified",
        "wandb secondary offline metadata",
    )
    updated["wandb"] = prefix + secondary

    changed_files = []
    for name, path in paths.items():
        if updated[name] != originals[name]:
            changed_files.append(str(path))

    for name, path in paths.items():
        if updated[name] != originals[name]:
            path.write_text(updated[name])

    return {"changed": len(changed_files), "files": changed_files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("miles_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(apply_offline_wandb_role_patch(args.miles_root), sort_keys=True))


if __name__ == "__main__":
    main()
