#!/usr/bin/env python3
"""Run Observatory phases or config-driven scripts with durable fail-closed state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .common import config_fingerprint


SCHEMA_VERSION = "nano_viz_queue_state.v1"
VALID_PHASES = {
    "canary-av",
    "canary-ar",
    "token-logprobs",
    "lattice-pilot",
    "lattice-full",
    "alternate-tellings",
    "functional-pilot",
    "functional-full",
    "trace-extract",
    "trace-describe",
    "poetry-prepare",
    "poetry-extract",
    "poetry-describe",
    "poetry-score",
    "poetry-reconstruct",
    "poetry-intervene",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def queue_environment(
    code_root: Path, overrides: dict[str, Any] | None = None
) -> dict[str, str]:
    python_paths = [
        str(code_root / "scripts"),
        str(code_root / "external" / "natural_language_autoencoders"),
        str(code_root),
    ]
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        python_paths.append(inherited)
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(python_paths),
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "offline",
    }
    for key, value in (overrides or {}).items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
            raise ValueError("queue environment overrides must be scalar string-keyed values")
        environment[key] = str(value)
    return environment


def build_command(
    *, python_bin: str, code_root: Path, config_path: Path, phase: str
) -> list[str]:
    if phase not in VALID_PHASES:
        raise ValueError(f"unsupported Observatory phase: {phase}")
    return [
        python_bin,
        "-m",
        "observatory.run_model_batches",
        "--config",
        str(config_path),
        "--phase",
        phase,
    ]


def build_script_command(
    *, python_bin: str, code_root: Path, config_path: Path, script: str
) -> list[str]:
    relative = Path(script)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
        raise ValueError(f"unsafe Observatory queue script: {script}")
    target = code_root / relative
    return [python_bin, str(target), "--config", str(config_path)]


def build_module_command(
    *, python_bin: str, config_path: Path, module: str
) -> list[str]:
    parts = module.split(".")
    if (
        not module.startswith("observatory.")
        or any(not part.isidentifier() for part in parts)
    ):
        raise ValueError(f"unsafe Observatory queue module: {module}")
    return [python_bin, "-m", module, "--config", str(config_path)]


def build_item_command(
    *,
    item: dict[str, Any],
    python_bin: str,
    code_root: Path,
    config_path: Path,
) -> tuple[list[str], dict[str, str]]:
    phase = item.get("phase")
    script = item.get("script")
    module = item.get("module")
    if sum(value is not None for value in (phase, script, module)) != 1:
        raise ValueError(
            "queue item must define exactly one of phase, script, or module"
        )
    if phase is not None:
        phase_name = str(phase)
        return (
            build_command(
                python_bin=python_bin,
                code_root=code_root,
                config_path=config_path,
                phase=phase_name,
            ),
            {"kind": "phase", "phase": phase_name},
        )
    if module is not None:
        module_name = str(module)
        return (
            build_module_command(
                python_bin=python_bin,
                config_path=config_path,
                module=module_name,
            ),
            {"kind": "module", "module": module_name},
        )
    script_name = str(script)
    return (
        build_script_command(
            python_bin=python_bin,
            code_root=code_root,
            config_path=config_path,
            script=script_name,
        ),
        {"kind": "script", "script": script_name},
    )


def run_queue(queue_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(queue_path.read_text())
    defaults = document.get("defaults") or {}
    code_root = Path(defaults["code_root"])
    config_path = Path(defaults["config"])
    config_sha256 = config_fingerprint(yaml.safe_load(config_path.read_text()))
    python_bin = str(defaults.get("python") or sys.executable)
    state_path = Path(defaults["state_json"])
    log_dir = Path(defaults["log_dir"])
    items = document.get("items") or []
    names = [str(item["name"]) for item in items]
    if not names or len(names) != len(set(names)):
        raise ValueError("queue item names must be non-empty and unique")
    state = (
        json.loads(state_path.read_text())
        if state_path.is_file()
        else {
            "schema_version": SCHEMA_VERSION,
            "queue": str(queue_path),
            "created_at": utc_now(),
            "items": {},
        }
    )
    state.update(
        status="running",
        queue=str(queue_path),
        config_sha256=config_sha256,
        updated_at=utc_now(),
    )
    write_state(state_path, state)
    log_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        name = str(item["name"])
        current = state["items"].get(name) or {}
        if (
            current.get("status") == "complete"
            and current.get("config_sha256") == config_sha256
        ):
            continue
        command, descriptor = build_item_command(
            item=item,
            python_bin=python_bin,
            code_root=code_root,
            config_path=config_path,
        )
        log_path = log_dir / f"{name}.log"
        state["items"][name] = {
            **descriptor,
            "status": "running",
            "started_at": utc_now(),
            "command": command,
            "log": str(log_path),
            "config_sha256": config_sha256,
        }
        state["updated_at"] = utc_now()
        write_state(state_path, state)
        with log_path.open("a") as log:
            completed = subprocess.run(
                command,
                cwd=code_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=queue_environment(code_root, defaults.get("environment")),
            )
        state["items"][name].update(
            {
                "status": "complete" if completed.returncode == 0 else "failed",
                "finished_at": utc_now(),
                "returncode": completed.returncode,
            }
        )
        state["updated_at"] = utc_now()
        write_state(state_path, state)
        if completed.returncode != 0:
            state["status"] = "failed"
            write_state(state_path, state)
            return state
    state["status"] = "complete"
    state["updated_at"] = utc_now()
    write_state(state_path, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        state = run_queue(args.queue)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
