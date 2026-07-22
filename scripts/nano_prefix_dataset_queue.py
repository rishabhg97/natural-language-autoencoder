#!/usr/bin/env python3
"""Sequential queue for config-driven Nano prefix dataset and critic jobs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml


VALID_STATUSES = {"pending", "running", "complete", "failed", "blocked"}
DRY_RUN_ENV_KEYS = ("PYTHONPATH", "WANDB_MODE")


class PrefixQueueError(ValueError):
    """Raised when a prefix queue is malformed or a job fails."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_queue(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text())
    if not isinstance(payload, dict):
        raise PrefixQueueError(f"queue must be a mapping: {source}")
    if payload.get("schema_version") != "nano_prefix_dataset_queue.v1":
        raise PrefixQueueError(
            "schema_version must be nano_prefix_dataset_queue.v1"
        )
    defaults = payload.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise PrefixQueueError("defaults must be a mapping")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise PrefixQueueError("items must be a non-empty list")
    names: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise PrefixQueueError(f"item {index} must be a mapping")
        item.setdefault("status", "pending")
        if item["status"] not in VALID_STATUSES:
            raise PrefixQueueError(
                f"item {index} has invalid status {item['status']!r}"
            )
        for field in ("name", "config"):
            if not item.get(field):
                raise PrefixQueueError(f"item {index} requires {field}")
        name = str(item["name"])
        if name in names:
            raise PrefixQueueError(f"duplicate item name: {name}")
        names.add(name)
        expected = item.get("expected_artifacts") or []
        if not isinstance(expected, list) or not expected:
            raise PrefixQueueError(
                f"item {index} requires non-empty expected_artifacts"
            )
    return payload


def write_queue(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    temporary.replace(destination)


def _resolve_path(value: str | Path, *, queue_path: Path, code_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    code_candidate = code_root / path
    if code_candidate.exists():
        return code_candidate
    return queue_path.parent / path


def _canonical_json_sha256(path: Path, ignored_paths: list[str]) -> str:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PrefixQueueError(f"artifact is not readable JSON: {path}") from exc
    for dotted_path in ignored_paths:
        parts = dotted_path.split(".")
        if not parts or any(not part for part in parts):
            raise PrefixQueueError(
                f"invalid ignored JSON path {dotted_path!r} for {path}"
            )
        parent = payload
        for part in parts[:-1]:
            if not isinstance(parent, dict) or part not in parent:
                raise PrefixQueueError(
                    f"ignored JSON path {dotted_path!r} is absent from {path}"
                )
            parent = parent[part]
        if not isinstance(parent, dict) or parts[-1] not in parent:
            raise PrefixQueueError(
                f"ignored JSON path {dotted_path!r} is absent from {path}"
            )
        del parent[parts[-1]]
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _code_root(defaults: dict[str, Any]) -> Path:
    override = os.environ.get("NANO_QUEUE_CODE_ROOT")
    value = override or defaults.get("code_root")
    if not value:
        raise PrefixQueueError("defaults.code_root is required")
    return Path(value).resolve()


def build_run_spec(
    queue_doc: dict[str, Any],
    item: dict[str, Any],
    *,
    queue_path: str | Path,
) -> dict[str, Any]:
    queue_path = Path(queue_path)
    defaults = queue_doc.get("defaults") or {}
    code_root = _code_root(defaults)
    python_bin = str(defaults.get("python") or sys.executable)
    config_path = _resolve_path(
        item["config"],
        queue_path=queue_path,
        code_root=code_root,
    )
    if not config_path.is_file():
        raise PrefixQueueError(f"prefix config is missing: {config_path}")
    run_dir = _resolve_path(
        item.get("run_dir", queue_path.parent / str(item["name"])),
        queue_path=queue_path,
        code_root=code_root,
    )
    report_json = _resolve_path(
        item.get("report_json", run_dir / "launch_report.json"),
        queue_path=queue_path,
        code_root=code_root,
    )
    log_path = _resolve_path(
        item.get("log_path", run_dir / "run.log"),
        queue_path=queue_path,
        code_root=code_root,
    )
    expected = []
    for value in item["expected_artifacts"]:
        raw = {"path": value} if isinstance(value, (str, Path)) else dict(value)
        if not raw.get("path"):
            raise PrefixQueueError("expected_artifacts entries require path")
        sha_value = raw.get("sha256")
        expected_sha = None if sha_value is None else str(sha_value).lower()
        if expected_sha is not None and re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
            raise PrefixQueueError(
                "expected_artifacts sha256 must be a 64-character hexadecimal digest"
            )
        canonical_value = raw.get("canonical_json_sha256")
        canonical_sha = (
            None if canonical_value is None else str(canonical_value).lower()
        )
        if canonical_sha is not None and re.fullmatch(
            r"[0-9a-f]{64}", canonical_sha
        ) is None:
            raise PrefixQueueError(
                "expected_artifacts canonical_json_sha256 must be a "
                "64-character hexadecimal digest"
            )
        ignored_paths = raw.get("ignore_json_paths") or []
        if not isinstance(ignored_paths, list) or not all(
            isinstance(value, str) and value for value in ignored_paths
        ):
            raise PrefixQueueError(
                "expected_artifacts ignore_json_paths must be a list of strings"
            )
        if ignored_paths and canonical_sha is None:
            raise PrefixQueueError(
                "ignore_json_paths requires canonical_json_sha256"
            )
        expected.append(
            {
                "path": str(
                    _resolve_path(
                        raw["path"],
                        queue_path=queue_path,
                        code_root=code_root,
                    )
                ),
                "sha256": expected_sha,
                "canonical_json_sha256": canonical_sha,
                "ignore_json_paths": ignored_paths,
            }
        )
    command = [
        python_bin,
        str(code_root / "scripts" / "nano_prefix_dataset_config.py"),
        str(config_path),
        "--run",
        "--report-json",
        str(report_json),
    ]
    env = os.environ.copy()
    env.update(
        {
            str(key): str(value)
            for key, value in (defaults.get("environment") or {}).items()
        }
    )
    env["WANDB_MODE"] = "offline"
    env["PYTHONPATH"] = ":".join(
        [
            str(code_root / "external" / "natural_language_autoencoders"),
            str(code_root),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(":")
    return {
        "item_name": str(item["name"]),
        "code_root": str(code_root),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "report_json": str(report_json),
        "log_path": str(log_path),
        "expected_artifacts": expected,
        "command": command,
        "env": env,
    }


def public_run_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a dry-run payload without inherited process credentials."""
    public = {key: value for key, value in spec.items() if key != "env"}
    environment = spec.get("env") or {}
    public["environment"] = {
        key: environment[key]
        for key in DRY_RUN_ENV_KEYS
        if key in environment
    }
    return public


def _run_logged(spec: dict[str, Any]) -> None:
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(spec["log_path"])
    with log_path.open("a") as log_handle:
        completed = subprocess.run(
            spec["command"],
            cwd=spec["code_root"],
            env=spec["env"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise PrefixQueueError(
            f"prefix job exited {completed.returncode}; see {log_path}"
        )


def process_next(queue_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    queue_path = Path(queue_path)
    queue_doc = load_queue(queue_path)
    index = next(
        (
            index
            for index, item in enumerate(queue_doc["items"])
            if item["status"] == "pending"
        ),
        None,
    )
    if index is None:
        return {"status": "empty"}
    item = queue_doc["items"][index]
    spec = build_run_spec(queue_doc, item, queue_path=queue_path)
    if dry_run:
        return {"status": "dry_run", **public_run_spec(spec)}

    item.update(
        status="running",
        started_at=utc_now(),
        run_dir=spec["run_dir"],
        log_path=spec["log_path"],
        report_json=spec["report_json"],
    )
    write_queue(queue_path, queue_doc)
    try:
        _run_logged(spec)
        missing = [
            artifact["path"]
            for artifact in spec["expected_artifacts"]
            if not Path(artifact["path"]).exists()
        ]
        if missing:
            raise PrefixQueueError(
                f"prefix job completed with missing artifacts: {missing}"
            )
        mismatched = []
        for artifact in spec["expected_artifacts"]:
            expected_sha = artifact["sha256"]
            path = Path(artifact["path"])
            if expected_sha is not None:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                digest = None
            if expected_sha is not None and digest != expected_sha:
                mismatched.append(
                    {
                        "path": artifact["path"],
                        "mode": "file_sha256",
                        "expected": expected_sha,
                        "actual": digest,
                    }
                )
            canonical_sha = artifact["canonical_json_sha256"]
            if canonical_sha is not None:
                canonical_digest = _canonical_json_sha256(
                    path,
                    artifact["ignore_json_paths"],
                )
                if canonical_digest != canonical_sha:
                    mismatched.append(
                        {
                            "path": artifact["path"],
                            "mode": "canonical_json_sha256",
                            "ignored_paths": artifact["ignore_json_paths"],
                            "expected": canonical_sha,
                            "actual": canonical_digest,
                        }
                    )
        if mismatched:
            raise PrefixQueueError(
                f"prefix job artifact sha256 mismatch: {mismatched}"
            )
        queue_doc = load_queue(queue_path)
        queue_doc["items"][index].update(
            status="complete",
            gate_passed=True,
            completed_at=utc_now(),
        )
        write_queue(queue_path, queue_doc)
        return {"status": "complete", "item_name": item["name"]}
    except BaseException as exc:
        queue_doc = load_queue(queue_path)
        queue_doc["items"][index].update(
            status="failed",
            gate_passed=False,
            failed_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        write_queue(queue_path, queue_doc)
        raise


def queue_status(queue_path: str | Path) -> dict[str, Any]:
    queue_doc = load_queue(queue_path)
    counts: dict[str, int] = {}
    for item in queue_doc["items"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"counts": counts, "items": queue_doc["items"]}


def reset_active(queue_path: str | Path) -> dict[str, Any]:
    queue_doc = load_queue(queue_path)
    reset = []
    for item in queue_doc["items"]:
        if item["status"] != "running":
            continue
        item.update(
            status="pending",
            reset_at=utc_now(),
            reset_reason="manual stale-active reset",
        )
        reset.append(str(item["name"]))
    write_queue(queue_path, queue_doc)
    return {"reset": reset}


def queue_lock(queue_path: Path):
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    handle = lock_path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--reset-active", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-until-empty", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.status:
        print(json.dumps(queue_status(args.queue), indent=2, sort_keys=True))
        return 0
    if args.reset_active:
        print(json.dumps(reset_active(args.queue), sort_keys=True))
        return 0
    with queue_lock(args.queue):
        while True:
            result = process_next(args.queue, dry_run=args.dry_run)
            print(json.dumps(result, default=str, sort_keys=True), flush=True)
            if args.dry_run or result["status"] == "empty":
                return 0
            if not args.run_until_empty:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
