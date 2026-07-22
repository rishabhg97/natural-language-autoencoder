#!/usr/bin/env python3
"""Lightweight YAML queue for Nano AR-SFT HPO probes."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nano_ar_hpo_study  # noqa: E402
import nano_av_runner  # noqa: E402
from nano_queue_status import ACTIVE_HPO_STATUSES, VALID_HPO_STATUSES, status_counts  # noqa: E402


VALID_STATUSES = VALID_HPO_STATUSES
ACTIVE_STATUSES = ACTIVE_HPO_STATUSES
MAX_AUTOMATED_EVAL_LIMIT = 512


class QueueError(ValueError):
    """Raised when an AR HPO queue manifest is unsafe or malformed."""


def load_queue(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict):
        raise QueueError(f"queue YAML must contain a mapping: {source}")
    return validate_queue(data, source=source)


def validate_queue(data: dict[str, Any], *, source: Path) -> dict[str, Any]:
    if data.get("schema_version") != "nano_ar_hpo_queue.v1":
        raise QueueError("schema_version must be nano_ar_hpo_queue.v1")
    defaults = data.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        raise QueueError("defaults must be a mapping")
    validation_limit = int(defaults.get("validation_limit", 512))
    test_limit = int(defaults.get("test_limit", 512))
    if validation_limit > MAX_AUTOMATED_EVAL_LIMIT:
        raise QueueError("validation_limit above 512 is not allowed in automated queue")
    if test_limit > MAX_AUTOMATED_EVAL_LIMIT:
        raise QueueError("test_limit above 512 is not allowed in automated queue")

    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise QueueError("items must be a non-empty list")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise QueueError(f"item {index} must be a mapping")
        item.setdefault("status", "pending")
        if item["status"] not in VALID_STATUSES:
            raise QueueError(f"item {index} has invalid status {item['status']!r}")
        if not item.get("name") or not item.get("config"):
            raise QueueError(f"item {index} requires name and config")
    return data


def next_pending_index(queue_doc: dict[str, Any]) -> int | None:
    for index, item in enumerate(queue_doc["items"]):
        if item.get("status") == "pending":
            return index
    return None


def write_queue(path: str | Path, queue_doc: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(queue_doc, sort_keys=False))
    tmp.replace(destination)


def update_item(path: str | Path, index: int, **fields: Any) -> dict[str, Any]:
    queue_doc = load_queue(path)
    item = queue_doc["items"][index]
    item.update({key: value for key, value in fields.items() if value is not None})
    write_queue(path, queue_doc)
    return item


def reset_active_items(path: str | Path, *, reason: str = "manual active reset") -> dict[str, Any]:
    queue_doc = load_queue(path)
    reset_items: list[str] = []
    reset_at = utc_now()
    for item in queue_doc["items"]:
        status = item.get("status")
        if status not in ACTIVE_STATUSES:
            continue
        item["previous_status"] = status
        item["status"] = "pending"
        item["reset_at"] = reset_at
        item["reset_reason"] = reason
        reset_items.append(str(item["name"]))
    if reset_items:
        write_queue(path, queue_doc)
    return {"queue": str(path), "reset_count": len(reset_items), "items": reset_items}


def _check_eval_limit(name: str, value: int) -> None:
    if value > MAX_AUTOMATED_EVAL_LIMIT:
        raise QueueError(f"{name} above 512 is not allowed in automated queue")


def build_eval_command(
    *,
    python_bin: str,
    checkpoint_dir: Path,
    train_parquet: Path,
    validation_parquet: Path,
    test_parquet: Path | None,
    report_json: Path,
    validation_limit: int,
    test_limit: int,
    eval_splits: list[str],
    batch_size: int,
    controls: list[str],
) -> list[str]:
    _check_eval_limit("validation_limit", validation_limit)
    _check_eval_limit("test_limit", test_limit)
    if not eval_splits or len(set(eval_splits)) != len(eval_splits) or not set(
        eval_splits
    ).issubset({"validation", "test"}):
        raise QueueError("eval_splits must be a non-empty unique validation/test list")
    command = [
        python_bin,
        "scripts/eval_nano_ar_miles_checkpoint.py",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--train-parquet",
        str(train_parquet),
        "--validation-parquet",
        str(validation_parquet),
        "--validation-limit",
        str(validation_limit),
        "--test-limit",
        str(test_limit),
        "--eval-splits",
        *[str(split) for split in eval_splits],
        "--batch-size",
        str(batch_size),
        "--controls",
        *controls,
        "--report-json",
        str(report_json),
    ]
    if "test" in eval_splits:
        if test_parquet is None:
            raise QueueError("test_parquet is required when eval_splits includes test")
        command.extend(["--test-parquet", str(test_parquet)])
    return command


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_path(value: str | Path, queue_path: Path, queue_doc: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    code_root = queue_doc.get("defaults", {}).get("code_root")
    if code_root:
        candidate = Path(code_root) / path
        if candidate.exists():
            return candidate
    candidate = queue_path.parent / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def expected_checkpoint_for_plan(plan: dict[str, Any]) -> Path:
    spec = plan["spec"]
    training = spec["training"]
    checkpoint = spec["checkpoint"]
    if checkpoint.get("finetune") and training.get("resume_steps") is not None:
        iteration = int(training["resume_steps"])
    else:
        iteration = int(plan["num_rollout"])
    return Path(plan["run_dir"]) / "checkpoints" / f"iter_{iteration:07d}"


def eval_paths_for_plan(plan: dict[str, Any]) -> tuple[Path, Path, Path]:
    train_parquet = Path(plan["train_parquet"])
    manifest = plan.get("split_manifest")
    if isinstance(manifest, dict):
        validation = Path(manifest["validation"]["path"])
        test = Path(manifest["test"]["path"])
    else:
        split_dir = train_parquet.parent
        validation = split_dir / "validation.parquet"
        test = split_dir / "test.parquet"
    return train_parquet, validation, test


def _with_python(command: list[str], python_bin: str) -> list[str]:
    if command and command[0] == "python":
        return [python_bin, *command[1:]]
    return command


def _run_logged(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n# started_utc={utc_now()}\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"# completed_utc={utc_now()}\n")


def _env_for_run(plan: dict[str, Any], code_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (plan.get("environment") or {}).items()})
    env["WANDB_MODE"] = "offline"
    pythonpath_parts = [
        str(code_root / "external" / "natural_language_autoencoders"),
        str(code_root / "external" / "natural_language_autoencoders" / "Miles"),
        str(code_root),
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return env


def _record_trial(
    *,
    queue_doc: dict[str, Any],
    item: dict[str, Any],
    config_path: Path,
    eval_report: Path,
    train_log: Path,
    run_dir: Path,
) -> None:
    study_jsonl = queue_doc.get("defaults", {}).get("study_jsonl")
    if not study_jsonl:
        return
    record = nano_ar_hpo_study.build_trial_record(
        trial_name=str(item["name"]),
        config_path=config_path,
        eval_report_path=eval_report,
        train_log_path=train_log,
        run_dir=run_dir,
        status="complete",
        notes=item.get("notes"),
        task="ar",
    )
    nano_ar_hpo_study.upsert_trial(Path(study_jsonl), record)


def _item_eval_controls(item: dict[str, Any], queue_doc: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    if item.get("controls"):
        return [str(value) for value in item["controls"]]
    defaults = queue_doc.get("defaults", {})
    if defaults.get("controls"):
        return [str(value) for value in defaults["controls"]]
    eval_spec = spec.get("eval") or {}
    return [str(value) for value in eval_spec.get("controls", [])]


def process_next_item(queue_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    queue_path = Path(queue_path)
    queue_doc = load_queue(queue_path)
    index = next_pending_index(queue_doc)
    if index is None:
        return {"status": "idle"}

    item = queue_doc["items"][index]
    defaults = queue_doc.get("defaults", {})
    python_bin = str(defaults.get("python") or sys.executable)
    config_path = resolve_path(item["config"], queue_path, queue_doc)
    spec = nano_av_runner.load_and_validate_spec(config_path)
    plan = nano_av_runner.prepare_run(spec, run_id=item.get("run_id") or item["name"])
    run_dir = Path(plan["run_dir"])
    code_root = Path(spec["paths"]["code_root"])
    expected_checkpoint = Path(item.get("expected_checkpoint") or expected_checkpoint_for_plan(plan))
    train_parquet, validation_parquet, test_parquet = eval_paths_for_plan(plan)
    eval_spec = spec.get("eval") or {}
    validation_limit = int(item.get("validation_limit") or defaults.get("validation_limit") or eval_spec.get("validation_limit") or 512)
    test_limit = int(item.get("test_limit") or defaults.get("test_limit") or eval_spec.get("test_limit") or 512)
    eval_splits = list(
        item.get("eval_splits")
        or defaults.get("eval_splits")
        or eval_spec.get("eval_splits")
        or ["validation"]
    )
    batch_size = int(item.get("batch_size") or defaults.get("batch_size") or 4)
    controls = _item_eval_controls(item, queue_doc, spec)
    report_json = Path(
        item.get("eval_report")
        or run_dir / f"eval_{expected_checkpoint.name}_v{validation_limit}_t{test_limit}_winrates_report.json"
    )
    eval_command = build_eval_command(
        python_bin=python_bin,
        checkpoint_dir=expected_checkpoint,
        train_parquet=train_parquet,
        validation_parquet=validation_parquet,
        test_parquet=test_parquet,
        report_json=report_json,
        validation_limit=validation_limit,
        test_limit=test_limit,
        eval_splits=eval_splits,
        batch_size=batch_size,
        controls=controls,
    )
    train_command = _with_python([str(part) for part in plan["command"]], python_bin)
    train_log = run_dir / "train.log"
    eval_log = report_json.with_suffix(".log")

    if dry_run:
        return {
            "status": "dry_run",
            "item_index": index,
            "item_name": item["name"],
            "run_dir": run_dir,
            "expected_checkpoint": expected_checkpoint,
            "train_command": train_command,
            "eval_command": eval_command,
        }

    try:
        update_item(
            queue_path,
            index,
            status="training",
            started_at=utc_now(),
            run_dir=str(run_dir),
            expected_checkpoint=str(expected_checkpoint),
            checkpoint_dir=str(expected_checkpoint),
            train_log=str(train_log),
        )
        env = _env_for_run(plan, code_root)
        _run_logged(train_command, cwd=code_root, env=env, log_path=train_log)
        if not expected_checkpoint.exists():
            raise QueueError(f"training completed without expected checkpoint: {expected_checkpoint}")
        nano_ar_hpo_study.assert_lr_decay_canary_for_run(config_path, train_log)
        update_item(
            queue_path,
            index,
            status="eval_running",
            eval_started_at=utc_now(),
            checkpoint_dir=str(expected_checkpoint),
            eval_report=str(report_json),
            eval_log=str(eval_log),
        )
        _run_logged(eval_command, cwd=code_root, env=env, log_path=eval_log)
        _record_trial(
            queue_doc=queue_doc,
            item=item,
            config_path=config_path,
            eval_report=report_json,
            train_log=train_log,
            run_dir=run_dir,
        )
        update_item(
            queue_path,
            index,
            status="complete",
            completed_at=utc_now(),
            checkpoint_dir=str(expected_checkpoint),
            eval_report=str(report_json),
            eval_log=str(eval_log),
        )
        return {
            "status": "complete",
            "item_index": index,
            "item_name": item["name"],
            "checkpoint_dir": expected_checkpoint,
            "eval_report": report_json,
        }
    except Exception as exc:
        update_item(queue_path, index, status="failed", failed_at=utc_now(), failure=str(exc))
        return {"status": "failed", "item_index": index, "item_name": item["name"], "failure": str(exc)}


def queue_status(queue_path: str | Path) -> dict[str, Any]:
    queue_doc = load_queue(queue_path)
    counts = status_counts(queue_doc["items"], VALID_STATUSES)
    return {"queue": str(queue_path), "counts": counts, "items": queue_doc["items"]}


@contextlib.contextmanager
def queue_lock(queue_path: str | Path):
    lock_path = Path(queue_path).with_suffix(Path(queue_path).suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QueueError(f"queue watcher already active: {lock_path}") from exc
        handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def watch_queue(
    queue_path: Path,
    *,
    poll_seconds: int,
    dry_run: bool,
    once: bool,
    stop_when_idle: bool = False,
    stop_on_failure: bool = True,
) -> int:
    with queue_lock(queue_path):
        while True:
            result = process_next_item(queue_path, dry_run=dry_run)
            print(json.dumps(result, default=str, sort_keys=True), flush=True)
            if once or dry_run:
                return 0
            if result["status"] == "failed" and stop_on_failure:
                return 1
            if result["status"] == "idle":
                if stop_when_idle:
                    return 0
                time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--status", action="store_true", help="Print queue status and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare the next item and print commands without running.")
    parser.add_argument("--once", action="store_true", help="Process at most one item.")
    parser.add_argument("--run-until-empty", action="store_true", help="Process pending items sequentially and exit when the queue is idle.")
    parser.add_argument("--reset-active", action="store_true", help="Reset stale training/eval_running items to pending before running or printing status.")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue to later pending items even if one item fails.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    if args.reset_active:
        print(json.dumps(reset_active_items(args.queue), indent=2, sort_keys=True))
    if args.status:
        print(json.dumps(queue_status(args.queue), indent=2, sort_keys=True))
        return 0
    return watch_queue(
        args.queue,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
        once=args.once,
        stop_when_idle=args.run_until_empty,
        stop_on_failure=not args.continue_on_failure,
    )


if __name__ == "__main__":
    raise SystemExit(main())
