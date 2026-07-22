#!/usr/bin/env python3
"""Sequential queue for Nano AV->AR round-trip gate evals."""

from __future__ import annotations

import argparse
import copy
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
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
import nano_av_probe_queue  # noqa: E402
import nano_roundtrip_eval_config  # noqa: E402
import nano_source_provenance  # noqa: E402
import eval_nano_av_ar_roundtrip_gate as roundtrip_gate  # noqa: E402
from nano_queue_status import PENDING_STATUS, TERMINAL_STATUSES, status_counts  # noqa: E402


SCHEMA_VERSION = "nano_roundtrip_queue.v1"
ACTIVE_STATUSES = {"running", "scoring"}
ROUNDTRIP_TERMINAL_STATUSES = {*TERMINAL_STATUSES, "blocked_waiting_on_parse_health"}
VALID_STATUSES = {PENDING_STATUS, *ACTIVE_STATUSES, *ROUNDTRIP_TERMINAL_STATUSES}
class RoundtripQueueError(ValueError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_queue(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise RoundtripQueueError(f"queue YAML must use schema_version {SCHEMA_VERSION}")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise RoundtripQueueError("queue YAML must contain non-empty items")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RoundtripQueueError(f"item {index} must be a mapping")
        item.setdefault("status", PENDING_STATUS)
        if item["status"] not in VALID_STATUSES:
            raise RoundtripQueueError(f"item {index} has invalid status {item['status']!r}")
        if not item.get("name") or not item.get("config"):
            raise RoundtripQueueError(f"item {index} requires name and config")
    return data


def write_queue(path: str | Path, queue_doc: dict[str, Any]) -> None:
    destination = Path(path)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(queue_doc, sort_keys=False))
    tmp.replace(destination)


def update_item(path: str | Path, index: int, **fields: Any) -> None:
    queue_doc = load_queue(path)
    queue_doc["items"][index].update({key: value for key, value in fields.items() if value is not None})
    write_queue(path, queue_doc)


def reset_active_items(path: str | Path, *, reason: str = "manual active reset") -> dict[str, Any]:
    queue_doc = load_queue(path)
    reset_items: list[str] = []
    reset_at = utc_now()
    for item in queue_doc["items"]:
        status = item.get("status")
        if status not in ACTIVE_STATUSES:
            continue
        item["previous_status"] = status
        item["status"] = PENDING_STATUS
        item["reset_at"] = reset_at
        item["reset_reason"] = reason
        reset_items.append(str(item["name"]))
    if reset_items:
        write_queue(path, queue_doc)
    return {"queue": str(path), "reset_count": len(reset_items), "items": reset_items}


def next_pending_index(queue_doc: dict[str, Any]) -> int | None:
    for index, item in enumerate(queue_doc["items"]):
        if item.get("status") == PENDING_STATUS:
            return index
    return None


def queue_status(queue_path: str | Path) -> dict[str, Any]:
    queue_doc = load_queue(queue_path)
    return {"queue": str(queue_path), "counts": status_counts(queue_doc["items"], VALID_STATUSES), "items": queue_doc["items"]}


def dry_run_queue(queue_path: str | Path) -> dict[str, Any]:
    """Resolve every queue item and prove generation-protocol parity."""

    path = Path(queue_path)
    queue_doc = load_queue(path)
    prepare_spec = _checkpoint_prepare_spec(queue_doc)
    dry_run_preparation = None
    if prepare_spec is not None:
        model_material = str(prepare_spec["dcp_checkpoint"]).encode()
        tokenizer_material = str(prepare_spec["origin_hf_dir"]).encode()
        dry_run_preparation = {
            "output_hf_dir": str(prepare_spec["output_hf_dir"]),
            "av_model_fingerprint": (
                "dcp_model_sha256:" + hashlib.sha256(model_material).hexdigest()
            ),
            "av_tokenizer_fingerprint": (
                "tokenizer_files_sha256:"
                + hashlib.sha256(tokenizer_material).hexdigest()
            ),
        }
    resolved_items = []
    protocol_hashes: set[str] = set()
    for item in queue_doc["items"]:
        config_path = resolve_path(item["config"], path, queue_doc)
        if not config_path.exists() and not Path(item["config"]).is_absolute():
            local_candidate = SCRIPT_DIR.parent / Path(item["config"])
            if local_candidate.exists():
                config_path = local_candidate
        config = nano_roundtrip_eval_config.load_raw_config(config_path)
        config = nano_roundtrip_eval_config.validate_config(
            apply_prepared_checkpoint(config, dry_run_preparation)
        )
        command = nano_roundtrip_eval_config.build_command(
            config,
            config_path=config_path,
            python_bin=python_for_item(config, queue_doc, item),
        )
        args = roundtrip_gate.parse_args(command[2:])
        protocol = roundtrip_gate.build_generation_protocol(args)
        protocol_sha256 = roundtrip_gate.generation_protocol_sha256(protocol)
        protocol_hashes.add(protocol_sha256)
        resolved_items.append(
            {
                "name": item["name"],
                "config": str(config_path),
                "command": command,
                "generation_protocol": protocol,
                "generation_protocol_sha256": protocol_sha256,
                "model_fingerprint": args.av_model_fingerprint,
                "identity_materialized": dry_run_preparation is None,
            }
        )
    return {
        "queue": str(path),
        "items": resolved_items,
        "protocol_sha256s": sorted(protocol_hashes),
        "protocols_match": len(protocol_hashes) == 1,
    }


def resolve_path(value: str | Path, queue_path: Path, queue_doc: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    code_root = queue_doc.get("defaults", {}).get("code_root")
    if code_root:
        return Path(code_root) / path
    return queue_path.parent / path


def _config_path(value: Any, *, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def report_json_for_config(config: dict[str, Any], *, config_path: Path) -> Path:
    return _config_path(config["paths"]["report_json"], base=config_path.parent)


def generated_jsonl_for_config(config: dict[str, Any], *, config_path: Path) -> Path | None:
    value = (config.get("paths") or {}).get("generated_jsonl")
    if value in {None, ""}:
        return None
    return _config_path(value, base=config_path.parent)


def code_root_for_config(config: dict[str, Any], queue_doc: dict[str, Any]) -> Path:
    value = os.environ.get("NANO_QUEUE_CODE_ROOT") or (
        queue_doc.get("defaults") or {}
    ).get("code_root") or (
        config.get("paths") or {}
    ).get("code_root")
    return Path(value) if value else Path.cwd()


def python_for_item(config: dict[str, Any], queue_doc: dict[str, Any], item: dict[str, Any]) -> str:
    return str(item.get("python") or (queue_doc.get("defaults") or {}).get("python") or config.get("python") or sys.executable)


def env_for_run(*, code_root: Path, queue_doc: dict[str, Any], item: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    for source in ((queue_doc.get("defaults") or {}).get("environment"), item.get("environment")):
        if isinstance(source, dict):
            env.update({str(key): str(value) for key, value in source.items()})
    env["WANDB_MODE"] = "offline"
    pythonpath = [
        str(code_root / "external" / "natural_language_autoencoders"),
        str(code_root / "external" / "natural_language_autoencoders" / "Miles"),
        str(code_root),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    return env


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n# started_utc={utc_now()}\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"# completed_utc={utc_now()}\n")


def _directory_stat_signature(path: Path) -> str:
    records = [
        (str(source.relative_to(path)), source.stat().st_size, source.stat().st_mtime_ns)
        for source in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    canonical = json.dumps(records, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def fingerprint_tokenizer_files(path: str | Path) -> dict[str, Any]:
    try:
        return nano_source_provenance.fingerprint_tokenizer_files(path)
    except nano_source_provenance.SourceProvenanceError as exc:
        raise RoundtripQueueError(str(exc)) from exc


def _hf_checkpoint_complete(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            model_files = {str(value) for value in (index.get("weight_map") or {}).values()}
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return bool(model_files) and all(
            (path / filename).is_file() and (path / filename).stat().st_size > 0
            for filename in model_files
        )
    model_files = list(path.glob("*.safetensors"))
    return bool(model_files) and all(item.stat().st_size > 0 for item in model_files)


def _reuse_existing_hf_checkpoint(
    spec: dict[str, Any],
    *,
    checkpoint_dir: Path,
    model_dir: Path,
    origin_hf_dir: Path,
    output_hf_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    source_value = spec.get("source_fingerprint_report_json")
    if not source_value:
        raise RoundtripQueueError(
            "reuse_existing_hf requires source_fingerprint_report_json"
        )
    source_path = Path(source_value)
    if not source_path.is_file():
        raise RoundtripQueueError(f"source fingerprint report is missing: {source_path}")
    source_hf_dir = Path(spec.get("existing_hf_source_dir") or output_hf_dir)
    if not _hf_checkpoint_complete(source_hf_dir):
        raise RoundtripQueueError(
            f"existing HF checkpoint is incomplete: {source_hf_dir}"
        )
    source = json.loads(source_path.read_text())
    expected_model = spec.get("expected_model_fingerprint")
    expected_tokenizer = spec.get("expected_tokenizer_fingerprint")
    if not expected_model or not expected_tokenizer:
        raise RoundtripQueueError(
            "reuse_existing_hf requires expected model and tokenizer fingerprints"
        )
    checks = {
        "dcp_checkpoint": (source.get("dcp_checkpoint"), str(checkpoint_dir)),
        "origin_hf_dir": (source.get("origin_hf_dir"), str(origin_hf_dir)),
        "av_model_fingerprint": (source.get("av_model_fingerprint"), expected_model),
        "av_tokenizer_fingerprint": (
            source.get("av_tokenizer_fingerprint"),
            expected_tokenizer,
        ),
        "model_stat_signature": (
            source.get("model_stat_signature"),
            _directory_stat_signature(model_dir),
        ),
        "tokenizer_stat_signature": (
            source.get("tokenizer_stat_signature"),
            _directory_stat_signature(origin_hf_dir),
        ),
    }
    mismatches = {
        name: {"source": actual, "expected": expected}
        for name, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise RoundtripQueueError(
            "existing HF source fingerprint report does not match current inputs: "
            + json.dumps(mismatches, sort_keys=True)
        )
    hf_stage = None
    if source_hf_dir.resolve() != output_hf_dir.resolve():
        manifest_path = output_hf_dir / ".nano_hf_stage_manifest.json"
        if output_hf_dir.exists():
            if not _hf_checkpoint_complete(output_hf_dir) or not manifest_path.is_file():
                raise RoundtripQueueError(
                    "existing HF stage is incomplete or lacks its manifest: "
                    f"{output_hf_dir}"
                )
            hf_stage = json.loads(manifest_path.read_text())
            current_signature = _directory_stat_signature(source_hf_dir)
            if (
                hf_stage.get("source_hf_dir") != str(source_hf_dir)
                or hf_stage.get("source_stat_signature") != current_signature
            ):
                raise RoundtripQueueError(
                    "existing HF stage manifest does not match its source"
                )
            hf_stage["reused"] = True
        else:
            stage_workers = int(spec.get("existing_hf_stage_workers", 4))
            stage_task_bytes = int(
                spec.get("existing_hf_stage_task_bytes", 512 * 1024 * 1024)
            )
            source_signature = _directory_stat_signature(source_hf_dir)
            staged_fingerprint = nano_source_provenance.fingerprint_and_copy_directory(
                source_hf_dir,
                output_hf_dir,
                label="existing_av_hf",
                workers=stage_workers,
                task_size=stage_task_bytes,
            )
            hf_stage = {
                "source_hf_dir": str(source_hf_dir),
                "destination_hf_dir": str(output_hf_dir),
                "source_stat_signature": source_signature,
                "stage_workers": stage_workers,
                "stage_task_bytes": stage_task_bytes,
                "fingerprint": staged_fingerprint,
                "reused": False,
            }
            write_queue_manifest = output_hf_dir / ".nano_hf_stage_manifest.json"
            write_queue_manifest.write_text(
                json.dumps(hf_stage, indent=2, sort_keys=True) + "\n"
            )
    if not _hf_checkpoint_complete(output_hf_dir):
        raise RoundtripQueueError(
            f"reused HF checkpoint is incomplete: {output_hf_dir}"
        )
    report = copy.deepcopy(source)
    report.update(
        {
            "output_hf_dir": str(output_hf_dir),
            "prepared_at": utc_now(),
            "reused_existing_hf": True,
            "source_fingerprint_report_json": str(source_path),
            "existing_hf_source_dir": str(source_hf_dir),
            "existing_hf_stage": hf_stage,
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    return report


def _checkpoint_prepare_spec(queue_doc: dict[str, Any]) -> dict[str, Any] | None:
    value = (queue_doc.get("defaults") or {}).get("av_checkpoint_prepare")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RoundtripQueueError("defaults.av_checkpoint_prepare must be a mapping")
    required = (
        "dcp_checkpoint",
        "origin_hf_dir",
        "output_hf_dir",
        "fingerprint_report_json",
        "convert_log",
    )
    missing = [name for name in required if not value.get(name)]
    if missing:
        raise RoundtripQueueError(
            f"defaults.av_checkpoint_prepare is missing keys: {missing}"
        )
    return value


def prepare_av_checkpoint(
    queue_doc: dict[str, Any],
    *,
    code_root: Path,
    env: dict[str, str],
) -> dict[str, Any] | None:
    spec = _checkpoint_prepare_spec(queue_doc)
    if spec is None:
        return None
    checkpoint_dir = Path(spec["dcp_checkpoint"])
    model_dir = checkpoint_dir / "model"
    origin_hf_dir = Path(spec["origin_hf_dir"])
    output_hf_dir = Path(spec["output_hf_dir"])
    report_path = Path(spec["fingerprint_report_json"])
    convert_log = Path(spec["convert_log"])
    staged_checkpoint_value = spec.get("stage_dcp_checkpoint")
    staged_checkpoint = (
        Path(staged_checkpoint_value) if staged_checkpoint_value else None
    )
    staged_model_dir = (
        staged_checkpoint / "model" if staged_checkpoint is not None else None
    )
    cleanup_staged = bool(
        spec.get("cleanup_staged_dcp_after_conversion", True)
    )
    stage_copy_workers = int(spec.get("stage_copy_workers", 1))
    stage_copy_task_bytes = int(
        spec.get("stage_copy_task_bytes", 512 * 1024 * 1024)
    )
    if stage_copy_workers < 1:
        raise RoundtripQueueError("stage_copy_workers must be at least 1")
    if stage_copy_task_bytes < 1:
        raise RoundtripQueueError("stage_copy_task_bytes must be at least 1")
    if not model_dir.is_dir():
        raise RoundtripQueueError(f"AV DCP model directory is missing: {model_dir}")
    if not (origin_hf_dir / "config.json").is_file():
        raise RoundtripQueueError(f"origin HF checkpoint is invalid: {origin_hf_dir}")
    if bool(spec.get("reuse_existing_hf", False)):
        return _reuse_existing_hf_checkpoint(
            spec,
            checkpoint_dir=checkpoint_dir,
            model_dir=model_dir,
            origin_hf_dir=origin_hf_dir,
            output_hf_dir=output_hf_dir,
            report_path=report_path,
        )

    model_signature = _directory_stat_signature(model_dir)
    tokenizer_signature = _directory_stat_signature(origin_hf_dir)
    report: dict[str, Any] | None = None
    if report_path.is_file():
        candidate = json.loads(report_path.read_text())
        if (
            candidate.get("model_stat_signature") == model_signature
            and candidate.get("tokenizer_stat_signature") == tokenizer_signature
            and candidate.get("dcp_checkpoint") == str(checkpoint_dir)
            and candidate.get("origin_hf_dir") == str(origin_hf_dir)
        ):
            report = candidate

    needs_conversion = not (output_hf_dir / "config.json").is_file()
    staged_model: dict[str, Any] | None = None
    if staged_model_dir is not None and needs_conversion:
        if staged_checkpoint.exists():
            shutil.rmtree(staged_checkpoint)
        staged_model = nano_source_provenance.fingerprint_and_copy_directory(
            model_dir,
            staged_model_dir,
            label="av_dcp_model",
            workers=stage_copy_workers,
            task_size=stage_copy_task_bytes,
        )

    if report is None:
        model = staged_model or nano_source_provenance.fingerprint_directory(
            model_dir,
            label="av_dcp_model",
        )
        tokenizer = fingerprint_tokenizer_files(origin_hf_dir)
        report = {
            "schema_version": "nano_roundtrip_checkpoint_prepare.v1",
            "created_at": utc_now(),
            "dcp_checkpoint": str(checkpoint_dir),
            "origin_hf_dir": str(origin_hf_dir),
            "output_hf_dir": str(output_hf_dir),
            "stage_copy_workers": stage_copy_workers,
            "stage_copy_task_bytes": stage_copy_task_bytes,
            "model_stat_signature": model_signature,
            "tokenizer_stat_signature": tokenizer_signature,
            "model": model,
            "tokenizer": tokenizer,
            "av_model_fingerprint": f"dcp_model_sha256:{model['sha256']}",
            "av_tokenizer_fingerprint": f"tokenizer_files_sha256:{tokenizer['sha256']}",
        }
    elif staged_model is not None and staged_model["sha256"] != report["model"]["sha256"]:
        raise RoundtripQueueError(
            "staged AV DCP fingerprint differs from the cached source fingerprint"
        )

    if needs_conversion:
        if output_hf_dir.exists():
            shutil.rmtree(output_hf_dir)
        conversion_checkpoint = staged_checkpoint or checkpoint_dir
        command = nano_av_probe_queue.build_convert_command(
            python_bin=str(spec.get("python") or sys.executable),
            code_root=code_root,
            checkpoint_dir=conversion_checkpoint,
            origin_hf_dir=origin_hf_dir,
            output_dir=output_hf_dir,
            torch_dtype=str(spec.get("torch_dtype") or "bfloat16"),
        )
        try:
            run_logged(command, cwd=code_root, env=env, log_path=convert_log)
        finally:
            if staged_checkpoint is not None and cleanup_staged:
                shutil.rmtree(staged_checkpoint, ignore_errors=True)
    if not (output_hf_dir / "config.json").is_file():
        raise RoundtripQueueError(
            f"DCP conversion completed without HF config.json: {output_hf_dir}"
        )

    report["prepared_at"] = utc_now()
    if staged_checkpoint is not None:
        report["staged_dcp_checkpoint"] = str(staged_checkpoint)
        report["staged_dcp_cleanup"] = cleanup_staged
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    return report


def apply_prepared_checkpoint(
    config: dict[str, Any],
    preparation: dict[str, Any] | None,
) -> dict[str, Any]:
    if preparation is None:
        return config
    resolved = copy.deepcopy(config)
    resolved["paths"]["av_hf_checkpoint"] = preparation["output_hf_dir"]
    resolved["eval"]["av_model_fingerprint"] = preparation["av_model_fingerprint"]
    resolved["eval"]["av_tokenizer_fingerprint"] = preparation[
        "av_tokenizer_fingerprint"
    ]
    return resolved


def cleanup_prepared_checkpoint(queue_path: Path) -> dict[str, Any] | None:
    queue_doc = load_queue(queue_path)
    spec = _checkpoint_prepare_spec(queue_doc)
    if spec is None or not bool(spec.get("cleanup_after_queue", False)):
        return None
    if any(item.get("status") != "complete" for item in queue_doc["items"]):
        return None
    output_hf_dir = Path(spec["output_hf_dir"])
    if output_hf_dir.exists():
        shutil.rmtree(output_hf_dir)
    queue_doc["defaults"]["av_checkpoint_prepare"]["cleaned_at"] = utc_now()
    write_queue(queue_path, queue_doc)
    return {"cleaned": str(output_hf_dir)}


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RoundtripQueueError(f"round-trip command completed without report: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise RoundtripQueueError(f"round-trip report is not a JSON object: {path}")
    return data


def _validate_generation_report(
    report: dict[str, Any],
    *,
    config: dict[str, Any],
    generated_jsonl: Path | None,
) -> dict[str, Any]:
    if report.get("schema_version") != "nano_roundtrip_generation_report.v1":
        raise RoundtripQueueError(
            "generation-only item did not produce a generation report"
        )
    eval_config = config.get("eval") or {}
    eval_splits = [str(value) for value in eval_config.get("eval_splits") or []]
    if not eval_splits:
        raise RoundtripQueueError("generation-only item has no eval_splits")
    expected_rows = 0
    for split in eval_splits:
        limit = eval_config.get(f"{split}_limit")
        if not isinstance(limit, int) or limit <= 0:
            raise RoundtripQueueError(
                f"generation-only item has invalid {split}_limit: {limit!r}"
            )
        expected_rows += limit
    if report.get("row_count") != expected_rows:
        raise RoundtripQueueError(
            "generation-only row count mismatch: "
            f"expected={expected_rows} observed={report.get('row_count')}"
        )
    if generated_jsonl is None:
        raise RoundtripQueueError("generation-only item has no generated_jsonl")
    reported_path = Path(str(report.get("generated_jsonl") or ""))
    if reported_path != generated_jsonl:
        raise RoundtripQueueError(
            "generation-only merged cache path mismatch: "
            f"expected={generated_jsonl} observed={reported_path}"
        )
    if not generated_jsonl.is_file():
        raise RoundtripQueueError(
            f"generation-only merged cache is missing: {generated_jsonl}"
        )
    observed_rows = sum(1 for line in generated_jsonl.open() if line.strip())
    if observed_rows != expected_rows:
        raise RoundtripQueueError(
            "generation-only merged cache row count mismatch: "
            f"expected={expected_rows} observed={observed_rows}"
        )
    protocol_sha = str(report.get("generation_protocol_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", protocol_sha):
        raise RoundtripQueueError(
            "generation-only report lacks a valid generation protocol hash"
        )
    return {
        "row_count": expected_rows,
        "generated_jsonl": str(generated_jsonl),
        "generation_protocol_sha256": protocol_sha,
    }


def _record_hpo_trial(
    *,
    queue_doc: dict[str, Any],
    item: dict[str, Any],
    config_path: Path,
    report_json: Path,
    log_path: Path,
) -> None:
    defaults = queue_doc.get("defaults") or {}
    study_jsonl = defaults.get("study_jsonl")
    if not study_jsonl:
        return
    record = nano_ar_hpo_study.build_trial_record(
        trial_name=str(item["name"]),
        config_path=config_path,
        roundtrip_report_path=report_json,
        train_log_path=log_path,
        run_dir=report_json.parent,
        status="complete",
        notes=item.get("notes"),
        task="av_roundtrip",
    )
    nano_ar_hpo_study.upsert_trial(Path(study_jsonl), record)
    optuna_json = defaults.get("optuna_json")
    if optuna_json:
        payload = nano_ar_hpo_study.export_optuna_payload(nano_ar_hpo_study.load_trials(Path(study_jsonl)), task="av_roundtrip")
        Path(optuna_json).parent.mkdir(parents=True, exist_ok=True)
        Path(optuna_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_active_process_lines() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid,etime,cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def launch_guard_for_item(queue_doc: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in ((queue_doc.get("defaults") or {}).get("launch_guard"), item.get("launch_guard")):
        if isinstance(source, dict):
            merged.update(source)
    return merged


def active_process_matches(process_lines: list[str], patterns: list[str] | tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for line in process_lines:
        for pattern in patterns:
            if re.search(str(pattern), line):
                matches.append(line)
                break
    return matches


def process_next(queue_path: Path, *, active_process_lines: list[str] | None = None) -> dict[str, Any]:
    queue_doc = load_queue(queue_path)
    index = next_pending_index(queue_doc)
    if index is None:
        return {"status": "idle"}
    item = queue_doc["items"][index]
    launch_guard = launch_guard_for_item(queue_doc, item)
    patterns = [str(pattern) for pattern in launch_guard.get("block_if_process_matches") or []]
    if patterns:
        lines = active_process_lines if active_process_lines is not None else read_active_process_lines()
        matches = active_process_matches(lines, patterns)
        if matches:
            return {
                "status": "blocked_active_process",
                "item": item["name"],
                "patterns": patterns,
                "matches": matches[:10],
            }
    config_path = resolve_path(item["config"], queue_path, queue_doc)
    config = nano_roundtrip_eval_config.load_raw_config(config_path)
    code_root = code_root_for_config(config, queue_doc)
    env = env_for_run(code_root=code_root, queue_doc=queue_doc, item=item)
    preparation = prepare_av_checkpoint(queue_doc, code_root=code_root, env=env)
    config = nano_roundtrip_eval_config.validate_config(
        apply_prepared_checkpoint(config, preparation)
    )
    python_bin = python_for_item(config, queue_doc, item)
    command = nano_roundtrip_eval_config.build_command(config, config_path=config_path, python_bin=python_bin)
    report_json = report_json_for_config(config, config_path=config_path)
    generated_jsonl = generated_jsonl_for_config(config, config_path=config_path)
    log_path = Path(item.get("log_path") or report_json.with_suffix(".log"))

    update_item(
        queue_path,
        index,
        status="running",
        started_at=utc_now(),
        config_resolved=str(config_path),
        report_json=str(report_json),
        generated_jsonl=str(generated_jsonl) if generated_jsonl else None,
        log_path=str(log_path),
        prepared_av_hf=None if preparation is None else preparation["output_hf_dir"],
        av_model_fingerprint=None
        if preparation is None
        else preparation["av_model_fingerprint"],
        av_tokenizer_fingerprint=None
        if preparation is None
        else preparation["av_tokenizer_fingerprint"],
    )
    try:
        run_logged(command, cwd=code_root, env=env, log_path=log_path)
        report = _load_report(report_json)
        generation_only = bool((config.get("eval") or {}).get("generation_only"))
        generation_evidence = None
        if generation_only:
            generation_evidence = _validate_generation_report(
                report,
                config=config,
                generated_jsonl=generated_jsonl,
            )
            gate_passed = None
        else:
            gate = report.get("gate") if isinstance(report.get("gate"), dict) else {}
            gate_passed = bool(gate.get("passed"))
            _record_hpo_trial(
                queue_doc=queue_doc,
                item=item,
                config_path=config_path,
                report_json=report_json,
                log_path=log_path,
            )
        update_item(
            queue_path,
            index,
            status="complete",
            completed_at=utc_now(),
            gate_passed=gate_passed,
            generation_evidence=generation_evidence,
        )
        return {
            "status": "complete",
            "item": item["name"],
            "report_json": str(report_json),
            "gate_passed": gate_passed,
            "generation_evidence": generation_evidence,
        }
    except Exception as exc:
        update_item(queue_path, index, status="failed", failed_at=utc_now(), failure=str(exc))
        return {"status": "failed", "item": item["name"], "failure": str(exc)}


@contextlib.contextmanager
def queue_lock(queue_path: Path):
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _status_cmd(args: argparse.Namespace) -> int:
    print(json.dumps(queue_status(args.queue), indent=2, sort_keys=True))
    return 0


def _reset_active_cmd(args: argparse.Namespace) -> int:
    print(json.dumps(reset_active_items(args.queue, reason=args.reason), indent=2, sort_keys=True))
    return 0


def _dry_run_cmd(args: argparse.Namespace) -> int:
    result = dry_run_queue(args.queue)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["protocols_match"] else 1


def _run_once(args: argparse.Namespace) -> int:
    with queue_lock(args.queue):
        result = process_next(args.queue)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 1 if result.get("status") == "failed" else 0


def _run_loop(args: argparse.Namespace) -> int:
    processed = 0
    while True:
        with queue_lock(args.queue):
            result = process_next(args.queue)
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] == "idle":
            cleanup = cleanup_prepared_checkpoint(args.queue)
            if cleanup is not None:
                print(json.dumps({"status": "cleanup", **cleanup}, sort_keys=True), flush=True)
            return 0
        if result["status"] == "blocked_active_process":
            return 0
        processed += 1
        if result["status"] == "failed" and not args.continue_on_failure:
            return 1
        if args.max_items and processed >= args.max_items:
            return 0
        time.sleep(args.sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Print queue status.")
    status.add_argument("queue", type=Path)
    status.set_defaults(func=_status_cmd)

    reset = subparsers.add_parser("reset-active", help="Reset stale running/scoring items to pending.")
    reset.add_argument("queue", type=Path)
    reset.add_argument("--reason", default="manual active reset")
    reset.set_defaults(func=_reset_active_cmd)

    dry_run = subparsers.add_parser(
        "dry-run",
        help="Resolve every item and verify generation-protocol parity.",
    )
    dry_run.add_argument("queue", type=Path)
    dry_run.set_defaults(func=_dry_run_cmd)

    run_once = subparsers.add_parser("run-once", help="Run the next pending item.")
    run_once.add_argument("queue", type=Path)
    run_once.set_defaults(func=_run_once)

    run_loop = subparsers.add_parser("run-loop", help="Run pending items sequentially.")
    run_loop.add_argument("queue", type=Path)
    run_loop.add_argument("--sleep-seconds", type=float, default=30.0)
    run_loop.add_argument("--max-items", type=int)
    run_loop.add_argument("--continue-on-failure", action="store_true")
    run_loop.set_defaults(func=_run_loop)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
