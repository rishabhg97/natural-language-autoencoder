#!/usr/bin/env python3
"""YAML queue driver for Nano NLA RL smoke/medium/hero runs.

The queue owns experiment parameters. The shell launcher stays stable and only
receives environment variables plus CLI overrides rendered from YAML.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_queue_status import ACTIVE_HPO_STATUSES, VALID_HPO_STATUSES, status_counts  # noqa: E402
from nano_checkpoint_retention import (  # noqa: E402
    RetentionPolicy,
    build_cleanup_plan,
    execute_cleanup,
)
from nano_source_provenance import (  # noqa: E402
    collect_source_provenance,
    fingerprint_hf_model_files,
    fingerprint_tokenizer_files,
    hf_model_stat_signature,
    tokenizer_stat_signature,
    verify_source_policy,
    write_provenance,
)


VALID_STATUSES = VALID_HPO_STATUSES
ACTIVE_STATUSES = ACTIVE_HPO_STATUSES


class QueueError(ValueError):
    """Raised when an RL queue manifest is unsafe or malformed."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _redact_launch_value(value: Any, *, key: str | None = None) -> Any:
    sensitive_markers = ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "CREDENTIAL")
    if key is not None and any(marker in key.upper() for marker in sensitive_markers):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(child_key): _redact_launch_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_launch_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _cli_option_value(command: list[str] | None, option: str) -> str | None:
    if not command or option not in command:
        return None
    index = command.index(option)
    if index + 1 >= len(command):
        raise QueueError(f"CLI option has no value: {option}")
    return str(command[index + 1])


def _set_cli_option(command: list[str] | None, option: str, value: str) -> None:
    if command is None:
        return
    if option in command:
        index = command.index(option)
        if index + 1 >= len(command):
            raise QueueError(f"CLI option has no value: {option}")
        existing = str(command[index + 1])
        if existing != value:
            raise QueueError(
                f"refusing to replace frozen CLI option {option}: "
                f"existing={existing!r} resolved={value!r}"
            )
        return
    command.extend([option, value])


def _roundtrip_identity_required(eval_spec: dict[str, Any]) -> bool:
    commands = (
        eval_spec.get("eval_command"),
        eval_spec.get("generation_command"),
        eval_spec.get("score_command"),
    )
    required_flags = {
        "--require-generation-protocol-match",
        "--reuse-generated",
        "--resume-generated",
    }
    return any(
        flag in command
        for command in commands
        if command is not None
        for flag in required_flags
    )


def _infer_origin_hf_dir(eval_spec: dict[str, Any]) -> Path:
    configured = eval_spec.get("origin_hf_dir")
    if configured:
        return Path(str(configured))
    inferred = _cli_option_value(
        eval_spec.get("converter_command"), "--origin-hf-dir"
    )
    if inferred:
        return Path(inferred)
    raise QueueError(
        "round-trip generation identity requires origin_hf_dir or a frozen "
        "converter command containing --origin-hf-dir"
    )


def _resolve_roundtrip_generation_identity(
    eval_spec: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve content identities after DCP-to-HF conversion.

    Dynamic RL checkpoints cannot declare their model hash before training.
    The queue fingerprints the completed temporary HF artifact once, caches
    the report behind stat signatures, and injects one identity into both the
    generation and scoring commands.
    """

    model_fingerprint = eval_spec.get("av_model_fingerprint") or _cli_option_value(
        eval_spec.get("eval_command"), "--av-model-fingerprint"
    )
    tokenizer_fingerprint = eval_spec.get(
        "av_tokenizer_fingerprint"
    ) or _cli_option_value(
        eval_spec.get("eval_command"), "--av-tokenizer-fingerprint"
    )
    if bool(model_fingerprint) != bool(tokenizer_fingerprint):
        raise QueueError(
            "round-trip generation identity requires both model and tokenizer "
            "fingerprints"
        )

    report: dict[str, Any] | None = None
    if not model_fingerprint and _roundtrip_identity_required(eval_spec):
        mode = str(eval_spec.get("av_fingerprint_mode") or "auto_hf")
        if mode != "auto_hf":
            raise QueueError(
                "round-trip generation identity requires explicit fingerprints "
                "or av_fingerprint_mode=auto_hf"
            )
        hf_dir = Path(str(eval_spec["hf_output_dir"]))
        origin_hf_dir = _infer_origin_hf_dir(eval_spec)
        workers = int(eval_spec.get("fingerprint_workers") or 1)
        if workers < 1:
            raise QueueError("post-eval fingerprint_workers must be at least 1")
        report_path = Path(
            str(
                eval_spec.get("fingerprint_report_json")
                or Path(str(eval_spec["report_json"])).with_name(
                    f"{eval_spec['iter_name']}_generation_identity.json"
                )
            )
        )
        model_signature = hf_model_stat_signature(hf_dir)
        tokenizer_signature = tokenizer_stat_signature(origin_hf_dir)
        if report_path.is_file():
            candidate = json.loads(report_path.read_text())
            if (
                candidate.get("model_stat_signature") == model_signature
                and candidate.get("tokenizer_stat_signature")
                == tokenizer_signature
                and candidate.get("hf_model_dir") == str(hf_dir)
                and candidate.get("origin_hf_dir") == str(origin_hf_dir)
            ):
                report = candidate
        if report is None:
            model = fingerprint_hf_model_files(hf_dir, workers=workers)
            tokenizer = fingerprint_tokenizer_files(origin_hf_dir)
            report = {
                "schema_version": "nano_roundtrip_generation_identity.v1",
                "created_at": utc_now(),
                "mode": mode,
                "hf_model_dir": str(hf_dir),
                "origin_hf_dir": str(origin_hf_dir),
                "fingerprint_workers": workers,
                "model_stat_signature": model_signature,
                "tokenizer_stat_signature": tokenizer_signature,
                "model": model,
                "tokenizer": tokenizer,
                "av_model_fingerprint": f"hf_model_sha256:{model['sha256']}",
                "av_tokenizer_fingerprint": (
                    f"tokenizer_files_sha256:{tokenizer['sha256']}"
                ),
            }
            _write_text_atomic(
                report_path,
                json.dumps(report, indent=2, sort_keys=True) + "\n",
            )
        model_fingerprint = str(report["av_model_fingerprint"])
        tokenizer_fingerprint = str(report["av_tokenizer_fingerprint"])
        eval_spec["fingerprint_report_json"] = str(report_path)

    if model_fingerprint and tokenizer_fingerprint:
        for key in ("eval_command", "generation_command", "score_command"):
            command = eval_spec.get(key)
            _set_cli_option(
                command,
                "--av-model-fingerprint",
                str(model_fingerprint),
            )
            _set_cli_option(
                command,
                "--av-tokenizer-fingerprint",
                str(tokenizer_fingerprint),
            )
        eval_spec["av_model_fingerprint"] = str(model_fingerprint)
        eval_spec["av_tokenizer_fingerprint"] = str(tokenizer_fingerprint)
    return report


def _validate_hf_checkpoint_for_reuse(path: Path) -> None:
    if not (path / "config.json").is_file():
        raise QueueError(f"reusable post-eval HF checkpoint lacks config.json: {path}")
    try:
        hf_model_stat_signature(path)
    except Exception as exc:
        raise QueueError(f"post-eval HF checkpoint is incomplete: {path}: {exc}") from exc


def freeze_launch_contract(
    *,
    queue_path: str | Path,
    queue_doc: dict[str, Any],
    item_index: int,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Persist one immutable, redacted launch snapshot before queue mutation."""

    run_dir = Path(str(spec["run_dir"]))
    contract_dir = run_dir / "launch_contract"
    queue_snapshot = _redact_launch_value(queue_doc)
    resolved_spec = _redact_launch_value(
        {
            key: spec.get(key)
            for key in (
                "item_name",
                "cwd",
                "code_root",
                "nla_root",
                "command",
                "env",
                "run_dir",
                "log_path",
                "resource_total_gpus",
                "rollout_batch_plan",
                "sglang_service",
                "source_provenance",
                "preregistration",
                "runtime_contracts",
                "required_gate_reports",
                "checkpoint_retention",
                "post_eval_specs",
            )
            if key in spec
        }
    )
    material = {
        "schema_version": "nano_immutable_launch_contract.v1",
        "queue_path": str(Path(queue_path).resolve()),
        "item_index": int(item_index),
        "queue_snapshot": queue_snapshot,
        "resolved_spec": resolved_spec,
    }
    contract_sha256 = _canonical_sha256(material)
    contract_path = contract_dir / "launch_contract.json"
    queue_snapshot_path = contract_dir / "queue_snapshot.yaml"
    resolved_spec_path = contract_dir / "resolved_spec.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text())
        if existing.get("contract_sha256") != contract_sha256:
            raise QueueError(
                "immutable launch contract mismatch for existing run directory: "
                f"{run_dir}"
            )
        return {
            "contract_path": str(contract_path),
            "contract_sha256": contract_sha256,
            "queue_snapshot_path": str(queue_snapshot_path),
            "resolved_spec_path": str(resolved_spec_path),
        }

    _write_text_atomic(
        queue_snapshot_path,
        yaml.safe_dump(queue_snapshot, sort_keys=False),
    )
    _write_text_atomic(
        resolved_spec_path,
        json.dumps(resolved_spec, indent=2, sort_keys=True) + "\n",
    )
    contract = {
        **material,
        "created_at": utc_now(),
        "contract_sha256": contract_sha256,
        "queue_snapshot_sha256": hashlib.sha256(
            queue_snapshot_path.read_bytes()
        ).hexdigest(),
        "resolved_spec_sha256": hashlib.sha256(
            resolved_spec_path.read_bytes()
        ).hexdigest(),
    }
    _write_text_atomic(
        contract_path,
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return {
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "queue_snapshot_path": str(queue_snapshot_path),
        "resolved_spec_path": str(resolved_spec_path),
    }


def load_launch_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path)
    if not contract_path.is_file():
        raise QueueError(f"immutable launch contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text())
    material = {
        key: contract.get(key)
        for key in (
            "schema_version",
            "queue_path",
            "item_index",
            "queue_snapshot",
            "resolved_spec",
        )
    }
    actual = _canonical_sha256(material)
    if contract.get("contract_sha256") != actual:
        raise QueueError(
            "immutable launch contract hash mismatch: "
            f"expected={contract.get('contract_sha256')} actual={actual}"
        )
    if material["schema_version"] != "nano_immutable_launch_contract.v1":
        raise QueueError("unsupported immutable launch contract schema")
    return contract


def load_queue(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict):
        raise QueueError(f"queue YAML must contain a mapping: {source}")
    return validate_queue(data, source=source)


def validate_queue(data: dict[str, Any], *, source: Path) -> dict[str, Any]:
    if data.get("schema_version") != "nano_rl_queue.v1":
        raise QueueError("schema_version must be nano_rl_queue.v1")
    defaults = data.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        raise QueueError("defaults must be a mapping")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise QueueError("items must be a non-empty list")
    item_names: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise QueueError(f"item {index} must be a mapping")
        item.setdefault("status", "pending")
        if item["status"] not in VALID_STATUSES:
            raise QueueError(f"item {index} has invalid status {item['status']!r}")
        if not item.get("name"):
            raise QueueError(f"item {index} requires name")
        name = str(item["name"])
        if name in item_names:
            raise QueueError(f"item {index} duplicates queue item name {name!r}")
        item_names.add(name)
        launch = item.get("launch")
        if launch is not None and not isinstance(launch, dict):
            raise QueueError(f"item {index} launch must be a mapping")
        for field in ("rl_parquet", "instruct_model", "actor_sft_ckpt", "critic_sl_ckpt", "run_dir"):
            if not item.get(field):
                raise QueueError(f"item {index} requires {field}")
    for index, item in enumerate(items):
        depends_on = item.get("depends_on")
        if depends_on is None:
            continue
        if not isinstance(depends_on, dict):
            raise QueueError(f"item {index} depends_on must be a mapping")
        dependency_name = str(depends_on.get("item") or "")
        if not dependency_name:
            raise QueueError(f"item {index} depends_on.item is required")
        if dependency_name not in item_names:
            raise QueueError(f"item {index} depends_on unknown item {dependency_name!r}")
        if dependency_name == str(item["name"]):
            raise QueueError(f"item {index} cannot depend on itself")
    return data


def write_queue(path: str | Path, queue_doc: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(queue_doc, sort_keys=False))
    tmp.replace(destination)


def update_item(
    path: str | Path,
    index: int | None = None,
    *,
    item_name: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    queue_doc = load_queue(path)
    if item_name is not None:
        matches = [
            item for item in queue_doc["items"] if str(item.get("name")) == item_name
        ]
        if len(matches) != 1:
            raise QueueError(
                f"queue item name must resolve exactly once: {item_name!r}"
            )
        item = matches[0]
    elif index is not None:
        item = queue_doc["items"][index]
    else:
        raise QueueError("update_item requires item_name or index")
    item.update({key: value for key, value in fields.items() if value is not None})
    write_queue(path, queue_doc)
    return item


def _launch_config(defaults: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return _merge_mapping(defaults.get("launch"), item.get("launch"))


def _launch_approved(defaults: dict[str, Any], item: dict[str, Any]) -> bool:
    launch = _launch_config(defaults, item)
    if not _as_bool(launch.get("requires_approval"), default=True):
        return True
    return _as_bool(launch.get("approved"), default=False)


def next_pending_index(queue_doc: dict[str, Any]) -> int | None:
    defaults = queue_doc.get("defaults") or {}
    for index, item in enumerate(queue_doc["items"]):
        if item.get("status") == "pending" and _launch_approved(defaults, item):
            return index
    return None


def promote_ready_blocked_items(queue_doc: dict[str, Any]) -> list[str]:
    """Promote blocked items only when their explicit dependency has passed."""

    defaults = queue_doc.get("defaults") or {}
    items = queue_doc["items"]
    items_by_name = {str(item["name"]): item for item in items}
    promoted: list[str] = []
    for item in items:
        if item.get("status") != "blocked":
            continue
        depends_on = item.get("depends_on")
        if not isinstance(depends_on, dict):
            continue
        dependency_name = str(depends_on["item"])
        dependency = items_by_name[dependency_name]
        if dependency.get("status") != "complete":
            continue
        requires_gate = _as_bool(depends_on.get("require_gate_pass"), default=False)
        gate_passed = _as_bool(dependency.get("gate_passed"), default=False)
        if requires_gate and not gate_passed:
            continue
        if not _launch_approved(defaults, item):
            continue
        item["status"] = "pending"
        item["dependency_item"] = dependency_name
        item["dependency_gate_passed"] = gate_passed
        item["dependency_promoted_at"] = utc_now()
        promoted.append(str(item["name"]))
    return promoted


def set_item_approval(
    path: str | Path,
    item_name: str,
    *,
    approved: bool,
    approved_by: str,
) -> dict[str, Any]:
    queue_doc = load_queue(path)
    matches = [item for item in queue_doc["items"] if str(item.get("name")) == str(item_name)]
    if not matches:
        raise QueueError(f"queue item not found: {item_name}")
    item = matches[0]
    launch = dict(item.get("launch") or {})
    launch["requires_approval"] = True
    launch["approved"] = bool(approved)
    launch["approved_by"] = str(approved_by)
    launch["approval_updated_at"] = utc_now()
    item["launch"] = launch
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


def _merge_mapping(*sources: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        if isinstance(source, dict):
            merged.update(source)
    return merged


def _resolve_path(value: str | Path, *, queue_path: Path, code_root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if code_root is not None:
        return code_root / path
    return queue_path.parent / path


def _optional_path(value: Any, *, queue_path: Path, code_root: Path | None = None) -> Path | None:
    if value is None:
        return None
    return _resolve_path(str(value), queue_path=queue_path, code_root=code_root)


def _resolve_script_path(value: str, *, queue_path: Path, code_root: Path, nla_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "external" and parts[1] == "natural_language_autoencoders":
        return code_root / path
    if parts and parts[0] in {"scripts", "configs", "docs", "tests"}:
        return code_root / path
    code_candidate = code_root / path
    if code_candidate.exists():
        return code_candidate
    return nla_root / path


def _as_cli_name(name: str) -> str:
    return "--" + name.replace("_", "-")


def _append_cli_arg(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(_as_cli_name(name))
        return
    if isinstance(value, (list, tuple)):
        command.append(_as_cli_name(name))
        command.extend(str(part) for part in value)
        return
    command.extend([_as_cli_name(name), str(value)])


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "none", "null", ""}:
        return False
    return default


def _resource_config(defaults: dict[str, Any], item: dict[str, Any]) -> dict[str, int]:
    raw = _merge_mapping(defaults.get("resources"), item.get("resources"))
    return {
        "actor_gpus": int(raw.get("actor_gpus", 1)),
        "critic_gpus": int(raw.get("critic_gpus", 1)),
        "rollout_gpus": int(raw.get("rollout_gpus", 1)),
        "actor_nodes": int(raw.get("actor_nodes", 1)),
        "critic_nodes": int(raw.get("critic_nodes", raw.get("actor_nodes", 1))),
        "min_actor_gpus": int(raw.get("min_actor_gpus", 1)),
    }


def _training_config(defaults: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return _merge_mapping(defaults.get("training"), item.get("training"))


def _critic_update_mode(
    training: dict[str, Any],
    runtime_env: dict[str, Any],
) -> str:
    """Resolve the critic optimizer contract without ambiguous env inheritance."""

    configured = training.get("critic_update_mode")
    mode = None if configured is None else str(configured).strip().lower()
    if mode is not None and mode not in {"frozen", "online"}:
        raise QueueError(
            "training.critic_update_mode must be either frozen or online"
        )

    freeze_raw = runtime_env.get("NLA_FREEZE_CRITIC_TRAIN")
    freeze_from_env: bool | None = None
    if freeze_raw is not None:
        if isinstance(freeze_raw, bool):
            freeze_from_env = freeze_raw
        else:
            freeze_text = str(freeze_raw).strip().lower()
            if freeze_text in {"1", "true", "yes", "on"}:
                freeze_from_env = True
            elif freeze_text in {"0", "false", "no", "off"}:
                freeze_from_env = False
            else:
                raise QueueError(
                    "env.NLA_FREEZE_CRITIC_TRAIN must be an explicit boolean"
                )

    if mode is None:
        if freeze_from_env is None:
            return "online"
        return "frozen" if freeze_from_env else "online"

    expected_freeze = mode == "frozen"
    if freeze_from_env is not None and freeze_from_env != expected_freeze:
        raise QueueError(
            "training.critic_update_mode conflicts with "
            "env.NLA_FREEZE_CRITIC_TRAIN"
        )
    return mode


def _training_bool(training: dict[str, Any], key: str, *, default: bool = False) -> bool:
    return _as_bool(training.get(key), default=default)


def _training_list(training: dict[str, Any], key: str) -> list[str]:
    value = training.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split() if part]
    if isinstance(value, list):
        return [str(part) for part in value]
    return [str(value)]


def _save_iterations_config(
    training: dict[str, Any],
    rollout: dict[str, Any],
) -> list[int] | None:
    raw = training.get("save_iterations")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise QueueError("training.save_iterations must be a non-empty list")
    try:
        values = [int(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise QueueError("training.save_iterations must contain integers") from exc
    if values != sorted(set(values)) or any(value <= 0 for value in values):
        raise QueueError(
            "training.save_iterations must be positive, unique, and strictly increasing"
        )
    num_rollout = _positive_int(
        rollout.get("num_rollout"),
        field="rollout.num_rollout",
    )
    if values[-1] != num_rollout:
        raise QueueError(
            "training.save_iterations must include the final rollout iteration "
            f"{num_rollout}"
        )
    if any(value > num_rollout for value in values):
        raise QueueError(
            "training.save_iterations cannot exceed rollout.num_rollout"
        )
    save_interval = training.get("save_interval")
    if save_interval not in (None, "", "none", "None", "null", "Null"):
        raise QueueError(
            "training.save_iterations cannot be combined with training.save_interval"
        )
    return values


def _drift_guard_config(training: dict[str, Any]) -> dict[str, Any]:
    raw = training.get("drift_guard")
    if raw is None:
        return {"enabled": False}
    if not isinstance(raw, dict):
        raise QueueError("training.drift_guard must be a mapping")
    enabled = _as_bool(raw.get("enabled"), default=False)
    if not enabled:
        return {"enabled": False}
    threshold = float(raw.get("max_logprob_abs_diff", 0.75))
    consecutive_steps = _positive_int(
        raw.get("consecutive_steps", 2),
        field="training.drift_guard.consecutive_steps",
    )
    if not math.isfinite(threshold) or threshold < 0.0:
        raise QueueError(
            "training.drift_guard.max_logprob_abs_diff must be finite and non-negative"
        )
    function_path = str(
        raw.get("function_path", "nla.train_guard.check_train_metrics")
    ).strip()
    if not function_path:
        raise QueueError("training.drift_guard.function_path must not be empty")
    return {
        "enabled": True,
        "function_path": function_path,
        "max_logprob_abs_diff": threshold,
        "consecutive_steps": consecutive_steps,
        "metric": str(
            raw.get("metric", "train/train_rollout_logprob_abs_diff")
        ),
    }


def _metric_guard_rules(training: dict[str, Any], rollout: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    actor_rules: list[dict[str, Any]] = []
    drift = _drift_guard_config(training)
    if drift["enabled"]:
        actor_rules.append(
            {
                "metric": drift["metric"],
                "comparison": "max",
                "threshold": drift["max_logprob_abs_diff"],
                "consecutive_steps": drift["consecutive_steps"],
                "role_prefixes": ["actor"],
            }
        )
    raw_actor_rules = training.get("guard_rules") or []
    raw_rollout_rules = rollout.get("guard_rules") or []
    if not isinstance(raw_actor_rules, list) or not isinstance(raw_rollout_rules, list):
        raise QueueError("training.guard_rules and rollout.guard_rules must be lists")
    actor_rules.extend(dict(rule) for rule in raw_actor_rules)
    rollout_rules = [dict(rule) for rule in raw_rollout_rules]
    for label, rules in (("training.guard_rules", actor_rules), ("rollout.guard_rules", rollout_rules)):
        for rule in rules:
            if not rule.get("metric"):
                raise QueueError(f"{label} entries require metric")
            comparison = str(rule.get("comparison", "max"))
            if comparison not in {"max", "min", "increasing"}:
                raise QueueError(f"{label} comparison must be max, min, or increasing")
            _positive_int(rule.get("consecutive_steps", 1), field=f"{label}.consecutive_steps")
    return {"actor": actor_rules, "rollout": rollout_rules}


def _preregistration_config(
    defaults: dict[str, Any],
    item: dict[str, Any],
    *,
    queue_path: Path,
    code_root: Path,
    training: dict[str, Any],
    rollout: dict[str, Any],
    runtime_env: dict[str, Any],
    metric_guard_rules: dict[str, list[dict[str, Any]]],
    roundtrip_configs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    raw = _merge_mapping(
        defaults.get("preregistration"), item.get("preregistration")
    )
    if not raw:
        return None
    if raw.get("schema_version") != "nano_rl_preregistration.v1":
        raise QueueError(
            "preregistration.schema_version must be nano_rl_preregistration.v1"
        )
    phase = str(raw.get("phase") or "")
    if phase not in {"stability_probe", "confirmatory_train"}:
        raise QueueError(
            "preregistration.phase must be stability_probe or confirmatory_train"
        )
    if str(raw.get("selection_split") or "") != "validation":
        raise QueueError("preregistration.selection_split must be validation")
    if str(raw.get("test_policy") or "") != "sealed":
        raise QueueError("preregistration.test_policy must be sealed")

    if raw.get("registered_seed") is None:
        raise QueueError("preregistration.registered_seed is required")
    rollout_seed_value = rollout.get("seed", rollout.get("rollout_seed"))
    if rollout_seed_value is None:
        raise QueueError("preregistered runs require rollout.seed")
    registered_seed = int(raw["registered_seed"])
    rollout_seed = int(rollout_seed_value)
    if registered_seed != rollout_seed:
        raise QueueError(
            "preregistration.registered_seed must match rollout.seed: "
            f"{registered_seed} != {rollout_seed}"
        )

    allowed_eval_splits = [str(value) for value in raw.get("allowed_eval_splits", [])]
    if allowed_eval_splits != ["validation"]:
        raise QueueError(
            "preregistration.allowed_eval_splits must be exactly [validation] before selection lock"
        )
    if not roundtrip_configs:
        raise QueueError("preregistered runs require post_eval.roundtrip")
    for config in roundtrip_configs:
        eval_splits = [str(value) for value in config.get("eval_splits", [])]
        if not eval_splits:
            raise QueueError(
                "preregistered round-trip evals must declare eval_splits explicitly"
            )
        forbidden = sorted(set(eval_splits) - set(allowed_eval_splits))
        if forbidden:
            raise QueueError(
                "preregistered test split is sealed; forbidden eval_splits="
                f"{forbidden}"
            )

    registration_path = _resolve_path(
        str(raw.get("registration_path") or ""),
        queue_path=queue_path,
        code_root=code_root,
    )
    if not registration_path.is_file():
        raise QueueError(
            f"preregistration.registration_path is missing: {registration_path}"
        )
    registration_sha256 = hashlib.sha256(registration_path.read_bytes()).hexdigest()
    expected_registration_sha256 = str(
        raw.get("registration_sha256") or ""
    ).lower()
    if not expected_registration_sha256:
        raise QueueError("preregistration.registration_sha256 is required")
    if expected_registration_sha256 != registration_sha256:
        raise QueueError(
            "preregistration sha256 mismatch: "
            f"expected={expected_registration_sha256} actual={registration_sha256}"
        )

    artifact_names = [
        "content_family_manifest",
        "content_family_coverage",
        "sft_baseline_report",
        "family_seal_report",
        "kernel_compatibility_report",
    ]
    if phase == "confirmatory_train":
        artifact_names.append("power_report")
    artifacts: dict[str, dict[str, str]] = {}
    for name in artifact_names:
        value = raw.get(name)
        if not value:
            raise QueueError(f"preregistration.{name} is required")
        path = _resolve_path(str(value), queue_path=queue_path, code_root=code_root)
        if not path.is_file():
            raise QueueError(f"preregistration.{name} is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = str(raw.get(f"{name}_sha256") or "").lower()
        if not expected:
            raise QueueError(f"preregistration.{name}_sha256 is required")
        if expected != digest:
            raise QueueError(
                f"preregistration.{name}_sha256 mismatch: expected={expected} actual={digest}"
            )
        artifacts[name] = {"path": str(path), "sha256": digest}
        if name.endswith("_report"):
            try:
                report = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise QueueError(
                    f"preregistration.{name} must be valid JSON: {path}"
                ) from exc
            if name in {
                "family_seal_report",
                "kernel_compatibility_report",
                "power_report",
            } and report.get("passed") is not True:
                raise QueueError(f"preregistration.{name} did not pass: {path}")

    if str(raw.get("guard_failure_action") or "") != "abort":
        raise QueueError("preregistration.guard_failure_action must be abort")
    for name in ("NLA_ASSERT_PACKED_EQUIV", "NLA_ASSERT_ACTOR_PACKED_EQUIV"):
        if not _as_bool(runtime_env.get(name), default=False):
            raise QueueError(f"preregistered runs require {name}=1")
    kernel_mode = str(runtime_env.get("NLA_TRAIN_MAMBA_KERNEL_MODE") or "")
    if kernel_mode not in {"torch", "unfused_torch_conv"}:
        raise QueueError(
            "preregistered runs require an explicit stable "
            "NLA_TRAIN_MAMBA_KERNEL_MODE"
        )
    actor_metrics = {str(rule["metric"]) for rule in metric_guard_rules["actor"]}
    rollout_metrics = {str(rule["metric"]) for rule in metric_guard_rules["rollout"]}
    required_actor = {
        str(value) for value in raw.get("required_actor_guard_metrics", [])
    }
    required_rollout = {
        str(value) for value in raw.get("required_rollout_guard_metrics", [])
    }
    if not required_actor or not required_rollout:
        raise QueueError(
            "preregistration requires non-empty actor and rollout guard metric declarations"
        )
    missing_actor = sorted(required_actor - actor_metrics)
    missing_rollout = sorted(required_rollout - rollout_metrics)
    if missing_actor or missing_rollout:
        raise QueueError(
            "preregistered guard metrics are not enforced by the resolved queue: "
            f"actor={missing_actor} rollout={missing_rollout}"
        )

    save_iterations = _save_iterations_config(training, rollout)
    declared_checkpoints = [int(value) for value in raw.get("checkpoint_iterations", [])]
    if not declared_checkpoints:
        raise QueueError("preregistration.checkpoint_iterations must not be empty")
    if save_iterations != declared_checkpoints:
        raise QueueError(
            "preregistration.checkpoint_iterations must exactly match "
            f"training.save_iterations: {declared_checkpoints} != {save_iterations}"
        )

    primary_endpoint = str(raw.get("primary_endpoint") or "").strip()
    secondary_endpoints = [
        str(value).strip() for value in raw.get("secondary_endpoints", [])
    ]
    if not primary_endpoint or not secondary_endpoints or not all(secondary_endpoints):
        raise QueueError("preregistration endpoints must be predeclared")

    guard_policy = {
        "actor_rules": metric_guard_rules["actor"],
        "rollout_rules": metric_guard_rules["rollout"],
        "training": {
            key: training.get(key)
            for key in (
                "actor_lr",
                "kl_loss_type",
                "kl_loss_coef",
                "actor_micro_batch",
                "clip_grad",
            )
        },
        "rollout": {
            key: rollout.get(key)
            for key in (
                "global_batch_size",
                "n_samples_per_prompt",
                "max_response_len",
                "max_context_len",
            )
        },
        "reward": {
            key: runtime_env.get(key)
            for key in (
                "NLA_FAILED_EXTRACTION_REWARD",
                "NLA_LOG_MSE_REWARD",
            )
        },
        "runtime_checks": {
            key: runtime_env.get(key)
            for key in (
                "NLA_ASSERT_PACKED_EQUIV",
                "NLA_ASSERT_ACTOR_PACKED_EQUIV",
                "NLA_ACTOR_PACKED_EQUIV_RTOL",
                "NLA_ACTOR_PACKED_EQUIV_ATOL",
                "NLA_TRAIN_MAMBA_KERNEL_MODE",
            )
        },
        "failure_action": "abort",
    }
    guard_policy_sha256 = _canonical_sha256(guard_policy)
    expected_guard_sha256 = str(raw.get("guard_policy_sha256") or "").lower()
    if phase == "confirmatory_train" and not expected_guard_sha256:
        raise QueueError(
            "confirmatory runs require preregistration.guard_policy_sha256"
        )
    if expected_guard_sha256 and expected_guard_sha256 != guard_policy_sha256:
        raise QueueError(
            "preregistration guard policy sha256 mismatch: "
            f"expected={expected_guard_sha256} actual={guard_policy_sha256}"
        )

    return {
        "schema_version": "nano_rl_preregistration.v1",
        "phase": phase,
        "registration_path": str(registration_path),
        "registration_sha256": registration_sha256,
        "registered_seed": registered_seed,
        "selection_split": "validation",
        "allowed_eval_splits": allowed_eval_splits,
        "test_policy": "sealed",
        "checkpoint_iterations": declared_checkpoints,
        "primary_endpoint": primary_endpoint,
        "secondary_endpoints": secondary_endpoints,
        "artifacts": artifacts,
        "guard_policy": guard_policy,
        "guard_policy_sha256": guard_policy_sha256,
        "mamba_kernel_mode": kernel_mode,
    }


def _source_policy(defaults: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    policy = _merge_mapping(defaults.get("source"), item.get("source"))
    # Source capture is universal. Optional policy fields only add stricter
    # expectations; they no longer turn provenance collection on or off.
    return policy or {"capture": True, "allow_missing_during_render": True}


def _required_gate_reports(
    defaults: dict[str, Any],
    item: dict[str, Any],
    *,
    queue_path: Path,
) -> list[dict[str, Any]]:
    raw_reports: list[Any] = []
    for source in (defaults.get("required_gate_reports"), item.get("required_gate_reports")):
        raw_reports.extend(_as_list(source))
    legacy = item.get("validity_gate_report")
    if legacy:
        raw_reports.append({"path": legacy, "field": "passed", "expected": True})
    reports: list[dict[str, Any]] = []
    for raw in raw_reports:
        value = {"path": raw} if isinstance(raw, (str, Path)) else dict(raw)
        if not value.get("path"):
            raise QueueError("required_gate_reports entries require path")
        reports.append(
            {
                "path": str(_resolve_path(str(value["path"]), queue_path=queue_path)),
                "field": str(value.get("field", "passed")),
                "expected": value.get("expected", True),
            }
        )
    return reports


def _runtime_contracts_config(
    defaults: dict[str, Any],
    item: dict[str, Any],
    *,
    queue_path: Path,
) -> list[dict[str, Any]]:
    raw_contracts: list[Any] = []
    for source in (defaults.get("runtime_contracts"), item.get("runtime_contracts")):
        raw_contracts.extend(_as_list(source))
    contracts: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_contracts):
        value = dict(raw)
        if not value.get("path"):
            raise QueueError(f"runtime_contracts[{index}] requires path")
        contract = {
            "name": str(value.get("name") or f"runtime_contract_{index}"),
            "path": str(_resolve_path(str(value["path"]), queue_path=queue_path)),
            "sha256": str(value.get("sha256") or "").lower(),
            "contains": [str(marker) for marker in _as_list(value.get("contains"))],
            "forbids": [str(marker) for marker in _as_list(value.get("forbids"))],
        }
        if not contract["sha256"] and not contract["contains"] and not contract["forbids"]:
            raise QueueError(f"runtime contract {contract['name']!r} has no assertions")
        contracts.append(contract)
    return contracts


def _validate_runtime_contracts(contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for contract in contracts:
        name = str(contract["name"])
        path = Path(str(contract["path"]))
        if not path.is_file():
            raise QueueError(f"runtime contract {name!r} file is missing: {path}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        expected_digest = str(contract.get("sha256") or "").lower()
        if expected_digest and digest != expected_digest:
            raise QueueError(
                f"runtime contract {name!r} sha256 mismatch: "
                f"expected={expected_digest} actual={digest} path={path}"
            )
        text_payload = payload.decode("utf-8", errors="replace")
        missing_markers = [
            marker for marker in contract.get("contains") or [] if marker not in text_payload
        ]
        if missing_markers:
            raise QueueError(
                f"runtime contract {name!r} missing required markers: {missing_markers}"
            )
        forbidden_markers = [
            marker for marker in contract.get("forbids") or [] if marker in text_payload
        ]
        if forbidden_markers:
            raise QueueError(
                f"runtime contract {name!r} contains forbidden markers: {forbidden_markers}"
            )
        reports.append(
            {
                "name": name,
                "path": str(path),
                "sha256": digest,
                "bytes": len(payload),
                "contains": list(contract.get("contains") or []),
                "forbids": list(contract.get("forbids") or []),
            }
        )
    return reports


def _checkpoint_retention_config(
    defaults: dict[str, Any],
    item: dict[str, Any],
    *,
    training: dict[str, Any],
    rollout: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    raw = _merge_mapping(defaults.get("checkpoint_retention"), item.get("checkpoint_retention"))
    if not raw or not _as_bool(raw.get("enabled"), default=True):
        return {"enabled": False}
    keep_iterations = sorted({int(value) for value in raw.get("keep_iterations", [])})
    if not keep_iterations:
        raise QueueError("checkpoint_retention.keep_iterations must not be empty")
    num_rollout = _positive_int(rollout.get("num_rollout"), field="rollout.num_rollout")
    save_iterations = _save_iterations_config(training, rollout)
    save_interval = None
    if save_iterations is None:
        save_interval = _positive_int(
            training.get("save_interval"),
            field="training.save_interval",
        )
    if num_rollout not in keep_iterations:
        raise QueueError("checkpoint retention must keep the final rollout iteration")
    if save_iterations is not None:
        invalid = [value for value in keep_iterations if value not in save_iterations]
        expected_saved = save_iterations
    else:
        assert save_interval is not None
        invalid = [
            value
            for value in keep_iterations
            if value <= 0 or value > num_rollout or value % save_interval
        ]
        expected_saved = list(range(save_interval, num_rollout + 1, save_interval))
    if invalid:
        raise QueueError(
            "checkpoint_retention.keep_iterations must name configured saved iterations; "
            f"invalid={invalid}, expected_saved={expected_saved}"
        )
    max_transient_checkpoints = int(raw.get("max_transient_checkpoints", len(expected_saved)))
    if len(expected_saved) > max_transient_checkpoints:
        raise QueueError(
            "save_interval would exceed checkpoint_retention.max_transient_checkpoints: "
            f"{len(expected_saved)} > {max_transient_checkpoints}"
        )
    return {
        "enabled": True,
        "checkpoint_root": str(run_dir / str(raw.get("relative_root", "actor"))),
        "keep_iterations": keep_iterations,
        "manifest_path": str(run_dir / str(raw.get("manifest", "checkpoint_retention_manifest.json"))),
        "apply": _as_bool(raw.get("apply"), default=True),
        "save_interval": save_interval,
        "save_iterations": save_iterations,
        "expected_saved_iterations": expected_saved,
        "max_transient_checkpoints": max_transient_checkpoints,
    }


def _validate_training_runtime_config(
    training: dict[str, Any],
    *,
    sglang_mode: str | None = None,
    critic_update_mode: str | None = None,
) -> None:
    qkv_format = str(training.get("qkv_format", "thd")).strip().lower()
    if qkv_format not in {"thd", "bshd"}:
        raise QueueError("training.qkv_format must be thd or bshd")
    if _training_bool(training, "async_training") and _training_bool(training, "colocate"):
        raise QueueError("training.async_training cannot be combined with training.colocate; Miles train_async asserts this")

    ref_placement = str(training.get("ref_log_probs_placement", "actor")).strip().lower()
    if ref_placement not in {"actor", "actor_cpu_offload"}:
        raise QueueError(
            "training.ref_log_probs_placement=critic is not supported by the current Miles FSDP runtime; "
            "ref log-probs are computed by the actor group via its local ref_model"
        )

    if _training_bool(training, "colocate") and sglang_mode == "external":
        raise QueueError(
            "training.colocate requires Miles-managed/internal rollout engines; "
            "the current Nano queue uses managed external SGLang"
        )

    kl_loss_type = str(training.get("kl_loss_type", "k1")).strip()
    if kl_loss_type not in {"k1", "k2", "k3", "low_var_kl"}:
        raise QueueError(
            "training.kl_loss_type must be one of k1, k2, k3, or low_var_kl"
        )

    resolved_critic_mode = critic_update_mode or _critic_update_mode(training, {})
    if resolved_critic_mode == "online":
        try:
            critic_lr = float(training.get("critic_lr", "1e-5"))
        except (TypeError, ValueError) as exc:
            raise QueueError(
                "training.critic_lr must be numeric for an online critic"
            ) from exc
        if critic_lr <= 0:
            raise QueueError(
                "training.critic_lr must be positive for an online critic"
            )
        try:
            min_retained = float(
                training.get("min_critic_retained_fraction", 0.95)
            )
        except (TypeError, ValueError) as exc:
            raise QueueError(
                "training.min_critic_retained_fraction must be numeric"
            ) from exc
        if not 0.0 <= min_retained <= 1.0:
            raise QueueError(
                "training.min_critic_retained_fraction must be between 0 and 1"
            )


def _validate_live_sync_runtime(
    *,
    trainer_python: str,
    sglang_service: dict[str, Any],
    runtime_env: dict[str, Any],
) -> None:
    if sglang_service.get("mode") != "external" or not sglang_service.get("managed"):
        return
    if _as_bool(runtime_env.get("NLA_SKIP_ROLLOUT_WEIGHT_SYNC")):
        return
    sglang_python = sglang_service.get("python")
    if not sglang_python:
        raise QueueError(
            "managed external live weight sync requires sglang.python so the "
            "trainer and SGLang NCCL runtimes can be validated"
        )
    trainer_runtime = os.path.normpath(str(trainer_python))
    sglang_runtime = os.path.normpath(str(sglang_python))
    if trainer_runtime != sglang_runtime:
        raise QueueError(
            "managed external live weight sync requires one unified Python runtime "
            "for trainer and SGLang because they join the same NCCL process group: "
            f"trainer={trainer_runtime!r}, sglang={sglang_runtime!r}"
        )


def _rollout_config(defaults: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return _merge_mapping(defaults.get("rollout"), item.get("rollout"))


def _positive_int(value: Any, *, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise QueueError(f"{field} must be positive, got {parsed}")
    return parsed


def _nonnegative_int(value: Any, *, field: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise QueueError(f"{field} must be nonnegative, got {parsed}")
    return parsed


def _optional_positive_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field=field)


def _rollout_batch_plan(rollout: dict[str, Any]) -> dict[str, Any]:
    """Resolve rollout batch semantics before rendering Miles CLI args.

    Miles/NLA RL is safest in the Qwen-tested shape where one rollout is one
    optimizer step: rollout_batch_size prompts times n samples per prompt equals
    global_batch_size. Some historical Nano systems smokes intentionally used
    smaller global batches, so strict matching is opt-in per queue.
    """
    rollout_batch_size = _positive_int(
        rollout.get("rollout_batch_size", 128),
        field="rollout.rollout_batch_size",
    )
    n_samples_per_prompt = _positive_int(
        rollout.get("n_samples_per_prompt", 4),
        field="rollout.n_samples_per_prompt",
    )
    generated_samples = rollout_batch_size * n_samples_per_prompt
    global_batch_size = _positive_int(
        rollout.get("global_batch_size", generated_samples),
        field="rollout.global_batch_size",
    )
    require_match = _as_bool(
        rollout.get(
            "require_global_batch_match",
            rollout.get("require_global_batch_equals_rollout", False),
        )
    )
    matches = global_batch_size == generated_samples
    if require_match and not matches:
        raise QueueError(
            "rollout.global_batch_size must equal "
            "rollout.rollout_batch_size * rollout.n_samples_per_prompt "
            f"when require_global_batch_match is true: {global_batch_size} != "
            f"{rollout_batch_size} * {n_samples_per_prompt} ({generated_samples})"
        )
    return {
        "rollout_batch_size": rollout_batch_size,
        "n_samples_per_prompt": n_samples_per_prompt,
        "generated_samples": generated_samples,
        "global_batch_size": global_batch_size,
        "global_batch_matches_rollout": matches,
        "require_global_batch_match": require_match,
    }


def _validate_actor_batch_plan(
    *,
    generated_samples: int,
    global_batch_size: int,
    actor_gpus: int,
    actor_micro_batch: int,
    required: bool,
) -> dict[str, Any]:
    generated_samples = _positive_int(
        generated_samples,
        field="rollout.generated_samples",
    )
    global_batch_size = _positive_int(
        global_batch_size,
        field="rollout.global_batch_size",
    )
    actor_gpus = _positive_int(actor_gpus, field="resources.actor_gpus")
    actor_micro_batch = _positive_int(
        actor_micro_batch,
        field="training.actor_micro_batch",
    )
    warnings: list[str] = []
    if generated_samples != global_batch_size:
        message = (
            f"generated_samples={generated_samples} must equal "
            f"global_batch_size={global_batch_size}"
        )
        if required:
            raise QueueError(message)
        warnings.append(message)

    divisor = actor_gpus * actor_micro_batch
    remainder = global_batch_size % divisor
    if remainder:
        message = (
            f"global_batch_size={global_batch_size} must be divisible by "
            f"actor_gpus={actor_gpus} * actor_micro_batch={actor_micro_batch}"
        )
        if required:
            raise QueueError(message)
        warnings.append(message)

    plan: dict[str, Any] = {
        "actor_gpus": actor_gpus,
        "actor_micro_batch": actor_micro_batch,
        "actor_batch_divisor": divisor,
        "samples_per_actor": global_batch_size // actor_gpus,
        "effective_trained_samples": global_batch_size - remainder,
        "dropped_or_truncated_samples": remainder,
        "require_exact_actor_batch": required,
    }
    if warnings:
        plan["warning"] = "; ".join(warnings)
    return plan


def _validate_critic_batch_plan(
    *,
    global_batch_size: int,
    critic_gpus: int,
    critic_micro_batch: int,
    required: bool,
) -> dict[str, Any]:
    """Describe critic retention when every generated sample is usable.

    Runtime parse/filter failures can reduce the usable count further. The NLA
    critic repartitioner records that dynamic loss separately and enforces
    ``min_critic_retained_fraction``. This static plan catches topology and
    microbatch alignment mistakes before a costly launch.
    """
    global_batch_size = _positive_int(
        global_batch_size,
        field="rollout.global_batch_size",
    )
    critic_gpus = _positive_int(critic_gpus, field="resources.critic_gpus")
    critic_micro_batch = _positive_int(
        critic_micro_batch,
        field="training.actor_micro_batch",
    )
    divisor = critic_gpus * critic_micro_batch
    remainder = global_batch_size % divisor
    if remainder and required:
        raise QueueError(
            f"global_batch_size={global_batch_size} must be divisible by "
            f"critic_gpus={critic_gpus} * shared_micro_batch={critic_micro_batch}"
        )

    retained = global_batch_size - remainder
    plan: dict[str, Any] = {
        "critic_gpus": critic_gpus,
        "critic_micro_batch": critic_micro_batch,
        "critic_batch_divisor": divisor,
        "full_usable_critic_retained_samples": retained,
        "full_usable_critic_alignment_drop": remainder,
        "full_usable_critic_retained_fraction": retained / global_batch_size,
        "require_exact_critic_batch": required,
    }
    if remainder:
        plan["critic_batch_warning"] = (
            f"full usable critic batch drops {remainder}/{global_batch_size} "
            "samples for data-parallel microbatch alignment"
        )
    return plan


def _wandb_config(defaults: dict[str, Any], item: dict[str, Any], *, run_dir: Path, queue_path: Path) -> dict[str, str]:
    raw = _merge_mapping(defaults.get("wandb"), item.get("wandb"))
    directory = raw.get("dir", raw.get("directory", run_dir / "wandb"))
    return {
        "mode": str(raw.get("mode", "offline")),
        "dir": str(_resolve_path(directory, queue_path=queue_path)),
        "project": str(raw.get("project", "nano30b-nla-pilot")),
        "group": str(raw.get("group", "nano-rl")),
        "run_id": str(raw.get("run_id", "")),
    }


def _extra_args(defaults: dict[str, Any], item: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for source in (defaults.get("extra_args"), item.get("extra_args")):
        if isinstance(source, list):
            args.extend(str(part) for part in source)
    return args


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _split_host_port(addr: str) -> tuple[str, int]:
    if addr.startswith("["):
        host, port = addr.rsplit("]:", 1)
        return host[1:], int(port)
    host, port = addr.rsplit(":", 1)
    return host, int(port)


def _normalize_start_command(command: Any) -> list[str]:
    if isinstance(command, str):
        return ["bash", "-lc", command]
    if isinstance(command, dict):
        if "shell" in command:
            return ["bash", "-lc", str(command["shell"])]
        if "argv" in command:
            command = command["argv"]
    if isinstance(command, list) and command:
        return [str(part) for part in command]
    raise QueueError(f"invalid sglang start command: {command!r}")


def _set_command_option(command: list[str], option: str, value: int | str) -> list[str]:
    rewritten = list(command)
    replacement = str(value)
    index = 0
    replaced = False
    while True:
        try:
            option_index = rewritten.index(option, index)
        except ValueError:
            break
        value_index = option_index + 1
        if value_index >= len(rewritten):
            rewritten.append(replacement)
            replaced = True
            break
        rewritten[value_index] = replacement
        replaced = True
        index = value_index + 1
    if not replaced:
        rewritten.extend([option, replacement])
    return rewritten


def _sglang_model_staging_config(raw: dict[str, Any], start_commands: list[list[str]]) -> dict[str, Any]:
    staging_raw = raw.get("model_staging") or raw.get("model_stage") or {}
    if not isinstance(staging_raw, dict):
        raise QueueError("sglang.model_staging must be a mapping when provided")
    if not _as_bool(staging_raw.get("enabled"), default=False):
        return {"enabled": False}

    source = staging_raw.get("source_model_path")
    if source is None:
        for command in start_commands:
            source = _command_option(command, "--model-path")
            if source is not None:
                break
    target = staging_raw.get("target_path") or staging_raw.get("target_model_path")
    if source is None:
        raise QueueError("sglang.model_staging.enabled=true requires source_model_path or a start command --model-path")
    if target is None:
        raise QueueError("sglang.model_staging.enabled=true requires target_path")

    release_raw = staging_raw.get("release_after_health") or {}
    if not isinstance(release_raw, dict):
        raise QueueError("sglang.model_staging.release_after_health must be a mapping")
    release_enabled = _as_bool(release_raw.get("enabled"), default=False)
    release_globs = [str(pattern) for pattern in _as_list(release_raw.get("globs"))]
    if release_enabled and not release_globs:
        raise QueueError(
            "sglang.model_staging.release_after_health.enabled=true requires globs"
        )

    return {
        "enabled": True,
        "source_model_path": str(source),
        "target_path": str(target),
        "reuse_existing": _as_bool(staging_raw.get("reuse_existing"), default=True),
        "clean": _as_bool(staging_raw.get("clean"), default=True),
        "copy_workers": _optional_positive_int(
            staging_raw.get("copy_workers", 1),
            field="sglang.model_staging.copy_workers",
        ),
        "copy_chunk_bytes": _nonnegative_int(
            staging_raw.get("copy_chunk_bytes", 0),
            field="sglang.model_staging.copy_chunk_bytes",
        ),
        "release_after_health": {
            "enabled": release_enabled,
            "globs": release_globs,
        },
    }


def _input_staging_config(
    defaults: dict[str, Any],
    item: dict[str, Any],
    *,
    queue_path: Path,
) -> list[dict[str, Any]]:
    raw = item.get("input_staging", defaults.get("input_staging", []))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise QueueError("input_staging must be a list of staging entries")

    entries: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_env_keys: set[str] = set()
    seen_targets: set[str] = set()
    for index, value in enumerate(raw):
        field = f"input_staging[{index}]"
        if not isinstance(value, dict):
            raise QueueError(f"{field} must be a mapping")
        if not _as_bool(value.get("enabled"), default=True):
            continue

        env_key = str(value.get("env_key", "")).strip()
        if not env_key:
            raise QueueError(f"{field}.env_key must be non-empty")
        name = str(value.get("name", env_key.lower())).strip()
        if not name:
            raise QueueError(f"{field}.name must be non-empty")
        target_raw = value.get("target_path")
        if target_raw is None or not str(target_raw).strip():
            raise QueueError(f"{field}.target_path must be non-empty")
        target_path = str(_resolve_path(target_raw, queue_path=queue_path))

        if name in seen_names:
            raise QueueError(f"duplicate input_staging name: {name}")
        if env_key in seen_env_keys:
            raise QueueError(f"duplicate input_staging env_key: {env_key}")
        if target_path in seen_targets:
            raise QueueError(f"duplicate input_staging target_path: {target_path}")
        seen_names.add(name)
        seen_env_keys.add(env_key)
        seen_targets.add(target_path)

        source_raw = value.get("source_path")
        entries.append(
            {
                "name": name,
                "env_key": env_key,
                "source_path": (
                    None
                    if source_raw is None
                    else str(_resolve_path(source_raw, queue_path=queue_path))
                ),
                "target_path": target_path,
                "reuse_existing": _as_bool(
                    value.get("reuse_existing"), default=True
                ),
                "clean": _as_bool(value.get("clean"), default=True),
                "copy_workers": _positive_int(
                    value.get("copy_workers", 1),
                    field=f"{field}.copy_workers",
                ),
                "copy_chunk_bytes": _nonnegative_int(
                    value.get("copy_chunk_bytes", 0),
                    field=f"{field}.copy_chunk_bytes",
                ),
            }
        )
    return entries


def _sglang_service_config(defaults: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    raw = _merge_mapping(defaults.get("sglang"), item.get("sglang"))
    mode = str(raw.get("mode", "internal")).lower()
    if mode not in {"internal", "external"}:
        raise QueueError(f"sglang.mode must be internal or external, got {mode!r}")

    engine_addrs = [str(addr) for addr in _as_list(raw.get("engine_addrs"))]
    router_addr = raw.get("router_addr")
    router_ip = raw.get("router_ip")
    router_port = raw.get("router_port")
    if router_addr is not None:
        router_ip, router_port = _split_host_port(str(router_addr))

    managed = bool(raw.get("managed", False))
    start_commands = [_normalize_start_command(command) for command in _as_list(raw.get("start_commands"))]
    tensor_parallel_size = _optional_positive_int(
        raw.get("tensor_parallel_size", raw.get("tp_size")),
        field="sglang.tensor_parallel_size",
    )
    base_gpu_id = _optional_positive_int(raw.get("base_gpu_id"), field="sglang.base_gpu_id")
    rollout_num_gpus_per_engine = _optional_positive_int(
        raw.get("rollout_num_gpus_per_engine", raw.get("gpus_per_engine", tensor_parallel_size)),
        field="sglang.rollout_num_gpus_per_engine",
    )
    if tensor_parallel_size is not None and rollout_num_gpus_per_engine is not None:
        if tensor_parallel_size != rollout_num_gpus_per_engine:
            raise QueueError(
                "sglang.tensor_parallel_size must match "
                "sglang.rollout_num_gpus_per_engine for one GPU-partitioned engine: "
                f"{tensor_parallel_size} != {rollout_num_gpus_per_engine}"
            )
    if tensor_parallel_size is not None:
        start_commands = [_set_command_option(command, "--tp-size", tensor_parallel_size) for command in start_commands]
    if base_gpu_id is not None:
        start_commands = [_set_command_option(command, "--base-gpu-id", base_gpu_id) for command in start_commands]
    if managed and not start_commands:
        raise QueueError("sglang.managed=true requires at least one start_commands entry")
    if mode == "external" and not engine_addrs:
        raise QueueError("sglang.mode=external requires engine_addrs")

    return {
        "mode": mode,
        "python": None if raw.get("python") is None else str(raw["python"]),
        "engine_addrs": engine_addrs,
        "router_ip": None if router_ip is None else str(router_ip),
        "router_port": None if router_port is None else int(router_port),
        "managed": managed,
        "start_commands": start_commands,
        "health_urls": [str(url) for url in _as_list(raw.get("health_urls"))],
        "timeout_seconds": float(raw.get("timeout_seconds", 900)),
        "poll_seconds": float(raw.get("poll_seconds", 2)),
        "terminate_on_exit": bool(raw.get("terminate_on_exit", managed)),
        "cwd": None if raw.get("cwd") is None else str(raw["cwd"]),
        "env": {str(k): str(v) for k, v in (raw.get("env") or {}).items()},
        "model_staging": _sglang_model_staging_config(raw, start_commands),
        "tensor_parallel_size": tensor_parallel_size,
        "base_gpu_id": base_gpu_id,
        "rollout_num_gpus_per_engine": rollout_num_gpus_per_engine,
    }


def _validate_sglang_resource_layout(resources: dict[str, int], service: dict[str, Any]) -> None:
    if service.get("mode") != "external":
        return

    rollout_gpus_per_engine = service.get("rollout_num_gpus_per_engine")
    engine_count = max(len(service.get("engine_addrs") or []), len(service.get("start_commands") or []), 1)
    if rollout_gpus_per_engine is not None:
        expected_rollout_gpus = int(rollout_gpus_per_engine) * engine_count
        if expected_rollout_gpus != resources["rollout_gpus"]:
            raise QueueError(
                "sglang rollout GPU allocation mismatch: "
                f"resources.rollout_gpus={resources['rollout_gpus']} but "
                f"sglang.rollout_num_gpus_per_engine={rollout_gpus_per_engine} "
                f"* engines={engine_count} gives {expected_rollout_gpus}"
            )

    base_gpu_id = service.get("base_gpu_id")
    if base_gpu_id is not None:
        actor_total_gpus = resources["actor_gpus"] * resources["actor_nodes"]
        if int(base_gpu_id) != actor_total_gpus:
            raise QueueError(
                "sglang.base_gpu_id must start immediately after actor GPUs for the "
                f"static Nano RL layout: expected {actor_total_gpus}, got {base_gpu_id}"
            )


def _nonzero_number(value: Any) -> bool:
    try:
        return float(str(value)) != 0.0
    except (TypeError, ValueError):
        return False


def _command_option(command: list[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(command):
        return None
    return command[value_index]


def _replace_command_option(
    command: list[str],
    option: str,
    *,
    old_value: str,
    new_value: str,
) -> tuple[list[str], int]:
    replaced = 0
    rewritten = list(command)
    index = 0
    while True:
        try:
            option_index = rewritten.index(option, index)
        except ValueError:
            break
        value_index = option_index + 1
        if value_index < len(rewritten) and rewritten[value_index] == old_value:
            rewritten[value_index] = new_value
            replaced += 1
        index = value_index + 1
    return rewritten, replaced


def preflight_missing_paths(spec: dict[str, Any]) -> list[dict[str, str]]:
    """Return required local filesystem paths missing before a run launches."""
    env = spec["env"]
    checks: list[tuple[str, str | None]] = [
        ("TRAIN_ENTRYPOINT", env.get("TRAIN_ENTRYPOINT")),
        ("rl_script", spec["command"][1] if len(spec.get("command", [])) > 1 else None),
        ("RL_PARQUET", env.get("RL_PARQUET")),
        ("INSTRUCT_MODEL", env.get("INSTRUCT_MODEL")),
        ("ACTOR_SFT_CKPT", env.get("ACTOR_SFT_CKPT")),
        ("ACTOR_LOAD_CKPT", env.get("ACTOR_LOAD_CKPT")),
        ("ACTOR_SIDECAR_SOURCE", env.get("ACTOR_SIDECAR_SOURCE")),
        ("CRITIC_SL_CKPT", env.get("CRITIC_SL_CKPT")),
    ]
    if _nonzero_number(env.get("KL_LOSS_COEF")):
        checks.append(("ACTOR_REF_CKPT", env.get("ACTOR_REF_CKPT")))
    service = spec.get("sglang_service") or {}
    if service.get("mode") == "external" and service.get("managed"):
        staging = service.get("model_staging") or {}
        if staging.get("enabled"):
            checks.append(("sglang.model_staging.source_model_path", staging.get("source_model_path")))
        for index, command in enumerate(service.get("start_commands") or []):
            if command:
                checks.append((f"sglang.start_commands[{index}][0]", command[0]))
            model_path = _command_option(command, "--model-path")
            if model_path is not None:
                checks.append((f"sglang.start_commands[{index}].--model-path", model_path))
    for index, report in enumerate(spec.get("required_gate_reports") or []):
        checks.append((f"required_gate_reports[{index}]", report["path"]))

    missing: list[dict[str, str]] = []
    for label, value in checks:
        if value is None:
            missing.append({"label": label, "path": "<missing>"})
            continue
        path = Path(value)
        if not path.exists():
            missing.append({"label": label, "path": str(path)})
    return missing


def preflight_run_spec(spec: dict[str, Any]) -> None:
    missing = preflight_missing_paths(spec)
    if missing:
        formatted = ", ".join(f"{entry['label']}={entry['path']}" for entry in missing)
        raise QueueError(f"required RL launch paths are missing: {formatted}")
    spec["runtime_contract_report"] = _validate_runtime_contracts(
        spec.get("runtime_contracts") or []
    )
    for report in spec.get("required_gate_reports") or []:
        path = Path(report["path"])
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise QueueError(f"required gate report is unreadable: {path}: {exc}") from exc
        field = str(report.get("field", "passed"))
        value: Any = payload
        for part in field.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        expected = report.get("expected", True)
        if value != expected:
            raise QueueError(
                f"required gate report did not pass: {path} field={field!r} value={value!r}"
            )


def post_eval_preflight_missing_paths(eval_specs: list[dict[str, Any]]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for eval_spec in eval_specs:
        for label, value in eval_spec.get("required_paths", {}).items():
            if not Path(value).exists():
                missing.append(
                    {
                        "label": f"post_eval.{eval_spec['name']}.{label}",
                        "path": str(value),
                    }
                )
    return missing


def _nested_mapping(source: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = source
    for key in keys:
        if not isinstance(value, dict):
            return {}
        value = value.get(key)
    return value if isinstance(value, dict) else {}


def _roundtrip_post_eval_config(defaults: dict[str, Any], item: dict[str, Any]) -> dict[str, Any] | None:
    raw = _merge_mapping(
        _nested_mapping(defaults, "post_eval", "roundtrip"),
        _nested_mapping(item, "post_eval", "roundtrip"),
    )
    if not raw or not _as_bool(raw.get("enabled")):
        return None
    eval_splits = [
        str(value) for value in raw.get("eval_splits", ["validation"])
    ]
    if not eval_splits or len(eval_splits) != len(set(eval_splits)):
        raise QueueError("post_eval.roundtrip.eval_splits must be non-empty and unique")
    invalid_splits = sorted(set(eval_splits) - {"validation", "test"})
    if invalid_splits:
        raise QueueError(
            f"post_eval.roundtrip.eval_splits contains invalid values: {invalid_splits}"
        )
    raw["eval_splits"] = eval_splits
    return raw


def _roundtrip_post_eval_configs(defaults: dict[str, Any], item: dict[str, Any]) -> list[dict[str, Any]]:
    raw = _roundtrip_post_eval_config(defaults, item)
    if raw is None:
        return []
    checkpoints = raw.get("checkpoints")
    if checkpoints is None:
        return [raw]
    if not isinstance(checkpoints, list) or not checkpoints:
        raise QueueError("post_eval.roundtrip.checkpoints must be a non-empty list")
    common = {key: value for key, value in raw.items() if key != "checkpoints"}
    configs: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise QueueError(f"post_eval.roundtrip.checkpoints[{index}] must be a mapping")
        merged = _merge_mapping(common, checkpoint)
        name = str(merged.get("name") or f"checkpoint-{merged.get('iteration', index)}")
        if name in names:
            raise QueueError(f"duplicate round-trip checkpoint eval name: {name}")
        names.add(name)
        merged["name"] = name
        configs.append(merged)
    return configs


def _reward_gate_correlation_post_eval_config(defaults: dict[str, Any], item: dict[str, Any]) -> dict[str, Any] | None:
    raw = _merge_mapping(
        _nested_mapping(defaults, "post_eval", "reward_gate_correlation"),
        _nested_mapping(item, "post_eval", "reward_gate_correlation"),
    )
    if not raw or not _as_bool(raw.get("enabled")):
        return None
    return raw


def _checkpoint_iteration(rollout: dict[str, Any], post_eval: dict[str, Any]) -> int:
    value = post_eval.get("iteration", rollout.get("num_rollout"))
    if value is None:
        raise QueueError("post_eval.roundtrip requires iteration or rollout.num_rollout")
    return int(value)


def _iteration_name(iteration: int) -> str:
    if iteration < 0:
        raise QueueError(f"checkpoint iteration must be non-negative, got {iteration}")
    return f"iter_{iteration:07d}"


def _append_optional_eval_arg(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    _append_cli_arg(command, name, value)


def build_roundtrip_post_eval_spec(
    queue_doc: dict[str, Any],
    item: dict[str, Any],
    run_spec: dict[str, Any],
    *,
    queue_path: str | Path,
    post_eval_override: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    queue_path = Path(queue_path)
    defaults = queue_doc.get("defaults") or {}
    post_eval = post_eval_override or _roundtrip_post_eval_config(defaults, item)
    if post_eval is None:
        return None

    code_root = Path(run_spec["code_root"])
    nla_root = Path(run_spec["nla_root"])
    rollout = _rollout_config(defaults, item)
    iteration = _checkpoint_iteration(rollout, post_eval)
    iter_name = _iteration_name(iteration)
    run_dir = Path(run_spec["run_dir"])
    input_dir = _resolve_path(
        post_eval.get("input_dir", run_dir / "actor" / iter_name),
        queue_path=queue_path,
    )
    output_dir = _resolve_path(
        post_eval.get("hf_output_dir", run_dir / f"hf_{iter_name}_tmp"),
        queue_path=queue_path,
    )
    report_json = _resolve_path(
        post_eval.get("report_json", run_dir / f"roundtrip_{iter_name}_v{post_eval.get('validation_limit', 64)}_t{post_eval.get('test_limit', 64)}_report.json"),
        queue_path=queue_path,
    )
    generated_jsonl = _resolve_path(
        post_eval.get("generated_jsonl", report_json.with_suffix("").with_name(report_json.with_suffix("").name + "_generated.jsonl")),
        queue_path=queue_path,
    )
    log_path = _resolve_path(
        post_eval.get(
            "log_path",
            run_dir
            / f"roundtrip_{iter_name}_v{post_eval.get('validation_limit', 64)}_t{post_eval.get('test_limit', 64)}.log",
        ),
        queue_path=queue_path,
    )
    python_bin = str(post_eval.get("python", defaults.get("post_eval_python", run_spec["env"]["PYTHON"])))
    converter_script = _resolve_script_path(
        str(post_eval.get("converter_script", "external/natural_language_autoencoders/tools/convert_fsdp_to_hf.py")),
        queue_path=queue_path,
        code_root=code_root,
        nla_root=nla_root,
    )
    eval_script = _resolve_script_path(
        str(post_eval.get("eval_script", "scripts/eval_nano_av_ar_roundtrip_gate.py")),
        queue_path=queue_path,
        code_root=code_root,
        nla_root=nla_root,
    )

    origin_hf_dir = _resolve_path(
        post_eval.get("origin_hf_dir", item["instruct_model"]),
        queue_path=queue_path,
    )
    converter_command = None
    if not _as_bool(post_eval.get("skip_conversion")):
        converter_command = [
            python_bin,
            str(converter_script),
            "--input-dir",
            str(input_dir),
            "--origin-hf-dir",
            str(origin_hf_dir),
            "--output-dir",
            str(output_dir),
        ]
    remote_code_patch_command = None
    if converter_command is not None and _as_bool(post_eval.get("patch_remote_code"), default=True):
        remote_code_patch_command = [
            python_bin,
            "-m",
            str(post_eval.get("remote_code_patch_module", "nla.remote_code_patches")),
            str(output_dir),
        ]

    eval_splits = [str(value) for value in post_eval["eval_splits"]]
    critic_input_dir = None
    critic_hf_output_dir = None
    critic_converter_command = None
    critic_remote_code_patch_command = None
    configured_critic_input = post_eval.get("critic_input_dir")
    configured_ar_checkpoint = post_eval.get("ar_checkpoint_dir")
    if configured_critic_input is not None:
        if configured_ar_checkpoint is not None:
            raise QueueError(
                "post_eval.roundtrip must configure either critic_input_dir or "
                "ar_checkpoint_dir, not both"
            )
        critic_input_dir = _resolve_path(
            str(configured_critic_input),
            queue_path=queue_path,
        )
        if _as_bool(post_eval.get("convert_critic_checkpoint"), default=False):
            critic_hf_output_dir = _resolve_path(
                post_eval.get(
                    "critic_hf_output_dir",
                    run_dir / f"critic_hf_{iter_name}_tmp",
                ),
                queue_path=queue_path,
            )
            critic_converter_command = [
                python_bin,
                str(converter_script),
                "--input-dir",
                str(critic_input_dir),
                "--origin-hf-dir",
                str(origin_hf_dir),
                "--output-dir",
                str(critic_hf_output_dir),
            ]
            if _as_bool(post_eval.get("patch_critic_remote_code"), default=True):
                critic_remote_code_patch_command = [
                    python_bin,
                    "-m",
                    str(post_eval.get("remote_code_patch_module", "nla.remote_code_patches")),
                    str(critic_hf_output_dir),
                ]
            ar_checkpoint_dir = critic_hf_output_dir
        else:
            ar_checkpoint_dir = critic_input_dir
    else:
        ar_checkpoint_dir = configured_ar_checkpoint

    required_eval_paths = {
        "train_parquet": post_eval.get("train_parquet"),
        "validation_parquet": post_eval.get("validation_parquet"),
    }
    if ar_checkpoint_dir is None:
        required_eval_paths["ar_checkpoint_dir"] = None
    elif critic_input_dir is None:
        required_eval_paths["ar_checkpoint_dir"] = ar_checkpoint_dir
    if "test" in eval_splits:
        required_eval_paths["test_parquet"] = post_eval.get("test_parquet")
    require_family_inference = _as_bool(
        post_eval.get("require_family_level_inference")
    )
    for family_path_name in (
        "content_family_manifest",
        "content_family_coverage",
    ):
        family_path = post_eval.get(family_path_name)
        if family_path is not None or require_family_inference:
            required_eval_paths[family_path_name] = family_path
    missing = [name for name, value in required_eval_paths.items() if not value]
    if missing:
        raise QueueError(f"post_eval.roundtrip missing required fields: {', '.join(missing)}")

    eval_command = [
        python_bin,
        str(eval_script),
        "--av-hf-checkpoint",
        str(output_dir),
        "--ar-checkpoint-dir",
        str(_resolve_path(str(ar_checkpoint_dir), queue_path=queue_path)),
        "--train-parquet",
        str(_resolve_path(str(required_eval_paths["train_parquet"]), queue_path=queue_path)),
        "--validation-parquet",
        str(_resolve_path(str(required_eval_paths["validation_parquet"]), queue_path=queue_path)),
        "--report-json",
        str(report_json),
        "--generated-jsonl",
        str(generated_jsonl),
    ]
    if "test" in eval_splits:
        eval_command.extend(
            [
                "--test-parquet",
                str(
                    _resolve_path(
                        str(required_eval_paths["test_parquet"]),
                        queue_path=queue_path,
                    )
                ),
            ]
        )
    required_paths = {
        name: str(_resolve_path(str(value), queue_path=queue_path))
        for name, value in required_eval_paths.items()
    }
    if post_eval.get("baseline_report_json") is not None and not _as_bool(
        post_eval.get("baseline_generated_by_previous")
    ):
        required_paths["baseline_report_json"] = str(
            _resolve_path(str(post_eval["baseline_report_json"]), queue_path=queue_path)
        )
    for name in (
        "baseline_report_json",
        "eval_splits",
        "validation_limit",
        "test_limit",
        "max_new_tokens",
        "generation_backend",
        "generation_workers",
        "generation_worker_devices",
        "generation_controls",
        "generated_text_fallback",
        "stop_text",
        "content_family_manifest",
        "content_family_coverage",
        "selection_strategy",
        "selection_seed",
        "seed",
        "min_control_win_fraction",
        "min_baseline_win_fraction",
        "min_baseline_relative_improvement",
        "require_baseline_ci_positive",
        "require_clustered_baseline_ci",
        "require_baseline_dataset_match",
        "bootstrap_samples",
        "bootstrap_seed",
        "permutation_samples",
        "permutation_seed",
        "min_independent_families",
        "min_closed_fraction",
        "min_usable_fraction",
        "control_margin",
        "baseline_margin",
        "injection_scale",
        "ar_batch_size",
        "ar_max_length",
        "av_attn_implementation",
        "av_device_map",
        "ar_device_map",
        "ar_low_cpu_mem_usage",
        "collect_ar_device_profile",
        "av_model_fingerprint",
        "av_tokenizer_fingerprint",
        "require_generation_protocol_match",
        "require_family_level_inference",
    ):
        value = post_eval.get(name)
        if name == "test_limit" and "test" not in eval_splits:
            continue
        if name in {
            "baseline_report_json",
            "content_family_manifest",
            "content_family_coverage",
        } and value is not None:
            value = _resolve_path(str(value), queue_path=queue_path)
        _append_optional_eval_arg(eval_command, name, value)
    if _as_bool(post_eval.get("stream_generated"), default=True):
        eval_command.append("--stream-generated")
    if _as_bool(post_eval.get("resume_generated")):
        eval_command.append("--resume-generated")
    if _as_bool(post_eval.get("reuse_generated")):
        eval_command.append("--reuse-generated")
    if _as_bool(post_eval.get("local_files_only")):
        eval_command.append("--local-files-only")

    generation_command = None
    score_command = None
    scoring_python = post_eval.get("scoring_python", defaults.get("post_eval_scoring_python"))
    if scoring_python is not None:
        scoring_python_bin = str(_resolve_path(str(scoring_python), queue_path=queue_path))
        if not _as_bool(post_eval.get("reuse_generated")):
            generation_command = [*eval_command, "--generation-only"]
        score_command = [scoring_python_bin, *eval_command[1:]]
        if "--reuse-generated" not in score_command:
            score_command.append("--reuse-generated")

    return {
        "name": str(post_eval.get("name") or iter_name),
        "iteration": iteration,
        "iter_name": iter_name,
        "input_dir": str(input_dir),
        "hf_output_dir": str(output_dir),
        "origin_hf_dir": str(origin_hf_dir),
        "report_json": str(report_json),
        "generated_jsonl": str(generated_jsonl),
        "log_path": str(log_path),
        "av_fingerprint_mode": str(
            post_eval.get("av_fingerprint_mode") or "auto_hf"
        ),
        "fingerprint_workers": int(post_eval.get("fingerprint_workers") or 1),
        "fingerprint_report_json": str(
            _resolve_path(
                post_eval.get(
                    "fingerprint_report_json",
                    run_dir / f"{iter_name}_generation_identity.json",
                ),
                queue_path=queue_path,
            )
        ),
        "av_model_fingerprint": post_eval.get("av_model_fingerprint"),
        "av_tokenizer_fingerprint": post_eval.get("av_tokenizer_fingerprint"),
        "converter_command": converter_command,
        "remote_code_patch_command": remote_code_patch_command,
        "critic_input_dir": None if critic_input_dir is None else str(critic_input_dir),
        "critic_hf_output_dir": (
            None if critic_hf_output_dir is None else str(critic_hf_output_dir)
        ),
        "critic_converter_command": critic_converter_command,
        "critic_remote_code_patch_command": critic_remote_code_patch_command,
        "eval_command": eval_command,
        "generation_command": generation_command,
        "score_command": score_command,
        "cleanup_hf": _as_bool(post_eval.get("cleanup_hf"), default=True),
        "cleanup_critic_hf": _as_bool(
            post_eval.get("cleanup_critic_hf"),
            default=critic_hf_output_dir is not None,
        ),
        "cleanup_actor_checkpoint": _as_bool(post_eval.get("cleanup_actor_checkpoint")),
        "require_previous_gate_pass": _as_bool(post_eval.get("require_previous_gate_pass")),
        "run_reward_gate_correlation": _as_bool(post_eval.get("run_reward_gate_correlation")),
        "reuse_generated": _as_bool(post_eval.get("reuse_generated")),
        "required_paths": required_paths,
    }


def build_roundtrip_post_eval_specs(
    queue_doc: dict[str, Any],
    item: dict[str, Any],
    run_spec: dict[str, Any],
    *,
    queue_path: str | Path,
) -> list[dict[str, Any]]:
    defaults = queue_doc.get("defaults") or {}
    return [
        build_roundtrip_post_eval_spec(
            queue_doc,
            item,
            run_spec,
            queue_path=queue_path,
            post_eval_override=config,
        )
        for config in _roundtrip_post_eval_configs(defaults, item)
    ]


def build_run_spec(queue_doc: dict[str, Any], item: dict[str, Any], *, queue_path: str | Path) -> dict[str, Any]:
    queue_path = Path(queue_path)
    defaults = queue_doc.get("defaults") or {}
    code_root = _resolve_path(defaults.get("code_root", "."), queue_path=queue_path)
    nla_root = _resolve_path(
        defaults.get("nla_root", "external/natural_language_autoencoders"),
        queue_path=queue_path,
        code_root=code_root,
    )
    miles_root = _optional_path(defaults.get("miles_root"), queue_path=queue_path)
    training = _training_config(defaults, item)
    train_entrypoint = _optional_path(defaults.get("train_entrypoint"), queue_path=queue_path)
    if train_entrypoint is None:
        if _training_bool(training, "async_training"):
            train_entrypoint = (miles_root / "train_async.py") if miles_root is not None else (nla_root / "train_async.py")
        else:
            train_entrypoint = (miles_root / "train.py") if miles_root is not None else (nla_root / "train.py")
    rl_script = _resolve_script_path(
        defaults.get("rl_script", "configs/rl.sh"),
        queue_path=queue_path,
        code_root=code_root,
        nla_root=nla_root,
    )
    python_bin = str(defaults.get("python", os.environ.get("PYTHON", "python")))
    resources = _resource_config(defaults, item)
    total_gpus = resources["actor_gpus"] * resources["actor_nodes"] + resources["critic_gpus"] * resources["critic_nodes"] + resources["rollout_gpus"]
    workspace_gpus = defaults.get("workspace_gpus", item.get("workspace_gpus"))
    workspace_gpus_value = None if workspace_gpus is None else int(workspace_gpus)
    if workspace_gpus_value is not None and total_gpus > workspace_gpus_value:
        raise QueueError(f"RL topology requires {total_gpus} GPUs but workspace_gpus is {workspace_gpus_value}")
    actor_total_gpus = resources["actor_gpus"] * resources["actor_nodes"]
    if actor_total_gpus < resources["min_actor_gpus"]:
        raise QueueError(
            "RL actor topology requires at least "
            f"{resources['min_actor_gpus']} actor GPUs but configured {actor_total_gpus}"
        )

    rollout = _rollout_config(defaults, item)
    rollout_batch = _rollout_batch_plan(rollout)
    actor_micro_batch = _positive_int(
        training.get("actor_micro_batch", training.get("micro_batch_size", 4)),
        field="training.actor_micro_batch",
    )
    rollout_batch.update(
        _validate_actor_batch_plan(
            generated_samples=rollout_batch["generated_samples"],
            global_batch_size=rollout_batch["global_batch_size"],
            actor_gpus=actor_total_gpus,
            actor_micro_batch=actor_micro_batch,
            required=_training_bool(training, "require_exact_actor_batch"),
        )
    )
    critic_total_gpus = resources["critic_gpus"] * resources["critic_nodes"]
    rollout_batch.update(
        _validate_critic_batch_plan(
            global_batch_size=rollout_batch["global_batch_size"],
            critic_gpus=critic_total_gpus,
            critic_micro_batch=actor_micro_batch,
            required=_training_bool(training, "require_exact_critic_batch"),
        )
    )
    sglang_service = _sglang_service_config(defaults, item)
    input_staging = _input_staging_config(
        defaults,
        item,
        queue_path=queue_path,
    )
    runtime_env = _merge_mapping(defaults.get("env"), item.get("env"))
    critic_update_mode = _critic_update_mode(training, runtime_env)
    runtime_env["NLA_FREEZE_CRITIC_TRAIN"] = (
        "1" if critic_update_mode == "frozen" else "0"
    )
    _validate_live_sync_runtime(
        trainer_python=python_bin,
        sglang_service=sglang_service,
        runtime_env=runtime_env,
    )
    _validate_training_runtime_config(
        training,
        sglang_mode=sglang_service["mode"],
        critic_update_mode=critic_update_mode,
    )
    drift_guard = _drift_guard_config(training)
    metric_guard_rules = _metric_guard_rules(training, rollout)
    _validate_sglang_resource_layout(resources, sglang_service)
    run_dir = _resolve_path(item["run_dir"], queue_path=queue_path)
    log_path = _resolve_path(item.get("log_path", str(run_dir / "train.log")), queue_path=queue_path)
    wandb = _wandb_config(defaults, item, run_dir=run_dir, queue_path=queue_path)
    source_policy = _source_policy(defaults, item)
    source_provenance = None
    if source_policy:
        if code_root.is_dir():
            critical_files: dict[str, Path] = {}
            rl_dataset_path = _resolve_path(
                item["rl_parquet"],
                queue_path=queue_path,
            )
            if rl_dataset_path.is_file() or not source_policy.get(
                "allow_missing_during_render"
            ):
                critical_files["rl_dataset"] = rl_dataset_path
            for name, value in (source_policy.get("critical_files") or {}).items():
                critical_files[str(name)] = _resolve_path(
                    value,
                    queue_path=queue_path,
                    code_root=code_root,
                )
            for name, value in (
                ("package_lock", source_policy.get("package_lock")),
                ("dataset_manifest", source_policy.get("dataset_manifest")),
                ("content_family_manifest", source_policy.get("content_family_manifest")),
                ("content_family_coverage", source_policy.get("content_family_coverage")),
            ):
                if value:
                    critical_files[name] = _resolve_path(
                        value,
                        queue_path=queue_path,
                        code_root=code_root,
                    )
            patches_root = _resolve_path(
                source_policy.get(
                    "miles_patches_root",
                    code_root
                    / "external"
                    / "natural_language_autoencoders"
                    / "nla"
                    / "miles_patches",
                ),
                queue_path=queue_path,
                code_root=code_root,
            )
            default_source_roots = (
                "scripts",
                "external/natural_language_autoencoders/nla",
                "external/natural_language_autoencoders/configs",
            )
            source_roots = source_policy.get("roots")
            if not source_roots:
                source_roots = tuple(
                    relative
                    for relative in default_source_roots
                    if (code_root / relative).is_dir()
                ) or (".",)
            provenance_python = shutil.which(python_bin) or sys.executable
            source_provenance = collect_source_provenance(
                code_root,
                queue_path=queue_path,
                roots=tuple(source_roots),
                miles_root=miles_root if miles_root is not None and miles_root.is_dir() else None,
                miles_patches_root=patches_root if patches_root.is_dir() else None,
                critical_files=critical_files,
                container_image_digest=source_policy.get("container_image_digest"),
                python_executable=provenance_python,
            )
            source_provenance["configured_python"] = python_bin
            verify_source_policy(source_policy, source_provenance)
        else:
            source_provenance = {
                "schema_version": "nano_source_provenance.v1",
                "deferred": True,
                "code_root": str(code_root),
                "expected_code_sha256": source_policy.get("expected_code_sha256"),
            }
        if source_policy.get("frozen_git_commit"):
            source_provenance["declared_git_commit"] = str(
                source_policy["frozen_git_commit"]
            )
    checkpoint_retention = _checkpoint_retention_config(
        defaults,
        item,
        training=training,
        rollout=rollout,
        run_dir=run_dir,
    )
    required_gate_reports = _required_gate_reports(defaults, item, queue_path=queue_path)
    runtime_contracts = _runtime_contracts_config(defaults, item, queue_path=queue_path)
    preregistration = _preregistration_config(
        defaults,
        item,
        queue_path=queue_path,
        code_root=code_root,
        training=training,
        rollout=rollout,
        runtime_env=runtime_env,
        metric_guard_rules=metric_guard_rules,
        roundtrip_configs=_roundtrip_post_eval_configs(defaults, item),
    )

    env = {
        str(key): str(value)
        for key, value in os.environ.items()
        if not str(key).startswith("NLA_")
        and str(key)
        not in {
            "KL_LOSS_TYPE",
            "WANDB_MODE",
            "WANDB_DIR",
            "WANDB_PROJECT",
            "WANDB_GROUP",
            "WANDB_RUN_ID",
        }
    }
    env.update({str(k): str(v) for k, v in runtime_env.items()})
    env["WANDB_MODE"] = str(wandb["mode"])
    env["WANDB_DIR"] = str(wandb["dir"])
    env["WANDB_PROJECT"] = str(wandb["project"])
    env["WANDB_GROUP"] = str(wandb["group"])
    if wandb["run_id"]:
        env["WANDB_RUN_ID"] = wandb["run_id"]
    env.setdefault("NLA_SYSTEM_METRICS", "1")
    env.setdefault("NLA_SYSTEM_METRICS_INTERVAL_STEPS", "1")
    env.setdefault("NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS", "1")
    env.setdefault("NLA_PHASE_METRICS", "1")
    env.setdefault("NLA_PHASE_METRICS_ALL_GPUS", "1")
    env.setdefault("NLA_PHASE_METRICS_WANDB", "1")
    env["PYTHON"] = python_bin
    env["TRAIN_ENTRYPOINT"] = str(train_entrypoint)
    env["RL_PARQUET"] = str(_resolve_path(item["rl_parquet"], queue_path=queue_path))
    env["INSTRUCT_MODEL"] = str(_resolve_path(item["instruct_model"], queue_path=queue_path))
    env["ACTOR_SFT_CKPT"] = str(_resolve_path(item["actor_sft_ckpt"], queue_path=queue_path))
    env["ACTOR_LOAD_CKPT"] = str(_resolve_path(item.get("actor_load_ckpt", item["actor_sft_ckpt"]), queue_path=queue_path))
    env["ACTOR_REF_CKPT"] = str(_resolve_path(item.get("actor_ref_ckpt", env["ACTOR_LOAD_CKPT"]), queue_path=queue_path))
    env["ACTOR_SIDECAR_SOURCE"] = str(
        _resolve_path(item.get("actor_sidecar_source", item["actor_sft_ckpt"]), queue_path=queue_path)
    )
    env["CRITIC_SL_CKPT"] = str(_resolve_path(item["critic_sl_ckpt"], queue_path=queue_path))
    env["RUN_DIR"] = str(run_dir)
    env["ACTOR_GPUS"] = str(resources["actor_gpus"])
    env["CRITIC_GPUS"] = str(resources["critic_gpus"])
    env["ROLLOUT_GPUS"] = str(resources["rollout_gpus"])
    env["ACTOR_NODES"] = str(resources["actor_nodes"])
    env["CRITIC_NODES"] = str(resources["critic_nodes"])
    if workspace_gpus_value is not None:
        env["NLA_WORKSPACE_GPUS"] = str(workspace_gpus_value)
    env["NLA_ACTOR_GPUS"] = str(resources["actor_gpus"])
    env["NLA_CRITIC_GPUS"] = str(resources["critic_gpus"])
    env["NLA_ROLLOUT_GPUS"] = str(resources["rollout_gpus"])
    if sglang_service.get("rollout_num_gpus_per_engine") is not None:
        env["NLA_ROLLOUT_GPUS_PER_ENGINE"] = str(sglang_service["rollout_num_gpus_per_engine"])
    if sglang_service.get("tensor_parallel_size") is not None:
        env["NLA_SGLANG_TP_SIZE"] = str(sglang_service["tensor_parallel_size"])
    if sglang_service.get("base_gpu_id") is not None:
        env["NLA_SGLANG_BASE_GPU_ID"] = str(sglang_service["base_gpu_id"])
    env["ACTOR_MICRO"] = str(actor_micro_batch)
    env["ACTOR_LR"] = str(training.get("actor_lr", "1e-6"))
    env["CRITIC_LR"] = str(training.get("critic_lr", "1e-5"))
    env["NLA_MIN_CRITIC_RETAINED_FRACTION"] = str(
        training.get("min_critic_retained_fraction", 0.95)
    )
    env["CLIP_GRAD"] = str(training.get("clip_grad", "1.0"))
    env["KL_LOSS_COEF"] = str(training.get("kl_loss_coef", "0.01"))
    env["KL_LOSS_TYPE"] = str(training.get("kl_loss_type", "k1")).strip()
    guard_env_keys = (
        "NLA_CUSTOM_TRAIN_GUARD_FUNCTION",
        "NLA_TRAIN_GUARD_MAX_LOGPROB_ABS_DIFF",
        "NLA_TRAIN_GUARD_CONSECUTIVE_STEPS",
        "NLA_TRAIN_GUARD_METRIC",
        "NLA_TRAIN_GUARD_RULES_JSON",
        "NLA_ROLLOUT_GUARD_RULES_JSON",
    )
    if metric_guard_rules["actor"]:
        env["NLA_CUSTOM_TRAIN_GUARD_FUNCTION"] = "nla.train_guard.check_train_metrics"
        env["NLA_TRAIN_GUARD_RULES_JSON"] = json.dumps(metric_guard_rules["actor"], sort_keys=True)
        if drift_guard["enabled"]:
            env["NLA_TRAIN_GUARD_MAX_LOGPROB_ABS_DIFF"] = str(
                drift_guard["max_logprob_abs_diff"]
            )
            env["NLA_TRAIN_GUARD_CONSECUTIVE_STEPS"] = str(
                drift_guard["consecutive_steps"]
            )
            env["NLA_TRAIN_GUARD_METRIC"] = drift_guard["metric"]
    elif drift_guard["enabled"]:
        env["NLA_CUSTOM_TRAIN_GUARD_FUNCTION"] = drift_guard["function_path"]
        env["NLA_TRAIN_GUARD_MAX_LOGPROB_ABS_DIFF"] = str(
            drift_guard["max_logprob_abs_diff"]
        )
        env["NLA_TRAIN_GUARD_CONSECUTIVE_STEPS"] = str(
            drift_guard["consecutive_steps"]
        )
        env["NLA_TRAIN_GUARD_METRIC"] = drift_guard["metric"]
    else:
        for key in guard_env_keys:
            env.pop(key, None)
    if metric_guard_rules["rollout"]:
        env["NLA_ROLLOUT_GUARD_RULES_JSON"] = json.dumps(metric_guard_rules["rollout"], sort_keys=True)
    env["GRADIENT_CHECKPOINTING"] = "1" if _training_bool(training, "gradient_checkpointing") else "0"
    env["OFFLOAD_TRAIN"] = "1" if _training_bool(training, "offload_train") else "0"
    env["OFFLOAD_ROLLOUT"] = "1" if _training_bool(training, "offload_rollout") else "0"
    env["OFFLOAD_ROLLOUT_LEVEL"] = " ".join(_training_list(training, "offload_rollout_level"))
    env["FSDP_CPU_OFFLOAD"] = "1" if _training_bool(training, "fsdp_cpu_offload") else "0"
    if training.get("fsdp_cpu_backend") is not None:
        env["FSDP_CPU_BACKEND"] = str(training["fsdp_cpu_backend"])
    env["COLOCATE"] = "1" if _training_bool(training, "colocate") else "0"
    env["NLA_REF_LOG_PROBS_PLACEMENT"] = str(training.get("ref_log_probs_placement", "actor")).strip().lower()
    env["ADVANTAGE_ESTIMATOR"] = str(training.get("advantage_estimator", "grpo"))
    env["NORMALIZE_ADVANTAGES"] = "1" if _as_bool(training.get("normalize_advantages")) else "0"
    env["REWARDS_NORMALIZATION"] = "1" if _as_bool(training.get("rewards_normalization"), default=True) else "0"
    env["GRPO_STD_NORMALIZATION"] = (
        "1" if _as_bool(training.get("grpo_std_normalization"), default=True) else "0"
    )
    save_iterations = _save_iterations_config(training, rollout)
    if save_iterations is not None:
        env["SAVE_INTERVAL"] = ""
        env["NLA_SAVE_ITERATIONS"] = ",".join(
            str(value) for value in save_iterations
        )
    else:
        save_interval = training.get("save_interval", 100)
        env["SAVE_INTERVAL"] = "" if save_interval is None else str(save_interval)
        env.pop("NLA_SAVE_ITERATIONS", None)
    env["LOSS_MASK_TYPE"] = str(training.get("loss_mask_type", "qwen"))
    env["QKV_FORMAT"] = str(training.get("qkv_format", "thd")).strip().lower()
    env["LR_DECAY_STYLE"] = str(training.get("lr_decay_style", "constant"))
    env["FINETUNE"] = "1" if _as_bool(training.get("finetune")) else "0"
    env["NO_LOAD_OPTIM"] = "1" if _as_bool(training.get("no_load_optim")) else "0"
    if training.get("fsdp_reduce_dtype") is not None:
        env["FSDP_REDUCE_DTYPE"] = str(training["fsdp_reduce_dtype"])
    if training.get("fsdp_disable_backward_prefetch") is not None:
        env["FSDP_DISABLE_BACKWARD_PREFETCH"] = (
            "1" if _as_bool(training.get("fsdp_disable_backward_prefetch")) else "0"
        )
    env["NANO_SGLANG_MODE"] = sglang_service["mode"]
    if sglang_service["python"] is not None:
        env["NANO_SGLANG_PYTHON"] = sglang_service["python"]
    if training.get("attn_implementation") is not None:
        env["ATTN_IMPLEMENTATION"] = str(training["attn_implementation"])
    if training.get("train_backend") is not None:
        env["TRAIN_BACKEND"] = str(training["train_backend"])
    if training.get("actor_cls") is not None:
        env["ACTOR_CLS"] = str(training["actor_cls"])
    env["NLA_ROLLOUT_PROMPT_BATCH"] = str(rollout_batch["rollout_batch_size"])
    env["NLA_ROLLOUT_SAMPLES_PER_PROMPT"] = str(rollout_batch["n_samples_per_prompt"])
    env["NLA_ROLLOUT_GENERATED_SAMPLES"] = str(rollout_batch["generated_samples"])
    env["NLA_ROLLOUT_GLOBAL_BATCH"] = str(rollout_batch["global_batch_size"])
    env["NLA_ROLLOUT_GLOBAL_MATCH"] = "1" if rollout_batch["global_batch_matches_rollout"] else "0"

    pythonpath = [str(nla_root), str(code_root)]
    if miles_root is not None:
        pythonpath.append(str(miles_root))
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)

    command = ["bash", str(rl_script)]
    rollout_cli = {
        "rollout_batch_size": rollout_batch["rollout_batch_size"],
        "global_batch_size": rollout_batch["global_batch_size"],
        "n_samples_per_prompt": rollout_batch["n_samples_per_prompt"],
        "rollout_max_response_len": rollout.get("max_response_len", rollout.get("rollout_max_response_len", 150)),
        "rollout_max_context_len": rollout.get("max_context_len", rollout.get("rollout_max_context_len", 300)),
        "num_rollout": rollout.get("num_rollout"),
        "rollout_seed": rollout.get("seed", rollout.get("rollout_seed")),
    }
    for name, value in rollout_cli.items():
        _append_cli_arg(command, name, value)
    if _as_bool(rollout.get("shuffle"), default=False):
        command.append("--rollout-shuffle")
    if sglang_service["mode"] == "external":
        command.append("--rollout-external")
        command.append("--rollout-external-engine-addrs")
        command.extend(sglang_service["engine_addrs"])
        if sglang_service["router_ip"] is not None:
            command.extend(["--sglang-router-ip", str(sglang_service["router_ip"])])
        if sglang_service["router_port"] is not None:
            command.extend(["--sglang-router-port", str(sglang_service["router_port"])])
    extra_args = _extra_args(defaults, item)
    if sglang_service.get("rollout_num_gpus_per_engine") is not None:
        if "--rollout-num-gpus-per-engine" in extra_args:
            raise QueueError(
                "rollout-num-gpus-per-engine must be configured through "
                "sglang.rollout_num_gpus_per_engine, not duplicated in extra_args"
            )
        command.extend(["--rollout-num-gpus-per-engine", str(sglang_service["rollout_num_gpus_per_engine"])])
    command.extend(extra_args)

    return {
        "item_name": str(item["name"]),
        "cwd": str(nla_root),
        "code_root": str(code_root),
        "nla_root": str(nla_root),
        "command": command,
        "env": env,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "resource_total_gpus": total_gpus,
        "critic_update_mode": critic_update_mode,
        "rollout_batch_plan": rollout_batch,
        "sglang_service": sglang_service,
        "input_staging": input_staging,
        "source_provenance": source_provenance,
        "preregistration": preregistration,
        "checkpoint_retention": checkpoint_retention,
        "required_gate_reports": required_gate_reports,
        "runtime_contracts": runtime_contracts,
    }


def build_reward_gate_correlation_post_eval_spec(
    queue_doc: dict[str, Any],
    item: dict[str, Any],
    run_spec: dict[str, Any],
    roundtrip_spec: dict[str, Any],
    *,
    queue_path: str | Path,
) -> dict[str, Any] | None:
    defaults = queue_doc.get("defaults") or {}
    correlation = _reward_gate_correlation_post_eval_config(defaults, item)
    if correlation is None:
        return None

    code_root = Path(run_spec["code_root"])
    run_dir = Path(run_spec["run_dir"])
    output_json = _resolve_path(
        correlation.get("output_json", run_dir / "reward_gate_correlation.json"),
        queue_path=Path(queue_path),
        code_root=code_root,
    )
    python_bin = str(defaults.get("post_eval_scoring_python", defaults.get("python", "python")))
    command = [
        python_bin,
        str(code_root / "scripts" / "analyze_rl_reward_gate_correlation.py"),
        "--roundtrip-report-json",
        str(roundtrip_spec["report_json"]),
        "--generated-jsonl",
        str(roundtrip_spec["generated_jsonl"]),
        "--critic-checkpoint-dir",
        str(_resolve_path(correlation["critic_checkpoint_dir"], queue_path=Path(queue_path), code_root=code_root)),
        "--validation-parquet",
        str(_resolve_path(correlation["validation_parquet"], queue_path=Path(queue_path), code_root=code_root)),
        "--test-parquet",
        str(_resolve_path(correlation["test_parquet"], queue_path=Path(queue_path), code_root=code_root)),
        "--output-json",
        str(output_json),
        "--batch-size",
        str(int(correlation.get("batch_size", 16))),
        "--device",
        str(correlation.get("device", "cuda")),
    ]
    return {
        "command": command,
        "output_json": str(output_json),
        "log_path": str(run_dir / "reward_gate_correlation.log"),
    }


def _http_ok(url: str, *, timeout: float = 2.0) -> bool:
    try:
        with urllib_request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (urllib_error.URLError, TimeoutError, OSError):
        return False


def _write_service_pid_report(run_dir: Path, processes: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "updated_at": utc_now(),
        "processes": [
            {
                "pid": proc["process"].pid,
                "command": proc["command"],
                "log_path": str(proc["log_path"]),
                "status": proc["status"],
            }
            for proc in processes
        ],
    }
    (run_dir / "sglang_service_pids.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def _wait_for_service_health(service: dict[str, Any], processes: list[dict[str, Any]]) -> None:
    health_urls = service.get("health_urls") or []
    if not health_urls:
        return
    deadline = time.monotonic() + float(service["timeout_seconds"])
    while time.monotonic() < deadline:
        failed = [proc for proc in processes if proc["process"].poll() is not None]
        if failed:
            names = ", ".join(str(proc["command"]) for proc in failed)
            raise RuntimeError(f"SGLang service command exited before healthcheck passed: {names}")
        if all(_http_ok(url) for url in health_urls):
            return
        time.sleep(float(service["poll_seconds"]))
    raise TimeoutError(f"SGLang service healthcheck timed out: {health_urls}")


def _tree_signature(
    root: Path,
    *,
    excluded_names: frozenset[str] = frozenset(),
) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.name in excluded_names or not path.is_file():
            continue
        stat = path.stat()
        file_count += 1
        total_bytes += stat.st_size
    return {"files": file_count, "bytes": total_bytes}


def _safe_remove_path(path: Path) -> None:
    if path == path.parent:
        raise QueueError(f"refusing to remove unsafe staging path: {path}")
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _copytree_with_workers(
    source: Path,
    target: Path,
    *,
    workers: int,
    chunk_bytes: int = 0,
) -> None:
    if workers <= 1 and chunk_bytes <= 0:
        shutil.copytree(source, target, symlinks=True)
        return

    target.mkdir(parents=True)
    directory_pairs: list[tuple[Path, Path]] = [(source, target)]
    file_pairs: list[tuple[Path, Path]] = []

    for root, directory_names, file_names in os.walk(source, followlinks=False):
        source_root = Path(root)
        relative_root = source_root.relative_to(source)
        target_root = target / relative_root

        for directory_name in list(directory_names):
            source_path = source_root / directory_name
            target_path = target_root / directory_name
            if source_path.is_symlink():
                target_path.symlink_to(os.readlink(source_path), target_is_directory=True)
                directory_names.remove(directory_name)
                continue
            target_path.mkdir()
            directory_pairs.append((source_path, target_path))

        for file_name in file_names:
            source_path = source_root / file_name
            target_path = target_root / file_name
            if source_path.is_symlink():
                target_path.symlink_to(os.readlink(source_path))
            else:
                file_pairs.append((source_path, target_path))

    if chunk_bytes <= 0:
        def copy_file(paths: tuple[Path, Path]) -> None:
            shutil.copy2(paths[0], paths[1], follow_symlinks=False)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nla-model-stage") as executor:
            for _ in executor.map(copy_file, file_pairs):
                pass
    else:
        chunk_jobs: list[tuple[Path, Path, int, int]] = []
        for source_path, target_path in file_pairs:
            size = source_path.stat().st_size
            with target_path.open("wb") as target_file:
                target_file.truncate(size)
            for offset in range(0, size, chunk_bytes):
                chunk_jobs.append(
                    (source_path, target_path, offset, min(chunk_bytes, size - offset))
                )

        def copy_chunk(job: tuple[Path, Path, int, int]) -> None:
            source_path, target_path, offset, length = job
            remaining = length
            position = offset
            with source_path.open("rb", buffering=0) as source_file, target_path.open(
                "r+b", buffering=0
            ) as target_file:
                source_file.seek(offset)
                target_file.seek(offset)
                while remaining:
                    block = source_file.read(min(16 * 1024 * 1024, remaining))
                    if not block:
                        raise OSError(
                            f"unexpected EOF while staging {source_path} at byte {position}"
                        )
                    unwritten = memoryview(block)
                    while unwritten:
                        written = target_file.write(unwritten)
                        if written is None or written <= 0:
                            raise OSError(
                                f"short write while staging {target_path} at byte {position}"
                            )
                        unwritten = unwritten[written:]
                    position += len(block)
                    remaining -= len(block)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nla-model-stage") as executor:
            for _ in executor.map(copy_chunk, chunk_jobs):
                pass

        for source_path, target_path in file_pairs:
            shutil.copystat(source_path, target_path, follow_symlinks=False)

    for source_path, target_path in reversed(directory_pairs):
        shutil.copystat(source_path, target_path, follow_symlinks=False)


def _stage_directory_tree(
    *,
    source: Path,
    target: Path,
    marker_name: str,
    compatible_marker_names: Iterable[str] = (),
    reuse_existing: bool,
    clean: bool,
    copy_workers: int,
    copy_chunk_bytes: int,
) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(f"staging source does not exist: {source}")
    if not source.is_dir():
        raise QueueError(f"staging source must be a directory: {source}")

    source_resolved = source.resolve()
    target_resolved = target.resolve(strict=False)
    if (
        source_resolved == target_resolved
        or source_resolved in target_resolved.parents
        or target_resolved in source_resolved.parents
    ):
        raise QueueError(
            f"staging source and target must not overlap: {source} -> {target}"
        )

    source_signature = _tree_signature(source)
    expected_marker = {
        "source_model_path": str(source),
        "source_signature": source_signature,
    }
    marker_names = frozenset({marker_name, *compatible_marker_names})
    if any(not name or Path(name).name != name for name in marker_names):
        raise QueueError("staging marker names must be simple file names")
    is_current = False
    if reuse_existing:
        marker_is_current = False
        for candidate_name in marker_names:
            marker_path = target / candidate_name
            if not marker_path.exists():
                continue
            try:
                marker_is_current = json.loads(marker_path.read_text()) == expected_marker
            except (OSError, json.JSONDecodeError):
                marker_is_current = False
            if marker_is_current:
                break
        is_current = marker_is_current and _tree_signature(
            target,
            excluded_names=marker_names,
        ) == source_signature

    staged_at: str | None = None
    if not is_current:
        if target.exists() or target.is_symlink():
            if not clean:
                raise QueueError(f"staged target exists and clean=false: {target}")
            _safe_remove_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_name(
            f"{target.name}.tmp.{os.getpid()}.{time.time_ns()}"
        )
        _safe_remove_path(temp_target)
        try:
            _copytree_with_workers(
                source,
                temp_target,
                workers=copy_workers,
                chunk_bytes=copy_chunk_bytes,
            )
            copied_signature = _tree_signature(temp_target)
            if copied_signature != source_signature:
                raise QueueError(
                    "staged tree signature mismatch: "
                    f"source={source_signature} copied={copied_signature}"
                )
            (temp_target / marker_name).write_text(
                json.dumps(expected_marker, indent=2, sort_keys=True) + "\n"
            )
            temp_target.rename(target)
            staged_at = utc_now()
        except Exception:
            _safe_remove_path(temp_target)
            raise

    return {
        "source_path": str(source),
        "target_path": str(target),
        "source_signature": source_signature,
        "copy_workers": copy_workers,
        "copy_chunk_bytes": copy_chunk_bytes,
        "reused_existing": is_current,
        "staged_at": staged_at,
    }


def _stage_run_inputs(spec: dict[str, Any]) -> dict[str, Any]:
    staging_entries = spec.get("input_staging") or []
    if not staging_entries:
        return spec

    staged_spec = dict(spec)
    staged_env = dict(spec["env"])
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "input_staging.json"
    report: dict[str, Any] = {
        "schema_version": "nano_rl_input_staging.v1",
        "status": "staging",
        "updated_at": utc_now(),
        "entries": [],
    }

    def write_report() -> None:
        report["updated_at"] = utc_now()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    write_report()
    try:
        for entry in staging_entries:
            env_key = str(entry["env_key"])
            source_value = entry.get("source_path") or staged_env.get(env_key)
            if source_value is None or not str(source_value).strip():
                raise QueueError(
                    f"input staging {entry['name']!r} has no source_path and "
                    f"environment key {env_key!r} is unset"
                )
            stage_result = _stage_directory_tree(
                source=Path(str(source_value)),
                target=Path(str(entry["target_path"])),
                marker_name=".nla_input_stage.json",
                compatible_marker_names=(".nla_sglang_model_stage.json",),
                reuse_existing=bool(entry["reuse_existing"]),
                clean=bool(entry["clean"]),
                copy_workers=int(entry["copy_workers"]),
                copy_chunk_bytes=int(entry["copy_chunk_bytes"]),
            )
            staged_env[env_key] = stage_result["target_path"]
            report["entries"].append(
                {
                    "name": entry["name"],
                    "env_key": env_key,
                    **stage_result,
                }
            )
            write_report()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        write_report()
        raise

    report["status"] = "complete"
    report["completed_at"] = utc_now()
    write_report()
    staged_spec["env"] = staged_env
    staged_spec["input_staging_report"] = report
    return staged_spec


def _stage_sglang_model(service: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    staging = service.get("model_staging") or {}
    if not staging.get("enabled"):
        return service

    source = Path(str(staging["source_model_path"]))
    target = Path(str(staging["target_path"]))
    stage_result = _stage_directory_tree(
        source=source,
        target=target,
        marker_name=".nla_sglang_model_stage.json",
        compatible_marker_names=(".nla_input_stage.json",),
        reuse_existing=bool(staging.get("reuse_existing", True)),
        clean=bool(staging.get("clean", True)),
        copy_workers=int(staging.get("copy_workers", 1)),
        copy_chunk_bytes=int(staging.get("copy_chunk_bytes", 0)),
    )
    source_signature = stage_result["source_signature"]
    is_current = bool(stage_result["reused_existing"])

    source_text = str(source)
    target_text = str(target)
    rewritten_commands: list[list[str]] = []
    total_replacements = 0
    for command in service.get("start_commands") or []:
        rewritten, replacements = _replace_command_option(
            command,
            "--model-path",
            old_value=source_text,
            new_value=target_text,
        )
        rewritten_commands.append(rewritten)
        total_replacements += replacements
    if total_replacements == 0:
        raise QueueError(
            "sglang.model_staging did not match any managed start command --model-path "
            f"for source {source_text}"
        )

    staged_service = dict(service)
    staged_env = dict(staged_service.get("env") or {})
    staged_env["NLA_SGLANG_STAGED_MODEL_PATH"] = target_text
    staged_service["env"] = staged_env
    staged_service["start_commands"] = rewritten_commands
    staged_service["model_staging"] = {
        **staging,
        "target_path": target_text,
        "source_signature": source_signature,
        "staged_at": stage_result["staged_at"],
        "reused_existing": is_current,
    }
    (run_dir / "sglang_model_staging.json").write_text(
        json.dumps(staged_service["model_staging"], indent=2, sort_keys=True)
    )
    return staged_service


def _release_sglang_staged_files(
    service: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any] | None:
    staging = service.get("model_staging") or {}
    release = staging.get("release_after_health") or {}
    if not release.get("enabled"):
        return None
    target = Path(str(staging["target_path"]))
    if not target.is_dir():
        raise QueueError(f"SGLang staged target does not exist for release: {target}")
    patterns = [str(value) for value in release.get("globs") or []]
    if not patterns:
        raise QueueError("sglang.model_staging.release_after_health.globs must be non-empty")

    released: dict[str, int] = {}
    for pattern in patterns:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise QueueError(f"unsafe SGLang staged release glob: {pattern}")
        for path in sorted(target.glob(pattern)):
            if path.is_dir() and not path.is_symlink():
                raise QueueError(f"SGLang staged release matched a directory: {path}")
            if not path.exists() and not path.is_symlink():
                continue
            released[str(path)] = path.stat().st_size
            path.unlink()

    report = {
        "schema_version": "nano_sglang_model_release.v1",
        "released_at": utc_now(),
        "target_path": str(target),
        "globs": patterns,
        "released_files": sorted(released),
        "released_bytes": sum(released.values()),
    }
    (run_dir / "sglang_model_release.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _terminate_service_processes(run_dir: Path, processes: list[dict[str, Any]]) -> None:
    for proc in processes:
        process = proc["process"]
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for proc in processes:
        process = proc["process"]
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        proc["status"] = "terminated"
    _write_service_pid_report(run_dir, processes)


@contextlib.contextmanager
def managed_sglang_service(spec: dict[str, Any]):
    service = spec.get("sglang_service") or {}
    processes: list[dict[str, Any]] = []
    run_dir = Path(spec["run_dir"])
    if service.get("mode") != "external":
        yield
        return
    try:
        if service.get("managed"):
            run_dir.mkdir(parents=True, exist_ok=True)
            service = _stage_sglang_model(service, run_dir)
            service_env = dict(spec["env"])
            service_env.update(service.get("env") or {})
            service_cwd = Path(service.get("cwd") or spec["cwd"])
            for index, command in enumerate(service["start_commands"]):
                log_path = run_dir / f"sglang_service_{index}.log"
                log_handle = log_path.open("a")
                log_handle.write(f"\n# started_utc={utc_now()}\n")
                log_handle.write(" ".join(command) + "\n")
                log_handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=service_cwd,
                    env=service_env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                log_handle.close()
                processes.append(
                    {
                        "process": process,
                        "command": command,
                        "log_path": log_path,
                        "status": "running",
                    }
                )
            _write_service_pid_report(run_dir, processes)
        _wait_for_service_health(service, processes)
        _release_sglang_staged_files(service, run_dir)
        yield
    finally:
        if processes and service.get("terminate_on_exit", True):
            _terminate_service_processes(run_dir, processes)
        elif processes:
            for proc in processes:
                if proc["process"].poll() is not None:
                    proc["status"] = f"exited:{proc['process'].returncode}"
            _write_service_pid_report(run_dir, processes)


def _run_logged(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n# started_utc={utc_now()}\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.write(f"# completed_utc={utc_now()}\n")


def _read_gate_passed(report_json: Path) -> bool | None:
    try:
        report = json.loads(report_json.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    gate = report.get("gate")
    if isinstance(gate, dict) and "passed" in gate:
        return bool(gate["passed"])
    if "gate_passed" in report:
        return bool(report["gate_passed"])
    if "passed" in report:
        return bool(report["passed"])
    return None


def apply_checkpoint_retention(run_spec: dict[str, Any]) -> dict[str, Any] | None:
    retention = run_spec.get("checkpoint_retention") or {}
    if not retention.get("enabled"):
        return None
    checkpoint_root = Path(retention["checkpoint_root"])
    if not checkpoint_root.is_dir():
        raise QueueError(f"checkpoint retention root does not exist: {checkpoint_root}")
    candidates = sorted(
        path
        for path in checkpoint_root.glob("iter_*")
        if path.is_dir() and path.name.removeprefix("iter_").isdigit()
    )
    by_iteration = {int(path.name.removeprefix("iter_")): path for path in candidates}
    missing = [value for value in retention["keep_iterations"] if value not in by_iteration]
    if missing:
        raise QueueError(f"checkpoint retention keep iterations are missing: {missing}")
    protected = {by_iteration[value] for value in retention["keep_iterations"]}
    plan = build_cleanup_plan(
        RetentionPolicy(output_root=checkpoint_root, protected=protected),
        candidates=candidates,
    )
    return execute_cleanup(
        plan,
        manifest_path=retention["manifest_path"],
        apply=bool(retention["apply"]),
    )


def validate_retention_protected_iterations(run_spec: dict[str, Any]) -> None:
    retention = run_spec.get("checkpoint_retention") or {}
    if not retention.get("enabled"):
        return
    required = {
        int(eval_spec["iteration"])
        for eval_spec in run_spec.get("post_eval_specs") or []
    }
    preregistration = run_spec.get("preregistration") or {}
    required.update(
        int(value) for value in preregistration.get("checkpoint_iterations") or []
    )
    retained = {int(value) for value in retention.get("keep_iterations") or []}
    missing = sorted(required - retained)
    if missing:
        raise QueueError(
            "checkpoint retention would remove evaluated or registered "
            f"iterations: {missing}"
        )


def run_roundtrip_post_eval(
    queue_path: str | Path,
    index: int,
    queue_doc: dict[str, Any],
    item: dict[str, Any],
    run_spec: dict[str, Any],
) -> dict[str, Any] | None:
    eval_specs = list(run_spec.get("post_eval_specs") or [])
    if not eval_specs:
        eval_specs = build_roundtrip_post_eval_specs(
            queue_doc, item, run_spec, queue_path=queue_path
        )
    if not eval_specs:
        return None
    cwd = Path(run_spec["code_root"])
    env = dict(run_spec["env"])
    results: list[dict[str, Any]] = []
    current_log_path = Path(eval_specs[0]["log_path"])
    update_item(
        queue_path,
        index,
        item_name=str(item["name"]),
        post_eval_status="running",
        post_eval_started_at=utc_now(),
        post_eval_log=str(current_log_path),
    )
    try:
        previous_gate_passed: bool | None = None
        for eval_spec in eval_specs:
            if eval_spec["require_previous_gate_pass"] and previous_gate_passed is not True:
                results.append(
                    {
                        "name": eval_spec["name"],
                        "status": "skipped",
                        "reason": "previous_gate_not_passed",
                        "gate_passed": False,
                    }
                )
                previous_gate_passed = False
                continue
            current_log_path = Path(eval_spec["log_path"])
            if eval_spec.get("converter_command"):
                _run_logged(eval_spec["converter_command"], cwd=cwd, env=env, log_path=current_log_path)
            if eval_spec.get("remote_code_patch_command"):
                _run_logged(
                    eval_spec["remote_code_patch_command"],
                    cwd=cwd,
                    env=env,
                    log_path=current_log_path,
                )
            identity_report = _resolve_roundtrip_generation_identity(eval_spec)
            if eval_spec.get("critic_converter_command"):
                _run_logged(
                    eval_spec["critic_converter_command"],
                    cwd=cwd,
                    env=env,
                    log_path=current_log_path,
                )
            if eval_spec.get("critic_remote_code_patch_command"):
                _run_logged(
                    eval_spec["critic_remote_code_patch_command"],
                    cwd=cwd,
                    env=env,
                    log_path=current_log_path,
                )
            if eval_spec.get("score_command"):
                if eval_spec.get("generation_command"):
                    _run_logged(eval_spec["generation_command"], cwd=cwd, env=env, log_path=current_log_path)
                _run_logged(eval_spec["score_command"], cwd=cwd, env=env, log_path=current_log_path)
            else:
                _run_logged(eval_spec["eval_command"], cwd=cwd, env=env, log_path=current_log_path)
            gate_passed = _read_gate_passed(Path(eval_spec["report_json"]))
            previous_gate_passed = gate_passed
            correlation_spec = None
            if eval_spec["run_reward_gate_correlation"]:
                correlation_spec = build_reward_gate_correlation_post_eval_spec(
                    queue_doc,
                    item,
                    run_spec,
                    eval_spec,
                    queue_path=queue_path,
                )
            if correlation_spec is not None:
                _run_logged(
                    correlation_spec["command"],
                    cwd=cwd,
                    env=env,
                    log_path=Path(correlation_spec["log_path"]),
                )
            if eval_spec["cleanup_hf"]:
                shutil.rmtree(eval_spec["hf_output_dir"], ignore_errors=True)
            if eval_spec["cleanup_critic_hf"] and eval_spec.get("critic_hf_output_dir"):
                shutil.rmtree(eval_spec["critic_hf_output_dir"], ignore_errors=True)
            if eval_spec["cleanup_actor_checkpoint"]:
                shutil.rmtree(eval_spec["input_dir"], ignore_errors=True)
            results.append(
                {
                    "name": eval_spec["name"],
                    "status": "complete",
                    "iteration": eval_spec["iteration"],
                    "report_json": eval_spec["report_json"],
                    "generated_jsonl": eval_spec["generated_jsonl"],
                    "gate_passed": gate_passed,
                    "generation_identity_report": eval_spec.get(
                        "fingerprint_report_json"
                    ),
                    "generation_identity_computed": identity_report is not None,
                    "reward_gate_correlation": (
                        None if correlation_spec is None else correlation_spec["output_json"]
                    ),
                }
            )
        completed_results = [result for result in results if result["status"] == "complete"]
        final_result = completed_results[-1] if completed_results else results[-1]
        gate_passed = bool(results) and all(result.get("gate_passed") is True for result in results)
        update_item(
            queue_path,
            index,
            item_name=str(item["name"]),
            post_eval_status="complete",
            post_eval_completed_at=utc_now(),
            post_eval_results=results,
            report_json=final_result.get("report_json"),
            generated_jsonl=final_result.get("generated_jsonl"),
            gate_passed=gate_passed,
            post_eval_hf_cleaned=all(spec["cleanup_hf"] for spec in eval_specs),
            post_eval_actor_checkpoint_cleaned=all(spec["cleanup_actor_checkpoint"] for spec in eval_specs),
        )
        return {
            "status": "complete",
            "results": results,
            "report_json": final_result.get("report_json"),
            "generated_jsonl": final_result.get("generated_jsonl"),
            "gate_passed": gate_passed,
        }
    except Exception as exc:
        update_item(
            queue_path,
            index,
            item_name=str(item["name"]),
            status="failed",
            failed_at=utc_now(),
            post_eval_status="failed",
            post_eval_failed_at=utc_now(),
            failure=f"post_eval.roundtrip failed: {exc}",
            post_eval_log=str(current_log_path),
        )
        raise


def process_next_item(queue_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    queue_path = Path(queue_path)
    queue_doc = load_queue(queue_path)
    if not dry_run and promote_ready_blocked_items(queue_doc):
        write_queue(queue_path, queue_doc)
        queue_doc = load_queue(queue_path)
    index = next_pending_index(queue_doc)
    if index is None:
        return {"status": "idle", "queue": str(queue_path)}
    item = queue_doc["items"][index]
    try:
        spec = build_run_spec(queue_doc, item, queue_path=queue_path)
        post_eval_specs = build_roundtrip_post_eval_specs(queue_doc, item, spec, queue_path=queue_path)
        spec["post_eval_specs"] = post_eval_specs
        validate_retention_protected_iterations(spec)
        post_eval_missing = post_eval_preflight_missing_paths(post_eval_specs)
        if dry_run:
            post_eval = (
                None
                if not post_eval_specs
                else post_eval_specs[0]
                if len(post_eval_specs) == 1
                else post_eval_specs
            )
            return {
                "status": "dry_run",
                "item_index": index,
                "item_name": item["name"],
                "command": spec["command"],
                "cwd": spec["cwd"],
                "run_dir": spec["run_dir"],
                "resource_total_gpus": spec["resource_total_gpus"],
                "sglang_service": spec["sglang_service"],
                "input_staging": spec.get("input_staging") or [],
                "post_eval": post_eval,
                "preflight_missing_paths": [*preflight_missing_paths(spec), *post_eval_missing],
            }
        spec = _stage_run_inputs(spec)
        preflight_run_spec(spec)
        if post_eval_missing:
            formatted = ", ".join(f"{entry['label']}={entry['path']}" for entry in post_eval_missing)
            raise QueueError(f"required RL post-eval paths are missing: {formatted}")
        launch_contract = freeze_launch_contract(
            queue_path=queue_path,
            queue_doc=queue_doc,
            item_index=index,
            spec=spec,
        )
        if spec.get("runtime_contract_report"):
            runtime_report_path = Path(spec["run_dir"]) / "runtime_contracts.json"
            runtime_report_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_runtime_contracts.v1",
                        "verified_at": utc_now(),
                        "contracts": spec["runtime_contract_report"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        if spec.get("source_provenance") is not None:
            provenance_path = Path(spec["run_dir"]) / "source_provenance.json"
            write_provenance(provenance_path, spec["source_provenance"])
        update_item(
            queue_path,
            index,
            item_name=str(item["name"]),
            status="training",
            started_at=utc_now(),
            run_dir=spec["run_dir"],
            train_log=spec["log_path"],
            resource_total_gpus=spec["resource_total_gpus"],
            source_provenance=spec.get("source_provenance"),
            launch_contract=launch_contract,
        )
        with managed_sglang_service(spec):
            _run_logged(spec["command"], cwd=Path(spec["cwd"]), env=spec["env"], log_path=Path(spec["log_path"]))
        # Retain the declared evaluation inputs before post-eval.  A successful
        # post-eval may intentionally remove its actor input to reclaim storage.
        # Applying retention afterward would then mistake that deliberate cleanup
        # for a missing protected checkpoint.
        retention_result = apply_checkpoint_retention(spec)
        if retention_result is not None:
            update_item(
                queue_path,
                index,
                item_name=str(item["name"]),
                checkpoint_retention_applied=bool(retention_result.get("apply_requested")),
                checkpoint_retention_manifest=spec["checkpoint_retention"]["manifest_path"],
                checkpoint_retention_deleted=retention_result.get("deleted", []),
                checkpoint_retention_phase="pre_post_eval",
            )
        post_eval_result = run_roundtrip_post_eval(queue_path, index, queue_doc, item, spec)
        update_item(
            queue_path,
            index,
            item_name=str(item["name"]),
            status="complete",
            completed_at=utc_now(),
        )
        return {
            "status": "complete",
            "item_index": index,
            "item_name": item["name"],
            "run_dir": spec["run_dir"],
            "train_log": spec["log_path"],
            "post_eval": post_eval_result,
            "checkpoint_retention": retention_result,
        }
    except Exception as exc:
        update_item(
            queue_path,
            index,
            item_name=str(item.get("name")),
            status="failed",
            failed_at=utc_now(),
            failure=str(exc),
        )
        return {"status": "failed", "item_index": index, "item_name": item.get("name"), "failure": str(exc)}


def process_post_eval_only(
    queue_path: str | Path,
    item_name: str,
    *,
    dry_run: bool = False,
    reuse_hf: bool = False,
    fingerprint_workers: int | None = None,
) -> dict[str, Any]:
    """Retry one named item's post-eval from its existing actor checkpoint."""

    queue_path = Path(queue_path)
    queue_doc = load_queue(queue_path)
    matches = [
        (index, item)
        for index, item in enumerate(queue_doc["items"])
        if str(item.get("name")) == str(item_name)
    ]
    if not matches:
        raise QueueError(f"post-eval-only item not found: {item_name}")
    index, item = matches[0]
    launch_contract_ref = item.get("launch_contract") or {}
    contract_path = launch_contract_ref.get("contract_path")
    if not contract_path:
        raise QueueError(
            "post-eval-only requires the immutable launch contract recorded at training launch"
        )
    contract = load_launch_contract(contract_path)
    if int(contract["item_index"]) != index:
        raise QueueError("post-eval-only item index does not match launch contract")
    frozen_queue_doc = contract["queue_snapshot"]
    frozen_item = frozen_queue_doc["items"][index]
    if str(frozen_item.get("name")) != str(item_name):
        raise QueueError("post-eval-only item name does not match launch contract")
    run_spec = contract["resolved_spec"]
    eval_specs = copy.deepcopy(list(run_spec.get("post_eval_specs") or []))
    if not eval_specs:
        raise QueueError(
            "post-eval-only launch contract does not contain frozen evaluation commands"
        )
    if not eval_specs:
        raise QueueError(f"item {item_name!r} does not enable post_eval.roundtrip")
    if fingerprint_workers is not None and fingerprint_workers < 1:
        raise QueueError("post-eval fingerprint workers must be at least 1")
    reused_hf_paths: list[Path] = []
    if reuse_hf:
        for eval_spec in eval_specs:
            hf_output_path = Path(eval_spec["hf_output_dir"])
            _validate_hf_checkpoint_for_reuse(hf_output_path)
            eval_spec["origin_hf_dir"] = str(_infer_origin_hf_dir(eval_spec))
            eval_spec["converter_command"] = None
            reused_hf_paths.append(hf_output_path)
    if fingerprint_workers is not None:
        for eval_spec in eval_specs:
            eval_spec["fingerprint_workers"] = fingerprint_workers
    retry_run_spec = copy.deepcopy(run_spec)
    retry_run_spec["post_eval_specs"] = eval_specs
    checkpoint_paths = [Path(spec["input_dir"]) for spec in eval_specs]
    cleanup_pairs = [
        (Path(spec["input_dir"]), Path(spec["hf_output_dir"]))
        for spec in eval_specs
        if (
            not reuse_hf
            and spec["cleanup_hf"]
            and Path(spec["hf_output_dir"]).exists()
        )
    ]
    if dry_run:
        return {
            "status": "dry_run",
            "mode": "post_eval_only",
            "item_index": index,
            "item_name": item_name,
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "checkpoints_exist": all(path.is_dir() for path in checkpoint_paths),
            "checkpoint_path": str(checkpoint_paths[0]) if len(checkpoint_paths) == 1 else None,
            "checkpoint_exists": checkpoint_paths[0].is_dir() if len(checkpoint_paths) == 1 else None,
            "retry_would_clean_hf": bool(cleanup_pairs),
            "retry_would_reuse_hf": [str(path) for path in reused_hf_paths],
            "fingerprint_workers": fingerprint_workers,
            "post_eval": eval_specs[0] if len(eval_specs) == 1 else eval_specs,
        }

    previous_status = str(item.get("status"))
    previous_failure = item.get("failure")
    try:
        missing_checkpoints = [path for path in checkpoint_paths if not path.is_dir()]
        if missing_checkpoints:
            raise QueueError(f"post-eval-only checkpoints do not exist: {missing_checkpoints}")
        for checkpoint_path, hf_output_path in cleanup_pairs:
            if hf_output_path.resolve() == checkpoint_path.resolve():
                raise QueueError("refusing to clean post-eval HF output because it matches the actor checkpoint")
            shutil.rmtree(hf_output_path)
        update_item(
            queue_path,
            index,
            item_name=str(item_name),
            status="eval_running",
            failure="",
            post_eval_retry_started_at=utc_now(),
            post_eval_retry_previous_status=previous_status,
            post_eval_retry_previous_failure=previous_failure,
            post_eval_retry_hf_cleaned_before_run=bool(cleanup_pairs),
            post_eval_retry_reused_hf=[str(path) for path in reused_hf_paths],
            post_eval_retry_fingerprint_workers=fingerprint_workers,
        )
        result = run_roundtrip_post_eval(
            queue_path,
            index,
            frozen_queue_doc,
            frozen_item,
            retry_run_spec,
        )
        completed_at = utc_now()
        update_item(
            queue_path,
            index,
            item_name=str(item_name),
            status="complete",
            completed_at=completed_at,
            post_eval_retry_completed_at=completed_at,
            failure="",
        )
        return {
            "status": "complete",
            "mode": "post_eval_only",
            "item_index": index,
            "item_name": item_name,
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "checkpoint_path": str(checkpoint_paths[0]) if len(checkpoint_paths) == 1 else None,
            "post_eval": result,
        }
    except Exception as exc:
        update_item(
            queue_path,
            index,
            item_name=str(item_name),
            status="failed",
            failed_at=utc_now(),
            failure=f"post-eval-only failed: {exc}",
        )
        return {
            "status": "failed",
            "mode": "post_eval_only",
            "item_index": index,
            "item_name": item_name,
            "failure": str(exc),
        }


def queue_status(queue_path: str | Path) -> dict[str, Any]:
    queue_doc = load_queue(queue_path)
    index = next_pending_index(queue_doc)
    defaults = queue_doc.get("defaults") or {}
    unapproved = [
        str(item["name"])
        for item in queue_doc["items"]
        if item.get("status") == "pending" and not _launch_approved(defaults, item)
    ]
    return {
        "queue": str(queue_path),
        "counts": status_counts(queue_doc["items"], VALID_STATUSES),
        "next_pending": None if index is None else str(queue_doc["items"][index]["name"]),
        "pending_unapproved": unapproved,
        "items": queue_doc["items"],
    }


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
    parser.add_argument("--dry-run", action="store_true", help="Render the next item without running it.")
    parser.add_argument("--once", action="store_true", help="Process at most one item.")
    parser.add_argument("--run-until-empty", action="store_true", help="Process pending items sequentially and exit when idle.")
    parser.add_argument("--reset-active", action="store_true", help="Reset stale training items to pending.")
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument("--arm", metavar="ITEM_NAME", help="Explicitly approve one queue item for launch.")
    approval.add_argument("--disarm", metavar="ITEM_NAME", help="Remove launch approval from one queue item.")
    parser.add_argument("--approved-by", default=os.environ.get("USER", "unknown"))
    parser.add_argument(
        "--post-eval-only",
        metavar="ITEM_NAME",
        help="Retry round-trip post-eval for one named item without replaying training.",
    )
    parser.add_argument(
        "--reuse-post-eval-hf",
        action="store_true",
        help=(
            "With --post-eval-only, reuse a structurally complete converted HF "
            "checkpoint instead of deleting and reconverting it."
        ),
    )
    parser.add_argument(
        "--post-eval-fingerprint-workers",
        type=int,
        help="Parallel file-hash workers for post-eval generation identity.",
    )
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue to later pending items after a failure.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    if args.arm or args.disarm:
        item = set_item_approval(
            args.queue,
            args.arm or args.disarm,
            approved=bool(args.arm),
            approved_by=args.approved_by,
        )
        print(json.dumps(item, indent=2, sort_keys=True))
        return 0
    if args.reset_active:
        print(json.dumps(reset_active_items(args.queue), indent=2, sort_keys=True))
        return 0
    if args.status:
        print(json.dumps(queue_status(args.queue), indent=2, sort_keys=True))
        return 0
    if args.post_eval_only:
        with queue_lock(args.queue):
            result = process_post_eval_only(
                args.queue,
                args.post_eval_only,
                dry_run=args.dry_run,
                reuse_hf=args.reuse_post_eval_hf,
                fingerprint_workers=args.post_eval_fingerprint_workers,
            )
        print(json.dumps(result, default=str, indent=2, sort_keys=True), flush=True)
        return 1 if result["status"] == "failed" else 0
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
