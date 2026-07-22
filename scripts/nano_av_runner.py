#!/usr/bin/env python3
"""Config-driven entrypoint helpers for Nano AV Miles/FSDP2 runs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_av_materialize_splits import materialize_splits  # noqa: E402


SPLIT_CACHE_SCHEMA_VERSION = 1
CRITIC_INITIALIZATION_VERIFICATION_SCHEMA = (
    "nano_critic_initialization_verification.v1"
)
MATERIALIZED_SPLIT_FILENAMES = (
    "train.parquet",
    "validation.parquet",
    "test.parquet",
    "train_padded.parquet",
)


class SpecValidationError(ValueError):
    """Raised when a Nano AV run spec is internally unsafe or incomplete."""


def _verify_critic_initialization_report(
    report_path: str | Path,
    *,
    critic_init_model_id: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(report_path)
    if not path.is_file():
        raise SpecValidationError(
            f"critic initialization verification report is missing: {path}"
        )
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecValidationError(
            f"critic initialization verification report is unreadable: {path}"
        ) from exc
    if report.get("schema_version") != CRITIC_INITIALIZATION_VERIFICATION_SCHEMA:
        raise SpecValidationError(
            "critic initialization verification has an unsupported schema"
        )
    if report.get("passed") is not True:
        raise SpecValidationError("critic initialization verification did not pass")
    if critic_init_model_id is not None:
        manifest_path = Path(critic_init_model_id) / "critic_initialization.json"
        if not manifest_path.is_file():
            raise SpecValidationError(
                f"critic initialization manifest is missing: {manifest_path}"
            )
        source = (report.get("sources") or {}).get("independent") or {}
        if source.get("path") != str(manifest_path):
            raise SpecValidationError(
                "critic initialization verification is not bound to the selected critic"
            )
        if source.get("sha256") != _sha256_file(manifest_path):
            raise SpecValidationError(
                "critic initialization manifest hash differs from the verification report"
            )
    return report


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _required_mapping(spec: dict[str, Any], key: str) -> dict[str, Any]:
    value = spec.get(key)
    if not isinstance(value, dict):
        raise SpecValidationError(f"{key!r} section is required")
    return value


def _fractions(dataset: dict[str, Any]) -> tuple[float, float, float]:
    values = dataset.get("fractions") or {}
    try:
        train = float(values["train"])
        validation = float(values["validation"])
        test = float(values["test"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SpecValidationError("dataset.fractions must define train, validation, and test") from exc
    if not math.isclose(train + validation + test, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SpecValidationError("dataset.fractions must sum to 1.0")
    return train, validation, test


def _objective(spec: dict[str, Any]) -> str:
    training = spec.get("training") or {}
    return str(training.get("objective", "av_sft"))


def _input_path_key(objective: str) -> str:
    if objective == "av_sft":
        return "input_av_sft"
    if objective == "ar_sft":
        return "input_ar_sft"
    raise SpecValidationError(f"training.objective must be av_sft or ar_sft, got {objective!r}")


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    run = _required_mapping(spec, "run")
    paths = _required_mapping(spec, "paths")
    dataset = _required_mapping(spec, "dataset")
    training = _required_mapping(spec, "training")
    checkpoint = _required_mapping(spec, "checkpoint")
    objective = _objective(spec)

    try:
        input_key = _input_path_key(objective)
    except SpecValidationError as exc:
        input_key = "input_av_sft"
        errors.append(str(exc))

    for section_name, section, keys in (
        ("run", run, ("name", "experiment_class", "output_root", "wandb_mode")),
        ("paths", paths, ("code_root", "miles_root", "model_id", input_key)),
        ("dataset", dataset, ("row_limit", "split_mode", "fractions", "materialize_splits", "final_batch_policy")),
        ("training", training, ("backend", "epochs", "global_batch_size", "micro_batch_size", "rollout_batch_size", "lr", "grad_norm_policy")),
        ("checkpoint", checkpoint, ("save_interval", "keep_last", "save_enabled")),
    ):
        for key in keys:
            if key not in section:
                errors.append(f"{section_name}.{key} is required")
    if objective == "av_sft" and "injection_scale" not in training:
        errors.append("training.injection_scale is required")
    if objective == "ar_sft" and "critic_init_model_id" not in paths:
        errors.append("paths.critic_init_model_id is required")

    train_fraction, validation_fraction, test_fraction = _fractions(dataset)
    experiment_class = str(run.get("experiment_class", ""))
    wandb_mode = str(run.get("wandb_mode", "")).lower()
    grad_norm_policy = str(training.get("grad_norm_policy", "clip"))

    if wandb_mode != "offline":
        errors.append("wandb_mode must be offline")
    if str(training.get("backend")) != "miles_fsdp2":
        errors.append("training.backend must be miles_fsdp2")
    split_mode = str(dataset.get("split_mode"))
    if split_mode not in {"doc", "content_component", "content_family_manifest"}:
        errors.append(
            "dataset.split_mode must be doc, content_component, or content_family_manifest"
        )
    if split_mode == "content_family_manifest" and not dataset.get(
        "content_family_manifest"
    ):
        errors.append(
            "dataset.content_family_manifest is required for content_family_manifest split mode"
        )
    if split_mode == "content_family_manifest" and not dataset.get(
        "content_family_manifest_sha256"
    ):
        errors.append(
            "dataset.content_family_manifest_sha256 is required for content_family_manifest split mode"
        )
    if int(training.get("global_batch_size", 0)) < int(training.get("micro_batch_size", 0)):
        errors.append("global_batch_size must be >= micro_batch_size")
    if int(training.get("rollout_batch_size", 0)) != int(training.get("global_batch_size", -1)):
        errors.append("rollout_batch_size should match global_batch_size for SFT diagnostics")
    if int(checkpoint.get("keep_last", 0)) < 1:
        errors.append("checkpoint.keep_last must be >= 1")
    if grad_norm_policy not in {"clip", "global_clip", "skip_diagnostic"}:
        errors.append("training.grad_norm_policy must be clip, global_clip, or skip_diagnostic")
    moe_routing_impl = str(training.get("moe_routing_impl", "segmented"))
    if moe_routing_impl not in {"segmented", "expert_scan"}:
        errors.append("training.moe_routing_impl must be segmented or expert_scan")
    mamba_kernel_mode = str(training.get("mamba_kernel_mode", "auto"))
    if mamba_kernel_mode not in {"auto", "torch", "unfused_torch_conv"}:
        errors.append(
            "training.mamba_kernel_mode must be auto, torch, or unfused_torch_conv"
        )
    if "resume_steps" in training and int(training["resume_steps"]) <= 0:
        errors.append("training.resume_steps must be positive when set")
    if "num_rollout" in training and int(training["num_rollout"]) <= 0:
        errors.append("training.num_rollout must be positive when set")
    if (
        "distributed_timeout_minutes" in training
        and int(training["distributed_timeout_minutes"]) <= 0
    ):
        errors.append("training.distributed_timeout_minutes must be positive")
    system_metrics = training.get("system_metrics")
    if system_metrics is not None:
        if not isinstance(system_metrics, dict):
            errors.append("training.system_metrics must be a mapping when set")
        else:
            if "interval_steps" in system_metrics and system_metrics["interval_steps"] is not None:
                try:
                    interval_steps = int(system_metrics["interval_steps"])
                except (TypeError, ValueError):
                    errors.append("training.system_metrics.interval_steps must be an integer")
                else:
                    if interval_steps <= 0:
                        errors.append("training.system_metrics.interval_steps must be positive")
            if (
                "nvidia_smi_interval_steps" in system_metrics
                and system_metrics["nvidia_smi_interval_steps"] is not None
            ):
                try:
                    nvidia_smi_interval_steps = int(system_metrics["nvidia_smi_interval_steps"])
                except (TypeError, ValueError):
                    errors.append("training.system_metrics.nvidia_smi_interval_steps must be an integer")
                else:
                    if nvidia_smi_interval_steps < 0:
                        errors.append("training.system_metrics.nvidia_smi_interval_steps must be >= 0")
    for key in ("max_sequence_tokens", "max_response_tokens", "max_tokens_per_gpu", "log_probs_max_tokens_per_gpu"):
        if key in training and training[key] is not None:
            try:
                value = int(training[key])
            except (TypeError, ValueError):
                errors.append(f"training.{key} must be an integer")
                continue
            if value <= 0:
                errors.append(f"training.{key} must be positive")
    if _as_bool(training.get("use_dynamic_batch_size")) and training.get("max_tokens_per_gpu") is None:
        errors.append("training.max_tokens_per_gpu is required when use_dynamic_batch_size is true")
    if _as_bool(training.get("assert_actor_packed_equivalence")) and _objective(spec) != "av_sft":
        errors.append("training.assert_actor_packed_equivalence is only valid for av_sft")
    for key in ("actor_packed_equivalence_rtol", "actor_packed_equivalence_atol"):
        if key in training and training[key] is not None:
            try:
                tolerance = float(training[key])
            except (TypeError, ValueError):
                errors.append(f"training.{key} must be numeric")
            else:
                if tolerance < 0:
                    errors.append(f"training.{key} must be non-negative")
    if (
        _as_bool(training.get("use_dynamic_batch_size"))
        and training.get("max_tokens_per_gpu") is not None
        and training.get("max_sequence_tokens") is not None
    ):
        try:
            max_tokens_per_gpu = int(training["max_tokens_per_gpu"])
            max_sequence_tokens = int(training["max_sequence_tokens"])
        except (TypeError, ValueError):
            pass
        else:
            if max_tokens_per_gpu < max_sequence_tokens and not _as_bool(
                training.get("allow_oversized_dynamic_batch")
            ):
                errors.append(
                    "training.max_tokens_per_gpu must be >= training.max_sequence_tokens "
                    "when use_dynamic_batch_size is true, unless "
                    "training.allow_oversized_dynamic_batch explicitly acknowledges "
                    "the Miles single-sample-overflow behavior"
                )
    if _as_bool(checkpoint.get("finetune")) and not checkpoint.get("resume_from"):
        errors.append("checkpoint.finetune requires checkpoint.resume_from")
    resume_optimizer_state_required = _as_bool(
        checkpoint.get("resume_optimizer_state_required")
    )
    if resume_optimizer_state_required and not checkpoint.get("resume_from"):
        errors.append(
            "checkpoint.resume_optimizer_state_required requires checkpoint.resume_from"
        )
    if resume_optimizer_state_required and _as_bool(checkpoint.get("finetune")):
        errors.append(
            "checkpoint.resume_optimizer_state_required is incompatible with finetune"
        )
    if (
        _as_bool(checkpoint.get("finetune"))
        and "resume_steps" in training
        and _as_bool(checkpoint.get("save_enabled"), True)
    ):
        resume_steps = int(training["resume_steps"])
        save_interval = int(checkpoint.get("save_interval", 0))
        if save_interval <= 0 or save_interval > resume_steps or resume_steps % save_interval != 0:
            errors.append(
                "finetune resume probes must save on the final resumed step: "
                "checkpoint.save_interval must divide training.resume_steps"
            )

    if experiment_class == "small-smoke" and int(dataset.get("row_limit", 0)) > 96:
        errors.append("small-smoke requires row_limit <= 96")
    if experiment_class == "medium-small" and int(dataset.get("row_limit", 0)) > 960:
        errors.append("medium-small requires row_limit <= 960")
    if experiment_class == "complete-performance":
        if int(dataset.get("row_limit", 0)) < 90000:
            errors.append("complete-performance requires row_limit >= 90000")
        if not (
            math.isclose(train_fraction, 0.9, abs_tol=1e-6)
            and math.isclose(validation_fraction, 0.05, abs_tol=1e-6)
            and math.isclose(test_fraction, 0.05, abs_tol=1e-6)
        ):
            errors.append("complete-performance requires train/validation/test = 0.9/0.05/0.05")
        if not _as_bool(dataset.get("materialize_splits")):
            errors.append("complete-performance requires materialized splits")
        if not _as_bool(checkpoint.get("save_enabled"), True):
            errors.append("complete-performance cannot disable checkpoint saves")
        if not (
            _as_bool(checkpoint.get("require_optimizer_state_for_hero"))
            or resume_optimizer_state_required
        ):
            errors.append("complete-performance requires optimizer/scheduler resume state")
        if _as_bool(checkpoint.get("no_save_optim")) and not resume_optimizer_state_required:
            errors.append("complete-performance cannot use no_save_optim")
        if grad_norm_policy == "skip_diagnostic":
            errors.append("skip_diagnostic grad norm is not allowed for complete-performance")
        if _as_bool(training.get("timing_debug")):
            errors.append("complete-performance cannot enable training.timing_debug")
    if _as_bool(dataset.get("cache_materialized_splits")) and not _as_bool(dataset.get("materialize_splits")):
        errors.append("dataset.cache_materialized_splits requires dataset.materialize_splits")
    if _as_bool(dataset.get("verify_materialized_splits")) and not _as_bool(dataset.get("materialize_splits")):
        errors.append("dataset.verify_materialized_splits requires dataset.materialize_splits")
    if str(dataset.get("split_cache_fingerprint", "stat")) not in {"stat", "sha256"}:
        errors.append("dataset.split_cache_fingerprint must be stat or sha256")

    if errors:
        raise SpecValidationError("; ".join(errors))
    return spec


def load_and_validate_spec(path: str | Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text())
    if not isinstance(spec, dict):
        raise SpecValidationError("run spec must be a YAML mapping")
    return validate_spec(spec)


def _verify_resume_optimizer_state(checkpoint_root: str | Path, iteration: int) -> None:
    iteration_dir = Path(checkpoint_root) / f"iter_{int(iteration):07d}"
    required = [
        iteration_dir / "optimizer" / ".metadata",
        iteration_dir / "lr_scheduler" / ".metadata",
    ]
    missing = [str(path.relative_to(iteration_dir)) for path in required if not path.is_file()]
    if missing:
        raise SpecValidationError(
            "checkpoint.resume_optimizer_state_required is missing DCP state at "
            f"{iteration_dir}: {', '.join(missing)}"
        )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "nano-av-run"


def default_run_id(spec: dict[str, Any], *, now: dt.datetime | None = None) -> str:
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    timestamp = now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{_slug(str(spec['run']['name']))}-{timestamp}"


def _sidecar_path_for(parquet_path: Path) -> Path:
    return parquet_path.with_name(parquet_path.name + ".nla_meta.yaml")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_content_family_manifest(
    path: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SpecValidationError(
            f"content family manifest is missing: {manifest_path}"
        )
    actual_sha256 = _sha256_file(manifest_path)
    if actual_sha256 != str(expected_sha256):
        raise SpecValidationError(
            "content family manifest hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise SpecValidationError("content family manifest is not valid JSON") from exc
    if manifest.get("schema_version") != "nano_content_family_manifest.v1":
        raise SpecValidationError(
            "content family manifest must use schema_version nano_content_family_manifest.v1"
        )
    if not manifest.get("doc_assignments") or not manifest.get("family_splits"):
        raise SpecValidationError(
            "content family manifest requires doc_assignments and family_splits"
        )
    return manifest


def _file_signature(path: Path, *, mode: str) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    signature: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if mode == "sha256":
        signature["sha256"] = _sha256_file(path)
    return signature


def _split_cache_key(
    parquet_path: str | Path,
    *,
    dataset: dict[str, Any],
    pad_train_to_multiple: int | None,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> str:
    parquet_path = Path(parquet_path)
    fingerprint_mode = str(dataset.get("split_cache_fingerprint", "stat"))
    payload = {
        "schema_version": SPLIT_CACHE_SCHEMA_VERSION,
        "source": _file_signature(parquet_path, mode=fingerprint_mode),
        "source_sidecar": _file_signature(_sidecar_path_for(parquet_path), mode=fingerprint_mode),
        "content_family_manifest": _file_signature(
            Path(dataset["content_family_manifest"]),
            mode="sha256",
        )
        if dataset.get("content_family_manifest")
        else None,
        "row_limit": int(dataset["row_limit"]),
        "split_mode": str(dataset.get("split_mode", "doc")),
        "seed": int(dataset.get("seed", 42)),
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "final_batch_policy": str(dataset.get("final_batch_policy")),
        "pad_train_to_multiple": pad_train_to_multiple,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _rewrite_split_manifest_paths(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    rewritten = json.loads(json.dumps(manifest))
    rewritten["output_dir"] = str(output_dir)
    split_paths = {
        "train": output_dir / "train.parquet",
        "validation": output_dir / "validation.parquet",
        "test": output_dir / "test.parquet",
    }
    for split_name, path in split_paths.items():
        if split_name in rewritten.get("splits", {}):
            rewritten["splits"][split_name]["path"] = str(path)
    rewritten["train"]["path"] = str(split_paths["train"])
    rewritten["train"]["padded_path"] = str(output_dir / "train_padded.parquet")
    rewritten["validation"] = rewritten["splits"]["validation"]
    rewritten["test"] = rewritten["splits"]["test"]
    return rewritten


def _copy_materialized_split_artifacts(source_dir: Path, destination_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for filename in MATERIALIZED_SPLIT_FILENAMES:
        source = source_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"cached split artifact missing: {source}")
        destination = destination_dir / filename
        shutil.copy2(source, destination)
        source_sidecar = _sidecar_path_for(source)
        if source_sidecar.exists():
            shutil.copy2(source_sidecar, _sidecar_path_for(destination))
    rewritten = _rewrite_split_manifest_paths(manifest, destination_dir)
    (destination_dir / "split_manifest.json").write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n")
    return rewritten


def _split_cache_entry_ready(cache_entry: Path) -> bool:
    manifest_path = cache_entry / "split_manifest.json"
    if not manifest_path.exists():
        return False
    return all((cache_entry / filename).exists() for filename in MATERIALIZED_SPLIT_FILENAMES)


def _materialized_split_paths(split_manifest: dict[str, Any]) -> dict[str, str]:
    return {
        "train": str(split_manifest["train"]["path"]),
        "validation": str(split_manifest["validation"]["path"]),
        "test": str(split_manifest["test"]["path"]),
    }


def _verify_materialized_splits(split_manifest: dict[str, Any], split_dir: Path) -> dict[str, Any]:
    from verify_nano_miles_av_dataset import materialized_split_content_report

    report = materialized_split_content_report(_materialized_split_paths(split_manifest))
    (split_dir / "split_content_verify.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    split_manifest["content_verification"] = report
    (split_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")
    return report


def _prepare_materialized_splits(
    *,
    source_parquet: str | Path,
    split_dir: Path,
    run_output_root: Path,
    dataset: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    train_fraction, validation_fraction, test_fraction = _fractions(dataset)
    pad_multiple = None
    if str(dataset.get("final_batch_policy")) == "pad_with_train_duplicates":
        pad_multiple = int(training["global_batch_size"])
    split_mode = str(dataset.get("split_mode", "doc"))

    if not _as_bool(dataset.get("cache_materialized_splits")):
        split_manifest = materialize_splits(
            source_parquet,
            split_dir,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=int(dataset.get("seed", 42)),
            row_limit=int(dataset["row_limit"]),
            pad_train_to_multiple=pad_multiple,
            split_mode=split_mode,
            content_family_manifest=dataset.get("content_family_manifest"),
        )
        split_manifest["split_cache"] = {"enabled": False}
    else:
        cache_root = Path(dataset.get("split_cache_dir") or run_output_root / "_split_cache")
        cache_key = _split_cache_key(
            source_parquet,
            dataset=dataset,
            pad_train_to_multiple=pad_multiple,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        cache_entry = cache_root / cache_key
        if _split_cache_entry_ready(cache_entry):
            cached_manifest = json.loads((cache_entry / "split_manifest.json").read_text())
            split_manifest = _copy_materialized_split_artifacts(cache_entry, split_dir, cached_manifest)
            cache_hit = True
        else:
            cache_entry.mkdir(parents=True, exist_ok=True)
            cached_manifest = materialize_splits(
                source_parquet,
                cache_entry,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                seed=int(dataset.get("seed", 42)),
                row_limit=int(dataset["row_limit"]),
                pad_train_to_multiple=pad_multiple,
                split_mode=split_mode,
                content_family_manifest=dataset.get("content_family_manifest"),
            )
            split_manifest = _copy_materialized_split_artifacts(cache_entry, split_dir, cached_manifest)
            cache_hit = False
        split_manifest["split_cache"] = {
            "enabled": True,
            "hit": cache_hit,
            "cache_key": cache_key,
            "cache_dir": str(cache_entry),
        }
        (split_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")

    if _as_bool(dataset.get("verify_materialized_splits")):
        _verify_materialized_splits(split_manifest, split_dir)
    return split_manifest


def _configured_run_environment(spec: dict[str, Any]) -> dict[str, str]:
    """Return the allowlisted config-derived environment for driver and Ray actors."""

    training = spec["training"]
    checkpoint = spec["checkpoint"]
    environment = {
        "NLA_KEEP_LOCAL": str(checkpoint["keep_last"]),
    }
    token_cap_env = {
        "max_sequence_tokens": "NLA_SFT_MAX_SEQUENCE_TOKENS",
        "max_response_tokens": "NLA_SFT_MAX_RESPONSE_TOKENS",
    }
    for key, env_name in token_cap_env.items():
        if training.get(key) is not None:
            environment[env_name] = str(int(training[key]))
    system_metrics = training.get("system_metrics") or {}
    if _as_bool(system_metrics.get("enabled")):
        environment["NLA_SYSTEM_METRICS"] = "1"
        environment["NLA_SYSTEM_METRICS_INTERVAL_STEPS"] = str(
            int(system_metrics.get("interval_steps", 1))
        )
        environment["NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS"] = str(
            int(system_metrics.get("nvidia_smi_interval_steps", 0))
        )
        if _as_bool(system_metrics.get("router_entropy")):
            environment["NLA_ROUTER_METRICS"] = "1"
    if training.get("pytorch_cuda_alloc_conf"):
        environment["PYTORCH_CUDA_ALLOC_CONF"] = str(
            training["pytorch_cuda_alloc_conf"]
        )
    if training.get("moe_routing_impl"):
        environment["NLA_MOE_ROUTING_IMPL"] = str(training["moe_routing_impl"])
    if training.get("mamba_kernel_mode"):
        environment["NLA_TRAIN_MAMBA_KERNEL_MODE"] = str(
            training["mamba_kernel_mode"]
        )
    if _as_bool(training.get("cuda_launch_blocking")):
        environment["CUDA_LAUNCH_BLOCKING"] = "1"
    if _as_bool(training.get("assert_actor_packed_equivalence")):
        environment["NLA_ASSERT_ACTOR_PACKED_EQUIV"] = "1"
        environment["NLA_ACTOR_PACKED_EQUIV_RTOL"] = str(
            float(training.get("actor_packed_equivalence_rtol", 0.02))
        )
        environment["NLA_ACTOR_PACKED_EQUIV_ATOL"] = str(
            float(training.get("actor_packed_equivalence_atol", 0.05))
        )
    return environment


def render_miles_command(
    spec: dict[str, Any],
    train_parquet: str | Path,
    run_dir: str | Path,
    *,
    num_rollout: int | None = None,
) -> list[str]:
    run = spec["run"]
    paths = spec["paths"]
    training = spec["training"]
    checkpoint = spec["checkpoint"]
    train_environment = _configured_run_environment(spec)

    run_dir = Path(run_dir)
    wandb_dir = run_dir / "wandb"
    objective = _objective(spec)
    common = [
        "python",
        str(Path(paths["miles_root"]) / "train.py"),
        "--train-backend",
        "fsdp",
        "--custom-actor-cls-path",
        "nla.train_actor.NLAFSDPActor",
        "--data-source-path",
        "nla.data_source.NLADataSource",
        "--prompt-data",
        str(train_parquet),
        "--input-key",
        "prompt",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(training.get("num_gpus", 2)),
        "--num-gpus-per-node",
        str(training.get("num_gpus", 2)),
        "--rollout-batch-size",
        str(training["rollout_batch_size"]),
        "--global-batch-size",
        str(training["global_batch_size"]),
        "--micro-batch-size",
        str(training["micro_batch_size"]),
        "--lr",
        str(training["lr"]),
        "--loss-mask-type",
        str(training.get("loss_mask_type", "qwen")),
        "--attn-implementation",
        str(training.get("attn_implementation", "eager")),
        "--use-wandb",
        "--wandb-mode",
        "offline",
        "--wandb-dir",
        str(wandb_dir),
        "--wandb-project",
        str(run.get("wandb_project", "nano30b-nla-pilot")),
        "--wandb-group",
        str(run.get("wandb_group", "nano-ar-miles-fsdp2-sft" if objective == "ar_sft" else "nano-av-miles-fsdp2-sft")),
        "--disable-wandb-random-suffix",
        "--rollout-shuffle",
        "--train-env-vars",
        json.dumps(train_environment, sort_keys=True, separators=(",", ":")),
    ]
    optional_training_flags = [
        ("min_lr", "--min-lr"),
        ("lr_decay_style", "--lr-decay-style"),
        ("lr_decay_iters", "--lr-decay-iters"),
        ("lr_warmup_iters", "--lr-warmup-iters"),
        ("lr_warmup_fraction", "--lr-warmup-fraction"),
        ("lr_warmup_init", "--lr-warmup-init"),
        ("critic_lr_warmup_iters", "--critic-lr-warmup-iters"),
        ("rollout_seed", "--rollout-seed"),
        ("distributed_timeout_minutes", "--distributed-timeout-minutes"),
    ]
    for key, flag in optional_training_flags:
        if key in training and training[key] is not None:
            common.extend([flag, str(training[key])])
    if objective == "ar_sft":
        command = common + [
            "--loss-type",
            "custom_loss",
            "--custom-loss-function-path",
            "nla.loss.nla_critic_loss",
            "--debug-train-only",
            "--disable-compute-advantages-and-returns",
            "--rollout-function-path",
            "nla.rollout.sft_critic.generate_rollout",
            "--hf-checkpoint",
            str(paths["critic_init_model_id"]),
            "--nla-model-is-critic",
        ]
    else:
        command = common + [
            "--loss-type",
            "sft_loss",
            "--debug-train-only",
            "--disable-compute-advantages-and-returns",
            "--rollout-function-path",
            "nla.rollout.sft_actor.generate_rollout",
            "--hf-checkpoint",
            str(paths["model_id"]),
            "--nla-injection-scale",
            str(training["injection_scale"]),
        ]
    if num_rollout is None:
        command.extend(["--num-epoch", str(training.get("epochs", 1))])
    else:
        command.extend(["--num-rollout", str(num_rollout)])
    if _as_bool(training.get("gradient_checkpointing"), True):
        command.append("--gradient-checkpointing")
    if _as_bool(training.get("timing_debug")):
        command.append("--nla-timing-debug")
    if str(training.get("grad_norm_policy")) == "skip_diagnostic":
        command.append("--nla-skip-grad-norm")
    if str(training.get("grad_norm_policy")) == "global_clip":
        command.append("--no-nla-local-grad-norm")
    if _as_bool(training.get("use_dynamic_batch_size")):
        command.append("--use-dynamic-batch-size")
        command.extend(["--max-tokens-per-gpu", str(int(training["max_tokens_per_gpu"]))])
        if training.get("log_probs_max_tokens_per_gpu") is not None:
            command.extend(
                ["--log-probs-max-tokens-per-gpu", str(int(training["log_probs_max_tokens_per_gpu"]))]
            )
    if _as_bool(checkpoint.get("save_enabled"), True):
        command.extend(["--save", str(run_dir / "checkpoints"), "--save-interval", str(checkpoint["save_interval"])])
    if _as_bool(checkpoint.get("no_save_optim")):
        command.append("--no-save-optim")
    if checkpoint.get("resume_from"):
        command.extend(["--load", str(checkpoint["resume_from"])])
        if _as_bool(checkpoint.get("finetune")):
            command.append("--finetune")
    return command


def prepare_run(spec: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    """Create a run directory, materialize configured splits, and write a launch plan."""

    spec = validate_spec(spec)
    run = spec["run"]
    dataset = spec["dataset"]
    paths = spec["paths"]
    training = spec["training"]
    checkpoint = spec["checkpoint"]

    if str(dataset.get("split_mode")) == "content_family_manifest":
        _verify_content_family_manifest(
            dataset["content_family_manifest"],
            str(dataset["content_family_manifest_sha256"]),
        )

    critic_verification = paths.get("critic_initialization_verification_report")
    if critic_verification:
        _verify_critic_initialization_report(
            critic_verification,
            critic_init_model_id=paths.get("critic_init_model_id"),
        )

    if run_id is None:
        run_id = default_run_id(spec)
    run_dir = Path(run["output_root"]) / run_id
    split_dir = run_dir / "splits"
    run_dir.mkdir(parents=True, exist_ok=True)

    objective = _objective(spec)
    input_key = _input_path_key(objective)
    train_parquet = Path(paths[input_key])
    split_manifest: dict[str, Any] | None = None
    if _as_bool(dataset.get("materialize_splits")):
        split_manifest = _prepare_materialized_splits(
            source_parquet=paths[input_key],
            split_dir=split_dir,
            run_output_root=Path(run["output_root"]),
            dataset=dataset,
            training=training,
        )
        train_info = split_manifest["train"]
        if str(dataset.get("final_batch_policy")) == "pad_with_train_duplicates":
            train_parquet = Path(train_info["padded_path"])
        else:
            train_parquet = Path(train_info["path"])

    train_rows = None
    if split_manifest is not None:
        train_rows = int(split_manifest["train"]["padded_row_count" if str(dataset.get("final_batch_policy")) == "pad_with_train_duplicates" else "row_count"])
    resume_from = checkpoint.get("resume_from")
    resume_steps = training.get("resume_steps")
    if resume_from and resume_steps is not None:
        tracker = Path(resume_from) / "latest_checkpointed_iteration.txt"
        if not tracker.exists():
            raise SpecValidationError(f"checkpoint.resume_from has no tracker: {tracker}")
        resume_start_rollout = int(tracker.read_text().strip())
        if _as_bool(checkpoint.get("resume_optimizer_state_required")):
            _verify_resume_optimizer_state(resume_from, resume_start_rollout)
        if _as_bool(checkpoint.get("finetune")):
            num_rollout = int(resume_steps)
        else:
            num_rollout = resume_start_rollout + int(resume_steps)
    elif "num_rollout" in training:
        resume_start_rollout = None
        num_rollout = int(training["num_rollout"])
    elif train_rows is None:
        resume_start_rollout = None
        num_rollout = None
    else:
        resume_start_rollout = None
        global_batch_size = int(training["global_batch_size"])
        epochs = int(training.get("epochs", 1))
        if train_rows % global_batch_size != 0:
            raise SpecValidationError(
                f"train rows {train_rows} must be divisible by global_batch_size {global_batch_size}; "
                "use final_batch_policy: pad_with_train_duplicates or change batch size"
            )
        num_rollout = (train_rows // global_batch_size) * epochs
        if num_rollout < 1:
            raise SpecValidationError("prepared run has zero optimizer steps")

    environment = _configured_run_environment(spec)
    command = render_miles_command(spec, train_parquet, run_dir, num_rollout=num_rollout)

    plan = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "spec": spec,
        "environment": environment,
        "train_parquet": str(train_parquet),
        "split_manifest": split_manifest,
        "num_rollout": num_rollout,
        "resume_start_rollout": resume_start_rollout,
        "command": command,
    }
    (run_dir / "run_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    (run_dir / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--train-parquet", type=Path, help="Materialized train split parquet for command rendering.")
    parser.add_argument("--run-dir", type=Path, help="Run directory for command rendering.")
    parser.add_argument("--prepare", action="store_true", help="Create run dir, materialize splits, and write run_plan.json.")
    parser.add_argument("--run-id", help="Stable run id for --prepare.")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    spec = load_and_validate_spec(args.spec)
    report: dict[str, Any] = {"spec": str(args.spec), "valid": True}
    if args.prepare:
        report["plan"] = prepare_run(spec, run_id=args.run_id)
    if args.print_command:
        if args.train_parquet is None or args.run_dir is None:
            raise SystemExit("--print-command requires --train-parquet and --run-dir")
        report["command"] = render_miles_command(spec, args.train_parquet, args.run_dir)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
