#!/usr/bin/env python3
"""Render, run, and analyze config-driven Nano round-trip research studies."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nano_roundtrip_eval_config  # noqa: E402


SCHEMA_VERSION = "nano_roundtrip_research_study.v1"
REPORT_SCHEMA_VERSION = "nano_roundtrip_research_report.v1"
ALLOWED_TEXT_SOURCES = ("sft", "rl")
ALLOWED_CRITICS = ("sft", "rl", "independent")
CONTROL_COLUMNS = (
    "token_position",
    "token_id",
    "token_text",
    "token_ids_prefix",
)


class StudyError(ValueError):
    """Raised when a research study is incomplete or identity-unsafe."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise StudyError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise StudyError(f"expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise StudyError(f"study config must use schema_version {SCHEMA_VERSION}")
    for section in ("paths", "texts", "critics", "evaluation", "execution"):
        if not isinstance(config.get(section), dict):
            raise StudyError(f"study config requires mapping: {section}")
    if set(config["texts"]) != set(ALLOWED_TEXT_SOURCES):
        raise StudyError(f"texts must be exactly {ALLOWED_TEXT_SOURCES}")
    if set(config["critics"]) != set(ALLOWED_CRITICS):
        raise StudyError(f"critics must be exactly {ALLOWED_CRITICS}")
    limit = int(config["evaluation"].get("validation_limit", 0))
    if limit <= 0 or limit > 512:
        raise StudyError("evaluation.validation_limit must be in [1, 512]")
    devices = [str(device) for device in config["execution"].get("gpu_devices", [])]
    if not devices or len(set(devices)) != len(devices):
        raise StudyError("execution.gpu_devices must be a non-empty unique list")
    return config


def _record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("split")),
        int(record.get("row_index", -1)),
        str(record.get("sample_uuid") or ""),
        str(record.get("content_family_id") or ""),
    )


def audit_generated_pair(config: dict[str, Any]) -> dict[str, Any]:
    expected_rows = int(config["evaluation"]["validation_limit"])
    summaries: dict[str, Any] = {}
    records_by_source: dict[str, list[dict[str, Any]]] = {}
    for source in ALLOWED_TEXT_SOURCES:
        path = Path(config["texts"][source]["generated_jsonl"])
        records = read_jsonl(path)
        records_by_source[source] = records
        identities = [_record_identity(record) for record in records]
        protocols = {
            str(record.get("generation_protocol_sha256") or "") for record in records
        }
        provenance = {
            str(record.get("generation_provenance_sha256") or "") for record in records
        }
        model_fingerprints = {
            str((record.get("generation_provenance") or {}).get("model_fingerprint") or "")
            for record in records
        }
        checkpoints = {
            str((record.get("generation_provenance") or {}).get("checkpoint") or "")
            for record in records
        }
        expected_model_fingerprint = str(
            config["texts"][source]["av_model_fingerprint"]
        )
        expected_checkpoint = str(config["texts"][source]["av_hf_checkpoint"])
        model_fingerprint_match = model_fingerprints == {
            expected_model_fingerprint
        }
        checkpoint_match = checkpoints == {expected_checkpoint}
        empty_targets = sum(
            not str(record.get("target_explanation") or "").strip()
            for record in records
        )
        empty_real = sum(
            not str(((record.get("controls") or {}).get("real") or {}).get("generated") or "").strip()
            for record in records
        )
        summaries[source] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(records),
            "unique_identities": len(set(identities)),
            "independent_families": len({identity[-1] for identity in identities if identity[-1]}),
            "generation_protocol_sha256": sorted(protocols),
            "generation_provenance_sha256": sorted(provenance),
            "generation_model_fingerprints": sorted(model_fingerprints),
            "expected_model_fingerprint": expected_model_fingerprint,
            "model_fingerprint_match": model_fingerprint_match,
            "generation_checkpoints": sorted(checkpoints),
            "expected_checkpoint": expected_checkpoint,
            "checkpoint_match": checkpoint_match,
            "empty_targets": empty_targets,
            "empty_real_generations": empty_real,
            "passed": (
                len(records) == expected_rows
                and len(set(identities)) == expected_rows
                and len(protocols) == 1
                and "" not in protocols
                and model_fingerprint_match
                and checkpoint_match
                and empty_targets == 0
                and empty_real == 0
            ),
        }
    sft = records_by_source["sft"]
    rl = records_by_source["rl"]
    identities_match = [_record_identity(record) for record in sft] == [
        _record_identity(record) for record in rl
    ]
    targets_match = [record.get("target_explanation") for record in sft] == [
        record.get("target_explanation") for record in rl
    ]
    protocol_match = summaries["sft"]["generation_protocol_sha256"] == summaries["rl"][
        "generation_protocol_sha256"
    ]
    passed = (
        all(summary["passed"] for summary in summaries.values())
        and identities_match
        and targets_match
        and protocol_match
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "phase": "preflight",
        "passed": passed,
        "expected_rows": expected_rows,
        "texts": summaries,
        "cross_text_checks": {
            "row_identity_match": identities_match,
            "target_explanation_match": targets_match,
            "generation_protocol_match": protocol_match,
        },
    }


def materialize_dataset_controls(config: dict[str, Any]) -> dict[str, Any]:
    """Restore exact source-token controls to a row-stable thin eval parquet."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = config["paths"]
    thin_path = Path(paths["thin_validation_parquet"])
    source_path = Path(paths["control_source_parquet"])
    output_path = Path(paths["validation_control_parquet"])
    join_columns = tuple(paths.get("control_join_columns", ("doc_id", "n_raw_tokens")))
    if not join_columns:
        raise StudyError("paths.control_join_columns must not be empty")

    thin = pq.read_table(thin_path)
    required_thin = set(join_columns) | {"detokenized_text_truncated"}
    missing_thin = sorted(required_thin - set(thin.column_names))
    if missing_thin:
        raise StudyError(f"thin validation parquet is missing columns: {missing_thin}")
    if any(column in thin.column_names for column in CONTROL_COLUMNS):
        raise StudyError(
            "thin validation parquet already contains control columns; refusing to "
            "silently replace them"
        )

    source_columns = list(
        dict.fromkeys(
            [*join_columns, "detokenized_text_truncated", *CONTROL_COLUMNS]
        )
    )
    source = pq.read_table(source_path, columns=source_columns)
    missing_source = sorted(set(source_columns) - set(source.column_names))
    if missing_source:
        raise StudyError(f"control source parquet is missing columns: {missing_source}")

    source_rows = source.to_pylist()
    source_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicate_source_keys: list[tuple[Any, ...]] = []
    for row in source_rows:
        key = tuple(row[column] for column in join_columns)
        if key in source_by_key:
            duplicate_source_keys.append(key)
        else:
            source_by_key[key] = row
    if duplicate_source_keys:
        raise StudyError(
            "control source join is ambiguous: "
            f"{len(duplicate_source_keys)} duplicate keys for {join_columns}"
        )

    thin_rows = thin.select(
        [*join_columns, "detokenized_text_truncated"]
    ).to_pylist()
    selected: list[dict[str, Any]] = []
    missing_keys: list[tuple[Any, ...]] = []
    text_mismatch_indices: list[int] = []
    empty_prefix_indices: list[int] = []
    for index, row in enumerate(thin_rows):
        key = tuple(row[column] for column in join_columns)
        source_row = source_by_key.get(key)
        if source_row is None:
            missing_keys.append(key)
            continue
        if row["detokenized_text_truncated"] != source_row["detokenized_text_truncated"]:
            text_mismatch_indices.append(index)
        if not source_row["token_ids_prefix"]:
            empty_prefix_indices.append(index)
        selected.append(source_row)

    passed = not (missing_keys or text_mismatch_indices or empty_prefix_indices)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "phase": "materialize_dataset_controls",
        "created_at": utc_now(),
        "passed": passed,
        "thin_validation_parquet": str(thin_path),
        "control_source_parquet": str(source_path),
        "output_parquet": str(output_path),
        "join_columns": list(join_columns),
        "thin_rows": int(thin.num_rows),
        "source_rows": int(source.num_rows),
        "matched_rows": len(selected),
        "missing_key_count": len(missing_keys),
        "text_mismatch_count": len(text_mismatch_indices),
        "empty_token_prefix_count": len(empty_prefix_indices),
    }
    if not passed:
        write_json(
            Path(paths["output_root"]) / "dataset_controls_materialization.json",
            report,
        )
        raise StudyError(
            "dataset-control materialization failed: "
            f"missing={len(missing_keys)} text_mismatch={len(text_mismatch_indices)} "
            f"empty_prefix={len(empty_prefix_indices)}"
        )

    enriched = thin
    for column in CONTROL_COLUMNS:
        source_type = source.schema.field(column).type
        enriched = enriched.append_column(
            column,
            pa.array([row[column] for row in selected], type=source_type),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(enriched, temporary, compression="zstd")
    temporary.replace(output_path)
    report.update(
        {
            "output_rows": int(enriched.num_rows),
            "output_columns": enriched.column_names,
            "output_sha256": sha256_file(output_path),
        }
    )
    write_json(
        Path(paths["output_root"]) / "dataset_controls_materialization.json",
        report,
    )
    return report


def _copy_file_with_sha256(source: Path, destination: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb") as writer:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(block)
            writer.write(block)
    shutil.copystat(source, destination)
    source_hash = digest.hexdigest()
    staged_hash = sha256_file(destination)
    if source_hash != staged_hash:
        raise StudyError(f"staged critic checksum mismatch: {source}")
    return {
        "relative_path": None,
        "size_bytes": int(source.stat().st_size),
        "sha256": source_hash,
    }


def multipart_file_identity(path: Path, *, part_size_bytes: int) -> dict[str, Any]:
    if part_size_bytes <= 0:
        raise StudyError("multipart part size must be positive")
    sha256 = hashlib.sha256()
    part_md5s: list[bytes] = []
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(part_size_bytes), b""):
            sha256.update(block)
            part_md5s.append(hashlib.md5(block, usedforsecurity=False).digest())
    if not part_md5s:
        part_md5s.append(hashlib.md5(b"", usedforsecurity=False).digest())
    if len(part_md5s) == 1:
        etag = part_md5s[0].hex()
    else:
        etag = (
            hashlib.md5(b"".join(part_md5s), usedforsecurity=False).hexdigest()
            + f"-{len(part_md5s)}"
        )
    return {
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256.hexdigest(),
        "multipart_etag": etag,
        "multipart_part_size_bytes": int(part_size_bytes),
    }


def _stage_s3_critic(
    config: dict[str, Any],
    *,
    name: str,
    spec: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    source_uri = str(spec["stage_s3_uri"])
    expected_size = int(spec["stage_model_size_bytes"])
    expected_etag = str(spec["stage_model_multipart_etag"]).strip('"')
    part_size = int(spec["stage_multipart_part_size_bytes"])
    model_path = destination / "model.safetensors"
    downloaded = False
    if not model_path.is_file() or model_path.stat().st_size != expected_size:
        temporary = destination.with_name(destination.name + ".download")
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)
        temporary.mkdir(parents=True, exist_ok=True)
        code_root = Path(config["paths"]["code_root"])
        command = [
            str(config["execution"].get("python", sys.executable)),
            "scripts/nano_s3.py",
            "cp-down",
            source_uri,
            str(temporary),
            "--recursive",
            "--timeout",
            "0",
            "--max-concurrent-requests",
            str(int(spec.get("stage_s3_max_concurrent_requests", 8))),
            "--multipart-chunksize-mb",
            str(int(spec.get("stage_s3_multipart_chunksize_mb", 128))),
        ]
        completed = subprocess.run(command, cwd=code_root, check=False)
        if completed.returncode != 0:
            raise StudyError(
                f"S3 critic staging failed for {name}: returncode={completed.returncode}"
            )
        temporary.replace(destination)
        model_path = destination / "model.safetensors"
        downloaded = True
    identity = multipart_file_identity(model_path, part_size_bytes=part_size)
    if identity["size_bytes"] != expected_size:
        raise StudyError(f"S3 critic size mismatch for {name}")
    if identity["multipart_etag"] != expected_etag:
        raise StudyError(
            f"S3 critic multipart ETag mismatch for {name}: "
            f"expected={expected_etag} actual={identity['multipart_etag']}"
        )
    files = [
        {
            "relative_path": str(path.relative_to(destination)),
            "size_bytes": int(path.stat().st_size),
            **(
                identity
                if path == model_path
                else {"sha256": sha256_file(path)}
            ),
        }
        for path in sorted(
            path for path in destination.rglob("*") if path.is_file()
        )
        if path.name != ".nano_stage_manifest.json"
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "phase": "critic_stage",
        "created_at": utc_now(),
        "source": source_uri,
        "source_type": "s3",
        "destination": str(destination),
        "downloaded": downloaded,
        "files": files,
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
    }


def stage_critic_checkpoints(config: dict[str, Any]) -> dict[str, Any]:
    """Stage configured critic HF directories on tmpfs with byte verification."""

    items: dict[str, Any] = {}
    for name in ALLOWED_CRITICS:
        spec = config["critics"][name]
        source_value = spec.get("stage_from")
        s3_source_value = spec.get("stage_s3_uri")
        destination_value = spec.get("runtime_checkpoint")
        if not source_value and not s3_source_value and not destination_value:
            items[name] = {"staged": False, "passed": True}
            continue
        if bool(source_value) == bool(s3_source_value):
            raise StudyError(
                f"critic {name} requires exactly one of stage_from or stage_s3_uri"
            )
        if not destination_value:
            raise StudyError(
                f"critic {name} requires runtime_checkpoint when staging"
            )
        destination = Path(destination_value)
        source_identity = str(source_value or s3_source_value)
        manifest_path = destination / ".nano_stage_manifest.json"
        if manifest_path.is_file():
            existing = load_json(manifest_path)
            files = existing.get("files") or []
            reusable = (
                existing.get("source") == source_identity
                and all(
                    (destination / item["relative_path"]).is_file()
                    and (destination / item["relative_path"]).stat().st_size
                    == int(item["size_bytes"])
                    for item in files
                )
            )
            if reusable:
                items[name] = {
                    "staged": True,
                    "reused": True,
                    "passed": True,
                    "source": source_identity,
                    "destination": str(destination),
                    "files": files,
                }
                continue
        if s3_source_value:
            stage_manifest = _stage_s3_critic(
                config,
                name=name,
                spec=spec,
                destination=destination,
            )
            write_json(manifest_path, stage_manifest)
            items[name] = {
                "staged": True,
                "reused": False,
                "passed": True,
                **stage_manifest,
            }
            continue
        source = Path(source_value)
        if not source.is_dir():
            raise StudyError(f"critic stage source is not a directory: {source}")
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)
        temporary.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
            relative = source_file.relative_to(source)
            item = _copy_file_with_sha256(source_file, temporary / relative)
            item["relative_path"] = str(relative)
            files.append(item)
        stage_manifest = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "phase": "critic_stage",
            "created_at": utc_now(),
            "source": str(source),
            "destination": str(destination),
            "files": files,
            "total_bytes": sum(int(item["size_bytes"]) for item in files),
        }
        write_json(temporary / ".nano_stage_manifest.json", stage_manifest)
        temporary.replace(destination)
        items[name] = {
            "staged": True,
            "reused": False,
            "passed": True,
            **stage_manifest,
        }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "phase": "critic_staging",
        "created_at": utc_now(),
        "passed": all(item["passed"] for item in items.values()),
        "critics": items,
    }
    write_json(
        Path(config["paths"]["output_root"]) / "critic_staging.json",
        report,
    )
    return report


def _score_config(
    config: dict[str, Any], *, source: str, critic: str, output_root: Path
) -> dict[str, Any]:
    paths = config["paths"]
    evaluation = config["evaluation"]
    text = config["texts"][source]
    records = read_jsonl(Path(text["generated_jsonl"]))
    if not records or not isinstance(records[0].get("generation_protocol"), dict):
        raise StudyError(f"missing generation protocol for text source: {source}")
    generation_protocol = records[0]["generation_protocol"]
    report_root = output_root / "grid"
    return {
        "schema_version": "nano_roundtrip_eval.v1",
        "python": str(config["execution"].get("python", "/workspace/interp/.venv/bin/python")),
        "paths": {
            "code_root": str(paths["code_root"]),
            "av_hf_checkpoint": str(text["av_hf_checkpoint"]),
            "ar_checkpoint_dir": str(
                config["critics"][critic].get("runtime_checkpoint")
                or config["critics"][critic]["checkpoint"]
            ),
            "train_parquet": str(paths["train_parquet"]),
            "validation_parquet": str(paths["validation_parquet"]),
            "validation_control_parquet": str(
                paths["validation_control_parquet"]
            ),
            "content_family_manifest": str(paths["content_family_manifest"]),
            "content_family_coverage": str(paths["content_family_coverage"]),
            "generated_jsonl": str(text["generated_jsonl"]),
            "report_json": str(report_root / f"{source}_text__{critic}_critic.json"),
            "prediction_cache_npz": str(
                report_root / f"{source}_text__{critic}_critic_predictions.npz"
            ),
        },
        "eval": {
            "eval_splits": ["validation"],
            "validation_limit": int(evaluation["validation_limit"]),
            "generation_controls": list(generation_protocol["controls"]),
            "dataset_controls": ["source_context", "source_raw"],
            "max_new_tokens": int(generation_protocol["max_new_tokens"]),
            "seed": int(generation_protocol["seed"]),
            "generation_prefix": str(generation_protocol["prefix"]),
            "stop_text": generation_protocol.get("stop_text"),
            "generated_text_fallback": str(
                generation_protocol["generated_text_fallback"]
            ),
            "generation_backend": str(generation_protocol["backend"]),
            "generation_workers": int(generation_protocol["worker_count"]),
            "reuse_generated": True,
            "injection_scale": str(generation_protocol["injection_scale"]),
            "ar_batch_size": int(evaluation.get("ar_batch_size", 4)),
            "ar_max_length": int(evaluation.get("ar_max_length", 1152)),
            "torch_dtype": str(generation_protocol["torch_dtype"]),
            "control_margin": float(evaluation.get("control_margin", 1e-6)),
            "min_control_win_fraction": 0.0,
            "min_closed_fraction": float(evaluation.get("min_closed_fraction", 0.95)),
            "min_usable_fraction": float(evaluation.get("min_usable_fraction", 0.99)),
            "bootstrap_samples": int(evaluation.get("bootstrap_samples", 10000)),
            "bootstrap_seed": int(evaluation["seed"]),
            "permutation_samples": int(evaluation.get("permutation_samples", 100000)),
            "permutation_seed": int(evaluation["seed"]),
            "selection_strategy": "family_stratified",
            "selection_seed": int(evaluation["selection_seed"]),
            "require_family_level_inference": True,
            "min_independent_families": int(evaluation["min_independent_families"]),
            "av_model_fingerprint": str(text["av_model_fingerprint"]),
            "av_tokenizer_fingerprint": str(text["av_tokenizer_fingerprint"]),
            "require_generation_protocol_match": True,
            "local_files_only": True,
            "av_device_map": "auto",
            "ar_device_map": "cuda:0",
            "ar_low_cpu_mem_usage": True,
            "collect_ar_device_profile": True,
        },
    }


def render(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    render_dir = Path(config["paths"]["render_dir"])
    output_root = Path(config["paths"]["output_root"])
    render_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for source in ALLOWED_TEXT_SOURCES:
        for critic in ALLOWED_CRITICS:
            name = f"{source}_text__{critic}_critic"
            path = render_dir / f"{name}.yaml"
            path.write_text(
                yaml.safe_dump(
                    _score_config(config, source=source, critic=critic, output_root=output_root),
                    sort_keys=False,
                )
            )
            jobs.append({"name": name, "config": str(path)})
    deterministic = [
        str(value)
        for value in (config.get("semantic_stress") or {}).get(
            "deterministic_transforms", []
        )
    ]
    external = dict((config.get("semantic_stress") or {}).get("external_transforms") or {})
    invariance_jobs: list[dict[str, Any]] = []
    for source in ALLOWED_TEXT_SOURCES:
        for critic in ALLOWED_CRITICS:
            name = f"{source}_text__{critic}_critic__semantic_stress"
            command = [
                str(config["execution"].get("python", "/workspace/interp/.venv/bin/python")),
                "scripts/eval_nano_roundtrip_invariance.py",
                "--generated-jsonl",
                str(config["texts"][source]["generated_jsonl"]),
                "--ar-checkpoint-dir",
                str(
                    config["critics"][critic].get("runtime_checkpoint")
                    or config["critics"][critic]["checkpoint"]
                ),
                "--train-parquet",
                str(config["paths"]["train_parquet"]),
                "--validation-parquet",
                str(config["paths"]["validation_parquet"]),
                "--eval-splits",
                "validation",
                "--validation-limit",
                str(config["evaluation"]["validation_limit"]),
                "--content-family-manifest",
                str(config["paths"]["content_family_manifest"]),
                "--content-family-coverage",
                str(config["paths"]["content_family_coverage"]),
                "--selection-strategy",
                "family_stratified",
                "--selection-seed",
                str(config["evaluation"]["selection_seed"]),
                "--require-family-level-inference",
                "--ar-device-map",
                "cuda:0",
                "--ar-batch-size",
                str(config["evaluation"].get("ar_batch_size", 4)),
                "--ar-max-length",
                str(config["evaluation"].get("ar_max_length", 1152)),
                "--deterministic-transforms",
                *deterministic,
                "--report-json",
                str(output_root / "semantic_stress" / f"{name}.json"),
            ]
            source_external = external.get(source, {})
            if source_external and not isinstance(source_external, dict):
                raise StudyError(
                    f"semantic_stress.external_transforms.{source} must be a mapping"
                )
            for transform, transform_path in source_external.items():
                command.extend(["--transform-jsonl", f"{transform}={transform_path}"])
            invariance_jobs.append({"name": name, "command": command})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_config": str(config_path),
        "rendered_at": utc_now(),
        "grid_jobs": jobs,
        "semantic_stress_jobs": invariance_jobs,
    }
    write_json(render_dir / "render_manifest.json", manifest)
    return manifest


def _run_job(
    *, name: str, command: list[str], device: str, code_root: Path, log_path: Path
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(device),
            "WANDB_MODE": "offline",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": f"{code_root / 'scripts'}:{code_root}",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    with log_path.open("a") as log:
        log.write(f"# started_utc={started} device={device}\n")
        log.write(json.dumps(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=code_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "name": name,
        "device": str(device),
        "started_at": started,
        "completed_at": utc_now(),
        "returncode": int(completed.returncode),
        "log_path": str(log_path),
    }


def run_plan(config_path: Path, config: dict[str, Any], *, phase: str) -> dict[str, Any]:
    preflight = audit_generated_pair(config)
    if not preflight["passed"]:
        raise StudyError("generated-text preflight failed; refusing to run GPU jobs")
    write_json(Path(config["paths"]["output_root"]) / "preflight.json", preflight)
    staging = stage_critic_checkpoints(config)
    if not staging["passed"]:
        raise StudyError("critic staging failed; refusing to run GPU jobs")
    manifest = render(config_path, config)
    code_root = Path(config["paths"]["code_root"])
    output_root = Path(config["paths"]["output_root"])
    devices = [str(device) for device in config["execution"]["gpu_devices"]]
    jobs: list[dict[str, Any]] = []
    resumed_results: list[dict[str, Any]] = []
    if phase == "grid":
        for item in manifest["grid_jobs"]:
            eval_config_path = Path(item["config"])
            eval_config = nano_roundtrip_eval_config.load_config(eval_config_path)
            command = nano_roundtrip_eval_config.build_command(
                eval_config, config_path=eval_config_path
            )
            report_path = Path(eval_config["paths"]["report_json"])
            if bool(config["execution"].get("resume_completed", True)) and report_path.is_file():
                try:
                    completed_report = load_json(report_path)
                    validation = completed_report["splits"]["validation"]
                    complete = int(validation["row_count"]) == int(
                        config["evaluation"]["validation_limit"]
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    complete = False
                if complete:
                    resumed_results.append(
                        {
                            "name": item["name"],
                            "device": None,
                            "started_at": None,
                            "completed_at": utc_now(),
                            "returncode": 0,
                            "log_path": None,
                            "report_path": str(report_path),
                            "resumed": True,
                        }
                    )
                    continue
            jobs.append({**item, "command": command})
    elif phase == "semantic_stress":
        jobs = list(manifest["semantic_stress_jobs"])
    else:
        raise StudyError(f"unknown plan phase: {phase}")
    max_parallel = min(
        int(config["execution"].get("max_parallel", len(devices))),
        len(devices),
        max(1, len(jobs)),
    )
    results: list[dict[str, Any]] = list(resumed_results)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = []
        for index, job in enumerate(jobs):
            device = devices[index % len(devices)]
            futures.append(
                pool.submit(
                    _run_job,
                    name=job["name"],
                    command=list(job["command"]),
                    device=device,
                    code_root=code_root,
                    log_path=output_root / phase / "logs" / f"{job['name']}.log",
                )
            )
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            write_json(
                output_root / phase / "run_state.json",
                {"phase": phase, "updated_at": utc_now(), "results": results},
            )
    passed = len(results) == len(jobs) + len(resumed_results) and all(
        result["returncode"] == 0 for result in results
    )
    report = {"phase": phase, "passed": passed, "jobs": sorted(results, key=lambda x: x["name"])}
    write_json(output_root / phase / "run_report.json", report)
    return report


def _variant_nmse(report: dict[str, Any], variant: str) -> float:
    value = report["splits"]["validation"]["variants"][variant]["directional_mse"]
    return float(value)


def _family_bootstrap_difference(
    left_report: dict[str, Any], right_report: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    left = left_report["splits"]["validation"]
    right = right_report["splits"]["validation"]
    if left["row_indices"] != right["row_indices"]:
        raise StudyError("paired reports have different row indices")
    if left["content_family_ids"] != right["content_family_ids"]:
        raise StudyError("paired reports have different content-family identities")
    left_losses = np.asarray(left["rowwise_directional_mse"]["av_real"], dtype=np.float64)
    right_losses = np.asarray(right["rowwise_directional_mse"]["av_real"], dtype=np.float64)
    differences = left_losses - right_losses
    result = _family_bootstrap_values(
        differences,
        left["content_family_ids"],
        samples=samples,
        seed=seed,
        positive_interpretation="lower right-hand loss is better",
    )
    result["positive_means_lower_right_is_better"] = True
    return result


def _family_bootstrap_values(
    differences: np.ndarray,
    content_family_ids: list[Any],
    *,
    samples: int,
    seed: int,
    positive_interpretation: str,
) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=np.float64)
    family_values: dict[str, list[float]] = {}
    if len(content_family_ids) != len(differences):
        raise StudyError("content-family and rowwise metric lengths differ")
    for family, value in zip(content_family_ids, differences):
        family_values.setdefault(str(family), []).append(float(value))
    family_means = np.asarray(
        [np.mean(values) for _, values in sorted(family_values.items())], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(family_means), size=(samples, len(family_means)))
    means = family_means[indices].mean(axis=1)
    return {
        "positive_interpretation": positive_interpretation,
        "families": len(family_means),
        "rows": len(differences),
        "mean": float(differences.mean()),
        "family_equal_weight_mean": float(family_means.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "row_win_fraction": float(np.mean(differences > 0.0)),
    }


def _semantic_transform_names(config: dict[str, Any]) -> list[str]:
    semantic = config.get("semantic_stress") or {}
    deterministic = [str(value) for value in semantic.get("deterministic_transforms", [])]
    external = semantic.get("external_transforms") or {}
    source_keys = [set((external.get(source) or {}).keys()) for source in ALLOWED_TEXT_SOURCES]
    if source_keys and any(keys != source_keys[0] for keys in source_keys[1:]):
        raise StudyError("semantic external transform names differ across text sources")
    return ["real", *deterministic, *sorted(source_keys[0] if source_keys else set())]


def _semantic_split(report: dict[str, Any]) -> dict[str, Any]:
    return report["combined_score"]["splits"]["validation"]


def _semantic_losses(split: dict[str, Any], transform: str) -> np.ndarray:
    variant = "av_real" if transform == "real" else f"av_{transform}"
    try:
        values = split["rowwise_directional_mse"][variant]
    except KeyError as exc:
        raise StudyError(f"semantic report is missing variant {variant}") from exc
    return np.asarray(values, dtype=np.float64)


def analyze_semantic_stress(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    transforms = _semantic_transform_names(config)
    samples = int(config["evaluation"].get("bootstrap_samples", 10000))
    seed = int(config["evaluation"]["seed"])
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    matrix: dict[str, Any] = {}
    transform_effects: dict[str, Any] = {}
    for source in ALLOWED_TEXT_SOURCES:
        matrix[source] = {}
        transform_effects[source] = {}
        for critic in ALLOWED_CRITICS:
            path = (
                output_root
                / "semantic_stress"
                / f"{source}_text__{critic}_critic__semantic_stress.json"
            )
            report = load_json(path)
            reports[(source, critic)] = report
            split = _semantic_split(report)
            if int(split["row_count"]) != int(config["evaluation"]["validation_limit"]):
                raise StudyError(f"semantic report has incomplete rows: {path}")
            if int(split["independent_family_count"]) < int(
                config["evaluation"]["min_independent_families"]
            ):
                raise StudyError(f"semantic report has insufficient families: {path}")
            baseline = _semantic_losses(split, "real")
            matrix[source][critic] = {}
            transform_effects[source][critic] = {}
            for transform in transforms:
                losses = _semantic_losses(split, transform)
                matrix[source][critic][transform] = float(losses.mean())
                transform_effects[source][critic][transform] = (
                    _family_bootstrap_values(
                        losses - baseline,
                        split["content_family_ids"],
                        samples=samples,
                        seed=seed,
                        positive_interpretation=(
                            "transformed text has higher loss than original text"
                        ),
                    )
                )

    actor_gain_after_transform: dict[str, Any] = {}
    actor_gain_retention: dict[str, Any] = {}
    for critic in ALLOWED_CRITICS:
        sft = _semantic_split(reports[("sft", critic)])
        rl = _semantic_split(reports[("rl", critic)])
        if sft["row_indices"] != rl["row_indices"]:
            raise StudyError("semantic SFT and RL reports have different row identities")
        if sft["content_family_ids"] != rl["content_family_ids"]:
            raise StudyError("semantic SFT and RL reports have different family identities")
        actor_gain_after_transform[critic] = {}
        for transform in transforms:
            gain = _semantic_losses(sft, transform) - _semantic_losses(rl, transform)
            actor_gain_after_transform[critic][transform] = _family_bootstrap_values(
                gain,
                sft["content_family_ids"],
                samples=samples,
                seed=seed,
                positive_interpretation="RL-generated text has lower loss than SFT text",
            )
        baseline_gain = actor_gain_after_transform[critic]["real"]["mean"]
        actor_gain_retention[critic] = {
            transform: (
                item["mean"] / baseline_gain if baseline_gain > 0.0 else None
            )
            for transform, item in actor_gain_after_transform[critic].items()
        }

    cross_source_contrasts: dict[str, Any] = {}
    for item in (config.get("semantic_stress") or {}).get(
        "cross_source_contrasts", []
    ):
        name = str(item["name"])
        if name in cross_source_contrasts:
            raise StudyError(f"duplicate semantic cross-source contrast: {name}")
        left_spec = dict(item["left"])
        right_spec = dict(item["right"])
        for side, spec in (("left", left_spec), ("right", right_spec)):
            if spec.get("source") not in ALLOWED_TEXT_SOURCES:
                raise StudyError(f"unknown {side} text source in contrast {name}")
            if spec.get("transform") not in transforms:
                raise StudyError(f"unknown {side} transform in contrast {name}")
        cross_source_contrasts[name] = {
            "left": left_spec,
            "right": right_spec,
            "by_critic": {},
        }
        for critic in ALLOWED_CRITICS:
            left_split = _semantic_split(reports[(left_spec["source"], critic)])
            right_split = _semantic_split(reports[(right_spec["source"], critic)])
            if left_split["row_indices"] != right_split["row_indices"]:
                raise StudyError(f"contrast {name} has different row identities")
            if left_split["content_family_ids"] != right_split["content_family_ids"]:
                raise StudyError(f"contrast {name} has different family identities")
            differences = _semantic_losses(
                left_split, left_spec["transform"]
            ) - _semantic_losses(right_split, right_spec["transform"])
            cross_source_contrasts[name]["by_critic"][critic] = (
                _family_bootstrap_values(
                    differences,
                    left_split["content_family_ids"],
                    samples=samples,
                    seed=seed,
                    positive_interpretation=str(item["positive_interpretation"]),
                )
            )

    report = {
        "schema_version": "nano_roundtrip_semantic_stress_analysis.v1",
        "phase": "semantic_stress_analysis",
        "created_at": utc_now(),
        "validation_only": True,
        "transforms": transforms,
        "matrix_directional_nmse": matrix,
        "transform_effects": transform_effects,
        "actor_gain_after_transform": actor_gain_after_transform,
        "actor_gain_retention": actor_gain_retention,
        "cross_source_contrasts": cross_source_contrasts,
        "interpretation_guard": (
            "External transforms are model-generated approximations to semantic "
            "equivalence. Treat robustness as evidence about the text channel, not "
            "proof that every transformation preserved all meaning."
        ),
    }
    write_json(output_root / "analysis" / "semantic_stress_analysis.json", report)
    return report


def _text_forensics(records: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [
        str(((record.get("controls") or {}).get("real") or {}).get("parsed", {}).get("explanation") or "")
        for record in records
    ]
    word_counts = np.asarray([len(text.split()) for text in texts], dtype=np.float64)
    char_counts = np.asarray([len(text) for text in texts], dtype=np.float64)
    unique_fractions = np.asarray(
        [len(set(text.lower().split())) / max(1, len(text.split())) for text in texts],
        dtype=np.float64,
    )
    closed = [
        bool(((record.get("controls") or {}).get("real") or {}).get("parsed", {}).get("closed"))
        for record in records
    ]
    return {
        "rows": len(records),
        "words_mean": float(word_counts.mean()),
        "words_median": float(np.median(word_counts)),
        "words_p95": float(np.quantile(word_counts, 0.95)),
        "characters_mean": float(char_counts.mean()),
        "unique_word_fraction_mean": float(unique_fractions.mean()),
        "closed_fraction": float(np.mean(closed)),
        "list_or_heading_fraction": float(
            np.mean([bool("\n-" in text or "\n*" in text or re_heading(text)) for text in texts])
        ),
    }


def re_heading(text: str) -> bool:
    return any(line.lstrip().startswith("#") for line in text.splitlines())


def _write_blinded_packet(
    sft_records: list[dict[str, Any]], rl_records: list[dict[str, Any]], *, output_root: Path, seed: int
) -> None:
    rng = random.Random(seed)
    packet: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    if len(sft_records) != len(rl_records):
        raise StudyError("SFT and RL generated-text row counts differ")
    for sft, rl in zip(sft_records, rl_records):
        row_key = f"validation:{int(sft['row_index'])}"
        values = {
            "sft": str(sft["controls"]["real"]["parsed"]["explanation"]),
            "rl": str(rl["controls"]["real"]["parsed"]["explanation"]),
        }
        order = ["sft", "rl"]
        rng.shuffle(order)
        packet.append(
            {
                "row_key": row_key,
                "text_a": values[order[0]],
                "text_b": values[order[1]],
                "questions": [
                    "Which text is clearer to a human reader?",
                    "Which text better preserves the teacher explanation's meaning?",
                    "Does either text contain opaque or code-like wording?",
                ],
            }
        )
        key.append({"row_key": row_key, "text_a": order[0], "text_b": order[1]})
    write_json(output_root / "analysis" / "blinded_text_packet.json", packet)
    write_json(output_root / "analysis" / "blinded_text_key.json", key)


def analyze(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    matrix: dict[str, Any] = {}
    for source in ALLOWED_TEXT_SOURCES:
        matrix[source] = {}
        for critic in ALLOWED_CRITICS:
            path = output_root / "grid" / f"{source}_text__{critic}_critic.json"
            report = load_json(path)
            reports[(source, critic)] = report
            split = report["splits"]["validation"]
            matrix[source][critic] = {
                "rows": int(split["row_count"]),
                "families": int(split["independent_family_count"]),
                "text_directional_nmse": _variant_nmse(report, "av_real"),
                "teacher_directional_nmse": _variant_nmse(report, "teacher"),
                "source_context_directional_nmse": _variant_nmse(report, "source_context"),
                "source_raw_directional_nmse": _variant_nmse(report, "source_raw"),
            }
    samples = int(config["evaluation"].get("bootstrap_samples", 10000))
    seed = int(config["evaluation"]["seed"])
    actor_gains = {
        critic: _family_bootstrap_difference(
            reports[("sft", critic)], reports[("rl", critic)], samples=samples, seed=seed
        )
        for critic in ALLOWED_CRITICS
    }
    joint_gain = (
        matrix["sft"]["sft"]["text_directional_nmse"]
        - matrix["rl"]["rl"]["text_directional_nmse"]
    )
    transfer = {
        critic: actor_gains[critic]["mean"] / joint_gain if joint_gain > 0 else None
        for critic in ALLOWED_CRITICS
    }
    sft_records = read_jsonl(Path(config["texts"]["sft"]["generated_jsonl"]))
    rl_records = read_jsonl(Path(config["texts"]["rl"]["generated_jsonl"]))
    _write_blinded_packet(sft_records, rl_records, output_root=output_root, seed=seed)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "phase": "grid_analysis",
        "created_at": utc_now(),
        "matrix": matrix,
        "actor_gain_by_fixed_critic": actor_gains,
        "matched_joint_gain": joint_gain,
        "actor_gain_transfer_fraction_of_joint": transfer,
        "coadaptation_interaction_rl_minus_sft_critic": (
            actor_gains["rl"]["mean"] - actor_gains["sft"]["mean"]
        ),
        "text_forensics": {
            "sft": _text_forensics(sft_records),
            "rl": _text_forensics(rl_records),
        },
        "interpretation_guard": (
            "The matched joint gain is not actor-only. Transfer through fixed SFT and "
            "independent critics estimates semantic portability; the interaction term "
            "estimates critic-specific co-adaptation."
        ),
    }
    write_json(output_root / "analysis" / "grid_analysis.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "materialize-controls",
            "stage-critics",
            "render",
            "preflight",
            "run-grid",
            "run-semantic-stress",
            "analyze",
            "analyze-semantic-stress",
        ),
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "materialize-controls":
        result = materialize_dataset_controls(config)
    elif args.command == "stage-critics":
        result = stage_critic_checkpoints(config)
    elif args.command == "render":
        result = render(args.config, config)
    elif args.command == "preflight":
        result = audit_generated_pair(config)
        write_json(Path(config["paths"]["output_root"]) / "preflight.json", result)
    elif args.command == "run-grid":
        result = run_plan(args.config, config, phase="grid")
    elif args.command == "run-semantic-stress":
        result = run_plan(args.config, config, phase="semantic_stress")
    elif args.command == "analyze-semantic-stress":
        result = analyze_semantic_stress(config)
    else:
        result = analyze(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
