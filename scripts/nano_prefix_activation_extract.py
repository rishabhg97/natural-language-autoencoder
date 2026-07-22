#!/usr/bin/env python3
"""Extract Nano residual activations from exact token-prefix source rows.

This extractor is intended for teacher-backed hero datasets where the source
parquet already contains the exact `token_ids_prefix` for each supervised row.
It avoids relying on external corpus streaming order, so sharding by rows is
safe and reproducible across layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


TEXT_OPEN = "<text>"
TEXT_CLOSE = "</text>"
DEFAULT_MODEL_ID = "/workspace/interp/models/nano-30b-a3b-bf16-hf"
SCHEMA_VERSION = "nano_prefix_activation_extract.v1"
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS = {":16:8", ":4096:8"}


def add_execution_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared numerical-execution controls used by extraction tools."""

    parser.add_argument(
        "--deterministic-algorithms",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    parser.add_argument("--cublas-workspace-config")
    parser.add_argument("--seed", type=int)


def validate_execution_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve and validate the numerical execution contract for extraction."""

    seed = getattr(args, "seed", None)
    profile = {
        "deterministic_algorithms": bool(
            getattr(args, "deterministic_algorithms", False)
        ),
        "allow_tf32": bool(getattr(args, "allow_tf32", False)),
        "cudnn_benchmark": bool(getattr(args, "cudnn_benchmark", False)),
        "float32_matmul_precision": str(
            getattr(args, "float32_matmul_precision", "highest")
        ),
        "cublas_workspace_config": str(
            getattr(args, "cublas_workspace_config", "") or ""
        ),
        "seed": None if seed is None else int(seed),
    }
    if bool(getattr(args, "publication_mode", False)):
        if not profile["deterministic_algorithms"]:
            raise ValueError(
                "publication extraction requires deterministic_algorithms=true"
            )
        if profile["allow_tf32"]:
            raise ValueError("publication extraction requires allow_tf32=false")
        if profile["cudnn_benchmark"]:
            raise ValueError("publication extraction requires cudnn_benchmark=false")
        if profile["float32_matmul_precision"] != "highest":
            raise ValueError(
                "publication extraction requires float32_matmul_precision=highest"
            )
        if (
            profile["cublas_workspace_config"]
            not in DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS
        ):
            raise ValueError(
                "publication extraction requires a deterministic "
                "cublas_workspace_config"
            )
        if profile["seed"] is None or int(profile["seed"]) < 0:
            raise ValueError("publication extraction requires a nonnegative seed")
    return profile


def configure_extraction_execution(
    args: argparse.Namespace,
    torch_module: Any,
) -> dict[str, Any]:
    """Apply the resolved numerical execution contract before model loading."""

    import numpy as np

    profile = validate_execution_profile(args)
    if profile["cublas_workspace_config"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = profile[
            "cublas_workspace_config"
        ]
    if profile["seed"] is not None:
        seed = int(profile["seed"])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch_module.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(
        bool(profile["deterministic_algorithms"])
    )
    torch_module.backends.cuda.matmul.allow_tf32 = bool(profile["allow_tf32"])
    torch_module.backends.cudnn.allow_tf32 = bool(profile["allow_tf32"])
    torch_module.backends.cudnn.benchmark = bool(profile["cudnn_benchmark"])
    torch_module.set_float32_matmul_precision(
        str(profile["float32_matmul_precision"])
    )
    return profile


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _ensure_script_path() -> None:
    script_dir = _script_dir()
    repo_root = script_dir.parent
    nla_root = repo_root / "external" / "natural_language_autoencoders"
    for path in (script_dir, repo_root, nla_root, nla_root / "Miles"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def _file_provenance(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def publication_provenance_from_args(
    args: argparse.Namespace,
    *,
    execution_profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not bool(getattr(args, "publication_mode", False)):
        return None
    model_report_path = getattr(args, "model_fingerprint_json", None)
    runtime_report_path = getattr(args, "runtime_provenance_json", None)
    if model_report_path is None or runtime_report_path is None:
        raise ValueError(
            "publication extraction requires model fingerprint and runtime provenance JSON"
        )
    if execution_profile is None:
        raise ValueError(
            "publication extraction requires a resolved execution profile"
        )
    model_report_path = Path(model_report_path).resolve()
    runtime_report_path = Path(runtime_report_path).resolve()
    if not model_report_path.is_file():
        raise ValueError(f"model fingerprint report does not exist: {model_report_path}")
    if not runtime_report_path.is_file():
        raise ValueError(f"runtime provenance report does not exist: {runtime_report_path}")
    model_report = json.loads(model_report_path.read_text())
    expected_model_root = Path(args.model_id).resolve()
    reported_model_root = Path(str(model_report.get("root") or "")).resolve()
    if reported_model_root != expected_model_root:
        raise ValueError(
            "model fingerprint root mismatch: "
            f"expected {expected_model_root}, got {reported_model_root}"
        )
    if not model_report.get("sha256") or int(model_report.get("file_count") or 0) <= 0:
        raise ValueError("model fingerprint report is incomplete")
    runtime_payload = json.loads(runtime_report_path.read_text())
    runtime = runtime_payload.get("runtime", runtime_payload)
    if not isinstance(runtime, dict) or not runtime.get("complete") or not runtime.get("sha256"):
        raise ValueError("publication extraction requires a complete runtime fingerprint")
    return {
        "schema_version": "nano_activation_extraction_provenance.v1",
        "model": {
            "root": str(reported_model_root),
            "sha256": str(model_report["sha256"]),
            "file_count": int(model_report["file_count"]),
            "total_bytes": int(model_report.get("total_bytes") or 0),
            "report": _file_provenance(model_report_path),
        },
        "runtime": {
            "sha256": str(runtime["sha256"]),
            "report": _file_provenance(runtime_report_path),
        },
        "source_parquet": _file_provenance(args.source_parquet),
        "execution": dict(execution_profile),
    }


def extract_explanation_from_prompt(prompt: str) -> str:
    start = prompt.find(TEXT_OPEN)
    if start < 0:
        return ""
    start += len(TEXT_OPEN)
    end = prompt.find(TEXT_CLOSE, start)
    if end < 0:
        return ""
    return prompt[start:end].strip()


def _column_value(row: dict[str, Any], name: str, default: Any = None) -> Any:
    value = row.get(name, default)
    return default if value is None else value


def _coerce_prefix(value: Any, *, row_index: int) -> list[int]:
    if value is None:
        raise ValueError(f"row {row_index}: token_ids_prefix is null")
    prefix = [int(item) for item in value]
    if not prefix:
        raise ValueError(f"row {row_index}: token_ids_prefix is empty")
    return prefix


def _record_from_row(row: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    prefix = _coerce_prefix(row.get("token_ids_prefix"), row_index=row_index)
    selected_position = len(prefix) - 1
    n_raw_tokens = int(_column_value(row, "n_raw_tokens", len(prefix)))
    if n_raw_tokens != len(prefix):
        raise ValueError(f"row {row_index}: n_raw_tokens mismatch {n_raw_tokens} != prefix length {len(prefix)}")

    token_position = int(_column_value(row, "token_position", selected_position))
    if token_position != selected_position:
        raise ValueError(f"row {row_index}: token_position mismatch {token_position} != {selected_position}")

    token_id = int(_column_value(row, "token_id", prefix[-1]))
    if token_id != int(prefix[-1]):
        raise ValueError(f"row {row_index}: token_id mismatch {token_id} != prefix tail {prefix[-1]}")

    explanation = row.get("api_explanation")
    if explanation is None:
        explanation = row.get("explanation")
    if explanation is None:
        explanation = row.get("teacher_explanation")
    if explanation is None:
        prompt = row.get("prompt")
        explanation = extract_explanation_from_prompt(prompt) if isinstance(prompt, str) else ""
    explanation = str(explanation).strip()
    if not explanation:
        raise ValueError(f"row {row_index}: empty explanation")

    doc_id = row.get("doc_id")
    if doc_id is None or not str(doc_id):
        raise ValueError(f"row {row_index}: missing doc_id")

    return {
        "source_row": row_index,
        "doc_id": str(doc_id),
        "token_ids_prefix": prefix,
        "selected_position": selected_position,
        "n_raw_tokens": n_raw_tokens,
        "token_position": token_position,
        "token_id": token_id,
        "detokenized_text_truncated": str(row.get("detokenized_text_truncated") or ""),
        "token_text": str(row.get("token_text") or ""),
        "api_explanation": explanation,
    }


def collect_source_records(
    source_parquet: str | Path,
    *,
    row_start: int = 0,
    row_limit: int | None = None,
    batch_size: int = 4096,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_parquet = Path(source_parquet)
    pf = pq.ParquetFile(source_parquet)
    schema_names = set(pf.schema_arrow.names)
    required = {"doc_id", "token_ids_prefix"}
    missing = sorted(required - schema_names)
    if missing:
        raise ValueError(f"{source_parquet} missing required columns: {missing}")

    wanted = [
        "prompt",
        "api_explanation",
        "explanation",
        "teacher_explanation",
        "doc_id",
        "token_ids_prefix",
        "n_raw_tokens",
        "token_position",
        "token_id",
        "token_text",
        "detokenized_text_truncated",
    ]
    columns = [name for name in wanted if name in schema_names]
    records: list[dict[str, Any]] = []
    skipped_before_start = 0
    rows_seen = 0
    rows_read = 0
    rows_kept = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        rows = batch.to_pylist()
        for row in rows:
            source_row = rows_seen
            rows_seen += 1
            if source_row < row_start:
                skipped_before_start += 1
                continue
            if row_limit is not None and rows_read >= row_limit:
                break
            rows_read += 1
            record = _record_from_row(row, row_index=source_row)
            records.append(record)
            rows_kept += 1
        if row_limit is not None and rows_read >= row_limit:
            break

    report = {
        "source": str(source_parquet),
        "source_rows_total": pf.metadata.num_rows,
        "row_start": row_start,
        "row_limit": row_limit,
        "rows_seen": rows_seen,
        "rows_read": rows_read,
        "rows_kept": rows_kept,
        "skipped_before_start": skipped_before_start,
    }
    return records, report


def group_records_by_doc(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups_by_doc: dict[str, dict[str, Any]] = {}
    ordered_groups: list[dict[str, Any]] = []
    for record in records:
        doc_id = str(record["doc_id"])
        group = groups_by_doc.get(doc_id)
        if group is None:
            group = {"doc_id": doc_id, "records": [], "token_ids": []}
            groups_by_doc[doc_id] = group
            ordered_groups.append(group)
        group["records"].append(record)
        prefix = [int(value) for value in record["token_ids_prefix"]]
        if len(prefix) > len(group["token_ids"]):
            group["token_ids"] = prefix

    for group in ordered_groups:
        longest = [int(value) for value in group["token_ids"]]
        for record in group["records"]:
            prefix = [int(value) for value in record["token_ids_prefix"]]
            if longest[: len(prefix)] != prefix:
                raise ValueError(
                    f"doc {group['doc_id']} row {record.get('source_row')}: "
                    "token_ids_prefix is not a prefix of the longest doc prefix"
                )
            selected_position = int(record["selected_position"])
            if selected_position >= len(longest):
                raise ValueError(
                    f"doc {group['doc_id']} row {record.get('source_row')}: "
                    f"selected position {selected_position} outside grouped prefix length {len(longest)}"
                )
    return ordered_groups


def plan_group_batch(groups: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    if not groups:
        raise ValueError("groups must be non-empty")
    max_len = max(len(group["token_ids"]) for group in groups)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    selected_positions: list[tuple[int, int]] = []
    batch_records: list[tuple[int, dict[str, Any]]] = []
    seen_positions: set[tuple[int, int]] = set()
    for batch_idx, group in enumerate(groups):
        prefix = [int(value) for value in group["token_ids"]]
        pad_len = max_len - len(prefix)
        input_ids.append(prefix + [int(pad_token_id)] * pad_len)
        attention_mask.append([1] * len(prefix) + [0] * pad_len)
        for record in group["records"]:
            pos = int(record["selected_position"])
            key = (batch_idx, pos)
            if key not in seen_positions:
                seen_positions.add(key)
                selected_positions.append(key)
            batch_records.append((batch_idx, record))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "selected_positions": selected_positions,
        "batch_records": batch_records,
    }


def _import_runtime_helpers() -> dict[str, Any]:
    _ensure_script_path()
    import torch
    from nano_ar_layer_sweep import _forward_selected_boundaries, parse_boundary_spec
    from nano_extraction_identity import _layer_mask_for_block, _module_execution_device, _move_optional_tensor
    from nano_introspection import (
        add_bool_optional_arg,
        classify_blocker,
        load_config_from_args,
        load_model_from_args,
        load_tokenizer_from_args,
        resolve_nano_module_paths,
    )
    from nano_realdata_stage0_extract import _model_start_device

    return {
        "torch": torch,
        "_forward_selected_boundaries": _forward_selected_boundaries,
        "_layer_mask_for_block": _layer_mask_for_block,
        "_module_execution_device": _module_execution_device,
        "_move_optional_tensor": _move_optional_tensor,
        "parse_boundary_spec": parse_boundary_spec,
        "add_bool_optional_arg": add_bool_optional_arg,
        "classify_blocker": classify_blocker,
        "load_config_from_args": load_config_from_args,
        "load_model_from_args": load_model_from_args,
        "load_tokenizer_from_args": load_tokenizer_from_args,
        "resolve_nano_module_paths": resolve_nano_module_paths,
        "_model_start_device": _model_start_device,
    }


def _activation_schema(d_model: int) -> pa.Schema:
    return pa.schema(
        [
            ("n_raw_tokens", pa.int64()),
            ("detokenized_text_truncated", pa.string()),
            ("activation_vector", pa.list_(pa.float32(), d_model)),
            ("activation_layer", pa.int64()),
            ("doc_id", pa.string()),
            ("token_position", pa.int64()),
            ("token_id", pa.int64()),
            ("token_text", pa.string()),
            ("token_ids_prefix", pa.list_(pa.int32())),
            ("api_explanation", pa.string()),
        ]
    )


def _rows_for_layer(
    *,
    batch_records: list[tuple[int, dict[str, Any]]],
    layer: int,
    captures: dict[tuple[int, int], list[float]],
    tokenizer: Any,
) -> dict[str, list[Any]]:
    rows = {name: [] for name in _activation_schema(len(next(iter(captures.values())))).names}
    for batch_idx, record in batch_records:
        pos = int(record["selected_position"])
        token_id = int(record["token_id"])
        token_text = record.get("token_text") or tokenizer.decode([token_id], skip_special_tokens=False)
        rows["n_raw_tokens"].append(int(record["n_raw_tokens"]))
        rows["detokenized_text_truncated"].append(str(record["detokenized_text_truncated"]))
        rows["activation_vector"].append(captures[(batch_idx, pos)])
        rows["activation_layer"].append(int(layer))
        rows["doc_id"].append(str(record["doc_id"]))
        rows["token_position"].append(int(record["token_position"]))
        rows["token_id"].append(token_id)
        rows["token_text"].append(str(token_text))
        rows["token_ids_prefix"].append([int(value) for value in record["token_ids_prefix"]])
        rows["api_explanation"].append(str(record["api_explanation"]))
    return rows


def _write_sidecar(
    *,
    output: Path,
    layer: int,
    row_count: int,
    d_model: int,
    args: argparse.Namespace,
    source_report: dict[str, Any],
    publication_provenance: dict[str, Any] | None,
) -> None:
    meta = {
        "schema_version": "nla_dataset_meta.v1",
        "dataset_id": f"base_prefix_{Path(args.model_id).name}_R{layer}_rows{source_report['row_start']}_{row_count}",
        "stage": "base_explained",
        "row_count": row_count,
        "created_by": "scripts.nano_prefix_activation_extract",
        "created_at": utc_now(),
        "extraction": {
            "base_model": args.model_id,
            "d_model": d_model,
            "layer_index": layer,
            "norm": "none",
            "source_parquet": str(args.source_parquet),
            "row_start": int(args.row_start),
            "row_limit": args.row_limit,
            "source_mode": "token_ids_prefix",
        },
        "parent_datasets": [str(args.source_parquet)],
    }
    if publication_provenance is not None:
        meta["publication_provenance"] = publication_provenance
    Path(str(output) + ".nla_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))


def extract_prefix_activations(args: argparse.Namespace) -> dict[str, Any]:
    requested_profile = validate_execution_profile(args)
    if requested_profile["cublas_workspace_config"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = requested_profile[
            "cublas_workspace_config"
        ]
    helpers = _import_runtime_helpers()
    torch = helpers["torch"]
    execution_profile = configure_extraction_execution(args, torch)
    layers = helpers["parse_boundary_spec"](args.layers)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_output or output_root / "extract_metadata.json")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "source_parquet": str(args.source_parquet),
        "output_root": str(output_root),
        "layers": layers,
        "row_counts": {f"R_{layer}": 0 for layer in layers},
        "blockers": [],
    }

    try:
        publication_provenance = publication_provenance_from_args(
            args,
            execution_profile=execution_profile,
        )
        if publication_provenance is not None:
            report["publication_provenance"] = publication_provenance
        records, source_report = collect_source_records(
            args.source_parquet,
            row_start=args.row_start,
            row_limit=args.row_limit,
            batch_size=args.source_batch_size,
        )
        report["source"] = source_report
        if not records:
            raise ValueError("source selection produced zero records")
        groups = group_records_by_doc(records)
        report["source"]["doc_groups"] = len(groups)

        tokenizer = helpers["load_tokenizer_from_args"](args)
        config, config_error = helpers["load_config_from_args"](args)
        if config_error is not None:
            report["blockers"].append(helpers["classify_blocker"]("remote-code load", config_error))
        model = helpers["load_model_from_args"](args, config)
        model.eval()
        d_model = int(getattr(config, "hidden_size"))
        schema = _activation_schema(d_model)

        layer_paths = {layer: output_root / f"R_{layer}" / "base.parquet" for layer in layers}
        for path in layer_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

        writers = {layer: pq.ParquetWriter(path, schema) for layer, path in layer_paths.items()}
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        with torch.no_grad():
            for start in range(0, len(groups), args.batch_size):
                batch_groups = groups[start : start + args.batch_size]
                planned = plan_group_batch(batch_groups, pad_token_id=int(pad_token_id))
                input_ids = torch.tensor(
                    planned["input_ids"],
                    dtype=torch.long,
                    device=helpers["_model_start_device"](model),
                )
                attention_mask = torch.tensor(planned["attention_mask"], dtype=torch.long, device=input_ids.device)
                captures = helpers["_forward_selected_boundaries"](
                    helpers=helpers,
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layers=layers,
                    selected_positions=planned["selected_positions"],
                )
                for layer in layers:
                    rows = _rows_for_layer(
                        batch_records=planned["batch_records"],
                        layer=layer,
                        captures=captures[layer],
                        tokenizer=tokenizer,
                    )
                    writers[layer].write_table(pa.Table.from_pydict(rows, schema=schema))
                    report["row_counts"][f"R_{layer}"] += len(planned["batch_records"])
                write_json(metadata_path, report)

        for layer, writer in writers.items():
            writer.close()
            path = layer_paths[layer]
            rows = int(report["row_counts"][f"R_{layer}"])
            _write_sidecar(
                output=path,
                layer=layer,
                row_count=rows,
                d_model=d_model,
                args=args,
                source_report=source_report,
                publication_provenance=publication_provenance,
            )
            write_json(
                path.with_suffix(path.suffix + ".metadata.json"),
                {
                    "schema_version": SCHEMA_VERSION,
                    "layer": layer,
                    "row_count": rows,
                    "output": str(path),
                    "source_parquet": str(args.source_parquet),
                    "publication_provenance": publication_provenance,
                },
            )
        report["completed_at"] = utc_now()
        report["sidecars"] = {f"R_{layer}": str(layer_paths[layer]) + ".nla_meta.yaml" for layer in layers}
        write_json(metadata_path, report)
        return report
    except Exception as exc:
        report["blockers"].append(
            {
                "kind": "prefix_activation_extract",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        )
        write_json(metadata_path, report)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--publication-mode", action="store_true")
    parser.add_argument("--model-fingerprint-json", type=Path)
    parser.add_argument("--runtime-provenance-json", type=Path)
    parser.add_argument("--layers", required=True, help="Layer boundaries, e.g. R33 or R25-R35.")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--source-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-id", default=os.environ.get("NANO_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--model-revision", default=os.environ.get("NANO_MODEL_REVISION"))
    parser.add_argument("--tokenizer-revision", default=os.environ.get("NANO_TOKENIZER_REVISION"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    add_execution_profile_arguments(parser)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(load_mode="full")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = extract_prefix_activations(args)
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
