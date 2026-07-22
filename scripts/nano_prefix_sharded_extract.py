#!/usr/bin/env python3
"""Run exact-prefix activation extraction as ordered, resumable GPU shards."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_ar_layer_sweep import parse_boundary_spec  # noqa: E402
from nano_prefix_activation_extract import add_execution_profile_arguments  # noqa: E402


SCHEMA_VERSION = "nano_prefix_sharded_extract.v1"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_devices(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        devices = [item.strip() for item in value.split(",") if item.strip()]
    else:
        devices = [str(item).strip() for item in value]
    if not devices or any(not item for item in devices):
        raise ValueError("at least one nonempty extraction device is required")
    if len(devices) != len(set(devices)):
        raise ValueError("extraction devices must be unique")
    if any("," in item for item in devices):
        raise ValueError("each extraction device must identify exactly one GPU")
    return devices


def plan_row_shards(
    *,
    total_rows: int,
    row_start: int,
    row_limit: int | None,
    devices: Sequence[str],
) -> list[dict[str, Any]]:
    """Plan balanced contiguous ranges while retaining source row order."""
    total_rows = int(total_rows)
    row_start = int(row_start)
    normalized_devices = parse_devices(devices)
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if row_start < 0 or row_start > total_rows:
        raise ValueError("row_start must fall within the source row count")
    selected_rows = total_rows - row_start if row_limit is None else int(row_limit)
    if selected_rows <= 0:
        raise ValueError("row selection must contain at least one row")
    if row_start + selected_rows > total_rows:
        raise ValueError(
            "row selection exceeds source row count: "
            f"start={row_start}, rows={selected_rows}, total={total_rows}"
        )

    active_devices = normalized_devices[: min(len(normalized_devices), selected_rows)]
    base_size, remainder = divmod(selected_rows, len(active_devices))
    shards: list[dict[str, Any]] = []
    cursor = row_start
    for shard_index, device in enumerate(active_devices):
        row_count = base_size + (1 if shard_index < remainder else 0)
        shards.append(
            {
                "shard_index": shard_index,
                "device": device,
                "row_start": cursor,
                "row_count": row_count,
                "row_end_exclusive": cursor + row_count,
            }
        )
        cursor += row_count
    if cursor != row_start + selected_rows:
        raise AssertionError("internal shard planning error")
    return shards


def _contiguous_document_runs(doc_ids: Sequence[Any]) -> list[dict[str, Any]]:
    if not doc_ids:
        raise ValueError("source contains no document IDs")
    runs: list[dict[str, Any]] = []
    closed_docs: set[str] = set()
    start = 0
    current = str(doc_ids[0])
    if not current:
        raise ValueError("source contains an empty document ID")
    for row_index in range(1, len(doc_ids)):
        doc_id = str(doc_ids[row_index])
        if not doc_id:
            raise ValueError(f"source row {row_index} has an empty document ID")
        if doc_id == current:
            continue
        closed_docs.add(current)
        if doc_id in closed_docs:
            raise ValueError(
                f"document {doc_id!r} occurs in noncontiguous source ranges"
            )
        runs.append(
            {
                "doc_id": current,
                "row_start": start,
                "row_end_exclusive": row_index,
            }
        )
        start = row_index
        current = doc_id
    runs.append(
        {
            "doc_id": current,
            "row_start": start,
            "row_end_exclusive": len(doc_ids),
        }
    )
    return runs


def plan_document_aligned_shards(
    *,
    doc_ids: Sequence[Any],
    row_start: int,
    row_limit: int | None,
    devices: Sequence[str],
    document_batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Balance contiguous shards without changing document batch geometry."""
    normalized_devices = parse_devices(devices)
    total_rows = len(doc_ids)
    row_start = int(row_start)
    row_limit = total_rows - row_start if row_limit is None else int(row_limit)
    document_batch_size = int(document_batch_size)
    if document_batch_size <= 0:
        raise ValueError("document_batch_size must be positive")
    if row_start < 0 or row_start > total_rows:
        raise ValueError("row_start must fall within the source row count")
    if row_limit <= 0 or row_start + row_limit > total_rows:
        raise ValueError("document-aligned row selection exceeds source row count")
    row_end = row_start + row_limit
    if row_start > 0 and str(doc_ids[row_start - 1]) == str(doc_ids[row_start]):
        raise ValueError("row selection starts inside document")
    if row_end < total_rows and str(doc_ids[row_end - 1]) == str(doc_ids[row_end]):
        raise ValueError("row selection ends inside document")

    all_runs = _contiguous_document_runs(doc_ids)
    selected_runs = [
        run
        for run in all_runs
        if int(run["row_start"]) >= row_start
        and int(run["row_end_exclusive"]) <= row_end
    ]
    if not selected_runs:
        raise ValueError("document-aligned selection contains no documents")
    if int(selected_runs[0]["row_start"]) != row_start or int(
        selected_runs[-1]["row_end_exclusive"]
    ) != row_end:
        raise ValueError("document-aligned selection does not cover exact row range")

    max_batch_aligned_shards = math.ceil(
        len(selected_runs) / document_batch_size
    )
    shard_count = min(len(normalized_devices), max_batch_aligned_shards)
    run_cursor = 0
    shards: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        remaining_shards = shard_count - shard_index - 1
        if remaining_shards == 0:
            boundary = len(selected_runs)
        else:
            minimum_remaining_docs = (
                (remaining_shards - 1) * document_batch_size + 1
            )
            maximum_boundary = len(selected_runs) - minimum_remaining_docs
            candidates = list(
                range(
                    run_cursor + document_batch_size,
                    maximum_boundary + 1,
                    document_batch_size,
                )
            )
            if not candidates:
                raise ValueError(
                    "not enough documents to preserve requested batch-aligned shards"
                )
            target_row_end = row_start + round(
                row_limit * (shard_index + 1) / shard_count
            )
            boundary = min(
                candidates,
                key=lambda index: (
                    abs(
                        int(selected_runs[index - 1]["row_end_exclusive"])
                        - target_row_end
                    ),
                    index,
                ),
            )
        shard_start = int(selected_runs[run_cursor]["row_start"])
        shard_end = int(selected_runs[boundary - 1]["row_end_exclusive"])
        shards.append(
            {
                "shard_index": shard_index,
                "device": normalized_devices[shard_index],
                "row_start": shard_start,
                "row_count": shard_end - shard_start,
                "row_end_exclusive": shard_end,
                "doc_count": boundary - run_cursor,
                "alignment": "document_batch",
                "document_batch_size": document_batch_size,
            }
        )
        run_cursor = boundary
    if run_cursor != len(selected_runs):
        raise AssertionError("internal document shard planning error")
    return shards


def _sidecar_path(parquet_path: Path) -> Path:
    return Path(str(parquet_path) + ".nla_meta.yaml")


def _metadata_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(parquet_path.suffix + ".metadata.json")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _read_shard_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing extraction shard: {path}")
    sidecar_path = _sidecar_path(path)
    metadata_path = _metadata_path(path)
    if not sidecar_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing extraction shard metadata for {path}")

    parquet = pq.ParquetFile(path)
    row_count = int(parquet.metadata.num_rows)
    sidecar = yaml.safe_load(sidecar_path.read_text()) or {}
    metadata = json.loads(metadata_path.read_text())
    if int(sidecar.get("row_count", -1)) != row_count:
        raise ValueError(f"shard sidecar row count mismatch for {path}")
    if int(metadata.get("row_count", -1)) != row_count:
        raise ValueError(f"shard metadata row count mismatch for {path}")
    sidecar_provenance = sidecar.get("publication_provenance")
    metadata_provenance = metadata.get("publication_provenance")
    if _canonical(sidecar_provenance) != _canonical(metadata_provenance):
        raise ValueError(f"shard publication provenance mismatch for {path}")
    return {
        "path": path,
        "parquet": parquet,
        "schema": parquet.schema_arrow,
        "row_count": row_count,
        "sidecar": sidecar,
        "metadata": metadata,
        "publication_provenance": sidecar_provenance,
    }


def merge_layer_shards(
    shard_paths: Sequence[str | Path],
    *,
    output: str | Path,
    layer: int,
    expected_rows: int,
    shard_plan: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stream ordered shards into one parquet and atomically publish its metadata."""
    paths = [Path(path) for path in shard_paths]
    if not paths:
        raise ValueError("at least one shard is required")
    descriptors = [_read_shard_descriptor(path) for path in paths]
    expected_rows = int(expected_rows)
    actual_rows = sum(int(item["row_count"]) for item in descriptors)
    if actual_rows != expected_rows:
        raise ValueError(
            f"merged shard rows {actual_rows} != expected rows {expected_rows}"
        )
    if shard_plan is not None and len(shard_plan) != len(descriptors):
        raise ValueError("shard plan count does not match extraction shard count")

    declared_ranges = []
    for descriptor in descriptors:
        extraction = descriptor["sidecar"].get("extraction") or {}
        declared_ranges.append(
            (extraction.get("row_start"), extraction.get("row_limit"))
        )
    if any(start is not None or limit is not None for start, limit in declared_ranges):
        if not all(start is not None for start, _ in declared_ranges):
            raise ValueError("every shard must declare row_start when any shard does")
        cursor = int(declared_ranges[0][0])
        for descriptor, (start, limit) in zip(descriptors, declared_ranges):
            if int(start) != cursor:
                raise ValueError(
                    "declared shard row ranges are not contiguous: "
                    f"expected start {cursor}, got {start} for {descriptor['path']}"
                )
            if limit is not None and int(limit) != int(descriptor["row_count"]):
                raise ValueError(
                    f"declared row_limit mismatch for shard {descriptor['path']}"
                )
            cursor += int(descriptor["row_count"])

    if shard_plan is not None:
        for descriptor, planned in zip(descriptors, shard_plan):
            if int(planned["row_count"]) != int(descriptor["row_count"]):
                raise ValueError(
                    f"shard plan row count mismatch for {descriptor['path']}"
                )
            extraction = descriptor["sidecar"].get("extraction") or {}
            if extraction.get("row_start") is not None and int(
                planned["row_start"]
            ) != int(extraction["row_start"]):
                raise ValueError(
                    f"shard plan row start mismatch for {descriptor['path']}"
                )

    reference_schema = descriptors[0]["schema"]
    reference_provenance = descriptors[0]["publication_provenance"]
    for descriptor in descriptors[1:]:
        if not descriptor["schema"].equals(reference_schema, check_metadata=True):
            raise ValueError(
                f"parquet schema mismatch for shard {descriptor['path']}"
            )
        if _canonical(descriptor["publication_provenance"]) != _canonical(
            reference_provenance
        ):
            raise ValueError(
                f"publication provenance mismatch for shard {descriptor['path']}"
            )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.name}.{uuid.uuid4().hex}.tmp"
    )
    writer = pq.ParquetWriter(temporary_output, reference_schema)
    try:
        for descriptor in descriptors:
            for batch in descriptor["parquet"].iter_batches(batch_size=1024):
                writer.write_batch(batch)
    except BaseException:
        writer.close()
        temporary_output.unlink(missing_ok=True)
        raise
    writer.close()
    if pq.ParquetFile(temporary_output).metadata.num_rows != expected_rows:
        temporary_output.unlink(missing_ok=True)
        raise ValueError("temporary merged parquet failed row-count verification")

    merged_sidecar = copy.deepcopy(descriptors[0]["sidecar"])
    merged_sidecar["row_count"] = expected_rows
    merged_sidecar["created_by"] = "scripts.nano_prefix_sharded_extract"
    merged_sidecar["created_at"] = utc_now()
    extraction = merged_sidecar.setdefault("extraction", {})
    shard_summaries = []
    for shard_index, descriptor in enumerate(descriptors):
        shard_extraction = descriptor["sidecar"].get("extraction") or {}
        summary = {
            "shard_index": shard_index,
            "path": str(descriptor["path"]),
            "row_count": int(descriptor["row_count"]),
            "row_start": shard_extraction.get("row_start"),
            "row_limit": shard_extraction.get("row_limit"),
        }
        if shard_plan is not None:
            summary["device"] = str(shard_plan[shard_index]["device"])
        shard_summaries.append(summary)
    extraction["sharding"] = {
        "schema_version": SCHEMA_VERSION,
        "shard_count": len(descriptors),
        "row_count": expected_rows,
        "ordered_by_source_row": True,
        "shards": shard_summaries,
    }
    starts = [item["row_start"] for item in shard_summaries]
    if all(start is not None for start in starts):
        extraction["row_start"] = int(starts[0])
        extraction["row_limit"] = expected_rows
    base_model = extraction.get("base_model")
    if base_model and extraction.get("row_start") is not None:
        merged_sidecar["dataset_id"] = (
            f"base_prefix_{Path(str(base_model)).name}_R{int(layer)}_"
            f"rows{int(extraction['row_start'])}_{expected_rows}"
        )

    merged_metadata = copy.deepcopy(descriptors[0]["metadata"])
    merged_metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "layer": int(layer),
            "row_count": expected_rows,
            "output": str(output),
            "publication_provenance": reference_provenance,
            "sharding": extraction["sharding"],
        }
    )
    temporary_sidecar = _sidecar_path(output).with_name(
        f".{_sidecar_path(output).name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_metadata = _metadata_path(output).with_name(
        f".{_metadata_path(output).name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_sidecar.write_text(yaml.safe_dump(merged_sidecar, sort_keys=False))
    temporary_metadata.write_text(
        json.dumps(merged_metadata, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_sidecar, _sidecar_path(output))
    os.replace(temporary_metadata, _metadata_path(output))
    os.replace(temporary_output, output)
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": int(layer),
        "row_count": expected_rows,
        "shard_count": len(descriptors),
        "output": str(output),
        "sidecar": str(_sidecar_path(output)),
        "metadata": str(_metadata_path(output)),
        "publication_provenance": reference_provenance,
    }


def _shard_is_complete(
    shard_root: Path,
    *,
    layers: Sequence[int],
    expected_rows: int,
) -> bool:
    try:
        for layer in layers:
            descriptor = _read_shard_descriptor(
                shard_root / f"R_{layer}" / "base.parquet"
            )
            if int(descriptor["row_count"]) != int(expected_rows):
                return False
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def _extract_command(
    args: argparse.Namespace,
    *,
    shard_root: Path,
    row_start: int,
    row_count: int,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "nano_prefix_activation_extract.py"),
        "--source-parquet",
        str(args.source_parquet),
        "--output-root",
        str(shard_root),
        "--metadata-output",
        str(shard_root / "extract_metadata.json"),
        "--layers",
        str(args.layers),
        "--row-start",
        str(row_start),
        "--row-limit",
        str(row_count),
        "--source-batch-size",
        str(args.source_batch_size),
        "--batch-size",
        str(args.batch_size),
        "--model-id",
        str(args.model_id),
        "--device-map",
        "auto",
        "--torch-dtype",
        str(args.torch_dtype),
    ]
    command.append(
        "--deterministic-algorithms"
        if bool(getattr(args, "deterministic_algorithms", False))
        else "--no-deterministic-algorithms"
    )
    command.append(
        "--allow-tf32"
        if bool(getattr(args, "allow_tf32", False))
        else "--no-allow-tf32"
    )
    command.append(
        "--cudnn-benchmark"
        if bool(getattr(args, "cudnn_benchmark", False))
        else "--no-cudnn-benchmark"
    )
    command.extend(
        [
            "--float32-matmul-precision",
            str(getattr(args, "float32_matmul_precision", "highest")),
        ]
    )
    if getattr(args, "cublas_workspace_config", None):
        command.extend(
            [
                "--cublas-workspace-config",
                str(args.cublas_workspace_config),
            ]
        )
    if getattr(args, "seed", None) is not None:
        command.extend(["--seed", str(int(args.seed))])
    if args.model_revision:
        command.extend(["--model-revision", str(args.model_revision)])
    if args.tokenizer_revision:
        command.extend(["--tokenizer-revision", str(args.tokenizer_revision)])
    if args.attn_implementation:
        command.extend(["--attn-implementation", str(args.attn_implementation)])
    if args.local_files_only:
        command.append("--local-files-only")
    command.append(
        "--trust-remote-code" if args.trust_remote_code else "--no-trust-remote-code"
    )
    if args.overwrite:
        command.append("--overwrite")
    if args.publication_mode:
        command.extend(
            [
                "--publication-mode",
                "--model-fingerprint-json",
                str(args.model_fingerprint_json),
                "--runtime-provenance-json",
                str(args.runtime_provenance_json),
            ]
        )
    return command


def run_sharded_extraction(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = int(pq.ParquetFile(args.source_parquet).metadata.num_rows)
    devices = parse_devices(args.devices)
    layers = parse_boundary_spec(args.layers)
    if args.shard_alignment == "row":
        plan = plan_row_shards(
            total_rows=source_rows,
            row_start=args.row_start,
            row_limit=args.row_limit,
            devices=devices,
        )
    else:
        doc_ids = pq.read_table(args.source_parquet, columns=["doc_id"])[
            "doc_id"
        ].to_pylist()
        plan = plan_document_aligned_shards(
            doc_ids=doc_ids,
            row_start=args.row_start,
            row_limit=args.row_limit,
            devices=devices,
            document_batch_size=(
                args.batch_size if args.shard_alignment == "document_batch" else 1
            ),
        )
    selected_rows = sum(int(item["row_count"]) for item in plan)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(
        args.metadata_output or output_root / "sharded_extract_metadata.json"
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "source_parquet": str(args.source_parquet),
        "source_rows": source_rows,
        "selected_rows": selected_rows,
        "layers": layers,
        "devices": devices,
        "shard_alignment": args.shard_alignment,
        "shards": [],
        "merges": {},
        "status": "starting",
    }

    pending: list[dict[str, Any]] = []
    for item in plan:
        shard_root = output_root / "shards" / f"shard-{item['shard_index']:02d}"
        shard_report = {
            **item,
            "output_root": str(shard_root),
            "log": str(shard_root / "extract.log"),
            "status": "pending",
        }
        if args.resume and _shard_is_complete(
            shard_root,
            layers=layers,
            expected_rows=int(item["row_count"]),
        ):
            shard_report["status"] = "reused"
        else:
            pending.append({"plan": item, "root": shard_root, "report": shard_report})
        report["shards"].append(shard_report)
    write_json(metadata_path, report)

    running: list[dict[str, Any]] = []
    try:
        for pending_index, item in enumerate(pending):
            shard_root = item["root"]
            shard_root.mkdir(parents=True, exist_ok=True)
            command = _extract_command(
                args,
                shard_root=shard_root,
                row_start=int(item["plan"]["row_start"]),
                row_count=int(item["plan"]["row_count"]),
            )
            log_handle = (shard_root / "extract.log").open("a")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(item["plan"]["device"])
            if args.cublas_workspace_config:
                env["CUBLAS_WORKSPACE_CONFIG"] = str(
                    args.cublas_workspace_config
                )
            if args.seed is not None:
                env["PYTHONHASHSEED"] = str(int(args.seed))
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            item["process"] = process
            item["log_handle"] = log_handle
            item["report"].update(
                {"status": "running", "pid": process.pid, "started_at": utc_now()}
            )
            running.append(item)
            report["status"] = "extracting"
            write_json(metadata_path, report)
            if pending_index + 1 < len(pending) and args.launch_stagger_seconds > 0:
                time.sleep(args.launch_stagger_seconds)

        failure: dict[str, Any] | None = None
        while running:
            for item in list(running):
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                item["log_handle"].close()
                item["report"].update(
                    {
                        "returncode": returncode,
                        "completed_at": utc_now(),
                        "status": "completed" if returncode == 0 else "failed",
                    }
                )
                running.remove(item)
                if returncode != 0 and failure is None:
                    failure = item["report"]
            write_json(metadata_path, report)
            if failure is not None:
                for item in running:
                    item["process"].terminate()
                for item in running:
                    try:
                        item["process"].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        item["process"].kill()
                        item["process"].wait()
                    item["log_handle"].close()
                    item["report"].update(
                        {
                            "returncode": item["process"].returncode,
                            "completed_at": utc_now(),
                            "status": "terminated_after_peer_failure",
                        }
                    )
                report["status"] = "failed"
                report["failure"] = failure
                write_json(metadata_path, report)
                raise RuntimeError(
                    "activation extraction shard failed; see "
                    f"{failure['log']} and {metadata_path}"
                )
            if running:
                time.sleep(args.poll_seconds)
    except BaseException:
        for item in running:
            if item["process"].poll() is None:
                item["process"].terminate()
            item["log_handle"].close()
        report["status"] = "failed"
        report["failed_at"] = utc_now()
        write_json(metadata_path, report)
        raise

    report["status"] = "merging"
    write_json(metadata_path, report)
    for layer in layers:
        shard_paths = [
            output_root
            / "shards"
            / f"shard-{item['shard_index']:02d}"
            / f"R_{layer}"
            / "base.parquet"
            for item in plan
        ]
        report["merges"][f"R_{layer}"] = merge_layer_shards(
            shard_paths,
            output=output_root / f"R_{layer}" / "base.parquet",
            layer=layer,
            expected_rows=selected_rows,
            shard_plan=plan,
        )
        write_json(metadata_path, report)
    report["status"] = "completed"
    report["completed_at"] = utc_now()
    write_json(metadata_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--devices", required=True)
    parser.add_argument(
        "--shard-alignment",
        choices=("row", "document", "document_batch"),
        default="document_batch",
    )
    parser.add_argument("--source-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation")
    add_execution_profile_arguments(parser)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--publication-mode", action="store_true")
    parser.add_argument("--model-fingerprint-json", type=Path)
    parser.add_argument("--runtime-provenance-json", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--launch-stagger-seconds", type=float, default=2.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.publication_mode and (
        args.model_fingerprint_json is None or args.runtime_provenance_json is None
    ):
        parser.error(
            "--publication-mode requires --model-fingerprint-json and "
            "--runtime-provenance-json"
        )
    if args.launch_stagger_seconds < 0 or args.poll_seconds <= 0:
        parser.error("launch stagger must be non-negative and poll interval positive")
    return args


def main(argv: list[str] | None = None) -> int:
    report = run_sharded_extraction(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
