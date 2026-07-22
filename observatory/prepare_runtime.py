#!/usr/bin/env python3
"""Prepare the one-GPU Observatory runtime and exact selected source rows."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import (
    ObservatoryConfigError,
    config_fingerprint,
    load_config,
    read_jsonl,
    resolve_path,
    sha256_file,
    write_json,
)


SCHEMA_VERSION = "nano_viz_runtime_prepare.v1"


def choose_token_ids(
    tokenizer: Any,
    texts: list[tuple[str, str]],
    *,
    expected_length: int,
) -> tuple[list[int], str, dict[str, int]]:
    lengths: dict[str, int] = {}
    candidates: list[tuple[list[int], str]] = []
    for source_name, text in texts:
        token_ids = [
            int(value)
            for value in tokenizer(str(text), add_special_tokens=False)["input_ids"]
        ]
        lengths[source_name] = len(token_ids)
        candidates.append((token_ids, source_name))
        if len(token_ids) == expected_length:
            return token_ids, source_name, lengths
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        for token_ids, source_name in candidates:
            if len(token_ids) + 1 == expected_length:
                lengths[f"{source_name}+bos"] = len(token_ids) + 1
                return [int(bos_token_id), *token_ids], f"{source_name}+bos", lengths
    closest, source_name = min(
        candidates,
        key=lambda candidate: (abs(len(candidate[0]) - expected_length), candidate[1]),
    )
    raise ObservatoryConfigError(
        "could not reproduce token_ids_prefix length "
        f"{expected_length}; candidate_lengths={lengths}; closest={source_name}:{len(closest)}"
    )


def _selected_source_rows(
    *,
    validation_parquet: Path,
    corpus_rows: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    wanted = {
        (str(row["doc_id"]), int(row["n_raw_tokens"])): row for row in corpus_rows
    }
    found: dict[tuple[str, int], dict[str, Any]] = {}
    parquet = pq.ParquetFile(validation_parquet)
    columns = [
        "prompt",
        "response",
        "activation_vector",
        "n_raw_tokens",
        "activation_layer",
        "doc_id",
        "detokenized_text_truncated",
    ]
    for row_group in range(parquet.num_row_groups):
        keys = parquet.read_row_group(
            row_group, columns=["doc_id", "n_raw_tokens"]
        ).to_pydict()
        matching_offsets: list[tuple[int, tuple[str, int]]] = []
        for offset, (doc_id, n_raw_tokens) in enumerate(
            zip(keys["doc_id"], keys["n_raw_tokens"], strict=True)
        ):
            key = (str(doc_id), int(n_raw_tokens))
            if key in wanted:
                matching_offsets.append((offset, key))
        if not matching_offsets:
            continue
        data = parquet.read_row_group(row_group, columns=columns)
        for offset, key in matching_offsets:
            if key in found:
                raise ObservatoryConfigError(f"duplicate validation source key {key}")
            found[key] = data.slice(offset, 1).to_pylist()[0]
        if len(found) == len(wanted):
            break
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ObservatoryConfigError(f"validation parquet is missing selected keys: {missing[:5]}")

    rows: list[dict[str, Any]] = []
    tokenization: list[dict[str, Any]] = []
    for key, corpus_row in sorted(wanted.items(), key=lambda item: int(item[1]["row_index"])):
        source = found[key]
        token_ids, source_name, lengths = choose_token_ids(
            tokenizer,
            [
                ("validation_detokenized", source["detokenized_text_truncated"]),
                ("cleared_panel_source", corpus_row["source_text"]),
            ],
            expected_length=int(source["n_raw_tokens"]),
        )
        source_prompt = source.get("prompt")
        if not isinstance(source_prompt, list) or not source_prompt or not all(
            isinstance(message, dict)
            and message.get("role")
            and "content" in message
            for message in source_prompt
        ):
            raise ObservatoryConfigError(
                f"validation source has invalid chat prompt for key {key}"
            )
        row = {
            "row_index": int(corpus_row["row_index"]),
            "doc_id": str(source["doc_id"]),
            "n_raw_tokens": int(source["n_raw_tokens"]),
            "token_position": int(source["n_raw_tokens"]) - 1,
            "token_id": int(token_ids[-1]),
            "token_text": tokenizer.decode([token_ids[-1]]),
            "detokenized_text_truncated": str(source["detokenized_text_truncated"]),
            "token_ids_prefix": token_ids,
            "activation_vector": [float(value) for value in source["activation_vector"]],
            "activation_layer": int(source["activation_layer"]),
            "split": "validation",
            "content_family_id": str(corpus_row["content_family_id"]),
            "api_explanation": str(corpus_row["target_explanation"]),
            "prompt": [
                {
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                }
                for message in source_prompt
            ],
            "response": str(source["response"]),
        }
        rows.append(row)
        tokenization.append(
            {
                "row_id": corpus_row["row_id"],
                "row_index": corpus_row["row_index"],
                "expected_length": source["n_raw_tokens"],
                "selected_source": source_name,
                "candidate_lengths": lengths,
                "tail_token_id": token_ids[-1],
            }
        )
    return rows, tokenization


def write_source_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("row_index", pa.int64()),
            ("doc_id", pa.string()),
            ("n_raw_tokens", pa.int64()),
            ("token_position", pa.int64()),
            ("token_id", pa.int64()),
            ("token_text", pa.string()),
            ("detokenized_text_truncated", pa.string()),
            ("token_ids_prefix", pa.list_(pa.int64())),
            ("activation_vector", pa.list_(pa.float32(), 2688)),
            ("activation_layer", pa.int64()),
            ("split", pa.string()),
            ("content_family_id", pa.string()),
            ("api_explanation", pa.string()),
            (
                "prompt",
                pa.list_(
                    pa.struct(
                        [
                            ("role", pa.string()),
                            ("content", pa.string()),
                        ]
                    )
                ),
            ),
            ("response", pa.string()),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    temporary.replace(path)


def _hf_checkpoint_ready(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        (path / "model.safetensors.index.json").is_file()
        or any(path.glob("*.safetensors"))
    )


def _safetensor_sizes(path: Path) -> dict[str, int]:
    return {
        str(candidate.relative_to(path)): candidate.stat().st_size
        for candidate in sorted(path.rglob("*.safetensors"))
    }


def _copy_tree_parallel(source: Path, destination: Path, *, workers: int) -> dict[str, int]:
    if workers < 1:
        raise ObservatoryConfigError("stage_copy_workers must be at least 1")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ObservatoryConfigError(f"checkpoint staging source is empty: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for directory in sorted(path for path in source.rglob("*") if path.is_dir()):
        (destination / directory.relative_to(source)).mkdir(parents=True, exist_ok=True)

    def copy_file(source_path: Path) -> int:
        destination_path = destination / source_path.relative_to(source)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_size = source_path.stat().st_size
        if source_size >= 256 * 1024 * 1024:
            subprocess.run(
                [
                    "dd",
                    f"if={source_path}",
                    f"of={destination_path}",
                    "bs=64M",
                    "iflag=direct",
                    "status=none",
                ],
                check=True,
            )
            shutil.copystat(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)
        if destination_path.stat().st_size != source_size:
            raise ObservatoryConfigError(
                f"checkpoint staging size mismatch: {source_path} -> {destination_path}"
            )
        return source_size

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        sizes = list(executor.map(copy_file, files))
    return {"files": len(files), "bytes": sum(sizes), "workers": workers}


class _ExclusiveLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def __enter__(self) -> "_ExclusiveLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise ObservatoryConfigError(
                f"another Observatory runtime preparation owns {self.path}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.path.unlink(missing_ok=True)


def stage_av_checkpoint(
    *,
    code_root: Path,
    python_bin: Path,
    dcp_checkpoint: Path,
    origin_hf: Path,
    output_hf: Path,
    force: bool,
    copy_workers: int,
) -> dict[str, Any]:
    output_hf.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_hf.parent / ".runtime_prepare.lock"
    with _ExclusiveLock(lock_path):
        if _hf_checkpoint_ready(output_hf) and not force:
            return {"status": "reused", "output_hf": str(output_hf)}
        if force and output_hf.exists():
            shutil.rmtree(output_hf)
        staged_dcp = output_hf.parent / f"dcp_{dcp_checkpoint.name}"
        if staged_dcp.exists():
            shutil.rmtree(staged_dcp)
        staged_dcp.mkdir(parents=True, exist_ok=True)
        source_model = dcp_checkpoint / "model"
        if not source_model.is_dir():
            raise ObservatoryConfigError(f"AV DCP model directory is missing: {source_model}")
        staging = _copy_tree_parallel(
            source_model, staged_dcp / "model", workers=copy_workers
        )
        command = [
            str(python_bin),
            str(
                code_root
                / "external"
                / "natural_language_autoencoders"
                / "tools"
                / "convert_fsdp_to_hf.py"
            ),
            "--input-dir",
            str(staged_dcp),
            "--origin-hf-dir",
            str(origin_hf),
            "--output-dir",
            str(output_hf),
            "--torch-dtype",
            "bfloat16",
        ]
        log_path = output_hf.parent / "av_dcp_to_hf.log"
        try:
            with log_path.open("w") as log:
                subprocess.run(
                    command,
                    cwd=code_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env={**os.environ, "PYTHONPATH": str(code_root)},
                )
        finally:
            shutil.rmtree(staged_dcp, ignore_errors=True)
        if not _hf_checkpoint_ready(output_hf):
            raise ObservatoryConfigError("AV conversion completed without a usable HF checkpoint")
        return {
            "status": "converted",
            "output_hf": str(output_hf),
            "log": str(log_path),
            "staging": staging,
        }


def stage_hf_checkpoint(
    *, source_checkpoint: Path, output_hf: Path, force: bool, copy_workers: int
) -> dict[str, Any]:
    """Stage an HF checkpoint on node-local tmpfs for predictable model loading."""

    source_hf = (
        source_checkpoint / "hf"
        if (source_checkpoint / "hf" / "config.json").is_file()
        else source_checkpoint
    )
    if not _hf_checkpoint_ready(source_hf):
        raise ObservatoryConfigError(f"HF checkpoint is incomplete: {source_hf}")
    source_weights = _safetensor_sizes(source_hf)
    if not source_weights:
        raise ObservatoryConfigError(f"HF checkpoint has no safetensors: {source_hf}")
    output_hf.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_hf.parent / f".{output_hf.name}.stage.lock"
    with _ExclusiveLock(lock_path):
        if (
            not force
            and _hf_checkpoint_ready(output_hf)
            and _safetensor_sizes(output_hf) == source_weights
        ):
            return {
                "status": "reused",
                "source_hf": str(source_hf),
                "output_hf": str(output_hf),
                "weight_files": len(source_weights),
                "weight_bytes": sum(source_weights.values()),
            }
        temporary = output_hf.with_name(output_hf.name + ".staging")
        if temporary.exists():
            shutil.rmtree(temporary)
        staging = _copy_tree_parallel(source_hf, temporary, workers=copy_workers)
        if not _hf_checkpoint_ready(temporary) or _safetensor_sizes(temporary) != source_weights:
            raise ObservatoryConfigError(
                f"staged HF checkpoint failed completeness validation: {temporary}"
            )
        if output_hf.exists():
            shutil.rmtree(output_hf)
        os.replace(temporary, output_hf)
        return {
            "status": "copied",
            "source_hf": str(source_hf),
            "output_hf": str(output_hf),
            "weight_files": len(source_weights),
            "weight_bytes": sum(source_weights.values()),
            "staging": staging,
        }


def runtime_versions() -> dict[str, str | None]:
    packages = ("torch", "transformers", "accelerate", "pyarrow", "safetensors", "tokenizers")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run(
    config_path: Path,
    *,
    force_av: bool = False,
    skip_av: bool = False,
    skip_critics: bool = False,
) -> dict[str, Any]:
    from transformers import PreTrainedTokenizerFast

    config = load_config(config_path)
    paths = config["paths"]
    models = config.get("models") or {}
    code_root = resolve_path(paths["code_root"], config_path=config_path)
    python_bin = Path(sys.executable)
    base_hf = resolve_path(models["base_hf"], config_path=config_path)
    corpus_dir = resolve_path(paths["corpus_dir"], config_path=config_path)
    corpus_rows = read_jsonl(corpus_dir / "rows.jsonl")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        base_hf, local_files_only=True
    )
    source_rows, tokenization = _selected_source_rows(
        validation_parquet=resolve_path(paths["validation_parquet"], config_path=config_path),
        corpus_rows=corpus_rows,
        tokenizer=tokenizer,
    )
    source_path = resolve_path(paths["source_base_selected_parquet"], config_path=config_path)
    write_source_parquet(source_path, source_rows)
    av_stage = {"status": "skipped"}
    if not skip_av:
        evaluation = config.get("evaluation") or {}
        av_stage = stage_av_checkpoint(
            code_root=code_root,
            python_bin=python_bin,
            dcp_checkpoint=resolve_path(models["av_dcp"], config_path=config_path),
            origin_hf=base_hf,
            output_hf=resolve_path(models["av_hf_staging"], config_path=config_path),
            force=force_av,
            copy_workers=int(evaluation.get("stage_copy_workers", 8)),
        )
    critic_stages: dict[str, Any] = {}
    if not skip_critics:
        evaluation = config.get("evaluation") or {}
        for name in ("primary_ar", "independent_ar"):
            critic_stages[name] = stage_hf_checkpoint(
                source_checkpoint=resolve_path(
                    models[f"{name}_source"], config_path=config_path
                ),
                output_hf=resolve_path(models[name], config_path=config_path),
                force=False,
                copy_workers=int(evaluation.get("stage_copy_workers", 8)),
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": len(source_rows) == 50 and all(
            item["expected_length"] in item["candidate_lengths"].values()
            for item in tokenization
        ),
        "config_sha256": config_fingerprint(config),
        "runtime_versions": runtime_versions(),
        "source_parquet": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "rows": len(source_rows),
            "d_model": len(source_rows[0]["activation_vector"]),
        },
        "tokenization": tokenization,
        "av_stage": av_stage,
        "critic_stages": critic_stages,
    }
    output_path = resolve_path(paths["model_outputs_dir"], config_path=config_path) / "runtime_prepare.json"
    write_json(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force-av", action="store_true")
    parser.add_argument("--skip-av", action="store_true")
    parser.add_argument("--skip-critics", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(
            args.config,
            force_av=args.force_av,
            skip_av=args.skip_av,
            skip_critics=args.skip_critics,
        )
    except (OSError, ValueError, subprocess.CalledProcessError, ObservatoryConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
