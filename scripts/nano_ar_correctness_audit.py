#!/usr/bin/env python3
"""Correctness audit for Nano AR critic checkpoints and split contracts."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


BOUNDARY_RE = re.compile(r"R_?(\d+)$", re.IGNORECASE)
LAYER_KEY_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")


def parse_boundary_name(boundary_name: str) -> int:
    match = BOUNDARY_RE.fullmatch(boundary_name.strip())
    if not match:
        raise ValueError(f"unsupported boundary name {boundary_name!r}; expected R27 or R_27")
    value = int(match.group(1))
    if value <= 0:
        raise ValueError(f"boundary must be positive, got {boundary_name!r}")
    return value


def expected_zero_based_layer(boundary_name: str) -> int:
    """Return the zero-based decoder block hooked by the extractor.

    Nano boundary labels use the extractor's module index directly: R33 hooks
    decoder block 33 and captures its output. This is not a one-based ordinal.
    """

    return parse_boundary_name(boundary_name)


def expected_extraction_layer_index(boundary_name: str) -> int:
    return parse_boundary_name(boundary_name)


def expected_hidden_layers(boundary_name: str) -> int:
    return expected_extraction_layer_index(boundary_name) + 1


def _checkpoint_tensor_keys(checkpoint_dir: Path) -> tuple[list[Path], set[str]]:
    try:
        from safetensors import safe_open
    except Exception as exc:  # noqa: BLE001 - diagnostic import path.
        raise RuntimeError(f"cannot inspect critic tensors: safetensors unavailable: {exc}") from exc

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path)
        files = sorted({checkpoint_dir / name for name in index.get("weight_map", {}).values()})
    else:
        files = sorted(
            path
            for path in checkpoint_dir.glob("model*.safetensors")
            if "megatron-compat" not in path.name and path.name != "value_head.safetensors"
        )
    keys: set[str] = set()
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return files, keys


def audit_checkpoint_tensor_layout(
    checkpoint_dir: str | Path,
    *,
    boundary_name: str,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    expected_last = expected_extraction_layer_index(boundary_name)
    expected_layers = list(range(expected_last + 1))
    report: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "expected_last_retained_block_index": expected_last,
        "expected_block_indices": expected_layers,
        "model_files": [],
        "observed_block_indices": [],
        "embedding_keys": [],
        "lm_head_keys": [],
        "final_norm_keys": [],
        "runtime_contract": "lm_head and final norm are stripped; value head reads raw post-block residual stream",
        "passed": False,
        "warnings": [],
    }
    files, keys = _checkpoint_tensor_keys(checkpoint_dir)
    report["model_files"] = [str(path) for path in files]
    if not files:
        report["warnings"].append("no model safetensor files found")
        return report

    block_indices = sorted(
        {
            int(match.group(1))
            for key in keys
            if (match := LAYER_KEY_RE.search(key)) is not None
        }
    )
    embedding_keys = sorted(
        key for key in keys if key.endswith(("embeddings.weight", "embed_tokens.weight", "word_embeddings.weight"))
    )
    lm_head_keys = sorted(key for key in keys if key.endswith("lm_head.weight"))
    final_norm_keys = sorted(
        key
        for key in keys
        if key.endswith(("norm_f.weight", "final_layernorm.weight", "ln_f.weight"))
        or key in {"model.norm.weight", "backbone.norm.weight"}
    )
    report["observed_block_indices"] = block_indices
    report["embedding_keys"] = embedding_keys
    report["lm_head_keys"] = lm_head_keys
    report["final_norm_keys"] = final_norm_keys
    report["passed"] = (
        block_indices == expected_layers
        and bool(embedding_keys)
        and not lm_head_keys
        and not final_norm_keys
    )
    if block_indices != expected_layers:
        report["warnings"].append(
            f"observed block indices {block_indices} != expected {expected_layers}"
        )
    if not embedding_keys:
        report["warnings"].append("input embedding tensor not found")
    if lm_head_keys:
        report["warnings"].append(f"critic checkpoint unexpectedly retains lm_head: {lm_head_keys}")
    if final_norm_keys:
        report["warnings"].append(
            f"critic checkpoint unexpectedly retains final norm: {final_norm_keys}"
        )
    return report


def audit_dataset_boundary(parquet_path: str | Path, *, boundary_name: str) -> dict[str, Any]:
    parquet_path = Path(parquet_path)
    sidecar_path = Path(str(parquet_path) + ".nla_meta.yaml")
    expected = expected_extraction_layer_index(boundary_name)
    report: dict[str, Any] = {
        "parquet_path": str(parquet_path),
        "sidecar_path": str(sidecar_path),
        "boundary_name": boundary_name,
        "expected_extraction_layer_index": expected,
        "observed_extraction_layer_index": None,
        "passed": False,
        "warnings": [],
    }
    if not sidecar_path.is_file():
        report["warnings"].append("missing parquet NLA sidecar")
        return report
    data = yaml.safe_load(sidecar_path.read_text()) or {}
    candidates = []
    for section_name in ("extraction", "critic"):
        section = data.get(section_name)
        if isinstance(section, dict):
            candidates.append(section.get("layer_index"))
            candidates.append(section.get("extraction_layer_index"))
    observed = next((int(value) for value in candidates if isinstance(value, int)), None)
    report["observed_extraction_layer_index"] = observed
    report["passed"] = observed == expected
    if observed != expected:
        report["warnings"].append(
            f"dataset extraction layer index {observed} != expected {expected}"
        )
    return report


def identity_distance(weight: Any) -> float | None:
    matrix = np.asarray(weight, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    identity = np.eye(matrix.shape[0], dtype=np.float32)
    denom = float(np.linalg.norm(identity, ord="fro"))
    if denom == 0.0:
        return None
    return float(np.linalg.norm(matrix - identity, ord="fro") / denom)


def doc_overlap_summary(split_doc_ids: dict[str, list[str]]) -> dict[str, Any]:
    doc_to_splits: dict[str, set[str]] = defaultdict(set)
    for split, doc_ids in split_doc_ids.items():
        for doc_id in doc_ids:
            if doc_id:
                doc_to_splits[str(doc_id)].add(split)
    overlap = {
        doc_id: sorted(splits)
        for doc_id, splits in sorted(doc_to_splits.items())
        if len(splits) > 1
    }
    return {
        "passed": not overlap,
        "overlap_count": len(overlap),
        "overlap_doc_ids": sorted(overlap),
        "overlap_sample": [
            {"doc_id": doc_id, "splits": splits}
            for doc_id, splits in list(overlap.items())[:20]
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_config_hidden_layers(config: dict[str, Any]) -> int | None:
    if isinstance(config.get("num_hidden_layers"), int):
        return int(config["num_hidden_layers"])
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and isinstance(text_config.get("num_hidden_layers"), int):
        return int(text_config["num_hidden_layers"])
    return None


def audit_checkpoint_config(checkpoint_dir: str | Path, *, boundary_name: str) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    expected = expected_hidden_layers(boundary_name)
    report: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "config_path": str(config_path),
        "boundary_name": boundary_name,
        "zero_based_last_layer_index": expected_zero_based_layer(boundary_name),
        "last_retained_block_index": expected_extraction_layer_index(boundary_name),
        "expected_hidden_layers": expected,
        "critic_config_num_hidden_layers": None,
        "passed": False,
        "warnings": [],
    }
    if not config_path.exists():
        report["warnings"].append("missing config.json")
        return report
    config = _load_json(config_path)
    actual = _read_config_hidden_layers(config)
    report["critic_config_num_hidden_layers"] = actual
    report["passed"] = actual == expected
    if actual != expected:
        report["warnings"].append(f"num_hidden_layers {actual} != expected {expected}")
    return report


def audit_model_sidecar(checkpoint_dir: str | Path, *, boundary_name: str) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    path = checkpoint_dir / "nla_meta.yaml"
    expected_num_layers = expected_hidden_layers(boundary_name)
    expected_extraction_index = expected_extraction_layer_index(boundary_name)
    report: dict[str, Any] = {
        "path": str(path),
        "critic_num_layers": None,
        "critic_extraction_layer_index": None,
        "expected_num_hidden_layers": expected_num_layers,
        "expected_extraction_layer_index": expected_extraction_index,
        "expected_zero_based_last_layer_index": expected_zero_based_layer(boundary_name),
        "expected_last_retained_block_index": expected_extraction_index,
        "passed": False,
        "warnings": [],
    }
    if not path.exists():
        report["warnings"].append("missing nla_meta.yaml")
        return report
    data = yaml.safe_load(path.read_text()) or {}
    critic_num_layers = data.get("critic_num_layers")
    if critic_num_layers is None and isinstance(data.get("nla"), dict):
        critic_num_layers = data["nla"].get("critic_num_layers")
    critic_extraction_layer_index = None
    if isinstance(data.get("critic"), dict):
        critic_extraction_layer_index = data["critic"].get("extraction_layer_index")
    report["critic_num_layers"] = critic_num_layers
    report["critic_extraction_layer_index"] = critic_extraction_layer_index
    num_layers_ok = critic_num_layers in (None, expected_num_layers)
    extraction_index_ok = critic_extraction_layer_index in (None, expected_extraction_index)
    report["passed"] = num_layers_ok and extraction_index_ok
    if critic_num_layers is None:
        report["warnings"].append("critic_num_layers absent from sidecar")
    elif critic_num_layers != expected_num_layers:
        report["warnings"].append(
            f"critic_num_layers {critic_num_layers} != expected {expected_num_layers}"
        )
    if critic_extraction_layer_index is None:
        report["warnings"].append("critic.extraction_layer_index absent from sidecar")
    elif critic_extraction_layer_index != expected_extraction_index:
        report["warnings"].append(
            "critic.extraction_layer_index "
            f"{critic_extraction_layer_index} != expected {expected_extraction_index}"
        )
    return report


def _load_value_head_weight(checkpoint_dir: Path) -> np.ndarray | None:
    head_path = checkpoint_dir / "value_head.safetensors"
    if not head_path.exists():
        return None
    try:
        from safetensors.torch import load_file
    except Exception as exc:  # noqa: BLE001 - diagnostic import path.
        raise RuntimeError(f"cannot read {head_path}: safetensors unavailable: {exc}") from exc
    tensors = load_file(str(head_path), device="cpu")
    for key in ("weight", "value_head.weight"):
        if key in tensors:
            return tensors[key].float().numpy()
    if len(tensors) == 1:
        return next(iter(tensors.values())).float().numpy()
    raise ValueError(f"could not identify value head weight in {head_path}; keys={sorted(tensors)}")


def audit_value_head(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    report: dict[str, Any] = {
        "path": str(checkpoint_dir / "value_head.safetensors"),
        "identity_distance": None,
        "passed": False,
        "warnings": [],
    }
    try:
        weight = _load_value_head_weight(checkpoint_dir)
    except Exception as exc:  # noqa: BLE001 - audit should report, not crash.
        report["warnings"].append(str(exc))
        return report
    if weight is None:
        report["warnings"].append("missing value_head.safetensors")
        return report
    distance = identity_distance(weight)
    report["identity_distance"] = distance
    report["shape"] = list(weight.shape)
    report["passed"] = distance is not None and math.isfinite(distance)
    return report


def _read_doc_ids(parquet_path: Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["doc_id"])
    return [str(item) for item in table.column("doc_id").to_pylist()]


def audit_split_overlap(
    *,
    train_parquet: str | Path,
    validation_parquet: str | Path,
    test_parquet: str | Path,
) -> dict[str, Any]:
    split_doc_ids = {
        "train": _read_doc_ids(Path(train_parquet)),
        "validation": _read_doc_ids(Path(validation_parquet)),
        "test": _read_doc_ids(Path(test_parquet)),
    }
    summary = doc_overlap_summary(split_doc_ids)
    summary["counts"] = {split: len(doc_ids) for split, doc_ids in split_doc_ids.items()}
    return summary


def build_audit_report(args: argparse.Namespace) -> dict[str, Any]:
    sections: dict[str, Any] = {
        "schema_version": "nano_ar_correctness_audit.v2",
        "boundary_name": args.boundary_name,
        "checkpoint_config": audit_checkpoint_config(args.checkpoint_dir, boundary_name=args.boundary_name),
        "checkpoint_tensor_layout": audit_checkpoint_tensor_layout(
            args.checkpoint_dir,
            boundary_name=args.boundary_name,
        ),
        "model_sidecar": audit_model_sidecar(args.checkpoint_dir, boundary_name=args.boundary_name),
        "value_head": audit_value_head(args.checkpoint_dir),
    }
    if args.train_parquet and args.validation_parquet and args.test_parquet:
        sections["dataset_boundary"] = audit_dataset_boundary(
            args.train_parquet,
            boundary_name=args.boundary_name,
        )
        sections["doc_split_overlap"] = audit_split_overlap(
            train_parquet=args.train_parquet,
            validation_parquet=args.validation_parquet,
            test_parquet=args.test_parquet,
        )
    passed = True
    for value in sections.values():
        if isinstance(value, dict) and value.get("passed") is False:
            passed = False
    sections["passed"] = passed
    return sections


def _markdown_section(name: str, report: dict[str, Any]) -> list[str]:
    lines = [f"## {name}", ""]
    passed = report.get("passed")
    if passed is not None:
        lines.append(f"- passed: `{passed}`")
    for key, value in report.items():
        if key in {"passed", "warnings"}:
            continue
        lines.append(f"- {key}: `{value}`")
    warnings = report.get("warnings") or []
    if warnings:
        lines.append("- warnings:")
        lines.extend(f"  - {item}" for item in warnings)
    lines.append("")
    return lines


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Nano AR Correctness Audit", "", f"- passed: `{report.get('passed')}`", ""]
    for name, section in report.items():
        if isinstance(section, dict):
            lines.extend(_markdown_section(name.replace("_", " ").title(), section))
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--train-parquet", type=Path)
    parser.add_argument("--validation-parquet", type=Path)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--boundary-name", default="R27")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    args = parser.parse_args()

    report = build_audit_report(args)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.report_md:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_report(args.report_md, report)
    if not args.report_json and not args.report_md:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
