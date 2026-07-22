#!/usr/bin/env python3
"""Build and audit a shared content-family manifest for Nano NLA datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_functional_eval_data import (  # noqa: E402
    FunctionalEvaluationError,
    assign_family_splits,
    build_content_families,
    build_family_exposure_report,
    content_family_overlap_report,
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise FunctionalEvaluationError("content-family build config must be an object")
    if config.get("schema_version") != "nano_content_family_build.v1":
        raise FunctionalEvaluationError(
            "content-family build config must use schema_version nano_content_family_build.v1"
        )
    if not isinstance(config.get("family_sources"), list) or not config["family_sources"]:
        raise FunctionalEvaluationError("family_sources must be a non-empty list")
    has_exposure = bool(config.get("exposure_sources"))
    has_candidates = bool(config.get("candidate_sources"))
    if has_exposure != has_candidates:
        raise FunctionalEvaluationError(
            "exposure_sources and candidate_sources must be provided together"
        )
    for key in ("exposure_sources", "candidate_sources"):
        if key in config and not isinstance(config[key], list):
            raise FunctionalEvaluationError(f"{key} must be a list")
    outputs = config.get("outputs") or {}
    if not outputs.get("manifest_json"):
        raise FunctionalEvaluationError("outputs.manifest_json is required")
    if has_exposure and not outputs.get("coverage_json"):
        raise FunctionalEvaluationError(
            "outputs.coverage_json is required when exposure sources are configured"
        )
    return config


def _resolved_path(value: Any, *, config_path: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else config_path.parent / path


def _parquet_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def iter_parquet_rows(
    source: dict[str, Any],
    *,
    config_path: Path,
    include_text: bool,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = _resolved_path(source.get("path"), config_path=config_path)
    if not path.is_file():
        raise FunctionalEvaluationError(f"source parquet does not exist: {path}")
    doc_id_field = str(source.get("doc_id_field") or "doc_id")
    text_field = str(source.get("text_field") or "detokenized_text_truncated")
    required = {doc_id_field}
    if include_text:
        required.add(text_field)
    missing = sorted(required - _parquet_columns(path))
    if missing:
        raise FunctionalEvaluationError(f"source {path} is missing columns: {missing}")
    columns = [doc_id_field] + ([text_field] if include_text else [])
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        doc_ids = batch.column(batch.schema.get_field_index(doc_id_field)).to_pylist()
        texts = (
            batch.column(batch.schema.get_field_index(text_field)).to_pylist()
            if include_text
            else [None] * len(doc_ids)
        )
        for doc_id, text in zip(doc_ids, texts):
            row = {"doc_id": doc_id}
            if include_text:
                row["source_text"] = text
            if source.get("split") is not None:
                row["split"] = str(source["split"])
            yield row


def _chain(iterables: Iterable[Iterable[dict[str, Any]]]) -> Iterator[dict[str, Any]]:
    for iterable in iterables:
        yield from iterable


def _source_metadata(
    sources: list[dict[str, Any]],
    *,
    config_path: Path,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    metadata = []
    for source in sources:
        path = _resolved_path(source.get("path"), config_path=config_path)
        metadata.append(
            {
                "name": str(source.get("name") or path.stem),
                "path": str(path),
                "split": source.get("split"),
                "row_count": int(pq.ParquetFile(path).metadata.num_rows),
                "doc_id_field": str(source.get("doc_id_field") or "doc_id"),
                "text_field": str(
                    source.get("text_field") or "detokenized_text_truncated"
                ),
            }
        )
    return metadata


def _candidate_iterables(
    sources: list[dict[str, Any]],
    *,
    config_path: Path,
    batch_size: int,
) -> dict[str, Iterable[dict[str, Any]]]:
    by_split: dict[str, list[Iterable[dict[str, Any]]]] = defaultdict(list)
    for source in sources:
        split = str(source.get("split") or "")
        if not split:
            raise FunctionalEvaluationError("candidate sources require a split")
        by_split[split].append(
            iter_parquet_rows(
                source,
                config_path=config_path,
                include_text=False,
                batch_size=batch_size,
            )
        )
    return {split: _chain(iterables) for split, iterables in by_split.items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _build_coverage(
    config: dict[str, Any],
    *,
    config_path: Path,
    manifest: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    if not config.get("exposure_sources") or not config.get("candidate_sources"):
        raise FunctionalEvaluationError(
            "coverage requires exposure_sources and candidate_sources"
        )
    exposure_sources = {
        str(source.get("name") or Path(str(source.get("path"))).stem): iter_parquet_rows(
            source,
            config_path=config_path,
            include_text=False,
            batch_size=batch_size,
        )
        for source in config["exposure_sources"]
    }
    candidates = _candidate_iterables(
        config["candidate_sources"],
        config_path=config_path,
        batch_size=batch_size,
    )
    coverage = build_family_exposure_report(
        manifest,
        candidate_rows_by_split=candidates,
        exposure_rows_by_source=exposure_sources,
        minimum_holdout_rows=int(
            (config.get("holdout") or {}).get("minimum_rows_per_split", 512)
        ),
    )
    coverage["build_config"] = str(config_path)
    return coverage


def run_coverage_only(
    config_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    config = load_config(resolved_config_path)
    resolved_manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(resolved_manifest_path.read_text())
    if manifest.get("schema_version") != "nano_content_family_manifest.v1":
        raise FunctionalEvaluationError(
            "reused manifest must use schema_version nano_content_family_manifest.v1"
        )
    coverage = _build_coverage(
        config,
        config_path=resolved_config_path,
        manifest=manifest,
        batch_size=int(config.get("batch_size") or 4096),
    )
    coverage["reused_manifest"] = str(resolved_manifest_path)
    coverage_path = _resolved_path(
        config["outputs"]["coverage_json"],
        config_path=resolved_config_path,
    )
    _write_json(coverage_path, coverage)
    return {
        "manifest_path": str(resolved_manifest_path),
        "coverage_path": str(coverage_path),
        "manifest": manifest,
        "coverage": coverage,
    }


def run_build(config_path: str | Path) -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    config = load_config(resolved_config_path)
    batch_size = int(config.get("batch_size") or 4096)
    algorithm = config.get("algorithm") or {}
    family_sources = list(config["family_sources"])
    family_rows = _chain(
        iter_parquet_rows(
            source,
            config_path=resolved_config_path,
            include_text=True,
            batch_size=batch_size,
        )
        for source in family_sources
    )
    manifest = build_content_families(
        family_rows,
        text_field="source_text",
        shingle_width=int(algorithm.get("shingle_width", 5)),
        similarity_threshold=float(algorithm.get("similarity_threshold", 0.8)),
        signature_size=int(algorithm.get("signature_size", 32)),
        candidate_min_shared=int(algorithm.get("candidate_min_shared", 4)),
        max_signature_bucket_size=int(
            algorithm.get("max_signature_bucket_size", 256)
        ),
    )
    manifest["build_config"] = str(resolved_config_path)
    manifest["source_tables"] = _source_metadata(
        family_sources,
        config_path=resolved_config_path,
    )

    split_assignment = config.get("split_assignment") or {}
    if split_assignment:
        manifest = assign_family_splits(
            manifest,
            split_weights={
                str(name): float(weight)
                for name, weight in (split_assignment.get("weights") or {}).items()
            },
            seed=int(split_assignment.get("seed", 0)),
        )

    observed_rows = _chain(
        iter_parquet_rows(
            source,
            config_path=resolved_config_path,
            include_text=False,
            batch_size=batch_size,
        )
        for source in family_sources
    )
    observed_rows_with_families = (
        {
            **row,
            "content_family_id": manifest["doc_assignments"].get(str(row["doc_id"])),
        }
        for row in observed_rows
    )
    manifest["observed_split_overlap"] = content_family_overlap_report(
        observed_rows_with_families
    )

    outputs = config["outputs"]
    manifest_path = _resolved_path(outputs["manifest_json"], config_path=resolved_config_path)
    _write_json(manifest_path, manifest)
    coverage = None
    coverage_path = None
    if config.get("exposure_sources"):
        coverage = _build_coverage(
            config,
            config_path=resolved_config_path,
            manifest=manifest,
            batch_size=batch_size,
        )
        coverage_path = _resolved_path(
            outputs["coverage_json"],
            config_path=resolved_config_path,
        )
        _write_json(coverage_path, coverage)
    return {
        "manifest_path": str(manifest_path),
        "coverage_path": None if coverage_path is None else str(coverage_path),
        "manifest": manifest,
        "coverage": coverage,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--reuse-manifest",
        type=Path,
        help="Skip clustering and recompute coverage from an existing manifest.",
    )
    args = parser.parse_args(argv)
    result = (
        run_coverage_only(args.config, args.reuse_manifest)
        if args.reuse_manifest
        else run_build(args.config)
    )
    summary = {
        "manifest_path": result["manifest_path"],
        "coverage_path": result["coverage_path"],
        "family_stats": result["manifest"]["stats"],
    }
    if result["coverage"] is not None:
        summary["holdout_decision"] = {
            "retain_existing_sft_checkpoints": result["coverage"][
                "retain_existing_sft_checkpoints"
            ],
            "clean_sft_retraining_required": result["coverage"][
                "clean_sft_retraining_required"
            ],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
