#!/usr/bin/env python3
"""Score semantic transformations of cached AV-generated explanations."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_nano_av_ar_roundtrip_gate as roundtrip  # noqa: E402
from nano_roundtrip_transforms import (  # noqa: E402
    DETERMINISTIC_TRANSFORMS,
    apply_transform_records,
    deterministic_transform_records,
    index_transform_records,
    read_jsonl,
    stable_row_key,
)


INVARIANCE_SCHEMA_VERSION = "nano_roundtrip_invariance.v1"


class InvarianceError(ValueError):
    """Raised when reports cannot form a strict paired invariance comparison."""


def _fve(report: dict[str, Any], split: str) -> float:
    value = (
        report.get("splits", {})
        .get(split, {})
        .get("variants", {})
        .get("av_real", {})
        .get("fve_nrm")
    )
    if not isinstance(value, (int, float)):
        raise InvarianceError(f"missing FVE for split={split}")
    return float(value)


def _row_indices(report: dict[str, Any], split: str) -> list[int] | None:
    values = report.get("splits", {}).get(split, {}).get("row_indices")
    if values is None:
        return None
    if not isinstance(values, list):
        raise InvarianceError(f"row_indices for split={split} must be a list")
    return [int(value) for value in values]


def summarize_invariance(
    raw_report: dict[str, Any],
    transformed_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    transforms: dict[str, Any] = {}
    for transform_name, report in transformed_reports.items():
        split_summaries: dict[str, Any] = {}
        for split in raw_report.get("splits", {}):
            raw_fve = _fve(raw_report, split)
            transformed_fve = _fve(report, split)
            if raw_fve <= 0.0:
                raise InvarianceError(f"raw FVE must be positive for split={split}, got {raw_fve}")
            raw_rows = _row_indices(raw_report, split)
            transformed_rows = _row_indices(report, split)
            if raw_rows is not None and transformed_rows is not None and raw_rows != transformed_rows:
                raise InvarianceError(
                    f"row identity mismatch for transform={transform_name} split={split}"
                )
            split_summaries[split] = {
                "raw_fve": raw_fve,
                "transformed_fve": transformed_fve,
                "fve_retention": transformed_fve / raw_fve,
                "fve_delta": transformed_fve - raw_fve,
                "row_count": len(raw_rows) if raw_rows is not None else None,
                "metrics": _metric_deltas(raw_report, report, split),
            }
        transforms[transform_name] = split_summaries
    return {
        "schema_version": INVARIANCE_SCHEMA_VERSION,
        "transforms": transforms,
    }


def _metric_deltas(
    raw_report: dict[str, Any], transformed_report: dict[str, Any], split: str
) -> dict[str, dict[str, float]]:
    raw = raw_report["splits"][split]["variants"]["av_real"]
    transformed = transformed_report["splits"][split]["variants"]["av_real"]
    output: dict[str, dict[str, float]] = {}
    for metric in ("directional_mse", "raw_mse", "centered_r2", "norm_ratio_mean"):
        if not isinstance(raw.get(metric), (int, float)) or not isinstance(
            transformed.get(metric), (int, float)
        ):
            continue
        output[metric] = {
            "raw": float(raw[metric]),
            "transformed": float(transformed[metric]),
            "delta": float(transformed[metric] - raw[metric]),
        }
    return output


def summarize_combined_invariance(
    report: dict[str, Any], transform_names: list[str]
) -> dict[str, Any]:
    transforms: dict[str, Any] = {}
    for transform_name in transform_names:
        split_summaries: dict[str, Any] = {}
        transformed_variant = f"av_{transform_name}"
        for split_name, split in report.get("splits", {}).items():
            variants = split.get("variants") or {}
            raw = variants.get("av_real") or {}
            transformed = variants.get(transformed_variant) or {}
            raw_fve = raw.get("fve_nrm")
            transformed_fve = transformed.get("fve_nrm")
            if not isinstance(raw_fve, (int, float)) or not isinstance(
                transformed_fve, (int, float)
            ):
                raise InvarianceError(
                    f"missing FVE for transform={transform_name} split={split_name}"
                )
            if raw_fve <= 0.0:
                raise InvarianceError(
                    f"raw FVE must be positive for split={split_name}, got {raw_fve}"
                )
            metrics: dict[str, dict[str, float]] = {}
            for metric in (
                "directional_mse",
                "raw_mse",
                "centered_r2",
                "norm_ratio_mean",
            ):
                if not isinstance(raw.get(metric), (int, float)) or not isinstance(
                    transformed.get(metric), (int, float)
                ):
                    continue
                metrics[metric] = {
                    "raw": float(raw[metric]),
                    "transformed": float(transformed[metric]),
                    "delta": float(transformed[metric] - raw[metric]),
                }
            split_summaries[split_name] = {
                "raw_fve": float(raw_fve),
                "transformed_fve": float(transformed_fve),
                "fve_retention": float(transformed_fve / raw_fve),
                "fve_delta": float(transformed_fve - raw_fve),
                "row_count": len(split.get("row_indices") or []),
                "metrics": metrics,
            }
        transforms[transform_name] = split_summaries
    return {
        "schema_version": INVARIANCE_SCHEMA_VERSION,
        "transforms": transforms,
    }


def merge_transform_controls(
    generated_records: list[dict[str, Any]],
    transformed_by_name: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    combined = copy.deepcopy(generated_records)
    for transform_name, transformed_records in transformed_by_name.items():
        if transform_name == "real":
            raise InvarianceError("transform name cannot be 'real'")
        if len(transformed_records) != len(combined):
            raise InvarianceError(
                f"row count mismatch for transform={transform_name}"
            )
        for destination, transformed in zip(combined, transformed_records):
            if stable_row_key(destination) != stable_row_key(transformed):
                raise InvarianceError(
                    f"row identity mismatch for transform={transform_name}"
                )
            payload = copy.deepcopy(
                (transformed.get("controls") or {}).get("real") or {}
            )
            if not str(payload.get("generated") or "").strip():
                raise InvarianceError(
                    f"empty transformed text for transform={transform_name}"
                )
            destination.setdefault("controls", {})[transform_name] = payload
    return combined


def filter_generated_records(
    records: list[dict[str, Any]],
    selected_by_split: dict[str, list[int]],
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("split")), int(record.get("row_index", -1)))
        if key in indexed:
            raise InvarianceError(f"duplicate generated row: {key}")
        indexed[key] = record
    selected: list[dict[str, Any]] = []
    missing: list[tuple[str, int]] = []
    for split_name, row_indices in selected_by_split.items():
        for row_index in row_indices:
            key = (split_name, int(row_index))
            record = indexed.get(key)
            if record is None:
                missing.append(key)
            else:
                selected.append(record)
    if missing:
        raise InvarianceError(
            f"generated JSONL is missing selected rows: {missing[:10]}"
        )
    return selected


def _external_transform_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise InvarianceError(
                f"--transform-jsonl must use NAME=PATH syntax, got {value!r}"
            )
        name, path = value.split("=", 1)
        if not name or not path or name in paths:
            raise InvarianceError(f"invalid or duplicate transform mapping: {value!r}")
        paths[name] = Path(path)
    return paths


def _score_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        ar_checkpoint_dir=args.ar_checkpoint_dir,
        critic_template_source=args.critic_template_source,
        critic_template=args.critic_template,
        train_parquet=args.train_parquet,
        validation_parquet=args.validation_parquet,
        test_parquet=args.test_parquet,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        generation_controls=["real"],
        dataset_controls=[],
        eval_splits=list(args.eval_splits),
        content_family_manifest=args.content_family_manifest,
        content_family_coverage=args.content_family_coverage,
        require_family_level_inference=args.require_family_level_inference,
        selection_strategy=args.selection_strategy,
        selection_seed=args.selection_seed,
        torch_dtype=args.torch_dtype,
        ar_device_map=args.ar_device_map,
        ar_low_cpu_mem_usage=True,
        collect_ar_device_profile=True,
        ar_batch_size=args.ar_batch_size,
        ar_max_length=args.ar_max_length,
        length_baseline_generated_jsonl=None,
        generated_text_fallback=args.generated_text_fallback,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    generated_records = roundtrip.read_generated_jsonl(args.generated_jsonl)
    rows, _, validation_indices, test_indices = roundtrip.load_eval_rows(
        args.train_parquet,
        args.validation_parquet,
        args.test_parquet,
        eval_splits=args.eval_splits,
        content_family_manifest=args.content_family_manifest,
        content_family_coverage=args.content_family_coverage,
        require_family_disjoint_splits=args.require_family_level_inference,
    )
    selected_by_split = roundtrip.select_eval_indices_by_split(
        rows,
        validation_indices=validation_indices,
        test_indices=test_indices,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        eval_splits=args.eval_splits,
        strategy=args.selection_strategy,
        seed=args.selection_seed,
    )
    generated_records = filter_generated_records(
        generated_records, selected_by_split
    )
    scoring_args = _score_args(args)
    transformed_records: dict[str, list[dict[str, Any]]] = {}

    for transform in args.deterministic_transforms:
        transform_records = deterministic_transform_records(
            generated_records,
            transform=transform,
            seed=args.seed,
        )
        transformed = apply_transform_records(
            generated_records,
            index_transform_records(transform_records),
            transform=transform,
        )
        transformed_records[transform] = transformed

    for transform, path in _external_transform_paths(args.transform_jsonl).items():
        transform_records = index_transform_records(read_jsonl(path))
        transformed = apply_transform_records(
            generated_records,
            transform_records,
            transform=transform,
        )
        transformed_records[transform] = transformed

    combined_records = merge_transform_controls(
        generated_records, transformed_records
    )
    transform_names = list(transformed_records)
    scoring_args.generation_controls = ["real", *transform_names]
    combined_report = roundtrip.score_generated_records(
        scoring_args, combined_records
    )
    summary = summarize_combined_invariance(combined_report, transform_names)
    return {
        **summary,
        "generated_jsonl": str(args.generated_jsonl),
        "scoring_passes": 1,
        "combined_score": combined_report,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-jsonl", type=Path, required=True)
    parser.add_argument("--ar-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--critic-template-source", type=Path)
    parser.add_argument("--critic-template")
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument(
        "--eval-splits", nargs="+", choices=("validation", "test"), default=["validation"]
    )
    parser.add_argument("--content-family-manifest", type=Path)
    parser.add_argument("--content-family-coverage", type=Path)
    parser.add_argument("--selection-strategy", choices=("row_order", "family_stratified"), default="row_order")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--require-family-level-inference", action="store_true")
    parser.add_argument("--validation-limit", type=int, default=512)
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--deterministic-transforms", nargs="+", default=list(DETERMINISTIC_TRANSFORMS))
    parser.add_argument("--transform-jsonl", action="append", default=[])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--ar-device-map", default="auto")
    parser.add_argument("--ar-batch-size", type=int, default=16)
    parser.add_argument("--ar-max-length", type=int)
    parser.add_argument("--generated-text-fallback", choices=("empty", "raw"), default="raw")
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if "test" in args.eval_splits and args.test_parquet is None:
        parser.error("--test-parquet is required when --eval-splits includes test")
    if args.validation_limit <= 0 or args.test_limit <= 0 or args.ar_batch_size <= 0:
        parser.error("validation/test limits and AR batch size must be positive")
    unknown = sorted(set(args.deterministic_transforms) - set(DETERMINISTIC_TRANSFORMS))
    if unknown:
        parser.error(f"unknown deterministic transforms: {unknown}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(args)
    roundtrip.write_json(args.report_json, report)
    print(json.dumps({"report_json": str(args.report_json), "transforms": list(report["transforms"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
