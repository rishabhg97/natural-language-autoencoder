#!/usr/bin/env python3
"""Audit frozen Nano round-trip predictions across validation-fitted subgroups."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calibrate_nano_activation_magnitude import (  # noqa: E402
    file_provenance,
    load_bound_score_report,
    load_cache_split,
    validate_cache_compatibility,
)
from nano_eval_core import clustered_paired_bootstrap_improvement  # noqa: E402


SCHEMA_VERSION = "nano_roundtrip_subgroup_audit.v1"
DIMENSIONS = (
    "n_raw_tokens",
    "target_word_count",
    "target_activation_norm",
    "sample_family_frequency",
)


class SubgroupAuditError(ValueError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise SubgroupAuditError(f"config must use schema_version {SCHEMA_VERSION}")
    paths = config.get("paths")
    protocol = config.get("protocol")
    if not isinstance(paths, dict) or not paths.get("report_json"):
        raise SubgroupAuditError("paths.report_json is required")
    datasets = paths.get("datasets")
    if not isinstance(datasets, dict) or len(datasets) < 2:
        raise SubgroupAuditError("paths.datasets requires fit and evaluation datasets")
    for name, item in datasets.items():
        required = {"cache_npz", "score_report_json", "generated_jsonl", "split"}
        if not isinstance(item, dict) or not required.issubset(item):
            raise SubgroupAuditError(f"dataset {name!r} is missing required paths")
    if not isinstance(protocol, dict):
        raise SubgroupAuditError("protocol is required")
    fit_name = str(protocol.get("fit_dataset") or "")
    if fit_name not in datasets or str(datasets[fit_name]["split"]) == "test":
        raise SubgroupAuditError("fit_dataset must exist and must not use test")
    if int(protocol.get("quantile_bins", 4)) < 2:
        raise SubgroupAuditError("quantile_bins must be at least 2")
    if int(protocol.get("bootstrap_samples", 10_000)) <= 0:
        raise SubgroupAuditError("bootstrap_samples must be positive")
    if float(protocol.get("magnitude_scalar", 1.0)) < 0.0:
        raise SubgroupAuditError("magnitude_scalar must be nonnegative")
    controls = list(protocol.get("controls") or [])
    if not controls or "av_real" in controls:
        raise SubgroupAuditError("controls must be non-empty and exclude av_real")
    return config


def read_generated_records(path: Path, split: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record.get("split")) != split:
            raise SubgroupAuditError(
                f"generated record split mismatch in {path}: {record.get('split')!r}"
            )
        row_index = int(record.get("row_index", -1))
        if row_index < 0 or row_index in records:
            raise SubgroupAuditError(f"invalid or duplicate row_index={row_index}")
        records[row_index] = record
    if not records:
        raise SubgroupAuditError(f"generated JSONL is empty: {path}")
    return records


def align_metadata(
    dataset: dict[str, Any],
    records: dict[int, dict[str, Any]],
) -> dict[str, np.ndarray]:
    row_indices = [int(value) for value in dataset["row_indices"]]
    if set(row_indices) != set(records) or len(row_indices) != len(records):
        raise SubgroupAuditError("cache and generated JSONL row identities differ")
    family_counts = Counter(str(value) for value in dataset["content_family_ids"])
    n_raw_tokens: list[float] = []
    target_words: list[float] = []
    family_frequency: list[float] = []
    for offset, row_index in enumerate(row_indices):
        record = records[row_index]
        if str(record.get("doc_id")) != str(dataset["doc_ids"][offset]):
            raise SubgroupAuditError(f"doc_id mismatch at row_index={row_index}")
        family = str(dataset["content_family_ids"][offset])
        if str(record.get("content_family_id")) != family:
            raise SubgroupAuditError(
                f"content_family_id mismatch at row_index={row_index}"
            )
        target = str(record.get("target_explanation") or "").strip()
        if not target:
            raise SubgroupAuditError(
                f"target_explanation is empty at row_index={row_index}"
            )
        raw_tokens = float(record.get("n_raw_tokens", math.nan))
        if not math.isfinite(raw_tokens) or raw_tokens < 0:
            raise SubgroupAuditError(f"invalid n_raw_tokens at row_index={row_index}")
        n_raw_tokens.append(raw_tokens)
        target_words.append(float(len(re.findall(r"\w+", target))))
        family_frequency.append(float(family_counts[family]))
    targets = np.asarray(dataset["targets"], dtype=np.float64)
    return {
        "n_raw_tokens": np.asarray(n_raw_tokens, dtype=np.float64),
        "target_word_count": np.asarray(target_words, dtype=np.float64),
        "target_activation_norm": np.linalg.norm(targets, axis=1),
        "sample_family_frequency": np.asarray(family_frequency, dtype=np.float64),
    }


def fit_quantile_edges(values: np.ndarray, bin_count: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < bin_count or not np.isfinite(values).all():
        raise SubgroupAuditError("quantile fit values must be finite and sufficiently large")
    quantiles = np.quantile(values, np.arange(1, bin_count) / bin_count)
    return [float(value) for value in np.unique(quantiles)]


def assign_bins(values: np.ndarray, edges: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise SubgroupAuditError("subgroup values must be a finite vector")
    return np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="right")


def rowwise_directional_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    numerator = np.sum(prediction * target, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return 2.0 * (1.0 - np.clip(cosine, -1.0, 1.0))


def rowwise_raw_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.square(prediction - target), axis=1)


def summarize_rows(
    dataset: dict[str, Any],
    indices: np.ndarray,
    *,
    controls: list[str],
    magnitude_scalar: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    predictions = dataset["predictions"]
    required = {"av_real", "teacher", *controls}
    missing = sorted(required - set(predictions))
    if missing:
        raise SubgroupAuditError(f"prediction cache lacks variants: {missing}")
    target = np.asarray(dataset["targets"], dtype=np.float64)[indices]
    train_mean = np.asarray(dataset["train_mean"], dtype=np.float64)
    candidate = np.asarray(predictions["av_real"], dtype=np.float64)[indices]
    teacher = np.asarray(predictions["teacher"], dtype=np.float64)[indices]
    families = [str(dataset["content_family_ids"][index]) for index in indices]
    candidate_directional = rowwise_directional_mse(candidate, target)
    teacher_directional = rowwise_directional_mse(teacher, target)
    calibrated = candidate * magnitude_scalar
    candidate_raw = rowwise_raw_mse(calibrated, target)
    mean_prediction = np.broadcast_to(train_mean, target.shape)
    mean_raw = rowwise_raw_mse(mean_prediction, target)
    controls_report: dict[str, Any] = {}
    for control_index, control in enumerate(controls):
        control_error = rowwise_directional_mse(
            np.asarray(predictions[control], dtype=np.float64)[indices],
            target,
        )
        controls_report[control] = clustered_paired_bootstrap_improvement(
            control_error,
            candidate_directional,
            families,
            seed=bootstrap_seed + control_index,
            resamples=bootstrap_samples,
        )
    raw_mse = float(candidate_raw.mean())
    mean_predictor_raw_mse = float(mean_raw.mean())
    return {
        "row_count": int(len(indices)),
        "family_count": int(len(set(families))),
        "av_real_directional_mse": float(candidate_directional.mean()),
        "teacher_directional_mse": float(teacher_directional.mean()),
        "av_real_minus_teacher_directional_mse": float(
            candidate_directional.mean() - teacher_directional.mean()
        ),
        "calibrated_raw_mse": raw_mse,
        "mean_predictor_raw_mse": mean_predictor_raw_mse,
        "calibrated_centered_raw_r2": float(1.0 - raw_mse / mean_predictor_raw_mse),
        "controls": controls_report,
    }


def evaluate_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    paths = config["paths"]
    protocol = config["protocol"]
    base = config_path.parent
    descriptors = {
        name: {
            key: (Path(value) if Path(value).is_absolute() else base / value)
            if key != "split"
            else str(value)
            for key, value in item.items()
        }
        for name, item in paths["datasets"].items()
    }
    loaded = {
        name: load_cache_split(item["cache_npz"], item["split"])
        for name, item in descriptors.items()
    }
    validate_cache_compatibility(loaded)
    score_reports = {
        name: load_bound_score_report(item["score_report_json"], item["cache_npz"])
        for name, item in descriptors.items()
    }
    metadata = {
        name: align_metadata(
            loaded[name],
            read_generated_records(item["generated_jsonl"], item["split"]),
        )
        for name, item in descriptors.items()
    }
    fit_name = str(protocol["fit_dataset"])
    quantile_bins = int(protocol.get("quantile_bins", 4))
    edges = {
        dimension: fit_quantile_edges(metadata[fit_name][dimension], quantile_bins)
        for dimension in DIMENSIONS
    }
    controls = [str(value) for value in protocol["controls"]]
    magnitude_scalar = float(protocol.get("magnitude_scalar", 1.0))
    bootstrap_samples = int(protocol.get("bootstrap_samples", 10_000))
    bootstrap_seed = int(protocol.get("bootstrap_seed", 0))
    min_rows = int(protocol.get("min_rows_per_group", 20))
    min_families = int(protocol.get("min_families_per_group", 10))
    datasets_report: dict[str, Any] = {}
    for dataset_offset, (name, dataset) in enumerate(loaded.items()):
        all_indices = np.arange(len(dataset["row_indices"]), dtype=np.int64)
        dimensions_report: dict[str, Any] = {}
        for dimension_offset, dimension in enumerate(DIMENSIONS):
            assignments = assign_bins(metadata[name][dimension], edges[dimension])
            groups = []
            for bin_index in sorted(set(int(value) for value in assignments)):
                indices = np.flatnonzero(assignments == bin_index)
                summary = summarize_rows(
                    dataset,
                    indices,
                    controls=controls,
                    magnitude_scalar=magnitude_scalar,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=(
                        bootstrap_seed
                        + dataset_offset * 10_000
                        + dimension_offset * 100
                        + bin_index
                    ),
                )
                summary.update(
                    {
                        "bin_index": bin_index,
                        "value_min": float(metadata[name][dimension][indices].min()),
                        "value_max": float(metadata[name][dimension][indices].max()),
                        "sufficient_rows": summary["row_count"] >= min_rows,
                        "sufficient_families": summary["family_count"] >= min_families,
                    }
                )
                groups.append(summary)
            dimensions_report[dimension] = {
                "fit_edges": edges[dimension],
                "groups": groups,
            }
        datasets_report[name] = {
            "split": descriptors[name]["split"],
            "row_count": int(len(all_indices)),
            "cache": file_provenance(descriptors[name]["cache_npz"]),
            "score_report": file_provenance(
                descriptors[name]["score_report_json"]
            ),
            "generated_jsonl": file_provenance(
                descriptors[name]["generated_jsonl"]
            ),
            "score_report_gate_passed": bool(
                (score_reports[name].get("gate") or {}).get("passed")
            ),
            "overall": summarize_rows(
                dataset,
                all_indices,
                controls=controls,
                magnitude_scalar=magnitude_scalar,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + dataset_offset * 10_000 + 9_000,
            ),
            "dimensions": dimensions_report,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "config": str(config_path),
        "publication_status": str(
            protocol.get("publication_status") or "exploratory"
        ),
        "fit": {
            "dataset": fit_name,
            "split": descriptors[fit_name]["split"],
            "quantile_bins_requested": quantile_bins,
            "edges": edges,
        },
        "protocol": {
            "controls": controls,
            "magnitude_scalar": magnitude_scalar,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "min_rows_per_group": min_rows,
            "min_families_per_group": min_families,
        },
        "datasets": datasets_report,
        "claim_boundary": (
            "Bins are fitted on validation only and applied unchanged to the "
            "exploratory exposed test. Subgroups are descriptive robustness "
            "evidence, not a pristine confirmatory analysis."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = evaluate_config(config, args.config)
    output = Path(config["paths"]["report_json"])
    if not output.is_absolute():
        output = args.config.parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report_json": str(output),
                "fit": report["fit"],
                "datasets": {
                    name: {
                        "split": item["split"],
                        "row_count": item["row_count"],
                        "overall": item["overall"],
                    }
                    for name, item in report["datasets"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
