#!/usr/bin/env python3
"""Fit and audit a low-capacity AR activation-magnitude calibration.

The calibration is deliberately separate from round-trip generation. It reads
immutable prediction caches, fits on a non-test split, and applies one shared
scalar transform to every requested AR input variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_eval_core import (  # noqa: E402
    activation_reconstruction_metrics,
    clustered_paired_bootstrap_improvement,
)


SCHEMA_VERSION = "nano_activation_magnitude_calibration.v1"
CACHE_SCHEMA_VERSION = "nano_roundtrip_prediction_cache.v1"
ALLOWED_METHODS = ("identity", "origin_scalar", "train_mean_scalar")
ROWWISE_KEYS = (
    "rowwise_directional_mse",
    "rowwise_unit_vector_feature_mse",
    "rowwise_raw_mse",
)


class CalibrationError(ValueError):
    pass


def file_provenance(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _as_path(value: Any, *, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationError(f"config must use schema_version {SCHEMA_VERSION}")
    paths = config.get("paths")
    protocol = config.get("protocol")
    if not isinstance(paths, dict) or not paths.get("report_json"):
        raise CalibrationError("paths.report_json is required")
    datasets = paths.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise CalibrationError("paths.datasets must be a non-empty mapping")
    for name, item in datasets.items():
        if (
            not isinstance(item, dict)
            or not item.get("cache_npz")
            or not item.get("score_report_json")
            or not item.get("split")
        ):
            raise CalibrationError(
                f"paths.datasets.{name} requires cache_npz, score_report_json, and split"
            )
    if not isinstance(protocol, dict):
        raise CalibrationError("protocol is required")
    fit_dataset = str(protocol.get("fit_dataset") or "")
    if fit_dataset not in datasets:
        raise CalibrationError("protocol.fit_dataset must name a configured dataset")
    fit_split = str(datasets[fit_dataset]["split"])
    if fit_split == "test" or "test" in fit_dataset.lower():
        raise CalibrationError("calibration fitting on test data is prohibited")
    methods = list(protocol.get("candidate_methods") or [])
    if not methods or len(methods) != len(set(methods)):
        raise CalibrationError("protocol.candidate_methods must be non-empty and unique")
    unknown = sorted(set(methods) - set(ALLOWED_METHODS))
    if unknown:
        raise CalibrationError(f"unsupported calibration methods: {unknown}")
    variants = list(protocol.get("evaluation_variants") or [])
    if not variants or len(variants) != len(set(variants)):
        raise CalibrationError("protocol.evaluation_variants must be non-empty and unique")
    if str(protocol.get("fit_variant") or "") not in variants:
        raise CalibrationError("protocol.fit_variant must be an evaluation variant")
    if int(protocol.get("bootstrap_samples", 10_000)) <= 0:
        raise CalibrationError("protocol.bootstrap_samples must be positive")
    if protocol.get("selection_metric", "raw_mse") != "raw_mse":
        raise CalibrationError("protocol.selection_metric must be raw_mse")
    if protocol.get("nonnegative_scalar", True) is not True:
        raise CalibrationError("protocol.nonnegative_scalar must be true")
    return config


def load_cache_split(path: Path, split: str) -> dict[str, Any]:
    if not path.is_file():
        raise CalibrationError(f"prediction cache does not exist: {path}")
    prefix = f"{split}__"
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "metadata_json",
            "train_mean",
            f"{prefix}row_indices",
            f"{prefix}doc_ids",
            f"{prefix}content_family_ids",
            f"{prefix}targets",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise CalibrationError(f"prediction cache is missing arrays: {missing}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise CalibrationError(
                f"prediction cache must use schema_version {CACHE_SCHEMA_VERSION}"
            )
        predictions = {
            key.removeprefix(f"{prefix}prediction__"): np.asarray(
                payload[key], dtype=np.float64
            )
            for key in payload.files
            if key.startswith(f"{prefix}prediction__")
        }
        result = {
            "metadata": metadata,
            "train_mean": np.asarray(payload["train_mean"], dtype=np.float64),
            "row_indices": np.asarray(payload[f"{prefix}row_indices"], dtype=np.int64),
            "doc_ids": np.asarray(payload[f"{prefix}doc_ids"], dtype=np.str_),
            "content_family_ids": np.asarray(
                payload[f"{prefix}content_family_ids"], dtype=np.str_
            ),
            "targets": np.asarray(payload[f"{prefix}targets"], dtype=np.float64),
            "predictions": predictions,
        }
    targets = result["targets"]
    if targets.ndim != 2 or targets.size == 0 or not np.isfinite(targets).all():
        raise CalibrationError("cache targets must be a non-empty finite matrix")
    if result["train_mean"].shape != (targets.shape[1],):
        raise CalibrationError("cache train_mean does not match activation dimension")
    row_count = targets.shape[0]
    for key in ("row_indices", "doc_ids", "content_family_ids"):
        if len(result[key]) != row_count:
            raise CalibrationError(f"cache {key} does not align with targets")
    for variant, prediction in predictions.items():
        if prediction.shape != targets.shape or not np.isfinite(prediction).all():
            raise CalibrationError(
                f"cache prediction {variant!r} must match finite target shape"
            )
    return result


def load_bound_score_report(path: Path, cache_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CalibrationError(f"score report does not exist: {path}")
    report = json.loads(path.read_text())
    cache = report.get("prediction_cache") if isinstance(report, dict) else None
    if not isinstance(cache, dict):
        raise CalibrationError("score report does not bind a prediction cache")
    observed = file_provenance(cache_path)
    for key in ("sha256", "size_bytes"):
        if cache.get(key) != observed[key]:
            raise CalibrationError(
                f"score report prediction-cache {key} does not match cache bytes"
            )
    if (report.get("gate") or {}).get("passed") is not True:
        raise CalibrationError("score report gate must pass before calibration")
    return report


def validate_cache_compatibility(loaded: dict[str, dict[str, Any]]) -> None:
    names = list(loaded)
    reference = loaded[names[0]]
    reference_metadata = reference["metadata"]
    reference_train = (reference_metadata.get("dataset_provenance") or {}).get("train")
    if not isinstance(reference_train, dict) or not reference_train.get("sha256"):
        raise CalibrationError("prediction cache is missing train dataset provenance")
    identity_keys = ("ar_checkpoint_dir", "ar_hf_dir", "critic_template_sha256")
    for name in names[1:]:
        candidate = loaded[name]
        metadata = candidate["metadata"]
        mismatches = [
            key
            for key in identity_keys
            if metadata.get(key) != reference_metadata.get(key)
        ]
        candidate_train = (metadata.get("dataset_provenance") or {}).get("train")
        if not isinstance(candidate_train, dict) or candidate_train.get("sha256") != reference_train["sha256"]:
            mismatches.append("train_dataset_sha256")
        if not np.array_equal(candidate["train_mean"], reference["train_mean"]):
            mismatches.append("train_mean")
        if candidate["targets"].shape[1] != reference["targets"].shape[1]:
            mismatches.append("activation_dimension")
        if mismatches:
            raise CalibrationError(
                f"prediction cache {name!r} is incompatible with {names[0]!r}: {mismatches}"
            )


def fit_method(
    method: str,
    prediction: np.ndarray,
    target: np.ndarray,
    train_mean: np.ndarray,
) -> dict[str, Any]:
    if method == "identity":
        return {"method": method, "scalar": 1.0, "anchor": "origin"}
    if method == "origin_scalar":
        centered_prediction = prediction
        centered_target = target
        anchor = "origin"
    elif method == "train_mean_scalar":
        centered_prediction = prediction - train_mean[None, :]
        centered_target = target - train_mean[None, :]
        anchor = "train_mean"
    else:
        raise CalibrationError(f"unsupported calibration method: {method}")
    denominator = float(np.sum(np.square(centered_prediction), dtype=np.float64))
    if denominator <= 0.0:
        raise CalibrationError(f"cannot fit {method}: prediction energy is zero")
    unconstrained = float(
        np.sum(centered_prediction * centered_target, dtype=np.float64) / denominator
    )
    scalar = max(0.0, unconstrained)
    return {
        "method": method,
        "scalar": scalar,
        "unconstrained_scalar": unconstrained,
        "nonnegative_constraint_active": scalar != unconstrained,
        "anchor": anchor,
    }


def apply_method(
    parameters: dict[str, Any],
    prediction: np.ndarray,
    train_mean: np.ndarray,
) -> np.ndarray:
    scalar = float(parameters["scalar"])
    if parameters["anchor"] == "train_mean":
        return train_mean[None, :] + scalar * (prediction - train_mean[None, :])
    return scalar * prediction


def metric_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    train_mean: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    metrics = activation_reconstruction_metrics(
        prediction,
        target,
        train_mean=train_mean,
    )
    rowwise_raw = np.asarray(metrics["rowwise_raw_mse"], dtype=np.float64)
    return ({key: value for key, value in metrics.items() if key not in ROWWISE_KEYS}, rowwise_raw)


def choose_cluster_ids(dataset: dict[str, Any]) -> tuple[str, list[str]]:
    families = [str(value) for value in dataset["content_family_ids"]]
    if families and all(families) and len(set(families)) >= 2:
        return "content_family_id", families
    docs = [str(value) for value in dataset["doc_ids"]]
    if docs and all(docs) and len(set(docs)) >= 2:
        return "doc_id", docs
    raise CalibrationError("at least two non-empty family or document clusters are required")


def evaluate_config(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    paths = config["paths"]
    protocol = config["protocol"]
    base = config_path.parent
    datasets = {
        name: {
            "path": _as_path(item["cache_npz"], base=base),
            "score_report_path": _as_path(item["score_report_json"], base=base),
            "split": str(item["split"]),
        }
        for name, item in paths["datasets"].items()
    }
    loaded = {
        name: load_cache_split(item["path"], item["split"])
        for name, item in datasets.items()
    }
    score_reports = {
        name: load_bound_score_report(item["score_report_path"], item["path"])
        for name, item in datasets.items()
    }
    validate_cache_compatibility(loaded)
    dimensions = {dataset["targets"].shape[1] for dataset in loaded.values()}
    if len(dimensions) != 1:
        raise CalibrationError("all prediction caches must have one activation dimension")

    fit_name = str(protocol["fit_dataset"])
    fit_variant = str(protocol["fit_variant"])
    fit_data = loaded[fit_name]
    if fit_variant not in fit_data["predictions"]:
        raise CalibrationError(f"fit variant {fit_variant!r} is absent from fit cache")
    parameters = {
        method: fit_method(
            method,
            fit_data["predictions"][fit_variant],
            fit_data["targets"],
            fit_data["train_mean"],
        )
        for method in protocol["candidate_methods"]
    }
    fit_metrics: dict[str, dict[str, Any]] = {}
    for method, fitted in parameters.items():
        calibrated = apply_method(
            fitted,
            fit_data["predictions"][fit_variant],
            fit_data["train_mean"],
        )
        fit_metrics[method], _ = metric_summary(
            calibrated, fit_data["targets"], fit_data["train_mean"]
        )
    selected = min(
        protocol["candidate_methods"],
        key=lambda method: (
            float(fit_metrics[method]["raw_mse"]),
            list(protocol["candidate_methods"]).index(method),
        ),
    )

    evaluation: dict[str, Any] = {}
    bootstrap_samples = int(protocol.get("bootstrap_samples", 10_000))
    bootstrap_seed = int(protocol.get("bootstrap_seed", 0))
    for dataset_name, dataset in loaded.items():
        cluster_unit, cluster_ids = choose_cluster_ids(dataset)
        variants: dict[str, Any] = {}
        for variant in protocol["evaluation_variants"]:
            if variant not in dataset["predictions"]:
                raise CalibrationError(
                    f"evaluation variant {variant!r} is absent from {dataset_name} cache"
                )
            prediction = dataset["predictions"][variant]
            candidate_metrics: dict[str, Any] = {}
            rowwise: dict[str, np.ndarray] = {}
            for method, fitted in parameters.items():
                calibrated = apply_method(fitted, prediction, dataset["train_mean"])
                candidate_metrics[method], rowwise[method] = metric_summary(
                    calibrated, dataset["targets"], dataset["train_mean"]
                )
            identity_key = "identity" if "identity" in rowwise else None
            paired = None
            if identity_key is not None and selected != identity_key:
                paired = clustered_paired_bootstrap_improvement(
                    rowwise[identity_key],
                    rowwise[selected],
                    cluster_ids,
                    seed=bootstrap_seed,
                    resamples=bootstrap_samples,
                )
                paired["cluster_unit"] = cluster_unit
                paired["metric"] = "rowwise_raw_mse"
            variants[variant] = {
                "candidates": candidate_metrics,
                "selected_method": selected,
                "selected_metrics": candidate_metrics[selected],
                "selected_vs_identity_clustered_bootstrap": paired,
            }
        evaluation[dataset_name] = {
            "cache": file_provenance(datasets[dataset_name]["path"]),
            "score_report": file_provenance(
                datasets[dataset_name]["score_report_path"]
            ),
            "score_report_gate": score_reports[dataset_name]["gate"],
            "cache_metadata": dataset["metadata"],
            "split": datasets[dataset_name]["split"],
            "row_count": int(dataset["targets"].shape[0]),
            "independent_cluster_unit": cluster_unit,
            "independent_cluster_count": len(set(cluster_ids)),
            "variants": variants,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "publication_status": str(protocol.get("publication_status") or "exploratory"),
        "claim_boundary": (
            "Post-hoc low-capacity magnitude calibration fit on validation only; "
            "it does not create an unexposed confirmatory test boundary."
        ),
        "config": file_provenance(config_path),
        "protocol": protocol,
        "fit": {
            "dataset": fit_name,
            "split": datasets[fit_name]["split"],
            "variant": fit_variant,
            "row_count": int(fit_data["targets"].shape[0]),
            "candidate_parameters": parameters,
            "candidate_metrics": fit_metrics,
            "selected_method": selected,
            "selection_metric": "raw_mse",
        },
        "evaluation": evaluation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = evaluate_config(config, config_path=args.config)
    report_path = _as_path(config["paths"]["report_json"], base=args.config.parent)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report_json": str(report_path), "fit": report["fit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
