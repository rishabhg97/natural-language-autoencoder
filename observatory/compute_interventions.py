#!/usr/bin/env python3
"""Derive Shapley, Court, control, and clustered aggregate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .bundle_common import (
    bundle_config_fingerprint,
    bundle_path,
    family_bootstrap_interval,
    load_bundle_config,
    read_json,
    write_parquet_atomic,
)
from .common import (
    ObservatoryConfigError,
    config_fingerprint,
    load_config,
    read_jsonl,
    sha256_file,
    stable_int,
    write_json,
)


SCHEMA_VERSION = "nano_viz_intervention_derivation.v1"


def exact_shapley(values_by_mask: dict[int, float], features: int = 4) -> list[float]:
    expected = set(range(1 << features))
    if set(values_by_mask) != expected:
        missing = sorted(expected - set(values_by_mask))
        raise ObservatoryConfigError(f"Shapley lattice is incomplete: {missing}")
    denominator = math.factorial(features)
    result: list[float] = []
    for feature in range(features):
        contribution = 0.0
        bit = 1 << feature
        for mask in expected:
            if mask & bit:
                continue
            size = int(mask).bit_count()
            weight = (
                math.factorial(size) * math.factorial(features - size - 1) / denominator
            )
            contribution += weight * (
                float(values_by_mask[mask | bit]) - float(values_by_mask[mask])
            )
        result.append(contribution)
    return result


def fit_balanced_threshold(positive: list[float], negative: list[float]) -> dict[str, float]:
    positives = np.asarray(positive, dtype=np.float64)
    negatives = np.asarray(negative, dtype=np.float64)
    if not len(positives) or not len(negatives):
        raise ObservatoryConfigError("Court threshold requires both classes")
    if not np.isfinite(positives).all() or not np.isfinite(negatives).all():
        raise ObservatoryConfigError("Court scores must be finite")
    candidates = np.unique(np.concatenate([positives, negatives]))
    candidates = np.concatenate(
        [
            [np.nextafter(candidates[0], -np.inf)],
            (candidates[:-1] + candidates[1:]) / 2.0,
            [np.nextafter(candidates[-1], np.inf)],
        ]
    )
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        true_positive = float(np.mean(positives >= threshold))
        true_negative = float(np.mean(negatives < threshold))
        balanced = (true_positive + true_negative) / 2.0
        candidate = (balanced, true_positive, float(threshold))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    threshold = best[2]
    return {
        "threshold": threshold,
        "balanced_accuracy": best[0],
        "positive_recall": float(np.mean(positives >= threshold)),
        "negative_recall": float(np.mean(negatives < threshold)),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        raise ObservatoryConfigError("Court vectors must have non-zero norm")
    return float(np.dot(left, right) / denominator)


def run(config_path: Path) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    bundle_config = load_bundle_config(config_path)
    source_config = load_config(bundle_config["source_config"])
    source_hash = config_fingerprint(source_config)
    paths = bundle_config["paths"]
    corpus_dir = bundle_path(paths["corpus_dir"], config_path=config_path)
    model_outputs = bundle_path(paths["model_outputs_dir"], config_path=config_path)
    derived_dir = bundle_path(paths["derived_dir"], config_path=config_path)
    lattice_report = read_json(model_outputs / "p2_lattice_full" / "lattice_full_report.json")
    if not lattice_report.get("passed") or lattice_report.get("config_sha256") != source_hash:
        raise ObservatoryConfigError("full lattice must pass under the source config")

    interventions = read_jsonl(corpus_dir / "interventions.jsonl")
    alternate_report = read_json(
        model_outputs / "p1_alternate_tellings" / "alternate_tellings_report.json"
    )
    if not alternate_report.get("passed") or alternate_report.get("config_sha256") != source_hash:
        raise ObservatoryConfigError("alternate tellings must pass under the source config")
    alternate_records = {
        str(record["cell_id"]): record
        for record in read_jsonl(alternate_report["records"]["path"])
    }
    for intervention in interventions:
        if intervention["family"] != "alternate_telling":
            continue
        record = alternate_records.get(str(intervention["cell_id"]))
        parsed = None if record is None else record.get("parsed")
        text = None if not isinstance(parsed, dict) else parsed.get("explanation")
        if not isinstance(text, str) or not text.strip():
            raise ObservatoryConfigError(
                f"alternate telling lacks usable text: {intervention['cell_id']}"
            )
        intervention["text"] = text.strip()
        intervention["text_sha256"] = hashlib.sha256(
            intervention["text"].encode("utf-8")
        ).hexdigest()
        intervention["state"] = "ready"
    if len(alternate_records) != 400:
        raise ObservatoryConfigError(
            f"expected 400 alternate tellings, found {len(alternate_records)}"
        )
    intervention_by_id = {str(row["cell_id"]): row for row in interventions}
    if len(intervention_by_id) != len(interventions):
        raise ObservatoryConfigError("intervention registry contains duplicate cells")
    prediction_rows: list[dict[str, Any]] = []
    prediction_vectors: dict[tuple[str, str], np.ndarray] = {}
    for shard in lattice_report["shards"]:
        shard_path = Path(shard["path"])
        if sha256_file(shard_path) != shard["sha256"]:
            raise ObservatoryConfigError(f"lattice shard hash mismatch: {shard_path}")
        for prediction in pq.read_table(shard_path).to_pylist():
            cell_id = str(prediction["cell_id"])
            intervention = intervention_by_id.get(cell_id)
            if intervention is None:
                raise ObservatoryConfigError(f"prediction has unknown cell: {cell_id}")
            critic = str(prediction["critic"])
            key = (critic, cell_id)
            if key in prediction_vectors:
                raise ObservatoryConfigError(f"duplicate prediction: {key}")
            vector = np.asarray(prediction.pop("prediction_vector"), dtype=np.float32)
            if vector.shape != (2688,) or not np.isfinite(vector).all():
                raise ObservatoryConfigError(f"invalid prediction vector: {key}")
            prediction_vectors[key] = vector
            prediction_rows.append(
                {
                    "cell_id": cell_id,
                    "control_group_id": intervention.get("control_group_id"),
                    "row_id": str(intervention["row_id"]),
                    "row_index": int(intervention["row_index"]),
                    "content_family_id": str(prediction["content_family_id"]),
                    "family": str(intervention["family"]),
                    "variant": str(intervention["variant"]),
                    "depth": str(intervention["depth"]),
                    "critic": critic,
                    "directional_mse": float(prediction["directional_mse"]),
                    "raw_mse": float(prediction["raw_mse"]),
                    "cosine": float(prediction["cosine"]),
                    "norm_ratio": float(prediction["norm_ratio"]),
                    "spec_json": json.dumps(intervention["spec"], sort_keys=True),
                    "text": intervention.get("text"),
                }
            )

    primary_by_cell = {
        row["cell_id"]: row for row in prediction_rows if row["critic"] == "primary"
    }
    shapley_rows: list[dict[str, Any]] = []
    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for intervention in interventions:
        by_row[int(intervention["row_index"])].append(intervention)
    section_names = ["syntax", "discourse", "register", "final_token"]
    for row_index, row_interventions in sorted(by_row.items()):
        values = {
            int(cell["spec"]["mask"]): 1.0 - float(primary_by_cell[cell["cell_id"]]["directional_mse"])
            for cell in row_interventions
            if cell["family"] == "section_ablation"
        }
        contributions = exact_shapley(values, features=4)
        efficiency_error = abs(
            sum(contributions) - (float(values[(1 << 4) - 1]) - float(values[0]))
        )
        for section_index, contribution in enumerate(contributions):
            identity = primary_by_cell[
                next(cell["cell_id"] for cell in row_interventions if cell["family"] == "identity")
            ]
            shapley_rows.append(
                {
                    "row_id": f"validation-{row_index}",
                    "row_index": row_index,
                    "content_family_id": str(identity["content_family_id"]),
                    "section_index": section_index,
                    "section": section_names[section_index],
                    "utility": "one_minus_directional_mse",
                    "shapley_value": float(contribution),
                    "efficiency_error": float(efficiency_error),
                }
            )

    identity_vectors = {
        (critic, int(row["row_index"])): prediction_vectors[(critic, row["cell_id"])]
        for row in prediction_rows
        if row["family"] == "identity"
        for critic in [str(row["critic"])]
    }
    court_rows: list[dict[str, Any]] = []
    scores_by_critic: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )
    for row in prediction_rows:
        if row["family"] not in {"paraphrase", "corruption"}:
            continue
        key = (str(row["critic"]), int(row["row_index"]))
        identity_vector = identity_vectors.get(key)
        if identity_vector is None:
            raise ObservatoryConfigError(f"Court row lacks identity prediction: {key}")
        score = cosine(prediction_vectors[(str(row["critic"]), row["cell_id"])], identity_vector)
        spec = json.loads(str(row["spec_json"]))
        label = "positive" if row["family"] == "paraphrase" else "context"
        if row["family"] == "paraphrase":
            scores_by_critic[str(row["critic"])]["positive"].append(score)
        elif float(spec.get("rate", 0.0)) == 0.5:
            label = "negative"
            scores_by_critic[str(row["critic"])]["negative"].append(score)
        court_rows.append({**row, "identity_cosine": score, "calibration_label": label})
    court_fits = {
        critic: fit_balanced_threshold(values["positive"], values["negative"])
        for critic, values in scores_by_critic.items()
    }
    for row in court_rows:
        threshold = court_fits[str(row["critic"])]["threshold"]
        row["semanticity_verdict"] = bool(float(row["identity_cosine"]) >= threshold)

    control_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for intervention in interventions:
        if intervention.get("control_group_id"):
            control_groups[str(intervention["control_group_id"])].append(intervention)
    incomplete_groups: list[str] = []
    for group_id, cells in control_groups.items():
        lanes_by_dose: dict[float, set[str]] = defaultdict(set)
        for cell in cells:
            spec = cell["spec"]
            lanes_by_dose[float(spec["dose"])].add(str(spec["lane"]))
        if any(lanes != {"edit", "paraphrase_placebo", "random_edit"} for lanes in lanes_by_dose.values()):
            incomplete_groups.append(group_id)

    statistics = bundle_config["statistics"]
    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: dict[str, Any] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        groups[(str(row["critic"]), str(row["family"]))].append(row)
    for group_index, ((critic, family), rows_for_group) in enumerate(sorted(groups.items())):
        for metric in ("directional_mse", "cosine"):
            interval = family_bootstrap_interval(
                [float(row[metric]) for row in rows_for_group],
                [str(row["content_family_id"]) for row in rows_for_group],
                samples=int(statistics["bootstrap_samples"]),
                confidence=float(statistics["confidence"]),
                seed=stable_int(int(statistics["seed"]), group_index, metric),
            )
            key = f"{critic}.{family}.{metric}"
            aggregate_json[key] = interval
            aggregate_rows.append(
                {"critic": critic, "family": family, "metric": metric, **interval}
            )

    metric_schema = pa.schema(
        [
            ("cell_id", pa.string()),
            ("control_group_id", pa.string()),
            ("row_id", pa.string()),
            ("row_index", pa.int64()),
            ("content_family_id", pa.string()),
            ("family", pa.string()),
            ("variant", pa.string()),
            ("depth", pa.string()),
            ("critic", pa.string()),
            ("directional_mse", pa.float32()),
            ("raw_mse", pa.float32()),
            ("cosine", pa.float32()),
            ("norm_ratio", pa.float32()),
            ("spec_json", pa.string()),
            ("text", pa.string()),
        ]
    )
    shapley_schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("row_index", pa.int64()),
            ("content_family_id", pa.string()),
            ("section_index", pa.int64()),
            ("section", pa.string()),
            ("utility", pa.string()),
            ("shapley_value", pa.float32()),
            ("efficiency_error", pa.float32()),
        ]
    )
    court_schema = pa.schema(
        list(metric_schema)
        + [
            ("identity_cosine", pa.float32()),
            ("calibration_label", pa.string()),
            ("semanticity_verdict", pa.bool_()),
        ]
    )
    aggregate_schema = pa.schema(
        [
            ("critic", pa.string()),
            ("family", pa.string()),
            ("metric", pa.string()),
            ("mean", pa.float64()),
            ("ci_low", pa.float64()),
            ("ci_high", pa.float64()),
            ("rows", pa.int64()),
            ("families", pa.int64()),
            ("bootstrap_samples", pa.int64()),
        ]
    )
    artifacts = {
        "intervention_metrics": derived_dir / "intervention_metrics.parquet",
        "shapley": derived_dir / "shapley.parquet",
        "court": derived_dir / "court.parquet",
        "aggregates": derived_dir / "aggregates.parquet",
    }
    write_parquet_atomic(artifacts["intervention_metrics"], prediction_rows, metric_schema)
    write_parquet_atomic(artifacts["shapley"], shapley_rows, shapley_schema)
    write_parquet_atomic(artifacts["court"], court_rows, court_schema)
    write_parquet_atomic(artifacts["aggregates"], aggregate_rows, aggregate_schema)
    aggregates_json_path = derived_dir / "aggregates.json"
    write_json(
        aggregates_json_path,
        {
            "schema_version": SCHEMA_VERSION,
            "fit_split": "validation",
            "family_clustered": True,
            "aggregates": aggregate_json,
            "court_thresholds": court_fits,
        },
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": (
            len(primary_by_cell) == len(interventions)
            and len(shapley_rows) == 200
            and not incomplete_groups
            and all(float(row["efficiency_error"]) <= 1e-9 for row in shapley_rows)
            and set(court_fits) == {"primary", "independent"}
        ),
        "source_config_sha256": source_hash,
        "bundle_config_sha256": bundle_config_fingerprint(bundle_config),
        "fit_split": "validation",
        "prediction_rows": len(prediction_rows),
        "primary_cells": len(primary_by_cell),
        "shapley_rows": len(shapley_rows),
        "court_rows": len(court_rows),
        "control_groups": len(control_groups),
        "incomplete_control_groups": incomplete_groups,
        "court_thresholds": court_fits,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        }
        | {
            "aggregates_json": {
                "path": str(aggregates_json_path),
                "sha256": sha256_file(aggregates_json_path),
            }
        },
    }
    write_json(derived_dir / "intervention_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.config)
    except (OSError, ValueError, ObservatoryConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
