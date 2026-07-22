#!/usr/bin/env python3
"""Fit validation-only Observatory geometry and native-space retrieval evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .bundle_common import (
    bundle_config_fingerprint,
    bundle_path,
    load_bundle_config,
    read_json,
    write_parquet_atomic,
)
from .common import (
    ObservatoryConfigError,
    config_fingerprint,
    load_config,
    sha256_file,
    write_json,
)


SCHEMA_VERSION = "nano_viz_geometry.v1"


def fit_validation_basis(vectors: np.ndarray, components: int) -> dict[str, np.ndarray]:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ObservatoryConfigError("geometry vectors must be a finite 2D matrix")
    component_count = min(int(components), matrix.shape[0] - 1, matrix.shape[1])
    if component_count < 1:
        raise ObservatoryConfigError("geometry requires at least one component")
    mean = matrix.mean(axis=0)
    _, singular_values, right = np.linalg.svd(matrix - mean, full_matrices=False)
    variance = np.square(singular_values)
    total = float(variance.sum())
    explained = variance[:component_count] / total if total else np.zeros(component_count)
    return {
        "mean": mean.astype(np.float32),
        "basis": right[:component_count].astype(np.float32),
        "singular_values": singular_values[:component_count].astype(np.float32),
        "explained_variance_ratio": explained.astype(np.float32),
    }


def target_retrieval_rank(
    prediction: np.ndarray, targets: np.ndarray, expected_index: int
) -> tuple[int, int, float]:
    candidate = np.asarray(prediction, dtype=np.float64)
    matrix = np.asarray(targets, dtype=np.float64)
    candidate_norm = np.linalg.norm(candidate)
    target_norms = np.linalg.norm(matrix, axis=1)
    if candidate_norm <= 0.0 or np.any(target_norms <= 0.0):
        raise ObservatoryConfigError("retrieval vectors must have non-zero norm")
    cosine = matrix @ candidate / (target_norms * candidate_norm)
    order = np.argsort(-cosine, kind="stable")
    locations = np.flatnonzero(order == int(expected_index))
    if len(locations) != 1:
        raise ObservatoryConfigError("expected retrieval target is not unique")
    return int(locations[0]) + 1, int(order[0]), float(cosine[expected_index])


def run(config_path: Path) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    bundle_config = load_bundle_config(config_path)
    source_config_path = Path(bundle_config["source_config"])
    source_config = load_config(source_config_path)
    source_hash = config_fingerprint(source_config)
    paths = bundle_config["paths"]
    model_outputs = bundle_path(paths["model_outputs_dir"], config_path=config_path)
    derived_dir = bundle_path(paths["derived_dir"], config_path=config_path)
    report_path = derived_dir / "geometry_report.json"

    lattice_report = read_json(model_outputs / "p2_lattice_full" / "lattice_full_report.json")
    if not lattice_report.get("passed") or lattice_report.get("config_sha256") != source_hash:
        raise ObservatoryConfigError("full lattice must pass under the source config")
    source_path = Path(source_config["paths"]["source_base_selected_parquet"])
    source_table = pq.read_table(
        source_path,
        columns=["row_index", "doc_id", "content_family_id", "activation_vector"],
    )
    sources = source_table.to_pylist()
    if len(sources) != 50:
        raise ObservatoryConfigError(f"geometry requires exactly 50 targets, found {len(sources)}")
    sources.sort(key=lambda row: int(row["row_index"]))
    targets = np.asarray([row["activation_vector"] for row in sources], dtype=np.float32)
    target_index = {int(row["row_index"]): index for index, row in enumerate(sources)}
    if len(target_index) != len(sources) or not np.isfinite(targets).all():
        raise ObservatoryConfigError("target identities must be unique and finite")

    geometry_config = bundle_config["geometry"]
    fitted = fit_validation_basis(targets, int(geometry_config["components"]))
    display_components = int(geometry_config["display_components"])
    if not 1 <= display_components <= fitted["basis"].shape[0]:
        raise ObservatoryConfigError("invalid display component count")
    basis = fitted["basis"]
    mean = fitted["mean"]

    rows: list[dict[str, Any]] = []
    retrieval: list[dict[str, Any]] = []
    target_projection = (targets - mean) @ basis.T
    for index, (source, coordinates) in enumerate(zip(sources, target_projection, strict=True)):
        rows.append(
            {
                "ref": f"target:validation-{int(source['row_index'])}",
                "kind": "target",
                "row_id": f"validation-{int(source['row_index'])}",
                "row_index": int(source["row_index"]),
                "content_family_id": str(source["content_family_id"]),
                "cell_id": None,
                "family": "target",
                "variant": "target",
                "critic": None,
                "x": float(coordinates[0]) if display_components >= 1 else None,
                "y": float(coordinates[1]) if display_components >= 2 else None,
                "z": float(coordinates[2]) if display_components >= 3 else None,
                "native_norm": float(np.linalg.norm(targets[index])),
                "target_cosine": 1.0,
                "directional_mse": 0.0,
                "vector_source": str(source_path),
                "vector_row": index,
            }
        )

    prediction_count = 0
    for shard in lattice_report["shards"]:
        shard_path = Path(shard["path"])
        if sha256_file(shard_path) != shard["sha256"]:
            raise ObservatoryConfigError(f"lattice shard hash mismatch: {shard_path}")
        table = pq.read_table(shard_path)
        for shard_row, prediction in enumerate(table.to_pylist()):
            row_index = int(prediction["row_index"])
            expected_index = target_index.get(row_index)
            if expected_index is None:
                raise ObservatoryConfigError(f"prediction references unknown row {row_index}")
            vector = np.asarray(prediction["prediction_vector"], dtype=np.float32)
            if vector.shape != (2688,) or not np.isfinite(vector).all():
                raise ObservatoryConfigError(f"invalid prediction vector for {prediction['cell_id']}")
            coordinates = (vector - mean) @ basis.T
            rows.append(
                {
                    "ref": f"prediction:{prediction['critic']}:{prediction['cell_id']}",
                    "kind": "prediction",
                    "row_id": f"validation-{row_index}",
                    "row_index": row_index,
                    "content_family_id": str(prediction["content_family_id"]),
                    "cell_id": str(prediction["cell_id"]),
                    "family": str(prediction["family"]),
                    "variant": str(prediction["variant"]),
                    "critic": str(prediction["critic"]),
                    "x": float(coordinates[0]) if display_components >= 1 else None,
                    "y": float(coordinates[1]) if display_components >= 2 else None,
                    "z": float(coordinates[2]) if display_components >= 3 else None,
                    "native_norm": float(np.linalg.norm(vector)),
                    "target_cosine": float(prediction["cosine"]),
                    "directional_mse": float(prediction["directional_mse"]),
                    "vector_source": str(shard_path),
                    "vector_row": shard_row,
                }
            )
            if prediction["family"] == "identity":
                rank, nearest_index, expected_cosine = target_retrieval_rank(
                    vector, targets, expected_index
                )
                retrieval.append(
                    {
                        "row_id": f"validation-{row_index}",
                        "row_index": row_index,
                        "content_family_id": str(prediction["content_family_id"]),
                        "critic": str(prediction["critic"]),
                        "rank": rank,
                        "nearest_row_id": f"validation-{int(sources[nearest_index]['row_index'])}",
                        "expected_cosine": expected_cosine,
                    }
                )
            prediction_count += 1

    geometry_schema = pa.schema(
        [
            ("ref", pa.string()),
            ("kind", pa.string()),
            ("row_id", pa.string()),
            ("row_index", pa.int64()),
            ("content_family_id", pa.string()),
            ("cell_id", pa.string()),
            ("family", pa.string()),
            ("variant", pa.string()),
            ("critic", pa.string()),
            ("x", pa.float32()),
            ("y", pa.float32()),
            ("z", pa.float32()),
            ("native_norm", pa.float32()),
            ("target_cosine", pa.float32()),
            ("directional_mse", pa.float32()),
            ("vector_source", pa.string()),
            ("vector_row", pa.int64()),
        ]
    )
    retrieval_schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("row_index", pa.int64()),
            ("content_family_id", pa.string()),
            ("critic", pa.string()),
            ("rank", pa.int64()),
            ("nearest_row_id", pa.string()),
            ("expected_cosine", pa.float32()),
        ]
    )
    geometry_path = derived_dir / "geometry.parquet"
    retrieval_path = derived_dir / "retrieval.parquet"
    basis_path = derived_dir / "geometry_basis.npz"
    write_parquet_atomic(geometry_path, rows, geometry_schema)
    write_parquet_atomic(retrieval_path, retrieval, retrieval_schema)
    derived_dir.mkdir(parents=True, exist_ok=True)
    with basis_path.open("wb") as handle:
        np.savez_compressed(handle, **fitted)

    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": (
            len(sources) == 50
            and prediction_count == sum(int(shard["rows"]) for shard in lattice_report["shards"])
            and len(retrieval) == 100
            and all(row["rank"] >= 1 for row in retrieval)
        ),
        "source_config_sha256": source_hash,
        "bundle_config_sha256": bundle_config_fingerprint(bundle_config),
        "fit_split": "validation",
        "claim_scope": "stored_snapshot",
        "targets": len(sources),
        "predictions": prediction_count,
        "identity_retrieval_rows": len(retrieval),
        "components": int(fitted["basis"].shape[0]),
        "explained_variance_ratio": [
            float(value) for value in fitted["explained_variance_ratio"]
        ],
        "artifacts": {
            "geometry": {"path": str(geometry_path), "sha256": sha256_file(geometry_path)},
            "retrieval": {"path": str(retrieval_path), "sha256": sha256_file(retrieval_path)},
            "basis": {"path": str(basis_path), "sha256": sha256_file(basis_path)},
        },
    }
    write_json(report_path, report)
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
