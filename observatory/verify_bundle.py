#!/usr/bin/env python3
"""Fail closed when an offline Observatory bundle is incomplete or inconsistent."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .build_bundle import SCHEMA_VERSION, manifest_bundle_id
from .bundle_common import (
    bundle_config_fingerprint,
    bundle_path,
    family_bootstrap_interval,
    load_bundle_config,
    read_json,
)
from .common import (
    ObservatoryConfigError,
    config_fingerprint,
    load_config,
    sha256_file,
    stable_int,
    write_json,
)
from audit_nano_release_text import scan_sensitive_text  # noqa: E402


REQUIRED_FILES = {
    "rows.parquet",
    "metrics.parquet",
    "explanations.parquet",
    "interventions.parquet",
    "behavior.parquet",
    "token_trajectories.parquet",
    "geometry.parquet",
    "geometry_basis.npz",
    "retrieval.parquet",
    "shapley.parquet",
    "court.parquet",
    "aggregates.parquet",
    "aggregates.json",
    "vector_index.parquet",
    "vectors/all.f16.bin",
    "assets/bundle_config.yaml",
    "assets/source_config.yaml",
    "assets/claim_ledger.json",
    "provenance.json",
}


def _duplicates(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    duplicate: set[Any] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return sorted(duplicate, key=str)


def validate_vector_layout(index_rows: list[dict[str, Any]], binary_bytes: int) -> list[str]:
    """Validate a packed float16 vector file without loading it into RAM."""

    errors: list[str] = []
    expected_offset = 0
    refs: list[str] = []
    for row_number, row in enumerate(index_rows):
        ref = str(row.get("ref"))
        refs.append(ref)
        offset = int(row.get("offset_elements", -1))
        length = int(row.get("length_elements", -1))
        if offset != expected_offset:
            errors.append(
                f"vector index row {row_number} has offset {offset}, expected {expected_offset}"
            )
        if length != 2688:
            errors.append(f"vector index row {row_number} has length {length}, expected 2688")
        if row.get("dtype") != "float16_le":
            errors.append(f"vector index row {row_number} has invalid dtype")
        expected_offset = offset + max(length, 0)
    if _duplicates(refs):
        errors.append("vector index contains duplicate refs")
    expected_bytes = expected_offset * np.dtype("<f2").itemsize
    if binary_bytes != expected_bytes:
        errors.append(f"vector binary has {binary_bytes} bytes, expected {expected_bytes}")
    return errors


def _require_columns(table: Any, required: set[str], label: str, errors: list[str]) -> None:
    missing = sorted(required - set(table.column_names))
    if missing:
        errors.append(f"{label} missing columns: {missing}")


def _finite(table: Any, columns: Iterable[str], label: str, errors: list[str]) -> None:
    for column in columns:
        if column not in table.column_names:
            continue
        values = np.asarray(table[column].to_numpy(zero_copy_only=False), dtype=np.float64)
        if not np.isfinite(values).all():
            errors.append(f"{label}.{column} contains non-finite values")


def _verify_control_groups(interventions: list[dict[str, Any]], errors: list[str]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interventions:
        if row.get("control_group_id"):
            groups[str(row["control_group_id"])].append(row)
    if not groups:
        errors.append("intervention lattice contains no control groups")
        return
    for group_id, rows in groups.items():
        lanes_by_dose: dict[float, set[str]] = defaultdict(set)
        try:
            for row in rows:
                spec = json.loads(str(row["spec_json"]))
                lanes_by_dose[float(spec["dose"])].add(str(spec["lane"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors.append(f"control group has invalid spec JSON: {group_id}")
            continue
        if any(
            lanes != {"edit", "paraphrase_placebo", "random_edit"}
            for lanes in lanes_by_dose.values()
        ):
            errors.append(f"control group is incomplete: {group_id}")


def _verify_sensitive_text(tables: dict[str, Any], errors: list[str]) -> dict[str, int]:
    findings: Counter[str] = Counter()
    fields = {
        "rows": ("source_text", "target_explanation", "av_explanation"),
        "interventions": ("text",),
        "explanations": ("text",),
        "behavior": ("baseline_continuation_text", "patched_continuation_text"),
    }
    for table_name, columns in fields.items():
        table = tables.get(table_name)
        if table is None:
            continue
        available = [column for column in columns if column in table.column_names]
        for row in table.select(available).to_pylist():
            for column in available:
                text = row.get(column)
                if isinstance(text, str) and text:
                    findings.update(scan_sensitive_text(text))
    result = dict(sorted(findings.items()))
    if result:
        errors.append(f"sensitive-text scan produced findings: {result}")
    return result


def _verify_aggregates(
    metrics: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    bundle_config: dict[str, Any],
    errors: list[str],
) -> None:
    statistics = bundle_config["statistics"]
    tolerance = float(bundle_config["verification"]["aggregate_tolerance"])
    observed = {
        (str(row["critic"]), str(row["family"]), str(row["metric"])): row
        for row in aggregate_rows
    }
    if len(observed) != len(aggregate_rows):
        errors.append("aggregates.parquet contains duplicate group keys")
        return
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        groups[(str(row["critic"]), str(row["family"]))].append(row)
    expected_keys: set[tuple[str, str, str]] = set()
    for group_index, ((critic, family), rows) in enumerate(sorted(groups.items())):
        for metric in ("directional_mse", "cosine"):
            key = (critic, family, metric)
            expected_keys.add(key)
            expected = family_bootstrap_interval(
                [float(row[metric]) for row in rows],
                [str(row["content_family_id"]) for row in rows],
                samples=int(statistics["bootstrap_samples"]),
                confidence=float(statistics["confidence"]),
                seed=stable_int(int(statistics["seed"]), group_index, metric),
            )
            actual = observed.get(key)
            if actual is None:
                errors.append(f"aggregate is missing: {key}")
                continue
            for field in ("mean", "ci_low", "ci_high"):
                if abs(float(actual[field]) - float(expected[field])) > tolerance:
                    errors.append(f"aggregate mismatch: {key}.{field}")
            for field in ("rows", "families", "bootstrap_samples"):
                if int(actual[field]) != int(expected[field]):
                    errors.append(f"aggregate count mismatch: {key}.{field}")
    if set(observed) != expected_keys:
        errors.append("aggregate group keys do not exactly match metric groups")


def _vector_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(np.float64)
    target = target.astype(np.float64)
    prediction_norm = float(np.linalg.norm(prediction))
    target_norm = float(np.linalg.norm(target))
    if prediction_norm <= 0.0 or target_norm <= 0.0:
        raise ObservatoryConfigError("bundle vectors must have non-zero norms")
    prediction_unit = prediction / prediction_norm
    target_unit = target / target_norm
    return {
        "directional_mse": float(np.square(prediction_unit - target_unit).sum()),
        "raw_mse": float(np.square(prediction - target).mean()),
        "cosine": float(np.dot(prediction_unit, target_unit)),
        "norm_ratio": prediction_norm / target_norm,
    }


def _verify_vector_metrics(
    *,
    vector_data: np.memmap,
    vector_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    checks: int,
    rtol: float,
    atol: float,
    errors: list[str],
) -> None:
    index = {str(row["ref"]): row for row in vector_rows}

    def vector(ref: str) -> np.ndarray:
        row = index[ref]
        start = int(row["offset_elements"])
        stop = start + int(row["length_elements"])
        return np.asarray(vector_data[start:stop], dtype=np.float32)

    selected = np.linspace(0, len(metrics) - 1, min(checks, len(metrics)), dtype=np.int64)
    for metric_index in selected:
        row = metrics[int(metric_index)]
        prediction_ref = f"prediction:{row['critic']}:{row['cell_id']}"
        target_ref = f"target:{row['row_id']}"
        if prediction_ref not in index or target_ref not in index:
            errors.append(f"metric recomputation lacks vector refs: {prediction_ref}")
            continue
        recomputed = _vector_metrics(vector(prediction_ref), vector(target_ref))
        for metric, value in recomputed.items():
            if not np.isclose(value, float(row[metric]), rtol=rtol, atol=atol):
                errors.append(
                    f"vector metric mismatch: {row['critic']}:{row['cell_id']}:{metric} "
                    f"stored={float(row[metric]):.8g} recomputed={value:.8g}"
                )


def _verify_manifest_files(
    bundle_dir: Path,
    manifest: dict[str, Any],
    errors: list[str],
) -> None:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        errors.append("manifest files must be a list")
        return
    listed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("manifest contains a non-object file entry")
            continue
        relative = str(entry.get("path") or "")
        relative_path = Path(relative)
        if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"manifest contains unsafe path: {relative!r}")
            continue
        if relative in listed:
            errors.append(f"manifest contains duplicate path: {relative}")
            continue
        listed.add(relative)
        path = bundle_dir / relative_path
        if not path.is_file() or path.is_symlink():
            errors.append(f"manifest file missing or symlinked: {relative}")
            continue
        if int(entry.get("bytes", -1)) != path.stat().st_size:
            errors.append(f"manifest byte count mismatch: {relative}")
        if entry.get("sha256") != sha256_file(path):
            errors.append(f"manifest sha256 mismatch: {relative}")
        if not entry.get("schema_version"):
            errors.append(f"manifest schema_version missing: {relative}")
    actual = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != "observatory_manifest.json"
    }
    if listed != actual:
        errors.append(
            "manifest file set mismatch: "
            f"missing={sorted(actual - listed)}, stale={sorted(listed - actual)}"
        )
    missing_required = sorted(REQUIRED_FILES - listed)
    if missing_required:
        errors.append(f"bundle missing required files: {missing_required}")


def _verify_relations(
    tables: dict[str, Any],
    manifest: dict[str, Any],
    expected: dict[str, int],
    errors: list[str],
) -> None:
    rows = tables["rows"].to_pylist()
    interventions = tables["interventions"].to_pylist()
    metrics = tables["metrics"].to_pylist()
    behavior = tables["behavior"].to_pylist()
    trajectories = tables["trajectories"].to_pylist()
    explanations = tables["explanations"].to_pylist()
    vectors = tables["vectors"].to_pylist()
    geometry = tables["geometry"].to_pylist()

    row_ids = {str(row["row_id"]) for row in rows}
    cell_ids = {str(row["cell_id"]) for row in interventions}
    explanation_refs = {str(row["ref"]) for row in explanations}
    vector_refs = {str(row["ref"]) for row in vectors}

    if len(rows) != expected["rows"]:
        errors.append(f"rows.parquet has {len(rows)} rows, expected {expected['rows']}")
    if len(interventions) != expected["interventions"]:
        errors.append(
            f"interventions.parquet has {len(interventions)} rows, expected {expected['interventions']}"
        )
    if len(behavior) != expected["behavior"]:
        errors.append(f"behavior.parquet has {len(behavior)} rows, expected {expected['behavior']}")
    if len(trajectories) != expected["trajectories"]:
        errors.append(
            f"token_trajectories.parquet has {len(trajectories)} rows, "
            f"expected {expected['trajectories']}"
        )

    if len(row_ids) != len(rows) or _duplicates(row["row_index"] for row in rows):
        errors.append("rows.parquet row IDs or row indices are not unique")
    if any(row.get("population") != "QUALIFIED" or row.get("split") != "validation" for row in rows):
        errors.append("rows.parquet is not exclusively QUALIFIED validation data")
    if any(row.get("source_text_release_status") != "privacy_cleared_panel" for row in rows):
        errors.append("rows.parquet contains source text outside the privacy-cleared panel")
    if any(row.get("claim_scope") != "stored_snapshot" for row in rows):
        errors.append("rows.parquet contains an invalid claim scope")

    if len(cell_ids) != len(interventions):
        errors.append("interventions.parquet contains duplicate cell IDs")
    if any(row.get("state") != "ready" or not str(row.get("text") or "").strip() for row in interventions):
        errors.append("interventions.parquet contains unresolved or empty interventions")
    if any(str(row["row_id"]) not in row_ids for row in interventions):
        errors.append("interventions.parquet references unknown rows")
    _verify_control_groups(interventions, errors)

    metric_keys = [(str(row["critic"]), str(row["cell_id"])) for row in metrics]
    if _duplicates(metric_keys):
        errors.append("metrics.parquet contains duplicate critic/cell pairs")
    if any(str(row["cell_id"]) not in cell_ids or str(row["row_id"]) not in row_ids for row in metrics):
        errors.append("metrics.parquet contains unknown row or cell references")
    primary_cells = {str(row["cell_id"]) for row in metrics if row["critic"] == "primary"}
    if primary_cells != cell_ids:
        errors.append("primary critic metrics do not cover the complete intervention lattice")
    if set(str(row["critic"]) for row in metrics) != {"primary", "independent"}:
        errors.append("metrics.parquet must contain primary and independent critics")

    if _duplicates(row["cell_id"] for row in behavior):
        errors.append("behavior.parquet contains duplicate cells")
    if any(str(row["cell_id"]) not in cell_ids or str(row["row_id"]) not in row_ids for row in behavior):
        errors.append("behavior.parquet contains unknown row or cell references")
    if any(len(row.get("baseline_continuation_token_ids") or []) != 32 for row in behavior):
        errors.append("behavior.parquet baseline continuations are not exactly 32 tokens")
    if any(len(row.get("patched_continuation_token_ids") or []) != 32 for row in behavior):
        errors.append("behavior.parquet patched continuations are not exactly 32 tokens")

    trajectory_keys = [(str(row["row_id"]), int(row["position"])) for row in trajectories]
    if _duplicates(trajectory_keys):
        errors.append("token_trajectories.parquet contains duplicate row/position pairs")
    if any(str(row["row_id"]) not in row_ids for row in trajectories):
        errors.append("token_trajectories.parquet references unknown rows")
    if any(str(row["description_ref"]) not in explanation_refs for row in trajectories):
        errors.append("token trajectories reference missing descriptions")
    if not all(bool(row["description_usable"]) for row in trajectories):
        errors.append("token trajectories include unusable descriptions")

    if len(explanation_refs) != len(explanations):
        errors.append("explanations.parquet contains duplicate refs")
    if any(not str(row.get("text") or "").strip() for row in explanations):
        errors.append("explanations.parquet contains empty text")
    if any(str(row["row_id"]) not in row_ids for row in explanations):
        errors.append("explanations.parquet references unknown rows")

    target_refs = {f"target:{row_id}" for row_id in row_ids}
    prediction_refs = {f"prediction:{critic}:{cell_id}" for critic, cell_id in metric_keys}
    trace_refs = {str(row["ref"]) for row in trajectories}
    if vector_refs != target_refs | prediction_refs | trace_refs:
        errors.append("vector index refs do not exactly cover targets, predictions, and traces")
    geometry_refs = {str(row["ref"]) for row in geometry}
    if geometry_refs != target_refs | prediction_refs:
        errors.append("geometry refs do not exactly cover targets and predictions")

    counts = manifest.get("counts") or {}
    observed_counts = {
        "rows": len(rows),
        "interventions": len(interventions),
        "behavior": len(behavior),
        "trajectories": len(trajectories),
        "vectors": len(vectors),
    }
    if counts != observed_counts:
        errors.append(f"manifest counts mismatch: observed={observed_counts}, manifest={counts}")


def run(config_path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    bundle_config = load_bundle_config(config_path)
    source_config = load_config(bundle_config["source_config"])
    source_hash = config_fingerprint(source_config)
    bundle_hash = bundle_config_fingerprint(bundle_config)
    paths = bundle_config["paths"]
    bundle_dir = bundle_path(paths["bundle_dir"], config_path=config_path)
    derived_dir = bundle_path(paths["derived_dir"], config_path=config_path)
    report_path = derived_dir / "bundle_verify_report.json"
    errors: list[str] = []

    manifest_path = bundle_dir / "observatory_manifest.json"
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, ObservatoryConfigError) as exc:
        manifest = {}
        errors.append(f"cannot read manifest: {exc}")

    if manifest:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            errors.append("manifest schema version mismatch")
        if manifest.get("source_config_sha256") != source_hash:
            errors.append("manifest source config hash mismatch")
        if manifest.get("bundle_config_sha256") != bundle_hash:
            errors.append("manifest bundle config hash mismatch")
        if manifest.get("population") != "QUALIFIED" or manifest.get("split") != "validation":
            errors.append("manifest population/split scope mismatch")
        payload = {key: value for key, value in manifest.items() if key != "bundle_id"}
        if manifest.get("bundle_id") != manifest_bundle_id(payload):
            errors.append("manifest bundle ID mismatch")
        _verify_manifest_files(bundle_dir, manifest, errors)

    provenance_path = bundle_dir / "provenance.json"
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        if provenance.get("source_config_sha256") != source_hash:
            errors.append("provenance source config hash mismatch")
        if provenance.get("bundle_config_sha256") != bundle_hash:
            errors.append("provenance bundle config hash mismatch")
        privacy_card = provenance.get("privacy_card") or {}
        if not privacy_card.get("automatic_gate_passed"):
            errors.append("inherited source-text privacy gate did not pass")
        if not provenance.get("code_bindings") or not provenance.get("runtime"):
            errors.append("provenance lacks code or runtime bindings")
    if (bundle_dir / "assets" / "source_config.yaml").is_file() and sha256_file(
        bundle_dir / "assets" / "source_config.yaml"
    ) != sha256_file(Path(bundle_config["source_config"])):
        errors.append("bundled source config differs from the evaluated source config")
    if (bundle_dir / "assets" / "bundle_config.yaml").is_file() and sha256_file(
        bundle_dir / "assets" / "bundle_config.yaml"
    ) != sha256_file(config_path):
        errors.append("bundled bundle config differs from the active bundle config")

    parquet_paths = {
        "rows": "rows.parquet",
        "interventions": "interventions.parquet",
        "metrics": "metrics.parquet",
        "behavior": "behavior.parquet",
        "trajectories": "token_trajectories.parquet",
        "explanations": "explanations.parquet",
        "vectors": "vector_index.parquet",
        "geometry": "geometry.parquet",
        "retrieval": "retrieval.parquet",
        "shapley": "shapley.parquet",
        "court": "court.parquet",
        "aggregates": "aggregates.parquet",
    }
    tables: dict[str, Any] = {}
    for label, relative in parquet_paths.items():
        try:
            tables[label] = pq.read_table(bundle_dir / relative)
        except (OSError, ValueError) as exc:
            errors.append(f"cannot read {relative}: {exc}")

    required_columns = {
        "rows": {"row_id", "row_index", "population", "split", "content_family_id"},
        "interventions": {"cell_id", "row_id", "family", "variant", "state", "text"},
        "metrics": {"cell_id", "row_id", "critic", "directional_mse", "raw_mse", "cosine", "norm_ratio"},
        "behavior": {"cell_id", "row_id", "metrics_json", "wake_json", "generation_protocol_json"},
        "trajectories": {"ref", "row_id", "position", "description_ref", "description_usable"},
        "explanations": {"ref", "row_id", "kind", "text", "parse_state"},
        "vectors": {"ref", "kind", "offset_elements", "length_elements", "dtype"},
        "geometry": {"ref", "kind", "row_id", "x", "y", "z", "native_norm"},
    }
    for label, columns in required_columns.items():
        if label in tables:
            _require_columns(tables[label], columns, label, errors)
    if set(required_columns) <= set(tables) and manifest:
        verification = bundle_config["verification"]
        expected = {
            "rows": int(verification["require_exact_rows"]),
            "interventions": int(verification["require_exact_lattice_cells"]),
            "behavior": int(verification["require_exact_behavior_cells"]),
            "trajectories": int(verification["require_exact_trace_rows"]),
        }
        _verify_relations(tables, manifest, expected, errors)

    for label, columns in {
        "metrics": ["directional_mse", "raw_mse", "cosine", "norm_ratio"],
        "geometry": ["x", "y", "z", "native_norm"],
        "retrieval": ["rank", "expected_cosine"],
        "shapley": ["shapley_value", "efficiency_error"],
        "court": ["directional_mse", "cosine", "identity_cosine"],
        "aggregates": ["mean", "ci_low", "ci_high"],
    }.items():
        if label in tables:
            _finite(tables[label], columns, label, errors)

    if "metrics" in tables and "aggregates" in tables:
        _verify_aggregates(
            tables["metrics"].to_pylist(),
            tables["aggregates"].to_pylist(),
            bundle_config,
            errors,
        )

    sensitive_findings = _verify_sensitive_text(tables, errors)

    basis_path = bundle_dir / "geometry_basis.npz"
    if basis_path.is_file():
        try:
            with np.load(basis_path) as basis:
                mean = np.asarray(basis["mean"])
                components = np.asarray(basis["basis"])
                variance = np.asarray(basis["explained_variance_ratio"])
            if mean.shape != (2688,):
                errors.append(f"geometry basis mean has invalid shape: {mean.shape}")
            expected_components = int(bundle_config["geometry"]["components"])
            if components.shape != (expected_components, 2688):
                errors.append(f"geometry basis has invalid shape: {components.shape}")
            if variance.shape != (expected_components,):
                errors.append(f"geometry variance has invalid shape: {variance.shape}")
            if not all(np.isfinite(value).all() for value in (mean, components, variance)):
                errors.append("geometry basis contains non-finite values")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"cannot validate geometry basis: {exc}")

    if "vectors" in tables:
        vector_rows = tables["vectors"].to_pylist()
        binary_path = bundle_dir / "vectors" / "all.f16.bin"
        binary_bytes = binary_path.stat().st_size if binary_path.is_file() else -1
        errors.extend(validate_vector_layout(vector_rows, binary_bytes))
        if binary_bytes >= 0 and vector_rows:
            vector_data = np.memmap(binary_path, mode="r", dtype="<f2")
            checks = min(int(bundle_config["verification"]["vector_spot_checks"]), len(vector_rows))
            indices = np.linspace(0, len(vector_rows) - 1, checks, dtype=np.int64)
            for index in indices:
                row = vector_rows[int(index)]
                start = int(row["offset_elements"])
                stop = start + int(row["length_elements"])
                if stop > vector_data.size or not np.isfinite(vector_data[start:stop].astype(np.float32)).all():
                    errors.append(f"vector spot check failed: {row['ref']}")
                    break
            if "metrics" in tables:
                _verify_vector_metrics(
                    vector_data=vector_data,
                    vector_rows=vector_rows,
                    metrics=tables["metrics"].to_pylist(),
                    checks=checks,
                    rtol=float(bundle_config["verification"]["vector_metric_rtol"]),
                    atol=float(bundle_config["verification"]["vector_metric_atol"]),
                    errors=errors,
                )

    report = {
        "schema_version": "nano_viz_bundle_verifier.v1",
        "passed": not errors,
        "source_config_sha256": source_hash,
        "bundle_config_sha256": bundle_hash,
        "bundle_id": manifest.get("bundle_id"),
        "bundle_dir": str(bundle_dir),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "sensitive_text_findings": sensitive_findings,
        "errors": errors,
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
