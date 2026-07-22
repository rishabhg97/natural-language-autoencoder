#!/usr/bin/env python3
"""Build an immutable family-disjoint round-trip development parquet.

The builder removes every content family observed in one or more sealed
round-trip reports.  It is intentionally dataset-agnostic so later layers or
protocols can create an HPO-only evaluation pool without changing evaluator
selection behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import copy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SCHEMA_VERSION = "nano_roundtrip_family_holdout.v1"
BOUNDARY_SCHEMA_VERSION = "nano_roundtrip_family_boundary.v1"
DATASET_SIDECAR_SUFFIX = ".nla_meta.yaml"


class RoundtripFamilyHoldoutError(ValueError):
    """Raised when a requested family boundary is incomplete or ambiguous."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve(value: str | Path, *, config_path: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else config_path.parent / path


def _sidecar_path_for_parquet(path: Path) -> Path:
    return Path(f"{path}{DATASET_SIDECAR_SUFFIX}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RoundtripFamilyHoldoutError(f"{label} must contain a JSON object: {path}")
    return value


def _read_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise RoundtripFamilyHoldoutError(f"{label} must contain a YAML mapping: {path}")
    return value


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"holdout config is missing: {config_path}")
    value = yaml.safe_load(config_path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise RoundtripFamilyHoldoutError(
            f"config must use schema_version {SCHEMA_VERSION}: {config_path}"
        )
    inputs = value.get("inputs")
    outputs = value.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise RoundtripFamilyHoldoutError("config requires inputs and outputs mappings")
    required_inputs = (
        "source_validation_parquet",
        "content_family_manifest",
        "exclusion_reports",
    )
    required_outputs = ("validation_parquet", "boundary_json", "report_json")
    missing = [name for name in required_inputs if not inputs.get(name)]
    missing.extend(f"outputs.{name}" for name in required_outputs if not outputs.get(name))
    if missing:
        raise RoundtripFamilyHoldoutError(f"config is missing required values: {missing}")
    if not isinstance(inputs["exclusion_reports"], list) or not inputs["exclusion_reports"]:
        raise RoundtripFamilyHoldoutError("inputs.exclusion_reports must be a non-empty list")
    if not str(value.get("role") or "").strip():
        raise RoundtripFamilyHoldoutError("config requires a non-empty role")
    return config_path, value


def _load_doc_assignments(path: Path) -> dict[str, str]:
    manifest = _read_json(path, label="content family manifest")
    assignments = manifest.get("doc_assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise RoundtripFamilyHoldoutError(
            f"content family manifest has no doc_assignments: {path}"
        )
    normalized = {
        str(doc_id): str(family_id)
        for doc_id, family_id in assignments.items()
        if str(doc_id).strip() and str(family_id).strip()
    }
    if not normalized:
        raise RoundtripFamilyHoldoutError(
            f"content family manifest has no usable doc assignments: {path}"
        )
    return normalized


def _load_exclusion_source(
    raw: Any,
    *,
    config_path: Path,
) -> tuple[set[str], dict[str, Any]]:
    spec = raw if isinstance(raw, dict) else {"report_json": raw}
    report_value = spec.get("report_json") or spec.get("path")
    if not report_value:
        raise RoundtripFamilyHoldoutError("each exclusion report requires report_json")
    split = str(spec.get("split") or "validation")
    report_path = _resolve(report_value, config_path=config_path).resolve()
    report = _read_json(report_path, label="exclusion report")
    splits = report.get("splits")
    split_payload = splits.get(split) if isinstance(splits, dict) else None
    if not isinstance(split_payload, dict):
        raise RoundtripFamilyHoldoutError(
            f"exclusion report has no splits.{split}: {report_path}"
        )
    raw_family_ids = split_payload.get("content_family_ids")
    if not isinstance(raw_family_ids, list) or not raw_family_ids:
        raise RoundtripFamilyHoldoutError(
            f"exclusion report has no content family ids for {split}: {report_path}"
        )
    family_ids = {str(value).strip() for value in raw_family_ids if str(value).strip()}
    if not family_ids:
        raise RoundtripFamilyHoldoutError(
            f"exclusion report has no usable content family ids for {split}: {report_path}"
        )
    metadata = {
        "report_json": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "split": split,
        "row_count": len(raw_family_ids),
        "family_count": len(family_ids),
        "generation_protocol_sha256": report.get("generation_protocol_sha256"),
    }
    return family_ids, metadata


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))
    temporary.replace(path)


def _derive_dataset_sidecar(
    source_meta: dict[str, Any],
    *,
    source_path: Path,
    source_meta_path: Path,
    source_parquet_sha256: str,
    source_meta_sha256: str,
    output_path: Path,
    output_row_count: int,
    boundary_path: Path,
    excluded_family_count: int,
    role: str,
) -> dict[str, Any]:
    if source_meta.get("kind") != "nla_dataset":
        raise RoundtripFamilyHoldoutError(
            f"source validation sidecar must have kind nla_dataset: {source_meta_path}"
        )
    if not isinstance(source_meta.get("extraction"), dict):
        raise RoundtripFamilyHoldoutError(
            f"source validation sidecar has no extraction mapping: {source_meta_path}"
        )
    if not isinstance(source_meta.get("tokens"), dict):
        raise RoundtripFamilyHoldoutError(
            f"source validation sidecar has no tokens mapping: {source_meta_path}"
        )

    derived = copy.deepcopy(source_meta)
    derived["dataset_id"] = output_path.stem
    derived["row_count"] = int(output_row_count)
    derived["created_by"] = "scripts.build_nano_roundtrip_family_holdout"
    lineage = copy.deepcopy(derived.get("lineage") or {})
    lineage.update(
        {
            "source_validation_parquet": str(source_path),
            "source_validation_parquet_sha256": source_parquet_sha256,
            "source_validation_nla_meta_yaml": str(source_meta_path),
            "source_validation_nla_meta_yaml_sha256": source_meta_sha256,
            "family_holdout_boundary_json": str(boundary_path),
            "family_holdout_excluded_content_family_count": int(excluded_family_count),
            "family_holdout_role": role,
        }
    )
    derived["lineage"] = lineage
    return derived


def _existing_report_if_exact(
    *,
    report_path: Path,
    boundary_path: Path,
    output_path: Path,
    output_meta_path: Path,
    identity: dict[str, Any],
) -> dict[str, Any] | None:
    existing_paths = [
        path for path in (report_path, boundary_path, output_path, output_meta_path) if path.exists()
    ]
    if not existing_paths:
        return None
    if not all(path.is_file() for path in (report_path, boundary_path, output_path, output_meta_path)):
        raise RoundtripFamilyHoldoutError(
            "holdout outputs are partially present; preserve them and choose new output paths"
        )
    report = _read_json(report_path, label="existing holdout report")
    if report.get("identity") != identity:
        raise RoundtripFamilyHoldoutError(
            "holdout outputs already exist with a different immutable input identity"
        )
    if report.get("output_validation_parquet_sha256") != _sha256_file(output_path):
        raise RoundtripFamilyHoldoutError("existing holdout parquet checksum mismatch")
    if report.get("boundary_sha256") != _sha256_file(boundary_path):
        raise RoundtripFamilyHoldoutError("existing holdout boundary checksum mismatch")
    if report.get("output_validation_nla_meta_yaml_sha256") != _sha256_file(output_meta_path):
        raise RoundtripFamilyHoldoutError("existing holdout sidecar checksum mismatch")
    if not report.get("passed"):
        raise RoundtripFamilyHoldoutError("existing holdout report is not passing")
    return report


def run_build(config: str | Path) -> dict[str, Any]:
    config_path, value = _load_config(config)
    inputs = value["inputs"]
    outputs = value["outputs"]
    source_path = _resolve(inputs["source_validation_parquet"], config_path=config_path).resolve()
    source_meta_path = _resolve(
        inputs.get("source_validation_nla_meta_yaml") or _sidecar_path_for_parquet(source_path),
        config_path=config_path,
    ).resolve()
    manifest_path = _resolve(inputs["content_family_manifest"], config_path=config_path).resolve()
    output_path = _resolve(outputs["validation_parquet"], config_path=config_path).resolve()
    output_meta_path = _resolve(
        outputs.get("validation_nla_meta_yaml") or _sidecar_path_for_parquet(output_path),
        config_path=config_path,
    ).resolve()
    boundary_path = _resolve(outputs["boundary_json"], config_path=config_path).resolve()
    report_path = _resolve(outputs["report_json"], config_path=config_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source validation parquet is missing: {source_path}")
    source_meta = _read_yaml_mapping(source_meta_path, label="source validation sidecar")

    assignments = _load_doc_assignments(manifest_path)
    excluded_families: set[str] = set()
    exclusion_sources: list[dict[str, Any]] = []
    for raw_source in inputs["exclusion_reports"]:
        source_families, source_metadata = _load_exclusion_source(
            raw_source,
            config_path=config_path,
        )
        excluded_families.update(source_families)
        exclusion_sources.append(source_metadata)
    if not excluded_families:
        raise RoundtripFamilyHoldoutError("family boundary would exclude no families")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "role": str(value["role"]),
        "config_sha256": _sha256_file(config_path),
        "source_validation_parquet": str(source_path),
        "source_validation_parquet_sha256": _sha256_file(source_path),
        "source_validation_nla_meta_yaml": str(source_meta_path),
        "source_validation_nla_meta_yaml_sha256": _sha256_file(source_meta_path),
        "content_family_manifest": str(manifest_path),
        "content_family_manifest_sha256": _sha256_file(manifest_path),
        "exclusion_sources": exclusion_sources,
        "excluded_content_family_ids_sha256": _canonical_sha256(
            sorted(excluded_families)
        ),
    }
    existing = _existing_report_if_exact(
        report_path=report_path,
        boundary_path=boundary_path,
        output_path=output_path,
        output_meta_path=output_meta_path,
        identity=identity,
    )
    if existing is not None:
        return existing

    table = pq.read_table(source_path)
    if "doc_id" not in table.column_names:
        raise RoundtripFamilyHoldoutError(
            f"source validation parquet has no doc_id column: {source_path}"
        )
    doc_ids = table.column("doc_id").to_pylist()
    source_family_ids: list[str] = []
    missing_doc_ids: list[str] = []
    for raw_doc_id in doc_ids:
        doc_id = str(raw_doc_id or "").strip()
        family_id = assignments.get(doc_id)
        if not doc_id or family_id is None:
            missing_doc_ids.append(doc_id or "<empty>")
        else:
            source_family_ids.append(family_id)
    if missing_doc_ids:
        raise RoundtripFamilyHoldoutError(
            "source validation rows are not fully covered by the content family manifest: "
            f"count={len(missing_doc_ids)} sample={missing_doc_ids[:5]}"
        )
    keep_mask = [family_id not in excluded_families for family_id in source_family_ids]
    filtered = table.filter(pa.array(keep_mask, type=pa.bool_()))
    output_family_ids = [
        family_id for family_id, keep in zip(source_family_ids, keep_mask) if keep
    ]
    overlap = sorted(set(output_family_ids) & excluded_families)
    if overlap:
        raise AssertionError(f"family filter retained excluded families: {overlap[:5]}")
    if not filtered.num_rows or not output_family_ids:
        raise RoundtripFamilyHoldoutError("family boundary removed every validation row")

    boundary = {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "role": str(value["role"]),
        "source_validation_parquet": str(source_path),
        "source_validation_parquet_sha256": identity["source_validation_parquet_sha256"],
        "content_family_manifest": str(manifest_path),
        "content_family_manifest_sha256": identity["content_family_manifest_sha256"],
        "exclusion_sources": exclusion_sources,
        "excluded_content_family_ids": sorted(excluded_families),
        "excluded_content_family_ids_sha256": identity[
            "excluded_content_family_ids_sha256"
        ],
        "output_validation_row_count": int(filtered.num_rows),
        "output_content_family_count": len(set(output_family_ids)),
        "output_validation_nla_meta_yaml": str(output_meta_path),
    }
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    pq.write_table(filtered, temporary_output, compression="zstd")
    temporary_output.replace(output_path)
    output_meta = _derive_dataset_sidecar(
        source_meta,
        source_path=source_path,
        source_meta_path=source_meta_path,
        source_parquet_sha256=identity["source_validation_parquet_sha256"],
        source_meta_sha256=identity["source_validation_nla_meta_yaml_sha256"],
        output_path=output_path,
        output_row_count=int(filtered.num_rows),
        boundary_path=boundary_path,
        excluded_family_count=len(excluded_families),
        role=str(value["role"]),
    )
    _write_yaml_atomic(output_meta_path, output_meta)
    _write_json_atomic(boundary_path, boundary)

    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "identity": identity,
        "output_validation_parquet": str(output_path),
        "output_validation_parquet_sha256": _sha256_file(output_path),
        "output_validation_nla_meta_yaml": str(output_meta_path),
        "output_validation_nla_meta_yaml_sha256": _sha256_file(output_meta_path),
        "boundary_json": str(boundary_path),
        "boundary_sha256": _sha256_file(boundary_path),
        "source_validation_row_count": int(table.num_rows),
        "source_content_family_count": len(set(source_family_ids)),
        "excluded_content_family_count": len(excluded_families),
        "excluded_validation_row_count": int(table.num_rows - filtered.num_rows),
        "output_validation_row_count": int(filtered.num_rows),
        "output_content_family_count": len(set(output_family_ids)),
        "output_excluded_family_overlap_count": len(overlap),
    }
    _write_json_atomic(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_build(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
