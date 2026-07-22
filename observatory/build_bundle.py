#!/usr/bin/env python3
"""Assemble the verified static Observatory bundle from qualified evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
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
    canonical_json,
    config_fingerprint,
    load_config,
    read_jsonl,
    sha256_file,
    write_json,
)


SCHEMA_VERSION = "nano_viz_bundle.v1"
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent


def manifest_bundle_id(manifest_without_id: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(manifest_without_id).encode("utf-8")).hexdigest()


def _require_report(
    path: Path,
    source_hash: str,
    *,
    bundle_hash: str | None = None,
) -> dict[str, Any]:
    report = read_json(path)
    if not report.get("passed"):
        raise ObservatoryConfigError(f"required report did not pass: {path}")
    observed = report.get("config_sha256") or report.get("source_config_sha256")
    if observed != source_hash:
        raise ObservatoryConfigError(f"report config hash mismatch: {path}")
    observed_bundle_hash = report.get("bundle_config_sha256")
    if bundle_hash is not None and observed_bundle_hash != bundle_hash:
        raise ObservatoryConfigError(f"report bundle config hash mismatch: {path}")
    return report


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _require_bound_artifact(binding: Any, label: str) -> Path:
    if not isinstance(binding, dict) or not binding.get("path") or not binding.get("sha256"):
        raise ObservatoryConfigError(f"artifact binding is incomplete: {label}")
    path = Path(str(binding["path"]))
    if not path.is_file():
        raise ObservatoryConfigError(f"bound artifact is missing: {label}: {path}")
    if sha256_file(path) != binding["sha256"]:
        raise ObservatoryConfigError(f"bound artifact hash mismatch: {label}: {path}")
    return path


def run(config_path: Path) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    bundle_config = load_bundle_config(config_path)
    source_config = load_config(bundle_config["source_config"])
    source_hash = config_fingerprint(source_config)
    bundle_hash = bundle_config_fingerprint(bundle_config)
    paths = bundle_config["paths"]
    corpus_dir = bundle_path(paths["corpus_dir"], config_path=config_path)
    model_outputs = bundle_path(paths["model_outputs_dir"], config_path=config_path)
    derived_dir = bundle_path(paths["derived_dir"], config_path=config_path)
    bundle_dir = bundle_path(paths["bundle_dir"], config_path=config_path)
    staging = bundle_dir.with_name(bundle_dir.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    report_paths = {
        "e0": model_outputs / "e0" / "qualification_report.json",
        "e1_av": model_outputs / "e1_canary" / "canary_av_report.json",
        "e1_ar": model_outputs / "e1_canary" / "canary_ar_report.json",
        "e2": model_outputs / "e2_token_logprobs" / "token_logprobs_report.json",
        "e3": model_outputs / "e3_lattice_pilot" / "lattice_pilot_report.json",
        "p1_tellings": model_outputs / "p1_alternate_tellings" / "alternate_tellings_report.json",
        "p2": model_outputs / "p2_lattice_full" / "lattice_full_report.json",
        "e4": model_outputs / "e4_functional_pilot" / "functional_pilot_report.json",
        "p3": model_outputs / "p3_functional_full" / "functional_full_report.json",
        "e5": model_outputs / "e5_trace_pilot" / "trace_extract_report.json",
        "p1_trace": model_outputs / "p1_trace_descriptions" / "trace_descriptions_report.json",
        "geometry": derived_dir / "geometry_report.json",
        "interventions": derived_dir / "intervention_report.json",
    }
    reports = {
        name: _require_report(
            path,
            source_hash,
            bundle_hash=bundle_hash if name in {"geometry", "interventions"} else None,
        )
        for name, path in report_paths.items()
    }
    alternate_path = _require_bound_artifact(
        reports["p1_tellings"].get("records"), "p1_tellings.records"
    )
    behavior_path = _require_bound_artifact(
        reports["p3"].get("functional_rows"), "p3.functional_rows"
    )
    trace_path = _require_bound_artifact(reports["e5"].get("trajectories"), "e5.trajectories")
    trace_descriptions_path = _require_bound_artifact(
        reports["p1_trace"].get("records"), "p1_trace.records"
    )
    p2_shard_paths = [
        _require_bound_artifact(shard, f"p2.shards[{index}]")
        for index, shard in enumerate(reports["p2"].get("shards") or [])
    ]
    if len(p2_shard_paths) != len(reports["p2"].get("shards") or []):
        raise ObservatoryConfigError("P2 shard binding count mismatch")
    for report_name in ("geometry", "interventions"):
        for artifact_name, binding in (reports[report_name].get("artifacts") or {}).items():
            _require_bound_artifact(binding, f"{report_name}.{artifact_name}")

    source_rows = read_jsonl(corpus_dir / "rows.jsonl")
    if len(source_rows) != 50 or any(row["population"] != "QUALIFIED" for row in source_rows):
        raise ObservatoryConfigError("bundle rows must be the 50 QUALIFIED validation rows")
    rows_schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("row_index", pa.int64()),
            ("source_row_index", pa.int64()),
            ("population", pa.string()),
            ("split", pa.string()),
            ("doc_id", pa.string()),
            ("content_family_id", pa.string()),
            ("n_raw_tokens", pa.int64()),
            ("token_position", pa.int64()),
            ("activation_norm", pa.float32()),
            ("source_text", pa.string()),
            ("target_explanation", pa.string()),
            ("av_explanation", pa.string()),
            ("source_text_release_status", pa.string()),
            ("claim_scope", pa.string()),
            ("stratum_json", pa.string()),
        ]
    )
    bundle_rows = [
        {
            **{key: row.get(key) for key in rows_schema.names if key != "stratum_json"},
            "stratum_json": json.dumps(row.get("stratum") or {}, sort_keys=True),
        }
        for row in source_rows
    ]
    write_parquet_atomic(staging / "rows.parquet", bundle_rows, rows_schema)

    interventions = read_jsonl(corpus_dir / "interventions.jsonl")
    alternate_records = {
        str(record["cell_id"]): record
        for record in read_jsonl(alternate_path)
    }
    intervention_rows: list[dict[str, Any]] = []
    for row in interventions:
        text = row.get("text")
        state = str(row["state"])
        if row["family"] == "alternate_telling":
            record = alternate_records.get(str(row["cell_id"]))
            parsed = None if record is None else record.get("parsed")
            text = None if not isinstance(parsed, dict) else parsed.get("explanation")
            state = "ready"
        if not isinstance(text, str) or not text.strip():
            raise ObservatoryConfigError(f"bundle intervention lacks text: {row['cell_id']}")
        intervention_rows.append(
            {
                "cell_id": str(row["cell_id"]),
                "control_group_id": row.get("control_group_id"),
                "row_id": str(row["row_id"]),
                "row_index": int(row["row_index"]),
                "family": str(row["family"]),
                "variant": str(row["variant"]),
                "depth": str(row["depth"]),
                "state": state,
                "text": text.strip(),
                "text_sha256": hashlib.sha256(text.strip().encode("utf-8")).hexdigest(),
                "spec_json": json.dumps(row["spec"], sort_keys=True),
            }
        )
    intervention_schema = pa.schema(
        [
            ("cell_id", pa.string()),
            ("control_group_id", pa.string()),
            ("row_id", pa.string()),
            ("row_index", pa.int64()),
            ("family", pa.string()),
            ("variant", pa.string()),
            ("depth", pa.string()),
            ("state", pa.string()),
            ("text", pa.string()),
            ("text_sha256", pa.string()),
            ("spec_json", pa.string()),
        ]
    )
    write_parquet_atomic(staging / "interventions.parquet", intervention_rows, intervention_schema)

    explanations: list[dict[str, Any]] = []
    for row in source_rows:
        for kind, text in (
            ("teacher", row["target_explanation"]),
            ("qualified_av", row["av_explanation"]),
        ):
            explanations.append(
                {
                    "ref": f"{kind}:{row['row_id']}",
                    "row_id": str(row["row_id"]),
                    "cell_id": None,
                    "position": None,
                    "kind": kind,
                    "text": str(text),
                    "parse_state": "usable_closed",
                    "token_ids_json": None,
                    "token_logprobs_json": None,
                    "protocol_sha256": None,
                }
            )
    for record in alternate_records.values():
        parsed = record["parsed"]
        explanations.append(
            {
                "ref": f"alternate:{record['cell_id']}",
                "row_id": str(record["row_id"]),
                "cell_id": str(record["cell_id"]),
                "position": None,
                "kind": "alternate_telling",
                "text": str(parsed["explanation"]),
                "parse_state": "usable_closed" if parsed.get("closed") else "usable_open",
                "token_ids_json": json.dumps(record["token_ids"]),
                "token_logprobs_json": json.dumps(record["token_logprobs"]),
                "protocol_sha256": reports["p1_tellings"]["config_sha256"],
            }
        )
    trace_descriptions = read_jsonl(trace_descriptions_path)
    for record in trace_descriptions:
        parsed = record["parsed"]
        explanations.append(
            {
                "ref": f"trace:{record['row_id']}:{int(record['position'])}",
                "row_id": str(record["row_id"]),
                "cell_id": None,
                "position": int(record["position"]),
                "kind": "trace_description",
                "text": str(parsed["explanation"]),
                "parse_state": "usable_closed" if parsed.get("closed") else "usable_open",
                "token_ids_json": None,
                "token_logprobs_json": None,
                "protocol_sha256": reports["p1_trace"]["config_sha256"],
            }
        )
    explanation_schema = pa.schema(
        [
            ("ref", pa.string()),
            ("row_id", pa.string()),
            ("cell_id", pa.string()),
            ("position", pa.int64()),
            ("kind", pa.string()),
            ("text", pa.string()),
            ("parse_state", pa.string()),
            ("token_ids_json", pa.string()),
            ("token_logprobs_json", pa.string()),
            ("protocol_sha256", pa.string()),
        ]
    )
    write_parquet_atomic(staging / "explanations.parquet", explanations, explanation_schema)

    behavior_records = read_jsonl(behavior_path)
    behavior_rows = [
        {
            "cell_id": str(row["cell_id"]),
            "row_id": f"validation-{int(row['row_index'])}",
            "row_index": int(row["row_index"]),
            "content_family_id": str(row["content_family_id"]),
            "variant": str(row["variant"]),
            "metrics_json": json.dumps(row["metrics"], sort_keys=True),
            "original_topk_json": json.dumps(row["original_topk"], sort_keys=True),
            "patched_topk_json": json.dumps(row["patched_topk"], sort_keys=True),
            "wake_json": json.dumps(row["wake"], sort_keys=True),
            "baseline_continuation_token_ids": row["baseline_continuation_token_ids"],
            "baseline_continuation_text": str(row["baseline_continuation_text"]),
            "patched_continuation_token_ids": row["patched_continuation_token_ids"],
            "patched_continuation_text": str(row["patched_continuation_text"]),
            "generation_protocol_json": json.dumps(row["generation_protocol"], sort_keys=True),
        }
        for row in behavior_records
    ]
    behavior_schema = pa.schema(
        [
            ("cell_id", pa.string()),
            ("row_id", pa.string()),
            ("row_index", pa.int64()),
            ("content_family_id", pa.string()),
            ("variant", pa.string()),
            ("metrics_json", pa.string()),
            ("original_topk_json", pa.string()),
            ("patched_topk_json", pa.string()),
            ("wake_json", pa.string()),
            ("baseline_continuation_token_ids", pa.list_(pa.int64())),
            ("baseline_continuation_text", pa.string()),
            ("patched_continuation_token_ids", pa.list_(pa.int64())),
            ("patched_continuation_text", pa.string()),
            ("generation_protocol_json", pa.string()),
        ]
    )
    write_parquet_atomic(staging / "behavior.parquet", behavior_rows, behavior_schema)

    trace_table = pq.read_table(trace_path)
    trace_rows = trace_table.to_pylist()
    description_by_key = {
        (int(row["row_index"]), int(row["position"])): row for row in trace_descriptions
    }
    trajectory_rows = [
        {
            "ref": f"trace:{row['row_id']}:{int(row['position'])}",
            "row_id": str(row["row_id"]),
            "row_index": int(row["row_index"]),
            "doc_id": str(row["doc_id"]),
            "content_family_id": str(row["content_family_id"]),
            "position": int(row["position"]),
            "n_context_tokens": int(row["n_context_tokens"]),
            "token_id": int(row["token_id"]),
            "token_text": str(row["token_text"]),
            "description_ref": f"trace:{row['row_id']}:{int(row['position'])}",
            "description_usable": bool(
                description_by_key[(int(row["row_index"]), int(row["position"]))]["parsed"]["usable"]
            ),
        }
        for row in trace_rows
    ]
    trajectory_schema = pa.schema(
        [
            ("ref", pa.string()),
            ("row_id", pa.string()),
            ("row_index", pa.int64()),
            ("doc_id", pa.string()),
            ("content_family_id", pa.string()),
            ("position", pa.int64()),
            ("n_context_tokens", pa.int64()),
            ("token_id", pa.int64()),
            ("token_text", pa.string()),
            ("description_ref", pa.string()),
            ("description_usable", pa.bool_()),
        ]
    )
    write_parquet_atomic(staging / "token_trajectories.parquet", trajectory_rows, trajectory_schema)

    for source, destination in (
        (derived_dir / "intervention_metrics.parquet", staging / "metrics.parquet"),
        (derived_dir / "geometry.parquet", staging / "geometry.parquet"),
        (derived_dir / "retrieval.parquet", staging / "retrieval.parquet"),
        (derived_dir / "shapley.parquet", staging / "shapley.parquet"),
        (derived_dir / "court.parquet", staging / "court.parquet"),
        (derived_dir / "aggregates.parquet", staging / "aggregates.parquet"),
        (derived_dir / "aggregates.json", staging / "aggregates.json"),
        (derived_dir / "geometry_basis.npz", staging / "geometry_basis.npz"),
    ):
        _copy(source, destination)

    vector_dir = staging / "vectors"
    vector_dir.mkdir()
    vector_path = vector_dir / "all.f16.bin"
    vector_index: list[dict[str, Any]] = []
    offset = 0

    def append_vector(
        handle: Any,
        *,
        ref: str,
        kind: str,
        row_id: str,
        cell_id: str | None,
        critic: str | None,
        position: int | None,
        vector: Any,
    ) -> None:
        nonlocal offset
        array = np.asarray(vector, dtype="<f2")
        if array.shape != (2688,) or not np.isfinite(array.astype(np.float32)).all():
            raise ObservatoryConfigError(f"invalid bundle vector: {ref}")
        handle.write(array.tobytes(order="C"))
        vector_index.append(
            {
                "ref": ref,
                "kind": kind,
                "row_id": row_id,
                "cell_id": cell_id,
                "critic": critic,
                "position": position,
                "offset_elements": offset,
                "length_elements": 2688,
                "dtype": "float16_le",
            }
        )
        offset += 2688

    selected_source = pq.read_table(source_config["paths"]["source_base_selected_parquet"]).to_pylist()
    with vector_path.open("wb") as handle:
        for row in selected_source:
            row_id = f"validation-{int(row['row_index'])}"
            append_vector(
                handle,
                ref=f"target:{row_id}",
                kind="target",
                row_id=row_id,
                cell_id=None,
                critic=None,
                position=None,
                vector=row["activation_vector"],
            )
        for shard, shard_path in zip(reports["p2"]["shards"], p2_shard_paths, strict=True):
            table = pq.read_table(shard_path)
            if table.num_rows != int(shard["rows"]):
                raise ObservatoryConfigError(f"P2 shard row count mismatch: {shard_path}")
            for row in table.to_pylist():
                append_vector(
                    handle,
                    ref=f"prediction:{row['critic']}:{row['cell_id']}",
                    kind="prediction",
                    row_id=str(row["row_id"]),
                    cell_id=str(row["cell_id"]),
                    critic=str(row["critic"]),
                    position=None,
                    vector=row["prediction_vector"],
                )
        for row in trace_rows:
            append_vector(
                handle,
                ref=f"trace:{row['row_id']}:{int(row['position'])}",
                kind="trace",
                row_id=str(row["row_id"]),
                cell_id=None,
                critic=None,
                position=int(row["position"]),
                vector=row["activation_vector"],
            )
    vector_index_schema = pa.schema(
        [
            ("ref", pa.string()),
            ("kind", pa.string()),
            ("row_id", pa.string()),
            ("cell_id", pa.string()),
            ("critic", pa.string()),
            ("position", pa.int64()),
            ("offset_elements", pa.int64()),
            ("length_elements", pa.int64()),
            ("dtype", pa.string()),
        ]
    )
    write_parquet_atomic(staging / "vector_index.parquet", vector_index, vector_index_schema)

    assets_reports = staging / "assets" / "reports"
    for name, path in report_paths.items():
        _copy(path, assets_reports / f"{name}.json")
    _copy(config_path, staging / "assets" / "bundle_config.yaml")
    _copy(Path(bundle_config["source_config"]), staging / "assets" / "source_config.yaml")
    claim_ledger = {
        "schema_version": SCHEMA_VERSION,
        "claims": {
            "stored_snapshot_channel": "qualified",
            "fresh_forward_trace": "exploratory",
            "functional_interventions": "validation_only_exploratory",
            "test_set": "not_opened_for_dashboard_lattice",
        },
        "limitations": reports["e0"].get("limitations", []),
    }
    write_json(staging / "assets" / "claim_ledger.json", claim_ledger)
    code_files = sorted(
        [*PACKAGE_DIR.glob("*.py")]
        + list((REPO_ROOT / "scripts").glob("nano_viz_*.py"))
        + [REPO_ROOT / "scripts" / "_observatory_entrypoint.py"]
        + list((REPO_ROOT / "configs" / "nano_viz").glob("*.yaml"))
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_config_sha256": source_hash,
        "bundle_config_sha256": bundle_hash,
        "population": "QUALIFIED",
        "split": "validation",
        "report_bindings": {
            name: {"source_path": str(path), "sha256": sha256_file(path)}
            for name, path in report_paths.items()
        },
        "source_provenance": reports["e0"].get("input_provenance", {}),
        "privacy_card": reports["e0"].get("privacy_card", {}),
        "code_bindings": {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in code_files
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    write_json(staging / "provenance.json", provenance)

    schema_by_name = {
        "rows.parquet": "nano_viz_rows.v1",
        "metrics.parquet": "nano_viz_metrics.v1",
        "explanations.parquet": "nano_viz_explanations.v1",
        "interventions.parquet": "nano_viz_interventions.v1",
        "behavior.parquet": "nano_viz_behavior.v1",
        "token_trajectories.parquet": "nano_viz_trajectories.v1",
        "geometry.parquet": "nano_viz_geometry.v1",
        "vector_index.parquet": "nano_viz_vector_index.v1",
        "vectors/all.f16.bin": "float16_le.2688.v1",
        "geometry_basis.npz": "numpy_npz_pca_basis.v1",
    }
    files = []
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative = path.relative_to(staging).as_posix()
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "schema_version": schema_by_name.get(relative, "source_bound.v1"),
            }
        )
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_config_sha256": source_hash,
        "bundle_config_sha256": bundle_hash,
        "population": "QUALIFIED",
        "split": "validation",
        "counts": {
            "rows": len(bundle_rows),
            "interventions": len(intervention_rows),
            "behavior": len(behavior_rows),
            "trajectories": len(trajectory_rows),
            "vectors": len(vector_index),
        },
        "files": files,
    }
    manifest = {**manifest_payload, "bundle_id": manifest_bundle_id(manifest_payload)}
    write_json(staging / "observatory_manifest.json", manifest)
    shutil.rmtree(bundle_dir, ignore_errors=True)
    staging.replace(bundle_dir)

    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "source_config_sha256": source_hash,
        "bundle_config_sha256": bundle_hash,
        "bundle_id": manifest["bundle_id"],
        "bundle_dir": str(bundle_dir),
        "counts": manifest["counts"],
        "bundle_bytes": sum(item["bytes"] for item in files),
        "manifest_sha256": sha256_file(bundle_dir / "observatory_manifest.json"),
    }
    write_json(derived_dir / "bundle_build_report.json", report)
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
