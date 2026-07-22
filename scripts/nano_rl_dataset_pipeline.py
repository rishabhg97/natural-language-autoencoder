#!/usr/bin/env python3
"""Build or reuse a verified family-clean Nano RL dataset from one YAML config."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_nano_r33_rl_dataset import build_dataset, sha256_file  # noqa: E402
from verify_nano_r33_rl_dataset import verify_dataset  # noqa: E402


SCHEMA_VERSION = "nano_rl_dataset_pipeline.v1"


class RLDatasetPipelineError(ValueError):
    """Raised when a configured RL dataset cannot be safely produced."""


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _required_path(paths: dict[str, Any], name: str) -> Path:
    value = paths.get(name)
    if value is None or not str(value).strip():
        raise RLDatasetPipelineError(f"paths.{name} is required")
    return Path(str(value))


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise RLDatasetPipelineError("pipeline config must be a mapping")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RLDatasetPipelineError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    return config


def run_pipeline(
    config_path: str | Path,
    *,
    overwrite: bool | None = None,
    reuse_verified: bool | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    paths = config.get("paths") or {}
    expectations = config.get("expectations") or {}
    runtime = config.get("runtime") or {}
    if not isinstance(paths, dict) or not isinstance(expectations, dict):
        raise RLDatasetPipelineError("paths and expectations must be mappings")
    if not isinstance(runtime, dict):
        raise RLDatasetPipelineError("runtime must be a mapping")

    base_parquet = _required_path(paths, "base_parquet")
    actor_sidecar_source = _required_path(paths, "actor_sidecar_source")
    split_manifest = _required_path(paths, "split_manifest")
    family_manifest = _required_path(paths, "content_family_manifest")
    family_coverage = _required_path(paths, "content_family_coverage")
    output = _required_path(paths, "output")
    build_report_path = _required_path(paths, "build_report_json")
    verify_report_path = _required_path(paths, "verify_report_json")
    pipeline_report_path = _required_path(paths, "pipeline_report_json")

    expected_rows = int(expectations.get("rows", 0))
    expected_d_model = int(expectations.get("d_model", 0))
    expected_layer = int(expectations.get("activation_layer", -1))
    batch_size = int(runtime.get("batch_size", 4_096))
    if expected_rows <= 0:
        raise RLDatasetPipelineError("expectations.rows must be positive")
    if expected_d_model <= 0:
        raise RLDatasetPipelineError("expectations.d_model must be positive")
    if expected_layer < 0:
        raise RLDatasetPipelineError(
            "expectations.activation_layer must be non-negative"
        )
    if batch_size <= 0:
        raise RLDatasetPipelineError("runtime.batch_size must be positive")

    overwrite_value = bool(runtime.get("overwrite", False)) if overwrite is None else overwrite
    reuse_value = (
        bool(runtime.get("reuse_verified", True))
        if reuse_verified is None
        else reuse_verified
    )

    verification: dict[str, Any] | None = None
    action = "built"
    if output.is_file():
        verification = verify_dataset(
            dataset=output,
            split_manifest=split_manifest,
            content_family_manifest=family_manifest,
            content_family_coverage=family_coverage,
            expected_rows=expected_rows,
            expected_d_model=expected_d_model,
            batch_size=batch_size,
        )
        _write_json_atomic(verify_report_path, verification)
        if verification["passed"] and reuse_value:
            action = "reused_verified"
        elif not overwrite_value:
            reason = "reuse is disabled" if verification["passed"] else (
                "existing output failed verification: "
                + ", ".join(verification["blockers"])
            )
            raise RLDatasetPipelineError(
                f"{output} exists but {reason}; enable overwrite explicitly"
            )

    build_report: dict[str, Any] | None = None
    if action != "reused_verified":
        build_report = build_dataset(
            base_parquet=base_parquet,
            actor_sidecar_source=actor_sidecar_source,
            split_manifest=split_manifest,
            output=output,
            report_json=build_report_path,
            content_family_manifest=family_manifest,
            content_family_coverage=family_coverage,
            expected_rows=expected_rows,
            expected_layer=expected_layer,
            batch_size=batch_size,
            overwrite=overwrite_value,
        )
        verification = verify_dataset(
            dataset=output,
            split_manifest=split_manifest,
            content_family_manifest=family_manifest,
            content_family_coverage=family_coverage,
            expected_rows=expected_rows,
            expected_d_model=expected_d_model,
            batch_size=batch_size,
        )
        _write_json_atomic(verify_report_path, verification)

    assert verification is not None
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": bool(verification["passed"]),
        "action": action,
        "created_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "output": str(output),
        "output_sha256": verification.get("dataset_sha256"),
        "rows": verification.get("rows"),
        "d_model_counts": verification.get("d_model_counts"),
        "build_report": None if build_report is None else str(build_report_path),
        "verify_report": str(verify_report_path),
        "blockers": list(verification.get("blockers") or []),
    }
    _write_json_atomic(pipeline_report_path, report)
    if not report["passed"]:
        raise RLDatasetPipelineError(
            "newly built RL dataset failed verification: "
            + ", ".join(report["blockers"])
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--no-reuse", action="store_false", dest="reuse_verified", default=None)
    args = parser.parse_args()
    try:
        report = run_pipeline(
            args.config,
            overwrite=args.overwrite,
            reuse_verified=args.reuse_verified,
        )
    except (RLDatasetPipelineError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
