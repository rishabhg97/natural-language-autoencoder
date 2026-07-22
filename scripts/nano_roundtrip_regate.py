#!/usr/bin/env python3
"""Recompute a round-trip promotion gate from completed evaluation reports."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_nano_av_ar_roundtrip_gate import (  # noqa: E402
    build_gate_summary,
    validate_activation_metric_reports,
    write_json,
)


SCHEMA_VERSION = "nano_roundtrip_regate.v1"
REQUIRED_PATHS = ("candidate_report_json", "baseline_report_json", "output_json")
REQUIRED_GATE_FIELDS = (
    "control_margin",
    "baseline_margin",
    "min_control_win_fraction",
    "min_baseline_win_fraction",
    "min_baseline_relative_improvement",
    "require_baseline_ci_positive",
    "require_clustered_baseline_ci",
    "require_baseline_dataset_match",
    "bootstrap_samples",
    "bootstrap_seed",
    "permutation_samples",
    "permutation_seed",
    "min_closed_fraction",
    "min_usable_fraction",
    "require_generation_protocol_match",
    "require_family_level_inference",
    "min_independent_families",
)


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("regate config must be a mapping")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )
    paths = payload.get("paths")
    gate = payload.get("gate")
    if not isinstance(paths, dict) or not isinstance(gate, dict):
        raise ValueError("regate config requires paths and gate mappings")
    missing_paths = [name for name in REQUIRED_PATHS if not paths.get(name)]
    missing_gate = [name for name in REQUIRED_GATE_FIELDS if name not in gate]
    if missing_paths or missing_gate:
        raise ValueError(
            f"incomplete regate config: missing_paths={missing_paths} missing_gate={missing_gate}"
        )
    return payload


def regate_reports(
    candidate_report: dict[str, Any],
    baseline_report: dict[str, Any],
    *,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    candidate_splits = candidate_report.get("splits") or {}
    baseline_splits = baseline_report.get("splits") or {}
    validate_activation_metric_reports(candidate_splits)
    validate_activation_metric_reports(baseline_splits)

    gate = build_gate_summary(
        candidate_splits,
        control_margin=float(gate_config["control_margin"]),
        baseline_report=baseline_report,
        dataset_provenance=candidate_report.get("dataset_provenance"),
        baseline_margin=float(gate_config["baseline_margin"]),
        min_control_win_fraction=float(gate_config["min_control_win_fraction"]),
        min_baseline_win_fraction=float(gate_config["min_baseline_win_fraction"]),
        min_baseline_relative_improvement=float(
            gate_config["min_baseline_relative_improvement"]
        ),
        require_baseline_ci_positive=bool(
            gate_config["require_baseline_ci_positive"]
        ),
        require_clustered_baseline_ci=bool(
            gate_config["require_clustered_baseline_ci"]
        ),
        require_baseline_dataset_match=bool(
            gate_config["require_baseline_dataset_match"]
        ),
        bootstrap_samples=int(gate_config["bootstrap_samples"]),
        bootstrap_seed=int(gate_config["bootstrap_seed"]),
        permutation_samples=int(gate_config["permutation_samples"]),
        permutation_seed=int(gate_config["permutation_seed"]),
        min_closed_fraction=float(gate_config["min_closed_fraction"]),
        min_usable_fraction=float(gate_config["min_usable_fraction"]),
        generation_protocol=candidate_report.get("generation_protocol"),
        require_generation_protocol_match=bool(
            gate_config["require_generation_protocol_match"]
        ),
        require_family_level_inference=bool(
            gate_config["require_family_level_inference"]
        ),
        min_independent_families=int(gate_config["min_independent_families"]),
    )
    output = copy.deepcopy(candidate_report)
    output["gate"] = gate
    return output


def run_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = load_config(path)
    paths = config["paths"]
    candidate_path = _resolve_path(paths["candidate_report_json"], base=path.parent)
    baseline_path = _resolve_path(paths["baseline_report_json"], base=path.parent)
    output_path = _resolve_path(paths["output_json"], base=path.parent)
    candidate = json.loads(candidate_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    output = regate_reports(candidate, baseline, gate_config=config["gate"])
    output["regate_provenance"] = {
        "schema_version": SCHEMA_VERSION,
        "config": str(path),
        "candidate_report": str(candidate_path),
        "candidate_report_sha256": _sha256(candidate_path),
        "baseline_report": str(baseline_path),
        "baseline_report_sha256": _sha256(baseline_path),
        "gate_config": copy.deepcopy(config["gate"]),
    }
    write_json(output_path, output)
    return {
        "output_json": str(output_path),
        "gate_passed": bool(output["gate"]["passed"]),
        "publication_status": output["gate"]["publication_status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_config(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
