#!/usr/bin/env python3
"""Fail-closed promotion gate for primary and independent R33 AR critics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SPLITS = ("validation", "test")
DATASETS = ("train", "validation", "test")


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _dataset_hashes(report: dict[str, Any]) -> dict[str, str] | None:
    provenance = report.get("dataset_provenance") or {}
    hashes = {
        name: str((provenance.get(name) or {}).get("sha256") or "")
        for name in DATASETS
    }
    return hashes if all(hashes.values()) else None


def _generated_text_hash(report: dict[str, Any]) -> str | None:
    value = (report.get("generated_jsonl_provenance") or {}).get("sha256")
    return str(value) if value else None


def _generation_protocol_hash(report: dict[str, Any]) -> str | None:
    value = report.get("generation_protocol_sha256")
    return str(value) if value else None


def _split_identity(report: dict[str, Any], split: str) -> tuple[list[Any], list[str]] | None:
    payload = (report.get("splits") or {}).get(split) or {}
    row_indices = payload.get("row_indices")
    row_keys = payload.get("row_keys")
    if not isinstance(row_indices, list) or not row_indices:
        return None
    if not isinstance(row_keys, list) or len(row_keys) != len(row_indices):
        return None
    doc_ids = payload.get("doc_ids")
    if isinstance(doc_ids, list) and len(doc_ids) == len(row_indices):
        canonical_doc_ids = [str(doc_id) for doc_id in doc_ids]
    else:
        canonical_doc_ids = []
        for row_key in row_keys:
            if not isinstance(row_key, dict) or not row_key.get("doc_id"):
                return None
            canonical_doc_ids.append(str(row_key["doc_id"]))

    # The parquet hash plus row index identifies the exact row. Document IDs
    # provide an independent provenance check. Optional row-key fields such as
    # n_raw_tokens may be added by later enrichment and are not row identity.
    return row_indices, canonical_doc_ids


def _split_metrics(report: dict[str, Any], split: str) -> dict[str, Any] | None:
    payload = ((report.get("gate") or {}).get("splits") or {}).get(split)
    if not isinstance(payload, dict):
        return None
    paired = payload.get("baseline_paired_improvement") or {}
    win = payload.get("baseline_rowwise_win_rate") or {}
    length_analysis = ((report.get("splits") or {}).get(split) or {}).get(
        "length_analysis"
    ) or {}
    return {
        "relative_improvement": _finite_number(paired.get("relative_improvement")),
        "clustered_ci95_low": _finite_number(paired.get("bootstrap_ci95_low")),
        "independent_unit": paired.get("independent_unit"),
        "independent_unit_count": paired.get("independent_unit_count"),
        "row_win_fraction": _finite_number(win.get("candidate_better_fraction")),
        "baseline_row_identity_match": payload.get("baseline_row_identity_match") is True,
        "baseline_dataset_hash_match": payload.get("baseline_dataset_hash_match") is True,
        "parse_health_passed": (payload.get("parse_health") or {}).get("passed") is True,
        "best_length_matched_relative_improvement": _finite_number(
            length_analysis.get("best_length_matched_relative_improvement")
        ),
    }


def build_cross_critic_gate(
    *,
    primary_candidate: dict[str, Any],
    independent_candidate: dict[str, Any],
    primary_sft: dict[str, Any],
    independent_sft: dict[str, Any],
    min_primary_relative_improvement: float = 0.10,
    min_independent_relative_improvement: float = 0.05,
    min_row_win_fraction: float = 0.50,
    min_independent_to_primary_gain_ratio: float = 0.75,
    require_family_clustered_ci_positive: bool = True,
    require_length_control_gain: bool = True,
) -> dict[str, Any]:
    reports = {
        "primary_candidate": primary_candidate,
        "independent_candidate": independent_candidate,
        "primary_sft": primary_sft,
        "independent_sft": independent_sft,
    }
    dataset_hashes = {name: _dataset_hashes(report) for name, report in reports.items()}
    dataset_identity = (
        all(value is not None for value in dataset_hashes.values())
        and len({json.dumps(value, sort_keys=True) for value in dataset_hashes.values()}) == 1
    )
    generated_text_hashes = {
        name: _generated_text_hash(report) for name, report in reports.items()
    }
    candidate_generated_text_identity = bool(
        generated_text_hashes["primary_candidate"]
        and generated_text_hashes["primary_candidate"]
        == generated_text_hashes["independent_candidate"]
    )
    sft_generated_text_identity = bool(
        generated_text_hashes["primary_sft"]
        and generated_text_hashes["primary_sft"]
        == generated_text_hashes["independent_sft"]
    )
    generation_protocol_hashes = {
        name: _generation_protocol_hash(report) for name, report in reports.items()
    }
    generation_protocol_identity = bool(
        all(generation_protocol_hashes.values())
        and len(set(generation_protocol_hashes.values())) == 1
    )

    split_results: dict[str, Any] = {}
    for split in SPLITS:
        identities = {name: _split_identity(report, split) for name, report in reports.items()}
        row_identity = (
            all(value is not None for value in identities.values())
            and len({json.dumps(value, sort_keys=True) for value in identities.values()}) == 1
        )
        primary = _split_metrics(primary_candidate, split)
        independent = _split_metrics(independent_candidate, split)
        primary_relative = None if primary is None else primary["relative_improvement"]
        independent_relative = None if independent is None else independent["relative_improvement"]
        gain_ratio = (
            independent_relative / primary_relative
            if primary_relative is not None
            and independent_relative is not None
            and primary_relative > 0.0
            else None
        )
        checks = {
            "dataset_identity": bool(dataset_identity),
            "row_identity": bool(row_identity),
            "candidate_generated_text_identity": candidate_generated_text_identity,
            "sft_generated_text_identity": sft_generated_text_identity,
            "generation_protocol_identity": generation_protocol_identity,
            "primary_metrics_present": primary is not None and primary_relative is not None,
            "independent_metrics_present": independent is not None and independent_relative is not None,
            "primary_relative_improvement": (
                primary_relative is not None
                and primary_relative >= min_primary_relative_improvement
            ),
            "independent_relative_improvement": (
                independent_relative is not None
                and independent_relative >= min_independent_relative_improvement
            ),
            "primary_family_clustered_ci_positive": bool(
                not require_family_clustered_ci_positive
                or (
                    primary
                    and primary["independent_unit"] == "content_family_id"
                    and primary["clustered_ci95_low"] is not None
                    and primary["clustered_ci95_low"] > 0.0
                )
            ),
            "independent_family_clustered_ci_positive": bool(
                not require_family_clustered_ci_positive
                or (
                    independent
                    and independent["independent_unit"] == "content_family_id"
                    and independent["clustered_ci95_low"] is not None
                    and independent["clustered_ci95_low"] > 0.0
                )
            ),
            "primary_row_wins": bool(
                primary
                and primary["row_win_fraction"] is not None
                and primary["row_win_fraction"] > min_row_win_fraction
            ),
            "independent_row_wins": bool(
                independent
                and independent["row_win_fraction"] is not None
                and independent["row_win_fraction"] > min_row_win_fraction
            ),
            "primary_baseline_binding": bool(
                primary
                and primary["baseline_row_identity_match"]
                and primary["baseline_dataset_hash_match"]
            ),
            "independent_baseline_binding": bool(
                independent
                and independent["baseline_row_identity_match"]
                and independent["baseline_dataset_hash_match"]
            ),
            "parse_health": bool(
                primary
                and independent
                and primary["parse_health_passed"]
                and independent["parse_health_passed"]
            ),
            "gain_ratio": (
                gain_ratio is not None
                and gain_ratio >= min_independent_to_primary_gain_ratio
            ),
            "primary_length_control_gain": bool(
                not require_length_control_gain
                or (
                    primary
                    and primary["best_length_matched_relative_improvement"] is not None
                    and primary["best_length_matched_relative_improvement"] > 0.0
                )
            ),
            "independent_length_control_gain": bool(
                not require_length_control_gain
                or (
                    independent
                    and independent["best_length_matched_relative_improvement"] is not None
                    and independent["best_length_matched_relative_improvement"] > 0.0
                )
            ),
        }
        split_results[split] = {
            "passed": all(checks.values()),
            "checks": checks,
            "primary": primary,
            "independent": independent,
            "gain_ratio": gain_ratio,
        }

    return {
        "schema_version": "nano_cross_critic_gate.v1",
        "passed": all(result["passed"] for result in split_results.values()),
        "thresholds": {
            "min_primary_relative_improvement": float(min_primary_relative_improvement),
            "min_independent_relative_improvement": float(min_independent_relative_improvement),
            "min_row_win_fraction_exclusive": float(min_row_win_fraction),
            "min_independent_to_primary_gain_ratio": float(min_independent_to_primary_gain_ratio),
            "require_family_clustered_ci_positive": bool(
                require_family_clustered_ci_positive
            ),
            "require_length_control_gain": bool(require_length_control_gain),
        },
        "dataset_hashes": dataset_hashes,
        "generated_text_hashes": generated_text_hashes,
        "generation_protocol_hashes": generation_protocol_hashes,
        "splits": split_results,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-candidate-report", type=Path, required=True)
    parser.add_argument("--independent-candidate-report", type=Path, required=True)
    parser.add_argument("--primary-sft-report", type=Path, required=True)
    parser.add_argument("--independent-sft-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-primary-relative-improvement", type=float, default=0.10)
    parser.add_argument("--min-independent-relative-improvement", type=float, default=0.05)
    parser.add_argument("--min-row-win-fraction", type=float, default=0.50)
    parser.add_argument("--min-independent-to-primary-gain-ratio", type=float, default=0.75)
    parser.add_argument(
        "--no-require-family-clustered-ci-positive",
        dest="require_family_clustered_ci_positive",
        action="store_false",
    )
    parser.add_argument(
        "--no-require-length-control-gain",
        dest="require_length_control_gain",
        action="store_false",
    )
    parser.set_defaults(
        require_family_clustered_ci_positive=True,
        require_length_control_gain=True,
    )
    args = parser.parse_args()

    paths = {
        "primary_candidate": args.primary_candidate_report,
        "independent_candidate": args.independent_candidate_report,
        "primary_sft": args.primary_sft_report,
        "independent_sft": args.independent_sft_report,
    }
    result = build_cross_critic_gate(
        primary_candidate=_read_json(paths["primary_candidate"]),
        independent_candidate=_read_json(paths["independent_candidate"]),
        primary_sft=_read_json(paths["primary_sft"]),
        independent_sft=_read_json(paths["independent_sft"]),
        min_primary_relative_improvement=args.min_primary_relative_improvement,
        min_independent_relative_improvement=args.min_independent_relative_improvement,
        min_row_win_fraction=args.min_row_win_fraction,
        min_independent_to_primary_gain_ratio=args.min_independent_to_primary_gain_ratio,
        require_family_clustered_ci_positive=args.require_family_clustered_ci_positive,
        require_length_control_gain=args.require_length_control_gain,
    )
    result["sources"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
