#!/usr/bin/env python3
"""Apply the composite validity gate before promoting fixed-AR R33 RL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_eval_core import clustered_paired_bootstrap_improvement  # noqa: E402


SCHEMA_VERSION = "nano_r33_validity_gate.v1"


class ValidityGateError(ValueError):
    """Raised when evidence cannot be evaluated safely."""


def _check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    observed: Any,
    threshold: Any,
    split: str,
    evidence: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "threshold": threshold,
            "split": split,
            "evidence": evidence,
        }
    )


def _finite_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValidityGateError(f"{name} must be a non-empty 1D array")
    if not np.isfinite(array).all():
        raise ValidityGateError(f"{name} must contain only finite values")
    return array


def _functional_metric(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    try:
        return _finite_array([row[metric] for row in rows], name=f"functional.{metric}")
    except KeyError as exc:
        raise ValidityGateError(f"functional rows are missing {metric}") from exc


def _all_row_keys_match(row_keys: dict[str, Any], required_rows: int) -> tuple[bool, dict[str, Any]]:
    required_sources = ("sft", "candidate", "invariance", "functional")
    missing = [source for source in required_sources if source not in row_keys]
    normalized = {
        source: [str(value) for value in row_keys.get(source, [])]
        for source in required_sources
    }
    reference = normalized["sft"]
    passed = (
        not missing
        and len(reference) == required_rows
        and len(set(reference)) == required_rows
        and all(normalized[source] == reference for source in required_sources[1:])
    )
    return passed, {
        "counts": {source: len(values) for source, values in normalized.items()},
        "missing_sources": missing,
        "all_equal": all(
            normalized[source] == reference for source in required_sources[1:]
        ),
    }


def evaluate_gate(bundle: dict[str, Any]) -> dict[str, Any]:
    thresholds = bundle.get("thresholds")
    splits = bundle.get("splits")
    if not isinstance(thresholds, dict) or not isinstance(splits, dict):
        raise ValidityGateError("bundle requires thresholds and splits mappings")
    required_splits = tuple(str(value) for value in bundle.get("eval_splits", ["validation"]))
    if not required_splits or len(set(required_splits)) != len(required_splits) or not set(
        required_splits
    ).issubset({"validation", "test"}):
        raise ValidityGateError(
            "eval_splits must be a non-empty unique validation/test list"
        )
    missing_splits = [split for split in required_splits if split not in splits]
    if missing_splits:
        raise ValidityGateError(f"bundle is missing splits: {missing_splits}")

    required_rows = int(thresholds["required_rows"])
    relative_threshold = float(thresholds["roundtrip_relative_improvement"])
    topk_regression = float(thresholds["functional_topk_max_regression"])
    invariance_threshold = float(thresholds["invariance_fve_retention"])
    control_threshold = float(thresholds["control_win_fraction"])
    usable_threshold = float(thresholds["usable_fraction"])
    closed_threshold = float(thresholds["closed_fraction"])
    qualitative_threshold = float(thresholds["qualitative_flag_fraction"])
    bootstrap_resamples = int(thresholds.get("bootstrap_resamples", 10_000))
    bootstrap_seed = int(thresholds.get("bootstrap_seed", 0))
    min_independent_families = int(
        thresholds.get("min_independent_families", min(required_rows, 30))
    )
    if required_rows <= 0 or bootstrap_resamples <= 0:
        raise ValidityGateError("required_rows and bootstrap_resamples must be positive")

    checks: list[dict[str, Any]] = []
    evidence = bundle.get("evidence", {})
    for split in required_splits:
        split_data = splits[split]
        if not isinstance(split_data, dict):
            raise ValidityGateError(f"split {split!r} must be a mapping")

        row_identity_passed, row_identity_observed = _all_row_keys_match(
            split_data.get("row_keys", {}), required_rows
        )
        family_ids = [
            str(value) for value in split_data.get("content_family_ids", [])
        ]
        independent_family_count = len(set(family_ids))
        _check(
            checks,
            name=f"independent_families:{split}",
            passed=(
                len(family_ids) == required_rows
                and all(family_ids)
                and independent_family_count >= min_independent_families
            ),
            observed={
                "row_count": len(family_ids),
                "independent_family_count": independent_family_count,
            },
            threshold={
                "required_rows": required_rows,
                "minimum_independent_families": min_independent_families,
            },
            split=split,
            evidence=evidence,
        )
        _check(
            checks,
            name=f"row_identity:{split}",
            passed=row_identity_passed,
            observed=row_identity_observed,
            threshold={"required_rows": required_rows, "exact_order_match": True},
            split=split,
            evidence=evidence,
        )

        roundtrip = split_data.get("roundtrip", {})
        sft_nmse = _finite_array(roundtrip.get("sft_nmse"), name=f"{split}.sft_nmse")
        candidate_nmse = _finite_array(
            roundtrip.get("candidate_nmse"), name=f"{split}.candidate_nmse"
        )
        if sft_nmse.shape != candidate_nmse.shape:
            raise ValidityGateError(f"{split} round-trip arrays must be paired")
        sft_mean = float(sft_nmse.mean())
        candidate_mean = float(candidate_nmse.mean())
        relative_improvement = (
            (sft_mean - candidate_mean) / sft_mean if sft_mean > 0.0 else float("-inf")
        )
        _check(
            checks,
            name=f"roundtrip_relative_improvement:{split}",
            passed=relative_improvement >= relative_threshold,
            observed={
                "sft_nmse": sft_mean,
                "candidate_nmse": candidate_mean,
                "relative_improvement": relative_improvement,
            },
            threshold={"minimum_relative_improvement": relative_threshold},
            split=split,
            evidence=evidence,
        )
        nmse_paired = clustered_paired_bootstrap_improvement(
            sft_nmse,
            candidate_nmse,
            family_ids,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
        _check(
            checks,
            name=f"roundtrip_paired_ci:{split}",
            passed=float(nmse_paired["ci95_low"]) > 0.0,
            observed=nmse_paired,
            threshold={"ci95_low_strictly_above": 0.0},
            split=split,
            evidence=evidence,
        )

        functional = split_data.get("functional", {})
        gold_functional = functional.get("stored_gold")
        sft_functional = functional.get("sft")
        candidate_functional = functional.get("candidate")
        if (
            not isinstance(gold_functional, list)
            or not isinstance(sft_functional, list)
            or not isinstance(candidate_functional, list)
        ):
            raise ValidityGateError(f"{split} functional rows are required")
        gold_kl = _functional_metric(gold_functional, "kl_original_to_patched")
        sft_kl = _functional_metric(sft_functional, "kl_original_to_patched")
        candidate_kl = _functional_metric(
            candidate_functional, "kl_original_to_patched"
        )
        if gold_kl.shape != sft_kl.shape or sft_kl.shape != candidate_kl.shape:
            raise ValidityGateError(f"{split} functional KL arrays must be paired")
        _check(
            checks,
            name=f"functional_gold_reference:{split}",
            passed=True,
            observed={
                "stored_gold_mean_kl": float(gold_kl.mean()),
                "sft_excess_kl": float(sft_kl.mean() - gold_kl.mean()),
                "candidate_excess_kl": float(candidate_kl.mean() - gold_kl.mean()),
            },
            threshold={"required": True, "exact_row_alignment": True},
            split=split,
            evidence=evidence,
        )
        kl_paired = clustered_paired_bootstrap_improvement(
            sft_kl,
            candidate_kl,
            family_ids,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
        _check(
            checks,
            name=f"functional_kl:{split}",
            passed=(
                float(candidate_kl.mean()) < float(sft_kl.mean())
                and float(kl_paired["ci95_low"]) > 0.0
            ),
            observed={
                "sft_mean": float(sft_kl.mean()),
                "candidate_mean": float(candidate_kl.mean()),
                "paired": kl_paired,
            },
            threshold={"candidate_lower": True, "ci95_low_strictly_above": 0.0},
            split=split,
            evidence=evidence,
        )
        for metric in ("top_10_overlap", "top_50_overlap"):
            gold_values = _functional_metric(gold_functional, metric)
            sft_values = _functional_metric(sft_functional, metric)
            candidate_values = _functional_metric(candidate_functional, metric)
            regression = float(sft_values.mean() - candidate_values.mean())
            _check(
                checks,
                name=f"functional_{metric}:{split}",
                passed=regression <= topk_regression,
                observed={
                    "sft_mean": float(sft_values.mean()),
                    "candidate_mean": float(candidate_values.mean()),
                    "stored_gold_mean": float(gold_values.mean()),
                    "candidate_gap_to_stored_gold": float(
                        gold_values.mean() - candidate_values.mean()
                    ),
                    "regression": regression,
                },
                threshold={"maximum_regression": topk_regression},
                split=split,
                evidence=evidence,
            )

        invariance = split_data.get("invariance_retention", {})
        if not isinstance(invariance, dict) or not invariance:
            raise ValidityGateError(f"{split} invariance retention is required")
        minimum_retention = min(float(value) for value in invariance.values())
        _check(
            checks,
            name=f"invariance_fve_retention:{split}",
            passed=minimum_retention >= invariance_threshold,
            observed={"minimum": minimum_retention, "transforms": invariance},
            threshold={"minimum": invariance_threshold},
            split=split,
            evidence=evidence,
        )

        control_wins = split_data.get("control_win_fractions", {})
        if not isinstance(control_wins, dict) or not control_wins:
            raise ValidityGateError(f"{split} control win fractions are required")
        minimum_control_win = min(float(value) for value in control_wins.values())
        _check(
            checks,
            name=f"control_win_fraction:{split}",
            passed=minimum_control_win >= control_threshold,
            observed={"minimum": minimum_control_win, "controls": control_wins},
            threshold={"minimum_for_every_control": control_threshold},
            split=split,
            evidence=evidence,
        )

        parse = split_data.get("parse", {})
        usable = float(parse.get("usable_fraction", float("-inf")))
        closed = float(parse.get("closed_fraction", float("-inf")))
        _check(
            checks,
            name=f"parse_health:{split}",
            passed=usable >= usable_threshold and closed >= closed_threshold,
            observed={"usable_fraction": usable, "closed_fraction": closed},
            threshold={
                "minimum_usable_fraction": usable_threshold,
                "minimum_closed_fraction": closed_threshold,
            },
            split=split,
            evidence=evidence,
        )

        leakage = split_data.get("leakage", {})
        injection_count = int(leakage.get("injection_marker_count", -1))
        cjk_count = int(leakage.get("cjk_count", -1))
        _check(
            checks,
            name=f"leakage:{split}",
            passed=injection_count == 0 and cjk_count == 0,
            observed={
                "injection_marker_count": injection_count,
                "cjk_count": cjk_count,
            },
            threshold={"maximum_each": 0},
            split=split,
            evidence=evidence,
        )

        qualitative = split_data.get("qualitative", {})
        qualitative_rows = int(qualitative.get("row_count", 0))
        reviewed_rows = int(qualitative.get("reviewed_count", 0))
        flagged = int(qualitative.get("flagged_count", qualitative_rows + 1))
        flag_fraction = flagged / qualitative_rows if qualitative_rows > 0 else float("inf")
        _check(
            checks,
            name=f"qualitative_review_complete:{split}",
            passed=qualitative_rows >= 50 and reviewed_rows == qualitative_rows,
            observed={
                "row_count": qualitative_rows,
                "reviewed_count": reviewed_rows,
            },
            threshold={"minimum_rows": 50, "reviewed_rows": "all"},
            split=split,
            evidence=evidence,
        )
        _check(
            checks,
            name=f"qualitative_flag_fraction:{split}",
            passed=(
                qualitative_rows >= 50
                and reviewed_rows == qualitative_rows
                and flagged >= 0
                and flag_fraction <= qualitative_threshold
            ),
            observed={
                "row_count": qualitative_rows,
                "reviewed_count": reviewed_rows,
                "flagged_count": flagged,
                "flag_fraction": flag_fraction,
            },
            threshold={
                "minimum_rows": 50,
                "maximum_flag_fraction": qualitative_threshold,
            },
            split=split,
            evidence=evidence,
        )

    blockers = [check["name"] for check in checks if not check["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_name": bundle.get("candidate_name"),
        "eval_splits": list(required_splits),
        "passed": not blockers,
        "checks": checks,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    from nano_validity_evidence import load_gate_bundle

    bundle = load_gate_bundle(args.config, candidate_name=args.candidate)
    report = evaluate_gate(bundle)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
