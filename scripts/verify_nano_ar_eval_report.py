#!/usr/bin/env python3
"""Verify a bounded Nano AR evaluation without overstating magnitude recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "nano_ar_eval_verifier.v1"


class VerificationError(ValueError):
    """Raised when an AR evaluation report violates its declared contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise VerificationError(f"config must use schema_version {SCHEMA_VERSION}")
    expected = value.get("expected")
    if not isinstance(expected, dict):
        raise VerificationError("expected section is required")
    splits = expected.get("eval_splits")
    if not isinstance(splits, list) or not splits or len(splits) != len(set(splits)):
        raise VerificationError("expected.eval_splits must be a non-empty unique list")
    if set(splits) - {"validation", "test"}:
        raise VerificationError("expected.eval_splits supports validation and test only")
    if not value.get("report_json") or not value.get("output_json"):
        raise VerificationError("report_json and output_json are required")
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def verify(config: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(config["report_json"])
    if not report_path.is_file():
        raise VerificationError(f"AR evaluation report is missing: {report_path}")
    report = json.loads(report_path.read_text())
    expected = config["expected"]
    eval_splits = [str(split) for split in expected["eval_splits"]]
    errors: list[str] = []
    if list(report.get("eval_splits") or []) != eval_splits:
        errors.append(
            "eval_splits mismatch: "
            f"expected={eval_splits} observed={report.get('eval_splits')}"
        )

    report_splits = report.get("splits")
    if not isinstance(report_splits, dict):
        report_splits = {}
        errors.append("splits section is missing")
    required_controls = [str(value) for value in expected.get("required_controls") or []]
    teacher_must_beat = [str(value) for value in expected.get("teacher_must_beat") or []]
    min_gap = float(expected.get("min_teacher_control_nmse_gap", 0.0))
    split_results: dict[str, Any] = {}

    for split in eval_splits:
        split_report = report_splits.get(split) or {}
        controls = split_report.get("controls") or {}
        expected_count = int(expected["counts"][split])
        if split_report.get("row_count") != expected_count:
            errors.append(
                f"{split} row count mismatch: expected={expected_count} "
                f"observed={split_report.get('row_count')}"
            )
        missing = sorted(set(required_controls) - set(controls))
        if missing:
            errors.append(f"{split} missing controls: {missing}")
        control_results: dict[str, Any] = {}
        for control in required_controls:
            entry = controls.get(control) or {}
            if entry.get("row_count") != expected_count:
                errors.append(f"{split} {control} row count mismatch")
            for metric in ("normalized_mse", "raw_mse", "cosine_mean", "fve_nrm"):
                if not _finite(entry.get(metric)):
                    errors.append(f"{split} {control} {metric} is not finite")
            control_results[control] = entry

        teacher = controls.get("teacher") or {}
        teacher_nmse = teacher.get("normalized_mse")
        teacher_cosine = teacher.get("cosine_mean")
        teacher_fve = teacher.get("fve_nrm")
        if _finite(teacher_nmse) and float(teacher_nmse) > float(expected["max_teacher_normalized_mse"]):
            errors.append(f"{split} teacher normalized_mse exceeds threshold")
        if _finite(teacher_cosine) and float(teacher_cosine) < float(expected["min_teacher_cosine"]):
            errors.append(f"{split} teacher cosine_mean is below threshold")
        if _finite(teacher_fve) and float(teacher_fve) < float(expected["min_teacher_fve_nrm"]):
            errors.append(f"{split} teacher fve_nrm is below threshold")

        gaps: dict[str, float | None] = {}
        win_rates = split_report.get("rowwise_win_rates") or {}
        for control in teacher_must_beat:
            control_nmse = (controls.get(control) or {}).get("normalized_mse")
            gap = None
            if _finite(teacher_nmse) and _finite(control_nmse):
                gap = float(control_nmse) - float(teacher_nmse)
                if gap <= min_gap:
                    errors.append(f"{split} teacher does not beat {control}: gap={gap}")
            gaps[control] = gap
            minimum_win = float(
                (expected.get("min_teacher_rowwise_win_fraction") or {}).get(control, 0.0)
            )
            win = (win_rates.get(f"teacher_vs_{control}") or {}).get(
                "teacher_better_fraction"
            )
            if not _finite(win) or float(win) < minimum_win:
                errors.append(
                    f"{split} teacher rowwise win fraction versus {control} is below "
                    f"{minimum_win}: observed={win}"
                )

        centered_raw_r2 = teacher.get("centered_raw_r2")
        raw_claim_supported = _finite(centered_raw_r2) and float(centered_raw_r2) >= float(
            expected.get("min_centered_raw_r2_for_magnitude_claim", 0.0)
        )
        if expected.get("require_raw_magnitude_claim") and not raw_claim_supported:
            errors.append(f"{split} raw-magnitude claim is not supported")
        split_results[split] = {
            "expected_count": expected_count,
            "teacher": teacher,
            "teacher_control_nmse_gaps": gaps,
            "raw_magnitude_claim_supported": raw_claim_supported,
        }

    if bool(expected.get("forbid_unrequested_rows", True)):
        for split in sorted({"validation", "test"} - set(eval_splits)):
            entry = report_splits.get(split)
            if isinstance(entry, dict) and int(entry.get("row_count") or 0) > 0:
                errors.append(f"unrequested split {split} was evaluated")

    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not errors,
        "claim_scope": "directional_activation_reconstruction",
        "report_json": str(report_path),
        "report_sha256": _sha256(report_path),
        "eval_splits": eval_splits,
        "split_results": split_results,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    result = verify(config)
    output = Path(config["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
