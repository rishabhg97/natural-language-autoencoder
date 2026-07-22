#!/usr/bin/env python3
"""Verify a bounded Nano AV evaluation report against a fail-closed contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "nano_av_eval_verifier.v1"


class VerificationError(ValueError):
    """Raised when an AV evaluation report violates its declared contract."""


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
    controls = expected.get("controls")
    if not isinstance(controls, list) or "real" not in controls:
        raise VerificationError("expected.controls must include real")
    counts = expected.get("counts")
    if not isinstance(counts, dict) or any(
        not isinstance(counts.get(split), int) or counts[split] <= 0 for split in splits
    ):
        raise VerificationError("expected.counts must contain a positive count per split")
    if not value.get("report_json") or not value.get("output_json"):
        raise VerificationError("report_json and output_json are required")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(config: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(config["report_json"])
    if not report_path.is_file():
        raise VerificationError(f"AV evaluation report is missing: {report_path}")
    report = json.loads(report_path.read_text())
    expected = config["expected"]
    eval_splits = [str(split) for split in expected["eval_splits"]]
    reported_splits = [str(split) for split in report.get("eval_splits") or []]
    errors: list[str] = []
    if reported_splits != eval_splits:
        errors.append(
            f"eval_splits mismatch: expected={eval_splits} observed={reported_splits}"
        )

    loss_summary = report.get("loss_summary")
    if not isinstance(loss_summary, dict):
        errors.append("loss_summary is missing")
        loss_summary = {}
    controls = [str(control) for control in expected["controls"]]
    missing_controls = sorted(set(controls) - set(loss_summary))
    if missing_controls:
        errors.append(f"missing controls: {missing_controls}")

    split_results: dict[str, Any] = {}
    min_gap = float(expected.get("min_real_control_gap", 0.0))
    for split in eval_splits:
        split_controls: dict[str, Any] = {}
        expected_count = int(expected["counts"][split])
        real_entry = (loss_summary.get("real") or {}).get(split) or {}
        real_count = real_entry.get("count")
        real_loss = real_entry.get("loss")
        if real_count != expected_count:
            errors.append(
                f"real {split} count mismatch: expected={expected_count} observed={real_count}"
            )
        if not isinstance(real_loss, (int, float)) or not math.isfinite(float(real_loss)):
            errors.append(f"real {split} loss is not finite")
            real_loss = None
        for control in controls:
            entry = (loss_summary.get(control) or {}).get(split) or {}
            count = entry.get("count")
            loss = entry.get("loss")
            if count != expected_count:
                errors.append(
                    f"{control} {split} count mismatch: expected={expected_count} observed={count}"
                )
            if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
                errors.append(f"{control} {split} loss is not finite")
                loss = None
            gap = None
            passed = control == "real"
            if control != "real" and real_loss is not None and loss is not None:
                gap = float(loss) - float(real_loss)
                passed = gap > min_gap
                if not passed:
                    errors.append(
                        f"real does not beat {control} on {split}: gap={gap} min>{min_gap}"
                    )
            split_controls[control] = {
                "count": count,
                "loss": loss,
                "control_minus_real": gap,
                "passed": passed,
            }
        split_results[split] = {
            "expected_count": expected_count,
            "controls": split_controls,
        }

    if bool(expected.get("forbid_unrequested_rows", True)):
        for split in sorted({"validation", "test"} - set(eval_splits)):
            for control in controls:
                entry = (loss_summary.get(control) or {}).get(split) or {}
                count = entry.get("count", 0)
                if count not in (0, None):
                    errors.append(
                        f"unrequested split {split} was evaluated for {control}: count={count}"
                    )

    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not errors,
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
