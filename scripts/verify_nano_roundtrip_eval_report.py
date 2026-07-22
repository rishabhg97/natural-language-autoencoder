#!/usr/bin/env python3
"""Verify a dataset-bound AV-to-AR round-trip report for release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_nano_av_ar_roundtrip_gate import (  # noqa: E402
    MODEL_FINGERPRINT_RE,
    TOKENIZER_FINGERPRINT_RE,
    paired_improvement_summary,
)


SCHEMA_VERSION = "nano_roundtrip_eval_verifier.v1"


class VerificationError(ValueError):
    """Raised when verifier configuration or evidence is malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise VerificationError(f"config must use schema_version {SCHEMA_VERSION}")
    expected = config.get("expected")
    if not isinstance(expected, dict):
        raise VerificationError("expected section is required")
    splits = expected.get("eval_splits")
    if not isinstance(splits, list) or not splits or len(splits) != len(set(splits)):
        raise VerificationError("expected.eval_splits must be a non-empty unique list")
    if set(splits) - {"validation", "test"}:
        raise VerificationError("eval_splits supports validation and test only")
    datasets = (expected.get("generation_identity") or {}).get("dataset_sha256")
    expected_dataset_labels = {"train", "validation", *splits}
    if not isinstance(datasets, dict) or set(datasets) != expected_dataset_labels:
        raise VerificationError(
            "generation dataset hashes must contain exactly train, validation, "
            "and eval_splits"
        )
    for label, digest in datasets.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise VerificationError(f"invalid expected dataset hash for {label}")
    if not config.get("report_json") or not config.get("output_json"):
        raise VerificationError("report_json and output_json are required")
    return config


def verify(config: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(config["report_json"])
    if not report_path.is_file():
        raise VerificationError(f"round-trip report is missing: {report_path}")
    report = json.loads(report_path.read_text())
    expected = config["expected"]
    errors: list[str] = []
    eval_splits = [str(value) for value in expected["eval_splits"]]
    if report.get("eval_splits") != eval_splits:
        errors.append("eval_splits mismatch")
    if bool(expected.get("forbid_unrequested_rows", True)):
        extras = sorted(set((report.get("splits") or {})) - set(eval_splits))
        if extras:
            errors.append(f"unrequested splits were evaluated: {extras}")

    gate = report.get("gate") or {}
    if gate.get("passed") is not True:
        errors.append("round-trip gate did not pass")
    if gate.get("current_generation_protocol_compatible") is not True:
        errors.append("generation protocol compatibility did not pass")
    if gate.get("require_family_level_inference") is not True:
        errors.append("family-level inference was not required")

    expected_identity = expected["generation_identity"]
    provenance = report.get("validated_generation_provenance") or {}
    model_fingerprint = str(provenance.get("model_fingerprint") or "")
    tokenizer_fingerprint = str(provenance.get("tokenizer_fingerprint") or "")
    if not MODEL_FINGERPRINT_RE.fullmatch(model_fingerprint):
        errors.append("generation model fingerprint is invalid")
    elif model_fingerprint != expected_identity["model_fingerprint"]:
        errors.append("generation model fingerprint mismatch")
    if not TOKENIZER_FINGERPRINT_RE.fullmatch(tokenizer_fingerprint):
        errors.append("generation tokenizer fingerprint is invalid")
    elif tokenizer_fingerprint != expected_identity["tokenizer_fingerprint"]:
        errors.append("generation tokenizer fingerprint mismatch")
    datasets = provenance.get("datasets") or {}
    expected_datasets = expected_identity["dataset_sha256"]
    if set(datasets) != set(expected_datasets):
        errors.append("generation dataset set mismatch")
    for split, digest in expected_datasets.items():
        if (datasets.get(split) or {}).get("sha256") != digest:
            errors.append(f"generation dataset hash mismatch for {split}")

    expected_ar = str(expected["ar_checkpoint_dir"])
    if report.get("ar_checkpoint_dir") != expected_ar:
        errors.append("AR checkpoint directory mismatch")

    controls = [str(value) for value in expected["controls"]]
    primary_variant = str(expected.get("primary_variant", "av_real"))
    split_results: dict[str, Any] = {}
    for split in eval_splits:
        split_report = (report.get("splits") or {}).get(split) or {}
        expected_count = int(expected["counts"][split])
        row_count = split_report.get("row_count")
        if row_count != expected_count:
            errors.append(f"{split} row count mismatch")
        family_ids = [str(value) for value in split_report.get("content_family_ids") or []]
        family_count = len(set(family_ids))
        minimum_families = int(expected["min_independent_families"])
        if len(family_ids) != expected_count or family_count < minimum_families:
            errors.append(f"{split} independent family evidence is insufficient")
        variants = split_report.get("variants") or {}
        primary = variants.get(primary_variant) or {}
        primary_directional = primary.get("directional_mse")
        teacher_directional = (variants.get("teacher") or {}).get("directional_mse")
        primary_raw = primary.get("raw_mse")
        centered_r2 = primary.get("centered_raw_r2")
        norm_ratio = primary.get("norm_ratio_mean")
        if not _finite(primary_directional) or float(primary_directional) > float(
            expected["max_primary_directional_mse"]
        ):
            errors.append(f"{split} primary directional MSE exceeds threshold")
        if not _finite(teacher_directional) or not _finite(primary_directional):
            errors.append(f"{split} teacher comparison is missing")
        elif float(primary_directional) - float(teacher_directional) > float(
            expected["max_primary_gap_to_teacher"]
        ):
            errors.append(f"{split} primary is too far behind teacher")
        if not all(_finite(value) for value in (primary_raw, centered_r2, norm_ratio)):
            errors.append(f"{split} raw metric disclosure is incomplete")
        if bool(expected.get("require_raw_magnitude_claim", False)) and (
            not _finite(centered_r2) or float(centered_r2) <= 0.0
        ):
            errors.append(f"{split} raw-magnitude claim is unsupported")

        parse = (split_report.get("generation_parse") or {}).get("real") or {}
        closed = parse.get("closed_fraction")
        usable = parse.get("usable_fraction")
        if not _finite(closed) or float(closed) < float(expected["min_closed_fraction"]):
            errors.append(f"{split} closed parse fraction is too low")
        if not _finite(usable) or float(usable) < float(expected["min_usable_fraction"]):
            errors.append(f"{split} usable parse fraction is too low")

        rowwise = split_report.get("rowwise_directional_mse") or {}
        primary_rows = np.asarray(rowwise.get(primary_variant) or [], dtype=np.float64)
        control_results: dict[str, Any] = {}
        for control in controls:
            control_rows = np.asarray(rowwise.get(control) or [], dtype=np.float64)
            if (
                primary_rows.shape != (expected_count,)
                or control_rows.shape != (expected_count,)
                or not np.isfinite(primary_rows).all()
                or not np.isfinite(control_rows).all()
            ):
                errors.append(f"{split} paired rows are invalid for {control}")
                continue
            inference = paired_improvement_summary(
                primary_rows,
                control_rows,
                content_family_ids=family_ids,
                bootstrap_samples=int(expected.get("bootstrap_samples", 10_000)),
                bootstrap_seed=int(expected.get("bootstrap_seed", 0)),
                permutation_samples=int(expected.get("permutation_samples", 100_000)),
                permutation_seed=int(expected.get("permutation_seed", 0)),
            )
            row_wins = float(np.mean(control_rows > primary_rows))
            passed = (
                inference["independent_unit"] == "content_family_id"
                and inference["independent_unit_count"] >= minimum_families
                and inference["independent_unit_mean_delta"]
                >= float(expected["min_control_margin"])
                and inference["bootstrap_ci95_low"] > 0.0
                and inference["sign_flip_p_value"]
                <= float(expected["max_sign_flip_p_value"])
                and row_wins >= float(expected["min_rowwise_win_fraction"])
            )
            if not passed:
                errors.append(f"{split} primary does not beat control {control}")
            control_results[control] = {
                "passed": bool(passed),
                "rowwise_win_fraction": row_wins,
                **inference,
            }
        split_results[split] = {
            "row_count": row_count,
            "independent_family_count": family_count,
            "primary_directional_mse": primary_directional,
            "teacher_directional_mse": teacher_directional,
            "primary_raw_mse": primary_raw,
            "primary_centered_raw_r2": centered_r2,
            "primary_norm_ratio_mean": norm_ratio,
            "parse": {"closed_fraction": closed, "usable_fraction": usable},
            "controls": control_results,
        }

    raw_claim_supported = all(
        _finite(result["primary_centered_raw_r2"])
        and float(result["primary_centered_raw_r2"]) > 0.0
        for result in split_results.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not errors,
        "report_json": str(report_path),
        "report_sha256": _sha256(report_path),
        "eval_splits": eval_splits,
        "claim_scope": "directional_av_to_ar_reconstruction",
        "raw_magnitude_claim_supported": raw_claim_supported,
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
