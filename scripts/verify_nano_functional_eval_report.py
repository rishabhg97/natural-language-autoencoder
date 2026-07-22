#!/usr/bin/env python3
"""Verify identity-bound, family-clustered Nano functional recovery evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "nano_functional_eval_verifier.v1"
MODEL_FINGERPRINT_RE = re.compile(r"^(?:dcp_model|hf_model)_sha256:[0-9a-f]{64}$")
TOKENIZER_FINGERPRINT_RE = re.compile(r"^tokenizer_files_sha256:[0-9a-f]{64}$")


class VerificationError(ValueError):
    """Raised when functional evidence violates its declared contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


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
    generation = expected.get("generation_identity")
    if not isinstance(generation, dict):
        raise VerificationError("expected.generation_identity is required")
    for name in ("model_fingerprint", "tokenizer_fingerprint"):
        if not isinstance(generation.get(name), str) or not generation[name]:
            raise VerificationError(
                f"expected.generation_identity.{name} is required"
            )
    dataset_hashes = generation.get("dataset_sha256")
    required_datasets = {"train", "validation", *splits}
    if not isinstance(dataset_hashes, dict) or set(dataset_hashes) != required_datasets:
        raise VerificationError(
            "expected.generation_identity.dataset_sha256 must contain exactly "
            f"{sorted(required_datasets)}"
        )
    for split, digest in dataset_hashes.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise VerificationError(
                f"expected generation dataset hash is invalid for {split}"
            )
    for name in (
        "ar_checkpoint_dir",
        "target_model",
        "source_base_parquet",
        "mean_activation_parquet",
    ):
        if not isinstance(expected.get(name), str) or not expected[name]:
            raise VerificationError(f"expected.{name} is required")
    if not isinstance(expected.get("boundary"), int) or expected["boundary"] <= 0:
        raise VerificationError("expected.boundary must be a positive integer")
    return value


def verify(config: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(config["report_json"])
    if not report_path.is_file():
        raise VerificationError(f"functional report is missing: {report_path}")
    report = json.loads(report_path.read_text())
    expected = config["expected"]
    errors: list[str] = []
    metadata = report.get("metadata") or {}
    eval_splits = [str(value) for value in expected["eval_splits"]]
    if list(metadata.get("eval_splits") or []) != eval_splits:
        errors.append(
            f"eval_splits mismatch: expected={eval_splits} "
            f"observed={metadata.get('eval_splits')}"
        )
    for name in (
        "ar_checkpoint_dir",
        "target_model",
        "source_base_parquet",
        "mean_activation_parquet",
        "boundary",
    ):
        if metadata.get(name) != expected[name]:
            errors.append(
                f"{name} mismatch: expected={expected[name]!r} "
                f"observed={metadata.get(name)!r}"
            )
    minimum_families = int(expected["min_independent_families"])
    independent_families = metadata.get("independent_family_count")
    if not isinstance(independent_families, int) or independent_families < minimum_families:
        errors.append(
            "independent family count is below threshold: "
            f"observed={independent_families} required={minimum_families}"
        )

    identity = metadata.get("generation_identity") or {}
    protocol = identity.get("protocol") or {}
    provenance = identity.get("provenance") or {}
    expected_generation = expected["generation_identity"]
    if str(protocol.get("prefix") or ""):
        errors.append("generation protocol prefix must be empty")
    model_fingerprint = str(provenance.get("model_fingerprint") or "")
    if not MODEL_FINGERPRINT_RE.fullmatch(model_fingerprint):
        errors.append("generation model fingerprint is missing or invalid")
    elif model_fingerprint != expected_generation["model_fingerprint"]:
        errors.append("generation model fingerprint does not match preregistration")
    tokenizer_fingerprint = str(provenance.get("tokenizer_fingerprint") or "")
    if not TOKENIZER_FINGERPRINT_RE.fullmatch(tokenizer_fingerprint):
        errors.append("generation tokenizer fingerprint is missing or invalid")
    elif tokenizer_fingerprint != expected_generation["tokenizer_fingerprint"]:
        errors.append("generation tokenizer fingerprint does not match preregistration")
    datasets = provenance.get("datasets") or {}
    expected_dataset_hashes = expected_generation["dataset_sha256"]
    if set(datasets) != set(expected_dataset_hashes):
        errors.append(
            "generation dataset set mismatch: "
            f"expected={sorted(expected_dataset_hashes)} observed={sorted(datasets)}"
        )
    for split in ("train", *eval_splits):
        digest = (datasets.get(split) or {}).get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"generation provenance lacks a content hash for {split}")
        elif digest != expected_dataset_hashes[split]:
            errors.append(f"generation dataset hash mismatch for {split}")

    gate = report.get("gate") or {}
    if gate.get("identity_passed") is not True:
        errors.append("reinjection identity gate did not pass")
    stored_replay_supported = (
        gate.get("stored_activation_replay_within_tolerance") is True
    )
    if expected.get("require_stored_activation_replay") and not stored_replay_supported:
        errors.append("stored activation replay exceeded identity tolerance")

    report_splits = report.get("splits") or {}
    required_variants = [str(value) for value in expected["required_variants"]]
    variant_counts = {
        str(name): int(value)
        for name, value in (expected.get("variant_counts") or {}).items()
    }
    variant_min_counts = {
        str(name): int(value)
        for name, value in (expected.get("variant_min_counts") or {}).items()
    }
    control_variants = [str(value) for value in expected["candidate_control_variants"]]
    positive_ci_metrics = [str(value) for value in expected["positive_ci_metrics"]]
    positive_mean_metrics = [str(value) for value in expected["positive_mean_metrics"]]
    minimum_fraction = float(expected.get("min_family_better_fraction", 0.5))
    split_results: dict[str, Any] = {}
    for split in eval_splits:
        split_report = report_splits.get(split) or {}
        variants = split_report.get("variants") or {}
        expected_count = int(expected["counts"][split])
        missing = sorted(set(required_variants) - set(variants))
        if missing:
            errors.append(f"{split} missing variants: {missing}")
        for variant in required_variants:
            observed_count = (variants.get(variant) or {}).get("row_count")
            required_count = variant_counts.get(variant, expected_count)
            minimum_count = variant_min_counts.get(variant)
            if minimum_count is not None:
                if not isinstance(observed_count, int) or observed_count < minimum_count:
                    errors.append(
                        f"{split} {variant} row count is below {minimum_count}: "
                        f"observed={observed_count}"
                    )
            elif observed_count != required_count:
                errors.append(f"{split} {variant} row count mismatch")
        comparisons = split_report.get("paired_candidate_vs_variants") or {}
        control_results: dict[str, Any] = {}
        for control in control_variants:
            metrics = comparisons.get(control) or {}
            control_minimum_rows = variant_min_counts.get(control)
            control_expected_rows = variant_counts.get(control, expected_count)
            control_minimum_families = int(
                (expected.get("control_min_independent_families") or {}).get(
                    control, minimum_families
                )
            )
            metric_results: dict[str, Any] = {}
            for metric in positive_ci_metrics + positive_mean_metrics:
                effect = metrics.get(metric) or {}
                mean = effect.get("mean_improvement")
                ci_low = effect.get("ci95_low")
                fraction = effect.get("candidate_better_fraction")
                paired_rows = effect.get("row_count")
                if control_minimum_rows is not None:
                    row_count_passed = (
                        isinstance(paired_rows, int)
                        and paired_rows >= control_minimum_rows
                    )
                else:
                    row_count_passed = paired_rows == control_expected_rows
                clustered = (
                    effect.get("unit") == "cluster"
                    and int(effect.get("cluster_count") or 0)
                    >= control_minimum_families
                )
                passed = (
                    _finite(mean)
                    and float(mean) > 0.0
                    and clustered
                    and row_count_passed
                )
                if metric in positive_ci_metrics:
                    passed = passed and _finite(ci_low) and float(ci_low) > 0.0
                if metric in positive_ci_metrics:
                    passed = (
                        passed
                        and _finite(fraction)
                        and float(fraction) >= minimum_fraction
                    )
                if not passed:
                    errors.append(
                        f"{split} candidate does not pass {metric} versus {control}"
                    )
                metric_results[metric] = {"passed": bool(passed), **effect}
            control_results[control] = metric_results
        split_results[split] = {
            "expected_count": expected_count,
            "candidate_vs_controls": control_results,
        }

    if bool(expected.get("forbid_unrequested_rows", True)):
        for split in sorted({"validation", "test"} - set(eval_splits)):
            if split in report_splits:
                errors.append(f"unrequested split {split} was evaluated")

    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not errors,
        "report_json": str(report_path),
        "report_sha256": _sha256(report_path),
        "eval_splits": eval_splits,
        "independent_family_count": independent_families,
        "fresh_forward_activation_claim_supported": stored_replay_supported,
        "claim_scope": (
            "fresh_forward_functional_recovery"
            if stored_replay_supported
            else "stored_snapshot_counterfactual_reinjection"
        ),
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
