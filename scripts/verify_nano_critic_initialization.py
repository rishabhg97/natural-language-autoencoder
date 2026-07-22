#!/usr/bin/env python3
"""Fail-closed comparison of primary and transfer-critic initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INITIALIZATION_SCHEMA = "nano_critic_initialization.v1"
REPORT_SCHEMA = "nano_critic_initialization_verification.v1"


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def verify_initializations(
    primary: dict[str, Any],
    independent: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}

    for label, manifest in (("primary", primary), ("independent", independent)):
        valid_schema = manifest.get("schema_version") == INITIALIZATION_SCHEMA
        checks[f"{label}_schema"] = valid_schema
        if not valid_schema:
            errors.append(f"{label} schema_version must be {INITIALIZATION_SCHEMA}")

    for field in (
        "base_model",
        "dataset_sidecar",
        "extraction_layer_index",
        "torch_dtype",
    ):
        matches = primary.get(field) == independent.get(field) and primary.get(field) is not None
        checks[f"shared_{field}"] = matches
        if not matches:
            errors.append(f"{field} must match")

    primary_head = _mapping(primary.get("value_head"))
    independent_head = _mapping(independent.get("value_head"))
    primary_router = _mapping(primary.get("router"))
    independent_router = _mapping(independent.get("router"))

    checks["primary_identity_head"] = primary_head.get("mode") == "identity"
    if not checks["primary_identity_head"]:
        errors.append("primary value head must use identity initialization")
    checks["primary_pretrained_router"] = primary_router.get("mode") == "pretrained"
    if not checks["primary_pretrained_router"]:
        errors.append("primary router must use pretrained initialization")
    checks["independent_seeded_head"] = independent_head.get("mode") == "seeded_givens"
    if not checks["independent_seeded_head"]:
        errors.append("independent value head must use seeded_givens")
    checks["independent_seeded_router"] = (
        independent_router.get("mode") == "seeded_relative_noise"
    )
    if not checks["independent_seeded_router"]:
        errors.append("independent router must use seeded_relative_noise")

    head_seed = independent_head.get("seed")
    router_seed = independent_router.get("seed")
    checks["shared_independent_seed"] = (
        isinstance(head_seed, int) and head_seed == router_seed
    )
    if not checks["shared_independent_seed"]:
        errors.append("independent head and router must share an explicit integer seed")

    primary_head_hash = primary_head.get("after_sha256")
    independent_head_hash = independent_head.get("after_sha256")
    checks["distinct_value_head_hash"] = (
        isinstance(primary_head_hash, str)
        and isinstance(independent_head_hash, str)
        and primary_head_hash != independent_head_hash
    )
    if not checks["distinct_value_head_hash"]:
        errors.append("value-head hashes must differ")

    primary_router_before = primary_router.get("before_sha256")
    primary_router_after = primary_router.get("after_sha256")
    independent_router_before = independent_router.get("before_sha256")
    independent_router_after = independent_router.get("after_sha256")
    checks["primary_router_unchanged"] = (
        isinstance(primary_router_before, str)
        and primary_router_before == primary_router_after
    )
    if not checks["primary_router_unchanged"]:
        errors.append("primary router hash must remain unchanged")
    checks["shared_router_start"] = (
        isinstance(primary_router_before, str)
        and primary_router_before == independent_router_before
    )
    if not checks["shared_router_start"]:
        errors.append("primary and independent router pre-initialization hashes must match")
    checks["independent_router_changed"] = (
        isinstance(independent_router_before, str)
        and isinstance(independent_router_after, str)
        and independent_router_before != independent_router_after
    )
    if not checks["independent_router_changed"]:
        errors.append("independent router hash must change")
    checks["router_parameters_present"] = int(
        independent_router.get("parameter_count") or 0
    ) > 0
    if not checks["router_parameters_present"]:
        errors.append("independent router parameter_count must be positive")

    return {
        "schema_version": REPORT_SCHEMA,
        "passed": not errors,
        "independent_seed": head_seed,
        "checks": checks,
        "errors": errors,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args(argv)

    primary = json.loads(args.primary.read_text())
    independent = json.loads(args.independent.read_text())
    report = verify_initializations(primary, independent)
    report["sources"] = {
        "primary": {
            "path": str(args.primary),
            "sha256": _sha256_file(args.primary),
        },
        "independent": {
            "path": str(args.independent),
            "sha256": _sha256_file(args.independent),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
