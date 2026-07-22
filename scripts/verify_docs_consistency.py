#!/usr/bin/env python3
"""Fail when active R33 claim documents omit publication invalidation state."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import yaml


PUBLICATION_INVALIDATION_MARKER = (
    "<!-- R33-HERO-BASELINE-PROTOCOL-INVALIDATED -->"
)
SELECTED_HERO_ID = "r33-corrected-k3-hero-lr1e5-update342-resume228-retry3"
LEGACY_HEADLINE_RE = re.compile(
    r"(?:30\.97\s*%|32\.34\s*%|0\.3096944419939734|0\.3233814903044525)"
)
REGISTRY_INVALIDATION_EXPECTATIONS = (
    (("publication_valid",), False),
    (
        ("publication_invalid_reason",),
        "stored_activation_identity_failure_and_exploratory_test_exposure",
    ),
    (("corrected_effect_pending",), False),
    (("metrics", "corrected_salvage", "cross_critic_gate_passed"), True),
    (("provenance", "activation_fidelity_publication_ready"), False),
)

DEFAULT_PATHS = (
    "README.md",
    "docs/current_state.md",
    "docs/execution_log.md",
    "docs/experiment_logbook.md",
    "docs/nano30b-nla-core-plan.md",
    "docs/nano_av_job_tracker.md",
    "docs/nano_av_run_history.md",
    "docs/rl_logbook.md",
    "docs/runs/r33_av_hpo_202606.md",
    "docs/runs/r33_gate_matrix.md",
    "docs/runs/r33_rl_hero_20260708.md",
    "runs/registry/experiments.yaml",
)

CANONICAL_REQUIREMENTS = {
    "README.md": (
        "docs/methods/measurement_contract.md",
        "qualified family-clean R33 SFT AV+AR pair",
        "independently initialized and trained",
        "not a pristine upstream checkout",
    ),
    "docs/current_state.md": (
        "centered raw R2",
        "publication_ready=false",
        "described as the final immutable launch source",
    ),
    "docs/runs/r33_gate_matrix.md": (
        "fresh-forward fidelity failed",
        "Independent critic/AR | passed on validation",
        "centered raw R2",
    ),
    "docs/methods/measurement_contract.md": (
        "nemotron-3-super-v3",
        "76b78d2c34a251f004d53eb5d53766fa01879e2bf3744bc4d80d4fcc1d17825e",
        "not a pristine upstream checkout",
    ),
    "external/natural_language_autoencoders/README.md": (
        "Nano30B production fork",
        "NANO_FORK.md",
    ),
}

FORBIDDEN_ACTIVE_CLAIMS = (
    "primary activation-fidelity proof",
    "Current selected checkpoint",
    "immutable launch snapshot",
)

CLEAN_REGISTRY_STATUSES = {
    "r33-deterministic-snapshot-full275396": (
        "snapshot_replay_passed_fresh_forward_fidelity_failed"
    ),
    "r33-family-clean-primary-ar-sft-20260710": (
        "selected_pair_component_validation_passed"
    ),
    "r33-family-clean-primary-av-sft-20260710": (
        "selected_pair_component_validation_passed"
    ),
    "r33-clean-sft-av-ar-iter1291-20260715": "qualified",
    "r33-clean-sft-publication-followup-20260716": "completed_claim_bounded",
}


def _validate_registry(path: Path, text: str) -> list[str]:
    if SELECTED_HERO_ID not in text:
        return []
    try:
        document = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        return [f"{path}: invalid experiment registry YAML: {exc}"]
    experiments = document.get("experiments") or []
    selected = next(
        (
            experiment
            for experiment in experiments
            if experiment.get("id") == SELECTED_HERO_ID
        ),
        None,
    )
    if selected is None:
        return [f"{path}: selected R33 hero registry entry is missing"]

    mismatches: list[str] = []
    for keys, expected in REGISTRY_INVALIDATION_EXPECTATIONS:
        value = selected
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value != expected:
            mismatches.append(f"{'.'.join(keys)}={expected!r}")
    if not mismatches:
        return []
    return [
        f"{path}: selected R33 hero is missing structured publication "
        f"invalidation fields: {', '.join(mismatches)}"
    ]


def validate_paths(paths: Iterable[Path]) -> list[str]:
    issues: list[str] = []
    for path in paths:
        if not path.is_file():
            issues.append(f"{path}: required claim document is missing")
            continue
        text = path.read_text()
        if path.name == "experiments.yaml":
            issues.extend(_validate_registry(path, text))
            continue
        if LEGACY_HEADLINE_RE.search(text) and PUBLICATION_INVALIDATION_MARKER not in text:
            issues.append(
                f"{path}: legacy R33 hero headline is missing publication "
                "invalidation marker"
            )
    return issues


def validate_repository_contract(root: Path) -> list[str]:
    issues: list[str] = []
    for relative, required_fragments in CANONICAL_REQUIREMENTS.items():
        path = root / relative
        if not path.is_file():
            issues.append(f"{path}: required claim document is missing")
            continue
        text = path.read_text()
        for fragment in required_fragments:
            if fragment not in text:
                issues.append(f"{path}: missing canonical statement {fragment!r}")
        if relative in {
            "README.md",
            "docs/current_state.md",
            "docs/runs/r33_gate_matrix.md",
        }:
            for phrase in FORBIDDEN_ACTIVE_CLAIMS:
                if phrase in text:
                    issues.append(f"{path}: contains stale active claim {phrase!r}")

    registry_path = root / "runs/registry/experiments.yaml"
    if registry_path.is_file():
        document = yaml.safe_load(registry_path.read_text()) or {}
        by_id = {
            item.get("id"): item
            for item in document.get("experiments") or []
            if isinstance(item, dict)
        }
        for run_id, expected_status in CLEAN_REGISTRY_STATUSES.items():
            item = by_id.get(run_id)
            if item is None:
                issues.append(f"{registry_path}: missing clean lineage {run_id}")
            elif item.get("status") != expected_status:
                issues.append(
                    f"{registry_path}: {run_id} status must be "
                    f"{expected_status!r}"
                )
    return issues


def _resolve_paths(values: Sequence[str], *, root: Path) -> list[Path]:
    selected = values or list(DEFAULT_PATHS)
    return [path if path.is_absolute() else root / path for path in map(Path, selected)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="claim documents to validate")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root for default and relative paths",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    issues = validate_paths(_resolve_paths(args.paths, root=root))
    if not args.paths:
        issues.extend(validate_repository_contract(root))
    if issues:
        for issue in issues:
            print(issue)
        return 1
    print("R33 claim documents are internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
