#!/usr/bin/env python3
"""Audit an exact Nano release bundle without printing matched secret values."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.audit_nano_release_text import SECRET_PATTERNS
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from audit_nano_release_text import SECRET_PATTERNS


SCHEMA_VERSION = "nano_release_bundle_audit.v1"
INTERNAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("local_home_path", re.compile(r"/Users/[A-Za-z0-9._-]+/")),
    ("runai_workspace_path", re.compile(r"/workspace/(?:interp|models)/")),
    ("internal_s3", re.compile(r"s3://team-ipp-[^\s'\"]+", re.I)),
    (
        "cluster_hostname",
        re.compile(r"\b[A-Za-z0-9.-]+\.ipp[0-9a-z.]*\.nvidia\.com\b", re.I),
    ),
    ("internal_s3_endpoint", re.compile(r"\bpdx\.s8k\.io\b", re.I)),
)

DEFAULT_FORBIDDEN_GLOBS = (
    ".env",
    ".env.*",
    "credentials",
    "*.pem",
    "*.key",
    "*.parquet",
    "*.arrow",
    "*.safetensors",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.bin",
    "*.npy",
    "*.npz",
    "*.wandb",
    "*.docx",
    "**/.env",
    "**/.env.*",
    "**/credentials",
    "**/*.pem",
    "**/*.key",
    "**/*.parquet",
    "**/*.arrow",
    "**/*.safetensors",
    "**/*.pt",
    "**/*.pth",
    "**/*.ckpt",
    "**/*.bin",
    "**/*.npy",
    "**/*.npz",
    "**/*.wandb",
    "**/*.docx",
)


class ReleaseBundleAuditError(ValueError):
    """Raised when a bundle audit cannot be performed deterministically."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches(path: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finding_allowed(
    *, path: str, kind: str, count: int, rules: list[dict[str, Any]]
) -> bool:
    for rule in rules:
        if not fnmatch.fnmatch(path, str(rule["path_glob"])):
            continue
        if kind not in set(rule.get("kinds", [])):
            continue
        if count > int(rule.get("max_count", 0)):
            continue
        return True
    return False


def audit_bundle(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseBundleAuditError(
            f"config schema_version must be {SCHEMA_VERSION!r}"
        )
    root = Path(config["root"])
    if root.is_symlink() or not root.is_dir():
        raise ReleaseBundleAuditError("root must be an existing non-symlink directory")
    root = root.resolve()

    exclude_globs = list(config.get("exclude_globs", []))
    forbidden_globs = list(
        config.get("forbidden_globs", DEFAULT_FORBIDDEN_GLOBS)
    )
    fail_pattern_kinds = set(
        config.get(
            "fail_pattern_kinds",
            [name for name, _ in SECRET_PATTERNS]
            + [name for name, _ in INTERNAL_PATTERNS],
        )
    )
    allowed_findings = list(config.get("allowed_findings", []))
    for rule in allowed_findings:
        if not isinstance(rule, dict) or not rule.get("path_glob"):
            raise ReleaseBundleAuditError(
                "each allowed_findings entry requires path_glob"
            )
        if int(rule.get("max_count", 0)) < 1:
            raise ReleaseBundleAuditError(
                "each allowed_findings entry requires positive max_count"
            )
    max_text_bytes = int(config.get("max_text_bytes", 5 * 1024 * 1024))
    if max_text_bytes <= 0:
        raise ReleaseBundleAuditError("max_text_bytes must be positive")

    files: list[dict[str, Any]] = []
    forbidden_paths: list[dict[str, str]] = []
    findings: list[dict[str, Any]] = []
    symlinks: list[str] = []
    binary_files: list[str] = []
    oversized_text_files: list[str] = []

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if _matches(relative, exclude_globs):
            continue
        if path.is_symlink():
            symlinks.append(relative)
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        digest = _sha256(path)
        files.append({"path": relative, "bytes": size, "sha256": digest})

        matched_globs = [
            pattern for pattern in forbidden_globs if fnmatch.fnmatch(relative, pattern)
        ]
        if matched_globs:
            forbidden_paths.append(
                {"path": relative, "matched_glob": matched_globs[0]}
            )

        if size > max_text_bytes:
            oversized_text_files.append(relative)
            continue
        raw = path.read_bytes()
        if b"\x00" in raw:
            binary_files.append(relative)
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            binary_files.append(relative)
            continue

        counts: Counter[str] = Counter()
        for kind, pattern in (*SECRET_PATTERNS, *INTERNAL_PATTERNS):
            count = sum(1 for _ in pattern.finditer(text))
            if count:
                counts[kind] += count
        if counts:
            allowed_kinds = sorted(
                kind
                for kind, count in counts.items()
                if _finding_allowed(
                    path=relative,
                    kind=kind,
                    count=count,
                    rules=allowed_findings,
                )
            )
            findings.append(
                {
                    "path": relative,
                    "finding_counts": dict(sorted(counts.items())),
                    "allowed_kinds": allowed_kinds,
                    "failed_kinds": sorted(
                        (set(counts) & fail_pattern_kinds) - set(allowed_kinds)
                    ),
                }
            )

    manifest_payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    failed_finding_files = [item for item in findings if item["failed_kinds"]]
    passed = not (
        forbidden_paths
        or symlinks
        or binary_files
        or oversized_text_files
        or failed_finding_files
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "automatic_gate_passed": passed,
        "claim_boundary": (
            "Static pattern and file-type triage only. A passing report does "
            "not establish legal clearance, privacy, vulnerability absence, "
            "or safe behavior of released code or models."
        ),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "tree_manifest_sha256": hashlib.sha256(
            manifest_payload.encode("utf-8")
        ).hexdigest(),
        "forbidden_paths": forbidden_paths,
        "symlinks": symlinks,
        "binary_files": binary_files,
        "oversized_text_files": oversized_text_files,
        "finding_files": findings,
        "failed_finding_files": failed_finding_files,
        "fail_pattern_kinds": sorted(fail_pattern_kinds),
        "allowed_findings": allowed_findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ReleaseBundleAuditError("config root must be a mapping")
    report = audit_bundle(config)
    _write_json_atomic(Path(config["output_json"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["automatic_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
