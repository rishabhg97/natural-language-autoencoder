#!/usr/bin/env python3
"""Freeze a verifier-bound Nano NLA AV+AR checkpoint-pair manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


CONFIG_SCHEMA_VERSION = "nano_nla_checkpoint_pair_release.v1"
MANIFEST_SCHEMA_VERSION = "nano_nla_checkpoint_pair_manifest.v1"
PLACEHOLDERS = ("TODO", "TBD", "CHANGEME", "<PLACEHOLDER>")


class PairManifestError(ValueError):
    """Raised when release evidence is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PairManifestError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise PairManifestError(f"{label} must contain a JSON object: {path}")
    return payload


def _lookup(payload: dict[str, Any], dotted_path: str, *, label: str) -> Any:
    value: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise PairManifestError(f"{label} is missing field {dotted_path!r}")
        value = value[key]
    return value


def _require_text(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PairManifestError(f"{label} must be non-empty")
    upper = text.upper()
    if any(marker in upper for marker in PLACEHOLDERS):
        raise PairManifestError(f"{label} contains a placeholder: {text!r}")
    return text


def _artifact_entry(spec: dict[str, Any], *, label: str) -> dict[str, Any]:
    path = Path(_require_text(spec.get("path"), label=f"{label}.path"))
    if not path.is_file():
        raise PairManifestError(f"{label} does not exist: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _checkpoint_entry(spec: dict[str, Any], *, label: str) -> dict[str, Any]:
    checkpoint_path = Path(
        _require_text(spec.get("path"), label=f"checkpoints.{label}.path")
    )
    if not checkpoint_path.exists():
        raise PairManifestError(
            f"checkpoints.{label}.path does not exist: {checkpoint_path}"
        )
    fingerprint_path = Path(
        _require_text(
            spec.get("fingerprint_report"),
            label=f"checkpoints.{label}.fingerprint_report",
        )
    )
    fingerprint = _load_json(
        fingerprint_path,
        label=f"checkpoints.{label}.fingerprint_report",
    )
    fingerprint_field = _require_text(
        spec.get("fingerprint_field", "sha256"),
        label=f"checkpoints.{label}.fingerprint_field",
    )
    observed = _require_text(
        _lookup(
            fingerprint,
            fingerprint_field,
            label=f"checkpoints.{label}.fingerprint_report",
        ),
        label=f"checkpoints.{label}.observed_fingerprint",
    )
    expected = _require_text(
        spec.get("expected_fingerprint"),
        label=f"checkpoints.{label}.expected_fingerprint",
    )
    if observed != expected:
        raise PairManifestError(
            f"checkpoints.{label} fingerprint mismatch: "
            f"expected={expected!r} observed={observed!r}"
        )
    return {
        "path": str(checkpoint_path),
        "fingerprint": observed,
        "fingerprint_field": fingerprint_field,
        "fingerprint_report": {
            "path": str(fingerprint_path),
            "size_bytes": fingerprint_path.stat().st_size,
            "sha256": _sha256(fingerprint_path),
        },
    }


def _evidence_entry(spec: dict[str, Any], *, label: str) -> dict[str, Any]:
    entry = _artifact_entry(spec, label=f"evidence.{label}")
    path = Path(entry["path"])
    report = _load_json(path, label=f"evidence.{label}")
    pass_field = _require_text(
        spec.get("pass_field", "passed"),
        label=f"evidence.{label}.pass_field",
    )
    observed = _lookup(report, pass_field, label=f"evidence.{label}")
    expected = spec.get("pass_value", True)
    if observed != expected:
        raise PairManifestError(
            f"evidence.{label} did not pass: field={pass_field!r} "
            f"expected={expected!r} observed={observed!r}"
        )
    entry.update({"pass_field": pass_field, "pass_value": observed})
    return entry


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise PairManifestError(
            f"schema_version must be {CONFIG_SCHEMA_VERSION!r}"
        )
    release_id = _require_text(config.get("release_id"), label="release_id")
    claim_scope = _require_text(config.get("claim_scope"), label="claim_scope")
    limitations = config.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise PairManifestError("limitations must be a non-empty list")
    limitations = [
        _require_text(value, label=f"limitations[{index}]")
        for index, value in enumerate(limitations)
    ]

    checkpoint_specs = config.get("checkpoints")
    if not isinstance(checkpoint_specs, dict) or set(checkpoint_specs) != {"av", "ar"}:
        raise PairManifestError("checkpoints must contain exactly av and ar")
    checkpoints = {
        label: _checkpoint_entry(spec, label=label)
        for label, spec in sorted(checkpoint_specs.items())
    }

    evidence_specs = config.get("evidence")
    if not isinstance(evidence_specs, dict) or not evidence_specs:
        raise PairManifestError("evidence must be a non-empty mapping")
    evidence = {
        label: _evidence_entry(spec, label=label)
        for label, spec in sorted(evidence_specs.items())
    }

    artifact_specs = config.get("artifacts") or {}
    if not isinstance(artifact_specs, dict):
        raise PairManifestError("artifacts must be a mapping")
    artifacts = {
        label: _artifact_entry(spec, label=f"artifacts.{label}")
        for label, spec in sorted(artifact_specs.items())
    }

    metadata = config.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise PairManifestError("metadata must be a mapping")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "release_id": release_id,
        "qualified": True,
        "claim_scope": claim_scope,
        "limitations": limitations,
        "checkpoints": checkpoints,
        "evidence": evidence,
        "artifacts": artifacts,
        "metadata": metadata,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise PairManifestError("config must contain a YAML mapping")
    manifest = build_manifest(config)
    output = Path(
        _require_text(config.get("output_json"), label="output_json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
