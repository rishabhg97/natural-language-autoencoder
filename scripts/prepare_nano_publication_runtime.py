#!/usr/bin/env python3
"""Freeze model and runtime provenance required by publication extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_source_provenance import (  # noqa: E402
    DEFAULT_SOURCE_ROOTS,
    SourceProvenanceError,
    collect_source_provenance,
    verify_source_policy,
    write_provenance,
)


SCHEMA_VERSION = "nano_publication_runtime.v1"
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PublicationRuntimeError(ValueError):
    """Raised when the publication runtime cannot be frozen safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path, *, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise PublicationRuntimeError(
            f"config must use schema_version {SCHEMA_VERSION}"
        )
    paths = config.get("paths") or {}
    for key in (
        "code_root",
        "miles_root",
        "miles_patches_root",
        "model_root",
        "model_fingerprint_source",
    ):
        if not paths.get(key):
            raise PublicationRuntimeError(f"paths.{key} is required")
    outputs = config.get("outputs") or {}
    for key in ("model_fingerprint_json", "runtime_provenance_json"):
        if not outputs.get(key):
            raise PublicationRuntimeError(f"outputs.{key} is required")
    expected_model = str(config.get("expected_model_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_model):
        raise PublicationRuntimeError("expected_model_sha256 must be a SHA-256 hex digest")
    image_digest = str(config.get("container_image_digest") or "")
    if not IMAGE_DIGEST_RE.fullmatch(image_digest):
        raise PublicationRuntimeError(
            "container_image_digest must be an immutable sha256 digest"
        )
    critical = config.get("critical_files") or {}
    if not isinstance(critical, dict) or not critical:
        raise PublicationRuntimeError("critical_files must be a non-empty mapping")
    return config


def run_prepare(config_path: str | Path, *, force: bool = False) -> dict[str, Any]:
    resolved_config = Path(config_path).resolve()
    config = _load_config(resolved_config)
    paths = config["paths"]
    outputs = config["outputs"]
    model_output = _resolve(outputs["model_fingerprint_json"], config_path=resolved_config)
    runtime_output = _resolve(outputs["runtime_provenance_json"], config_path=resolved_config)
    if not force and (model_output.exists() or runtime_output.exists()):
        raise PublicationRuntimeError(
            f"publication runtime output already exists: {model_output} or {runtime_output}"
        )

    model_root = _resolve(paths["model_root"], config_path=resolved_config).resolve()
    model_source = _resolve(
        paths["model_fingerprint_source"], config_path=resolved_config
    ).resolve()
    if not model_root.is_dir() or not model_source.is_file():
        raise PublicationRuntimeError("model root or model fingerprint source is missing")
    model_report = json.loads(model_source.read_text())
    if Path(str(model_report.get("root") or "")).resolve() != model_root:
        raise PublicationRuntimeError("model fingerprint root does not match model_root")
    if model_report.get("sha256") != config["expected_model_sha256"]:
        raise PublicationRuntimeError("model fingerprint does not match expected_model_sha256")
    if int(model_report.get("file_count") or 0) <= 0:
        raise PublicationRuntimeError("model fingerprint has no files")

    critical_files = {
        str(name): _resolve(value, config_path=resolved_config)
        for name, value in config["critical_files"].items()
    }
    critical_files["publication_runtime_config"] = resolved_config
    try:
        provenance = collect_source_provenance(
            _resolve(paths["code_root"], config_path=resolved_config),
            roots=tuple(config.get("source_roots") or DEFAULT_SOURCE_ROOTS),
            miles_root=_resolve(paths["miles_root"], config_path=resolved_config),
            miles_patches_root=_resolve(
                paths["miles_patches_root"], config_path=resolved_config
            ),
            critical_files=critical_files,
            container_image_digest=str(config["container_image_digest"]),
            python_executable=sys.executable,
        )
        verify_source_policy({"require_complete_runtime": True}, provenance)
    except SourceProvenanceError as exc:
        raise PublicationRuntimeError(str(exc)) from exc

    normalized_model_report = {
        **model_report,
        "root": str(model_root),
        "source_report": {
            "path": str(model_source),
            "sha256": _sha256_file(model_source),
        },
    }
    provenance["publication_runtime"] = {
        "config": str(resolved_config),
        "config_sha256": _sha256_file(resolved_config),
        "model_fingerprint_source": str(model_source),
        "model_fingerprint_sha256": str(model_report["sha256"]),
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(
        json.dumps(normalized_model_report, indent=2, sort_keys=True) + "\n"
    )
    write_provenance(runtime_output, provenance)
    return {
        "schema_version": "nano_publication_runtime_report.v1",
        "passed": True,
        "model_fingerprint_json": str(model_output),
        "model_sha256": str(model_report["sha256"]),
        "runtime_provenance_json": str(runtime_output),
        "runtime_sha256": str(provenance["runtime"]["sha256"]),
        "source_sha256": str(provenance["source"]["sha256"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    report = run_prepare(args.config, force=args.force)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
