#!/usr/bin/env python3
"""Build a deterministic, redacted Nano NLA public-release candidate tree."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml


SCHEMA_VERSION = "nano_public_release_bundle.v1"
MANIFEST_NAME = "bundle_manifest.json"
REDACTIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "local_home",
        re.compile(r"/Users/[A-Za-z0-9._-]+/"),
        "${NANO_LOCAL_HOME}/",
    ),
    (
        "interp_root",
        re.compile(r"/workspace/interp/"),
        "${NANO_INTERP_ROOT}/",
    ),
    (
        "model_root",
        re.compile(r"/workspace/models/"),
        "${NANO_MODEL_ROOT}/",
    ),
    (
        "internal_s3_root",
        re.compile(
            r"s3://team-ipp-[A-Za-z0-9._-]+/(?:nano30b-nla-pilot/)?",
            re.IGNORECASE,
        ),
        "${NANO_INTERNAL_S3_ROOT}/",
    ),
    (
        "cluster_hostname",
        re.compile(
            r"\b[A-Za-z0-9.-]+\.ipp[0-9a-z.]*\.nvidia\.com\b",
            re.IGNORECASE,
        ),
        "${NANO_CLUSTER_HOST}",
    ),
    (
        "internal_s3_endpoint",
        re.compile(r"\bpdx\.s8k\.io\b", re.IGNORECASE),
        "${NANO_S3_ENDPOINT}",
    ),
)


class ReleaseBundleError(ValueError):
    """Raised when a release candidate cannot be staged safely."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ReleaseBundleError(f"unsafe bundle destination: {value!r}")
    return path.as_posix()


def _redact(text: str) -> tuple[str, dict[str, int]]:
    counts: Counter[str] = Counter()
    for kind, pattern, replacement in REDACTIONS:
        text, count = pattern.subn(replacement, text)
        if count:
            counts[kind] += count
    return text, dict(sorted(counts.items()))


def _read_text(path: Path, *, max_source_bytes: int) -> str:
    size = path.stat().st_size
    if size > max_source_bytes:
        raise ReleaseBundleError(
            f"source file exceeds max_source_bytes={max_source_bytes}: {path}"
        )
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise ReleaseBundleError(f"binary source file is not permitted: {path}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseBundleError(f"non-UTF-8 source file is not permitted: {path}") from exc


def _selected_source_files(
    root: Path,
    *,
    include_globs: list[str],
    exclude_globs: list[str],
) -> list[tuple[Path, str]]:
    selected: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not _matches(relative, include_globs):
            continue
        if _matches(relative, exclude_globs):
            continue
        if path.is_symlink():
            raise ReleaseBundleError(f"symlink is not permitted: {relative}")
        selected.append((path, relative))
    return selected


def _stage_file(
    source: Path,
    destination: Path,
    *,
    relative: str,
    max_source_bytes: int,
) -> dict[str, Any]:
    original = _read_text(source, max_source_bytes=max_source_bytes)
    redacted, redactions = _redact(original)
    encoded = redacted.encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    os.chmod(destination, source.stat().st_mode & 0o777)
    return {
        "path": relative,
        "bytes": len(encoded),
        "source_sha256": _sha256_bytes(original.encode("utf-8")),
        "staged_sha256": _sha256_bytes(encoded),
        "redactions": redactions,
    }


def build_bundle(config: dict[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseBundleError(
            f"config schema_version must be {SCHEMA_VERSION!r}"
        )
    source_root = Path(config["source_root"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    if not source_root.is_dir():
        raise ReleaseBundleError(f"source_root is not a directory: {source_root}")
    if output_root == source_root:
        raise ReleaseBundleError("output_root must not equal source_root")

    include_globs = list(config.get("include_globs") or [])
    exclude_globs = list(config.get("exclude_globs") or [])
    if not include_globs:
        raise ReleaseBundleError("include_globs must be non-empty")
    source_prefix_value = str(config.get("source_prefix", "")).strip("/")
    source_prefix = _safe_relative(source_prefix_value) if source_prefix_value else ""
    max_source_bytes = int(config.get("max_source_bytes", 32 * 1024 * 1024))
    if max_source_bytes <= 0:
        raise ReleaseBundleError("max_source_bytes must be positive")

    temporary = output_root.with_name(output_root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    if output_root.exists() and not config.get("replace_existing", False):
        raise ReleaseBundleError(f"output_root already exists: {output_root}")
    temporary.mkdir(parents=True)

    entries: list[dict[str, Any]] = []
    destinations: set[str] = set()
    try:
        for source, relative in _selected_source_files(
            source_root,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        ):
            bundle_relative = (
                f"{source_prefix}/{relative}" if source_prefix else relative
            )
            destinations.add(bundle_relative)
            entries.append(
                _stage_file(
                    source,
                    temporary / bundle_relative,
                    relative=bundle_relative,
                    max_source_bytes=max_source_bytes,
                )
            )

        for item in config.get("extra_files") or []:
            if not isinstance(item, dict):
                raise ReleaseBundleError("extra_files entries must be mappings")
            source = Path(item["source"])
            if not source.is_absolute():
                source = source_root / source
            source = source.resolve()
            if not source.is_file() or source.is_symlink():
                raise ReleaseBundleError(f"extra source is not a regular file: {source}")
            relative = _safe_relative(str(item["destination"]))
            if relative in destinations:
                raise ReleaseBundleError(f"duplicate bundle destination: {relative}")
            destinations.add(relative)
            entries.append(
                _stage_file(
                    source,
                    temporary / relative,
                    relative=relative,
                    max_source_bytes=max_source_bytes,
                )
            )

        entries.sort(key=lambda item: item["path"])
        tree_payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
        redaction_totals: Counter[str] = Counter()
        for entry in entries:
            redaction_totals.update(entry["redactions"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "release_id": str(config["release_id"]),
            "claim_boundary": str(config["claim_boundary"]),
            "weights_included": False,
            "legal_clearance_granted": False,
            "file_count": len(entries),
            "total_bytes": sum(item["bytes"] for item in entries),
            "tree_manifest_sha256": _sha256_bytes(tree_payload.encode("utf-8")),
            "config_sha256": _sha256(config_path) if config_path else None,
            "redaction_totals": dict(sorted(redaction_totals.items())),
            "files": entries,
        }
        (temporary / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(temporary, output_root)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ReleaseBundleError("config root must be a mapping")
    manifest = build_bundle(config, config_path=config_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
