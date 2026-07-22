#!/usr/bin/env python3
"""Package an audited Nano public-release candidate as a deterministic archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


SCHEMA_VERSION = "nano_public_release_archive.v1"


class ReleaseArchiveError(ValueError):
    """Raised when an archive cannot be bound to a passing tree audit."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReleaseArchiveError(f"symlink is not permitted: {path}")
        if not path.is_file():
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries


def _tree_hash(entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _archive_entries(path: Path, *, archive_root_name: str) -> list[dict[str, Any]]:
    prefix = archive_root_name.rstrip("/") + "/"
    entries: list[dict[str, Any]] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise ReleaseArchiveError(
                    f"archive contains non-regular member: {member.name}"
                )
            if not member.name.startswith(prefix):
                raise ReleaseArchiveError(
                    f"archive member is outside root {archive_root_name!r}: "
                    f"{member.name}"
                )
            relative = member.name[len(prefix) :]
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or not relative:
                raise ReleaseArchiveError(f"unsafe archive member: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ReleaseArchiveError(f"could not read archive member: {member.name}")
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            entries.append(
                {"path": pure.as_posix(), "bytes": size, "sha256": digest.hexdigest()}
            )
    return sorted(entries, key=lambda item: item["path"])


def package_bundle(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseArchiveError(
            f"config schema_version must be {SCHEMA_VERSION!r}"
        )
    root = Path(config["bundle_root"]).resolve()
    audit_path = Path(config["audit_report"]).resolve()
    output = Path(config["output_archive"]).resolve()
    attestation_path = Path(config["attestation_json"]).resolve()
    archive_root_name = str(config["archive_root_name"]).strip("/")
    if not root.is_dir() or not audit_path.is_file():
        raise ReleaseArchiveError("bundle root and audit report must exist")
    if not archive_root_name or ".." in PurePosixPath(archive_root_name).parts:
        raise ReleaseArchiveError("archive_root_name is unsafe")

    audit = json.loads(audit_path.read_text())
    if audit.get("automatic_gate_passed") is not True:
        raise ReleaseArchiveError("bundle security audit did not pass")
    if Path(audit.get("root", "")).resolve() != root:
        raise ReleaseArchiveError("audit report root does not match bundle_root")

    entries = _tree_entries(root)
    tree_sha = _tree_hash(entries)
    if tree_sha != audit.get("tree_manifest_sha256"):
        raise ReleaseArchiveError(
            "bundle tree changed after audit: "
            f"expected {audit.get('tree_manifest_sha256')}, got {tree_sha}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for entry in entries:
                    source = root / entry["path"]
                    info = archive.gettarinfo(
                        str(source),
                        arcname=f"{archive_root_name}/{entry['path']}",
                    )
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)
    os.replace(temporary, output)

    archived_entries = _archive_entries(output, archive_root_name=archive_root_name)
    archived_tree_sha = _tree_hash(archived_entries)
    if archived_entries != entries or archived_tree_sha != tree_sha:
        raise ReleaseArchiveError("archive content does not match audited tree")

    attestation = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_id": str(config["release_id"]),
        "archive_root_name": archive_root_name,
        "archive_sha256": _sha256(output),
        "archive_bytes": output.stat().st_size,
        "file_count": len(entries),
        "audited_tree_manifest_sha256": tree_sha,
        "archive_tree_manifest_sha256": archived_tree_sha,
        "audit_report_sha256": _sha256(audit_path),
        "security_audit_passed": True,
        "weights_included": False,
        "legal_clearance_granted": False,
    }
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_attestation = attestation_path.with_suffix(
        attestation_path.suffix + ".tmp"
    )
    temporary_attestation.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_attestation, attestation_path)
    return attestation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ReleaseArchiveError("config root must be a mapping")
    attestation = package_bundle(config)
    print(json.dumps(attestation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
