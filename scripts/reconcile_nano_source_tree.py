#!/usr/bin/env python3
"""Verify and reconcile launch-critical source roots against a signed manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_source_provenance import fingerprint_source_tree, sha256_file, source_file_manifest  # noqa: E402


class SourceReconcileError(ValueError):
    """Raised when expected source files are missing or modified."""


def reconcile_source_tree(
    *,
    code_root: str | Path,
    manifest: dict[str, Any],
    apply: bool = False,
) -> dict[str, Any]:
    root = Path(code_root).resolve()
    expected_fingerprint = manifest.get("fingerprint") or {}
    roots = tuple(expected_fingerprint.get("roots") or ())
    if not roots:
        raise SourceReconcileError("source manifest has no roots")
    expected_files = {
        str(item["path"]): item
        for item in manifest.get("files") or []
        if isinstance(item, dict) and item.get("path")
    }
    if not expected_files:
        raise SourceReconcileError("source manifest has no files")

    missing: list[str] = []
    mismatched: list[dict[str, Any]] = []
    for relative, expected in expected_files.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected.get("sha256"):
            mismatched.append(
                {"path": relative, "expected": expected.get("sha256"), "actual": actual_hash}
            )
    if missing or mismatched:
        raise SourceReconcileError(
            f"source tree does not contain the manifest payload: missing={missing[:10]} mismatched={mismatched[:10]}"
        )

    actual_manifest = source_file_manifest(root, roots=roots)
    actual_paths = {str(item["path"]) for item in actual_manifest["files"]}
    extras = sorted(actual_paths - set(expected_files))
    deleted: list[str] = []
    if apply:
        for relative in extras:
            path = root / relative
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                raise SourceReconcileError(f"refusing to delete unsafe source path: {path}")
            path.unlink()
            deleted.append(relative)
        for relative_root in roots:
            for directory in sorted((root / relative_root).rglob("*"), reverse=True):
                if directory.is_dir() and not directory.is_symlink():
                    try:
                        directory.rmdir()
                    except OSError:
                        pass

    final = fingerprint_source_tree(root, roots=roots)
    expected_hash = expected_fingerprint.get("sha256")
    matched = final["sha256"] == expected_hash
    if apply and not matched:
        raise SourceReconcileError(
            f"source fingerprint still mismatches after reconciliation: expected={expected_hash} actual={final['sha256']}"
        )
    return {
        "schema_version": "nano_source_reconcile.v1",
        "apply": bool(apply),
        "code_root": str(root),
        "expected_sha256": expected_hash,
        "actual_sha256": final["sha256"],
        "matched": matched,
        "extra_files": extras,
        "deleted_files": deleted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest_json.read_text())
    report = reconcile_source_tree(code_root=args.code_root, manifest=manifest, apply=args.apply)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
