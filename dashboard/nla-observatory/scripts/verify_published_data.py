#!/usr/bin/env python3
"""Verify the committed, checkpoint-free dashboard bundle before publication."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


SCHEMA = "nla_observatory_dashboard.v2"
EXPECTED_PROTOCOL = "fcc431ec4450adb8817cd946d6c194fa2a45b53b0c6c42c8682c1e9f12f94d4d"
EXPECTED_CONTROLS = {"av_shuffled", "av_zero", "av_mean", "av_none"}


class VerifyError(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def load(path: Path) -> dict:
    check(path.is_file(), f"missing file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerifyError(f"invalid JSON: {path}: {exc}") from exc
    check(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_numbers(value: object, context: str) -> None:
    if isinstance(value, float):
        check(math.isfinite(value), f"non-finite number in {context}")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_numbers(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_numbers(child, f"{context}[{index}]")


def verify(data_dir: Path) -> None:
    manifest = load(data_dir / "manifest.json")
    check(manifest.get("schema_version") == SCHEMA, "manifest schema mismatch")

    entries = manifest.get("files")
    check(isinstance(entries, list) and entries, "manifest has no files")
    listed: set[str] = set()
    for entry in entries:
        check(isinstance(entry, dict), "invalid manifest file entry")
        relative = entry.get("path")
        check(isinstance(relative, str) and relative, "manifest file path is invalid")
        check(relative not in listed, f"duplicate manifest path: {relative}")
        listed.add(relative)
        target = data_dir / relative
        check(target.is_file(), f"manifest file missing: {relative}")
        check(target.stat().st_size == entry.get("bytes"), f"size mismatch: {relative}")
        check(sha256(target) == entry.get("sha256"), f"hash mismatch: {relative}")

    present = {
        str(path.relative_to(data_dir))
        for path in data_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    check(present == listed, f"unlisted or missing public files: {present ^ listed}")

    channel = load(data_dir / "channel.json")
    result = channel.get("matched_online_rl")
    check(isinstance(result, dict), "matched online-RL result is missing")
    check(result.get("status") == "validation_only_matched", "RL scope changed")
    check(result.get("row_count") == 122, "RL row count changed")
    check(result.get("independent_family_count") == 122, "RL family count changed")
    check(result.get("max_new_tokens") == 384, "RL generation budget changed")
    check(result.get("generation_protocol_sha256") == EXPECTED_PROTOCOL,
          "RL protocol hash changed")
    check(result["sft"].get("roundtrip_nmse") == 0.309055, "SFT headline changed")
    check(result["rl"].get("roundtrip_nmse") == 0.224386, "RL headline changed")
    check(result["rl"]["roundtrip_nmse"] < result["sft"]["roundtrip_nmse"],
          "RL no longer improves on SFT")
    check({control.get("key") for control in result["rl"].get("controls", [])}
          == EXPECTED_CONTROLS, "RL controls are incomplete")

    audit = load(data_dir / "audit.json")
    claims = audit.get("claim_ledger", {}).get("claims", {})
    check(claims.get("matched_online_rl_roundtrip") == "validation_only_confirmatory",
          "RL claim boundary changed")

    for path in data_dir.rglob("*.json"):
        finite_numbers(load(path), str(path.relative_to(data_dir)))


def main() -> int:
    data_dir = Path(__file__).resolve().parents[1] / "public" / "data"
    try:
        verify(data_dir)
    except VerifyError as exc:
        print(f"PUBLISHED DATA VERIFY FAILED: {exc}", file=sys.stderr)
        return 1
    print("VERIFIED: committed dashboard data is complete, finite, and hash-bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

