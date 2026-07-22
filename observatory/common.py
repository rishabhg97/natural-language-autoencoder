#!/usr/bin/env python3
"""Shared configuration, hashing, and serialization helpers for NLA Observatory jobs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import yaml


CONFIG_SCHEMA = "nano_viz_offline_observatory.v1"


class ObservatoryConfigError(ValueError):
    """Raised when an Observatory config or artifact violates its contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(seed: int, *parts: Any) -> int:
    payload = canonical_json([int(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def resolve_path(value: str | Path, *, config_path: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return path
    return Path(config_path).resolve().parent / path


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ObservatoryConfigError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text())
    config = _require_mapping(config, name="config")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ObservatoryConfigError(
            f"config schema_version must be {CONFIG_SCHEMA!r}"
        )
    for section in ("paths", "selection", "grid", "evaluation", "gates"):
        _require_mapping(config.get(section), name=section)
    selection = config["selection"]
    expected = {
        "deep_dive_rows": 50,
        "behavior_rows": 24,
        "canary_rows": 8,
    }
    for key, exact in expected.items():
        observed = int(selection.get(key, -1))
        if observed != exact:
            raise ObservatoryConfigError(
                f"selection.{key} must be {exact}, got {observed}"
            )
    film_rows = int(selection.get("film_rows", 0))
    if not 8 <= film_rows <= 12:
        raise ObservatoryConfigError("selection.film_rows must be between 8 and 12")
    if int(selection.get("seed", -1)) < 0:
        raise ObservatoryConfigError("selection.seed must be non-negative")
    return config


def config_fingerprint(config: dict[str, Any]) -> str:
    return sha256_json(config)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    count = 0
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    temporary.replace(target)
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ObservatoryConfigError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            rows.append(value)
    return rows


def git_revision(root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
