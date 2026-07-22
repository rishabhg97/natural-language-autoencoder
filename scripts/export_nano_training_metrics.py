#!/usr/bin/env python3
"""Export compact, reviewable training curves from Nano Miles text logs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "nano_training_metric_curves.v1"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(r"\b(?:step|perf)\s+(\d+):\s+(\{.*\})\s*$")


class MetricExportError(ValueError):
    """Raised when a complete finite metric curve cannot be recovered."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_curves(
    path: Path,
    *,
    expected_updates: int,
    include_prefixes: tuple[str, ...],
    exclude_metrics: set[str],
) -> list[dict[str, float | int]]:
    if not path.is_file():
        raise MetricExportError(f"train log does not exist: {path}")
    by_step: dict[int, dict[str, float | int]] = {}
    with path.open(errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line).rstrip("\n")
            match = METRIC_RE.search(line)
            if not match:
                continue
            try:
                payload = ast.literal_eval(match.group(2))
            except (SyntaxError, ValueError) as exc:
                raise MetricExportError(
                    f"could not parse metric payload at step {match.group(1)}"
                ) from exc
            if not isinstance(payload, dict):
                raise MetricExportError("metric payload is not a mapping")
            step = int(match.group(1))
            selected = by_step.setdefault(step, {})
            for key, value in payload.items():
                if key in exclude_metrics or not key.startswith(include_prefixes):
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise MetricExportError(
                        f"nonfinite metric {key!r} at step {step} in {path}"
                    )
                selected[str(key)] = int(value) if isinstance(value, int) else numeric

    expected = list(range(expected_updates))
    observed = sorted(by_step)
    if observed != expected:
        final = observed[-1] if observed else None
        raise MetricExportError(
            f"steps do not match 0..{expected_updates - 1}: "
            f"observed={len(observed)}, final={final}"
        )
    return [{"step": step, **dict(sorted(by_step[step].items()))} for step in expected]


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise MetricExportError(
            f"config schema_version must be {SCHEMA_VERSION!r}"
        )
    prefixes = tuple(str(value) for value in config.get("include_prefixes") or [])
    if not prefixes:
        raise MetricExportError("include_prefixes must be non-empty")
    excluded = {str(value) for value in config.get("exclude_metrics") or []}
    raw_runs = config.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise MetricExportError("runs must be a non-empty list")

    runs = []
    for item in raw_runs:
        if not isinstance(item, dict):
            raise MetricExportError("run entries must be mappings")
        path = Path(item["train_log"])
        expected_updates = int(item["expected_updates"])
        runs.append(
            {
                "name": str(item["name"]),
                "component": str(item["component"]),
                "gpu_type": str(item["gpu_type"]),
                "gpu_count": int(item["gpu_count"]),
                "train_log_sha256": _sha256(path),
                "expected_updates": expected_updates,
                "steps": parse_curves(
                    path,
                    expected_updates=expected_updates,
                    include_prefixes=prefixes,
                    exclude_metrics=excluded,
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_semantics": (
            "Rank-aggregated metrics recovered from the immutable Miles text "
            "logs. System samples are logger-time observations, not profiler traces."
        ),
        "include_prefixes": list(prefixes),
        "exclude_metrics": sorted(excluded),
        "runs": runs,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise MetricExportError("config root must be a mapping")
    report = build_report(config)
    _write_json_atomic(Path(config["output_json"]), report)
    print(json.dumps({
        "schema_version": report["schema_version"],
        "runs": [
            {"name": run["name"], "steps": len(run["steps"])}
            for run in report["runs"]
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
