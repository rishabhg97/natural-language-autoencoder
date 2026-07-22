#!/usr/bin/env python3
"""Shared contracts for deterministic Observatory derivation and bundling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .common import ObservatoryConfigError, canonical_json


BUNDLE_CONFIG_SCHEMA = "nano_viz_bundle_config.v1"


def load_bundle_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != BUNDLE_CONFIG_SCHEMA:
        raise ObservatoryConfigError(
            f"bundle config schema_version must be {BUNDLE_CONFIG_SCHEMA!r}"
        )
    for section in ("paths", "geometry", "statistics", "verification"):
        if not isinstance(value.get(section), dict):
            raise ObservatoryConfigError(f"bundle config {section} must be a mapping")
    source = Path(str(value.get("source_config") or ""))
    if not source.is_absolute():
        source = (config_path.resolve().parent / source).resolve()
    value["source_config"] = str(source)
    return value


def bundle_config_fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def bundle_path(value: str | Path, *, config_path: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


def write_parquet_atomic(path: str | Path, rows: list[dict[str, Any]], schema: Any) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    temporary.replace(target)


def family_bootstrap_interval(
    values: Iterable[float],
    family_ids: Iterable[str],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    values_array = np.asarray(list(values), dtype=np.float64)
    families_array = np.asarray(list(family_ids), dtype=np.str_)
    if values_array.ndim != 1 or len(values_array) != len(families_array):
        raise ObservatoryConfigError("values and family_ids must be aligned vectors")
    if not len(values_array) or not np.isfinite(values_array).all():
        raise ObservatoryConfigError("bootstrap values must be non-empty and finite")
    unique = np.unique(families_array)
    family_means = np.asarray(
        [values_array[families_array == family].mean() for family in unique],
        dtype=np.float64,
    )
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise ObservatoryConfigError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    draws = family_means[
        rng.integers(0, len(family_means), size=(samples, len(family_means)))
    ].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(family_means.mean()),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "rows": int(len(values_array)),
        "families": int(len(unique)),
        "bootstrap_samples": int(samples),
    }


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ObservatoryConfigError(f"expected JSON object: {path}")
    return value
