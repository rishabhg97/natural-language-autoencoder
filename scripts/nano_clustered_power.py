#!/usr/bin/env python3
"""Estimate confirmatory power from paired validation-family residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


class ClusteredPowerError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_payload(report: dict[str, Any], split: str) -> dict[str, Any]:
    payload = (report.get("splits") or {}).get(split)
    if not isinstance(payload, dict):
        raise ClusteredPowerError(f"report has no {split!r} split")
    return payload


def _row_identity(payload: dict[str, Any]) -> list[Any]:
    identities = payload.get("row_keys") or payload.get("row_indices")
    if not isinstance(identities, list) or not identities:
        raise ClusteredPowerError("split has no row_keys or row_indices")
    return identities


def paired_family_deltas(
    candidate_report: dict[str, Any],
    baseline_report: dict[str, Any],
    *,
    split: str,
    variant: str,
) -> tuple[np.ndarray, float, int]:
    candidate = _split_payload(candidate_report, split)
    baseline = _split_payload(baseline_report, split)
    if _row_identity(candidate) != _row_identity(baseline):
        raise ClusteredPowerError("candidate and baseline row identities differ")
    candidate_families = candidate.get("content_family_ids")
    baseline_families = baseline.get("content_family_ids")
    if candidate_families != baseline_families or not candidate_families:
        raise ClusteredPowerError("candidate and baseline family identities differ")
    candidate_losses = (candidate.get("rowwise_directional_mse") or {}).get(variant)
    baseline_losses = (baseline.get("rowwise_directional_mse") or {}).get(variant)
    if not isinstance(candidate_losses, list) or not isinstance(baseline_losses, list):
        raise ClusteredPowerError(
            f"both reports require rowwise_directional_mse.{variant}"
        )
    row_count = len(candidate_families)
    if len(candidate_losses) != row_count or len(baseline_losses) != row_count:
        raise ClusteredPowerError("rowwise loss and family lengths differ")
    grouped: dict[str, list[float]] = defaultdict(list)
    for family_id, candidate_loss, baseline_loss in zip(
        candidate_families,
        candidate_losses,
        baseline_losses,
    ):
        delta = float(baseline_loss) - float(candidate_loss)
        if not math.isfinite(delta):
            raise ClusteredPowerError("paired losses contain nonfinite values")
        grouped[str(family_id)].append(delta)
    family_deltas = np.asarray(
        [float(np.mean(grouped[key], dtype=np.float64)) for key in sorted(grouped)],
        dtype=np.float64,
    )
    baseline_mean = float(np.mean(np.asarray(baseline_losses, dtype=np.float64)))
    if len(family_deltas) < 2 or baseline_mean <= 0.0:
        raise ClusteredPowerError(
            "power estimation requires at least two families and positive baseline loss"
        )
    return family_deltas, baseline_mean, row_count


def simulate_power(
    family_deltas: np.ndarray,
    *,
    target_absolute_gain: float,
    registered_family_count: int,
    simulations: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if registered_family_count < 2 or simulations <= 0:
        raise ClusteredPowerError(
            "registered_family_count must be at least 2 and simulations positive"
        )
    if target_absolute_gain <= 0.0 or not 0.0 < confidence < 1.0:
        raise ClusteredPowerError("target gain and confidence must be valid")
    centered = family_deltas - float(np.mean(family_deltas, dtype=np.float64))
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        centered,
        size=(simulations, registered_family_count),
        replace=True,
    )
    draws = draws + target_absolute_gain
    means = np.mean(draws, axis=1, dtype=np.float64)
    stds = np.std(draws, axis=1, ddof=1, dtype=np.float64)
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    lower_bounds = means - z_value * stds / math.sqrt(registered_family_count)
    power = float(np.mean(lower_bounds > 0.0, dtype=np.float64))
    observed_sigma = float(np.std(centered, ddof=1, dtype=np.float64))
    required_families = math.ceil(
        ((z_value + NormalDist().inv_cdf(0.80)) * observed_sigma / target_absolute_gain)
        ** 2
    )
    return {
        "power": power,
        "confidence": confidence,
        "simulations": simulations,
        "seed": seed,
        "registered_family_count": registered_family_count,
        "observed_family_sigma": observed_sigma,
        "normal_approx_required_families_for_80pct_power": max(2, required_families),
    }


def build_report(
    candidate_path: Path,
    baseline_path: Path,
    *,
    split: str,
    variant: str,
    target_relative_gain: float,
    registered_family_count: int | None,
    simulations: int,
    seed: int,
    min_power: float,
) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    family_deltas, baseline_mean, row_count = paired_family_deltas(
        candidate,
        baseline,
        split=split,
        variant=variant,
    )
    family_count = int(registered_family_count or len(family_deltas))
    target_absolute_gain = baseline_mean * target_relative_gain
    simulation = simulate_power(
        family_deltas,
        target_absolute_gain=target_absolute_gain,
        registered_family_count=family_count,
        simulations=simulations,
        seed=seed,
    )
    return {
        "schema_version": "nano_clustered_power.v1",
        "passed": simulation["power"] >= min_power,
        "split": split,
        "variant": variant,
        "candidate_report": str(candidate_path.resolve()),
        "candidate_report_sha256": _sha256(candidate_path),
        "baseline_report": str(baseline_path.resolve()),
        "baseline_report_sha256": _sha256(baseline_path),
        "row_count": row_count,
        "pilot_family_count": len(family_deltas),
        "baseline_directional_mse": baseline_mean,
        "target_relative_gain": target_relative_gain,
        "target_absolute_gain": target_absolute_gain,
        "minimum_power": min_power,
        **simulation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--variant", default="av_real")
    parser.add_argument("--target-relative-gain", type=float, default=0.10)
    parser.add_argument("--registered-family-count", type=int)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--min-power", type=float, default=0.80)
    args = parser.parse_args(argv)
    report = build_report(
        args.candidate_report,
        args.baseline_report,
        split=args.split,
        variant=args.variant,
        target_relative_gain=args.target_relative_gain,
        registered_family_count=args.registered_family_count,
        simulations=args.simulations,
        seed=args.seed,
        min_power=args.min_power,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
