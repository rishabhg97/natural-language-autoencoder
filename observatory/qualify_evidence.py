#!/usr/bin/env python3
"""Qualify cached R33 publication evidence for the offline NLA Observatory."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from nano_eval_core import activation_reconstruction_metrics  # noqa: E402
from .common import (
    ObservatoryConfigError,
    config_fingerprint,
    load_config,
    read_jsonl,
    resolve_path,
    sha256_file,
    stable_int,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "nano_viz_evidence_qualification.v1"
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
PRIMARY_VARIANTS = (
    "teacher",
    "av_real",
    "av_mean",
    "av_none",
    "av_shuffled",
    "av_zero",
    "mean",
)


def _normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)


def _rowwise_directional_mse(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    delta = _normalize(predictions) - _normalize(targets)
    return np.square(delta).sum(axis=1)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(_normalize(left) * _normalize(right), axis=1)


def _load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cache:
        required = (
            "train_mean",
            "validation__row_indices",
            "validation__content_family_ids",
            "validation__doc_ids",
            "validation__targets",
        )
        missing = [key for key in required if key not in cache]
        if missing:
            raise ObservatoryConfigError(f"{path} is missing cache keys: {missing}")
        payload = {key: np.asarray(cache[key]) for key in cache.files if key != "metadata_json"}
        payload["metadata_json"] = (
            json.loads(str(cache["metadata_json"])) if "metadata_json" in cache else {}
        )
    row_count, width = payload["validation__targets"].shape
    if row_count != 512 or width != 2688:
        raise ObservatoryConfigError(
            f"qualified cache must be (512, 2688), got {(row_count, width)}"
        )
    if not all(
        np.isfinite(value).all()
        for key, value in payload.items()
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
    ):
        raise ObservatoryConfigError(f"{path} contains non-finite numeric values")
    return payload


def _prediction(cache: dict[str, Any], variant: str) -> np.ndarray:
    key = f"validation__prediction__{variant}"
    if key not in cache:
        raise ObservatoryConfigError(f"prediction cache is missing {key}")
    return np.asarray(cache[key], dtype=np.float64)


def _clustered_mean_ci(
    values: np.ndarray,
    families: np.ndarray,
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for value, family in zip(values, families, strict=True):
        grouped.setdefault(str(family), []).append(float(value))
    effects = np.asarray([np.mean(group) for group in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    chunk_size = max(1, min(8192, 2_000_000 // len(effects)))
    for start in range(0, resamples, chunk_size):
        stop = min(resamples, start + chunk_size)
        indices = rng.integers(0, len(effects), size=(stop - start, len(effects)))
        draws[start:stop] = effects[indices].mean(axis=1)
    return {
        "unit": "content_family_id",
        "row_count": int(len(values)),
        "family_count": int(len(effects)),
        "family_weighted_mean": float(effects.mean()),
        "row_weighted_mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def build_waterfall(cache: dict[str, Any], *, seed: int) -> dict[str, Any]:
    targets = np.asarray(cache["validation__targets"], dtype=np.float64)
    train_mean = np.asarray(cache["train_mean"], dtype=np.float64)
    families = cache["validation__content_family_ids"]
    variants: dict[str, Any] = {}
    for variant in PRIMARY_VARIANTS:
        predictions = _prediction(cache, variant)
        metrics = activation_reconstruction_metrics(
            predictions, targets, train_mean=train_mean
        )
        rowwise = np.asarray(metrics.pop("rowwise_directional_mse"), dtype=np.float64)
        metrics.pop("rowwise_unit_vector_feature_mse", None)
        metrics.pop("rowwise_raw_mse", None)
        variants[variant] = {
            **metrics,
            "family_clustered_directional_mse": _clustered_mean_ci(
                rowwise, families, seed=stable_int(seed, "waterfall", variant)
            ),
        }
    return {
        "metric": "directional_mse_equals_2_times_one_minus_cosine",
        "split": "validation",
        "variants": variants,
    }


def _fano_lower_bound(error_rate: float, gallery_size: int) -> float:
    if gallery_size <= 1:
        return 0.0
    error_rate = min(max(float(error_rate), 0.0), 1.0)
    binary_entropy = 0.0
    if 0.0 < error_rate < 1.0:
        binary_entropy = -error_rate * math.log2(error_rate) - (1.0 - error_rate) * math.log2(
            1.0 - error_rate
        )
    return max(
        0.0,
        math.log2(gallery_size)
        - binary_entropy
        - error_rate * math.log2(gallery_size - 1),
    )


def build_capacity_ladder(cache: dict[str, Any], *, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    targets = _normalize(cache["validation__targets"])
    predictions = _normalize(_prediction(cache, "av_real"))
    families = np.asarray(cache["validation__content_family_ids"])
    similarities = predictions @ targets.T
    gallery_sizes = (2, 4, 8, 16, 32, 64, 128, 256)
    ladder: list[dict[str, Any]] = []
    final_ranks = np.zeros(len(targets), dtype=np.int64)
    for gallery_size in gallery_sizes:
        ranks: list[int] = []
        for row_index in range(len(targets)):
            eligible = [
                candidate
                for candidate in range(len(targets))
                if candidate != row_index and families[candidate] != families[row_index]
            ]
            eligible.sort(
                key=lambda candidate: stable_int(
                    seed, "capacity", gallery_size, row_index, candidate
                )
            )
            gallery = [row_index, *eligible[: gallery_size - 1]]
            scores = similarities[row_index, gallery]
            rank = 1 + int(np.sum(scores > scores[0]))
            ranks.append(rank)
        ranks_array = np.asarray(ranks, dtype=np.int64)
        if gallery_size == gallery_sizes[-1]:
            final_ranks[:] = ranks_array
        accuracy = float(np.mean(ranks_array == 1))
        ladder.append(
            {
                "gallery_size": gallery_size,
                "gallery_bits": math.log2(gallery_size),
                "top1_accuracy": accuracy,
                "top5_accuracy": float(np.mean(ranks_array <= min(5, gallery_size))),
                "mean_reciprocal_rank": float(np.mean(1.0 / ranks_array)),
                "median_rank": float(np.median(ranks_array)),
                "fano_information_lower_bound_bits": _fano_lower_bound(
                    1.0 - accuracy, gallery_size
                ),
            }
        )
    wrong_pairs: collections.Counter[tuple[str, str]] = collections.Counter()
    nearest = np.argsort(-similarities, axis=1)
    for row_index, candidates in enumerate(nearest):
        for candidate in candidates:
            if candidate == row_index or families[candidate] == families[row_index]:
                continue
            wrong_pairs[(str(families[row_index]), str(families[candidate]))] += 1
            break
    return (
        {
            "variant": "av_real",
            "distance": "cosine",
            "family_exclusion": "same-family distractors excluded; exact target retained",
            "assumptions": {
                "gallery_prior": "uniform by construction",
                "decoder": "nearest stored target by cosine",
                "fano_scope": "classifier-derived lower bound, not channel capacity equality",
            },
            "ladder": ladder,
            "top_confusions": [
                {"source_family": pair[0], "retrieved_family": pair[1], "count": count}
                for pair, count in wrong_pairs.most_common(20)
            ],
        },
        final_ranks,
    )


def build_twin_critic(
    primary: dict[str, Any], independent: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    for identity_key in (
        "validation__row_indices",
        "validation__content_family_ids",
        "validation__doc_ids",
    ):
        if not np.array_equal(primary[identity_key], independent[identity_key]):
            raise ObservatoryConfigError(f"critic caches disagree on {identity_key}")
    targets = np.asarray(primary["validation__targets"], dtype=np.float64)
    if not np.allclose(targets, independent["validation__targets"], rtol=0.0, atol=0.0):
        raise ObservatoryConfigError("critic caches contain different targets")
    primary_predictions = _prediction(primary, "av_real")
    independent_predictions = _prediction(independent, "av_real")
    primary_error = primary_predictions - targets
    independent_error = independent_predictions - targets
    primary_dmse = _rowwise_directional_mse(primary_predictions, targets)
    independent_dmse = _rowwise_directional_mse(independent_predictions, targets)
    family_ids = primary["validation__content_family_ids"]
    return {
        "shared_teacher_confound": True,
        "prediction_cosine": {
            "mean": float(_cosine_rows(primary_predictions, independent_predictions).mean()),
            "min": float(_cosine_rows(primary_predictions, independent_predictions).min()),
        },
        "error_vector_cosine": {
            "mean": float(_cosine_rows(primary_error, independent_error).mean()),
            "median": float(np.median(_cosine_rows(primary_error, independent_error))),
        },
        "rowwise_directional_mse_correlation": float(
            np.corrcoef(primary_dmse, independent_dmse)[0, 1]
        ),
        "primary_directional_mse": _clustered_mean_ci(
            primary_dmse, family_ids, seed=stable_int(seed, "twin", "primary")
        ),
        "independent_directional_mse": _clustered_mean_ci(
            independent_dmse, family_ids, seed=stable_int(seed, "twin", "independent")
        ),
    }


def build_spectral_strip(
    cache: dict[str, Any], *, components: int = 64
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    targets = np.asarray(cache["validation__targets"], dtype=np.float64)
    predictions = _prediction(cache, "av_real")
    mean = targets.mean(axis=0)
    centered = targets - mean
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order][:components], 0.0)
    left = eigenvectors[:, order[:components]]
    singular = np.sqrt(eigenvalues)
    valid = singular > 1e-10
    basis = np.zeros((components, targets.shape[1]), dtype=np.float64)
    basis[valid] = (left[:, valid].T @ centered) / singular[valid, None]
    target_coords = centered @ basis.T
    prediction_coords = (predictions - mean) @ basis.T
    residual_coords = (predictions - targets) @ basis.T
    target_energy = np.mean(np.square(target_coords), axis=0)
    residual_energy = np.mean(np.square(residual_coords), axis=0)
    total_variance = float(np.sum(eigenvalues))
    rows = []
    for index in range(components):
        rows.append(
            {
                "pc": index + 1,
                "eigenvalue": float(eigenvalues[index]),
                "explained_variance_fraction": (
                    0.0 if total_variance == 0.0 else float(eigenvalues[index] / total_variance)
                ),
                "target_energy": float(target_energy[index]),
                "residual_energy": float(residual_energy[index]),
                "transmitted_fraction": (
                    None
                    if target_energy[index] <= 1e-12
                    else float(1.0 - residual_energy[index] / target_energy[index])
                ),
            }
        )
    return (
        {
            "fit_split": "validation",
            "variant": "av_real",
            "component_count": components,
            "components": rows,
        },
        {
            "mean": mean.astype(np.float32),
            "basis": basis.astype(np.float32),
            "eigenvalues": eigenvalues.astype(np.float32),
            "target_coords": target_coords.astype(np.float32),
            "prediction_coords": prediction_coords.astype(np.float32),
        },
    )


def _control_text(record: dict[str, Any], control: str) -> str:
    payload = (record.get("controls") or {}).get(control) or {}
    parsed = payload.get("parsed") or {}
    return str(parsed.get("explanation") or payload.get("generated") or "").strip()


def build_null_almanac(records: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {control: collections.Counter() for control in ("real", "zero", "none")}
    usable = collections.Counter()
    closed = collections.Counter()
    for record in records:
        for control in counters:
            payload = (record.get("controls") or {}).get(control) or {}
            parsed = payload.get("parsed") or {}
            text = _control_text(record, control)
            counters[control].update(token.lower() for token in WORD_RE.findall(text))
            usable[control] += bool(parsed.get("usable", text))
            closed[control] += bool(parsed.get("closed"))
    vocabulary = set(counters["real"]) | set(counters["zero"])
    real_total = sum(counters["real"].values()) + len(vocabulary)
    zero_total = sum(counters["zero"].values()) + len(vocabulary)
    log_odds = {
        token: math.log((counters["real"][token] + 1) / real_total)
        - math.log((counters["zero"][token] + 1) / zero_total)
        for token in vocabulary
    }
    return {
        "row_count": len(records),
        "scope": "preliminary_word_frequency; token_logprob_backfill_pending",
        "parse_health": {
            control: {
                "usable_fraction": usable[control] / len(records),
                "closed_fraction": closed[control] / len(records),
            }
            for control in counters
        },
        "real_enriched_words": [
            {"token": token, "log_odds_real_vs_zero": float(score)}
            for token, score in sorted(log_odds.items(), key=lambda item: item[1], reverse=True)[:40]
        ],
        "zero_enriched_words": [
            {"token": token, "log_odds_real_vs_zero": float(score)}
            for token, score in sorted(log_odds.items(), key=lambda item: item[1])[:40]
        ],
    }


def _source_card(path: Path, fields: tuple[str, ...]) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
        **{field: payload.get(field) for field in fields},
    }


def _source_raw_card(extract_root: Path) -> dict[str, Any] | None:
    candidates = list(extract_root.rglob("eval_iter_0001291_v512_t512_winrates_report.json"))
    if len(candidates) != 1:
        return None
    path = candidates[0]
    payload = json.loads(path.read_text())
    metrics = (((payload.get("splits") or {}).get("validation") or {}).get("controls") or {}).get(
        "source_raw"
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "metrics": metrics,
    }


def build_row_metrics(
    cache: dict[str, Any], independent: dict[str, Any], selection: dict[str, Any], ranks: np.ndarray
) -> list[dict[str, Any]]:
    row_positions = {
        int(row_index): position
        for position, row_index in enumerate(cache["validation__row_indices"])
    }
    targets = cache["validation__targets"]
    output = []
    for row_id in selection["deep_dive_row_ids"]:
        row_index = int(str(row_id).split("-")[-1])
        position = row_positions[row_index]
        variants = {
            variant: float(
                _rowwise_directional_mse(
                    _prediction(cache, variant)[position : position + 1],
                    targets[position : position + 1],
                )[0]
            )
            for variant in PRIMARY_VARIANTS
        }
        output.append(
            {
                "schema_version": "nano_viz_e0_row.v1",
                "row_id": row_id,
                "row_index": row_index,
                "content_family_id": str(cache["validation__content_family_ids"][position]),
                "directional_mse": variants,
                "independent_av_real_directional_mse": float(
                    _rowwise_directional_mse(
                        _prediction(independent, "av_real")[position : position + 1],
                        targets[position : position + 1],
                    )[0]
                ),
                "retrieval_rank_gallery256": int(ranks[position]),
            }
        )
    return output


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    paths = config["paths"]
    seed = int(config["selection"]["seed"])
    output_dir = resolve_path(paths["model_outputs_dir"], config_path=config_path) / "e0"
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_path = resolve_path(paths["validation_prediction_cache_npz"], config_path=config_path)
    independent_path = resolve_path(
        paths["independent_validation_prediction_cache_npz"], config_path=config_path
    )
    primary = _load_cache(primary_path)
    independent = _load_cache(independent_path)
    selection_path = resolve_path(paths["corpus_dir"], config_path=config_path) / "selection_manifest.json"
    selection = json.loads(selection_path.read_text())
    records = read_jsonl(
        resolve_path(paths["generated_validation_jsonl"], config_path=config_path)
    )
    if len(records) != 512 or {str(record.get("split")) for record in records} != {"validation"}:
        raise ObservatoryConfigError("generated evidence must contain 512 validation rows only")

    waterfall = build_waterfall(primary, seed=seed)
    capacity, ranks = build_capacity_ladder(primary, seed=seed)
    twin = build_twin_critic(primary, independent, seed=seed)
    spectral, spectral_arrays = build_spectral_strip(primary)
    null_almanac = build_null_almanac(records)
    evidence_root = resolve_path(paths["evidence_root"], config_path=config_path)
    publication_root = evidence_root / "r33-clean-sft-publication-evidence-20260716"
    drift = _source_card(
        publication_root / "activation_fidelity_validation64_mb8.json",
        ("publication_ready", "primary_fidelity_assessment", "activation_fidelity"),
    )
    magnitude = _source_card(
        publication_root / "roundtrip_magnitude_calibration_report.json",
        ("publication_status", "fit", "evaluation", "claim_boundary"),
    )
    privacy = _source_card(
        publication_root / "release_text_privacy_memorization_audit.json",
        ("automatic_gate_passed", "human_review_required", "claim_boundary"),
    )
    source_raw = _source_raw_card(
        resolve_path(paths["qualified_extract_dir"], config_path=config_path)
    )
    row_metrics = build_row_metrics(primary, independent, selection, ranks)
    write_jsonl(output_dir / "rows.jsonl", row_metrics)
    np.savez_compressed(
        output_dir / "spectral_geometry.npz",
        row_indices=primary["validation__row_indices"],
        **spectral_arrays,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": bool(privacy.get("automatic_gate_passed"))
        and len(row_metrics) == 50
        and twin["primary_directional_mse"]["family_count"] >= 100,
        "claim_scope": "stored_snapshot",
        "split": "validation",
        "config_sha256": config_fingerprint(config),
        "input_provenance": {
            "primary_cache_sha256": sha256_file(primary_path),
            "independent_cache_sha256": sha256_file(independent_path),
            "selection_manifest_sha256": sha256_file(selection_path),
        },
        "information_waterfall": waterfall,
        "capacity_ladder": capacity,
        "twin_critic": twin,
        "spectral_strip": spectral,
        "null_text_almanac": null_almanac,
        "drift_card": drift,
        "magnitude_card": magnitude,
        "privacy_card": privacy,
        "source_raw_floor": source_raw,
        "deep_dive_row_count": len(row_metrics),
        "limitations": [
            "PCA and Court-facing summaries are validation-fitted exploratory views.",
            "Twin critics share a teacher target and are not semantically independent.",
            "Null-text token log-probabilities require the E2 backfill before final use.",
        ],
    }
    write_json(output_dir / "qualification_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args.config)
    except (OSError, ValueError, ObservatoryConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "passed": report["passed"],
                "deep_dive_row_count": report["deep_dive_row_count"],
                "av_real_directional_mse": report["information_waterfall"]["variants"]["av_real"]["directional_mse"],
                "independent_av_real_family_mean": report["twin_critic"]["independent_directional_mse"]["family_weighted_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
