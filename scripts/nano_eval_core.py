#!/usr/bin/env python3
"""Dependency-light statistics shared by Nano NLA evaluation tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def select_requested_eval_splits(
    eval_splits: Iterable[str],
    **split_payloads: Any,
) -> dict[str, Any]:
    requested = [str(split) for split in eval_splits]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("eval_splits must be non-empty and unique")
    unknown = [split for split in requested if split not in split_payloads]
    if unknown:
        raise ValueError(f"unknown eval splits: {unknown}")
    return {split: split_payloads[split] for split in requested}


def shuffled_control_candidates(
    rows: list[dict[str, Any]],
    *,
    row_index: int,
) -> list[int]:
    row = rows[row_index]
    split = str(row.get("split") or "")
    if not split:
        raise ValueError(f"row {row_index} is missing split metadata")
    family = row.get("content_family_id")
    return [
        int(candidate["row_index"])
        for candidate in rows
        if int(candidate["row_index"]) != row_index
        and str(candidate.get("split") or "") == split
        and (
            family in {None, ""}
            or (
                candidate.get("content_family_id") not in {None, ""}
                and str(candidate.get("content_family_id")) != str(family)
            )
        )
    ]


def activation_reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    train_mean: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Report direction-only and raw-space activation reconstruction metrics.

    ``directional_mse`` is the squared Euclidean distance between independently
    L2-normalized prediction and target rows. It is exactly ``2 * (1-cosine)``
    and is invariant to positive scalar rescaling. ``raw_mse`` and
    ``centered_r2`` retain activation magnitude and therefore answer a
    different question. ``normalized_mse`` is a compatibility alias for the
    same dimension-independent directional quantity.
    """

    pred = np.asarray(prediction, dtype=np.float64)
    gold = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or gold.ndim != 2 or pred.shape != gold.shape or pred.size == 0:
        raise ValueError("prediction and target must have matching non-empty 2D shapes")
    if not np.isfinite(pred).all() or not np.isfinite(gold).all():
        raise ValueError("prediction and target must contain only finite values")

    mean = np.asarray(train_mean, dtype=np.float64)
    if mean.ndim == 2 and mean.shape[0] == 1:
        mean = mean[0]
    if mean.ndim != 1 or mean.shape[0] != gold.shape[1]:
        raise ValueError("train_mean must be a vector matching the activation dimension")
    if not np.isfinite(mean).all():
        raise ValueError("train_mean must contain only finite values")
    if eps <= 0:
        raise ValueError("eps must be positive")

    prediction_norms = np.linalg.norm(pred, axis=1)
    target_norms = np.linalg.norm(gold, axis=1)
    normalized_prediction = pred / np.maximum(prediction_norms[:, None], eps)
    normalized_target = gold / np.maximum(target_norms[:, None], eps)
    rowwise_directional_mse = np.sum(
        np.square(normalized_prediction - normalized_target),
        axis=1,
    )
    rowwise_unit_feature_mse = rowwise_directional_mse / float(gold.shape[1])
    rowwise_raw_mse = np.mean(np.square(pred - gold), axis=1)
    rowwise_mean_predictor_raw_mse = np.mean(np.square(gold - mean[None, :]), axis=1)
    mean_predictor_raw_mse = float(rowwise_mean_predictor_raw_mse.mean())
    raw_mse = float(rowwise_raw_mse.mean())
    valid_target_norms = target_norms > eps
    norm_ratio_mean = (
        float(np.mean(prediction_norms[valid_target_norms] / target_norms[valid_target_norms]))
        if np.any(valid_target_norms)
        else None
    )
    directional_mse = float(rowwise_directional_mse.mean())
    return {
        "directional_mse": directional_mse,
        "normalized_mse": directional_mse,
        "unit_vector_feature_mse": float(rowwise_unit_feature_mse.mean()),
        "cosine_mean": float(np.mean(np.sum(normalized_prediction * normalized_target, axis=1))),
        "raw_mse": raw_mse,
        "mean_predictor_raw_mse": mean_predictor_raw_mse,
        "centered_r2": (
            None
            if mean_predictor_raw_mse <= eps
            else float(1.0 - raw_mse / mean_predictor_raw_mse)
        ),
        "prediction_norm_mean": float(prediction_norms.mean()),
        "target_norm_mean": float(target_norms.mean()),
        "norm_ratio_mean": norm_ratio_mean,
        "rowwise_directional_mse": rowwise_directional_mse,
        "rowwise_unit_vector_feature_mse": rowwise_unit_feature_mse,
        "rowwise_raw_mse": rowwise_raw_mse,
    }


def _paired_finite_1d(
    baseline: Iterable[float],
    candidate: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if (
        baseline_array.ndim != 1
        or candidate_array.ndim != 1
        or baseline_array.size == 0
        or baseline_array.shape != candidate_array.shape
    ):
        raise ValueError("baseline and candidate must be non-empty paired 1D arrays")
    if not np.isfinite(baseline_array).all() or not np.isfinite(candidate_array).all():
        raise ValueError("paired arrays must contain only finite values")
    return baseline_array, candidate_array


def paired_bootstrap_improvement(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    seed: int = 0,
    resamples: int = 10_000,
) -> dict[str, float | int]:
    """Estimate paired improvement where positive means the candidate is lower.

    This orientation is suitable for loss, NMSE, KL, and divergence metrics.
    Higher-is-better callers should negate both paired arrays before calling.
    """

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    baseline_array, candidate_array = _paired_finite_1d(baseline, candidate)
    improvements = baseline_array - candidate_array
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        improvements.size,
        size=(resamples, improvements.size),
    )
    bootstrap_means = improvements[sample_indices].mean(axis=1)
    return {
        "count": int(improvements.size),
        "mean_improvement": float(improvements.mean()),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "candidate_better_fraction": float(np.mean(improvements > 0.0)),
        "tie_fraction": float(np.mean(improvements == 0.0)),
    }


def clustered_paired_bootstrap_improvement(
    baseline: Iterable[float],
    candidate: Iterable[float],
    cluster_ids: Iterable[str],
    *,
    seed: int = 0,
    resamples: int = 10_000,
) -> dict[str, float | int | str]:
    """Bootstrap paired improvements over equal-weight independent clusters."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    baseline_array, candidate_array = _paired_finite_1d(baseline, candidate)
    cluster_values = [str(value) for value in cluster_ids]
    if len(cluster_values) != baseline_array.size:
        raise ValueError("cluster IDs must align with paired arrays")
    if any(not value for value in cluster_values):
        raise ValueError("cluster IDs must be non-empty")
    grouped: dict[str, list[int]] = {}
    for index, cluster_id in enumerate(cluster_values):
        grouped.setdefault(cluster_id, []).append(index)
    if len(grouped) < 2:
        raise ValueError("clustered bootstrap requires at least two clusters")
    improvements = baseline_array - candidate_array
    cluster_effects = np.asarray(
        [float(improvements[indexes].mean()) for indexes in grouped.values()],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    chunk_size = max(1, min(65_536, 4_000_000 // len(cluster_effects)))
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        selected = rng.integers(
            0,
            len(cluster_effects),
            size=(stop - start, len(cluster_effects)),
        )
        bootstrap_means[start:stop] = cluster_effects[selected].mean(axis=1)
    return {
        "unit": "cluster",
        "row_count": int(improvements.size),
        "cluster_count": int(len(cluster_effects)),
        "mean_improvement": float(cluster_effects.mean()),
        "row_weighted_mean_improvement": float(improvements.mean()),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "candidate_better_fraction": float(np.mean(cluster_effects > 0.0)),
        "tie_fraction": float(np.mean(cluster_effects == 0.0)),
    }


def _finite_logits(values: Iterable[float], *, name: str) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    if logits.ndim != 1 or logits.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least two logits")
    if not np.isfinite(logits).all():
        raise ValueError(f"{name} must contain only finite logits")
    return logits


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    return shifted - np.log(np.exp(shifted).sum())


def _pearson(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_centered = lhs - lhs.mean()
    rhs_centered = rhs - rhs.mean()
    denominator = np.linalg.norm(lhs_centered) * np.linalg.norm(rhs_centered)
    if denominator == 0.0:
        return 1.0 if np.array_equal(lhs, rhs) else 0.0
    return float(np.dot(lhs_centered, rhs_centered) / denominator)


def functional_logit_metrics(
    original_logits: Iterable[float],
    patched_logits: Iterable[float],
    *,
    top_ks: tuple[int, ...] = (10, 50),
) -> dict[str, Any]:
    """Compare patched next-token logits against the unmodified target model."""

    original = _finite_logits(original_logits, name="original_logits")
    patched = _finite_logits(patched_logits, name="patched_logits")
    if original.shape != patched.shape:
        raise ValueError("original_logits and patched_logits must have matching shapes")

    original_log_probabilities = _log_softmax(original)
    patched_log_probabilities = _log_softmax(patched)
    original_probabilities = np.exp(original_log_probabilities)
    patched_probabilities = np.exp(patched_log_probabilities)
    midpoint_log_probabilities = np.logaddexp(
        original_log_probabilities,
        patched_log_probabilities,
    ) - np.log(2.0)
    kl_original_to_patched = np.sum(
        original_probabilities
        * (original_log_probabilities - patched_log_probabilities)
    )
    js_divergence = 0.5 * np.sum(
        original_probabilities
        * (original_log_probabilities - midpoint_log_probabilities)
    ) + 0.5 * np.sum(
        patched_probabilities
        * (patched_log_probabilities - midpoint_log_probabilities)
    )

    result: dict[str, Any] = {
        "vocab_size": int(original.size),
        "kl_original_to_patched": float(max(0.0, kl_original_to_patched)),
        "js_divergence": float(max(0.0, js_divergence)),
        "logit_pearson": _pearson(original, patched),
    }
    original_order = np.argsort(-original, kind="stable")
    patched_order = np.argsort(-patched, kind="stable")
    original_top1 = int(original_order[0])
    result["original_top1_token_id"] = original_top1
    result["original_top1_rank"] = int(np.flatnonzero(patched_order == original_top1)[0]) + 1

    for top_k in top_ks:
        if top_k <= 0 or top_k > original.size:
            raise ValueError(f"top_k={top_k} must be in [1, {original.size}]")
        original_set = set(int(index) for index in original_order[:top_k])
        patched_set = set(int(index) for index in patched_order[:top_k])
        result[f"top_{top_k}_overlap"] = len(original_set & patched_set) / top_k
    return result
