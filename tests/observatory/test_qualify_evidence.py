from __future__ import annotations

import numpy as np

from observatory.qualify_evidence import (
    _fano_lower_bound,
    build_capacity_ladder,
    build_spectral_strip,
    build_twin_critic,
)


def _cache(rows: int = 512, width: int = 16) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    targets = rng.normal(size=(rows, width)).astype(np.float32)
    return {
        "train_mean": targets.mean(axis=0),
        "validation__row_indices": np.arange(rows),
        "validation__content_family_ids": np.asarray([f"cf_{index}" for index in range(rows)]),
        "validation__doc_ids": np.asarray([f"doc_{index}" for index in range(rows)]),
        "validation__targets": targets,
        "validation__prediction__av_real": targets.copy(),
    }


def test_capacity_ladder_recovers_exact_targets() -> None:
    report, ranks = build_capacity_ladder(_cache(), seed=11)
    assert all(item["top1_accuracy"] == 1.0 for item in report["ladder"])
    assert np.all(ranks == 1)
    assert report["ladder"][-1]["fano_information_lower_bound_bits"] == 8.0


def test_twin_critic_reports_identical_predictions() -> None:
    primary = _cache(rows=32)
    independent = {key: value.copy() for key, value in primary.items()}
    report = build_twin_critic(primary, independent, seed=13)
    assert report["prediction_cosine"]["mean"] == 1.0
    assert report["rowwise_directional_mse_correlation"] != report[
        "rowwise_directional_mse_correlation"
    ]  # undefined when both error arrays are constant


def test_spectral_strip_is_finite_for_exact_recovery() -> None:
    report, arrays = build_spectral_strip(_cache(rows=64, width=16), components=8)
    assert report["component_count"] == 8
    assert arrays["basis"].shape == (8, 16)
    assert np.isfinite(arrays["basis"]).all()
    assert max(item["residual_energy"] for item in report["components"]) == 0.0


def test_fano_bound_has_expected_endpoints() -> None:
    assert _fano_lower_bound(0.0, 16) == 4.0
    assert _fano_lower_bound(15.0 / 16.0, 16) == 0.0
