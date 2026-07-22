from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from observatory.bundle_common import family_bootstrap_interval
from observatory.compute_geometry import (
    fit_validation_basis,
    target_retrieval_rank,
)
from observatory.compute_interventions import (
    exact_shapley,
    fit_balanced_threshold,
)
from observatory.build_bundle import _require_bound_artifact, manifest_bundle_id
from observatory.common import ObservatoryConfigError, sha256_file, stable_int
from observatory.verify_bundle import (
    _vector_metrics,
    _verify_aggregates,
    _verify_control_groups,
    validate_vector_layout,
)


def test_geometry_basis_is_deterministic_and_centered() -> None:
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(12, 8)).astype(np.float32)
    first = fit_validation_basis(vectors, 4)
    second = fit_validation_basis(vectors, 4)
    np.testing.assert_array_equal(first["mean"], second["mean"])
    np.testing.assert_array_equal(first["basis"], second["basis"])
    assert first["basis"].shape == (4, 8)
    assert np.isfinite(first["explained_variance_ratio"]).all()


def test_native_retrieval_rank_finds_expected_target() -> None:
    targets = np.eye(4, dtype=np.float32)
    rank, nearest, cosine = target_retrieval_rank(targets[2], targets, 2)
    assert rank == 1
    assert nearest == 2
    assert cosine == pytest.approx(1.0)


def test_family_bootstrap_weights_families_not_rows() -> None:
    result = family_bootstrap_interval(
        [0.0, 0.0, 0.0, 1.0],
        ["large", "large", "large", "small"],
        samples=1000,
        confidence=0.95,
        seed=11,
    )
    assert result["mean"] == pytest.approx(0.5)
    assert result["families"] == 2
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]


def test_exact_shapley_satisfies_efficiency_for_additive_game() -> None:
    values = {
        mask: sum((index + 1) for index in range(4) if mask & (1 << index))
        for mask in range(16)
    }
    contributions = exact_shapley(values)
    np.testing.assert_allclose(contributions, [1.0, 2.0, 3.0, 4.0])
    assert sum(contributions) == pytest.approx(values[15] - values[0])


def test_balanced_threshold_separates_ordered_scores() -> None:
    result = fit_balanced_threshold([0.8, 0.9], [0.1, 0.2])
    assert 0.2 < result["threshold"] < 0.8
    assert result["balanced_accuracy"] == pytest.approx(1.0)


def test_manifest_bundle_id_is_order_independent() -> None:
    first = {"schema_version": "v1", "counts": {"rows": 2, "cells": 3}}
    second = {"counts": {"cells": 3, "rows": 2}, "schema_version": "v1"}
    assert manifest_bundle_id(first) == manifest_bundle_id(second)


def test_vector_layout_accepts_contiguous_float16_vectors() -> None:
    rows = [
        {
            "ref": f"vector:{index}",
            "offset_elements": index * 2688,
            "length_elements": 2688,
            "dtype": "float16_le",
        }
        for index in range(3)
    ]
    assert validate_vector_layout(rows, 3 * 2688 * 2) == []


def test_vector_layout_rejects_gaps_duplicates_and_wrong_size() -> None:
    rows = [
        {
            "ref": "duplicate",
            "offset_elements": 0,
            "length_elements": 2688,
            "dtype": "float16_le",
        },
        {
            "ref": "duplicate",
            "offset_elements": 3000,
            "length_elements": 1024,
            "dtype": "float32",
        },
    ]
    errors = validate_vector_layout(rows, 1)
    assert any("offset" in error for error in errors)
    assert any("length" in error for error in errors)
    assert any("dtype" in error for error in errors)
    assert any("duplicate refs" in error for error in errors)
    assert any("binary" in error for error in errors)


def test_bound_artifact_requires_matching_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n")
    binding = {"path": str(artifact), "sha256": sha256_file(artifact)}
    assert _require_bound_artifact(binding, "test") == artifact
    binding["sha256"] = "0" * 64
    with pytest.raises(ObservatoryConfigError, match="hash mismatch"):
        _require_bound_artifact(binding, "test")


def test_vector_metrics_match_directional_contract() -> None:
    target = np.asarray([1.0, 0.0], dtype=np.float32)
    prediction = np.asarray([0.0, 2.0], dtype=np.float32)
    metrics = _vector_metrics(prediction, target)
    assert metrics["directional_mse"] == pytest.approx(2.0)
    assert metrics["cosine"] == pytest.approx(0.0)
    assert metrics["raw_mse"] == pytest.approx(2.5)
    assert metrics["norm_ratio"] == pytest.approx(2.0)


def test_control_groups_require_all_three_lanes_per_dose() -> None:
    complete = [
        {
            "control_group_id": "g1",
            "spec_json": '{"dose": 1.0, "lane": "' + lane + '"}',
        }
        for lane in ("edit", "paraphrase_placebo", "random_edit")
    ]
    errors: list[str] = []
    _verify_control_groups(complete, errors)
    assert errors == []
    _verify_control_groups(complete[:-1], errors)
    assert any("incomplete" in error for error in errors)


def test_aggregate_verifier_rederives_family_clustered_intervals() -> None:
    metrics = [
        {
            "critic": "primary",
            "family": "identity",
            "content_family_id": family,
            "directional_mse": dmse,
            "cosine": cosine,
        }
        for family, dmse, cosine in (
            ("a", 0.1, 0.95),
            ("a", 0.2, 0.90),
            ("b", 0.3, 0.85),
        )
    ]
    statistics = {"bootstrap_samples": 100, "confidence": 0.95, "seed": 7}
    aggregate_rows = []
    for metric in ("directional_mse", "cosine"):
        aggregate_rows.append(
            {
                "critic": "primary",
                "family": "identity",
                "metric": metric,
                **family_bootstrap_interval(
                    [row[metric] for row in metrics],
                    [row["content_family_id"] for row in metrics],
                    samples=100,
                    confidence=0.95,
                    seed=stable_int(7, 0, metric),
                ),
            }
        )
    config = {
        "statistics": statistics,
        "verification": {"aggregate_tolerance": 1.0e-9},
    }
    errors: list[str] = []
    _verify_aggregates(metrics, aggregate_rows, config, errors)
    assert errors == []
    aggregate_rows[0]["mean"] += 0.1
    _verify_aggregates(metrics, aggregate_rows, config, errors)
    assert any("aggregate mismatch" in error for error in errors)
