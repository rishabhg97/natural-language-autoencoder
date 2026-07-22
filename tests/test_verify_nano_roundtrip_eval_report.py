from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "verify_nano_roundtrip_eval_report.py"
    spec = importlib.util.spec_from_file_location("roundtrip_verifier", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _report(count: int = 120) -> dict:
    family_ids = [f"family-{index}" for index in range(count)]
    primary = [0.30] * count
    controls = {
        "av_shuffled": [0.95] * count,
        "av_zero": [0.90] * count,
        "av_mean": [0.80] * count,
        "av_none": [0.85] * count,
        "mean": [0.65] * count,
    }
    variants = {
        "av_real": {
            "directional_mse": 0.30,
            "raw_mse": 9.0,
            "centered_raw_r2": -0.2,
            "norm_ratio_mean": 1.4,
        },
        "teacher": {"directional_mse": 0.29},
    }
    variants.update(
        {name: {"directional_mse": values[0]} for name, values in controls.items()}
    )
    return {
        "eval_splits": ["validation"],
        "ar_checkpoint_dir": "/checkpoints/ar",
        "generation_protocol_sha256": "f" * 64,
        "validated_generation_provenance": {
            "model_fingerprint": "dcp_model_sha256:" + "a" * 64,
            "tokenizer_fingerprint": "tokenizer_files_sha256:" + "b" * 64,
            "datasets": {
                "train": {"sha256": "c" * 64},
                "validation": {"sha256": "d" * 64},
            },
        },
        "gate": {
            "passed": True,
            "current_generation_protocol_compatible": True,
            "require_family_level_inference": True,
        },
        "splits": {
            "validation": {
                "row_count": count,
                "independent_family_count": count,
                "content_family_ids": family_ids,
                "generation_parse": {
                    "real": {"closed_fraction": 1.0, "usable_fraction": 1.0}
                },
                "variants": variants,
                "rowwise_directional_mse": {"av_real": primary, **controls},
            }
        },
    }


def _config(report_path: Path, count: int = 120) -> dict:
    return {
        "report_json": str(report_path),
        "expected": {
            "eval_splits": ["validation"],
            "counts": {"validation": count},
            "min_independent_families": 100,
            "primary_variant": "av_real",
            "controls": ["av_shuffled", "av_zero", "av_mean", "av_none", "mean"],
            "max_primary_directional_mse": 0.35,
            "max_primary_gap_to_teacher": 0.05,
            "min_control_margin": 0.1344,
            "min_rowwise_win_fraction": 0.9,
            "min_closed_fraction": 0.95,
            "min_usable_fraction": 0.99,
            "max_sign_flip_p_value": 0.01,
            "bootstrap_samples": 1000,
            "bootstrap_seed": 7,
            "permutation_samples": 1000,
            "permutation_seed": 7,
            "require_raw_magnitude_claim": False,
            "forbid_unrequested_rows": True,
            "ar_checkpoint_dir": "/checkpoints/ar",
            "generation_identity": {
                "model_fingerprint": "dcp_model_sha256:" + "a" * 64,
                "tokenizer_fingerprint": "tokenizer_files_sha256:" + "b" * 64,
                "dataset_sha256": {
                    "train": "c" * 64,
                    "validation": "d" * 64,
                },
            },
        },
    }


def test_verifier_requires_clustered_control_separation(tmp_path: Path):
    module = _load_module()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()))

    result = module.verify(_config(report_path))

    assert result["passed"] is True
    assert result["raw_magnitude_claim_supported"] is False
    assert all(
        value["passed"]
        for value in result["split_results"]["validation"]["controls"].values()
    )


def test_verifier_rejects_wrong_dataset_hash(tmp_path: Path):
    module = _load_module()
    report = _report()
    report["validated_generation_provenance"]["datasets"]["validation"][
        "sha256"
    ] = "e" * 64
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    result = module.verify(_config(report_path))

    assert result["passed"] is False
    assert "generation dataset hash mismatch for validation" in result["errors"]


def test_verifier_rejects_control_without_margin(tmp_path: Path):
    module = _load_module()
    report = _report()
    report["splits"]["validation"]["rowwise_directional_mse"]["mean"] = [
        0.31
    ] * 120
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    result = module.verify(_config(report_path))

    assert result["passed"] is False
    assert "validation primary does not beat control mean" in result["errors"]


def test_test_config_binds_validation_boundary_hash(tmp_path: Path):
    module = _load_module()
    report_path = tmp_path / "report.json"
    config = _config(report_path)
    config["schema_version"] = module.SCHEMA_VERSION
    config["output_json"] = str(tmp_path / "verified.json")
    expected = config["expected"]
    expected["eval_splits"] = ["test"]
    expected["counts"] = {"test": 120}
    expected["generation_identity"]["dataset_sha256"]["test"] = "e" * 64
    config_path = tmp_path / "verify.yaml"
    config_path.write_text(yaml.safe_dump(config))

    loaded = module._load_config(config_path)

    assert set(loaded["expected"]["generation_identity"]["dataset_sha256"]) == {
        "train",
        "validation",
        "test",
    }
