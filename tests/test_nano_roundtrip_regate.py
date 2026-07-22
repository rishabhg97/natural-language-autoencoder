import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "nano_roundtrip_regate.py"
    spec = importlib.util.spec_from_file_location("nano_roundtrip_regate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regate_uses_stable_rows_and_strict_promotion_thresholds():
    regate = load_script()
    controls = ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
    candidate_losses = [0.19, 0.21]
    baseline_losses = [0.20, 0.20]
    row_keys = [
        {"doc_id": "doc-a", "n_raw_tokens": 11},
        {"doc_id": "doc-b", "n_raw_tokens": 13},
    ]

    def split(losses):
        return {
            "row_indices": [4, 8],
            "row_keys": row_keys,
            "doc_ids": ["doc-a", "doc-b"],
            "content_family_ids": ["family-a", "family-b"],
            "rowwise_directional_mse": {"av_real": losses},
            "rowwise_normalized_mse": {"av_real": losses},
            "rowwise_raw_mse": {"av_real": losses},
            "variants": {
                "av_real": {
                    "directional_mse": sum(losses) / len(losses),
                    "normalized_mse": sum(losses) / len(losses),
                    "raw_mse": sum(losses) / len(losses),
                    "centered_r2": 0.0,
                    "norm_ratio_mean": 1.0,
                },
                **{
                    name: {
                        "directional_mse": 0.5,
                        "normalized_mse": 0.5,
                        "raw_mse": 0.5,
                        "centered_r2": 0.0,
                        "norm_ratio_mean": 1.0,
                    }
                    for name in controls
                },
                "teacher": {
                    "directional_mse": 0.2,
                    "normalized_mse": 0.2,
                    "raw_mse": 0.2,
                    "centered_r2": 0.0,
                    "norm_ratio_mean": 1.0,
                },
            },
            "rowwise_win_rates": {
                f"av_real_vs_{name}": {"candidate_better_fraction": 1.0}
                for name in controls
            },
            "generation_parse": {
                "real": {"closed_fraction": 1.0, "usable_fraction": 1.0}
            },
        }

    provenance = {"train": {"sha256": "a"}, "validation": {"sha256": "b"}}
    protocol = {"schema_version": "nano_generation_protocol.v1", "prefix": ""}
    candidate = {
        "splits": {"validation": split(candidate_losses)},
        "dataset_provenance": provenance,
        "generation_protocol": protocol,
    }
    baseline = {
        "splits": {"validation": split(baseline_losses)},
        "dataset_provenance": provenance,
        "generation_protocol": protocol,
    }
    config = {
        "control_margin": 0.1,
        "baseline_margin": 0.0,
        "min_control_win_fraction": 0.8,
        "min_baseline_win_fraction": 0.6,
        "min_baseline_relative_improvement": 0.1,
        "require_baseline_ci_positive": True,
        "require_clustered_baseline_ci": True,
        "require_baseline_dataset_match": True,
        "bootstrap_samples": 100,
        "bootstrap_seed": 7,
        "permutation_samples": 100,
        "permutation_seed": 7,
        "min_closed_fraction": 0.95,
        "min_usable_fraction": 0.99,
        "require_generation_protocol_match": True,
        "require_family_level_inference": True,
        "min_independent_families": 2,
    }

    output = regate.regate_reports(candidate, baseline, gate_config=config)

    assert output["gate"]["passed"] is False
    validation = output["gate"]["splits"]["validation"]
    assert validation["baseline_row_identity_kind"] == "row_key"
    assert validation["baseline_row_overlap_count"] == 2
    assert validation["baseline_beaten"] is False
