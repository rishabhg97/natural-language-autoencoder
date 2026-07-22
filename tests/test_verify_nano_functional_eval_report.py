from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "verify_nano_functional_eval_report.py"
    spec = importlib.util.spec_from_file_location(
        "verify_nano_functional_eval_report", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def passing_report() -> dict:
    metrics = (
        "kl_original_to_patched",
        "js_divergence",
        "logit_pearson",
        "top_10_overlap",
        "top_50_overlap",
        "original_top1_rank",
    )
    effect = {
        "unit": "cluster",
        "row_count": 512,
        "cluster_count": 120,
        "mean_improvement": 0.1,
        "ci95_low": 0.01,
        "ci95_high": 0.2,
        "candidate_better_fraction": 0.7,
    }
    return {
        "metadata": {
            "eval_splits": ["validation"],
            "ar_checkpoint_dir": "/checkpoints/ar",
            "target_model": "/models/nano",
            "source_base_parquet": "/data/base.parquet",
            "mean_activation_parquet": "/data/train.parquet",
            "boundary": 33,
            "independent_family_count": 120,
            "generation_identity": {
                "protocol": {"prefix": ""},
                "provenance": {
                    "model_fingerprint": "dcp_model_sha256:" + "a" * 64,
                    "tokenizer_fingerprint": "tokenizer_files_sha256:" + "b" * 64,
                    "datasets": {
                        "train": {"sha256": "c" * 64},
                        "validation": {"sha256": "d" * 64},
                    },
                },
            },
        },
        "gate": {
            "identity_passed": True,
            "stored_activation_replay_within_tolerance": True,
        },
        "splits": {
            "validation": {
                "variants": {
                    name: {"row_count": 512}
                    for name in (
                        "candidate",
                        "teacher",
                        "stored_gold",
                        "mean",
                        "zero",
                        "shuffled",
                    )
                },
                "paired_candidate_vs_variants": {
                    control: {metric: dict(effect) for metric in metrics}
                    for control in ("mean", "zero", "shuffled")
                },
            }
        },
    }


def passing_config(report_path: pathlib.Path) -> dict:
    variants = (
        "candidate",
        "teacher",
        "stored_gold",
        "mean",
        "zero",
        "shuffled",
    )
    return {
        "report_json": str(report_path),
        "expected": {
            "eval_splits": ["validation"],
            "ar_checkpoint_dir": "/checkpoints/ar",
            "target_model": "/models/nano",
            "source_base_parquet": "/data/base.parquet",
            "mean_activation_parquet": "/data/train.parquet",
            "boundary": 33,
            "generation_identity": {
                "model_fingerprint": "dcp_model_sha256:" + "a" * 64,
                "tokenizer_fingerprint": "tokenizer_files_sha256:" + "b" * 64,
                "dataset_sha256": {
                    "train": "c" * 64,
                    "validation": "d" * 64,
                },
            },
            "counts": {"validation": 512},
            "min_independent_families": 100,
            "required_variants": list(variants),
            "variant_counts": {name: 512 for name in variants},
            "candidate_control_variants": ["mean", "zero", "shuffled"],
            "control_min_independent_families": {
                "mean": 100,
                "zero": 100,
                "shuffled": 30,
            },
            "positive_ci_metrics": [
                "kl_original_to_patched",
                "js_divergence",
            ],
            "positive_mean_metrics": [
                "logit_pearson",
                "top_10_overlap",
                "top_50_overlap",
                "original_top1_rank",
            ],
            "min_family_better_fraction": 0.5,
            "forbid_unrequested_rows": True,
        },
    }


class VerifyNanoFunctionalEvalReportTests(unittest.TestCase):
    def _verify(self, report: dict, mutate_config=None):
        verifier = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            report_path = pathlib.Path(tmp) / "report.json"
            report_path.write_text(json.dumps(report))
            config = passing_config(report_path)
            if mutate_config is not None:
                mutate_config(config)
            return verifier.verify(config)

    def test_passes_identity_bound_clustered_control_comparisons(self):
        result = self._verify(passing_report())

        self.assertTrue(result["passed"])
        self.assertEqual(result["independent_family_count"], 120)

    def test_rejects_different_content_addressed_model(self):
        report = passing_report()
        report["metadata"]["generation_identity"]["provenance"][
            "model_fingerprint"
        ] = "dcp_model_sha256:" + "e" * 64

        result = self._verify(report)

        self.assertFalse(result["passed"])
        self.assertIn(
            "generation model fingerprint does not match preregistration",
            result["errors"],
        )

    def test_rejects_different_content_addressed_dataset(self):
        report = passing_report()
        report["metadata"]["generation_identity"]["provenance"]["datasets"][
            "validation"
        ]["sha256"] = "e" * 64

        result = self._verify(report)

        self.assertFalse(result["passed"])
        self.assertIn(
            "generation dataset hash mismatch for validation",
            result["errors"],
        )

    def test_rejects_truncated_paired_comparison(self):
        report = passing_report()
        report["splits"]["validation"]["paired_candidate_vs_variants"]["mean"][
            "kl_original_to_patched"
        ]["row_count"] = 100

        result = self._verify(report)

        self.assertFalse(result["passed"])
        self.assertIn(
            "validation candidate does not pass kl_original_to_patched versus mean",
            result["errors"],
        )

    def test_test_config_binds_validation_boundary_hash(self):
        verifier = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = passing_config(root / "report.json")
            config["schema_version"] = verifier.SCHEMA_VERSION
            config["output_json"] = str(root / "verified.json")
            expected = config["expected"]
            expected["eval_splits"] = ["test"]
            expected["counts"] = {"test": 512}
            expected["generation_identity"]["dataset_sha256"]["test"] = "e" * 64
            config_path = root / "verify.yaml"
            config_path.write_text(yaml.safe_dump(config))

            loaded = verifier._load_config(config_path)

        self.assertEqual(
            set(loaded["expected"]["generation_identity"]["dataset_sha256"]),
            {"train", "validation", "test"},
        )


if __name__ == "__main__":
    unittest.main()
