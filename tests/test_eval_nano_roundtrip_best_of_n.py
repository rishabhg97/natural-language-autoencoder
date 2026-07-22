import importlib.util
import hashlib
import json
import pathlib
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "eval_nano_roundtrip_best_of_n.py"
    spec = importlib.util.spec_from_file_location("eval_nano_roundtrip_best_of_n", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoRoundtripBestOfNTests(unittest.TestCase):
    def test_matched_baseline_uses_only_selected_rows(self):
        module = load_script()
        report = {
            "splits": {
                "validation": {
                    "row_indices": [10, 20, 30],
                    "rowwise_directional_mse": {
                        "av_real": [0.1, 0.9, 0.3]
                    },
                }
            }
        }

        value = module.matched_report_directional_nmse(report, [30, 10])

        self.assertAlmostEqual(value, 0.2)
        with self.assertRaises(module.BestOfNError):
            module.matched_report_directional_nmse(report, [40])

    def test_oracle_curve_is_nested_and_improves_monotonically(self):
        module = load_script()
        losses = np.asarray(
            [
                [0.5, 0.4, 0.6],
                [0.3, 0.5, 0.4],
                [0.4, 0.2, 0.5],
                [0.2, 0.3, 0.3],
            ]
        )

        curves = module.summarize_best_of_n(losses, n_values=[1, 2, 4])

        self.assertGreaterEqual(
            curves["1"]["oracle_directional_nmse"],
            curves["2"]["oracle_directional_nmse"],
        )
        self.assertGreaterEqual(
            curves["2"]["oracle_directional_nmse"],
            curves["4"]["oracle_directional_nmse"],
        )
        self.assertAlmostEqual(curves["4"]["oracle_directional_nmse"], 7.0 / 30.0)

    def test_analysis_adds_paired_family_intervals_and_gain_fraction(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)

            def write_report(path, losses):
                path.write_text(
                    json.dumps(
                        {
                            "splits": {
                                "validation": {
                                    "row_indices": [10, 20],
                                    "content_family_ids": ["a", "b"],
                                    "rowwise_directional_mse": {
                                        "av_real": losses
                                    },
                                }
                            }
                        }
                    )
                )

            scores = root / "scores"
            scores.mkdir()
            write_report(scores / "sample_00.json", [0.4, 0.5])
            write_report(scores / "sample_01.json", [0.2, 0.6])
            greedy = root / "greedy.json"
            rl = root / "rl.json"
            write_report(greedy, [0.5, 0.7])
            write_report(rl, [0.3, 0.4])
            config = {
                "paths": {
                    "output_root": str(root),
                    "greedy_sft_report": str(greedy),
                    "matched_rl_report": str(rl),
                },
                "sampling": {
                    "samples_per_row": 2,
                    "n_values": [1, 2],
                    "seed": 7,
                },
                "analysis": {"bootstrap_samples": 1000, "seed": 11},
            }

            result = module.analyze(config)

        best = result["oracle_curves"]["2"]
        self.assertAlmostEqual(best["oracle_directional_nmse"], 0.35)
        self.assertAlmostEqual(best["improvement_vs_greedy_sft"], 0.25)
        self.assertAlmostEqual(best["improvement_vs_matched_rl"], 0.0)
        self.assertAlmostEqual(best["fraction_of_matched_rl_gain_explained"], 1.0)
        self.assertEqual(
            best["paired_comparisons"]["matched_rl"]["families"], 2
        )

    def test_sample_provenance_uses_prepared_checkpoint_and_current_datasets(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            train = root / "train.parquet"
            validation = root / "validation.parquet"
            train.write_bytes(b"train")
            validation.write_bytes(b"validation")
            config = {
                "paths": {
                    "train_parquet": str(train),
                    "validation_parquet": str(validation),
                }
            }
            prepared = {
                "output_hf_dir": str(root / "prepared-hf"),
                "av_model_fingerprint": "dcp_model_sha256:model",
                "av_tokenizer_fingerprint": "tokenizer_files_sha256:tokenizer",
            }

            provenance = module._sample_provenance(config, prepared)

            self.assertEqual(provenance["checkpoint"], str(root / "prepared-hf"))
            self.assertEqual(provenance["model_fingerprint"], prepared["av_model_fingerprint"])
            self.assertEqual(
                provenance["tokenizer_fingerprint"],
                prepared["av_tokenizer_fingerprint"],
            )
            self.assertEqual(
                provenance["datasets"]["train"]["sha256"],
                hashlib.sha256(train.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                provenance["datasets"]["validation"]["sha256"],
                hashlib.sha256(validation.read_bytes()).hexdigest(),
            )

    def test_score_config_pins_emitted_sampling_protocol(self):
        module = load_script()
        config = {
            "paths": {
                "code_root": "/code",
                "output_root": "/out",
                "ar_checkpoint_dir": "/critic",
                "train_parquet": "/data/train.parquet",
                "validation_parquet": "/data/validation.parquet",
                "content_family_manifest": "/data/families.json",
                "content_family_coverage": "/data/coverage.json",
            },
            "av_checkpoint_prepare": {
                "output_hf_dir": "/tmp/prepared-hf",
                "expected_model_fingerprint": "dcp_model_sha256:model",
                "expected_tokenizer_fingerprint": "tokenizer_files_sha256:tokenizer",
            },
            "evaluation": {
                "validation_limit": 122,
                "max_new_tokens": 384,
                "selection_seed": 20260718,
                "min_independent_families": 122,
            },
            "sampling": {"seed": 20260722},
            "execution": {"python": "/venv/bin/python"},
        }

        score_config = module._score_config(config, sample_index=3)

        self.assertEqual(
            score_config["paths"]["expected_generation_protocol_json"],
            "/out/generation_protocol.json",
        )
        self.assertEqual(
            score_config["paths"]["generated_jsonl"],
            "/out/generated/sample_03.jsonl",
        )
        self.assertTrue(score_config["eval"]["reuse_generated"])

    def test_source_records_allow_a_canary_subset(self):
        module = load_script()
        source_records = [
            {"row_index": 10, "target_explanation": "ten"},
            {"row_index": 20, "target_explanation": "twenty"},
            {"row_index": 30, "target_explanation": "thirty"},
        ]

        selected = module._source_records_for_selected(source_records, [30, 10])

        self.assertEqual(selected[30]["target_explanation"], "thirty")
        self.assertEqual(selected[10]["target_explanation"], "ten")

    def test_source_records_reject_missing_or_duplicate_rows(self):
        module = load_script()
        with self.assertRaises(module.BestOfNError):
            module._source_records_for_selected([{"row_index": 1}], [2])
        with self.assertRaises(module.BestOfNError):
            module._source_records_for_selected(
                [{"row_index": 1}, {"row_index": 1}], [1]
            )


if __name__ == "__main__":
    unittest.main()
