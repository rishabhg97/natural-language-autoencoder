from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RoundtripGateTests(unittest.TestCase):
    def test_prediction_cache_round_trips_metadata_and_arrays(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        with tempfile.TemporaryDirectory() as tmp:
            cache = pathlib.Path(tmp) / "predictions.npz"
            gate.write_prediction_cache(
                cache,
                split_payloads={
                    "validation": {
                        "row_indices": [7, 9],
                        "doc_ids": ["doc-7", "doc-9"],
                        "content_family_ids": ["family-a", "family-b"],
                        "targets": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                        "predictions": {
                            "teacher": np.asarray([[1.1, 2.1], [3.1, 4.1]]),
                            "av_real": np.asarray([[0.9, 1.9], [2.9, 3.9]]),
                        },
                    }
                },
                train_mean=np.asarray([0.5, 1.5]),
                metadata={"schema_version": gate.PREDICTION_CACHE_SCHEMA_VERSION},
            )

            with np.load(cache, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
                self.assertEqual(
                    metadata["schema_version"], gate.PREDICTION_CACHE_SCHEMA_VERSION
                )
                np.testing.assert_allclose(payload["train_mean"], [0.5, 1.5])
                np.testing.assert_allclose(
                    payload["validation__prediction__av_real"],
                    [[0.9, 1.9], [2.9, 3.9]],
                )
                self.assertEqual(
                    payload["validation__content_family_ids"].tolist(),
                    ["family-a", "family-b"],
                )

    def test_select_eval_indices_respects_validation_only_contract(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        rows = [
            {"row_index": 0, "split": "train"},
            {"row_index": 1, "split": "validation"},
            {"row_index": 2, "split": "validation"},
            {"row_index": 3, "split": "test"},
            {"row_index": 4, "split": "test"},
        ]

        selected = gate.select_eval_indices_by_split(
            rows,
            validation_indices=[1, 2],
            test_indices=[3, 4],
            validation_limit=1,
            test_limit=1,
            eval_splits=["validation"],
            strategy="row_order",
            seed=17,
        )

        self.assertEqual(selected, {"validation": [1]})
        self.assertNotIn("test", selected)

    def test_load_eval_rows_does_not_open_unrequested_test_split(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        def fake_read(path, split, offset):
            if split == "test":
                raise AssertionError("sealed test split was opened")
            return [
                {
                    "row_index": offset,
                    "source_row_index": 0,
                    "split": split,
                    "activation_vector": [1.0, 0.0],
                }
            ]

        with mock.patch.object(gate, "_read_rows", side_effect=fake_read):
            rows, train, validation, test = gate.load_eval_rows(
                pathlib.Path("train.parquet"),
                pathlib.Path("validation.parquet"),
                pathlib.Path("sealed-test.parquet"),
                eval_splits=["validation"],
            )

        self.assertEqual([row["split"] for row in rows], ["train", "validation"])
        self.assertEqual(train, [0])
        self.assertEqual(validation, [1])
        self.assertEqual(test, [])

    def test_validation_only_gate_has_no_phantom_test_split(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        variants = {
            "av_real": {
                "directional_mse": 0.2,
                "raw_mse": 2.0,
                "centered_r2": 0.1,
                "norm_ratio_mean": 1.0,
            },
            **{
                control: {"directional_mse": 0.5}
                for control in ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
            },
        }
        split = {
            "variants": variants,
            "generation_parse": {
                "real": {"closed_fraction": 1.0, "usable_fraction": 1.0}
            },
            "rowwise_win_rates": {
                f"av_real_vs_{control}": {"candidate_better_fraction": 1.0}
                for control in ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
            },
        }

        summary = gate.build_gate_summary(
            {"validation": split},
            control_margin=0.1,
            generation_protocol={"prefix": "", "backend": "legacy_batch"},
            require_generation_protocol_match=True,
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(set(summary["splits"]), {"validation"})
        self.assertTrue(summary["current_generation_protocol_compatible"])

    def test_activation_metric_schema_requires_raw_and_norm_diagnostics(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        incomplete = {
            "validation": {
                "variants": {"av_real": {"directional_mse": 0.2}}
            }
        }

        with self.assertRaisesRegex(ValueError, "missing required metrics"):
            gate.validate_activation_metric_reports(incomplete)

        gate.validate_activation_metric_reports(
            {
                "validation": {
                    "variants": {
                        "av_real": {
                            "directional_mse": 0.2,
                            "raw_mse": 2.0,
                            "centered_r2": -0.1,
                            "norm_ratio_mean": 0.9,
                        }
                    }
                }
            }
        )

    def test_generated_record_metadata_preserves_stable_provenance(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        row = {
            "row_index": 17,
            "source_row_index": 3,
            "split": "validation",
            "doc_id": "doc-9",
            "n_raw_tokens": 22,
            "token_position": 21,
            "token_id": 77,
            "sample_uuid": "sample-uuid",
        }

        metadata = gate.generated_record_metadata(row)

        for key, value in row.items():
            self.assertEqual(metadata[key], value)

    def test_teacher_prompt_extracts_explanation_tags(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        row = {"response": "<explanation>  concise summary  </explanation>"}

        self.assertEqual(gate.target_explanation(row), "concise summary")
        self.assertEqual(
            gate.teacher_prompt_for_row(row, "Summary: <text>{explanation}</text> <summary>"),
            "Summary: <text>concise summary</text> <summary>",
        )

    def test_summarize_variant_predictions_tracks_primary_wins(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        targets = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        train_targets = targets.copy()
        predictions = {
            "teacher": targets.copy(),
            "av_real": np.asarray([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32),
            "av_shuffled": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            "av_zero": np.zeros((2, 2), dtype=np.float32),
            "av_mean": np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
            "av_none": np.asarray([[0.2, 0.8], [0.8, 0.2]], dtype=np.float32),
            "mean": np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        }

        summary = gate.summarize_variant_predictions(
            predictions,
            targets=targets,
            train_targets=train_targets,
        )

        self.assertLess(
            summary["variants"]["av_real"]["normalized_mse"],
            summary["variants"]["av_shuffled"]["normalized_mse"],
        )
        self.assertEqual(
            summary["rowwise_win_rates"]["av_real_vs_av_shuffled"]["candidate_better_count"],
            2,
        )
        self.assertEqual(
            summary["rowwise_win_rates"]["teacher_vs_av_real"]["candidate"],
            "teacher",
        )
        self.assertEqual(
            summary["variants"]["av_real"]["directional_mse"],
            summary["variants"]["av_real"]["normalized_mse"],
        )
        self.assertIn("raw_mse", summary["variants"]["av_real"])
        self.assertIn("rowwise_directional_mse", summary)
        self.assertIn("rowwise_raw_mse", summary)
        self.assertEqual(
            summary["rowwise_directional_mse"],
            summary["rowwise_normalized_mse"],
        )

    def test_scaled_prediction_is_direction_perfect_but_raw_incorrect(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        targets = np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
        predictions = {"av_real": 10.0 * targets}

        summary = gate.summarize_variant_predictions(
            predictions,
            targets=targets,
            train_targets=targets,
        )

        metrics = summary["variants"]["av_real"]
        self.assertAlmostEqual(metrics["directional_mse"], 0.0, places=12)
        self.assertGreater(metrics["raw_mse"], 0.0)
        np.testing.assert_allclose(summary["rowwise_directional_mse"]["av_real"], [0.0, 0.0], atol=1e-12)
        self.assertTrue(all(value > 0.0 for value in summary["rowwise_raw_mse"]["av_real"]))

    def test_gate_reports_paired_directional_and_raw_baseline_effects(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        candidate_directional = [0.10, 0.12]
        baseline_directional = [0.20, 0.22]
        candidate_raw = [10.0, 12.0]
        baseline_raw = [20.0, 24.0]
        split = {
            "row_indices": [1, 2],
            "doc_ids": ["doc-a", "doc-b"],
            "rowwise_directional_mse": {"av_real": candidate_directional},
            "rowwise_normalized_mse": {"av_real": candidate_directional},
            "rowwise_raw_mse": {"av_real": candidate_raw},
            "generation_parse": {"real": {"closed_fraction": 1.0, "usable_fraction": 1.0}},
            "variants": {
                "av_real": {"directional_mse": 0.11, "normalized_mse": 0.11, "raw_mse": 11.0},
                "av_shuffled": {"directional_mse": 0.50, "normalized_mse": 0.50},
                "av_zero": {"directional_mse": 0.51, "normalized_mse": 0.51},
                "av_mean": {"directional_mse": 0.52, "normalized_mse": 0.52},
                "av_none": {"directional_mse": 0.53, "normalized_mse": 0.53},
                "mean": {"directional_mse": 0.54, "normalized_mse": 0.54},
            },
            "rowwise_win_rates": {
                f"av_real_vs_{control}": {"candidate_better_fraction": 1.0}
                for control in ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
            },
        }
        baseline = {
            "splits": {
                name: {
                    "row_indices": [1, 2],
                    "rowwise_directional_mse": {"av_real": baseline_directional},
                    "rowwise_normalized_mse": {"av_real": baseline_directional},
                    "rowwise_raw_mse": {"av_real": baseline_raw},
                    "variants": {
                        "av_real": {
                            "directional_mse": 0.21,
                            "normalized_mse": 0.21,
                            "raw_mse": 22.0,
                        }
                    },
                }
                for name in ("validation", "test")
            }
        }

        summary = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            bootstrap_samples=100,
        )

        validation = summary["splits"]["validation"]
        effects = validation["baseline_paired_improvement_by_metric"]
        self.assertEqual(set(effects), {"directional_mse", "raw_mse"})
        self.assertAlmostEqual(effects["directional_mse"]["mean_delta_baseline_minus_candidate"], 0.1)
        self.assertAlmostEqual(effects["raw_mse"]["mean_delta_baseline_minus_candidate"], 11.0)
        self.assertEqual(
            validation["baseline_paired_improvement"],
            effects["directional_mse"],
        )

    def test_gate_summary_requires_controls_and_optional_baseline(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        split = {
            "row_indices": [1, 2],
            "rowwise_normalized_mse": {"av_real": [0.20, 0.30]},
            "generation_parse": {"real": {"closed_fraction": 1.0, "usable_fraction": 1.0}},
            "variants": {
                "av_real": {"normalized_mse": 0.25},
                "av_shuffled": {"normalized_mse": 0.60},
                "av_zero": {"normalized_mse": 0.65},
                "av_mean": {"normalized_mse": 0.58},
                "av_none": {"normalized_mse": 0.61},
                "mean": {"normalized_mse": 0.70},
                "teacher": {"normalized_mse": 0.20},
            }
        }
        baseline = {
            "splits": {
                "validation": {
                    "row_indices": [1, 2],
                    "rowwise_normalized_mse": {"av_real": [0.32, 0.40]},
                    "variants": {"av_real": {"normalized_mse": 0.35}},
                },
                "test": {
                    "row_indices": [1, 2],
                    "rowwise_normalized_mse": {"av_real": [0.33, 0.41]},
                    "variants": {"av_real": {"normalized_mse": 0.36}},
                },
            }
        }

        summary = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            baseline_margin=0.02,
            min_closed_fraction=0.8,
            min_usable_fraction=0.95,
        )

        self.assertTrue(summary["passed"])
        self.assertTrue(summary["baseline_required"])
        self.assertTrue(summary["splits"]["validation"]["baseline_beaten"])
        self.assertTrue(summary["splits"]["validation"]["baseline_row_identity_match"])
        self.assertEqual(
            summary["splits"]["validation"]["baseline_rowwise_win_rate"]["candidate_better_count"],
            2,
        )

    def test_gate_summary_compares_baseline_on_overlapping_row_indices(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        split = {
            "row_indices": [11, 13],
            "rowwise_normalized_mse": {"av_real": [0.10, 0.12]},
            "generation_parse": {"real": {"closed_fraction": 1.0, "usable_fraction": 1.0}},
            "variants": {
                "av_real": {"normalized_mse": 0.11},
                "av_shuffled": {"normalized_mse": 0.30},
                "av_zero": {"normalized_mse": 0.31},
                "av_mean": {"normalized_mse": 0.32},
                "av_none": {"normalized_mse": 0.33},
                "mean": {"normalized_mse": 0.34},
                "teacher": {"normalized_mse": 0.09},
            },
            "rowwise_win_rates": {
                f"av_real_vs_{control}": {"candidate_better_fraction": 1.0}
                for control in ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
            },
        }
        baseline = {
            "splits": {
                split_name: {
                    "row_indices": [10, 11, 12, 13],
                    "rowwise_normalized_mse": {"av_real": [0.20, 0.16, 0.18, 0.15]},
                    "variants": {"av_real": {"normalized_mse": 0.1725}},
                }
                for split_name in ("validation", "test")
            }
        }

        summary = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            baseline_margin=0.0,
            min_control_win_fraction=0.9,
            min_closed_fraction=0.8,
            min_usable_fraction=0.95,
        )

        validation = summary["splits"]["validation"]
        self.assertTrue(summary["passed"])
        self.assertFalse(validation["baseline_row_identity_match"])
        self.assertEqual(validation["baseline_row_overlap_count"], 2)
        self.assertAlmostEqual(validation["baseline_primary_matched_normalized_mse"], 0.155)
        self.assertTrue(validation["baseline_beaten"])
        self.assertEqual(validation["baseline_rowwise_win_rate"]["candidate_better_count"], 2)

    def test_gate_summary_prefers_stable_row_keys_for_subset_alignment(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        controls = ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
        split = {
            "row_indices": [101, 303],
            "row_keys": [
                {"doc_id": "doc-a", "n_raw_tokens": 11},
                {"doc_id": "doc-b", "n_raw_tokens": 13},
            ],
            "doc_ids": ["doc-a", "doc-b"],
            "content_family_ids": ["family-a", "family-b"],
            "rowwise_normalized_mse": {"av_real": [0.10, 0.12]},
            "generation_parse": {
                "real": {"closed_fraction": 1.0, "usable_fraction": 1.0}
            },
            "variants": {
                "av_real": {"normalized_mse": 0.11},
                **{name: {"normalized_mse": 0.50} for name in controls},
                "teacher": {"normalized_mse": 0.09},
            },
            "rowwise_win_rates": {
                f"av_real_vs_{control}": {"candidate_better_fraction": 1.0}
                for control in controls
            },
        }
        baseline = {
            "splits": {
                split_name: {
                    "row_indices": [0, 1, 2, 3],
                    "row_keys": [
                        {"doc_id": "doc-x", "n_raw_tokens": 7},
                        {"n_raw_tokens": 13, "doc_id": "doc-b"},
                        {"doc_id": "doc-y", "n_raw_tokens": 9},
                        {"n_raw_tokens": 11, "doc_id": "doc-a"},
                    ],
                    "rowwise_normalized_mse": {
                        "av_real": [0.20, 0.15, 0.18, 0.16]
                    },
                    "variants": {"av_real": {"normalized_mse": 0.1725}},
                }
                for split_name in ("validation", "test")
            }
        }

        summary = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            baseline_margin=0.0,
            min_control_win_fraction=0.9,
            min_closed_fraction=0.8,
            min_usable_fraction=0.95,
        )

        validation = summary["splits"]["validation"]
        self.assertTrue(summary["passed"])
        self.assertEqual(validation["baseline_row_identity_kind"], "row_key")
        self.assertFalse(validation["baseline_row_identity_match"])
        self.assertEqual(validation["baseline_row_overlap_count"], 2)
        self.assertAlmostEqual(
            validation["baseline_primary_matched_normalized_mse"], 0.155
        )
        self.assertEqual(
            validation["baseline_paired_improvement"]["independent_unit_count"],
            2,
        )

    def test_gate_requires_clustered_ci_and_dataset_binding_when_configured(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        row_indices = [1, 2, 3, 4]
        candidate = [0.10, 0.11, 0.12, 0.13]
        baseline_losses = [0.20, 0.21, 0.22, 0.23]
        variants = {
            "av_real": {"normalized_mse": float(np.mean(candidate))},
            "av_shuffled": {"normalized_mse": 0.50},
            "av_zero": {"normalized_mse": 0.51},
            "av_mean": {"normalized_mse": 0.52},
            "av_none": {"normalized_mse": 0.53},
            "mean": {"normalized_mse": 0.54},
            "teacher": {"normalized_mse": 0.09},
        }
        split = {
            "row_indices": row_indices,
            "doc_ids": ["doc-a", "doc-a", "doc-b", "doc-b"],
            "rowwise_normalized_mse": {"av_real": candidate},
            "generation_parse": {"real": {"closed_fraction": 1.0, "usable_fraction": 1.0}},
            "variants": variants,
            "rowwise_win_rates": {
                f"av_real_vs_{control}": {"candidate_better_fraction": 1.0}
                for control in ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
            },
        }
        provenance = {
            name: {"sha256": name * 8}
            for name in ("train", "validation", "test")
        }
        baseline = {
            "dataset_provenance": provenance,
            "splits": {
                name: {
                    "row_indices": row_indices,
                    "rowwise_normalized_mse": {"av_real": baseline_losses},
                    "variants": {"av_real": {"normalized_mse": float(np.mean(baseline_losses))}},
                }
                for name in ("validation", "test")
            },
        }

        summary = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            dataset_provenance=provenance,
            min_baseline_win_fraction=0.5,
            min_baseline_relative_improvement=0.1,
            require_baseline_ci_positive=True,
            require_clustered_baseline_ci=True,
            require_baseline_dataset_match=True,
            bootstrap_samples=100,
        )

        self.assertTrue(summary["passed"])
        paired = summary["splits"]["test"]["baseline_paired_improvement"]
        self.assertEqual(paired["independent_unit"], "doc_id")
        self.assertEqual(paired["independent_unit_count"], 2)

        family_split = dict(split)
        family_split["content_family_ids"] = [
            "family-a",
            "family-a",
            "family-b",
            "family-b",
        ]
        family_clustered = gate.build_gate_summary(
            {"validation": family_split, "test": family_split},
            control_margin=0.01,
            baseline_report=baseline,
            dataset_provenance=provenance,
            min_baseline_win_fraction=0.5,
            min_baseline_relative_improvement=0.1,
            require_baseline_ci_positive=True,
            require_clustered_baseline_ci=True,
            require_baseline_dataset_match=True,
            require_family_level_inference=True,
            min_independent_families=2,
            bootstrap_samples=100,
        )

        self.assertTrue(family_clustered["passed"])
        family_paired = family_clustered["splits"]["test"][
            "baseline_paired_improvement"
        ]
        self.assertEqual(family_paired["independent_unit"], "content_family_id")
        self.assertEqual(family_paired["independent_unit_count"], 2)

        mismatched = dict(provenance)
        mismatched["test"] = {"sha256": "different"}
        rejected = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            dataset_provenance=mismatched,
            require_baseline_dataset_match=True,
            bootstrap_samples=100,
        )
        self.assertFalse(rejected["passed"])

    def test_paired_improvement_uses_content_families_and_sign_flip_inference(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        summary = gate.paired_improvement_summary(
            np.asarray([0.10, 0.12, 0.20, 0.22]),
            np.asarray([0.30, 0.32, 0.40, 0.42]),
            content_family_ids=["family-a", "family-a", "family-b", "family-b"],
            bootstrap_samples=500,
            bootstrap_seed=7,
            permutation_samples=1_000,
            permutation_seed=11,
        )

        self.assertEqual(summary["independent_unit"], "content_family_id")
        self.assertEqual(summary["independent_unit_count"], 2)
        self.assertGreater(summary["bootstrap_ci95_low"], 0.0)
        self.assertEqual(summary["sign_flip_method"], "exact")
        self.assertAlmostEqual(summary["sign_flip_p_value"], 0.25)

    def test_paired_improvement_exercises_monte_carlo_sign_flip_branch(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        families = [f"family-{index:02d}" for index in range(21)]
        candidate = np.linspace(0.10, 0.20, len(families))
        baseline = candidate + 0.02

        first = gate.paired_improvement_summary(
            candidate,
            baseline,
            content_family_ids=families,
            bootstrap_samples=200,
            bootstrap_seed=7,
            permutation_samples=2_000,
            permutation_seed=11,
        )
        second = gate.paired_improvement_summary(
            candidate,
            baseline,
            content_family_ids=families,
            bootstrap_samples=200,
            bootstrap_seed=7,
            permutation_samples=2_000,
            permutation_seed=11,
        )

        self.assertEqual(first["sign_flip_method"], "monte_carlo")
        self.assertEqual(first["sign_flip_p_value"], second["sign_flip_p_value"])
        self.assertGreaterEqual(first["sign_flip_p_value"], 0.0)
        self.assertLessEqual(first["sign_flip_p_value"], 1.0)

    def test_publication_gate_requires_enough_content_families_or_marks_pilot(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        row_indices = [1, 2, 3, 4]
        candidate = [0.10, 0.11, 0.12, 0.13]
        baseline_losses = [0.20, 0.21, 0.22, 0.23]
        split = {
            "row_indices": row_indices,
            "doc_ids": ["doc-a", "doc-a", "doc-b", "doc-b"],
            "content_family_ids": ["family-a", "family-a", "family-b", "family-b"],
            "rowwise_directional_mse": {"av_real": candidate},
            "rowwise_normalized_mse": {"av_real": candidate},
            "generation_parse": {"real": {"closed_fraction": 1.0, "usable_fraction": 1.0}},
            "variants": {
                "av_real": {"directional_mse": 0.115, "normalized_mse": 0.115},
                "av_shuffled": {"normalized_mse": 0.50},
                "av_zero": {"normalized_mse": 0.51},
                "av_mean": {"normalized_mse": 0.52},
                "av_none": {"normalized_mse": 0.53},
                "mean": {"normalized_mse": 0.54},
            },
            "rowwise_win_rates": {
                f"av_real_vs_{control}": {"candidate_better_fraction": 1.0}
                for control in ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
            },
        }
        baseline = {
            "splits": {
                name: {
                    "row_indices": row_indices,
                    "rowwise_directional_mse": {"av_real": baseline_losses},
                    "rowwise_normalized_mse": {"av_real": baseline_losses},
                    "variants": {"av_real": {"normalized_mse": 0.215}},
                }
                for name in ("validation", "test")
            }
        }

        pilot = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            require_family_level_inference=True,
            min_independent_families=100,
            bootstrap_samples=100,
        )
        adequately_sized_for_fixture = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            require_family_level_inference=True,
            min_independent_families=2,
            bootstrap_samples=100,
        )

        self.assertFalse(pilot["passed"])
        self.assertEqual(pilot["publication_status"], "small_sample_pilot")
        self.assertTrue(adequately_sized_for_fixture["passed"])
        self.assertEqual(adequately_sized_for_fixture["publication_status"], "confirmatory")
        paired = adequately_sized_for_fixture["splits"]["test"]["baseline_paired_improvement"]
        self.assertEqual(paired["independent_unit"], "content_family_id")

    def test_family_requirement_is_enforced_without_external_baseline(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        controls = ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
        split = {
            "content_family_ids": ["family-a", "family-b"],
            "variants": {
                "av_real": {"normalized_mse": 0.10},
                **{name: {"normalized_mse": 0.50} for name in controls},
            },
            "rowwise_win_rates": {
                f"av_real_vs_{name}": {"candidate_better_fraction": 1.0}
                for name in controls
            },
            "generation_parse": {
                "real": {"closed_fraction": 1.0, "usable_fraction": 1.0}
            },
        }

        rejected = gate.build_gate_summary(
            {"validation": split},
            control_margin=0.01,
            require_family_level_inference=True,
            min_independent_families=3,
        )
        accepted = gate.build_gate_summary(
            {"validation": split},
            control_margin=0.01,
            require_family_level_inference=True,
            min_independent_families=2,
        )

        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["publication_status"], "small_sample_pilot")
        self.assertTrue(accepted["passed"])
        self.assertEqual(
            accepted["publication_status"], "family_controlled_validation"
        )

    def test_gate_summary_rejects_missing_controls_and_ties(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        missing = {"variants": {"av_real": {"normalized_mse": 0.25}}}
        tied = {
            "variants": {
                "av_real": {"normalized_mse": 0.25},
                "av_shuffled": {"normalized_mse": 0.25},
                "av_zero": {"normalized_mse": 0.25},
                "av_mean": {"normalized_mse": 0.25},
                "av_none": {"normalized_mse": 0.25},
                "mean": {"normalized_mse": 0.25},
            }
        }

        missing_summary = gate.build_gate_summary({"validation": missing}, control_margin=0.02)
        tied_summary = gate.build_gate_summary({"validation": tied}, control_margin=0.02)

        self.assertFalse(missing_summary["passed"])
        self.assertEqual(
            set(missing_summary["splits"]["validation"]["missing_controls"]),
            {"av_shuffled", "av_zero", "av_mean", "av_none", "mean"},
        )
        self.assertFalse(tied_summary["passed"])

    def test_default_gate_margin_is_positive(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        args = gate.parse_args(
            [
                "--ar-checkpoint-dir",
                "/tmp/ar",
                "--train-parquet",
                "/tmp/train.parquet",
                "--validation-parquet",
                "/tmp/validation.parquet",
                "--test-parquet",
                "/tmp/test.parquet",
                "--report-json",
                "/tmp/report.json",
            ]
        )

        self.assertEqual(args.control_margin, 0.1)
        self.assertEqual(args.eval_splits, ["validation"])

    def test_gate_summary_supports_small_nmse_margins_with_rowwise_win_rate(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        variants = {
            "av_real": {"normalized_mse": 0.00017},
            "av_shuffled": {"normalized_mse": 0.00042},
            "av_zero": {"normalized_mse": 0.00043},
            "av_mean": {"normalized_mse": 0.00039},
            "av_none": {"normalized_mse": 0.00042},
            "mean": {"normalized_mse": 0.00030},
            "teacher": {"normalized_mse": 0.00015},
        }
        split = {
            "variants": variants,
            "generation_parse": {"real": {"closed_fraction": 1.0, "usable_fraction": 1.0}},
            "rowwise_win_rates": {
                f"av_real_vs_{control}": {"candidate_better_fraction": fraction}
                for control, fraction in {
                    "av_shuffled": 1.0,
                    "av_zero": 1.0,
                    "av_mean": 1.0,
                    "av_none": 1.0,
                    "mean": 0.95,
                }.items()
            },
        }

        summary = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=5e-5,
            min_control_win_fraction=0.9,
            min_closed_fraction=0.8,
            min_usable_fraction=0.95,
        )

        self.assertTrue(summary["passed"])
        self.assertTrue(summary["splits"]["validation"]["controls"]["mean"]["primary_beats"])
        self.assertEqual(summary["splits"]["validation"]["controls"]["mean"]["rowwise_win_fraction"], 0.95)

        strict_summary = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=5e-5,
            min_control_win_fraction=0.99,
            min_closed_fraction=0.8,
            min_usable_fraction=0.95,
        )

        self.assertFalse(strict_summary["passed"])
        self.assertFalse(strict_summary["splits"]["validation"]["controls"]["mean"]["primary_beats"])

    def test_generation_backend_cli_accepts_trusted_batched_mode_and_quarantines_cache(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        common = [
            "--ar-checkpoint-dir", "/tmp/ar",
            "--train-parquet", "/tmp/train.parquet",
            "--validation-parquet", "/tmp/validation.parquet",
            "--test-parquet", "/tmp/test.parquet",
            "--report-json", "/tmp/report.json",
        ]
        args = gate.parse_args([*common, "--generation-backend", "legacy_batch"])
        self.assertEqual(args.generation_backend, "legacy_batch")

        with self.assertRaises(SystemExit):
            gate.parse_args([*common, "--generation-backend", "cache"])
        diagnostic = gate.parse_args(
            [*common, "--generation-backend", "cache", "--allow-unsafe-cache-backend"]
        )
        self.assertEqual(diagnostic.generation_backend, "cache")

    def test_generation_protocol_captures_mechanics_separately_from_model_identity(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        args = gate.parse_args(
            [
                "--av-hf-checkpoint", "/tmp/av",
                "--ar-checkpoint-dir", "/tmp/ar",
                "--train-parquet", "/tmp/train.parquet",
                "--validation-parquet", "/tmp/validation.parquet",
                "--test-parquet", "/tmp/test.parquet",
                "--report-json", "/tmp/report.json",
                "--generation-backend", "legacy_batch",
                "--generation-prefix", "<explanation>\n",
                "--stop-text", "</explanation>",
                "--max-new-tokens", "256",
                "--generation-workers", "8",
                "--av-model-fingerprint", "dcp_model_sha256:" + "a" * 64,
                "--av-tokenizer-fingerprint", "tokenizer_files_sha256:" + "b" * 64,
            ]
        )

        protocol = gate.build_generation_protocol(args)
        provenance = gate.build_generation_provenance(args)

        self.assertEqual(protocol["backend"], "legacy_batch")
        self.assertEqual(protocol["prefix"], "<explanation>\n")
        self.assertEqual(
            protocol["tokenizer_fingerprint"],
            "tokenizer_files_sha256:" + "b" * 64,
        )
        self.assertEqual(protocol["worker_count"], 8)
        self.assertFalse(protocol["do_sample"])
        self.assertEqual(protocol["temperature"], 0.0)
        self.assertNotIn("model_fingerprint", protocol)
        self.assertEqual(
            provenance["model_fingerprint"],
            "dcp_model_sha256:" + "a" * 64,
        )
        self.assertEqual(len(gate.generation_protocol_sha256(protocol)), 64)

    def test_generation_worker_records_parent_fanout_and_fingerprints(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        args = gate.parse_args(
            [
                "--av-hf-checkpoint", "/tmp/av",
                "--ar-checkpoint-dir", "/tmp/ar",
                "--train-parquet", "/tmp/train.parquet",
                "--validation-parquet", "/tmp/validation.parquet",
                "--test-parquet", "/tmp/test.parquet",
                "--report-json", "/tmp/report.json",
                "--generation-backend", "legacy_batch",
                "--generation-workers", "8",
                "--av-model-fingerprint", "dcp_model_sha256:" + "a" * 64,
                "--av-tokenizer-fingerprint", "tokenizer_files_sha256:" + "b" * 64,
                "--require-generation-protocol-match",
            ]
        )

        command = gate.build_generation_worker_command(
            args,
            shard_index=2,
            shard_count=8,
            shard_jsonl=pathlib.Path("/tmp/generated-worker2.jsonl"),
        )
        worker_args = gate.parse_args(command[2:])

        self.assertEqual(worker_args.generation_workers, 1)
        self.assertEqual(worker_args.generation_parent_worker_count, 8)
        self.assertEqual(gate.build_generation_protocol(worker_args)["worker_count"], 8)
        self.assertEqual(
            worker_args.av_model_fingerprint,
            "dcp_model_sha256:" + "a" * 64,
        )
        self.assertEqual(
            worker_args.av_tokenizer_fingerprint,
            "tokenizer_files_sha256:" + "b" * 64,
        )
        self.assertTrue(worker_args.require_generation_protocol_match)

    def test_gate_rejects_generation_protocol_mismatch_when_required(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        candidate_protocol = {
            "schema_version": "nano_generation_protocol.v1",
            "backend": "legacy_batch",
            "prefix": "",
            "prefix_sha256": gate.sha256_text(""),
            "stop_text": "</explanation>",
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": 0.0,
            "seed": 1234,
            "injection_scale": "75",
            "torch_dtype": "bfloat16",
            "attention_implementation": None,
            "tokenizer_fingerprint": "tokenizer-sha",
            "worker_count": 8,
            "controls": ["real", "shuffled", "zero", "mean", "none"],
        }
        baseline_protocol = dict(candidate_protocol)
        baseline_protocol["prefix"] = "<explanation>"
        baseline_protocol["prefix_sha256"] = gate.sha256_text("<explanation>")
        candidate_losses = [0.10, 0.11]
        baseline_losses = [0.20, 0.21]
        split = {
            "row_indices": [1, 2],
            "doc_ids": ["doc-a", "doc-b"],
            "rowwise_normalized_mse": {"av_real": candidate_losses},
            "generation_parse": {"real": {"closed_fraction": 1.0, "usable_fraction": 1.0}},
            "variants": {
                "av_real": {"normalized_mse": 0.105},
                "av_shuffled": {"normalized_mse": 0.50},
                "av_zero": {"normalized_mse": 0.51},
                "av_mean": {"normalized_mse": 0.52},
                "av_none": {"normalized_mse": 0.53},
                "mean": {"normalized_mse": 0.54},
            },
            "rowwise_win_rates": {
                f"av_real_vs_{control}": {"candidate_better_fraction": 1.0}
                for control in ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
            },
        }
        baseline = {
            "generation_protocol": baseline_protocol,
            "splits": {
                name: {
                    "row_indices": [1, 2],
                    "rowwise_normalized_mse": {"av_real": baseline_losses},
                    "variants": {"av_real": {"normalized_mse": 0.205}},
                }
                for name in ("validation", "test")
            },
        }

        rejected = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            generation_protocol=candidate_protocol,
            require_generation_protocol_match=True,
            bootstrap_samples=100,
        )
        baseline["generation_protocol"] = candidate_protocol
        accepted = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            generation_protocol=candidate_protocol,
            require_generation_protocol_match=True,
            bootstrap_samples=100,
        )

        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["generation_protocol_parity"]["matched"])
        self.assertEqual(
            set(rejected["generation_protocol_parity"]["mismatched_fields"]),
            {"prefix", "prefix_sha256"},
        )
        self.assertTrue(accepted["passed"])
        self.assertTrue(accepted["generation_protocol_parity"]["matched"])

    def test_gate_rejects_each_generation_mechanics_mismatch(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        protocol = {
            "schema_version": "nano_generation_protocol.v1",
            "backend": "legacy_batch",
            "prefix": "",
            "prefix_sha256": gate.sha256_text(""),
            "stop_text": "</explanation>",
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": 0.0,
            "seed": 1234,
            "injection_scale": "75",
            "torch_dtype": "bfloat16",
            "attention_implementation": None,
            "tokenizer_fingerprint": "tokenizer-sha",
            "worker_count": 8,
            "controls": ["real", "shuffled", "zero", "mean", "none"],
            "generated_text_fallback": "empty",
        }
        split = {
            "row_indices": [1, 2],
            "doc_ids": ["doc-a", "doc-b"],
            "rowwise_normalized_mse": {"av_real": [0.10, 0.11]},
            "generation_parse": {"real": {"closed_fraction": 1.0, "usable_fraction": 1.0}},
            "variants": {
                "av_real": {"normalized_mse": 0.105},
                "av_shuffled": {"normalized_mse": 0.50},
                "av_zero": {"normalized_mse": 0.51},
                "av_mean": {"normalized_mse": 0.52},
                "av_none": {"normalized_mse": 0.53},
                "mean": {"normalized_mse": 0.54},
            },
            "rowwise_win_rates": {
                f"av_real_vs_{control}": {"candidate_better_fraction": 1.0}
                for control in ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
            },
        }
        mismatches = {
            "backend": "legacy",
            "stop_text": None,
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.7,
            "seed": 7,
            "injection_scale": "50",
            "torch_dtype": "float32",
            "attention_implementation": "eager",
            "tokenizer_fingerprint": "other-tokenizer-sha",
            "worker_count": 4,
            "controls": ["real"],
            "generated_text_fallback": "raw",
        }

        for field, other_value in mismatches.items():
            with self.subTest(field=field):
                baseline_protocol = {**protocol, field: other_value}
                baseline = {
                    "generation_protocol": baseline_protocol,
                    "splits": {
                        name: {
                            "row_indices": [1, 2],
                            "rowwise_normalized_mse": {"av_real": [0.20, 0.21]},
                            "variants": {"av_real": {"normalized_mse": 0.205}},
                        }
                        for name in ("validation", "test")
                    },
                }

                summary = gate.build_gate_summary(
                    {"validation": split, "test": split},
                    control_margin=0.01,
                    baseline_report=baseline,
                    generation_protocol=protocol,
                    require_generation_protocol_match=True,
                    bootstrap_samples=100,
                )

                self.assertFalse(summary["passed"])
                self.assertEqual(
                    summary["generation_protocol_parity"]["mismatched_fields"],
                    [field],
                )

    def test_publication_gate_rejects_matching_nonempty_prefix(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        protocol = {
            "schema_version": "nano_generation_protocol.v1",
            "backend": "legacy_batch",
            "prefix": "<explanation>\n",
            "prefix_sha256": gate.sha256_text("<explanation>\n"),
        }
        split = {
            "row_indices": [1],
            "doc_ids": ["doc-a"],
            "rowwise_normalized_mse": {"av_real": [0.1]},
            "generation_parse": {"real": {"closed_fraction": 1.0, "usable_fraction": 1.0}},
            "variants": {
                "av_real": {"normalized_mse": 0.1},
                "av_shuffled": {"normalized_mse": 0.5},
                "av_zero": {"normalized_mse": 0.5},
                "av_mean": {"normalized_mse": 0.5},
                "av_none": {"normalized_mse": 0.5},
                "mean": {"normalized_mse": 0.5},
            },
            "rowwise_win_rates": {
                f"av_real_vs_{control}": {"candidate_better_fraction": 1.0}
                for control in ("av_shuffled", "av_zero", "av_mean", "av_none", "mean")
            },
        }
        baseline = {
            "generation_protocol": protocol,
            "splits": {
                name: {
                    "row_indices": [1],
                    "rowwise_normalized_mse": {"av_real": [0.2]},
                    "variants": {"av_real": {"normalized_mse": 0.2}},
                }
                for name in ("validation", "test")
            },
        }

        summary = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            baseline_report=baseline,
            generation_protocol=protocol,
            require_generation_protocol_match=True,
            bootstrap_samples=100,
        )

        self.assertFalse(summary["passed"])
        self.assertTrue(summary["generation_protocol_parity"]["matched"])
        self.assertFalse(summary["generation_protocol_parity"]["publication_compatible"])
        self.assertIn("nonempty_generation_prefix", summary["generation_protocol_parity"]["publication_errors"])

    def test_cached_records_fail_closed_on_missing_or_mixed_protocol(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        protocol = {
            "schema_version": "nano_generation_protocol.v1",
            "backend": "legacy_batch",
            "prefix": "",
        }
        protocol_hash = gate.generation_protocol_sha256(protocol)
        valid = {
            "row_index": 1,
            "generation_protocol": protocol,
            "generation_protocol_sha256": protocol_hash,
        }

        resolved = gate.validate_generated_record_protocols(
            [valid], expected_protocol=protocol, require=True
        )

        self.assertEqual(resolved, protocol)
        with self.assertRaisesRegex(ValueError, "missing generation protocol"):
            gate.validate_generated_record_protocols(
                [{"row_index": 1}], expected_protocol=protocol, require=True
            )
        with self.assertRaisesRegex(ValueError, "mixed generation protocols"):
            other_protocol = {**protocol, "prefix": "<explanation>"}
            gate.validate_generated_record_protocols(
                [
                    valid,
                    {
                        **valid,
                        "row_index": 2,
                        "generation_protocol": other_protocol,
                        "generation_protocol_sha256": gate.generation_protocol_sha256(
                            other_protocol
                        ),
                    },
                ],
                expected_protocol=protocol,
                require=True,
            )

    def test_cached_generation_provenance_rejects_dataset_mismatch(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        provenance = {
            "model_fingerprint": "dcp_model_sha256:" + "a" * 64,
            "tokenizer_fingerprint": "tokenizer_files_sha256:" + "b" * 64,
            "checkpoint": "/models/av",
            "model_revision": None,
            "tokenizer_revision": None,
            "datasets": {
                "train": {"path": "/data/train", "size_bytes": 1, "sha256": "c" * 64},
                "validation": {"path": "/data/val", "size_bytes": 1, "sha256": "d" * 64},
            },
        }
        provenance["dataset_bundle_sha256"] = gate.hashlib.sha256(
            gate.json.dumps(
                provenance["datasets"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        record = {
            "row_index": 4,
            "generation_provenance": provenance,
            "generation_provenance_sha256": gate.generation_provenance_sha256(
                provenance
            ),
        }
        changed = json.loads(json.dumps(provenance))
        changed["datasets"]["validation"]["sha256"] = "e" * 64

        with self.assertRaisesRegex(ValueError, "model or dataset identity"):
            gate.validate_generated_record_provenance(
                [record], expected_provenance=changed, require=True
            )

    def test_cache_reuse_rejects_placeholder_fingerprint(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        with self.assertRaises(SystemExit):
            gate.parse_args(
                [
                    "--ar-checkpoint-dir", "/tmp/ar",
                    "--train-parquet", "/tmp/train.parquet",
                    "--validation-parquet", "/tmp/validation.parquet",
                    "--report-json", "/tmp/report.json",
                    "--generated-jsonl", "/tmp/generated.jsonl",
                    "--reuse-generated",
                    "--av-model-fingerprint", "queue-prepared-model",
                    "--av-tokenizer-fingerprint", "queue-prepared-tokenizer",
                ]
            )

    def test_repetition_detector_handles_phrases_longer_than_eight_tokens(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        phrase = [f"token-{index}" for index in range(9)]

        self.assertTrue(gate.has_repetition_loop(phrase * 3))

    def test_generation_only_uses_worker_fanout_when_requested(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            report_json = root / "report.json"
            generated_jsonl = root / "generated.jsonl"
            calls: list[tuple[str, pathlib.Path]] = []

            def fake_workers(args, output_path):
                calls.append(("workers", output_path))
                output_path.write_text('{"row_index": 1}\n')
                return [{"row_index": 1}]

            def fake_serial(*_args, **_kwargs):
                calls.append(("serial", pathlib.Path("unused")))
                return []

            with mock.patch.object(gate, "generate_roundtrip_records_with_workers", fake_workers), mock.patch.object(
                gate, "generate_roundtrip_records", fake_serial
            ):
                result = gate.main(
                    [
                        "--generation-only",
                        "--generation-workers",
                        "8",
                        "--av-hf-checkpoint",
                        str(root / "av"),
                        "--ar-checkpoint-dir",
                        str(root / "ar"),
                        "--train-parquet",
                        str(root / "train.parquet"),
                        "--validation-parquet",
                        str(root / "validation.parquet"),
                        "--test-parquet",
                        str(root / "test.parquet"),
                        "--report-json",
                        str(report_json),
                        "--generated-jsonl",
                        str(generated_jsonl),
                        "--stream-generated",
                    ]
                )
                generation_report = json.loads(report_json.read_text())

        self.assertEqual(result, 0)
        self.assertEqual(calls, [("workers", generated_jsonl)])
        self.assertEqual(generation_report["row_count"], 1)
        self.assertEqual(generation_report["generated_jsonl"], str(generated_jsonl))
        self.assertIsNone(generation_report["generation_protocol_sha256"])

    def test_generated_control_prompt_strips_explanation_tags(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        parsed = gate.parse_generated_explanation("<explanation>  generated summary  </explanation>")

        self.assertEqual(parsed["explanation"], "generated summary")
        self.assertTrue(parsed["closed"])

    def test_generated_control_parser_can_use_open_tag_or_raw_fallback(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        open_only = gate.parse_generated_explanation("<explanation> partial summary", fallback="empty")
        raw = gate.parse_generated_explanation("plain summary without tags", fallback="raw")
        empty = gate.parse_generated_explanation("plain summary without tags", fallback="empty")

        self.assertEqual(open_only["explanation"], "partial summary")
        self.assertEqual(open_only["extraction_mode"], "open_tag")
        self.assertFalse(open_only["closed"])
        self.assertFalse(open_only["empty"])
        self.assertEqual(raw["explanation"], "plain summary without tags")
        self.assertEqual(raw["extraction_mode"], "raw")
        self.assertTrue(empty["empty"])

    def test_parse_quality_rejects_fallbacks_and_repetition_loops(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        repeated = gate.parse_generated_explanation(
            "<explanation>alpha beta alpha beta alpha beta</explanation>"
        )
        raw = gate.parse_generated_explanation("answer 12 34", fallback="raw")
        clean = gate.parse_generated_explanation(
            "<explanation>answer 12 has supporting context</explanation>"
        )

        self.assertTrue(repeated["repetition_loop"])
        self.assertFalse(repeated["usable"])
        self.assertTrue(repeated["content_usable"])
        self.assertTrue(raw["fallback_only"])
        self.assertFalse(raw["usable"])
        self.assertGreater(raw["factual_number_density"], 0.0)
        self.assertTrue(clean["usable"])

    def test_parse_summary_revalidates_strict_usability(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        records = [
            {
                "controls": {
                    "real": {
                        "generated": "<explanation>clean concise answer</explanation>"
                    }
                }
            },
            {
                "controls": {
                    "real": {
                        "generated": (
                            "<explanation>alpha beta alpha beta alpha beta</explanation>"
                        )
                    }
                }
            },
            {
                "controls": {
                    "real": {
                        "generated": "raw fallback 42",
                        "parsed": {"extraction_mode": "raw"},
                    }
                }
            },
        ]

        summary = gate.summarize_generation_parse(records, "real")

        self.assertEqual(summary["closed_count"], 2)
        self.assertEqual(summary["repetition_loop_count"], 1)
        self.assertEqual(summary["fallback_only_count"], 1)
        self.assertEqual(summary["true_usable_count"], 1)
        self.assertAlmostEqual(summary["usable_fraction"], 1 / 3)
        self.assertGreater(summary["factual_number_density_mean"], 0.0)

    def test_length_control_analysis_reports_matched_gain_and_correlations(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        analysis = gate.summarize_length_control_analysis(
            candidate_token_counts=[6, 4, 5],
            sft_token_counts=[3, 5, 4],
            teacher_token_counts=[2, 2, 3],
            candidate_losses=[0.08, 0.12, 0.10],
            baseline_losses=[0.10, 0.15, 0.13],
            sft_length_matched_losses=[0.09, 0.13, 0.11],
            teacher_length_matched_losses=[0.11, 0.14, 0.12],
        )

        self.assertEqual(analysis["row_count"], 3)
        self.assertGreater(analysis["relative_improvement"], 0.0)
        self.assertGreater(
            analysis["sft_length_matched_relative_improvement"],
            0.0,
        )
        self.assertGreater(
            analysis["teacher_length_matched_relative_improvement"],
            0.0,
        )
        self.assertEqual(
            analysis["best_length_matched_relative_improvement"],
            analysis["sft_length_matched_relative_improvement"],
        )
        self.assertIn("sft_length_delta_vs_gain_pearson", analysis)
        self.assertIn("gain_per_generated_token_mean", analysis)

    def test_length_matched_explanations_align_rows_and_truncate_candidate(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        class SpaceTokenizer:
            def __call__(self, text, *, add_special_tokens=False):
                self.assert_no_special_tokens = not add_special_tokens
                return {"input_ids": text.split()}

            def decode(self, token_ids, **_kwargs):
                return " ".join(token_ids)

        candidate = [
            {
                "split": "validation",
                "row_index": 7,
                "doc_id": "doc-a",
                "content_family_id": "family-a",
                "controls": {
                    "real": {
                        "generated": (
                            "<explanation>one two three four five six</explanation>"
                        )
                    }
                },
            }
        ]
        baseline = [
            {
                "split": "validation",
                "row_index": 7,
                "doc_id": "doc-a",
                "content_family_id": "family-a",
                "controls": {
                    "real": {"generated": "<explanation>a b c</explanation>"}
                },
            }
        ]
        rows = {7: {"response": "<explanation>x y</explanation>"}}

        matched = gate.build_length_matched_explanations(
            SpaceTokenizer(),
            candidate_records=candidate,
            baseline_records=baseline,
            rows_by_index=rows,
            fallback="raw",
        )

        self.assertEqual(matched["candidate_token_counts"], [6])
        self.assertEqual(matched["sft_token_counts"], [3])
        self.assertEqual(matched["teacher_token_counts"], [2])
        self.assertEqual(matched["sft_length_matched_explanations"], ["one two three"])
        self.assertEqual(matched["teacher_length_matched_explanations"], ["one two"])

        baseline[0]["doc_id"] = "wrong-doc"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            gate.build_length_matched_explanations(
                SpaceTokenizer(),
                candidate_records=candidate,
                baseline_records=baseline,
                rows_by_index=rows,
                fallback="raw",
            )

    def test_length_tokenizer_accepts_batch_encoding_and_tensor_like_ids(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        from collections import UserDict

        class TensorLike:
            def tolist(self):
                return [10, 20, 30]

        class BatchEncodingLike(UserDict):
            pass

        class Tokenizer:
            def __call__(self, _text, *, add_special_tokens=False):
                self.add_special_tokens = add_special_tokens
                return BatchEncodingLike({"input_ids": TensorLike()})

        tokenizer = Tokenizer()
        self.assertEqual(gate._tokenize_explanation(tokenizer, "hello"), [10, 20, 30])
        self.assertFalse(tokenizer.add_special_tokens)

    def test_attach_length_control_analysis_requires_exact_baseline_rows(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        split = {
            "row_indices": [7, 8],
            "length_control_token_counts": {
                "candidate_token_counts": [6, 4],
                "sft_token_counts": [3, 5],
                "teacher_token_counts": [2, 2],
            },
            "rowwise_directional_mse": {
                "av_real": [0.08, 0.12],
                "av_real_sft_length_matched": [0.09, 0.13],
                "av_real_teacher_length_matched": [0.11, 0.14],
            },
        }
        baseline = {
            "splits": {
                "validation": {
                    "row_indices": [7, 8],
                    "rowwise_directional_mse": {"av_real": [0.10, 0.15]},
                }
            }
        }

        gate.attach_length_control_analysis({"validation": split}, baseline)

        self.assertGreater(
            split["length_analysis"]["best_length_matched_relative_improvement"],
            0.0,
        )

        baseline["splits"]["validation"]["row_indices"] = [8, 7]
        with self.assertRaisesRegex(ValueError, "row identity mismatch"):
            gate.attach_length_control_analysis({"validation": split}, baseline)

    def test_gate_summary_can_require_parse_health(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        split = {
            "generation_parse": {"real": {"closed_fraction": 0.25, "usable_fraction": 1.0}},
            "variants": {
                "av_real": {"normalized_mse": 0.20},
                "av_shuffled": {"normalized_mse": 0.40},
                "av_zero": {"normalized_mse": 0.40},
                "av_mean": {"normalized_mse": 0.40},
                "av_none": {"normalized_mse": 0.40},
                "mean": {"normalized_mse": 0.40},
            },
        }

        summary = gate.build_gate_summary(
            {"validation": split, "test": split},
            control_margin=0.01,
            min_closed_fraction=0.8,
            min_usable_fraction=0.95,
        )

        self.assertFalse(summary["passed"])
        self.assertFalse(summary["splits"]["validation"]["parse_health"]["passed"])

    def test_generated_jsonl_roundtrip(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        records = [
            {
                "schema_version": gate.ROUNDTRIP_SCHEMA_VERSION,
                "row_index": 1,
                "split": "validation",
                "controls": {"real": {"generated": "hello"}},
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "generated.jsonl"
            gate.write_generated_jsonl(path, records)
            loaded = gate.read_generated_jsonl(path)

        self.assertEqual(loaded, records)

    def test_shard_eval_indices_partitions_by_position(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")

        self.assertEqual(
            gate.shard_eval_indices([10, 11, 12, 13, 14], shard_index=0, shard_count=2),
            [10, 12, 14],
        )
        self.assertEqual(
            gate.shard_eval_indices([10, 11, 12, 13, 14], shard_index=1, shard_count=2),
            [11, 13],
        )

    def test_merge_generated_shards_preserves_eval_order(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shard_a = root / "a.jsonl"
            shard_b = root / "b.jsonl"
            out = root / "merged.jsonl"
            gate.write_generated_jsonl(
                shard_a,
                [
                    {"row_index": 12, "split": "validation", "controls": {}},
                    {"row_index": 20, "split": "test", "controls": {}},
                ],
            )
            gate.write_generated_jsonl(
                shard_b,
                [
                    {"row_index": 10, "split": "validation", "controls": {}},
                    {"row_index": 18, "split": "test", "controls": {}},
                ],
            )

            merged = gate.merge_generated_shards([shard_a, shard_b], out)
            loaded = gate.read_generated_jsonl(out)

        self.assertEqual([record["row_index"] for record in merged], [10, 12, 18, 20])
        self.assertEqual(loaded, merged)

    def test_multiworker_resume_validates_merged_cache_once(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        with tempfile.TemporaryDirectory() as tmp:
            generated = pathlib.Path(tmp) / "generated.jsonl"
            generated.write_text("{}\n")
            args = gate.argparse.Namespace(
                generation_workers=8,
                generation_parent_worker_count=None,
                resume_generated=True,
                stream_generated=True,
            )
            with mock.patch.object(
                gate,
                "generate_roundtrip_records",
                return_value=[{"row_index": 1}],
            ) as generate:
                records = gate.generate_roundtrip_records_with_workers(args, generated)

        self.assertEqual(records, [{"row_index": 1}])
        resume_args = generate.call_args.args[0]
        self.assertEqual(resume_args.generation_workers, 1)
        self.assertEqual(resume_args.generation_parent_worker_count, 8)
        self.assertEqual(generate.call_args.kwargs["stream_jsonl"], generated)

    def test_generated_record_coverage_rejects_missing_rows(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        records = [
            {"row_index": 10, "split": "validation", "controls": {"real": {}, "mean": {}}},
            {"row_index": 20, "split": "test", "controls": {"real": {}, "mean": {}}},
        ]

        with self.assertRaisesRegex(ValueError, "missing generated rows"):
            gate.validate_generated_record_coverage(
                records,
                expected_by_split={"validation": [10, 12], "test": [20]},
                controls_requested=["real", "mean"],
            )

    def test_generated_record_coverage_rejects_missing_controls(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        records = [
            {"row_index": 10, "split": "validation", "controls": {"real": {}}},
            {"row_index": 20, "split": "test", "controls": {"real": {}, "mean": {}}},
        ]

        with self.assertRaisesRegex(ValueError, "missing generated controls"):
            gate.validate_generated_record_coverage(
                records,
                expected_by_split={"validation": [10], "test": [20]},
                controls_requested=["real", "mean"],
            )

    def test_select_reusable_generated_records_keeps_complete_expected_rows(self):
        gate = load_script("eval_nano_av_ar_roundtrip_gate")
        records = [
            {"row_index": 10, "split": "validation", "controls": {"real": {}, "mean": {}}},
            {"row_index": 12, "split": "validation", "controls": {"real": {}}},
            {"row_index": 99, "split": "validation", "controls": {"real": {}, "mean": {}}},
            {"row_index": 20, "split": "test", "controls": {"real": {}, "mean": {}}},
        ]

        reusable = gate.select_reusable_generated_records(
            records,
            expected_by_split={"validation": [10, 12], "test": [20]},
            controls_requested=["real", "mean"],
        )

        self.assertEqual(
            [(record["split"], record["row_index"]) for record in reusable],
            [("validation", 10), ("test", 20)],
        )


if __name__ == "__main__":
    unittest.main()
