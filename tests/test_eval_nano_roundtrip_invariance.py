import importlib.util
import argparse
import json
import pathlib
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoRoundtripInvarianceTests(unittest.TestCase):
    def test_report_writer_serializes_numpy_metrics(self):
        module = load_script("eval_nano_roundtrip_invariance")
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            module.roundtrip.write_json(
                path,
                {"rowwise": np.asarray([0.2, 0.4], dtype=np.float32)},
            )
            payload = json.loads(path.read_text())

        self.assertEqual(len(payload["rowwise"]), 2)
        self.assertAlmostEqual(payload["rowwise"][0], 0.2, places=6)

    def test_score_args_supply_roundtrip_runtime_defaults(self):
        module = load_script("eval_nano_roundtrip_invariance")
        args = argparse.Namespace(
            ar_checkpoint_dir=pathlib.Path("/critic"),
            critic_template_source=None,
            critic_template=None,
            train_parquet=pathlib.Path("/train.parquet"),
            validation_parquet=pathlib.Path("/validation.parquet"),
            test_parquet=None,
            validation_limit=8,
            test_limit=8,
            eval_splits=["validation"],
            content_family_manifest=None,
            content_family_coverage=None,
            require_family_level_inference=True,
            selection_strategy="family_stratified",
            selection_seed=1,
            torch_dtype="bfloat16",
            ar_device_map="cuda:0",
            ar_batch_size=4,
            ar_max_length=1152,
            generated_text_fallback="raw",
        )

        scoring = module._score_args(args)

        self.assertTrue(scoring.ar_low_cpu_mem_usage)
        self.assertTrue(scoring.collect_ar_device_profile)
        self.assertIsNone(scoring.length_baseline_generated_jsonl)

    def test_validation_only_summary_does_not_require_test(self):
        module = load_script("eval_nano_roundtrip_invariance")
        metrics = {
            "fve_nrm": 0.6,
            "directional_mse": 0.4,
            "raw_mse": 2.0,
            "centered_r2": 0.3,
            "norm_ratio_mean": 0.9,
        }
        raw = {"splits": {"validation": {"variants": {"av_real": metrics}}}}
        changed = dict(metrics, fve_nrm=0.54, directional_mse=0.46)
        transformed = {
            "canonical": {
                "splits": {"validation": {"variants": {"av_real": changed}}}
            }
        }

        report = module.summarize_invariance(raw, transformed)

        summary = report["transforms"]["canonical"]["validation"]
        self.assertAlmostEqual(summary["fve_retention"], 0.9)
        self.assertAlmostEqual(
            summary["metrics"]["directional_mse"]["delta"], 0.06
        )
        self.assertNotIn("test", report["transforms"]["canonical"])

    def test_summarize_invariance_reports_fve_retention(self):
        module = load_script("eval_nano_roundtrip_invariance")
        raw = {
            "splits": {
                "validation": {"variants": {"av_real": {"fve_nrm": 0.60}}},
                "test": {"variants": {"av_real": {"fve_nrm": 0.50}}},
            }
        }
        transformed = {
            "format_normalized": {
                "splits": {
                    "validation": {"variants": {"av_real": {"fve_nrm": 0.57}}},
                    "test": {"variants": {"av_real": {"fve_nrm": 0.45}}},
                }
            }
        }

        report = module.summarize_invariance(raw, transformed)

        self.assertAlmostEqual(
            report["transforms"]["format_normalized"]["validation"]["fve_retention"],
            0.95,
        )
        self.assertAlmostEqual(
            report["transforms"]["format_normalized"]["test"]["fve_retention"],
            0.90,
        )

    def test_summarize_invariance_rejects_missing_fve(self):
        module = load_script("eval_nano_roundtrip_invariance")
        raw = {"splits": {"validation": {"variants": {"av_real": {"fve_nrm": 0.5}}}}}

        with self.assertRaisesRegex(module.InvarianceError, "missing FVE"):
            module.summarize_invariance(raw, {"unit_reordered": {"splits": {}}})

    def test_combined_summary_reads_named_control_variants(self):
        module = load_script("eval_nano_roundtrip_invariance")
        report = {
            "splits": {
                "validation": {
                    "row_indices": [2, 7],
                    "variants": {
                        "av_real": {
                            "fve_nrm": 0.6,
                            "directional_mse": 0.4,
                            "raw_mse": 2.0,
                        },
                        "av_light_paraphrase": {
                            "fve_nrm": 0.48,
                            "directional_mse": 0.52,
                            "raw_mse": 2.2,
                        },
                    },
                }
            }
        }

        summary = module.summarize_combined_invariance(
            report, ["light_paraphrase"]
        )

        item = summary["transforms"]["light_paraphrase"]["validation"]
        self.assertAlmostEqual(item["fve_retention"], 0.8)
        self.assertEqual(item["row_count"], 2)
        self.assertAlmostEqual(item["metrics"]["directional_mse"]["delta"], 0.12)

    def test_merge_transform_controls_preserves_row_identity(self):
        module = load_script("eval_nano_roundtrip_invariance")
        base = [
            {
                "split": "validation",
                "row_index": 3,
                "controls": {"real": {"generated": "source"}},
            }
        ]
        transformed = [
            {
                "split": "validation",
                "row_index": 3,
                "controls": {"real": {"generated": "paraphrase"}},
            }
        ]

        combined = module.merge_transform_controls(
            base, {"light_paraphrase": transformed}
        )

        self.assertEqual(combined[0]["controls"]["real"]["generated"], "source")
        self.assertEqual(
            combined[0]["controls"]["light_paraphrase"]["generated"],
            "paraphrase",
        )

    def test_filter_generated_records_selects_exact_subset(self):
        module = load_script("eval_nano_roundtrip_invariance")
        records = [
            {"split": "validation", "row_index": index}
            for index in (2, 7, 11)
        ]

        selected = module.filter_generated_records(
            records, {"validation": [11, 2]}
        )

        self.assertEqual([record["row_index"] for record in selected], [11, 2])
        with self.assertRaises(module.InvarianceError):
            module.filter_generated_records(
                records, {"validation": [11, 99]}
            )


if __name__ == "__main__":
    unittest.main()
