import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "audit_nano_roundtrip_subgroups.py"
    spec = importlib.util.spec_from_file_location(
        "audit_nano_roundtrip_subgroups", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AuditNanoRoundtripSubgroupsTests(unittest.TestCase):
    def test_config_rejects_test_as_fit_dataset(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": module.SCHEMA_VERSION,
                        "paths": {
                            "report_json": "report.json",
                            "datasets": {
                                "validation": {
                                    "cache_npz": "a.npz",
                                    "score_report_json": "a.json",
                                    "generated_jsonl": "a.jsonl",
                                    "split": "validation",
                                },
                                "test": {
                                    "cache_npz": "b.npz",
                                    "score_report_json": "b.json",
                                    "generated_jsonl": "b.jsonl",
                                    "split": "test",
                                },
                            },
                        },
                        "protocol": {
                            "fit_dataset": "test",
                            "controls": ["mean"],
                        },
                    }
                )
            )

            with self.assertRaisesRegex(module.SubgroupAuditError, "must not use test"):
                module.load_config(path)

    def test_validation_edges_apply_without_refitting(self):
        module = load_script()
        fit = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        test = np.asarray([-1.0, 0.5, 3.5, 10.0])

        edges = module.fit_quantile_edges(fit, 4)
        assigned = module.assign_bins(test, edges)

        self.assertEqual(edges, [1.75, 3.5, 5.25])
        self.assertEqual(assigned.tolist(), [0, 0, 2, 3])

    def test_alignment_uses_frozen_row_doc_and_family_identity(self):
        module = load_script()
        dataset = {
            "row_indices": np.asarray([10, 20]),
            "doc_ids": np.asarray(["doc:10", "doc:20"]),
            "content_family_ids": np.asarray(["f1", "f1"]),
            "targets": np.asarray([[3.0, 4.0], [0.0, 2.0]]),
        }
        records = {
            10: {
                "row_index": 10,
                "doc_id": "doc:10",
                "content_family_id": "f1",
                "n_raw_tokens": 100,
                "target_explanation": "ten useful words are represented here",
            },
            20: {
                "row_index": 20,
                "doc_id": "doc:20",
                "content_family_id": "f1",
                "n_raw_tokens": 200,
                "target_explanation": "another teacher explanation",
            },
        }

        aligned = module.align_metadata(dataset, records)

        self.assertEqual(aligned["n_raw_tokens"].tolist(), [100.0, 200.0])
        self.assertEqual(aligned["sample_family_frequency"].tolist(), [2.0, 2.0])
        self.assertEqual(aligned["target_activation_norm"].tolist(), [5.0, 2.0])

    def test_directional_metric_is_scale_invariant(self):
        module = load_script()
        target = np.asarray([[1.0, 2.0], [2.0, -1.0]])
        prediction = target * 3.0

        metric = module.rowwise_directional_mse(prediction, target)

        self.assertTrue(np.allclose(metric, 0.0))


if __name__ == "__main__":
    unittest.main()
