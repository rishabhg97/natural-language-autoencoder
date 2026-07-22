import importlib.util
import pathlib
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoAREvalReportExtensionTests(unittest.TestCase):
    def test_bootstrap_ci_reports_mean_interval_and_count(self):
        evaluator = load_script("eval_nano_ar_miles_checkpoint")
        values = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float32)

        ci = evaluator.bootstrap_ci(values, samples=200, seed=7)

        self.assertEqual(ci["n"], 4)
        self.assertAlmostEqual(ci["mean"], 0.375)
        self.assertLessEqual(ci["ci_low"], ci["mean"])
        self.assertGreaterEqual(ci["ci_high"], ci["mean"])

    def test_prediction_dump_rows_include_row_metrics(self):
        evaluator = load_script("eval_nano_ar_miles_checkpoint")

        rows = evaluator.build_prediction_dump_rows(
            split_name="validation",
            control_name="teacher",
            row_indices=[7],
            doc_ids=["doc-a"],
            predictions=np.array([[1.0, 0.0]], dtype=np.float32),
            targets=np.array([[0.0, 1.0]], dtype=np.float32),
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["split"], "validation")
        self.assertEqual(row["control"], "teacher")
        self.assertEqual(row["row_index"], 7)
        self.assertEqual(row["doc_id"], "doc-a")
        self.assertAlmostEqual(row["normalized_mse"], 2.0, places=6)
        self.assertAlmostEqual(row["cosine"], 0.0)
        self.assertAlmostEqual(row["pred_norm"], 1.0)
        self.assertAlmostEqual(row["gold_norm"], 1.0)


if __name__ == "__main__":
    unittest.main()
