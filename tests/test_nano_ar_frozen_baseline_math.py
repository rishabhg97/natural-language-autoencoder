import importlib.util
import pathlib
import unittest

import pytest

torch = pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoARFrozenBaselineMathTests(unittest.TestCase):
    def test_ridge_map_recovers_linear_targets(self):
        baseline = load_script("nano_ar_frozen_baseline")
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        true_weight = torch.tensor([[2.0, -1.0], [0.5, 1.5]])
        targets = features @ true_weight

        weight = baseline.fit_ridge_map(features, targets, ridge_alpha=1e-6)
        predictions = baseline.predict_linear_map(features, weight)

        self.assertLess(float(torch.nn.functional.mse_loss(predictions, targets)), 1e-8)

    def test_procrustes_map_recovers_orthogonal_transform(self):
        baseline = load_script("nano_ar_frozen_baseline")
        features = torch.eye(3)
        rotation = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        targets = features @ rotation

        weight = baseline.fit_procrustes_map(features, targets)
        predictions = baseline.predict_linear_map(features, weight)

        self.assertTrue(torch.allclose(predictions, targets, atol=1e-6))

    def test_closed_form_report_row_carries_readout_metadata(self):
        baseline = load_script("nano_ar_frozen_baseline")

        row = baseline.closed_form_report_row(
            method="ridge",
            readout_mode="mean_pool",
            split="validation",
            metrics={"normalized_mse": 0.3, "cosine_mean": 0.85},
        )

        self.assertEqual(row["method"], "ridge")
        self.assertEqual(row["readout_mode"], "mean_pool")
        self.assertEqual(row["split"], "validation")
        self.assertEqual(row["metrics"]["normalized_mse"], 0.3)


if __name__ == "__main__":
    unittest.main()
