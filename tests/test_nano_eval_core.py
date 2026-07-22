import importlib.util
import pathlib
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


class NanoEvalCoreTests(unittest.TestCase):
    def test_activation_metrics_distinguish_direction_from_raw_reconstruction(self):
        module = load_script("nano_eval_core")
        target = np.array([[1.0, 2.0]], dtype=np.float64)
        prediction = 10.0 * target

        result = module.activation_reconstruction_metrics(
            prediction,
            target,
            train_mean=np.zeros(2, dtype=np.float64),
        )

        self.assertAlmostEqual(result["directional_mse"], 0.0, places=12)
        self.assertAlmostEqual(result["normalized_mse"], 0.0, places=12)
        self.assertAlmostEqual(result["cosine_mean"], 1.0, places=12)
        self.assertAlmostEqual(result["raw_mse"], 202.5, places=12)
        self.assertAlmostEqual(result["mean_predictor_raw_mse"], 2.5, places=12)
        self.assertAlmostEqual(result["centered_r2"], -80.0, places=12)
        self.assertAlmostEqual(result["prediction_norm_mean"], 10.0 * np.sqrt(5.0), places=12)
        self.assertAlmostEqual(result["target_norm_mean"], np.sqrt(5.0), places=12)
        self.assertAlmostEqual(result["norm_ratio_mean"], 10.0, places=12)
        np.testing.assert_allclose(result["rowwise_directional_mse"], [0.0], atol=1e-12)
        np.testing.assert_allclose(result["rowwise_raw_mse"], [202.5], atol=1e-12)

    def test_activation_metrics_reject_shape_and_nonfinite_inputs(self):
        module = load_script("nano_eval_core")

        with self.assertRaisesRegex(ValueError, "matching non-empty 2D shapes"):
            module.activation_reconstruction_metrics(
                np.ones((2, 2)),
                np.ones((2, 3)),
                train_mean=np.zeros(2),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            module.activation_reconstruction_metrics(
                np.array([[1.0, np.nan]]),
                np.ones((1, 2)),
                train_mean=np.zeros(2),
            )

    def test_directional_mse_is_dimension_independent_two_one_minus_cosine(self):
        module = load_script("nano_eval_core")

        result = module.activation_reconstruction_metrics(
            np.array([[1.0, 0.0]]),
            np.array([[0.0, 1.0]]),
            train_mean=np.zeros(2),
        )

        self.assertAlmostEqual(result["cosine_mean"], 0.0, places=12)
        self.assertAlmostEqual(result["directional_mse"], 2.0, places=12)
        self.assertAlmostEqual(result["normalized_mse"], 2.0, places=12)
        self.assertAlmostEqual(result["unit_vector_feature_mse"], 1.0, places=12)
        np.testing.assert_allclose(result["rowwise_directional_mse"], [2.0])

    def test_paired_bootstrap_positive_when_candidate_loss_is_lower(self):
        module = load_script("nano_eval_core")
        baseline = np.array([4.0, 3.0, 2.0, 1.0])
        candidate = np.array([3.0, 2.0, 1.0, 0.5])

        result = module.paired_bootstrap_improvement(
            baseline,
            candidate,
            seed=7,
            resamples=2_000,
        )

        self.assertGreater(result["mean_improvement"], 0.0)
        self.assertGreater(result["ci95_low"], 0.0)
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["candidate_better_fraction"], 1.0)

    def test_paired_bootstrap_rejects_unpaired_or_nonfinite_values(self):
        module = load_script("nano_eval_core")

        with self.assertRaisesRegex(ValueError, "paired 1D arrays"):
            module.paired_bootstrap_improvement([1.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            module.paired_bootstrap_improvement([1.0, np.nan], [1.0, 2.0])

    def test_clustered_bootstrap_uses_independent_family_means(self):
        module = load_script("nano_eval_core")
        result = module.clustered_paired_bootstrap_improvement(
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 2.0, 2.0],
            ["large", "large", "small-a", "small-b"],
            seed=7,
            resamples=500,
        )

        self.assertEqual(result["cluster_count"], 3)
        self.assertEqual(result["row_count"], 4)
        self.assertAlmostEqual(result["row_weighted_mean_improvement"], 0.0)
        self.assertAlmostEqual(result["mean_improvement"], -1.0 / 3.0)

    def test_clustered_bootstrap_rejects_missing_cluster_ids(self):
        module = load_script("nano_eval_core")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            module.clustered_paired_bootstrap_improvement(
                [1.0, 1.0], [0.5, 0.5], ["family", ""]
            )

    def test_functional_metrics_identical_logits_are_perfect(self):
        module = load_script("nano_eval_core")
        logits = np.array([3.0, 2.0, 1.0, -1.0])

        result = module.functional_logit_metrics(logits, logits, top_ks=(2, 3))

        self.assertAlmostEqual(result["kl_original_to_patched"], 0.0, places=12)
        self.assertAlmostEqual(result["js_divergence"], 0.0, places=12)
        self.assertAlmostEqual(result["logit_pearson"], 1.0, places=12)
        self.assertEqual(result["top_2_overlap"], 1.0)
        self.assertEqual(result["top_3_overlap"], 1.0)
        self.assertEqual(result["original_top1_rank"], 1)

    def test_functional_metrics_detect_changed_ranking(self):
        module = load_script("nano_eval_core")

        result = module.functional_logit_metrics(
            np.array([4.0, 3.0, 2.0, 1.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            top_ks=(1, 2),
        )

        self.assertGreater(result["kl_original_to_patched"], 0.0)
        self.assertGreater(result["js_divergence"], 0.0)
        self.assertLess(result["logit_pearson"], 0.0)
        self.assertEqual(result["top_1_overlap"], 0.0)
        self.assertEqual(result["original_top1_rank"], 4)

    def test_functional_metrics_remain_finite_for_extreme_logits(self):
        module = load_script("nano_eval_core")

        result = module.functional_logit_metrics(
            np.array([1_000.0, 0.0, -1_000.0]),
            np.array([-1_000.0, 0.0, 1_000.0]),
            top_ks=(1, 2),
        )

        self.assertTrue(np.isfinite(result["kl_original_to_patched"]))
        self.assertTrue(np.isfinite(result["js_divergence"]))
        self.assertGreater(result["kl_original_to_patched"], 1_000.0)


if __name__ == "__main__":
    unittest.main()
