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


class NanoARInformationCeilingTests(unittest.TestCase):
    def test_canonicalize_explanation_strips_tags_and_normalizes_space(self):
        ceiling = load_script("nano_ar_information_ceiling")

        text = " <explanation>Syntax feature:\n  The token Foo. </explanation> "

        self.assertEqual(ceiling.canonicalize_explanation(text), "syntax feature the token foo")

    def test_duplicate_group_floor_uses_normalized_group_mean(self):
        ceiling = load_script("nano_ar_information_ceiling")
        rows = [
            {"explanation": "same", "activation_vector": [1.0, 0.0]},
            {"explanation": "same", "activation_vector": [0.0, 1.0]},
            {"explanation": "unique", "activation_vector": [1.0, 0.0]},
        ]

        report = ceiling.duplicate_group_floor(rows, min_group_size=2)

        self.assertEqual(report["group_count"], 1)
        self.assertEqual(report["covered_row_count"], 2)
        self.assertAlmostEqual(report["mean_nmse"], 0.29289323, places=6)

    def test_knn_floor_predicts_mean_of_nearest_train_vectors(self):
        ceiling = load_script("nano_ar_information_ceiling")
        train_features = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
        train_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        eval_features = np.array([[0.2], [2.8]], dtype=np.float32)
        eval_vectors = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)

        report = ceiling.knn_floor(
            train_features=train_features,
            train_vectors=train_vectors,
            eval_features=eval_features,
            eval_vectors=eval_vectors,
            k=1,
        )

        self.assertEqual(report["row_count"], 2)
        self.assertAlmostEqual(report["mean_nmse"], 0.0, places=6)

    def test_baseline_keys_include_token_position_and_local_window(self):
        ceiling = load_script("nano_ar_information_ceiling")
        row = {
            "target_token": "ing",
            "target_token_id": 123,
            "target_position": 47,
            "tokens_before": ["active", "listen"],
            "tokens_after": ["with", "callers"],
        }

        keys = ceiling.baseline_keys_for_row(row, position_bucket_size=10)

        self.assertEqual(keys["target_token"], "tok:ing")
        self.assertEqual(keys["target_token_id"], "tok_id:123")
        self.assertEqual(keys["position_bucket"], "pos:40-49")
        self.assertEqual(keys["local_window"], "local:active listen <T> with callers")

    def test_baseline_keys_accept_stage3_token_sidecar_names(self):
        ceiling = load_script("nano_ar_information_ceiling")
        row = {
            "token_text": "ing",
            "token_id": 123,
            "token_position": 47,
        }

        keys = ceiling.baseline_keys_for_row(row, position_bucket_size=10)

        self.assertEqual(keys["target_token"], "tok:ing")
        self.assertEqual(keys["target_token_id"], "tok_id:123")
        self.assertEqual(keys["position_bucket"], "pos:40-49")

    def test_feature_text_can_append_stage3_token_hints_to_explanation(self):
        ceiling = load_script("nano_ar_information_ceiling")
        row = {
            "explanation": "The model is continuing a verb.",
            "token_text": "ing",
            "token_id": 123,
            "token_position": 47,
        }

        text = ceiling.feature_text_for_row(row, "explanation_all_token_hints", position_bucket_size=10)

        self.assertIn("The model is continuing a verb.", text)
        self.assertIn("target_token=ing", text)
        self.assertIn("target_token_id=123", text)
        self.assertIn("position_bucket=pos:40-49", text)


if __name__ == "__main__":
    unittest.main()
