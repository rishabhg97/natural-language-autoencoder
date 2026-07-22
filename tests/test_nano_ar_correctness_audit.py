import importlib.util
import json
import pathlib
import tempfile
import unittest

import numpy as np
import pytest

pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoARCorrectnessAuditTests(unittest.TestCase):
    def test_expected_r27_boundary_values(self):
        audit = load_script("nano_ar_correctness_audit")

        self.assertEqual(audit.expected_zero_based_layer("R27"), 27)
        self.assertEqual(audit.expected_extraction_layer_index("R27"), 27)
        self.assertEqual(audit.expected_hidden_layers("R27"), 28)

    def test_doc_overlap_summary_marks_cross_split_overlap(self):
        audit = load_script("nano_ar_correctness_audit")

        summary = audit.doc_overlap_summary(
            {
                "train": ["doc-a", "doc-b", "doc-c"],
                "validation": ["doc-d", "doc-b"],
                "test": ["doc-e", "doc-a"],
            }
        )

        self.assertEqual(summary["overlap_count"], 2)
        self.assertFalse(summary["passed"])
        self.assertIn("doc-a", summary["overlap_doc_ids"])
        self.assertIn("doc-b", summary["overlap_doc_ids"])

    def test_identity_distance_is_zero_for_identity_and_positive_otherwise(self):
        audit = load_script("nano_ar_correctness_audit")

        identity = np.eye(4, dtype=np.float32)
        perturbed = identity.copy()
        perturbed[0, 1] = 0.25

        self.assertEqual(audit.identity_distance(identity), 0.0)
        self.assertGreater(audit.identity_distance(perturbed), 0.0)

    def test_checkpoint_config_audit_checks_expected_hidden_layers(self):
        audit = load_script("nano_ar_correctness_audit")

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = pathlib.Path(tmp)
            (ckpt / "config.json").write_text(json.dumps({"num_hidden_layers": 28}))

            report = audit.audit_checkpoint_config(ckpt, boundary_name="R27")

        self.assertTrue(report["passed"])
        self.assertEqual(report["critic_config_num_hidden_layers"], 28)
        self.assertEqual(report["expected_hidden_layers"], 28)

    def test_model_sidecar_accepts_extraction_layer_index(self):
        audit = load_script("nano_ar_correctness_audit")

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = pathlib.Path(tmp)
            (ckpt / "nla_meta.yaml").write_text("critic:\n  extraction_layer_index: 27\n")

            report = audit.audit_model_sidecar(ckpt, boundary_name="R27")

        self.assertTrue(report["passed"])
        self.assertEqual(report["critic_extraction_layer_index"], 27)

    def test_checkpoint_tensor_layout_matches_extractor_index(self):
        import torch
        from safetensors.torch import save_file

        audit = load_script("nano_ar_correctness_audit")

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = pathlib.Path(tmp)
            tensors = {
                "backbone.embeddings.weight": torch.ones(8, 4),
                **{
                    f"backbone.layers.{index}.mixer.weight": torch.ones(4, 4)
                    for index in range(28)
                },
            }
            save_file(tensors, ckpt / "model.safetensors")

            report = audit.audit_checkpoint_tensor_layout(ckpt, boundary_name="R27")

        self.assertTrue(report["passed"])
        self.assertEqual(report["expected_last_retained_block_index"], 27)
        self.assertEqual(report["observed_block_indices"], list(range(28)))
        self.assertEqual(report["lm_head_keys"], [])
        self.assertEqual(report["final_norm_keys"], [])

    def test_dataset_boundary_reads_extraction_sidecar(self):
        audit = load_script("nano_ar_correctness_audit")

        with tempfile.TemporaryDirectory() as tmp:
            parquet = pathlib.Path(tmp) / "train.parquet"
            pathlib.Path(str(parquet) + ".nla_meta.yaml").write_text(
                "extraction:\n  layer_index: 33\ncritic:\n  extraction_layer_index: 33\n"
            )

            report = audit.audit_dataset_boundary(parquet, boundary_name="R33")

        self.assertTrue(report["passed"])
        self.assertEqual(report["observed_extraction_layer_index"], 33)


if __name__ == "__main__":
    unittest.main()
