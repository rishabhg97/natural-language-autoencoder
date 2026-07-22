import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "verify_nano_critic_initialization.py"
    spec = importlib.util.spec_from_file_location("verify_nano_critic_initialization", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def manifest(*, independent: bool) -> dict:
    return {
        "schema_version": "nano_critic_initialization.v1",
        "base_model": "/models/nano",
        "dataset_sidecar": "/data/r33.parquet",
        "extraction_layer_index": 33,
        "torch_dtype": "bfloat16",
        "value_head": {
            "mode": "seeded_givens" if independent else "identity",
            "seed": 314159 if independent else None,
            "after_sha256": "b" * 64 if independent else "a" * 64,
        },
        "router": {
            "mode": "seeded_relative_noise" if independent else "pretrained",
            "seed": 314159 if independent else None,
            "parameter_count": 30,
            "before_sha256": "c" * 64,
            "after_sha256": "d" * 64 if independent else "c" * 64,
        },
    }


class VerifyNanoCriticInitializationTests(unittest.TestCase):
    def test_accepts_shared_provenance_with_distinct_seeded_initialization(self):
        verifier = load_script()

        report = verifier.verify_initializations(
            manifest(independent=False),
            manifest(independent=True),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["independent_seed"], 314159)
        self.assertEqual(report["errors"], [])

    def test_rejects_reused_value_head_initialization(self):
        verifier = load_script()
        primary = manifest(independent=False)
        independent = manifest(independent=True)
        independent["value_head"]["after_sha256"] = primary["value_head"]["after_sha256"]

        report = verifier.verify_initializations(primary, independent)

        self.assertFalse(report["passed"])
        self.assertIn("value-head hashes must differ", report["errors"])

    def test_rejects_different_base_provenance(self):
        verifier = load_script()
        independent = manifest(independent=True)
        independent["base_model"] = "/models/other"

        report = verifier.verify_initializations(
            manifest(independent=False),
            independent,
        )

        self.assertFalse(report["passed"])
        self.assertIn("base_model must match", report["errors"])


if __name__ == "__main__":
    unittest.main()
