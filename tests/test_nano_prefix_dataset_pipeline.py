import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class NanoPrefixDatasetPipelineTests(unittest.TestCase):
    def test_av_verifier_receives_tokenizer_model(self):
        script = (ROOT / "scripts" / "nano_prefix_dataset_pipeline.sh").read_text()
        av_section = script.split('scripts/verify_nano_miles_av_dataset.py "$AV_SFT"', 1)[1]
        av_section = av_section.split("fi", 1)[0]

        self.assertIn('--tokenizer-model "$MODEL"', av_section)

    def test_publication_mode_forwards_required_provenance(self):
        script = (ROOT / "scripts" / "nano_prefix_dataset_pipeline.sh").read_text()

        self.assertIn('PUBLICATION_MODE="${PUBLICATION_MODE:-0}"', script)
        self.assertIn('--model-fingerprint-json "$MODEL_FINGERPRINT_JSON"', script)
        self.assertIn('--runtime-provenance-json "$RUNTIME_PROVENANCE_JSON"', script)
        self.assertIn("--publication-mode", script)

    def test_verifiers_receive_frozen_content_family_manifest(self):
        script = (ROOT / "scripts" / "nano_prefix_dataset_pipeline.sh").read_text()

        self.assertIn('CONTENT_FAMILY_MANIFEST="${CONTENT_FAMILY_MANIFEST:-}"', script)
        self.assertIn('--content-family-manifest "$CONTENT_FAMILY_MANIFEST"', script)
        self.assertIn('--content-family-manifest-sha256 "$CONTENT_FAMILY_MANIFEST_SHA256"', script)

    def test_source_parquet_hash_is_verified_before_extraction(self):
        script = (ROOT / "scripts" / "nano_prefix_dataset_pipeline.sh").read_text()

        self.assertIn(
            'EXPECTED_SOURCE_PARQUET_SHA256="${EXPECTED_SOURCE_PARQUET_SHA256:-}"',
            script,
        )
        self.assertIn("source parquet SHA-256 mismatch", script)

    def test_multiple_devices_use_sharded_extraction_wrapper(self):
        script = (ROOT / "scripts" / "nano_prefix_dataset_pipeline.sh").read_text()

        self.assertIn('EXTRACT_DEVICES="${EXTRACT_DEVICES:-}"', script)
        self.assertIn('EXTRACT_SHARD_ALIGNMENT="${EXTRACT_SHARD_ALIGNMENT:-document_batch}"', script)
        self.assertIn("scripts/nano_prefix_sharded_extract.py", script)
        self.assertIn('--devices "$EXTRACT_DEVICES"', script)
        self.assertIn('--shard-alignment "$EXTRACT_SHARD_ALIGNMENT"', script)

    def test_extraction_forwards_deterministic_execution_profile(self):
        script = (ROOT / "scripts" / "nano_prefix_dataset_pipeline.sh").read_text()

        self.assertIn('--deterministic-algorithms', script)
        self.assertIn('--no-allow-tf32', script)
        self.assertIn('--no-cudnn-benchmark', script)
        self.assertIn(
            '--float32-matmul-precision "$EXTRACT_FLOAT32_MATMUL_PRECISION"',
            script,
        )
        self.assertIn(
            '--cublas-workspace-config "$EXTRACT_CUBLAS_WORKSPACE_CONFIG"',
            script,
        )
        self.assertIn('--seed "$EXTRACT_SEED"', script)
        self.assertIn('execution = provenance.get("execution") or {}', script)
        self.assertIn('execution.get("deterministic_algorithms") is not True', script)
        self.assertIn('execution.get("allow_tf32") is not False', script)


if __name__ == "__main__":
    unittest.main()
