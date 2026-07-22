import importlib.util
import pathlib
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "nano_prefix_dataset_config.py"
    spec = importlib.util.spec_from_file_location("nano_prefix_dataset_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoPrefixDatasetConfigTests(unittest.TestCase):
    def test_publication_configs_predeclare_primary_and_seeded_critic_initialization(self):
        runner = load_script()
        primary_path = (
            ROOT
            / "configs"
            / "nano_data"
            / "publication"
            / "r33_deterministic_full275396.yaml"
        )
        independent_path = (
            ROOT
            / "configs"
            / "nano_data"
            / "publication"
            / "r33_deterministic_independent_critic_init.yaml"
        )

        primary = runner.load_config(primary_path)
        independent = runner.load_config(independent_path)
        _, primary_env = runner.build_launch(primary)
        _, independent_env = runner.build_launch(independent)

        self.assertEqual(primary_env["CRITIC_VALUE_HEAD_INIT"], "identity")
        self.assertEqual(primary_env["CRITIC_ROUTER_INIT"], "pretrained")
        self.assertEqual(independent_env["CRITIC_VALUE_HEAD_INIT"], "seeded_givens")
        self.assertEqual(independent_env["CRITIC_INITIALIZATION_SEED"], "314159")
        self.assertEqual(independent_env["CRITIC_ROUTER_INIT"], "seeded_relative_noise")
        self.assertEqual(primary_env["OUT"], independent_env["OUT"])
        self.assertNotEqual(primary_env["CRITIC"], independent_env["CRITIC"])
        self.assertTrue(primary_env["CRITIC"].startswith("/workspace/models/"))
        self.assertTrue(independent_env["CRITIC"].startswith("/workspace/models/"))

    def test_publication_config_renders_pipeline_environment(self):
        runner = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "dataset.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_prefix_dataset_pipeline.v1
                    python: /venv/bin/python
                    paths:
                      code_root: /code
                      model: /models/nano
                      source_parquet: /data/teacher.parquet
                      contract: /data/teacher.nla_meta.yaml
                      out: /out/publication-r33
                      extract_root: /out/publication-r33/extract
                      critic: /out/publication-r33/critic
                      model_fingerprint_json: /evidence/model.json
                      runtime_provenance_json: /evidence/runtime.json
                      content_family_manifest: /evidence/families.json
                    dataset:
                      layer: 33
                      layers: R33
                      row_start: 0
                      row_limit: null
                      slug: r33_publication_full275396
                      expected_rows: 275396
                      expected_d_model: 2688
                      expected_source_parquet_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
                      content_family_manifest_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
                    extraction:
                      batch_size: 2
                      source_batch_size: 4096
                      publication_mode: true
                      deterministic_algorithms: true
                      allow_tf32: false
                      cudnn_benchmark: false
                      float32_matmul_precision: highest
                      cublas_workspace_config: ":4096:8"
                      seed: 20260709
                      devices: ["0", "1", "2", "3", "4", "5", "6", "7"]
                      shard_alignment: document_batch
                    critic_initialization:
                      value_head: seeded_givens
                      seed: 314159
                      value_head_rotation_radians: 0.2
                      router: seeded_relative_noise
                      router_relative_std: 0.01
                    build:
                      ar: true
                      av: true
                      prep_critic: true
                      verify: true
                    """
                )
            )

            config = runner.load_config(path)
            command, env = runner.build_launch(config)

        self.assertEqual(command, ["bash", "scripts/nano_prefix_dataset_pipeline.sh"])
        self.assertEqual(env["EXPECTED_ROWS"], "275396")
        self.assertEqual(env["EXPECTED_SOURCE_PARQUET_SHA256"], "b" * 64)
        self.assertEqual(env["PUBLICATION_MODE"], "1")
        self.assertEqual(env["MODEL_FINGERPRINT_JSON"], "/evidence/model.json")
        self.assertEqual(env["CONTENT_FAMILY_MANIFEST"], "/evidence/families.json")
        self.assertEqual(env["CONTENT_FAMILY_MANIFEST_SHA256"], "a" * 64)
        self.assertEqual(env["BUILD_AR"], "1")
        self.assertEqual(env["ROW_LIMIT"], "")
        self.assertEqual(env["CRITIC_VALUE_HEAD_INIT"], "seeded_givens")
        self.assertEqual(env["CRITIC_INITIALIZATION_SEED"], "314159")
        self.assertEqual(env["CRITIC_ROUTER_INIT"], "seeded_relative_noise")
        self.assertEqual(env["EXTRACT_DEVICES"], "0,1,2,3,4,5,6,7")
        self.assertEqual(env["EXTRACT_SHARD_ALIGNMENT"], "document_batch")
        self.assertEqual(env["EXTRACT_DETERMINISTIC_ALGORITHMS"], "1")
        self.assertEqual(env["EXTRACT_ALLOW_TF32"], "0")
        self.assertEqual(env["EXTRACT_CUDNN_BENCHMARK"], "0")
        self.assertEqual(env["EXTRACT_FLOAT32_MATMUL_PRECISION"], "highest")
        self.assertEqual(env["EXTRACT_CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(env["EXTRACT_SEED"], "20260709")

    def test_publication_config_requires_deterministic_execution_profile(self):
        runner = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "dataset.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_prefix_dataset_pipeline.v1
                    paths:
                      code_root: /code
                      model: /models/nano
                      source_parquet: /data/teacher.parquet
                      contract: /data/teacher.nla_meta.yaml
                      out: /out/publication-r33
                      model_fingerprint_json: /evidence/model.json
                      runtime_provenance_json: /evidence/runtime.json
                    dataset:
                      layer: 33
                      slug: r33
                      expected_rows: 275396
                      expected_d_model: 2688
                      expected_source_parquet_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
                    extraction:
                      publication_mode: true
                      deterministic_algorithms: false
                      allow_tf32: false
                      cudnn_benchmark: false
                      float32_matmul_precision: highest
                      cublas_workspace_config: ":4096:8"
                      seed: 20260709
                    build:
                      verify: false
                    """
                )
            )

            with self.assertRaisesRegex(
                runner.PrefixDatasetConfigError,
                "deterministic_algorithms",
            ):
                runner.load_config(path)

    def test_expected_source_hash_must_be_sha256(self):
        runner = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "dataset.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_prefix_dataset_pipeline.v1
                    paths:
                      code_root: /code
                      model: /models/nano
                      source_parquet: /data/teacher.parquet
                      contract: /data/teacher.nla_meta.yaml
                      out: /out/r33
                    dataset:
                      layer: 33
                      slug: r33
                      expected_rows: 275396
                      expected_d_model: 2688
                      expected_source_parquet_sha256: not-a-hash
                    """
                )
            )

            with self.assertRaisesRegex(
                runner.PrefixDatasetConfigError,
                "expected_source_parquet_sha256",
            ):
                runner.load_config(path)

    def test_seeded_critic_initialization_requires_explicit_seed(self):
        runner = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "dataset.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_prefix_dataset_pipeline.v1
                    paths:
                      code_root: /code
                      model: /models/nano
                      source_parquet: /data/teacher.parquet
                      contract: /data/teacher.nla_meta.yaml
                      out: /out/publication-r33
                    dataset:
                      layer: 33
                      slug: r33
                      expected_rows: 275396
                      expected_d_model: 2688
                    critic_initialization:
                      value_head: seeded_givens
                      router: pretrained
                    """
                )
            )

            with self.assertRaisesRegex(
                runner.PrefixDatasetConfigError,
                "critic_initialization.seed",
            ):
                runner.load_config(path)

    def test_publication_config_requires_provenance_paths(self):
        runner = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "dataset.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_prefix_dataset_pipeline.v1
                    paths:
                      code_root: /code
                      model: /models/nano
                      source_parquet: /data/teacher.parquet
                      contract: /data/teacher.nla_meta.yaml
                      out: /out/publication-r33
                    dataset:
                      layer: 33
                      slug: r33
                      expected_rows: 275396
                      expected_d_model: 2688
                    extraction:
                      publication_mode: true
                    """
                )
            )

            with self.assertRaisesRegex(
                runner.PrefixDatasetConfigError,
                "model_fingerprint_json",
            ):
                runner.load_config(path)

    def test_extraction_devices_must_be_unique_nonempty_strings(self):
        runner = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "dataset.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_prefix_dataset_pipeline.v1
                    paths:
                      code_root: /code
                      model: /models/nano
                      source_parquet: /data/teacher.parquet
                      contract: /data/teacher.nla_meta.yaml
                      out: /out/publication-r33
                    dataset:
                      layer: 33
                      slug: r33
                      expected_rows: 10
                      expected_d_model: 2688
                    extraction:
                      devices: ["0", "0"]
                    """
                )
            )

            with self.assertRaisesRegex(
                runner.PrefixDatasetConfigError,
                "extraction.devices must contain unique",
            ):
                runner.load_config(path)


if __name__ == "__main__":
    unittest.main()
