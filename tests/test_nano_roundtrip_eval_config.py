import importlib.util
import pathlib
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoRoundtripEvalConfigTests(unittest.TestCase):
    def test_dataset_controls_are_rendered(self):
        runner = load_script("nano_roundtrip_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "roundtrip.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_roundtrip_eval.v1
                    paths:
                      av_hf_checkpoint: /hf/av
                      ar_checkpoint_dir: /hf/ar
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      validation_control_parquet: /data/validation_controls.parquet
                      report_json: /out/report.json
                    eval:
                      validation_limit: 8
                      eval_splits: [validation]
                      dataset_controls: [source_context, source_raw]
                    """
                )
            )

            command = runner.build_command(runner.load_config(path), config_path=path)

        index = command.index("--dataset-controls")
        self.assertEqual(command[index + 1 : index + 3], ["source_context", "source_raw"])
    def test_config_renders_roundtrip_command(self):
        runner = load_script("nano_roundtrip_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = root / "roundtrip.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_roundtrip_eval.v1
                    python: /venv/bin/python
                    paths:
                      av_hf_checkpoint: /hf/av
                      ar_checkpoint_dir: /hf/ar
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      validation_control_parquet: /data/validation_controls.parquet
                      test_parquet: /data/test.parquet
                      report_json: /out/report.json
                      expected_generation_protocol_json: /out/protocol.json
                      prediction_cache_npz: /out/predictions.npz
                      baseline_report_json: /out/baseline.json
                      length_baseline_generated_jsonl: /out/baseline_generated.jsonl
                      content_family_manifest: /data/content_families.json
                      content_family_coverage: /data/content_family_coverage.json
                    eval:
                      validation_limit: 8
                      test_limit: 8
                      generation_controls: [real]
                      dataset_controls: [source_context, source_raw]
                      max_new_tokens: 64
                      seed: 20260708
                      generation_prefix: ""
                      stop_text: "</explanation>"
                      generated_text_fallback: empty
                      generation_backend: cache
                      generation_workers: 2
                      generation_max_parallel_workers: 1
                      generation_worker_devices: ["0", "1"]
                      stream_generated: true
                      resume_generated: true
                      generation_only: true
                      progress_every: 8
                      min_control_win_fraction: 0.9
                      min_baseline_win_fraction: 0.51
                      min_baseline_relative_improvement: 0.1
                      require_baseline_ci_positive: true
                      require_clustered_baseline_ci: true
                      require_baseline_dataset_match: true
                      bootstrap_samples: 1234
                      bootstrap_seed: 17
                      permutation_samples: 4321
                      permutation_seed: 19
                      selection_strategy: family_stratified
                      selection_seed: 20260708
                      require_family_level_inference: true
                      min_independent_families: 100
                      min_closed_fraction: 0.8
                      min_usable_fraction: 0.95
                      injection_scale: "75"
                      ar_device_map: cuda:0
                      ar_low_cpu_mem_usage: true
                      av_model_fingerprint: model-sha
                      av_tokenizer_fingerprint: tokenizer-sha
                      require_generation_protocol_match: true
                    """
                )
            )

            loaded = runner.load_config(config)
            command = runner.build_command(loaded, config_path=config)

        self.assertEqual(command[0], "/venv/bin/python")
        self.assertIn("scripts/eval_nano_av_ar_roundtrip_gate.py", command)
        self.assertEqual(command[command.index("--av-hf-checkpoint") + 1], "/hf/av")
        self.assertEqual(command[command.index("--ar-checkpoint-dir") + 1], "/hf/ar")
        self.assertEqual(
            command[command.index("--prediction-cache-npz") + 1],
            "/out/predictions.npz",
        )
        self.assertEqual(
            command[command.index("--expected-generation-protocol-json") + 1],
            "/out/protocol.json",
        )
        self.assertEqual(
            command[command.index("--length-baseline-generated-jsonl") + 1],
            "/out/baseline_generated.jsonl",
        )
        self.assertEqual(command[command.index("--generation-controls") + 1], "real")
        self.assertEqual(
            command[command.index("--validation-control-parquet") + 1],
            "/data/validation_controls.parquet",
        )
        self.assertEqual(
            command[
                command.index("--dataset-controls") + 1 :
                command.index("--dataset-controls") + 3
            ],
            ["source_context", "source_raw"],
        )
        self.assertEqual(command[command.index("--validation-limit") + 1], "8")
        self.assertEqual(command[command.index("--seed") + 1], "20260708")
        self.assertEqual(command[command.index("--generation-prefix") + 1], "")
        self.assertEqual(command[command.index("--stop-text") + 1], "</explanation>")
        self.assertEqual(command[command.index("--generation-backend") + 1], "cache")
        self.assertEqual(command[command.index("--generation-workers") + 1], "2")
        self.assertEqual(
            command[command.index("--generation-max-parallel-workers") + 1], "1"
        )
        self.assertEqual(command[command.index("--generation-worker-devices") + 1: command.index("--generation-worker-devices") + 3], ["0", "1"])
        self.assertIn("--av-low-cpu-mem-usage", command)
        self.assertEqual(command[command.index("--ar-device-map") + 1], "cuda:0")
        self.assertIn("--ar-low-cpu-mem-usage", command)
        self.assertIn("--stream-generated", command)
        self.assertIn("--resume-generated", command)
        self.assertIn("--generation-only", command)
        self.assertNotIn("--reuse-generated", command)
        self.assertEqual(command[command.index("--min-control-win-fraction") + 1], "0.9")
        self.assertEqual(command[command.index("--min-baseline-win-fraction") + 1], "0.51")
        self.assertEqual(
            command[command.index("--min-baseline-relative-improvement") + 1],
            "0.1",
        )
        self.assertIn("--require-baseline-ci-positive", command)
        self.assertIn("--require-clustered-baseline-ci", command)
        self.assertIn("--require-baseline-dataset-match", command)
        self.assertEqual(command[command.index("--bootstrap-samples") + 1], "1234")
        self.assertEqual(command[command.index("--bootstrap-seed") + 1], "17")
        self.assertEqual(command[command.index("--permutation-samples") + 1], "4321")
        self.assertEqual(command[command.index("--permutation-seed") + 1], "19")
        self.assertEqual(
            command[command.index("--content-family-manifest") + 1],
            "/data/content_families.json",
        )
        self.assertEqual(
            command[command.index("--content-family-coverage") + 1],
            "/data/content_family_coverage.json",
        )
        self.assertEqual(
            command[command.index("--selection-strategy") + 1],
            "family_stratified",
        )
        self.assertEqual(command[command.index("--selection-seed") + 1], "20260708")
        self.assertIn("--require-family-level-inference", command)
        self.assertEqual(
            command[command.index("--min-independent-families") + 1],
            "100",
        )
        self.assertEqual(command[command.index("--min-usable-fraction") + 1], "0.95")
        self.assertEqual(
            command[command.index("--av-model-fingerprint") + 1], "model-sha"
        )
        self.assertEqual(
            command[command.index("--av-tokenizer-fingerprint") + 1],
            "tokenizer-sha",
        )
        self.assertIn("--require-generation-protocol-match", command)

    def test_publication_protocol_match_requires_explicit_fingerprints(self):
        runner = load_script("nano_roundtrip_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "roundtrip.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_roundtrip_eval.v1
                    paths:
                      av_hf_checkpoint: /hf/av
                      ar_checkpoint_dir: /hf/ar
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      test_parquet: /data/test.parquet
                      report_json: /out/report.json
                    eval:
                      validation_limit: 8
                      test_limit: 8
                      require_generation_protocol_match: true
                    """
                )
            )

            with self.assertRaisesRegex(
                runner.RoundtripConfigError, "explicit model and tokenizer fingerprints"
            ):
                runner.load_config(path)

    def test_publication_protocol_match_rejects_nonempty_forced_prefix(self):
        runner = load_script("nano_roundtrip_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "roundtrip.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_roundtrip_eval.v1
                    paths:
                      av_hf_checkpoint: /hf/av
                      ar_checkpoint_dir: /hf/ar
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      test_parquet: /data/test.parquet
                      report_json: /out/report.json
                    eval:
                      validation_limit: 8
                      test_limit: 8
                      generation_prefix: "<explanation>"
                      av_model_fingerprint: model-sha
                      av_tokenizer_fingerprint: tokenizer-sha
                      require_generation_protocol_match: true
                    """
                )
            )

            with self.assertRaisesRegex(
                runner.RoundtripConfigError, "empty generation_prefix"
            ):
                runner.load_config(path)

    def test_length_controls_require_a_baseline_report(self):
        runner = load_script("nano_roundtrip_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "roundtrip.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_roundtrip_eval.v1
                    paths:
                      av_hf_checkpoint: /hf/av
                      ar_checkpoint_dir: /hf/ar
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      test_parquet: /data/test.parquet
                      report_json: /out/report.json
                      length_baseline_generated_jsonl: /out/baseline.jsonl
                    eval:
                      validation_limit: 8
                      test_limit: 8
                    """
                )
            )

            with self.assertRaisesRegex(
                runner.RoundtripConfigError,
                "requires paths.baseline_report_json",
            ):
                runner.load_config(path)

    def test_validation_only_config_renders_explicit_eval_splits(self):
        runner = load_script("nano_roundtrip_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "roundtrip.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_roundtrip_eval.v1
                    paths:
                      av_hf_checkpoint: /hf/av
                      ar_checkpoint_dir: /hf/ar
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      test_parquet: /data/test.parquet
                      report_json: /out/report.json
                    eval:
                      validation_limit: 8
                      test_limit: 8
                      eval_splits: [validation]
                    """
                )
            )

            loaded = runner.load_config(path)
            command = runner.build_command(loaded, config_path=path)

        index = command.index("--eval-splits")
        self.assertEqual(command[index + 1 : index + 2], ["validation"])

    def test_family_level_inference_requires_manifest_and_stratified_selection(self):
        runner = load_script("nano_roundtrip_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "roundtrip.yaml"
            base = textwrap.dedent(
                """
                schema_version: nano_roundtrip_eval.v1
                paths:
                  av_hf_checkpoint: /hf/av
                  ar_checkpoint_dir: /hf/ar
                  train_parquet: /data/train.parquet
                  validation_parquet: /data/validation.parquet
                  test_parquet: /data/test.parquet
                  report_json: /out/report.json
                eval:
                  validation_limit: 8
                  test_limit: 8
                  require_family_level_inference: true
                """
            )
            path.write_text(base)
            with self.assertRaisesRegex(
                runner.RoundtripConfigError, "content_family_manifest"
            ):
                runner.load_config(path)

            path.write_text(
                base.replace(
                    "report_json: /out/report.json",
                    "report_json: /out/report.json\n  content_family_manifest: /data/families.json",
                )
            )
            with self.assertRaisesRegex(
                runner.RoundtripConfigError, "content_family_coverage"
            ):
                runner.load_config(path)

            path.write_text(
                base.replace(
                    "report_json: /out/report.json",
                    "report_json: /out/report.json\n"
                    "  content_family_manifest: /data/families.json\n"
                    "  content_family_coverage: /data/coverage.json",
                )
            )
            with self.assertRaisesRegex(
                runner.RoundtripConfigError, "family_stratified"
            ):
                runner.load_config(path)

    def test_reuse_generated_mode_renders_and_conflicts_are_rejected(self):
        runner = load_script("nano_roundtrip_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "roundtrip.yaml"
            base = textwrap.dedent(
                """
                schema_version: nano_roundtrip_eval.v1
                paths:
                  av_hf_checkpoint: /hf/av
                  ar_checkpoint_dir: /hf/ar
                  train_parquet: /data/train.parquet
                  validation_parquet: /data/validation.parquet
                  test_parquet: /data/test.parquet
                  report_json: /out/report.json
                  generated_jsonl: /out/generated.jsonl
                eval:
                  validation_limit: 8
                  test_limit: 8
                  reuse_generated: true
                """
            )
            path.write_text(base)
            loaded = runner.load_config(path)
            command = runner.build_command(loaded, config_path=path)
            self.assertIn("--reuse-generated", command)
            self.assertNotIn("--generation-only", command)

            path.write_text(base.replace("reuse_generated: true", "reuse_generated: true\n  generation_only: true"))
            with self.assertRaisesRegex(runner.RoundtripConfigError, "mutually exclusive"):
                runner.load_config(path)

    def test_checked_in_r27_full_control_baselines_use_resumable_cached_generation(self):
        runner = load_script("nano_roundtrip_eval_config")
        for config in [
            ROOT / "configs" / "nano_roundtrip" / "r27_baseline_64_full_controls_prefix256.yaml",
            ROOT / "configs" / "nano_roundtrip" / "r27_baseline_256_full_controls_prefix256.yaml",
        ]:
            loaded = runner.load_config(config)
            command = runner.build_command(loaded, config_path=config)

            self.assertIn("--resume-generated", command, str(config))
            self.assertIn("--stream-generated", command, str(config))
            self.assertEqual(command[command.index("--generation-backend") + 1], "cache")
            self.assertEqual(command[command.index("--generation-workers") + 1], "2")
            self.assertEqual(command[command.index("--generation-worker-devices") + 1: command.index("--generation-worker-devices") + 3], ["0", "1"])

    def test_checked_in_r27_256_baseline_pins_full_control_accelerated_generation(self):
        runner = load_script("nano_roundtrip_eval_config")
        config = ROOT / "configs" / "nano_roundtrip" / "r27_baseline_256_full_controls_prefix256.yaml"

        loaded = runner.load_config(config)
        command = runner.build_command(loaded, config_path=config)

        self.assertEqual(command[command.index("--validation-limit") + 1], "256")
        self.assertEqual(command[command.index("--test-limit") + 1], "256")
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "256")
        self.assertEqual(
            command[command.index("--generation-controls") + 1: command.index("--generation-controls") + 6],
            ["real", "shuffled", "zero", "mean", "none"],
        )
        self.assertEqual(command[command.index("--generation-backend") + 1], "cache")
        self.assertEqual(command[command.index("--generation-workers") + 1], "2")
        self.assertEqual(command[command.index("--generation-worker-devices") + 1: command.index("--generation-worker-devices") + 3], ["0", "1"])
        self.assertIn("--stream-generated", command)
        self.assertIn("--resume-generated", command)

    def test_retry4_update16_diagnostic_uses_hardened_512_gate(self):
        runner = load_script("nano_roundtrip_eval_config")
        config = (
            ROOT
            / "configs"
            / "nano_roundtrip"
            / "r33_retry4_update16_v512_score.yaml"
        )

        loaded = runner.load_config(config)
        command = runner.build_command(loaded, config_path=config)

        self.assertEqual(command[command.index("--validation-limit") + 1], "512")
        self.assertEqual(command[command.index("--test-limit") + 1], "512")
        self.assertEqual(
            command[command.index("--min-baseline-relative-improvement") + 1],
            "0.1",
        )
        self.assertEqual(
            command[command.index("--min-baseline-win-fraction") + 1],
            "0.5",
        )
        self.assertIn("--require-baseline-ci-positive", command)
        self.assertIn("--require-clustered-baseline-ci", command)
        self.assertIn("--require-baseline-dataset-match", command)
        self.assertIn("--reuse-generated", command)

    def test_config_can_enable_score_phase_ar_device_profile(self):
        runner = load_script("nano_roundtrip_eval_config")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "roundtrip.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_roundtrip_eval.v1
                    paths:
                      av_hf_checkpoint: /hf/av
                      ar_checkpoint_dir: /hf/ar
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      report_json: /out/report.json
                    eval:
                      validation_limit: 8
                      collect_ar_device_profile: true
                    """
                )
            )

            command = runner.build_command(runner.load_config(path), config_path=path)

        self.assertIn("--collect-ar-device-profile", command)


if __name__ == "__main__":
    unittest.main()
