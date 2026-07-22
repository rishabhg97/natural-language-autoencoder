import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import textwrap
import types
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoAVRunnerSpecTests(unittest.TestCase):
    def test_frozen_content_family_manifest_must_match_expected_hash(self):
        runner = load_script("nano_av_runner")
        with tempfile.TemporaryDirectory() as tmp:
            manifest = pathlib.Path(tmp) / "families.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_content_family_manifest.v1",
                        "doc_assignments": {"doc-a": "family-a"},
                        "family_splits": {"family-a": "train"},
                    },
                    sort_keys=True,
                )
            )
            expected = runner._sha256_file(manifest)

            accepted = runner._verify_content_family_manifest(manifest, expected)
            with self.assertRaisesRegex(
                runner.SpecValidationError,
                "content family manifest hash mismatch",
            ):
                runner._verify_content_family_manifest(manifest, "0" * 64)

        self.assertEqual(accepted["schema_version"], "nano_content_family_manifest.v1")

    def test_content_family_manifest_split_requires_manifest_path(self):
        runner = load_script("nano_av_runner")
        spec = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "nano_ar"
                / "publication"
                / "r33_family_clean_sft.yaml"
            ).read_text()
        )
        spec["dataset"]["split_mode"] = "content_family_manifest"
        spec["dataset"].pop("content_family_manifest", None)

        with self.assertRaisesRegex(
            runner.SpecValidationError,
            "dataset.content_family_manifest is required",
        ):
            runner.validate_spec(spec)

    def test_distributed_timeout_minutes_must_be_positive(self):
        runner = load_script("nano_av_runner")
        spec = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "nano_ar"
                / "publication"
                / "r33_family_clean_sft.yaml"
            ).read_text()
        )
        spec["training"]["distributed_timeout_minutes"] = 0

        with self.assertRaisesRegex(
            runner.SpecValidationError,
            "training.distributed_timeout_minutes must be positive",
        ):
            runner.validate_spec(spec)

    def test_mamba_kernel_mode_must_be_known(self):
        runner = load_script("nano_av_runner")
        spec = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "nano_ar"
                / "publication"
                / "r33_family_clean_sft.yaml"
            ).read_text()
        )
        spec["training"]["mamba_kernel_mode"] = "mystery"

        with self.assertRaisesRegex(
            runner.SpecValidationError,
            "training.mamba_kernel_mode",
        ):
            runner.validate_spec(spec)

    def test_critic_initialization_verification_report_must_pass(self):
        runner = load_script("nano_av_runner")
        with tempfile.TemporaryDirectory() as tmp:
            report = pathlib.Path(tmp) / "critic-init-verification.json"
            report.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_critic_initialization_verification.v1",
                        "passed": False,
                    }
                )
            )

            with self.assertRaisesRegex(
                runner.SpecValidationError,
                "critic initialization verification did not pass",
            ):
                runner._verify_critic_initialization_report(report)

            report.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_critic_initialization_verification.v1",
                        "passed": True,
                    }
                )
            )
            accepted = runner._verify_critic_initialization_report(report)

        self.assertTrue(accepted["passed"])

    def _write_spec(self, text):
        tmp = tempfile.TemporaryDirectory()
        path = pathlib.Path(tmp.name) / "spec.yaml"
        path.write_text(textwrap.dedent(text))
        self.addCleanup(tmp.cleanup)
        return path

    def _write_tiny_av_dataset(
        self,
        root: pathlib.Path,
        *,
        docs: list[str],
        repeated_response: bool = False,
    ) -> pathlib.Path:
        parquet = root / "av_sft.parquet"
        table = pa.table(
            {
                "prompt": pa.array([f"prompt {i} <concept><INJECT></concept>" for i in range(len(docs))]),
                "response": pa.array(
                    [
                        "<explanation>same content</explanation>" if repeated_response else f"<explanation>row {i}</explanation>"
                        for i in range(len(docs))
                    ]
                ),
                "activation_vector": pa.array(
                    [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(docs))],
                    type=pa.list_(pa.float32(), 4),
                ),
                "doc_id": pa.array(docs),
            }
        )
        pq.write_table(table, parquet)
        parquet.with_name(parquet.name + ".nla_meta.yaml").write_text(
            yaml.safe_dump(
                {
                    "kind": "nla_dataset",
                    "schema_version": 1,
                    "dataset_id": "tiny-av",
                    "stage": "av_sft",
                    "row_count": len(docs),
                    "extraction": {"d_model": 4, "layer_index": 33},
                },
                sort_keys=False,
            )
        )
        return parquet

    def test_complete_performance_rejects_unsafe_hero_settings(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: unsafe-hero
              experiment_class: complete-performance
              output_root: /tmp/nano
              wandb_mode: online
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 99570
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: false
              final_batch_policy: drop_remainder
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 8
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: skip_diagnostic
              timing_debug: true
            checkpoint:
              save_interval: 500
              keep_last: 3
              save_enabled: false
              no_save_optim: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.load_and_validate_spec(spec_path)

        message = str(caught.exception)
        self.assertIn("wandb_mode must be offline", message)
        self.assertIn("complete-performance requires train/validation/test = 0.9/0.05/0.05", message)
        self.assertIn("complete-performance requires materialized splits", message)
        self.assertIn("complete-performance cannot disable checkpoint saves", message)
        self.assertIn("complete-performance requires optimizer/scheduler resume state", message)
        self.assertIn("complete-performance cannot use no_save_optim", message)
        self.assertIn("skip_diagnostic grad norm is not allowed", message)

    def test_medium_small_spec_renders_miles_command_from_materialized_train_split(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: medium
              experiment_class: medium-small
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 8
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: skip_diagnostic
              timing_debug: true
            checkpoint:
              save_interval: 10
              keep_last: 3
              save_enabled: true
              no_save_optim: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)
        command = runner.render_miles_command(spec, "/tmp/splits/train_padded.parquet", "/tmp/run")

        self.assertIn("--prompt-data", command)
        self.assertIn("/tmp/splits/train_padded.parquet", command)
        self.assertIn("--global-batch-size", command)
        self.assertIn("96", command)
        self.assertIn("--micro-batch-size", command)
        self.assertIn("8", command)
        self.assertIn("--nla-skip-grad-norm", command)
        self.assertIn("--nla-timing-debug", command)
        self.assertIn("--no-save-optim", command)
        self.assertNotIn("--wandb-mode online", " ".join(command))

    def test_ar_sft_spec_renders_qwen_faithful_critic_command(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: nano-ar-small
              experiment_class: small-smoke
              output_root: /tmp/nano-ar-out
              wandb_mode: offline
              wandb_group: nano-ar-miles-fsdp2-sft
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              critic_init_model_id: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r27-critic-init
              input_ar_sft: /tmp/ar_sft.parquet
            dataset:
              row_limit: 96
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              objective: ar_sft
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 8
              micro_batch_size: 1
              rollout_batch_size: 8
              lr: 1e-5
              grad_norm_policy: skip_diagnostic
              timing_debug: true
            checkpoint:
              save_interval: 1
              keep_last: 1
              save_enabled: true
              no_save_optim: false
              require_optimizer_state_for_hero: false
            eval:
              controls: [teacher, teacher_shuffled, blank, generic, mean]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)
        command = runner.render_miles_command(spec, "/tmp/splits/train_padded.parquet", "/tmp/run")

        self.assertIn("--loss-type", command)
        self.assertEqual(command[command.index("--loss-type") + 1], "custom_loss")
        self.assertIn("--custom-loss-function-path", command)
        self.assertIn("nla.loss.nla_critic_loss", command)
        self.assertIn("--rollout-function-path", command)
        self.assertIn("nla.rollout.sft_critic.generate_rollout", command)
        self.assertIn("--nla-model-is-critic", command)
        self.assertIn("--hf-checkpoint", command)
        self.assertEqual(
            command[command.index("--hf-checkpoint") + 1],
            "/workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r27-critic-init",
        )
        self.assertNotIn("--nla-injection-scale", command)

    def test_ar_sft_allows_batched_padded_critic_training_without_packed_acknowledgement(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: nano-ar-batched
              experiment_class: small-smoke
              output_root: /tmp/nano-ar-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              critic_init_model_id: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r27-critic-init
              input_ar_sft: /tmp/ar_sft.parquet
            dataset:
              row_limit: 96
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              objective: ar_sft
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 8
              micro_batch_size: 4
              rollout_batch_size: 8
              lr: 1e-5
              grad_norm_policy: skip_diagnostic
            checkpoint:
              save_interval: 1
              keep_last: 1
              save_enabled: true
              no_save_optim: false
              require_optimizer_state_for_hero: false
            eval:
              controls: [teacher, teacher_shuffled, blank, generic, mean]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)
        command = runner.render_miles_command(spec, "/tmp/splits/train_padded.parquet", "/tmp/run")

        self.assertIn("--micro-batch-size", command)
        self.assertEqual(command[command.index("--micro-batch-size") + 1], "4")
        self.assertIn("--nla-model-is-critic", command)

    def test_keep_last_one_is_valid_for_minimal_resume_retention(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: minimal-retention
              experiment_class: medium-small
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 8
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
            checkpoint:
              save_interval: 10
              keep_last: 1
              save_enabled: true
              no_save_optim: false
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)

        self.assertEqual(spec["checkpoint"]["keep_last"], 1)

    def test_checked_in_specs_are_valid(self):
        runner = load_script("nano_av_runner")
        specs = [
            ROOT / "configs" / "nano_av" / "small_smoke.yaml",
            ROOT / "configs" / "nano_av" / "medium_small_miles_fsdp2.yaml",
            ROOT / "configs" / "nano_av" / "hero_100k_miles_fsdp2.yaml",
            ROOT / "configs" / "nano_av" / "diagnostics" / "batch_scaling.yaml",
            ROOT / "configs" / "nano_av" / "diagnostics" / "resume_smoke.yaml",
            ROOT / "configs" / "nano_av" / "hpo" / "r33_100k_lr1e5_gb192_mb2_seq1152_dyn512.yaml",
            ROOT / "configs" / "nano_av" / "hpo" / "r33_dedup_av_20k_lr1e4_cosine_warmup5_gb192_mb1_seq1152_dyn1152.yaml",
            ROOT / "configs" / "nano_av" / "hpo" / "r33_dedup_av_20k_lr5e5_cosine_warmup5_gb192_mb1_seq1152_dyn1152.yaml",
            ROOT / "configs" / "nano_av" / "hpo" / "r33_dedup_av_full_lr1e4_cosine_warmup5_gb192_mb1_seq1152_dyn1152.yaml",
            ROOT / "configs" / "nano_ar" / "small_smoke.yaml",
            ROOT / "configs" / "nano_ar" / "medium_small_miles_fsdp2.yaml",
            ROOT / "configs" / "nano_ar" / "hero_100k_miles_fsdp2.yaml",
            ROOT / "configs" / "nano_ar" / "diagnostics" / "resume_smoke.yaml",
            ROOT / "configs" / "nano_ar" / "hpo" / "r33_100k_lr2e5_cosine_gb192_mb8.yaml",
            ROOT / "configs" / "nano_ar" / "hpo" / "r33_100k_lr1e5_cosine_gb192_mb8.yaml",
            ROOT / "configs" / "nano_ar" / "hpo" / "r33_dedup_smoke_20k_lr2e5_cosine_warmup20_gb192_mb96.yaml",
            ROOT / "configs" / "nano_ar" / "hpo" / "r33_dedup_full_lr5e5_cosine_warmup25_gb192_mb96.yaml",
            ROOT / "configs" / "nano_ar" / "hpo" / "r27_wide_probe_best1547_lr3e5_cosine_128steps.yaml",
            ROOT / "configs" / "nano_ar" / "hpo" / "r27_wide_probe_best1547_lr1e5_constant_128steps.yaml",
            ROOT / "configs" / "nano_ar" / "hpo" / "r27_wide_probe_best1547_lr5e6_cosine_128steps.yaml",
            ROOT / "configs" / "nano_ar" / "hpo" / "r27_wide_probe_fullscan_lr2e5_cosine_192steps.yaml",
            ROOT / "configs" / "nano_ar" / "hpo" / "r27_wide_probe_fullscan_lr5e5_cosine_128steps.yaml",
        ]

        for path in specs:
            with self.subTest(path=path):
                spec = runner.load_and_validate_spec(path)
                self.assertEqual(spec["run"]["wandb_mode"], "offline")

    def test_prepare_run_materializes_splits_and_writes_run_plan(self):
        runner = load_script("nano_av_runner")
        docs = ["doc-a", "doc-a", "doc-b", "doc-b", "doc-c", "doc-c", "doc-d", "doc-d"]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "av_sft.parquet"
            output_root = root / "runs"
            table = pa.table(
                {
                    "prompt": pa.array([f"prompt {i} <concept><INJECT></concept>" for i in range(len(docs))]),
                    "response": pa.array([f"<explanation>row {i}</explanation>" for i in range(len(docs))]),
                    "activation_vector": pa.array(
                        [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(docs))],
                        type=pa.list_(pa.float32(), 4),
                    ),
                    "doc_id": pa.array(docs),
                }
            )
            pq.write_table(table, parquet)
            parquet.with_name(parquet.name + ".nla_meta.yaml").write_text(
                yaml.safe_dump(
                    {
                        "kind": "nla_dataset",
                        "schema_version": 1,
                        "dataset_id": "tiny-av",
                        "stage": "av_sft",
                        "row_count": len(docs),
                        "extraction": {"d_model": 4, "layer_index": 27},
                    },
                    sort_keys=False,
                )
            )
            spec_path = self._write_spec(
                f"""
                run:
                  name: medium
                  experiment_class: medium-small
                  output_root: {output_root}
                  wandb_mode: offline
                paths:
                  code_root: /workspace/interp/code/nano30b-nla-pilot-current
                  miles_root: /workspace/interp/code/miles-051cd15
                  model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
                  input_av_sft: {parquet}
                dataset:
                  row_limit: 8
                  split_mode: doc
                  fractions: {{train: 0.5, validation: 0.25, test: 0.25}}
                  materialize_splits: true
                  final_batch_policy: pad_with_train_duplicates
                  expected_d_model: 4
                training:
                  backend: miles_fsdp2
                  epochs: 1
                  global_batch_size: 3
                  micro_batch_size: 1
                  rollout_batch_size: 3
                  lr: 1e-5
                  injection_scale: 75
                  grad_norm_policy: skip_diagnostic
                  timing_debug: true
                checkpoint:
                  save_interval: 10
                  keep_last: 3
                  save_enabled: true
                  require_optimizer_state_for_hero: false
                eval:
                  controls: [real, shuffled, zero, mean, none]
                """
            )

            spec = runner.load_and_validate_spec(spec_path)
            plan = runner.prepare_run(spec, run_id="unit-run")
            plan_file = json.loads((output_root / "unit-run" / "run_plan.json").read_text())
            split_manifest_exists = (output_root / "unit-run" / "splits" / "split_manifest.json").exists()

            self.assertEqual(plan, plan_file)
            self.assertEqual(plan["run_dir"], str(output_root / "unit-run"))
            self.assertEqual(plan["split_manifest"]["train"]["padded_row_count"], 6)
            self.assertIn(str(output_root / "unit-run" / "splits" / "train_padded.parquet"), plan["command"])
            self.assertIn("--num-rollout", plan["command"])
            self.assertEqual(plan["command"][plan["command"].index("--num-rollout") + 1], "2")
            self.assertEqual(plan["num_rollout"], 2)
            self.assertTrue(split_manifest_exists)

    def test_prepare_run_can_reuse_cached_materialized_splits_with_rewritten_paths(self):
        runner = load_script("nano_av_runner")
        docs = ["doc-a", "doc-a", "doc-b", "doc-b", "doc-c", "doc-c", "doc-d", "doc-d"]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = self._write_tiny_av_dataset(root, docs=docs)
            output_root = root / "runs"
            split_cache_dir = root / "split-cache"
            spec_path = self._write_spec(
                f"""
                run:
                  name: cached-medium
                  experiment_class: medium-small
                  output_root: {output_root}
                  wandb_mode: offline
                paths:
                  code_root: /workspace/interp/code/nano30b-nla-pilot-current
                  miles_root: /workspace/interp/code/miles-051cd15
                  model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
                  input_av_sft: {parquet}
                dataset:
                  row_limit: 8
                  split_mode: doc
                  fractions: {{train: 0.5, validation: 0.25, test: 0.25}}
                  materialize_splits: true
                  cache_materialized_splits: true
                  split_cache_dir: {split_cache_dir}
                  verify_materialized_splits: true
                  final_batch_policy: pad_with_train_duplicates
                  expected_d_model: 4
                training:
                  backend: miles_fsdp2
                  epochs: 1
                  global_batch_size: 3
                  micro_batch_size: 1
                  rollout_batch_size: 3
                  lr: 1e-5
                  injection_scale: 75
                  grad_norm_policy: skip_diagnostic
                  timing_debug: true
                checkpoint:
                  save_interval: 10
                  keep_last: 3
                  save_enabled: true
                  require_optimizer_state_for_hero: false
                eval:
                  controls: [real, shuffled, zero, mean, none]
                """
            )
            original_materialize = runner.materialize_splits
            calls: list[pathlib.Path] = []

            def counted_materialize(*args, **kwargs):
                calls.append(pathlib.Path(args[1]))
                return original_materialize(*args, **kwargs)

            runner.materialize_splits = counted_materialize
            try:
                spec = runner.load_and_validate_spec(spec_path)
                first = runner.prepare_run(spec, run_id="unit-cache-a")
                second = runner.prepare_run(spec, run_id="unit-cache-b")
            finally:
                runner.materialize_splits = original_materialize

            second_split_dir = output_root / "unit-cache-b" / "splits"
            self.assertEqual(len(calls), 1)
            self.assertEqual(first["split_manifest"]["split_cache"]["hit"], False)
            self.assertEqual(second["split_manifest"]["split_cache"]["hit"], True)
            self.assertEqual(second["split_manifest"]["train"]["path"], str(second_split_dir / "train.parquet"))
            self.assertEqual(second["split_manifest"]["train"]["padded_path"], str(second_split_dir / "train_padded.parquet"))
            self.assertTrue((second_split_dir / "split_manifest.json").exists())
            self.assertTrue((second_split_dir / "split_content_verify.json").exists())
            self.assertEqual(second["split_manifest"]["content_verification"]["doc_overlap_count"], 0)

    def test_prepare_run_fails_when_materialized_split_content_verification_finds_overlap(self):
        runner = load_script("nano_av_runner")
        docs = ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e", "doc-f"]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = self._write_tiny_av_dataset(root, docs=docs, repeated_response=True)
            output_root = root / "runs"
            spec_path = self._write_spec(
                f"""
                run:
                  name: bad-split-content
                  experiment_class: medium-small
                  output_root: {output_root}
                  wandb_mode: offline
                paths:
                  code_root: /workspace/interp/code/nano30b-nla-pilot-current
                  miles_root: /workspace/interp/code/miles-051cd15
                  model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
                  input_av_sft: {parquet}
                dataset:
                  row_limit: 6
                  split_mode: doc
                  fractions: {{train: 0.5, validation: 0.25, test: 0.25}}
                  materialize_splits: true
                  verify_materialized_splits: true
                  final_batch_policy: pad_with_train_duplicates
                  expected_d_model: 4
                training:
                  backend: miles_fsdp2
                  epochs: 1
                  global_batch_size: 3
                  micro_batch_size: 1
                  rollout_batch_size: 3
                  lr: 1e-5
                  injection_scale: 75
                  grad_norm_policy: skip_diagnostic
                  timing_debug: true
                checkpoint:
                  save_interval: 10
                  keep_last: 3
                  save_enabled: true
                  require_optimizer_state_for_hero: false
                eval:
                  controls: [real, shuffled, zero, mean, none]
                """
            )

            spec = runner.load_and_validate_spec(spec_path)
            with self.assertRaises(ValueError) as caught:
                runner.prepare_run(spec, run_id="bad-split-run")

            self.assertIn("content-hash cross-split overlap", str(caught.exception))

    def test_prepare_run_exports_optional_sft_token_caps(self):
        runner = load_script("nano_av_runner")
        docs = ["doc-a", "doc-a", "doc-b", "doc-b"]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "av_sft.parquet"
            output_root = root / "runs"
            table = pa.table(
                {
                    "prompt": pa.array([f"prompt {i} <concept><INJECT></concept>" for i in range(len(docs))]),
                    "response": pa.array([f"<explanation>row {i}</explanation>" for i in range(len(docs))]),
                    "activation_vector": pa.array(
                        [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(docs))],
                        type=pa.list_(pa.float32(), 4),
                    ),
                    "doc_id": pa.array(docs),
                }
            )
            pq.write_table(table, parquet)
            parquet.with_name(parquet.name + ".nla_meta.yaml").write_text(
                yaml.safe_dump(
                    {
                        "kind": "nla_dataset",
                        "schema_version": 1,
                        "dataset_id": "tiny-av",
                        "stage": "av_sft",
                        "row_count": len(docs),
                        "extraction": {"d_model": 4, "layer_index": 33},
                    },
                    sort_keys=False,
                )
            )
            spec_path = self._write_spec(
                f"""
                run:
                  name: capped-medium
                  experiment_class: medium-small
                  output_root: {output_root}
                  wandb_mode: offline
                paths:
                  code_root: /workspace/interp/code/nano30b-nla-pilot-current
                  miles_root: /workspace/interp/code/miles-051cd15
                  model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
                  input_av_sft: {parquet}
                dataset:
                  row_limit: 4
                  split_mode: doc
                  fractions: {{train: 0.5, validation: 0.25, test: 0.25}}
                  materialize_splits: true
                  final_batch_policy: pad_with_train_duplicates
                  expected_d_model: 4
                training:
                  backend: miles_fsdp2
                  epochs: 1
                  global_batch_size: 2
                  micro_batch_size: 1
                  rollout_batch_size: 2
                  lr: 1e-5
                  injection_scale: 75
                  grad_norm_policy: clip
                  max_sequence_tokens: 1152
                  max_response_tokens: 1024
                  pytorch_cuda_alloc_conf: expandable_segments:True
                  system_metrics:
                    enabled: true
                    interval_steps: 2
                    nvidia_smi_interval_steps: 8
                checkpoint:
                  save_interval: 1
                  keep_last: 1
                  save_enabled: true
                  require_optimizer_state_for_hero: false
                eval:
                  controls: [real, shuffled, zero, mean, none]
                """
            )

            spec = runner.load_and_validate_spec(spec_path)
            plan = runner.prepare_run(spec, run_id="capped-run")

            self.assertEqual(plan["environment"]["NLA_SFT_MAX_SEQUENCE_TOKENS"], "1152")
            self.assertEqual(plan["environment"]["NLA_SFT_MAX_RESPONSE_TOKENS"], "1024")
            self.assertEqual(plan["environment"]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
            self.assertEqual(plan["environment"]["NLA_SYSTEM_METRICS"], "1")
            self.assertEqual(plan["environment"]["NLA_SYSTEM_METRICS_INTERVAL_STEPS"], "2")
            self.assertEqual(plan["environment"]["NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS"], "8")

    def test_system_metrics_rejects_non_positive_interval(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: bad-system-metrics
              experiment_class: medium-small
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 4
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
              system_metrics:
                enabled: true
                interval_steps: 0
            checkpoint:
              save_interval: 10
              keep_last: 1
              save_enabled: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.load_and_validate_spec(spec_path)

        self.assertIn("training.system_metrics.interval_steps must be positive", str(caught.exception))

    def test_moe_routing_implementation_is_config_driven(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: expert-scan-routing
              experiment_class: tuning-probe
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: false
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 4
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
              moe_routing_impl: expert_scan
            checkpoint:
              save_interval: 10
              keep_last: 1
              save_enabled: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)
        plan = runner.prepare_run(spec, run_id="expert-scan-routing")

        self.assertEqual(plan["environment"]["NLA_MOE_ROUTING_IMPL"], "expert_scan")

    def test_moe_routing_implementation_rejects_unknown_value(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: invalid-moe-routing
              experiment_class: tuning-probe
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 4
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
              moe_routing_impl: mystery
            checkpoint:
              save_interval: 10
              keep_last: 1
              save_enabled: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.load_and_validate_spec(spec_path)

        self.assertIn("training.moe_routing_impl", str(caught.exception))

    def test_cuda_launch_blocking_is_config_driven(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: cuda-blocking-diagnostic
              experiment_class: tuning-probe
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: false
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 4
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
              cuda_launch_blocking: true
              lr_decay_iters: 1289
            checkpoint:
              save_interval: 10
              keep_last: 1
              save_enabled: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)
        plan = runner.prepare_run(spec, run_id="cuda-blocking-diagnostic")

        self.assertEqual(plan["environment"]["CUDA_LAUNCH_BLOCKING"], "1")
        decay_idx = plan["command"].index("--lr-decay-iters")
        self.assertEqual(plan["command"][decay_idx + 1], "1289")

    def test_configured_environment_is_forwarded_to_ray_workers_without_host_env(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: ray-worker-env
              experiment_class: tuning-probe
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: false
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 4
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
              max_sequence_tokens: 1152
              max_response_tokens: 1024
              pytorch_cuda_alloc_conf: expandable_segments:True
              moe_routing_impl: expert_scan
              mamba_kernel_mode: unfused_torch_conv
              cuda_launch_blocking: true
              assert_actor_packed_equivalence: true
              actor_packed_equivalence_rtol: 0.02
              actor_packed_equivalence_atol: 0.05
              system_metrics:
                enabled: true
                interval_steps: 2
                nvidia_smi_interval_steps: 4
                router_entropy: true
            checkpoint:
              save_interval: 10
              keep_last: 2
              save_enabled: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)
        previous = os.environ.get("NLA_HOST_ONLY_SENTINEL")
        os.environ["NLA_HOST_ONLY_SENTINEL"] = "must-not-leak"
        try:
            plan = runner.prepare_run(spec, run_id="ray-worker-env")
        finally:
            if previous is None:
                os.environ.pop("NLA_HOST_ONLY_SENTINEL", None)
            else:
                os.environ["NLA_HOST_ONLY_SENTINEL"] = previous

        command = plan["command"]
        train_env_index = command.index("--train-env-vars")
        worker_environment = json.loads(command[train_env_index + 1])
        self.assertEqual(worker_environment, plan["environment"])
        self.assertNotIn("NLA_HOST_ONLY_SENTINEL", worker_environment)
        self.assertEqual(worker_environment["NLA_MOE_ROUTING_IMPL"], "expert_scan")
        self.assertEqual(
            worker_environment["NLA_TRAIN_MAMBA_KERNEL_MODE"],
            "unfused_torch_conv",
        )
        self.assertEqual(worker_environment["CUDA_LAUNCH_BLOCKING"], "1")
        self.assertEqual(worker_environment["NLA_ROUTER_METRICS"], "1")
        self.assertEqual(worker_environment["NLA_ASSERT_ACTOR_PACKED_EQUIV"], "1")
        self.assertEqual(worker_environment["NLA_ACTOR_PACKED_EQUIV_RTOL"], "0.02")
        self.assertEqual(worker_environment["NLA_ACTOR_PACKED_EQUIV_ATOL"], "0.05")

    def test_dynamic_batching_flags_are_rendered_from_yaml(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: dynamic-medium
              experiment_class: medium-small
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 4
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
              use_dynamic_batch_size: true
              max_tokens_per_gpu: 1024
            checkpoint:
              save_interval: 10
              keep_last: 1
              save_enabled: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)
        command = runner.render_miles_command(spec, "/tmp/splits/train_padded.parquet", "/tmp/run")

        self.assertIn("--use-dynamic-batch-size", command)
        self.assertIn("--max-tokens-per-gpu", command)
        self.assertEqual(command[command.index("--max-tokens-per-gpu") + 1], "1024")

    def test_dynamic_batching_requires_positive_token_budget(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: bad-dynamic-budget
              experiment_class: medium-small
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 4
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
              use_dynamic_batch_size: true
              max_tokens_per_gpu: 0
            checkpoint:
              save_interval: 10
              keep_last: 1
              save_enabled: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.load_and_validate_spec(spec_path)

        self.assertIn("training.max_tokens_per_gpu must be positive", str(caught.exception))

    def test_dynamic_batching_requires_single_sample_to_fit_token_budget(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: bad-dynamic-sequence-budget
              experiment_class: medium-small
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 4
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
              use_dynamic_batch_size: true
              max_tokens_per_gpu: 512
              max_sequence_tokens: 1152
            checkpoint:
              save_interval: 10
              keep_last: 1
              save_enabled: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.load_and_validate_spec(spec_path)

        self.assertIn(
            "training.max_tokens_per_gpu must be >= training.max_sequence_tokens",
            str(caught.exception),
        )

    def test_training_token_caps_must_be_positive(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: bad-caps
              experiment_class: medium-small
              output_root: /tmp/nano-out
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 960
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 96
              micro_batch_size: 8
              rollout_batch_size: 96
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
              max_sequence_tokens: 0
              max_response_tokens: -3
            checkpoint:
              save_interval: 10
              keep_last: 1
              save_enabled: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.load_and_validate_spec(spec_path)

        message = str(caught.exception)
        self.assertIn("training.max_sequence_tokens must be positive", message)
        self.assertIn("training.max_response_tokens must be positive", message)

    def test_sft_actor_truncates_response_and_sequence_tokens(self):
        try:
            __import__("torch")
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in the local test environment")
        mask_utils = types.ModuleType("miles.utils.mask_utils")
        mask_utils.MultiTurnLossMaskGenerator = object
        processing_utils = types.ModuleType("miles.utils.processing_utils")
        processing_utils.load_tokenizer = lambda *args, **kwargs: None
        miles_utils = types.ModuleType("miles.utils")
        miles = types.ModuleType("miles")
        nla_schema = types.ModuleType("nla.schema")
        nla_schema.MM_ACTIVATION_KEY = "activation"
        nla = types.ModuleType("nla")

        old_modules = {
            name: sys.modules.get(name)
            for name in (
                "miles",
                "miles.utils",
                "miles.utils.mask_utils",
                "miles.utils.processing_utils",
                "nla",
                "nla.schema",
            )
        }
        try:
            sys.modules.update(
                {
                    "miles": miles,
                    "miles.utils": miles_utils,
                    "miles.utils.mask_utils": mask_utils,
                    "miles.utils.processing_utils": processing_utils,
                    "nla": nla,
                    "nla.schema": nla_schema,
                }
            )
            path = ROOT / "external" / "natural_language_autoencoders" / "nla" / "rollout" / "sft_actor.py"
            spec = importlib.util.spec_from_file_location("unit_sft_actor", path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
        finally:
            for name, old in old_modules.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old

        token_ids = list(range(20))
        loss_mask = [0] * 5 + [1] * 15

        capped_ids, capped_mask, response_length = module._truncate_sft_tokens(
            token_ids,
            loss_mask,
            response_length=15,
            max_sequence_tokens=10,
            max_response_tokens=8,
        )

        self.assertEqual(capped_ids, list(range(10)))
        self.assertEqual(capped_mask, [0] * 5 + [1] * 5)
        self.assertEqual(response_length, 5)

    def test_resume_smoke_plan_uses_latest_checkpoint_plus_resume_steps(self):
        runner = load_script("nano_av_runner")
        docs = ["doc-a", "doc-a", "doc-b", "doc-b"]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "av_sft.parquet"
            output_root = root / "runs"
            checkpoint_root = root / "checkpoints"
            checkpoint_root.mkdir()
            (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("9")
            table = pa.table(
                {
                    "prompt": pa.array([f"prompt {i} <concept><INJECT></concept>" for i in range(len(docs))]),
                    "response": pa.array([f"<explanation>row {i}</explanation>" for i in range(len(docs))]),
                    "activation_vector": pa.array(
                        [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(docs))],
                        type=pa.list_(pa.float32(), 4),
                    ),
                    "doc_id": pa.array(docs),
                }
            )
            pq.write_table(table, parquet)
            parquet.with_name(parquet.name + ".nla_meta.yaml").write_text(
                yaml.safe_dump(
                    {
                        "kind": "nla_dataset",
                        "schema_version": 1,
                        "dataset_id": "tiny-av",
                        "stage": "av_sft",
                        "row_count": len(docs),
                        "extraction": {"d_model": 4, "layer_index": 27},
                    },
                    sort_keys=False,
                )
            )
            spec_path = self._write_spec(
                f"""
                run:
                  name: resume
                  experiment_class: medium-small
                  output_root: {output_root}
                  wandb_mode: offline
                paths:
                  code_root: /workspace/interp/code/nano30b-nla-pilot-current
                  miles_root: /workspace/interp/code/miles-051cd15
                  model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
                  input_av_sft: {parquet}
                dataset:
                  row_limit: 4
                  split_mode: doc
                  fractions: {{train: 0.5, validation: 0.25, test: 0.25}}
                  materialize_splits: true
                  final_batch_policy: pad_with_train_duplicates
                  expected_d_model: 4
                training:
                  backend: miles_fsdp2
                  epochs: 1
                  resume_steps: 1
                  global_batch_size: 2
                  micro_batch_size: 1
                  rollout_batch_size: 2
                  lr: 1e-5
                  injection_scale: 75
                  grad_norm_policy: skip_diagnostic
                checkpoint:
                  resume_from: {checkpoint_root}
                  save_interval: 1000
                  keep_last: 3
                  save_enabled: false
                  no_save_optim: true
                  require_optimizer_state_for_hero: false
                eval:
                  controls: [real, shuffled, zero, mean, none]
                """
            )

            spec = runner.load_and_validate_spec(spec_path)
            plan = runner.prepare_run(spec, run_id="resume-run")

            self.assertEqual(plan["resume_start_rollout"], 9)
            self.assertEqual(plan["num_rollout"], 10)
            self.assertIn("--load", plan["command"])
            self.assertIn(str(checkpoint_root), plan["command"])
            self.assertIn("--num-rollout", plan["command"])
            self.assertEqual(plan["command"][plan["command"].index("--num-rollout") + 1], "10")
            self.assertNotIn("--save", plan["command"])

    def test_complete_performance_resume_can_save_model_only_final_checkpoint(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: complete-resume-model-only-final
              experiment_class: complete-performance
              output_root: /tmp/nano
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 90000
              split_mode: doc
              fractions: {train: 0.9, validation: 0.05, test: 0.05}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              resume_steps: 896
              global_batch_size: 192
              micro_batch_size: 48
              rollout_batch_size: 192
              lr: 5e-5
              injection_scale: 75
              grad_norm_policy: clip
            checkpoint:
              resume_from: /tmp/checkpoints
              finetune: false
              resume_optimizer_state_required: true
              save_interval: 1289
              keep_last: 1
              save_enabled: true
              no_save_optim: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)

        self.assertTrue(spec["checkpoint"]["resume_optimizer_state_required"])

    def test_required_resume_optimizer_state_checks_dcp_metadata(self):
        runner = load_script("nano_av_runner")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = pathlib.Path(tmp)
            iteration_dir = checkpoint_root / "iter_0000009"
            for component in ("optimizer", "lr_scheduler"):
                component_dir = iteration_dir / component
                component_dir.mkdir(parents=True)
                (component_dir / ".metadata").write_text("metadata")

            runner._verify_resume_optimizer_state(checkpoint_root, 9)

            (iteration_dir / "optimizer" / ".metadata").unlink()
            with self.assertRaises(runner.SpecValidationError) as caught:
                runner._verify_resume_optimizer_state(checkpoint_root, 9)

        self.assertIn("optimizer/.metadata", str(caught.exception))

    def test_finetune_resume_steps_are_fresh_rollout_budget(self):
        runner = load_script("nano_av_runner")
        docs = ["doc-a", "doc-a", "doc-b", "doc-b"]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "av_sft.parquet"
            output_root = root / "runs"
            checkpoint_root = root / "checkpoints"
            checkpoint_root.mkdir()
            (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("9")
            table = pa.table(
                {
                    "prompt": pa.array([f"prompt {i} <concept><INJECT></concept>" for i in range(len(docs))]),
                    "response": pa.array([f"<explanation>row {i}</explanation>" for i in range(len(docs))]),
                    "activation_vector": pa.array(
                        [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(docs))],
                        type=pa.list_(pa.float32(), 4),
                    ),
                    "doc_id": pa.array(docs),
                }
            )
            pq.write_table(table, parquet)
            parquet.with_name(parquet.name + ".nla_meta.yaml").write_text(
                yaml.safe_dump(
                    {
                        "kind": "nla_dataset",
                        "schema_version": 1,
                        "dataset_id": "tiny-av",
                        "stage": "av_sft",
                        "row_count": len(docs),
                        "extraction": {"d_model": 4, "layer_index": 27},
                    },
                    sort_keys=False,
                )
            )
            spec_path = self._write_spec(
                f"""
                run:
                  name: finetune-resume
                  experiment_class: medium-small
                  output_root: {output_root}
                  wandb_mode: offline
                paths:
                  code_root: /workspace/interp/code/nano30b-nla-pilot-current
                  miles_root: /workspace/interp/code/miles-051cd15
                  model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
                  input_av_sft: {parquet}
                dataset:
                  row_limit: 4
                  split_mode: doc
                  fractions: {{train: 0.5, validation: 0.25, test: 0.25}}
                  materialize_splits: true
                  final_batch_policy: pad_with_train_duplicates
                  expected_d_model: 4
                training:
                  backend: miles_fsdp2
                  epochs: 1
                  resume_steps: 1
                  global_batch_size: 2
                  micro_batch_size: 1
                  rollout_batch_size: 2
                  lr: 1e-5
                  injection_scale: 75
                  grad_norm_policy: skip_diagnostic
                checkpoint:
                  resume_from: {checkpoint_root}
                  finetune: true
                  save_interval: 1
                  keep_last: 1
                  save_enabled: true
                  no_save_optim: true
                  require_optimizer_state_for_hero: false
                eval:
                  controls: [real, shuffled, zero, mean, none]
                """
            )

            spec = runner.load_and_validate_spec(spec_path)
            plan = runner.prepare_run(spec, run_id="finetune-resume-run")

            self.assertEqual(plan["resume_start_rollout"], 9)
            self.assertEqual(plan["num_rollout"], 1)
            self.assertIn("--load", plan["command"])
            self.assertIn("--finetune", plan["command"])
            self.assertIn("--num-rollout", plan["command"])
            self.assertEqual(plan["command"][plan["command"].index("--num-rollout") + 1], "1")

    def test_resume_steps_must_be_positive(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: bad-resume
              experiment_class: medium-small
              output_root: /tmp/nano
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 96
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              resume_steps: 0
              global_batch_size: 8
              micro_batch_size: 1
              rollout_batch_size: 8
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: skip_diagnostic
            checkpoint:
              resume_from: /tmp/checkpoints
              save_interval: 1000
              keep_last: 3
              save_enabled: false
              no_save_optim: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.load_and_validate_spec(spec_path)

        self.assertIn("training.resume_steps must be positive", str(caught.exception))

    def test_checkpoint_finetune_renders_miles_flag(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: finetune
              experiment_class: medium-small
              output_root: /tmp/nano
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 96
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 8
              micro_batch_size: 1
              rollout_batch_size: 8
              lr: 5e-6
              injection_scale: 75
              grad_norm_policy: clip
            checkpoint:
              resume_from: /tmp/checkpoints
              finetune: true
              save_interval: 10
              keep_last: 1
              save_enabled: true
              no_save_optim: false
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)
        command = runner.render_miles_command(spec, "/tmp/train.parquet", "/tmp/run")

        self.assertIn("--load", command)
        self.assertIn("--finetune", command)

    def test_checkpoint_finetune_requires_resume_from(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: bad-finetune
              experiment_class: medium-small
              output_root: /tmp/nano
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 96
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 8
              micro_batch_size: 1
              rollout_batch_size: 8
              lr: 5e-6
              injection_scale: 75
              grad_norm_policy: clip
            checkpoint:
              finetune: true
              save_interval: 10
              keep_last: 1
              save_enabled: true
              no_save_optim: false
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.load_and_validate_spec(spec_path)

        self.assertIn("checkpoint.finetune requires checkpoint.resume_from", str(caught.exception))

    def test_complete_performance_rejects_timing_debug(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: bad-hero-timing
              experiment_class: complete-performance
              output_root: /tmp/nano
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 90000
              split_mode: doc
              fractions: {train: 0.9, validation: 0.05, test: 0.05}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 192
              micro_batch_size: 8
              rollout_batch_size: 192
              lr: 1e-5
              injection_scale: 75
              grad_norm_policy: clip
              timing_debug: true
            checkpoint:
              save_interval: 100
              keep_last: 1
              save_enabled: true
              no_save_optim: false
              require_optimizer_state_for_hero: true
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.load_and_validate_spec(spec_path)

        self.assertIn("complete-performance cannot enable training.timing_debug", str(caught.exception))

    def test_finetune_resume_steps_require_final_save_interval(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: bad-finetune-save
              experiment_class: medium-small
              output_root: /tmp/nano
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 96
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              resume_steps: 256
              global_batch_size: 8
              micro_batch_size: 1
              rollout_batch_size: 8
              lr: 5e-6
              injection_scale: 75
              grad_norm_policy: clip
            checkpoint:
              resume_from: /tmp/checkpoints
              finetune: true
              save_interval: 1547
              keep_last: 1
              save_enabled: true
              no_save_optim: true
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        with self.assertRaises(runner.SpecValidationError) as caught:
            runner.load_and_validate_spec(spec_path)

        self.assertIn("finetune resume probes must save on the final resumed step", str(caught.exception))

    def test_training_schedule_knobs_render_miles_lr_flags(self):
        runner = load_script("nano_av_runner")
        spec_path = self._write_spec(
            """
            run:
              name: schedule
              experiment_class: medium-small
              output_root: /tmp/nano
              wandb_mode: offline
            paths:
              code_root: /workspace/interp/code/nano30b-nla-pilot-current
              miles_root: /workspace/interp/code/miles-051cd15
              model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
              input_av_sft: /tmp/av_sft.parquet
            dataset:
              row_limit: 96
              split_mode: doc
              fractions: {train: 0.8, validation: 0.1, test: 0.1}
              materialize_splits: true
              final_batch_policy: pad_with_train_duplicates
              expected_d_model: 2688
            training:
              backend: miles_fsdp2
              epochs: 1
              global_batch_size: 8
              micro_batch_size: 1
              rollout_batch_size: 8
              lr: 2e-5
              min_lr: 2e-6
              lr_decay_style: cosine
              lr_warmup_iters: 25
              lr_warmup_init: 0.0
              rollout_seed: 314159
              distributed_timeout_minutes: 60
              injection_scale: 75
              grad_norm_policy: clip
            checkpoint:
              save_interval: 10
              keep_last: 1
              save_enabled: true
              no_save_optim: false
              require_optimizer_state_for_hero: false
            eval:
              controls: [real, shuffled, zero, mean, none]
            """
        )

        spec = runner.load_and_validate_spec(spec_path)
        command = runner.render_miles_command(spec, "/tmp/train.parquet", "/tmp/run")

        self.assertEqual(command[command.index("--lr") + 1], "2e-5")
        self.assertEqual(command[command.index("--min-lr") + 1], "2e-6")
        self.assertEqual(command[command.index("--lr-decay-style") + 1], "cosine")
        self.assertEqual(command[command.index("--lr-warmup-iters") + 1], "25")
        self.assertEqual(command[command.index("--lr-warmup-init") + 1], "0.0")
        self.assertEqual(command[command.index("--rollout-seed") + 1], "314159")
        self.assertEqual(command[command.index("--distributed-timeout-minutes") + 1], "60")


if __name__ == "__main__":
    unittest.main()
