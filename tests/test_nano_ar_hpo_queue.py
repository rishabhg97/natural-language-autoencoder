import importlib.util
import pathlib
import tempfile
import textwrap
import unittest
from unittest import mock

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoARHPOQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _write_queue(self, text):
        path = self.root / "queue.yaml"
        path.write_text(textwrap.dedent(text))
        return path

    def _write_ar_config(self):
        code_root = self.root / "code"
        config_dir = code_root / "configs" / "nano_ar" / "hpo"
        config_dir.mkdir(parents=True)
        split_dir = self.root / "splits"
        split_dir.mkdir()
        checkpoint_root = self.root / "base" / "checkpoints"
        checkpoint_root.mkdir(parents=True)
        (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("1547")
        config_path = config_dir / "trial.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""
                run:
                  name: unit-trial
                  experiment_class: tuning-probe
                  output_root: {self.root / "runs"}
                  wandb_mode: offline
                paths:
                  code_root: {code_root}
                  miles_root: /workspace/interp/code/miles-051cd15
                  model_id: /workspace/interp/models/nano-30b-a3b-bf16-hf
                  critic_init_model_id: /workspace/interp/outputs/nano30b-nla-pilot/miles-fsdp2-ar-sft-critic-init/nano-ar-r27-critic-init
                  input_ar_sft: {split_dir / "train_padded.parquet"}
                dataset:
                  row_limit: 192
                  split_mode: doc
                  fractions: {{train: 0.9, validation: 0.05, test: 0.05}}
                  materialize_splits: false
                  final_batch_policy: pad_with_train_duplicates
                  expected_d_model: 2688
                training:
                  objective: ar_sft
                  backend: miles_fsdp2
                  epochs: 1
                  resume_steps: 128
                  global_batch_size: 192
                  micro_batch_size: 8
                  rollout_batch_size: 192
                  lr: 2e-5
                  min_lr: 2e-6
                  lr_decay_style: cosine
                  lr_warmup_iters: 25
                  grad_norm_policy: clip
                  timing_debug: true
                  allow_packed_critic_training: true
                checkpoint:
                  resume_from: {checkpoint_root}
                  finetune: true
                  save_interval: 128
                  keep_last: 1
                  save_enabled: true
                  no_save_optim: true
                  require_optimizer_state_for_hero: false
                eval:
                  controls: [teacher, teacher_shuffled, blank, generic, mean, source_context, source_raw]
                  validation_limit: 512
                  test_limit: 512
                """
            )
        )
        return code_root, config_path

    def test_load_queue_rejects_long_eval_limits(self):
        queue = load_script("nano_ar_hpo_queue")
        queue_path = self._write_queue(
            """
            schema_version: nano_ar_hpo_queue.v1
            defaults:
              validation_limit: 2048
              test_limit: 512
            items:
              - name: too-long
                config: configs/nano_ar/hpo/example.yaml
                status: pending
            """
        )

        with self.assertRaisesRegex(queue.QueueError, "validation_limit"):
            queue.load_queue(queue_path)

    def test_next_pending_skips_complete_and_running_items(self):
        queue = load_script("nano_ar_hpo_queue")
        queue_path = self.root / "queue.yaml"
        loaded = queue.validate_queue(
            {
                "schema_version": "nano_ar_hpo_queue.v1",
                "defaults": {"validation_limit": 512, "test_limit": 512},
                "items": [
                    {"name": "done", "config": "a.yaml", "status": "complete"},
                    {"name": "active", "config": "b.yaml", "status": "training"},
                    {"name": "next", "config": "c.yaml", "status": "pending"},
                ],
            },
            source=queue_path,
        )

        self.assertEqual(queue.next_pending_index(loaded), 2)

    def test_checked_in_r33_100k_queue_is_valid(self):
        queue = load_script("nano_ar_hpo_queue")
        queue_path = ROOT / "configs" / "nano_ar" / "hpo" / "r33_100k_scaling_queue.yaml"

        loaded = queue.load_queue(queue_path)

        self.assertIsNone(queue.next_pending_index(loaded))
        self.assertEqual(len(loaded["items"]), 16)
        self.assertTrue(all(item["status"] != "pending" for item in loaded["items"]))

    def test_update_item_status_writes_yaml_with_artifacts(self):
        queue = load_script("nano_ar_hpo_queue")
        queue_path = self._write_queue(
            """
            schema_version: nano_ar_hpo_queue.v1
            defaults:
              validation_limit: 512
              test_limit: 512
            items:
              - name: trial-a
                config: configs/nano_ar/hpo/trial-a.yaml
                status: pending
            """
        )

        updated = queue.update_item(queue_path, 0, status="training", run_dir="/runs/a")
        reloaded = yaml.safe_load(queue_path.read_text())

        self.assertEqual(updated["status"], "training")
        self.assertEqual(reloaded["items"][0]["status"], "training")
        self.assertEqual(reloaded["items"][0]["run_dir"], "/runs/a")

    def test_process_next_records_checkpoint_dir_for_completed_item(self):
        queue = load_script("nano_ar_hpo_queue")
        code_root = self.root / "code"
        code_root.mkdir()
        config_path = code_root / "trial.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                run:
                  name: unit-trial
                  output_root: /unused
                paths:
                  code_root: /unused
                training:
                  lr_decay_style: cosine
                eval:
                  controls: [teacher]
                """
            )
        )
        queue_path = self._write_queue(
            f"""
            schema_version: nano_ar_hpo_queue.v1
            defaults:
              validation_limit: 8
              test_limit: 8
              batch_size: 2
            items:
              - name: trial-a
                config: {config_path}
                status: pending
            """
        )
        run_dir = self.root / "run"
        checkpoint_dir = run_dir / "checkpoints" / "iter_0000003"
        checkpoint_dir.mkdir(parents=True)
        train_parquet = self.root / "splits" / "train.parquet"
        validation_parquet = self.root / "splits" / "validation.parquet"
        test_parquet = self.root / "splits" / "test.parquet"
        train_parquet.parent.mkdir()
        for parquet in (train_parquet, validation_parquet, test_parquet):
            parquet.write_text("placeholder")
        plan = {
            "run_dir": str(run_dir),
            "num_rollout": 3,
            "spec": {
                "paths": {"code_root": str(code_root)},
                "training": {},
                "checkpoint": {},
                "eval": {"controls": ["teacher"]},
            },
            "train_parquet": str(train_parquet),
            "split_manifest": {
                "validation": {"path": str(validation_parquet)},
                "test": {"path": str(test_parquet)},
            },
            "command": ["python", "train.py"],
            "environment": {},
        }

        with mock.patch.object(queue.nano_av_runner, "load_and_validate_spec", return_value=plan["spec"]), \
             mock.patch.object(queue.nano_av_runner, "prepare_run", return_value=plan), \
             mock.patch.object(queue.nano_ar_hpo_study, "assert_lr_decay_canary_for_run"), \
             mock.patch.object(queue, "_run_logged"):
            result = queue.process_next_item(queue_path)

        reloaded = yaml.safe_load(queue_path.read_text())
        item = reloaded["items"][0]
        self.assertEqual(result["status"], "complete")
        self.assertEqual(item["status"], "complete")
        self.assertEqual(item["expected_checkpoint"], str(checkpoint_dir))
        self.assertEqual(item["checkpoint_dir"], str(checkpoint_dir))
        self.assertIn("eval_report", item)

    def test_eval_command_uses_512_limits_and_standard_controls(self):
        queue = load_script("nano_ar_hpo_queue")

        command = queue.build_eval_command(
            python_bin="/venv/bin/python",
            checkpoint_dir=pathlib.Path("/run/checkpoints/iter_0000128"),
            train_parquet=pathlib.Path("/splits/train_padded.parquet"),
            validation_parquet=pathlib.Path("/splits/validation.parquet"),
            test_parquet=pathlib.Path("/splits/test.parquet"),
            report_json=pathlib.Path("/run/eval.json"),
            validation_limit=512,
            test_limit=512,
            eval_splits=["validation"],
            batch_size=4,
            controls=[
                "teacher",
                "teacher_shuffled",
                "blank",
                "generic",
                "mean",
                "source_context",
                "source_raw",
            ],
        )

        self.assertEqual(command[:2], ["/venv/bin/python", "scripts/eval_nano_ar_miles_checkpoint.py"])
        self.assertIn("--validation-limit", command)
        self.assertEqual(command[command.index("--validation-limit") + 1], "512")
        self.assertEqual(command[command.index("--test-limit") + 1], "512")
        self.assertEqual(
            command[command.index("--eval-splits") + 1],
            "validation",
        )
        self.assertIn("--controls", command)
        self.assertIn("source_raw", command)

    def test_queue_lock_rejects_second_holder(self):
        queue = load_script("nano_ar_hpo_queue")
        queue_path = self._write_queue(
            """
            schema_version: nano_ar_hpo_queue.v1
            defaults:
              validation_limit: 512
              test_limit: 512
            items:
              - name: trial-a
                config: configs/nano_ar/hpo/trial-a.yaml
                status: pending
            """
        )

        with queue.queue_lock(queue_path):
            with self.assertRaisesRegex(queue.QueueError, "already active"):
                with queue.queue_lock(queue_path):
                    pass

    def test_reset_active_items_requeues_stale_training_and_eval_items(self):
        queue = load_script("nano_ar_hpo_queue")
        queue_path = self._write_queue(
            """
            schema_version: nano_ar_hpo_queue.v1
            defaults:
              validation_limit: 512
              test_limit: 512
            items:
              - name: stale-train
                config: configs/nano_ar/hpo/stale-train.yaml
                status: training
              - name: stale-eval
                config: configs/nano_ar/hpo/stale-eval.yaml
                status: eval_running
              - name: done
                config: configs/nano_ar/hpo/done.yaml
                status: complete
            """
        )

        result = queue.reset_active_items(queue_path, reason="unit stale reset")
        reloaded = yaml.safe_load(queue_path.read_text())

        self.assertEqual(result["reset_count"], 2)
        self.assertEqual(reloaded["items"][0]["status"], "pending")
        self.assertEqual(reloaded["items"][0]["previous_status"], "training")
        self.assertEqual(reloaded["items"][0]["reset_reason"], "unit stale reset")
        self.assertEqual(reloaded["items"][1]["status"], "pending")
        self.assertEqual(reloaded["items"][1]["previous_status"], "eval_running")
        self.assertEqual(reloaded["items"][2]["status"], "complete")

    def test_watch_queue_run_until_empty_stops_after_idle(self):
        queue = load_script("nano_ar_hpo_queue")
        queue_path = self._write_queue(
            """
            schema_version: nano_ar_hpo_queue.v1
            defaults:
              validation_limit: 512
              test_limit: 512
            items:
              - name: trial-a
                config: configs/nano_ar/hpo/trial-a.yaml
                status: pending
            """
        )

        with mock.patch.object(
            queue,
            "process_next_item",
            side_effect=[
                {"status": "complete", "item_name": "trial-a"},
                {"status": "idle"},
            ],
        ) as process_next:
            result = queue.watch_queue(
                queue_path,
                poll_seconds=0,
                dry_run=False,
                once=False,
                stop_when_idle=True,
            )

        self.assertEqual(result, 0)
        self.assertEqual(process_next.call_count, 2)

    def test_watch_queue_stops_on_failed_item_by_default(self):
        queue = load_script("nano_ar_hpo_queue")
        queue_path = self._write_queue(
            """
            schema_version: nano_ar_hpo_queue.v1
            defaults:
              validation_limit: 512
              test_limit: 512
            items:
              - name: trial-a
                config: configs/nano_ar/hpo/trial-a.yaml
                status: pending
              - name: trial-b
                config: configs/nano_ar/hpo/trial-b.yaml
                status: pending
            """
        )

        with mock.patch.object(queue, "process_next_item", return_value={"status": "failed", "item_name": "trial-a"}) as process_next:
            result = queue.watch_queue(
                queue_path,
                poll_seconds=0,
                dry_run=False,
                once=False,
                stop_when_idle=True,
            )

        self.assertEqual(result, 1)
        self.assertEqual(process_next.call_count, 1)

    def test_process_next_item_dry_run_prepares_train_and_eval_commands(self):
        queue = load_script("nano_ar_hpo_queue")
        code_root, config_path = self._write_ar_config()
        queue_path = self._write_queue(
            f"""
            schema_version: nano_ar_hpo_queue.v1
            defaults:
              code_root: {code_root}
              python: /venv/bin/python
              validation_limit: 512
              test_limit: 512
              batch_size: 4
              controls: [teacher, teacher_shuffled, blank, generic, mean, source_context, source_raw]
            items:
              - name: unit-trial
                config: {config_path.relative_to(code_root)}
                run_id: unit-trial-run
                status: pending
            """
        )

        result = queue.process_next_item(queue_path, dry_run=True)

        self.assertEqual(result["status"], "dry_run")
        self.assertIn("train_command", result)
        self.assertIn("eval_command", result)
        self.assertIn("--num-rollout", result["train_command"])
        self.assertEqual(result["train_command"][result["train_command"].index("--num-rollout") + 1], "128")
        self.assertEqual(result["expected_checkpoint"].name, "iter_0000128")
        self.assertEqual(result["eval_command"][result["eval_command"].index("--validation-limit") + 1], "512")
        self.assertIn(str(self.root / "splits" / "validation.parquet"), result["eval_command"])

    def test_independent_r33_critic_preserves_split_and_changes_training_order(self):
        config_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_seed314159.yaml"
        )
        queue_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_queue.yaml"
        )

        config = yaml.safe_load(config_path.read_text())
        queue = yaml.safe_load(queue_path.read_text())

        self.assertEqual(config["dataset"]["seed"], 42)
        self.assertEqual(config["dataset"]["split_mode"], "content_component")
        self.assertEqual(config["dataset"]["row_limit"], 275396)
        self.assertEqual(config["training"]["rollout_seed"], 314159)
        self.assertEqual(config["training"]["lr"], "5e-5")
        self.assertEqual(config["training"]["global_batch_size"], 192)
        self.assertEqual(config["training"]["micro_batch_size"], 96)
        self.assertEqual(config["checkpoint"]["save_interval"], 1289)
        self.assertTrue(config["checkpoint"]["no_save_optim"])
        self.assertEqual(config["eval"]["validation_limit"], 512)
        self.assertEqual(config["eval"]["test_limit"], 512)
        self.assertEqual(
            config["paths"]["code_root"],
            "/workspace/interp/code/nano30b-nla-pilot-hero-current",
        )
        self.assertEqual(len(queue["items"]), 1)
        self.assertEqual(queue["items"][0]["status"], "pending")
        self.assertEqual(
            queue["defaults"]["code_root"],
            "/workspace/interp/code/nano30b-nla-pilot-hero-current",
        )

    def test_independent_r33_critic_h100_retry_uses_four_way_exact_batch(self):
        config_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_seed314159_4gpu_mb48.yaml"
        )
        queue_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_retry_4gpu_queue.yaml"
        )

        config = yaml.safe_load(config_path.read_text())
        queue = yaml.safe_load(queue_path.read_text())

        self.assertEqual(config["training"]["num_gpus"], 4)
        self.assertEqual(config["training"]["global_batch_size"], 192)
        self.assertEqual(config["training"]["micro_batch_size"], 48)
        self.assertEqual(config["training"]["rollout_batch_size"], 192)
        self.assertEqual(config["training"]["rollout_seed"], 314159)
        self.assertEqual(config["training"]["lr"], "5e-5")
        self.assertEqual(config["dataset"]["seed"], 42)
        self.assertEqual(len(queue["items"]), 1)
        self.assertEqual(queue["items"][0]["status"], "pending")
        self.assertIn("mb48-4gpu", queue["items"][0]["run_id"])

    def test_independent_r33_critic_disables_optional_router_hook_after_cuda_fault(self):
        config_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_seed314159_4gpu_mb48_norouter.yaml"
        )
        queue_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_retry2_norouter_queue.yaml"
        )

        config = yaml.safe_load(config_path.read_text())
        queue = yaml.safe_load(queue_path.read_text())

        self.assertEqual(config["training"]["num_gpus"], 4)
        self.assertEqual(config["training"]["micro_batch_size"], 48)
        self.assertEqual(config["training"]["global_batch_size"], 192)
        self.assertFalse(config["training"]["system_metrics"]["router_entropy"])
        self.assertEqual(queue["items"][0]["status"], "pending")
        self.assertIn("norouter", queue["items"][0]["run_id"])

    def test_independent_r33_critic_can_use_expert_scan_moe_fallback(self):
        config_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_seed314159_4gpu_mb48_expertscan.yaml"
        )
        queue_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_retry3_expertscan_queue.yaml"
        )

        config = yaml.safe_load(config_path.read_text())
        queue = yaml.safe_load(queue_path.read_text())

        self.assertEqual(config["training"]["num_gpus"], 4)
        self.assertEqual(config["training"]["micro_batch_size"], 48)
        self.assertEqual(config["training"]["global_batch_size"], 192)
        self.assertEqual(config["training"]["moe_routing_impl"], "expert_scan")
        self.assertFalse(config["training"]["system_metrics"]["router_entropy"])
        self.assertEqual(queue["items"][0]["status"], "pending")
        self.assertIn("expertscan", queue["items"][0]["run_id"])

    def test_independent_r33_critic_cuda_blocking_diagnostic_saves_restart_state(self):
        config_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_seed314159_4gpu_mb48_cudablock_diag393.yaml"
        )
        queue_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_retry4_cudablock_diag_queue.yaml"
        )

        config = yaml.safe_load(config_path.read_text())
        queue = yaml.safe_load(queue_path.read_text())

        self.assertEqual(config["training"]["num_rollout"], 393)
        self.assertEqual(config["training"]["lr_decay_iters"], 1289)
        self.assertTrue(config["training"]["cuda_launch_blocking"])
        self.assertEqual(config["training"]["moe_routing_impl"], "expert_scan")
        self.assertEqual(config["checkpoint"]["save_interval"], 384)
        self.assertFalse(config["checkpoint"]["no_save_optim"])
        self.assertIn("cudablock", queue["items"][0]["run_id"])

    def test_independent_r33_critic_cuda_blocking_continuation_preserves_schedule(self):
        config_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_seed314159_4gpu_mb48_cudablock_resume393.yaml"
        )
        queue_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "hpo"
            / "r33_component_full_independent_critic_resume393_cudablock_queue.yaml"
        )

        config = yaml.safe_load(config_path.read_text())
        queue = yaml.safe_load(queue_path.read_text())

        self.assertNotIn("num_rollout", config["training"])
        self.assertEqual(config["training"]["resume_steps"], 896)
        self.assertEqual(config["training"]["lr_decay_iters"], 1289)
        self.assertTrue(config["training"]["cuda_launch_blocking"])
        self.assertEqual(config["training"]["moe_routing_impl"], "expert_scan")
        self.assertTrue(config["checkpoint"]["resume_from"].endswith("cudablock-diag393/checkpoints"))
        self.assertFalse(config["checkpoint"]["finetune"])
        self.assertTrue(config["checkpoint"]["resume_optimizer_state_required"])
        self.assertEqual(config["checkpoint"]["save_interval"], 1289)
        self.assertTrue(config["checkpoint"]["no_save_optim"])
        self.assertIn("cudablock-resume393", queue["items"][0]["run_id"])

    def test_publication_independent_critic_uses_seeded_init_and_shared_family_split(self):
        primary_path = (
            ROOT / "configs" / "nano_ar" / "publication" / "r33_family_clean_sft.yaml"
        )
        independent_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "publication"
            / "r33_family_clean_independent_sft.yaml"
        )
        queue_path = (
            ROOT
            / "configs"
            / "nano_ar"
            / "publication"
            / "r33_family_clean_independent_sft_queue.yaml"
        )

        primary = yaml.safe_load(primary_path.read_text())
        independent = yaml.safe_load(independent_path.read_text())
        queue = yaml.safe_load(queue_path.read_text())
        av = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "nano_av"
                / "publication"
                / "r33_family_clean_sft.yaml"
            ).read_text()
        )

        self.assertEqual(primary["paths"]["input_ar_sft"], independent["paths"]["input_ar_sft"])
        self.assertEqual(primary["dataset"]["seed"], independent["dataset"]["seed"])
        self.assertEqual(
            primary["dataset"]["split_cache_dir"],
            independent["dataset"]["split_cache_dir"],
        )
        self.assertNotEqual(
            primary["paths"]["critic_init_model_id"],
            independent["paths"]["critic_init_model_id"],
        )
        self.assertEqual(independent["training"]["rollout_seed"], 314159)
        self.assertEqual(
            primary["training"]["mamba_kernel_mode"], "unfused_torch_conv"
        )
        self.assertEqual(
            independent["training"]["mamba_kernel_mode"], "unfused_torch_conv"
        )
        self.assertEqual(independent["eval"]["eval_splits"], ["validation"])
        for config in (primary, independent, av):
            self.assertEqual(config["dataset"]["split_mode"], "content_family_manifest")
            self.assertEqual(config["dataset"]["seed"], 20260709)
            self.assertTrue(config["dataset"]["content_family_manifest"].endswith(
                "/r33_confirmatory_family_manifest.json"
            ))
            self.assertEqual(
                config["dataset"]["content_family_manifest_sha256"],
                "479cbab5d21cd031cb72a770eebb3428e0d5419ebf8cce38c2ca6025e49741b6",
            )
        self.assertEqual(queue["items"][0]["status"], "pending")
        self.assertEqual(
            queue["items"][0]["config"],
            "configs/nano_ar/publication/r33_family_clean_independent_sft.yaml",
        )


if __name__ == "__main__":
    unittest.main()
