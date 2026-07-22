import contextlib
import importlib.util
import io
import pathlib
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class NanoAVProbeQueueTests(unittest.TestCase):
    def _write_queue(self, root: pathlib.Path, text: str) -> pathlib.Path:
        path = root / "queue.yaml"
        path.write_text(textwrap.dedent(text))
        return path

    def test_shuffled_control_candidates_stay_in_split_and_exclude_family(self):
        evaluator = load_script("nano_eval_core")
        rows = [
            {"row_index": 0, "split": "validation", "content_family_id": "a"},
            {"row_index": 1, "split": "validation", "content_family_id": "a"},
            {"row_index": 2, "split": "validation", "content_family_id": "b"},
            {"row_index": 3, "split": "test", "content_family_id": "c"},
        ]

        self.assertEqual(
            evaluator.shuffled_control_candidates(rows, row_index=0),
            [2],
        )

    def test_converted_hf_checkpoint_sits_beside_checkpoint_root(self):
        queue = load_script("nano_av_probe_queue")
        checkpoint = pathlib.Path("/out/run/checkpoints/iter_0000128")

        self.assertEqual(queue.converted_hf_checkpoint_for_dcp(checkpoint), pathlib.Path("/out/run/hf_iter_0000128"))

    def test_process_next_dry_run_returns_all_commands_without_mutating_queue(self):
        queue = load_script("nano_av_probe_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text("placeholder: true\n")
            queue_path = self._write_queue(
                root,
                """
                schema_version: nano_av_probe_queue.v1
                defaults:
                  python: /venv/bin/python
                  eval_splits: [validation]
                items:
                  - name: av-dry-run
                    config: config.yaml
                    status: pending
                """,
            )
            original_queue = queue_path.read_text()
            run_dir = root / "run"
            plan = {
                "run_dir": str(run_dir),
                "num_rollout": 128,
                "train_parquet": str(run_dir / "splits" / "train.parquet"),
                "split_manifest": {
                    "validation": {"path": str(run_dir / "splits" / "validation.parquet")},
                    "test": {"path": str(run_dir / "splits" / "test.parquet")},
                },
                "command": ["python", "train.py", "--example"],
                "environment": {},
            }
            spec = {
                "paths": {"code_root": str(root), "model_id": "/models/nano"},
                "training": {"injection_scale": 75},
                "run": {"wandb_project": "project", "wandb_group": "group"},
                "eval": {
                    "validation_limit": 64,
                    "test_limit": 64,
                    "eval_splits": ["validation"],
                    "generation_examples": 0,
                    "roundtrip": {
                        "enabled": True,
                        "ar_checkpoint_dir": str(root / "ar-checkpoint"),
                        "validation_limit": 16,
                        "test_limit": 16,
                    },
                },
            }

            with mock.patch.object(
                queue.nano_av_runner, "load_and_validate_spec", return_value=spec
            ), mock.patch.object(
                queue.nano_av_runner, "prepare_run", return_value=plan
            ), mock.patch.object(
                queue, "run_logged", side_effect=AssertionError("dry run executed a command")
            ):
                result = queue.process_next(queue_path, dry_run=True)

            self.assertEqual(result["status"], "dry_run")
            self.assertEqual(result["item_name"], "av-dry-run")
            self.assertIn("train_command", result)
            self.assertIn("convert_command", result)
            self.assertIn("eval_command", result)
            self.assertIsNotNone(result["roundtrip_command"])
            self.assertEqual(
                result["roundtrip_command"][
                    result["roundtrip_command"].index("--ar-checkpoint-dir") + 1
                ],
                str(root / "ar-checkpoint"),
            )
            self.assertEqual(queue_path.read_text(), original_queue)

    def test_main_dry_run_forwards_flag_and_stops_after_one_item(self):
        queue = load_script("nano_av_probe_queue")
        with tempfile.TemporaryDirectory() as tmp:
            queue_path = self._write_queue(
                pathlib.Path(tmp),
                """
                schema_version: nano_av_probe_queue.v1
                items:
                  - name: av-dry-run
                    config: config.yaml
                    status: pending
                """,
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(
                sys, "argv", ["nano_av_probe_queue.py", str(queue_path), "--dry-run"]
            ), mock.patch.object(
                queue,
                "process_next",
                return_value={"status": "dry_run", "run_dir": pathlib.Path("/tmp/run")},
            ) as process_next, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                exit_code = queue.main()

            self.assertEqual(exit_code, 0)
            self.assertIn('"run_dir": "/tmp/run"', stdout.getvalue())
            process_next.assert_called_once_with(queue_path, dry_run=True)

    def test_eval_command_uses_converted_hf_checkpoint(self):
        queue = load_script("nano_av_probe_queue")
        command = queue.build_eval_command(
            python_bin="/venv/bin/python",
            hf_checkpoint=pathlib.Path("/out/run/hf_iter_0000128"),
            train_parquet=pathlib.Path("/out/run/splits/train_padded.parquet"),
            validation_parquet=pathlib.Path("/out/run/splits/validation.parquet"),
            test_parquet=pathlib.Path("/out/run/splits/test.parquet"),
            report_json=pathlib.Path("/out/run/eval.json"),
            validation_limit=512,
            test_limit=512,
            eval_splits=["validation"],
            generation_examples=4,
            injection_scale="75",
            wandb_project="nano30b-nla-pilot",
            wandb_group="nano-av-layer-probes",
        )

        self.assertEqual(command[command.index("--hf-checkpoint") + 1], "/out/run/hf_iter_0000128")
        self.assertEqual(command[command.index("--eval-splits") + 1], "validation")
        self.assertNotIn("/out/run/checkpoints/iter_0000128/hf", command)

    def test_roundtrip_command_uses_converted_av_checkpoint_and_ar_scorer(self):
        queue = load_script("nano_av_probe_queue")
        command = queue.build_roundtrip_command(
            python_bin="/venv/bin/python",
            av_hf_checkpoint=pathlib.Path("/out/run/hf_iter_0000128"),
            ar_checkpoint_dir=pathlib.Path("/models/ar_hf_iter_0000256"),
            train_parquet=pathlib.Path("/out/run/splits/train_padded.parquet"),
            validation_parquet=pathlib.Path("/out/run/splits/validation.parquet"),
            test_parquet=pathlib.Path("/out/run/splits/test.parquet"),
            report_json=pathlib.Path("/out/run/roundtrip.json"),
            generated_jsonl=pathlib.Path("/out/run/roundtrip_generated.jsonl"),
            prediction_cache_npz=pathlib.Path("/out/run/roundtrip_predictions.npz"),
            generation_controls=["real"],
            validation_limit=16,
            test_limit=16,
            max_new_tokens=64,
            injection_scale="75",
            ar_batch_size=2,
            torch_dtype="bfloat16",
            ar_device_map="cuda:0",
            ar_low_cpu_mem_usage=True,
            generation_prefix="<explanation>\n",
            stop_text="</explanation>",
            generated_text_fallback="empty",
            generation_backend="cache",
            generation_workers=2,
            generation_worker_devices=["0", "1"],
            stream_generated=True,
            resume_generated=True,
            progress_every=8,
            min_control_win_fraction=0.9,
            min_closed_fraction=0.8,
            min_usable_fraction=0.95,
            content_family_manifest=pathlib.Path("/data/content_families.json"),
            content_family_coverage=pathlib.Path("/data/content_family_coverage.json"),
            selection_strategy="family_stratified",
            selection_seed=20260708,
            permutation_samples=4321,
            permutation_seed=19,
            require_family_level_inference=True,
            min_independent_families=100,
            av_model_fingerprint="model-sha",
            av_tokenizer_fingerprint="tokenizer-sha",
            require_generation_protocol_match=True,
            generation_only=True,
        )

        self.assertEqual(command[command.index("--av-hf-checkpoint") + 1], "/out/run/hf_iter_0000128")
        self.assertEqual(command[command.index("--ar-checkpoint-dir") + 1], "/models/ar_hf_iter_0000256")
        self.assertEqual(
            command[command.index("--prediction-cache-npz") + 1],
            "/out/run/roundtrip_predictions.npz",
        )
        self.assertEqual(command[command.index("--generation-controls") + 1], "real")
        self.assertEqual(command[command.index("--validation-limit") + 1], "16")
        self.assertEqual(command[command.index("--ar-device-map") + 1], "cuda:0")
        self.assertIn("--ar-low-cpu-mem-usage", command)
        self.assertEqual(command[command.index("--generation-prefix") + 1], "<explanation>\n")
        self.assertEqual(command[command.index("--stop-text") + 1], "</explanation>")
        self.assertEqual(command[command.index("--generated-text-fallback") + 1], "empty")
        self.assertEqual(command[command.index("--generation-backend") + 1], "cache")
        self.assertEqual(command[command.index("--generation-workers") + 1], "2")
        self.assertEqual(command[command.index("--generation-worker-devices") + 1: command.index("--generation-worker-devices") + 3], ["0", "1"])
        self.assertIn("--stream-generated", command)
        self.assertIn("--resume-generated", command)
        self.assertIn("--generation-only", command)
        self.assertEqual(command[command.index("--min-control-win-fraction") + 1], "0.9")
        self.assertEqual(command[command.index("--min-closed-fraction") + 1], "0.8")
        self.assertEqual(
            command[command.index("--av-model-fingerprint") + 1], "model-sha"
        )
        self.assertEqual(
            command[command.index("--av-tokenizer-fingerprint") + 1],
            "tokenizer-sha",
        )
        self.assertIn("--require-generation-protocol-match", command)
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
        self.assertIn("--require-family-level-inference", command)
        self.assertIn("scripts/eval_nano_av_ar_roundtrip_gate.py", command)

    def test_convert_command_can_request_bf16_checkpoint_dtype(self):
        queue = load_script("nano_av_probe_queue")
        command = queue.build_convert_command(
            python_bin="/venv/bin/python",
            code_root=ROOT,
            checkpoint_dir=pathlib.Path("/out/run/checkpoints/iter_0000128"),
            origin_hf_dir=pathlib.Path("/models/nano"),
            output_dir=pathlib.Path("/out/run/hf_iter_0000128"),
            torch_dtype="bfloat16",
        )

        self.assertIn("--torch-dtype", command)
        self.assertEqual(command[command.index("--torch-dtype") + 1], "bfloat16")

    def test_eval_only_skips_training_only_when_checkpoint_exists(self):
        queue = load_script("nano_av_probe_queue")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = pathlib.Path(tmp) / "run" / "checkpoints" / "iter_0000128"

            self.assertFalse(queue.should_skip_training({"eval_only": True}, checkpoint))
            checkpoint.mkdir(parents=True)
            self.assertTrue(queue.should_skip_training({"eval_only": True}, checkpoint))
            self.assertTrue(queue.should_skip_training({"skip_training_if_checkpoint_exists": True}, checkpoint))
            self.assertFalse(queue.should_skip_training({}, checkpoint))

    def test_converted_hf_checkpoint_cleanup_defaults_to_true(self):
        queue = load_script("nano_av_probe_queue")

        self.assertTrue(queue.should_cleanup_converted_hf({}, {}))
        self.assertFalse(queue.should_cleanup_converted_hf({"cleanup_converted_hf_after_eval": False}, {}))
        self.assertFalse(queue.should_cleanup_converted_hf({}, {"cleanup_converted_hf_after_eval": False}))
        self.assertTrue(queue.should_cleanup_converted_hf({"cleanup_converted_hf_after_eval": False}, {"cleanup_converted_hf_after_eval": True}))

    def test_roundtrip_config_merges_defaults_spec_and_item(self):
        queue = load_script("nano_av_probe_queue")

        merged = queue.roundtrip_config_for_item(
            defaults={"roundtrip": {"enabled": True, "validation_limit": 128, "generation_controls": ["real"]}},
            spec={"eval": {"roundtrip": {"test_limit": 64, "max_new_tokens": 96}}},
            item={"roundtrip": {"validation_limit": 16, "ar_checkpoint_dir": "/ar"}},
        )
        disabled = queue.roundtrip_config_for_item(defaults={}, spec={}, item={})

        self.assertIsNone(disabled)
        self.assertEqual(merged["validation_limit"], 16)
        self.assertEqual(merged["test_limit"], 64)
        self.assertEqual(merged["max_new_tokens"], 96)
        self.assertEqual(merged["generation_controls"], ["real"])
        self.assertEqual(merged["ar_checkpoint_dir"], "/ar")

    def test_checked_in_r33_100k_queue_is_valid(self):
        queue = load_script("nano_av_probe_queue")
        queue_path = ROOT / "configs" / "nano_av" / "hpo" / "r33_100k_scaling_queue.yaml"

        loaded = queue.load_queue(queue_path)

        self.assertIsNone(queue.next_pending_index(loaded))
        self.assertEqual(len(loaded["items"]), 1)

    def test_checked_in_r33_dedup_queue_has_terminal_roundtrip_smoke(self):
        queue = load_script("nano_av_probe_queue")
        queue_path = ROOT / "configs" / "nano_av" / "hpo" / "r33_dedup_clean_queue.yaml"

        loaded = queue.load_queue(queue_path)
        first = loaded["items"][0]
        roundtrip = queue.roundtrip_config_for_item(
            defaults=loaded.get("defaults") or {},
            item=first,
            spec={"eval": {}},
        )

        self.assertIsNone(queue.next_pending_index(loaded))
        self.assertEqual(first["status"], "failed")
        self.assertIn("OOM", first["failure"])
        self.assertEqual(first["study_task"], "av_roundtrip")
        self.assertIsNotNone(roundtrip)
        self.assertEqual(roundtrip["validation_limit"], 64)
        self.assertEqual(roundtrip["test_limit"], 64)
        self.assertEqual(roundtrip["generation_controls"], ["real", "shuffled", "zero", "mean", "none"])
        self.assertEqual(roundtrip["control_margin"], 5e-5)
        self.assertEqual(roundtrip["min_control_win_fraction"], 0.9)
        self.assertTrue(roundtrip["resume_generated"])
        self.assertIn("r27_roundtrip_v256_t256_full_controls_prefix256_report.json", roundtrip["baseline_report_json"])
        self.assertIn("nano-ar-r33-dedup-clean56k-lr5e5-cosine-warmup25-gb192-mb96-128step-padded", roundtrip["ar_checkpoint_dir"])

    def test_load_queue_accepts_historical_terminal_statuses(self):
        queue = load_script("nano_av_probe_queue")
        with tempfile.TemporaryDirectory() as tmp:
            queue_path = self._write_queue(
                pathlib.Path(tmp),
                """
                schema_version: nano_av_probe_queue.v1
                items:
                  - name: cancelled
                    config: configs/nano_av/hpo/cancelled.yaml
                    status: cancelled
                  - name: blocked
                    config: configs/nano_av/hpo/blocked.yaml
                    status: blocked_missing_dataset
                """,
            )

            loaded = queue.load_queue(queue_path)

        self.assertIsNone(queue.next_pending_index(loaded))

    def test_reset_active_items_requeues_stale_training_and_eval_items(self):
        queue = load_script("nano_av_probe_queue")
        with tempfile.TemporaryDirectory() as tmp:
            queue_path = self._write_queue(
                pathlib.Path(tmp),
                """
                schema_version: nano_av_probe_queue.v1
                items:
                  - name: active-training
                    config: configs/nano_av/hpo/a.yaml
                    status: training
                  - name: active-eval
                    config: configs/nano_av/hpo/b.yaml
                    status: eval_running
                  - name: done
                    config: configs/nano_av/hpo/c.yaml
                    status: complete
                """,
            )

            result = queue.reset_active_items(queue_path, reason="unit test")
            loaded = queue.load_queue(queue_path)

        self.assertEqual(result["reset_count"], 2)
        self.assertEqual(loaded["items"][0]["status"], "pending")
        self.assertEqual(loaded["items"][0]["previous_status"], "training")
        self.assertEqual(loaded["items"][1]["status"], "pending")
        self.assertEqual(loaded["items"][1]["previous_status"], "eval_running")
        self.assertEqual(loaded["items"][2]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
