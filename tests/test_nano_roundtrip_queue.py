from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import yaml


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


class NanoRoundtripQueueTests(unittest.TestCase):
    def test_queue_code_root_overrides_template_config_root(self):
        queue = load_script("nano_roundtrip_queue")

        resolved = queue.code_root_for_config(
            {"paths": {"code_root": "/template/source"}},
            {"defaults": {"code_root": "/immutable/source"}},
        )

        self.assertEqual(resolved, pathlib.Path("/immutable/source"))

    def test_chained_source_environment_overrides_queue_root(self):
        queue = load_script("nano_roundtrip_queue")

        with mock.patch.dict("os.environ", {"NANO_QUEUE_CODE_ROOT": "/chain/source"}):
            resolved = queue.code_root_for_config(
                {"paths": {"code_root": "/template/source"}},
                {"defaults": {"code_root": "/queue/source"}},
            )

        self.assertEqual(resolved, pathlib.Path("/chain/source"))

    def test_prepared_checkpoint_overrides_hf_and_fingerprints(self):
        queue = load_script("nano_roundtrip_queue")
        config = {
            "paths": {"av_hf_checkpoint": "/old/hf"},
            "eval": {
                "av_model_fingerprint": "pending-model",
                "av_tokenizer_fingerprint": "pending-tokenizer",
            },
        }

        resolved = queue.apply_prepared_checkpoint(
            config,
            {
                "output_hf_dir": "/tmp/prepared-hf",
                "av_model_fingerprint": "dcp_model_sha256:model",
                "av_tokenizer_fingerprint": "tokenizer_files_sha256:tokenizer",
            },
        )

        self.assertEqual(resolved["paths"]["av_hf_checkpoint"], "/tmp/prepared-hf")
        self.assertEqual(
            resolved["eval"]["av_model_fingerprint"],
            "dcp_model_sha256:model",
        )
        self.assertEqual(
            resolved["eval"]["av_tokenizer_fingerprint"],
            "tokenizer_files_sha256:tokenizer",
        )
        self.assertEqual(config["paths"]["av_hf_checkpoint"], "/old/hf")

    def test_tokenizer_fingerprint_is_content_sensitive(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "tokenizer.json").write_text("one")
            (root / "config.json").write_text("ignored")
            first = queue.fingerprint_tokenizer_files(root)
            (root / "tokenizer.json").write_text("two")
            second = queue.fingerprint_tokenizer_files(root)

        self.assertEqual(first["files"], ["tokenizer.json"])
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_prepare_av_checkpoint_converts_once_and_caches_fingerprints(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = root / "checkpoint"
            model = checkpoint / "model"
            origin = root / "origin"
            output = root / "prepared"
            report_path = root / "fingerprint.json"
            model.mkdir(parents=True)
            origin.mkdir()
            (model / "rank0.distcp").write_bytes(b"model")
            (origin / "config.json").write_text("{}")
            (origin / "tokenizer.json").write_text("tokenizer")
            queue_doc = {
                "defaults": {
                    "av_checkpoint_prepare": {
                        "python": "/venv/bin/python",
                        "dcp_checkpoint": str(checkpoint),
                        "origin_hf_dir": str(origin),
                        "output_hf_dir": str(output),
                        "fingerprint_report_json": str(report_path),
                        "convert_log": str(root / "convert.log"),
                    }
                }
            }
            commands = []

            def fake_run_logged(command, *, cwd, env, log_path):
                commands.append(command)
                output.mkdir(parents=True)
                (output / "config.json").write_text("{}")

            with mock.patch.object(queue, "run_logged", side_effect=fake_run_logged):
                first = queue.prepare_av_checkpoint(
                    queue_doc,
                    code_root=root,
                    env={},
                )
                second = queue.prepare_av_checkpoint(
                    queue_doc,
                    code_root=root,
                    env={},
                )

        self.assertEqual(len(commands), 1)
        self.assertEqual(first["av_model_fingerprint"], second["av_model_fingerprint"])
        self.assertTrue(first["av_model_fingerprint"].startswith("dcp_model_sha256:"))
        self.assertTrue(
            first["av_tokenizer_fingerprint"].startswith("tokenizer_files_sha256:")
        )

    def test_prepare_av_checkpoint_reuses_attested_existing_hf(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = root / "checkpoint"
            model = checkpoint / "model"
            origin = root / "origin"
            output = root / "prepared"
            report_path = root / "fingerprint.json"
            source_report = root / "source-fingerprint.json"
            model.mkdir(parents=True)
            origin.mkdir()
            output.mkdir()
            (model / "rank0.distcp").write_bytes(b"model")
            (origin / "config.json").write_text("{}")
            (origin / "tokenizer.json").write_text("tokenizer")
            (output / "config.json").write_text("{}")
            (output / "model.safetensors").write_bytes(b"converted")
            source_report.write_text(
                json.dumps(
                    {
                        "dcp_checkpoint": str(checkpoint),
                        "origin_hf_dir": str(origin),
                        "model_stat_signature": queue._directory_stat_signature(model),
                        "tokenizer_stat_signature": queue._directory_stat_signature(origin),
                        "av_model_fingerprint": "dcp_model_sha256:model",
                        "av_tokenizer_fingerprint": "tokenizer_files_sha256:tokenizer",
                    }
                )
            )
            queue_doc = {
                "defaults": {
                    "av_checkpoint_prepare": {
                        "python": "/venv/bin/python",
                        "dcp_checkpoint": str(checkpoint),
                        "origin_hf_dir": str(origin),
                        "output_hf_dir": str(output),
                        "fingerprint_report_json": str(report_path),
                        "convert_log": str(root / "convert.log"),
                        "reuse_existing_hf": True,
                        "source_fingerprint_report_json": str(source_report),
                        "expected_model_fingerprint": "dcp_model_sha256:model",
                        "expected_tokenizer_fingerprint": "tokenizer_files_sha256:tokenizer",
                    }
                }
            }

            with mock.patch.object(queue, "run_logged") as run_logged:
                report = queue.prepare_av_checkpoint(
                    queue_doc, code_root=root, env={}
                )

        run_logged.assert_not_called()
        self.assertTrue(report["reused_existing_hf"])
        self.assertEqual(report["output_hf_dir"], str(output))

    def test_prepare_av_checkpoint_reuse_rejects_wrong_fingerprint(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = root / "checkpoint"
            model = checkpoint / "model"
            origin = root / "origin"
            output = root / "prepared"
            model.mkdir(parents=True)
            origin.mkdir()
            output.mkdir()
            (model / "rank0.distcp").write_bytes(b"model")
            (origin / "config.json").write_text("{}")
            (origin / "tokenizer.json").write_text("tokenizer")
            (output / "config.json").write_text("{}")
            (output / "model.safetensors").write_bytes(b"converted")
            source_report = root / "source-fingerprint.json"
            source_report.write_text(
                json.dumps(
                    {
                        "dcp_checkpoint": str(checkpoint),
                        "origin_hf_dir": str(origin),
                        "model_stat_signature": queue._directory_stat_signature(model),
                        "tokenizer_stat_signature": queue._directory_stat_signature(origin),
                        "av_model_fingerprint": "dcp_model_sha256:wrong",
                        "av_tokenizer_fingerprint": "tokenizer_files_sha256:tokenizer",
                    }
                )
            )
            queue_doc = {
                "defaults": {
                    "av_checkpoint_prepare": {
                        "dcp_checkpoint": str(checkpoint),
                        "origin_hf_dir": str(origin),
                        "output_hf_dir": str(output),
                        "fingerprint_report_json": str(root / "fingerprint.json"),
                        "convert_log": str(root / "convert.log"),
                        "reuse_existing_hf": True,
                        "source_fingerprint_report_json": str(source_report),
                        "expected_model_fingerprint": "dcp_model_sha256:model",
                        "expected_tokenizer_fingerprint": "tokenizer_files_sha256:tokenizer",
                    }
                }
            }

            with self.assertRaises(queue.RoundtripQueueError):
                queue.prepare_av_checkpoint(queue_doc, code_root=root, env={})

    def test_prepare_av_checkpoint_stages_attested_existing_hf(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = root / "checkpoint"
            model = checkpoint / "model"
            origin = root / "origin"
            source_hf = root / "source-hf"
            output = root / "staged-hf"
            model.mkdir(parents=True)
            origin.mkdir()
            source_hf.mkdir()
            (model / "rank0.distcp").write_bytes(b"model")
            (origin / "config.json").write_text("{}")
            (origin / "tokenizer.json").write_text("tokenizer")
            (source_hf / "config.json").write_text("{}")
            (source_hf / "model.safetensors").write_bytes(b"converted")
            source_report = root / "source-fingerprint.json"
            source_report.write_text(
                json.dumps(
                    {
                        "dcp_checkpoint": str(checkpoint),
                        "origin_hf_dir": str(origin),
                        "model_stat_signature": queue._directory_stat_signature(model),
                        "tokenizer_stat_signature": queue._directory_stat_signature(origin),
                        "av_model_fingerprint": "dcp_model_sha256:model",
                        "av_tokenizer_fingerprint": "tokenizer_files_sha256:tokenizer",
                    }
                )
            )
            queue_doc = {
                "defaults": {
                    "av_checkpoint_prepare": {
                        "dcp_checkpoint": str(checkpoint),
                        "origin_hf_dir": str(origin),
                        "output_hf_dir": str(output),
                        "fingerprint_report_json": str(root / "fingerprint.json"),
                        "convert_log": str(root / "convert.log"),
                        "reuse_existing_hf": True,
                        "existing_hf_source_dir": str(source_hf),
                        "existing_hf_stage_workers": 2,
                        "existing_hf_stage_task_bytes": 3,
                        "source_fingerprint_report_json": str(source_report),
                        "expected_model_fingerprint": "dcp_model_sha256:model",
                        "expected_tokenizer_fingerprint": "tokenizer_files_sha256:tokenizer",
                    }
                }
            }

            first = queue.prepare_av_checkpoint(queue_doc, code_root=root, env={})
            second = queue.prepare_av_checkpoint(queue_doc, code_root=root, env={})
            staged_bytes = (output / "model.safetensors").read_bytes()
            first_reused = first["existing_hf_stage"]["reused"]
            second_reused = second["existing_hf_stage"]["reused"]

        self.assertEqual(staged_bytes, b"converted")
        self.assertFalse(first_reused)
        self.assertTrue(second_reused)

    def test_prepare_av_checkpoint_stages_and_fingerprints_in_one_pass(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = root / "checkpoint"
            model = checkpoint / "model"
            staged_checkpoint = root / "staged-checkpoint"
            origin = root / "origin"
            output = root / "prepared"
            report_path = root / "fingerprint.json"
            model.mkdir(parents=True)
            origin.mkdir()
            (model / "rank0.distcp").write_bytes(b"model-bytes")
            (origin / "config.json").write_text("{}")
            (origin / "tokenizer.json").write_text("tokenizer")
            expected_model_sha = queue.nano_source_provenance.fingerprint_directory(
                model,
                label="av_dcp_model",
            )["sha256"]
            queue_doc = {
                "defaults": {
                    "av_checkpoint_prepare": {
                        "python": "/venv/bin/python",
                        "dcp_checkpoint": str(checkpoint),
                        "stage_dcp_checkpoint": str(staged_checkpoint),
                        "stage_copy_workers": 2,
                        "stage_copy_task_bytes": 5,
                        "cleanup_staged_dcp_after_conversion": True,
                        "origin_hf_dir": str(origin),
                        "output_hf_dir": str(output),
                        "fingerprint_report_json": str(report_path),
                        "convert_log": str(root / "convert.log"),
                    }
                }
            }
            commands = []

            def fake_run_logged(command, *, cwd, env, log_path):
                commands.append(command)
                conversion_input = pathlib.Path(
                    command[command.index("--input-dir") + 1]
                )
                self.assertEqual(conversion_input, staged_checkpoint)
                self.assertEqual(
                    (conversion_input / "model" / "rank0.distcp").read_bytes(),
                    b"model-bytes",
                )
                output.mkdir(parents=True)
                (output / "config.json").write_text("{}")

            with mock.patch.object(queue, "run_logged", side_effect=fake_run_logged):
                report = queue.prepare_av_checkpoint(
                    queue_doc,
                    code_root=root,
                    env={},
                )

        self.assertEqual(len(commands), 1)
        self.assertFalse(staged_checkpoint.exists())
        self.assertEqual(report["staged_dcp_cleanup"], True)
        self.assertEqual(report["stage_copy_workers"], 2)
        self.assertEqual(report["stage_copy_task_bytes"], 5)
        self.assertEqual(report["model"]["sha256"], expected_model_sha)

    def test_publication_configs_do_not_resolve_through_historical_hero_source(self):
        publication_root = ROOT / "configs/nano_roundtrip/publication"
        expected = "/workspace/interp/code/nano30b-nla-pilot-publication-current"

        checked = 0
        for path in publication_root.glob("*.yaml"):
            payload = yaml.safe_load(path.read_text()) or {}
            code_roots = []
            if isinstance(payload.get("paths"), dict) and payload["paths"].get(
                "code_root"
            ):
                code_roots.append(payload["paths"]["code_root"])
            if isinstance(payload.get("defaults"), dict) and payload[
                "defaults"
            ].get("code_root"):
                code_roots.append(payload["defaults"]["code_root"])
            for code_root in code_roots:
                checked += 1
                self.assertEqual(code_root, expected, path.name)

        self.assertGreater(checked, 0)

    def test_checked_in_publication_queue_dry_runs_from_local_repo(self):
        queue = load_script("nano_roundtrip_queue")

        result = queue.dry_run_queue(
            ROOT
            / "configs"
            / "nano_roundtrip"
            / "publication"
            / "r33_existing_hero_corrected_queue.yaml"
        )

        self.assertTrue(result["protocols_match"])
        self.assertEqual(len(result["items"]), 4)
        self.assertEqual(len(result["protocol_sha256s"]), 1)

    def test_checked_in_clean_sft_validation_queue_has_protocol_parity(self):
        queue = load_script("nano_roundtrip_queue")

        result = queue.dry_run_queue(
            ROOT
            / "configs"
            / "nano_roundtrip"
            / "publication"
            / "r33_clean_sft_validation_queue.yaml"
        )

        self.assertTrue(result["protocols_match"])
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(len(result["protocol_sha256s"]), 1)

    def test_dry_run_queue_reports_protocol_parity_without_mutation(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)

            def write_config(path: pathlib.Path, *, model_sha: str, max_tokens: int):
                model_fingerprint = "hf_model_sha256:" + hashlib.sha256(
                    model_sha.encode()
                ).hexdigest()
                tokenizer_fingerprint = "tokenizer_files_sha256:" + hashlib.sha256(
                    b"shared-tokenizer"
                ).hexdigest()
                path.write_text(
                    textwrap.dedent(
                        f"""
                        schema_version: nano_roundtrip_eval.v1
                        paths:
                          av_hf_checkpoint: /hf/av
                          ar_checkpoint_dir: /hf/ar
                          train_parquet: /data/train.parquet
                          validation_parquet: /data/validation.parquet
                          test_parquet: /data/test.parquet
                          report_json: /out/{path.stem}.json
                        eval:
                          validation_limit: 8
                          test_limit: 8
                          generation_controls: [real]
                          max_new_tokens: {max_tokens}
                          seed: 20260708
                          generation_prefix: ""
                          stop_text: "</explanation>"
                          generated_text_fallback: raw
                          generation_backend: legacy_batch
                          generation_workers: 2
                          av_model_fingerprint: {model_fingerprint}
                          av_tokenizer_fingerprint: {tokenizer_fingerprint}
                          require_generation_protocol_match: true
                        """
                    )
                )

            sft = root / "sft.yaml"
            hero = root / "hero.yaml"
            write_config(sft, model_sha="sft-sha", max_tokens=64)
            write_config(hero, model_sha="hero-sha", max_tokens=64)
            queue_path = root / "queue.yaml"
            queue_path.write_text(
                textwrap.dedent(
                    f"""
                    schema_version: nano_roundtrip_queue.v1
                    items:
                      - name: sft
                        config: {sft}
                        status: pending
                      - name: hero
                        config: {hero}
                        status: pending
                    """
                )
            )

            matched = queue.dry_run_queue(queue_path)
            statuses_after = [
                item["status"] for item in queue.load_queue(queue_path)["items"]
            ]
            write_config(hero, model_sha="hero-sha", max_tokens=65)
            mismatched = queue.dry_run_queue(queue_path)

        self.assertTrue(matched["protocols_match"])
        self.assertEqual(len(matched["protocol_sha256s"]), 1)
        self.assertEqual(statuses_after, ["pending", "pending"])
        self.assertFalse(mismatched["protocols_match"])
        self.assertEqual(len(mismatched["protocol_sha256s"]), 2)

    def test_queue_loads_pending_and_blocked_items(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            queue_path = root / "queue.yaml"
            queue_path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_roundtrip_queue.v1
                    defaults:
                      python: /venv/bin/python
                      code_root: /code
                    items:
                      - name: r27-64
                        config: configs/nano_roundtrip/r27.yaml
                        status: pending
                      - name: r27-256
                        config: configs/nano_roundtrip/r27-256.yaml
                        status: blocked_waiting_on_parse_health
                    """
                )
            )

            loaded = queue.load_queue(queue_path)

        self.assertEqual(queue.next_pending_index(loaded), 0)
        self.assertEqual(queue.status_counts(loaded["items"], queue.VALID_STATUSES)["pending"], 1)

    def test_reset_active_items_requeues_running_statuses(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            queue_path = root / "queue.yaml"
            queue_path.write_text(
                textwrap.dedent(
                    """
                    schema_version: nano_roundtrip_queue.v1
                    items:
                      - name: running
                        config: a.yaml
                        status: running
                      - name: scoring
                        config: b.yaml
                        status: scoring
                      - name: done
                        config: c.yaml
                        status: complete
                    """
                )
            )

            result = queue.reset_active_items(queue_path, reason="unit test")
            loaded = queue.load_queue(queue_path)

        self.assertEqual(result["reset_count"], 2)
        self.assertEqual(loaded["items"][0]["status"], "pending")
        self.assertEqual(loaded["items"][1]["status"], "pending")
        self.assertEqual(loaded["items"][2]["status"], "complete")

    def test_checked_in_r27_baseline_queue_reflects_completed_64_and_256(self):
        queue = load_script("nano_roundtrip_queue")
        loaded = queue.load_queue(ROOT / "configs" / "nano_roundtrip" / "r27_baseline_queue.yaml")
        items = {item["name"]: item for item in loaded["items"]}

        first = items["r27-baseline-roundtrip-v64-t64-full-controls-prefix256"]
        second = items["r27-baseline-roundtrip-v256-t256-full-controls-prefix256"]

        self.assertEqual(first["status"], "complete")
        self.assertTrue(first["gate_passed"])
        self.assertIn("r27_roundtrip_v64_t64_full_controls_prefix256_report.json", first["report_json"])
        self.assertIn("r27_roundtrip_v64_t64_full_controls_prefix256_generated.jsonl", first["generated_jsonl"])
        self.assertEqual(second["status"], "complete")
        self.assertTrue(second["gate_passed"])
        self.assertIn("r27_roundtrip_v256_t256_full_controls_prefix256_report.json", second["report_json"])
        self.assertIn("r27_roundtrip_v256_t256_full_controls_prefix256_generated.jsonl", second["generated_jsonl"])

    def test_process_next_respects_active_process_guard_without_mutating_queue(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config = root / "roundtrip.yaml"
            config.write_text(
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
                      validation_limit: 1
                      test_limit: 1
                    """
                )
            )
            queue_path = root / "queue.yaml"
            queue_path.write_text(
                textwrap.dedent(
                    f"""
                    schema_version: nano_roundtrip_queue.v1
                    defaults:
                      launch_guard:
                        block_if_process_matches: ["train.py"]
                    items:
                      - name: guarded
                        config: {config}
                        status: pending
                    """
                )
            )

            result = queue.process_next(queue_path, active_process_lines=["123 00:01 train.py --train-backend fsdp"])
            loaded = queue.load_queue(queue_path)

        self.assertEqual(result["status"], "blocked_active_process")
        self.assertEqual(loaded["items"][0]["status"], "pending")
        self.assertIn("train.py", result["matches"][0])

    def test_generation_only_report_validates_cache_without_scoring_gate(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            generated = root / "generated.jsonl"
            generated.write_text('{"row": 0}\n{"row": 1}\n')
            report = {
                "schema_version": "nano_roundtrip_generation_report.v1",
                "row_count": 2,
                "generated_jsonl": str(generated),
                "generation_protocol_sha256": "a" * 64,
            }
            result = queue._validate_generation_report(
                report,
                config={
                    "eval": {
                        "eval_splits": ["validation"],
                        "validation_limit": 2,
                        "generation_only": True,
                    }
                },
                generated_jsonl=generated,
            )

        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["generation_protocol_sha256"], "a" * 64)

    def test_generation_only_report_rejects_partial_cache(self):
        queue = load_script("nano_roundtrip_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            generated = root / "generated.jsonl"
            generated.write_text('{"row": 0}\n')
            report = {
                "schema_version": "nano_roundtrip_generation_report.v1",
                "row_count": 2,
                "generated_jsonl": str(generated),
                "generation_protocol_sha256": "a" * 64,
            }
            with self.assertRaisesRegex(
                queue.RoundtripQueueError,
                "merged cache row count mismatch",
            ):
                queue._validate_generation_report(
                    report,
                    config={
                        "eval": {
                            "eval_splits": ["validation"],
                            "validation_limit": 2,
                            "generation_only": True,
                        }
                    },
                    generated_jsonl=generated,
                )

    def test_checked_in_r27_guard_does_not_match_sync_filenames(self):
        queue = load_script("nano_roundtrip_queue")
        loaded = queue.load_queue(ROOT / "configs" / "nano_roundtrip" / "r27_baseline_queue.yaml")
        patterns = loaded["defaults"]["launch_guard"]["block_if_process_matches"]

        false_positive_lines = [
            "664733 00:00 bash -lc printf '%s' chunk >> /tmp/test_nano_av_probe_queue.py.b64",
            "664734 00:00 bash -lc printf '%s' chunk >> /tmp/nano_ar_hpo_queue.py.b64",
        ]
        real_process_lines = [
            "493173 01:16 /workspace/interp/.venv/bin/python scripts/nano_ar_hpo_queue.py configs/nano_ar/hpo/r33_dedup_clean_queue.yaml --run-until-empty",
            "493249 01:16 /workspace/interp/.venv/bin/python /workspace/interp/code/miles-051cd15/train.py --train-backend fsdp",
        ]

        self.assertEqual(queue.active_process_matches(false_positive_lines, patterns), [])
        self.assertEqual(len(queue.active_process_matches(real_process_lines, patterns)), 2)


if __name__ == "__main__":
    unittest.main()
