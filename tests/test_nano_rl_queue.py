import copy
import importlib.util
import hashlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
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


class NanoRLQueueTests(unittest.TestCase):
    def test_critic_update_mode_is_explicit_and_conflict_checked(self):
        queue = load_script("nano_rl_queue")

        self.assertEqual(queue._critic_update_mode({}, {}), "online")
        self.assertEqual(
            queue._critic_update_mode(
                {"critic_update_mode": "frozen"},
                {"NLA_FREEZE_CRITIC_TRAIN": "1"},
            ),
            "frozen",
        )
        self.assertEqual(
            queue._critic_update_mode(
                {"critic_update_mode": "online"},
                {"NLA_FREEZE_CRITIC_TRAIN": "0"},
            ),
            "online",
        )
        with self.assertRaisesRegex(queue.QueueError, "conflicts"):
            queue._critic_update_mode(
                {"critic_update_mode": "online"},
                {"NLA_FREEZE_CRITIC_TRAIN": "1"},
            )
        with self.assertRaisesRegex(queue.QueueError, "explicit boolean"):
            queue._critic_update_mode({}, {"NLA_FREEZE_CRITIC_TRAIN": "maybe"})

    def test_online_critic_requires_positive_learning_rate(self):
        queue = load_script("nano_rl_queue")

        with self.assertRaisesRegex(queue.QueueError, "must be positive"):
            queue._validate_training_runtime_config(
                {"critic_update_mode": "online", "critic_lr": 0},
                critic_update_mode="online",
            )

    def test_online_critic_retention_fraction_is_bounded(self):
        queue = load_script("nano_rl_queue")

        for value in (-0.01, 1.01, "invalid"):
            with self.subTest(value=value), self.assertRaises(queue.QueueError):
                queue._validate_training_runtime_config(
                    {
                        "critic_update_mode": "online",
                        "critic_lr": "2e-6",
                        "min_critic_retained_fraction": value,
                    },
                    critic_update_mode="online",
                )

        queue._validate_training_runtime_config(
            {
                "critic_update_mode": "online",
                "critic_lr": "2e-6",
                "min_critic_retained_fraction": 0.95,
            },
            critic_update_mode="online",
        )

    def test_queue_updates_are_bound_to_item_name_not_position(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "queue.yaml"
            common = {
                "status": "pending",
                "rl_parquet": "/data/rl.parquet",
                "instruct_model": "/models/base",
                "actor_sft_ckpt": "/runs/actor",
                "critic_sl_ckpt": "/runs/critic",
                "run_dir": "/runs/output",
            }
            path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "nano_rl_queue.v1",
                        "items": [
                            {"name": "first", **common},
                            {"name": "second", **common},
                        ],
                    },
                    sort_keys=False,
                )
            )

            queue.update_item(
                path,
                0,
                item_name="second",
                status="complete",
            )
            items = queue.load_queue(path)["items"]

        self.assertEqual(items[0]["status"], "pending")
        self.assertEqual(items[1]["status"], "complete")

    def test_immutable_launch_contract_rejects_config_mutation_and_redacts_secrets(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            queue_path = root / "queue.yaml"
            queue_doc = {
                "schema_version": "nano_rl_queue.v1",
                "defaults": {"training": {"actor_lr": "1e-5"}},
                "items": [{"name": "run", "status": "pending"}],
            }
            queue_path.write_text(yaml.safe_dump(queue_doc, sort_keys=False))
            spec = {
                "item_name": "run",
                "run_dir": str(root / "run"),
                "cwd": "/code",
                "command": ["bash", "train.sh"],
                "resource_total_gpus": 8,
                "rollout_batch_plan": {"global_batch_size": 384},
                "env": {
                    "WANDB_MODE": "offline",
                    "AWS_SECRET_ACCESS_KEY": "must-not-appear",
                },
                "source_provenance": {"runtime": {"sha256": "a" * 64}},
            }

            first = queue.freeze_launch_contract(
                queue_path=queue_path,
                queue_doc=queue_doc,
                item_index=0,
                spec=spec,
            )
            contract_text = pathlib.Path(first["contract_path"]).read_text()
            mutated = copy.deepcopy(queue_doc)
            mutated["defaults"]["training"]["actor_lr"] = "2e-5"

            self.assertNotIn("must-not-appear", contract_text)
            self.assertIn("<redacted>", contract_text)
            with self.assertRaisesRegex(queue.QueueError, "immutable launch contract mismatch"):
                queue.freeze_launch_contract(
                    queue_path=queue_path,
                    queue_doc=mutated,
                    item_index=0,
                    spec=spec,
                )

    def _write_queue(self, root: pathlib.Path, text: str) -> pathlib.Path:
        path = root / "queue.yaml"
        path.write_text(textwrap.dedent(text))
        return path

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def test_builds_smoke_command_from_yaml_without_script_edits(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            rl_parquet = root / "rl.parquet"
            actor = root / "actor"
            critic = root / "critic" / "hf"
            model = root / "model"
            for path in (rl_parquet,):
                path.write_text("placeholder")
            for path in (actor, critic, model):
                path.mkdir(parents=True)
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  python: /venv/bin/python
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  workspace_gpus: 4
                  env:
                    WANDB_MODE: offline
                  resources:
                    actor_gpus: 1
                    critic_gpus: 1
                    rollout_gpus: 2
                  training:
                    actor_lr: 1e-6
                    critic_lr: 1e-5
                    actor_micro_batch: 1
                    attn_implementation: eager
                    fsdp_reduce_dtype: bfloat16
                    fsdp_disable_backward_prefetch: true
                    kl_loss_coef: 0
                    kl_loss_type: k3
                    drift_guard:
                      enabled: true
                      max_logprob_abs_diff: 0.75
                      consecutive_steps: 2
                    save_interval: 1
                    finetune: true
                    no_load_optim: true
                  rollout:
                    rollout_batch_size: 8
                    global_batch_size: 8
                    n_samples_per_prompt: 2
                    max_response_len: 64
                    max_context_len: 256
                    num_rollout: 1
                  wandb:
                    project: nano30b-nla-pilot-test
                    group: rl-smoke-tests
                items:
                  - name: r33-rl-smoke
                    status: pending
                    rl_parquet: {rl_parquet}
                    instruct_model: {model}
                    actor_sft_ckpt: {actor}
                    critic_sl_ckpt: {critic}
                    run_dir: {root / "runs" / "r33-smoke"}
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            item = queue_doc["items"][0]
            with mock.patch.dict(
                os.environ,
                {
                    "NLA_SKIP_INJECTION": "1",
                    "KL_LOSS_TYPE": "k1",
                    "WANDB_MODE": "online",
                },
            ):
                spec = queue.build_run_spec(queue_doc, item, queue_path=queue_path)

        self.assertEqual(spec["resource_total_gpus"], 4)
        self.assertEqual(spec["cwd"], str(nla_root))
        self.assertEqual(spec["env"]["TRAIN_ENTRYPOINT"], str(miles_root / "train.py"))
        self.assertEqual(spec["env"]["RL_PARQUET"], str(rl_parquet))
        self.assertEqual(spec["env"]["ACTOR_GPUS"], "1")
        self.assertEqual(spec["env"]["CRITIC_GPUS"], "1")
        self.assertEqual(spec["env"]["ROLLOUT_GPUS"], "2")
        self.assertEqual(spec["env"]["ATTN_IMPLEMENTATION"], "eager")
        self.assertEqual(spec["env"]["FSDP_REDUCE_DTYPE"], "bfloat16")
        self.assertEqual(spec["env"]["FSDP_DISABLE_BACKWARD_PREFETCH"], "1")
        self.assertEqual(spec["env"]["KL_LOSS_TYPE"], "k3")
        self.assertNotIn("NLA_SKIP_INJECTION", spec["env"])
        self.assertEqual(
            spec["env"]["NLA_CUSTOM_TRAIN_GUARD_FUNCTION"],
            "nla.train_guard.check_train_metrics",
        )
        self.assertEqual(
            spec["env"]["NLA_TRAIN_GUARD_MAX_LOGPROB_ABS_DIFF"], "0.75"
        )
        self.assertEqual(spec["env"]["NLA_TRAIN_GUARD_CONSECUTIVE_STEPS"], "2")
        self.assertEqual(spec["env"]["WANDB_MODE"], "offline")
        self.assertEqual(spec["env"]["WANDB_DIR"], str(root / "runs" / "r33-smoke" / "wandb"))
        self.assertEqual(spec["env"]["WANDB_PROJECT"], "nano30b-nla-pilot-test")
        self.assertEqual(spec["env"]["WANDB_GROUP"], "rl-smoke-tests")
        self.assertEqual(spec["env"]["NLA_SYSTEM_METRICS"], "1")
        self.assertEqual(spec["env"]["NLA_SYSTEM_METRICS_INTERVAL_STEPS"], "1")
        self.assertEqual(spec["env"]["NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS"], "1")
        self.assertEqual(spec["env"]["NLA_PHASE_METRICS"], "1")
        self.assertEqual(spec["env"]["NLA_PHASE_METRICS_ALL_GPUS"], "1")
        self.assertEqual(spec["env"]["NLA_PHASE_METRICS_WANDB"], "1")
        self.assertIsNotNone(spec["source_provenance"])
        self.assertEqual(spec["env"]["FINETUNE"], "1")
        self.assertEqual(spec["env"]["NO_LOAD_OPTIM"], "1")
        self.assertEqual(spec["command"], [
            "bash",
            str(nla_root / "configs" / "rl.sh"),
            "--rollout-batch-size",
            "8",
            "--global-batch-size",
            "8",
            "--n-samples-per-prompt",
            "2",
            "--rollout-max-response-len",
            "64",
            "--rollout-max-context-len",
            "256",
            "--num-rollout",
            "1",
        ])
        self.assertNotIn("--sglang-context-length", spec["command"])

    def test_renders_high_throughput_topology_from_structured_sglang_config(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  python: /venv/bin/python
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  workspace_gpus: 8
                  resources:
                    actor_gpus: 5
                    critic_gpus: 2
                    rollout_gpus: 1
                    min_actor_gpus: 5
                  sglang:
                    mode: external
                    python: /venv/bin/python
                    engine_addrs: [127.0.0.1:31000]
                    managed: true
                    tensor_parallel_size: 1
                    base_gpu_id: 5
                    rollout_num_gpus_per_engine: 1
                    start_commands:
                    - - /venv/bin/python
                      - -m
                      - sglang.launch_server
                      - --model-path
                      - /models/rollout
                    health_urls:
                    - http://127.0.0.1:31000/health_generate
                items:
                  - name: tp1-actor5
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/tp1-actor5
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            spec = queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

        self.assertEqual(spec["resource_total_gpus"], 8)
        self.assertEqual(spec["env"]["ACTOR_GPUS"], "5")
        self.assertEqual(spec["env"]["CRITIC_GPUS"], "2")
        self.assertEqual(spec["env"]["ROLLOUT_GPUS"], "1")
        self.assertEqual(spec["env"]["NLA_WORKSPACE_GPUS"], "8")
        self.assertEqual(spec["env"]["NLA_ROLLOUT_GPUS_PER_ENGINE"], "1")
        self.assertEqual(spec["env"]["NLA_SGLANG_TP_SIZE"], "1")
        self.assertIn("--rollout-num-gpus-per-engine", spec["command"])
        self.assertEqual(spec["command"][spec["command"].index("--rollout-num-gpus-per-engine") + 1], "1")
        start_command = spec["sglang_service"]["start_commands"][0]
        self.assertEqual(start_command[start_command.index("--tp-size") + 1], "1")
        self.assertEqual(start_command[start_command.index("--base-gpu-id") + 1], "5")

    def test_rejects_sglang_topology_that_exceeds_rollout_gpu_allocation(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  workspace_gpus: 8
                  resources:
                    actor_gpus: 5
                    critic_gpus: 2
                    rollout_gpus: 1
                  sglang:
                    mode: external
                    engine_addrs: [127.0.0.1:31000]
                    tensor_parallel_size: 2
                    rollout_num_gpus_per_engine: 2
                items:
                  - name: bad-topology
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/bad-topology
                """,
            )

            queue_doc = queue.load_queue(queue_path)

        with self.assertRaisesRegex(queue.QueueError, "rollout GPU allocation"):
            queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

    def test_renders_structured_throughput_runtime_knobs(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('sync train')\n")
            (miles_root / "train_async.py").write_text("print('async train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  training:
                    async_training: true
                    gradient_checkpointing: false
                    offload_train: true
                    offload_rollout: true
                    offload_rollout_level: [kv_cache]
                    fsdp_cpu_offload: true
                    fsdp_cpu_backend: gloo
                    ref_log_probs_placement: actor
                items:
                  - name: throughput-knobs
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/throughput-knobs
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            spec = queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

        self.assertEqual(spec["env"]["TRAIN_ENTRYPOINT"], str(miles_root / "train_async.py"))
        self.assertEqual(spec["env"]["GRADIENT_CHECKPOINTING"], "0")
        self.assertEqual(spec["env"]["OFFLOAD_TRAIN"], "1")
        self.assertEqual(spec["env"]["OFFLOAD_ROLLOUT"], "1")
        self.assertEqual(spec["env"]["OFFLOAD_ROLLOUT_LEVEL"], "kv_cache")
        self.assertEqual(spec["env"]["FSDP_CPU_OFFLOAD"], "1")
        self.assertEqual(spec["env"]["FSDP_CPU_BACKEND"], "gloo")
        self.assertEqual(spec["env"]["NLA_REF_LOG_PROBS_PLACEMENT"], "actor")

    def test_rejects_async_colocate_combination_before_miles_assertion(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('sync train')\n")
            (miles_root / "train_async.py").write_text("print('async train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  training:
                    async_training: true
                    colocate: true
                items:
                  - name: bad-async-colocate
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/bad-async-colocate
                """,
            )

            queue_doc = queue.load_queue(queue_path)

        with self.assertRaisesRegex(queue.QueueError, "async_training.*colocate"):
            queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

    def test_rejects_ref_log_probs_on_critic_until_runtime_support_exists(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  training:
                    ref_log_probs_placement: critic
                items:
                  - name: unsupported-ref-placement
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/unsupported-ref-placement
                """,
            )

            queue_doc = queue.load_queue(queue_path)

        with self.assertRaisesRegex(queue.QueueError, "ref_log_probs_placement=critic"):
            queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

    def test_rejects_split_python_runtimes_for_live_external_weight_sync(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  python: /trainer/bin/python
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  workspace_gpus: 3
                  resources:
                    actor_gpus: 1
                    critic_gpus: 1
                    rollout_gpus: 1
                  sglang:
                    mode: external
                    python: /server/bin/python
                    managed: true
                    engine_addrs: [127.0.0.1:31000]
                    tensor_parallel_size: 1
                    base_gpu_id: 1
                    rollout_num_gpus_per_engine: 1
                    start_commands:
                      - [/server/bin/python, -m, sglang.launch_server]
                items:
                  - name: split-runtime-live-sync
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/split-runtime-live-sync
                """,
            )
            queue_doc = queue.load_queue(queue_path)

        with self.assertRaisesRegex(queue.QueueError, "unified Python runtime"):
            queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

    def test_allows_split_python_runtimes_when_weight_sync_is_skipped(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  python: /trainer/bin/python
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  workspace_gpus: 3
                  env:
                    NLA_SKIP_ROLLOUT_WEIGHT_SYNC: "1"
                  resources:
                    actor_gpus: 1
                    critic_gpus: 1
                    rollout_gpus: 1
                  sglang:
                    mode: external
                    python: /server/bin/python
                    managed: true
                    engine_addrs: [127.0.0.1:31000]
                    tensor_parallel_size: 1
                    base_gpu_id: 1
                    rollout_num_gpus_per_engine: 1
                    start_commands:
                      - [/server/bin/python, -m, sglang.launch_server]
                items:
                  - name: split-runtime-skip-sync
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/split-runtime-skip-sync
                """,
            )
            queue_doc = queue.load_queue(queue_path)
            spec = queue.build_run_spec(
                queue_doc,
                queue_doc["items"][0],
                queue_path=queue_path,
            )

        self.assertEqual(spec["env"]["NLA_SKIP_ROLLOUT_WEIGHT_SYNC"], "1")

    def test_null_save_interval_disables_miles_checkpoints(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  training:
                    save_interval: null
                items:
                  - name: no-final-save
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /runs/actor
                    critic_sl_ckpt: /runs/critic/hf
                    run_dir: {root / "runs" / "no-final-save"}
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            spec = queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

        self.assertEqual(spec["env"]["SAVE_INTERVAL"], "")
        self.assertNotIn("--save-interval", spec["command"])

    def test_reset_active_cli_exits_without_processing_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            queue_path = self._write_queue(
                root,
                """
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: /missing/code
                  python: /missing/python
                items:
                  - name: active-run
                    status: training
                    rl_parquet: /missing/rl.parquet
                    instruct_model: /missing/model
                    actor_sft_ckpt: /missing/actor
                    critic_sl_ckpt: /missing/critic
                    run_dir: /missing/run
                """,
            )

            result = subprocess.run(
                [sys.executable, str(SCRIPTS / "nano_rl_queue.py"), str(queue_path), "--reset-active"],
                text=True,
                capture_output=True,
            )
            queue_doc = yaml.safe_load(queue_path.read_text())

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(queue_doc["items"][0]["status"], "pending")
        self.assertNotIn("failure", queue_doc["items"][0])

    def test_actor_load_and_sidecar_paths_can_be_separate(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                items:
                  - name: split-actor-paths
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /runs/actor/checkpoints/iter_0000123
                    actor_load_ckpt: /runs/actor/checkpoints
                    actor_ref_ckpt: /runs/actor/ref-checkpoints
                    actor_sidecar_source: /runs/actor/checkpoints/iter_0000123
                    critic_sl_ckpt: /runs/critic/hf
                    run_dir: {root / "runs" / "split-actor-paths"}
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            spec = queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

        self.assertEqual(spec["env"]["ACTOR_SFT_CKPT"], "/runs/actor/checkpoints/iter_0000123")
        self.assertEqual(spec["env"]["ACTOR_LOAD_CKPT"], "/runs/actor/checkpoints")
        self.assertEqual(spec["env"]["ACTOR_REF_CKPT"], "/runs/actor/ref-checkpoints")
        self.assertEqual(spec["env"]["ACTOR_SIDECAR_SOURCE"], "/runs/actor/checkpoints/iter_0000123")

    def test_training_signal_controls_are_config_driven(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  training:
                    advantage_estimator: reinforce_plus_plus_baseline
                    normalize_advantages: true
                    rewards_normalization: false
                    grpo_std_normalization: false
                    kl_loss_coef: 0.0003
                items:
                  - name: signal-controls
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /runs/actor
                    critic_sl_ckpt: /runs/critic/hf
                    run_dir: {root / "runs" / "signal-controls"}
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            spec = queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

        self.assertEqual(spec["env"]["ADVANTAGE_ESTIMATOR"], "reinforce_plus_plus_baseline")
        self.assertEqual(spec["env"]["NORMALIZE_ADVANTAGES"], "1")
        self.assertEqual(spec["env"]["REWARDS_NORMALIZATION"], "0")
        self.assertEqual(spec["env"]["GRPO_STD_NORMALIZATION"], "0")
        self.assertEqual(spec["env"]["KL_LOSS_COEF"], "0.0003")

    def test_rollout_batch_plan_tracks_full_rollout_global_batch(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  rollout:
                    rollout_batch_size: 4
                    n_samples_per_prompt: 8
                    global_batch_size: 32
                    require_global_batch_match: true
                items:
                  - name: qwen-style-batch
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /runs/actor
                    critic_sl_ckpt: /runs/critic/hf
                    run_dir: {root / "runs" / "qwen-style-batch"}
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            spec = queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

        self.assertEqual(spec["rollout_batch_plan"]["generated_samples"], 32)
        self.assertTrue(spec["rollout_batch_plan"]["global_batch_matches_rollout"])
        self.assertEqual(spec["env"]["NLA_ROLLOUT_PROMPT_BATCH"], "4")
        self.assertEqual(spec["env"]["NLA_ROLLOUT_SAMPLES_PER_PROMPT"], "8")
        self.assertEqual(spec["env"]["NLA_ROLLOUT_GENERATED_SAMPLES"], "32")
        self.assertEqual(spec["env"]["NLA_ROLLOUT_GLOBAL_BATCH"], "32")
        self.assertEqual(spec["env"]["NLA_ROLLOUT_GLOBAL_MATCH"], "1")

    def test_strict_rollout_batch_plan_rejects_partial_global_batch(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  rollout:
                    rollout_batch_size: 2
                    n_samples_per_prompt: 2
                    global_batch_size: 2
                    require_global_batch_match: true
                items:
                  - name: partial-batch
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /runs/actor
                    critic_sl_ckpt: /runs/critic/hf
                    run_dir: {root / "runs" / "partial-batch"}
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            with self.assertRaisesRegex(queue.QueueError, "global_batch_size must equal"):
                queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

    def test_roundtrip_post_eval_spec_builds_converter_and_baseline_eval_commands(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            scripts_root = code_root / "scripts"
            tools_root = nla_root / "tools"
            (nla_root / "configs").mkdir(parents=True)
            tools_root.mkdir(parents=True)
            scripts_root.mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (tools_root / "convert_fsdp_to_hf.py").write_text("print('convert')\n")
            (scripts_root / "eval_nano_av_ar_roundtrip_gate.py").write_text("print('eval')\n")
            (miles_root / "train.py").write_text("print('train')\n")
            run_dir = root / "runs" / "post-eval"
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  python: /train/python
                  post_eval_python: /eval/python
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  rollout:
                    num_rollout: 8
                  post_eval:
                    roundtrip:
                      enabled: true
                      origin_hf_dir: /models/base
                      ar_checkpoint_dir: /runs/ar/hf
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      test_parquet: /data/test.parquet
                      baseline_report_json: /reports/sft_baseline.json
                      eval_splits: [validation]
                      validation_limit: 64
                      test_limit: 64
                      max_new_tokens: 128
                      generation_backend: cache
                      generation_workers: 4
                      generation_worker_devices: ["0", "1", "2", "3"]
                      generation_controls: [real, mean, shuffled]
                      generated_text_fallback: raw
                      stop_text: "</explanation>"
                      seed: 20260709
                      av_model_fingerprint: hf_model_sha256:{"a" * 64}
                      av_tokenizer_fingerprint: tokenizer_files_sha256:{"b" * 64}
                      min_control_win_fraction: 0.9
                      min_closed_fraction: 0.8
                      min_usable_fraction: 0.95
                      cleanup_hf: true
                      cleanup_actor_checkpoint: true
                items:
                  - name: post-eval
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /runs/actor
                    critic_sl_ckpt: /runs/critic/hf
                    run_dir: {run_dir}
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            item = queue_doc["items"][0]
            run_spec = queue.build_run_spec(queue_doc, item, queue_path=queue_path)
            eval_spec = queue.build_roundtrip_post_eval_spec(queue_doc, item, run_spec, queue_path=queue_path)

        self.assertIsNotNone(eval_spec)
        assert eval_spec is not None
        self.assertEqual(eval_spec["iter_name"], "iter_0000008")
        self.assertEqual(eval_spec["input_dir"], str(run_dir / "actor" / "iter_0000008"))
        self.assertEqual(eval_spec["hf_output_dir"], str(run_dir / "hf_iter_0000008_tmp"))
        self.assertEqual(eval_spec["report_json"], str(run_dir / "roundtrip_iter_0000008_v64_t64_report.json"))
        self.assertEqual(eval_spec["generated_jsonl"], str(run_dir / "roundtrip_iter_0000008_v64_t64_report_generated.jsonl"))
        self.assertEqual(eval_spec["converter_command"][:2], ["/eval/python", str(tools_root / "convert_fsdp_to_hf.py")])
        self.assertIn("--origin-hf-dir", eval_spec["converter_command"])
        self.assertEqual(eval_spec["converter_command"][eval_spec["converter_command"].index("--origin-hf-dir") + 1], "/models/base")
        self.assertEqual(
            eval_spec["remote_code_patch_command"],
            ["/eval/python", "-m", "nla.remote_code_patches", str(run_dir / "hf_iter_0000008_tmp")],
        )
        eval_command = eval_spec["eval_command"]
        self.assertEqual(eval_command[:2], ["/eval/python", str(scripts_root / "eval_nano_av_ar_roundtrip_gate.py")])
        self.assertEqual(eval_command[eval_command.index("--av-hf-checkpoint") + 1], str(run_dir / "hf_iter_0000008_tmp"))
        self.assertEqual(eval_command[eval_command.index("--baseline-report-json") + 1], "/reports/sft_baseline.json")
        self.assertEqual(eval_command[eval_command.index("--generation-backend") + 1], "cache")
        self.assertEqual(eval_command[eval_command.index("--generation-workers") + 1], "4")
        self.assertEqual(eval_command[eval_command.index("--seed") + 1], "20260709")
        self.assertEqual(
            eval_command[eval_command.index("--stop-text") + 1],
            "</explanation>",
        )
        self.assertEqual(
            eval_command[eval_command.index("--av-model-fingerprint") + 1],
            "hf_model_sha256:" + "a" * 64,
        )
        self.assertEqual(
            eval_command[eval_command.index("--av-tokenizer-fingerprint") + 1],
            "tokenizer_files_sha256:" + "b" * 64,
        )
        self.assertEqual(
            eval_command[
                eval_command.index("--eval-splits") + 1 :
                eval_command.index("--eval-splits") + 2
            ],
            ["validation"],
        )
        self.assertIn("--stream-generated", eval_command)
        self.assertEqual(eval_command[eval_command.index("--min-control-win-fraction") + 1], "0.9")
        self.assertTrue(eval_spec["cleanup_hf"])
        self.assertTrue(eval_spec["cleanup_actor_checkpoint"])

    def test_roundtrip_identity_auto_fingerprints_once_and_injects_all_commands(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            hf = root / "hf"
            origin = root / "origin"
            hf.mkdir()
            origin.mkdir()
            (hf / "config.json").write_text("{}\n")
            (hf / "model.safetensors").write_bytes(b"weights")
            (origin / "tokenizer.json").write_text('{"version":"1"}\n')
            report = root / "identity.json"
            base_command = [
                "/python",
                "/eval.py",
                "--require-generation-protocol-match",
            ]
            base_spec = {
                "iter_name": "iter_0000002",
                "hf_output_dir": str(hf),
                "origin_hf_dir": str(origin),
                "report_json": str(root / "report.json"),
                "fingerprint_report_json": str(report),
                "fingerprint_workers": 2,
                "eval_command": list(base_command),
                "generation_command": [*base_command, "--generation-only"],
                "score_command": [*base_command, "--reuse-generated"],
            }

            spec = copy.deepcopy(base_spec)
            first = queue._resolve_roundtrip_generation_identity(spec)
            cached_spec = copy.deepcopy(base_spec)
            with mock.patch.object(
                queue,
                "fingerprint_hf_model_files",
                side_effect=AssertionError("cached identity should be reused"),
            ):
                second = queue._resolve_roundtrip_generation_identity(cached_spec)
            report_exists = report.is_file()

        assert first is not None
        assert second is not None
        self.assertTrue(report_exists)
        for command_name in ("eval_command", "generation_command", "score_command"):
            command = spec[command_name]
            self.assertIn("--av-model-fingerprint", command)
            self.assertIn("--av-tokenizer-fingerprint", command)
        self.assertEqual(
            first["av_model_fingerprint"], second["av_model_fingerprint"]
        )

    def test_reusable_post_eval_hf_requires_complete_index(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "config.json").write_text("{}\n")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map":{"a":"missing.safetensors"}}\n'
            )

            with self.assertRaisesRegex(queue.QueueError, "incomplete"):
                queue._validate_hf_checkpoint_for_reuse(root)

    def test_roundtrip_eval_script_resolves_under_remote_code_root_without_local_stat(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = pathlib.Path("/workspace/interp/code/nano30b-nla-pilot-current")
            miles_root = pathlib.Path("/workspace/interp/code/miles-051cd15")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  python: /workspace/interp/.venvs/sglang-cu130/bin/python
                  post_eval_python: /workspace/interp/.venv/bin/python
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  rollout:
                    num_rollout: 8
                  post_eval:
                    roundtrip:
                      enabled: true
                      ar_checkpoint_dir: /runs/ar/hf
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      test_parquet: /data/test.parquet
                items:
                  - name: remote-code-root
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /runs/actor
                    critic_sl_ckpt: /runs/critic/hf
                    run_dir: /runs/rl
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            item = queue_doc["items"][0]
            run_spec = queue.build_run_spec(queue_doc, item, queue_path=queue_path)
            eval_spec = queue.build_roundtrip_post_eval_spec(queue_doc, item, run_spec, queue_path=queue_path)

        assert eval_spec is not None
        self.assertEqual(
            eval_spec["eval_command"][1],
            "/workspace/interp/code/nano30b-nla-pilot-current/scripts/eval_nano_av_ar_roundtrip_gate.py",
        )
        self.assertEqual(
            eval_spec["converter_command"][1],
            "/workspace/interp/code/nano30b-nla-pilot-current/external/natural_language_autoencoders/tools/convert_fsdp_to_hf.py",
        )

    def test_roundtrip_can_convert_and_score_online_critic_checkpoint(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            tools_root = nla_root / "tools"
            scripts_root = code_root / "scripts"
            tools_root.mkdir(parents=True)
            scripts_root.mkdir(parents=True)
            (nla_root / "configs").mkdir(parents=True)
            (nla_root / "configs" / "rl.sh").write_text("#!/bin/bash\n")
            (tools_root / "convert_fsdp_to_hf.py").write_text("# converter\n")
            (scripts_root / "eval_nano_av_ar_roundtrip_gate.py").write_text("# eval\n")
            run_dir = root / "run"
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  nla_root: {nla_root}
                  python: /eval/python
                  rollout:
                    rollout_batch_size: 1
                    global_batch_size: 1
                    n_samples_per_prompt: 1
                    num_rollout: 2
                  post_eval:
                    roundtrip:
                      enabled: true
                      origin_hf_dir: /models/base
                      critic_input_dir: {run_dir / "critic" / "iter_0000002"}
                      convert_critic_checkpoint: true
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      eval_splits: [validation]
                items:
                  - name: online-critic-eval
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /runs/actor
                    critic_sl_ckpt: /runs/critic/hf
                    run_dir: {run_dir}
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            item = queue_doc["items"][0]
            run_spec = queue.build_run_spec(queue_doc, item, queue_path=queue_path)
            eval_spec = queue.build_roundtrip_post_eval_spec(
                queue_doc,
                item,
                run_spec,
                queue_path=queue_path,
            )

        assert eval_spec is not None
        critic_hf = run_dir / "critic_hf_iter_0000002_tmp"
        self.assertEqual(
            eval_spec["critic_converter_command"][
                eval_spec["critic_converter_command"].index("--input-dir") + 1
            ],
            str(run_dir / "critic" / "iter_0000002"),
        )
        self.assertEqual(
            eval_spec["eval_command"][
                eval_spec["eval_command"].index("--ar-checkpoint-dir") + 1
            ],
            str(critic_hf),
        )
        self.assertTrue(eval_spec["cleanup_critic_hf"])

    def test_post_eval_only_retries_named_failed_item_without_training(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            scripts_root = code_root / "scripts"
            tools_root = nla_root / "tools"
            (nla_root / "configs").mkdir(parents=True)
            tools_root.mkdir(parents=True)
            scripts_root.mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (tools_root / "convert_fsdp_to_hf.py").write_text("print('convert')\n")
            (scripts_root / "eval_nano_av_ar_roundtrip_gate.py").write_text("print('eval')\n")
            (miles_root / "train.py").write_text("print('train')\n")
            run_dir = root / "runs" / "retry-post-eval"
            (run_dir / "actor" / "iter_0000008").mkdir(parents=True)
            stale_hf_dir = run_dir / "hf_iter_0000008_tmp"
            stale_hf_dir.mkdir(parents=True)
            (stale_hf_dir / "stale.txt").write_text("retry must rebuild this directory\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  python: /train/python
                  post_eval_python: /eval/python
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  rollout:
                    num_rollout: 8
                  post_eval:
                    roundtrip:
                      enabled: true
                      ar_checkpoint_dir: /runs/ar/hf
                      train_parquet: /data/train.parquet
                      validation_parquet: /data/validation.parquet
                      test_parquet: /data/test.parquet
                items:
                  - name: retry-post-eval
                    status: failed
                    failure: old post-eval failure
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /runs/actor
                    critic_sl_ckpt: /runs/critic/hf
                    run_dir: {run_dir}
                """,
            )
            launch_queue = queue.load_queue(queue_path)
            launch_item = launch_queue["items"][0]
            launch_spec = queue.build_run_spec(
                launch_queue, launch_item, queue_path=queue_path
            )
            launch_spec["post_eval_specs"] = queue.build_roundtrip_post_eval_specs(
                launch_queue,
                launch_item,
                launch_spec,
                queue_path=queue_path,
            )
            launch_contract = queue.freeze_launch_contract(
                queue_path=queue_path,
                queue_doc=launch_queue,
                item_index=0,
                spec=launch_spec,
            )
            queue.update_item(queue_path, 0, launch_contract=launch_contract)
            edited_queue = queue.load_queue(queue_path)
            edited_queue["defaults"]["post_eval"]["roundtrip"][
                "validation_parquet"
            ] = "/data/edited-after-launch.parquet"
            queue.write_queue(queue_path, edited_queue)

            with mock.patch.object(
                queue,
                "run_roundtrip_post_eval",
                return_value={"status": "complete", "gate_passed": True},
            ) as run_post_eval:
                result = queue.process_post_eval_only(queue_path, "retry-post-eval")

            updated = queue.load_queue(queue_path)["items"][0]

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["mode"], "post_eval_only")
        self.assertEqual(updated["status"], "complete")
        self.assertEqual(updated["post_eval_retry_previous_status"], "failed")
        self.assertEqual(updated["post_eval_retry_previous_failure"], "old post-eval failure")
        self.assertTrue(updated["post_eval_retry_hf_cleaned_before_run"])
        self.assertFalse(stale_hf_dir.exists())
        run_post_eval.assert_called_once()
        frozen_spec = run_post_eval.call_args.args[4]
        self.assertEqual(
            frozen_spec["post_eval_specs"][0]["required_paths"][
                "validation_parquet"
            ],
            "/data/validation.parquet",
        )

    def test_rl_shell_omits_save_interval_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            capture_path = root / "argv.json"
            fake_python = root / "fake_python.py"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "if len(sys.argv) > 1 and sys.argv[1] == '-c':\n"
                "    raise SystemExit(1)\n"
                f"pathlib.Path({str(capture_path)!r}).write_text(json.dumps(sys.argv))\n"
            )
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": str(fake_python),
                    "TRAIN_ENTRYPOINT": str(root / "train.py"),
                    "RL_PARQUET": "/data/rl.parquet",
                    "INSTRUCT_MODEL": "/models/base",
                    "ACTOR_SFT_CKPT": "/runs/actor",
                    "ACTOR_LOAD_CKPT": "/runs/actor/checkpoints",
                    "ACTOR_REF_CKPT": "/runs/actor/ref-checkpoints",
                    "ACTOR_SIDECAR_SOURCE": "/runs/actor/iter_0000001",
                    "CRITIC_SL_CKPT": "/runs/critic/hf",
                    "RUN_DIR": str(root / "run"),
                    "KL_LOSS_COEF": "0",
                    "WANDB_MODE": "disabled",
                    "SAVE_INTERVAL": "",
                    "FINETUNE": "1",
                    "NO_LOAD_OPTIM": "1",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "NLA_EMBED_DUMP_DIR": str(root / "shm"),
                    "NLA_CRITIC_REWARD_LAYOUT_MSE_RATIO_TOL": "0.03",
                    "NLA_CRITIC_TRAIN_MODE_MSE_RATIO_TOL": "0.07",
                    "NLA_SYSTEM_METRICS": "1",
                    "NLA_SYSTEM_METRICS_INTERVAL_STEPS": "1",
                    "NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS": "1",
                }
            )

            subprocess.run(
                ["bash", str(ROOT / "external/natural_language_autoencoders/configs/rl.sh")],
                check=True,
                env=env,
                cwd=ROOT,
            )
            argv = json.loads(capture_path.read_text())

        self.assertNotIn("--save-interval", argv)
        self.assertIn("--finetune", argv)
        self.assertIn("--no-load-optim", argv)
        self.assertEqual(argv[argv.index("--ref-load") + 1], "/runs/actor/ref-checkpoints")
        self.assertEqual(argv[argv.index("--load") + 1], "/runs/actor/checkpoints")
        self.assertEqual(argv[argv.index("--nla-sidecar-source") + 1], "/runs/actor/iter_0000001")
        train_env_vars = json.loads(argv[argv.index("--train-env-vars") + 1])
        self.assertEqual(train_env_vars["PYTORCH_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(train_env_vars["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(train_env_vars["NLA_SYSTEM_METRICS"], "1")
        self.assertEqual(train_env_vars["NLA_SYSTEM_METRICS_INTERVAL_STEPS"], "1")
        self.assertEqual(train_env_vars["NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS"], "1")
        self.assertEqual(train_env_vars["NLA_PHASE_METRICS"], "1")
        self.assertEqual(train_env_vars["NLA_PHASE_METRICS_ALL_GPUS"], "1")
        self.assertEqual(train_env_vars["NLA_PHASE_METRICS_WANDB"], "1")
        self.assertEqual(train_env_vars["NLA_CRITIC_REWARD_LAYOUT_MSE_RATIO_TOL"], "0.03")
        self.assertEqual(train_env_vars["NLA_CRITIC_TRAIN_MODE_MSE_RATIO_TOL"], "0.07")

    def test_external_sglang_service_renders_rpc_boundary(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  python: /workspace/interp/.venv/bin/python
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  workspace_gpus: 3
                  resources:
                    actor_gpus: 1
                    critic_gpus: 1
                    rollout_gpus: 1
                  sglang:
                    mode: external
                    python: /workspace/interp/.venvs/sglang-cu130/bin/python
                    engine_addrs:
                    - 127.0.0.1:31000
                    router_addr: 127.0.0.1:32000
                    managed: false
                    health_urls:
                    - http://127.0.0.1:31000/health_generate
                items:
                  - name: external-sglang
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/external
                """,
            )

            queue_doc = queue.load_queue(queue_path)
            spec = queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

        self.assertEqual(spec["env"]["NANO_SGLANG_MODE"], "external")
        self.assertEqual(spec["env"]["NANO_SGLANG_PYTHON"], "/workspace/interp/.venvs/sglang-cu130/bin/python")
        self.assertIn("--rollout-external", spec["command"])
        self.assertEqual(
            spec["command"][spec["command"].index("--rollout-external-engine-addrs") + 1],
            "127.0.0.1:31000",
        )
        self.assertEqual(spec["command"][spec["command"].index("--sglang-router-ip") + 1], "127.0.0.1")
        self.assertEqual(spec["command"][spec["command"].index("--sglang-router-port") + 1], "32000")
        self.assertEqual(spec["sglang_service"]["mode"], "external")
        self.assertFalse(spec["sglang_service"]["managed"])
        self.assertEqual(spec["sglang_service"]["health_urls"], ["http://127.0.0.1:31000/health_generate"])

    def test_managed_external_sglang_service_starts_before_training_and_stops(self):
        queue = load_script("nano_rl_queue")
        port = self._free_port()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            run_dir = root / "run"
            rl_parquet = root / "data" / "rl.parquet"
            instruct_model = root / "models" / "base"
            actor = root / "ckpts" / "actor"
            critic = root / "ckpts" / "critic" / "hf"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            rl_parquet.parent.mkdir(parents=True)
            rl_parquet.write_text("placeholder")
            for path in (instruct_model, actor, critic):
                path.mkdir(parents=True)
            (nla_root / "configs" / "rl.sh").write_text(
                "#!/usr/bin/env bash\n"
                "mkdir -p \"$RUN_DIR\"\n"
                "printf '%s\\n' \"$NANO_SGLANG_MODE\" > \"$RUN_DIR/train_saw_mode.txt\"\n"
            )
            (miles_root / "train.py").write_text("print('unused')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  python: {sys.executable}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  workspace_gpus: 3
                  resources:
                    actor_gpus: 1
                    critic_gpus: 1
                    rollout_gpus: 1
                  sglang:
                    mode: external
                    python: {sys.executable}
                    engine_addrs:
                    - 127.0.0.1:{port}
                    managed: true
                    terminate_on_exit: true
                    timeout_seconds: 15
                    poll_seconds: 0.25
                    start_commands:
                    - - {sys.executable}
                      - -m
                      - http.server
                      - "{port}"
                      - --bind
                      - 127.0.0.1
                    health_urls:
                    - http://127.0.0.1:{port}/
                items:
                  - name: managed-external-sglang
                    status: pending
                    rl_parquet: {rl_parquet}
                    instruct_model: {instruct_model}
                    actor_sft_ckpt: {actor}
                    critic_sl_ckpt: {critic}
                    run_dir: {run_dir}
                """,
            )

            queue.set_item_approval(
                queue_path,
                "managed-external-sglang",
                approved=True,
                approved_by="test",
            )
            result = queue.process_next_item(queue_path)

            self.assertEqual(result["status"], "complete")
            self.assertEqual((run_dir / "train_saw_mode.txt").read_text().strip(), "external")
            pid_report = json.loads((run_dir / "sglang_service_pids.json").read_text())

        self.assertEqual(len(pid_report["processes"]), 1)
        self.assertEqual(pid_report["processes"][0]["status"], "terminated")

    def test_managed_external_sglang_service_can_stage_model_before_start(self):
        queue = load_script("nano_rl_queue")
        port = self._free_port()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            run_dir = root / "run"
            rl_parquet = root / "data" / "rl.parquet"
            instruct_model = root / "models" / "base"
            actor = root / "ckpts" / "actor"
            critic = root / "ckpts" / "critic" / "hf"
            source_model = root / "rollout_hf"
            staged_model = root / "staged" / "rollout_hf"
            capture_path = root / "service_argv.json"
            fake_service = root / "fake_sglang_service.py"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            rl_parquet.parent.mkdir(parents=True)
            rl_parquet.write_text("placeholder")
            for path in (instruct_model, actor, critic, source_model):
                path.mkdir(parents=True)
            (source_model / "config.json").write_text("{}\n")
            (source_model / "weights.safetensors").write_bytes(b"tiny")
            (source_model / "remote_code").mkdir()
            (source_model / "remote_code" / "modeling_test.py").write_text("MODEL = 'test'\n")
            fake_service.write_text(
                "import argparse, http.server, json, os, pathlib, socketserver, sys\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--model-path', required=True)\n"
                "parser.add_argument('--port', type=int, required=True)\n"
                "args = parser.parse_args()\n"
                "pathlib.Path(os.environ['CAPTURE_PATH']).write_text(json.dumps({\n"
                "    'argv': sys.argv,\n"
                "    'model_path': args.model_path,\n"
                "    'staged_env': os.environ.get('NLA_SGLANG_STAGED_MODEL_PATH'),\n"
                "}))\n"
                "handler = http.server.SimpleHTTPRequestHandler\n"
                "with socketserver.TCPServer(('127.0.0.1', args.port), handler) as httpd:\n"
                "    httpd.serve_forever()\n"
            )
            (nla_root / "configs" / "rl.sh").write_text(
                "#!/usr/bin/env bash\n"
                "mkdir -p \"$RUN_DIR\"\n"
                "printf '%s\\n' \"$NLA_SGLANG_MODE\" > \"$RUN_DIR/train_saw_mode.txt\"\n"
            )
            (miles_root / "train.py").write_text("print('unused')\n")
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  python: {sys.executable}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  workspace_gpus: 3
                  resources:
                    actor_gpus: 1
                    critic_gpus: 1
                    rollout_gpus: 1
                  sglang:
                    mode: external
                    python: {sys.executable}
                    engine_addrs:
                    - 127.0.0.1:{port}
                    managed: true
                    terminate_on_exit: true
                    timeout_seconds: 15
                    poll_seconds: 0.25
                    env:
                      CAPTURE_PATH: {capture_path}
                    model_staging:
                      enabled: true
                      source_model_path: {source_model}
                      target_path: {staged_model}
                      copy_workers: 4
                    start_commands:
                    - - {sys.executable}
                      - {fake_service}
                      - --model-path
                      - {source_model}
                      - --port
                      - "{port}"
                    health_urls:
                    - http://127.0.0.1:{port}/
                items:
                  - name: managed-external-sglang-stage-model
                    status: pending
                    rl_parquet: {rl_parquet}
                    instruct_model: {instruct_model}
                    actor_sft_ckpt: {actor}
                    critic_sl_ckpt: {critic}
                    run_dir: {run_dir}
                """,
            )

            queue.set_item_approval(
                queue_path,
                "managed-external-sglang-stage-model",
                approved=True,
                approved_by="test",
            )
            result = queue.process_next_item(queue_path)
            captured = json.loads(capture_path.read_text())
            pid_report = json.loads((run_dir / "sglang_service_pids.json").read_text())
            staged_has_config = (staged_model / "config.json").exists()
            staged_has_weights = (staged_model / "weights.safetensors").exists()
            staged_has_remote_code = (staged_model / "remote_code" / "modeling_test.py").exists()
            staging_report = json.loads((run_dir / "sglang_model_staging.json").read_text())

        self.assertEqual(result["status"], "complete")
        self.assertEqual(captured["model_path"], str(staged_model))
        self.assertEqual(captured["staged_env"], str(staged_model))
        self.assertTrue(staged_has_config)
        self.assertTrue(staged_has_weights)
        self.assertTrue(staged_has_remote_code)
        self.assertEqual(staging_report["copy_workers"], 4)
        self.assertEqual(
            pid_report["processes"][0]["command"][pid_report["processes"][0]["command"].index("--model-path") + 1],
            str(staged_model),
        )

    def test_rejects_resource_request_above_workspace_gpu_count(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            queue_path = self._write_queue(
                root,
                """
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: /code
                  python: /venv/bin/python
                  workspace_gpus: 4
                  resources:
                    actor_gpus: 2
                    critic_gpus: 2
                    rollout_gpus: 2
                items:
                  - name: too-large
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/too-large
                """,
            )

            queue_doc = queue.load_queue(queue_path)

        with self.assertRaisesRegex(queue.QueueError, "requires 6 GPUs but workspace_gpus is 4"):
            queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

    def test_rejects_queue_below_declared_min_actor_gpus(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            queue_path = self._write_queue(
                root,
                """
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: /code
                  python: /venv/bin/python
                  workspace_gpus: 3
                  resources:
                    actor_gpus: 1
                    critic_gpus: 1
                    rollout_gpus: 1
                    min_actor_gpus: 2
                items:
                  - name: undersharded-actor
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/undersharded
                """,
            )

            queue_doc = queue.load_queue(queue_path)

        with self.assertRaisesRegex(queue.QueueError, "requires at least 2 actor GPUs but configured 1"):
            queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

    def test_status_reports_pending_and_terminal_items(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            queue_path = self._write_queue(
                root,
                """
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: /code
                  python: /venv/bin/python
                items:
                  - name: pending-one
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/pending
                  - name: complete-one
                    status: complete
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic/hf
                    run_dir: /runs/complete
                """,
            )

            status = queue.queue_status(queue_path)

        self.assertEqual(status["counts"]["pending"], 1)
        self.assertEqual(status["counts"]["complete"], 1)
        self.assertIsNone(status["next_pending"])

    def test_checked_in_r33_3gpu_smoke_queue_is_blocked_after_actor_oom(self):
        queue = load_script("nano_rl_queue")
        queue_path = ROOT / "configs" / "nano_rl" / "r33_component_full_smoke_queue.yaml"

        queue_doc = queue.load_queue(queue_path)
        item = queue_doc["items"][0]

        self.assertEqual(item["status"], "blocked")
        self.assertIn("single actor H200", item["failure"])
        with self.assertRaisesRegex(queue.QueueError, "requires at least 2 actor GPUs but configured 1"):
            queue.build_run_spec(queue_doc, item, queue_path=queue_path)

    def test_checked_in_r33_smoke_queue_targets_4_gpu_trainable_topology(self):
        queue = load_script("nano_rl_queue")
        queue_path = ROOT / "configs" / "nano_rl" / "r33_component_full_smoke_queue_4h200.yaml"

        queue_doc = queue.load_queue(queue_path)
        item = queue_doc["items"][0]
        spec = queue.build_run_spec(queue_doc, item, queue_path=queue_path)

        self.assertIn(item["status"], {"pending", "running", "complete", "failed"})
        self.assertEqual(spec["resource_total_gpus"], 4)
        self.assertEqual(spec["env"]["ACTOR_GPUS"], "2")
        self.assertEqual(spec["env"]["CRITIC_GPUS"], "1")
        self.assertEqual(spec["env"]["ROLLOUT_GPUS"], "1")
        self.assertEqual(spec["env"]["ATTN_IMPLEMENTATION"], "eager")
        self.assertEqual(spec["env"]["NLA_WEIGHT_UPDATE_LOG_EVERY"], "100")
        self.assertEqual(spec["env"]["NLA_SKIP_ROLLOUT_WEIGHT_SYNC"], "1")
        self.assertEqual(spec["env"]["NANO_SGLANG_MODE"], "external")
        self.assertEqual(spec["env"]["NANO_SGLANG_PYTHON"], "/workspace/interp/.venvs/sglang-cu130/bin/python")
        self.assertTrue(spec["sglang_service"]["managed"])
        self.assertIn("/workspace/interp/.venvs/sglang-cu130/bin/python", spec["sglang_service"]["start_commands"][0])
        self.assertIn("--base-gpu-id", spec["sglang_service"]["start_commands"][0])
        self.assertIn("2", spec["sglang_service"]["start_commands"][0])
        self.assertIn("--attention-backend", spec["sglang_service"]["start_commands"][0])
        self.assertIn("fa3", spec["sglang_service"]["start_commands"][0])
        self.assertIn("actor_sft_hf_iter_0001291", " ".join(spec["sglang_service"]["start_commands"][0]))
        self.assertIn("rl_R33_fullscan275396_smoke512.parquet", spec["env"]["RL_PARQUET"])
        self.assertIn("iter_0001291", spec["env"]["ACTOR_SFT_CKPT"])
        self.assertIn("iter_0001289/hf", spec["env"]["CRITIC_SL_CKPT"])
        self.assertIn("--rollout-external", spec["command"])

    def test_checked_in_r33_signal_ladder_uses_hf_ref_for_kl(self):
        queue = load_script("nano_rl_queue")
        queue_path = ROOT / "configs" / "nano_rl" / "r33_component_full_signal_ladder_queue_4h200_len512.yaml"

        queue_doc = queue.load_queue(queue_path)
        for item in queue_doc["items"]:
            with self.subTest(item=item["name"]):
                spec = queue.build_run_spec(queue_doc, item, queue_path=queue_path)
                self.assertIn("actor_sft_hf_iter_0001291", spec["env"]["ACTOR_REF_CKPT"])
                self.assertNotEqual(spec["env"]["ACTOR_LOAD_CKPT"], spec["env"]["ACTOR_REF_CKPT"])
                self.assertNotEqual(spec["env"]["KL_LOSS_COEF"], "0")
                self.assertTrue(str(item["run_dir"]).endswith("_refhf"))

    def test_checked_in_r33_8gpu_ladder_targets_large_batch_topology(self):
        queue = load_script("nano_rl_queue")
        queue_path = ROOT / "configs" / "nano_rl" / "r33_component_full_signal_ladder_queue_8h200_len512.yaml"

        queue_doc = queue.load_queue(queue_path)
        item = queue_doc["items"][0]
        spec = queue.build_run_spec(queue_doc, item, queue_path=queue_path)

        self.assertEqual(spec["resource_total_gpus"], 8)
        self.assertEqual(spec["env"]["ACTOR_GPUS"], "4")
        self.assertEqual(spec["env"]["CRITIC_GPUS"], "2")
        self.assertEqual(spec["env"]["ROLLOUT_GPUS"], "2")
        self.assertEqual(
            spec["env"]["HF_MODULES_CACHE"],
            "/dev/shm/nano30b-nla-pilot/hf_modules_cache/r33_component_full_signal_ladder_8h200",
        )
        self.assertEqual(spec["env"]["NLA_PREWARM_HF_MODULES"], "1")
        self.assertEqual(spec["rollout_batch_plan"]["generated_samples"], 16)
        self.assertTrue(spec["rollout_batch_plan"]["global_batch_matches_rollout"])
        self.assertEqual(spec["env"]["NLA_ROLLOUT_GLOBAL_MATCH"], "1")
        self.assertEqual(spec["command"][spec["command"].index("--global-batch-size") + 1], "16")
        self.assertIn("--rollout-external-engine-addrs", spec["command"])
        engine_index = spec["command"].index("--rollout-external-engine-addrs")
        self.assertEqual(spec["command"][engine_index + 1 : engine_index + 3], ["127.0.0.1:31000", "127.0.0.1:31001"])
        start_commands = spec["sglang_service"]["start_commands"]
        self.assertEqual(len(start_commands), 2)
        self.assertEqual(start_commands[0][start_commands[0].index("--base-gpu-id") + 1], "4")
        self.assertEqual(start_commands[1][start_commands[1].index("--base-gpu-id") + 1], "5")
        self.assertTrue(spec["sglang_service"]["model_staging"]["enabled"])
        self.assertEqual(
            spec["sglang_service"]["model_staging"]["target_path"],
            "/dev/shm/nano30b-nla-pilot/sglang_models/r33_component_full_actor_sft_iter_0001291",
        )
        post_eval = queue.build_roundtrip_post_eval_spec(queue_doc, item, spec, queue_path=queue_path)
        assert post_eval is not None
        generation_command = post_eval["generation_command"]
        score_command = post_eval["score_command"]
        self.assertEqual(generation_command[0], "/workspace/interp/.venvs/sglang-cu130/bin/python")
        self.assertIn("--generation-only", generation_command)
        self.assertEqual(score_command[0], "/workspace/interp/.venv/bin/python")
        self.assertIn("--reuse-generated", score_command)
        self.assertEqual(score_command[score_command.index("--generated-jsonl") + 1], post_eval["generated_jsonl"])

    def test_checked_in_r33_8h100_ladder_uses_tp2_rollout_engine(self):
        queue = load_script("nano_rl_queue")
        queue_path = ROOT / "configs" / "nano_rl" / "r33_component_full_signal_ladder_queue_8h100_len512.yaml"

        queue_doc = queue.load_queue(queue_path)
        item = queue_doc["items"][0]
        spec = queue.build_run_spec(queue_doc, item, queue_path=queue_path)

        self.assertEqual(spec["resource_total_gpus"], 8)
        self.assertEqual(spec["env"]["ACTOR_GPUS"], "4")
        self.assertEqual(spec["env"]["CRITIC_GPUS"], "2")
        self.assertEqual(spec["env"]["ROLLOUT_GPUS"], "2")
        self.assertEqual(spec["rollout_batch_plan"]["generated_samples"], 16)
        self.assertTrue(spec["rollout_batch_plan"]["global_batch_matches_rollout"])
        engine_index = spec["command"].index("--rollout-external-engine-addrs")
        self.assertEqual(spec["command"][engine_index + 1], "127.0.0.1:31000")
        self.assertNotEqual(
            spec["command"][engine_index + 2] if len(spec["command"]) > engine_index + 2 else None,
            "127.0.0.1:31000",
        )
        self.assertIn("--rollout-num-gpus-per-engine", spec["command"])
        self.assertEqual(spec["command"][spec["command"].index("--rollout-num-gpus-per-engine") + 1], "2")
        self.assertEqual(spec["sglang_service"]["engine_addrs"], ["127.0.0.1:31000"])
        start_commands = spec["sglang_service"]["start_commands"]
        self.assertEqual(len(start_commands), 1)
        self.assertEqual(start_commands[0][start_commands[0].index("--tp-size") + 1], "2")
        self.assertEqual(start_commands[0][start_commands[0].index("--base-gpu-id") + 1], "4")
        self.assertIn("--disable-custom-all-reduce", start_commands[0])
        self.assertEqual(queue_doc["items"][1]["rollout"]["global_batch_size"], 64)
        post_eval = queue.build_roundtrip_post_eval_spec(queue_doc, item, spec, queue_path=queue_path)
        assert post_eval is not None
        self.assertEqual(post_eval["generation_command"][0], "/workspace/interp/.venvs/sglang-cu130/bin/python")
        self.assertIn("--generation-only", post_eval["generation_command"])
        self.assertEqual(post_eval["score_command"][0], "/workspace/interp/.venv/bin/python")
        self.assertIn("--reuse-generated", post_eval["score_command"])

    def test_checked_in_r33_qwen_comparable_queue_uses_512_generation_updates(self):
        queue = load_script("nano_rl_queue")
        queue_path = ROOT / "configs" / "nano_rl" / "r33_component_qwen_comparable_queue_8h100_len512.yaml"

        queue_doc = queue.load_queue(queue_path)
        self.assertEqual(queue_doc["defaults"]["env"]["NLA_ROLLOUT_SUMMARY_STDOUT"], "1")
        self.assertEqual(queue_doc["defaults"]["env"]["NLA_FAILED_EXTRACTION_REWARD"], "-2.0")
        expected = {
            "r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp2": 2,
            "r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp1-actor5": 2,
            "r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp1-actor5-cpuoffload-nockpt-mb2": 2,
            "r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-async-tp2": 2,
            "r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5": 2,
            "r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5-mb4": 2,
            "r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5-mb8": 2,
            "r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5-mb16": 2,
            "r33-component-rl-8h100-tier1-medium-rb30-n16-gb480-lr5e6-rollout16-mb16-v64t64": 16,
            "r33-component-rl-8h100-stageb-advcheck-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64": 3,
            "r33-component-rl-8h100-stageb-advcheck2-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64": 3,
            "r33-component-rl-8h100-stageb-advcheck3-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64": 3,
            "r33-component-rl-8h100-stageb-advcheck4-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64": 3,
            "r33-component-rl-8h100-tier1-probe-rb64-n8-gb512-lr5e6-rollout8-v256t256": 8,
            "r33-component-rl-8h100-tier1-probe-rb64-n8-gb512-lr1e5-rollout8-v256t256": 8,
            "r33-component-rl-8h100-tier1-medium-rb64-n8-gb512-lr5e6-rollout32": 32,
            "r33-component-rl-8h100-tier1-confirm-rb64-n8-gb512-lr5e6-rollout64": 64,
            "r33-component-rl-8h100-tier2-minhero-rb64-n8-gb512-lr5e6-rollout150": 150,
            "r33-component-rl-8h100-tier2-hero-rb64-n8-gb512-lr5e6-rollout256": 256,
        }
        items = {item["name"]: item for item in queue_doc["items"]}
        self.assertEqual(set(items), set(expected))

        for name, num_rollout in expected.items():
            item = items[name]
            rollout = item["rollout"]
            training = item["training"]
            if "rb30-n16" in name:
                expected_rollout_batch_size = 30
                expected_samples_per_prompt = 16
                expected_global_batch_size = 480
            elif "rb32-n16" in name:
                expected_rollout_batch_size = 32
                expected_samples_per_prompt = 16
                expected_global_batch_size = 512
            else:
                expected_rollout_batch_size = 64
                expected_samples_per_prompt = 8
                expected_global_batch_size = 512
            self.assertEqual(rollout["rollout_batch_size"], expected_rollout_batch_size)
            self.assertEqual(rollout["n_samples_per_prompt"], expected_samples_per_prompt)
            self.assertEqual(rollout["global_batch_size"], expected_global_batch_size)
            self.assertEqual(rollout["num_rollout"], num_rollout)
            self.assertEqual(
                rollout["rollout_batch_size"] * rollout["n_samples_per_prompt"],
                expected_global_batch_size,
            )
            expected_lr = "1e-5" if "lr1e5" in name else "5e-6"
            self.assertEqual(training["actor_lr"], expected_lr)
            if "mb16" in name:
                expected_actor_mb = 16
            elif "tp1-actor5-mb8" in name:
                expected_actor_mb = 8
            elif "tp1-actor5-mb4" in name:
                expected_actor_mb = 4
            elif "cpuoffload-nockpt-mb2" in name:
                expected_actor_mb = 2
            else:
                expected_actor_mb = 1
            self.assertEqual(training["actor_micro_batch"], expected_actor_mb)
            self.assertEqual(training["kl_loss_coef"], 0.0003)

        fit_name = "r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp2"
        self.assertEqual(items[fit_name]["status"], "complete")
        probe_name = "r33-component-rl-8h100-tier1-probe-rb64-n8-gb512-lr5e6-rollout8-v256t256"
        self.assertEqual(items[probe_name]["status"], "complete")
        n16_name = "r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5"
        self.assertEqual(items[n16_name]["status"], "blocked")
        n16_mb4_name = "r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5-mb4"
        self.assertEqual(items[n16_mb4_name]["status"], "cancelled")
        n16_mb8_name = "r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5-mb8"
        self.assertEqual(items[n16_mb8_name]["status"], "cancelled")
        n16_mb16_name = "r33-component-rl-8h100-tier1-fit-rb32-n16-gb512-lr5e6-tp1-actor5-mb16"
        self.assertEqual(items[n16_mb16_name]["status"], "complete")
        n16_medium_name = "r33-component-rl-8h100-tier1-medium-rb30-n16-gb480-lr5e6-rollout16-mb16-v64t64"
        self.assertEqual(items[n16_medium_name]["status"], "cancelled")
        stageb_advcheck_name = "r33-component-rl-8h100-stageb-advcheck-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64"
        self.assertEqual(items[stageb_advcheck_name]["status"], "cancelled")
        stageb_advcheck2_name = "r33-component-rl-8h100-stageb-advcheck2-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64"
        self.assertEqual(items[stageb_advcheck2_name]["status"], "cancelled")
        stageb_advcheck3_name = "r33-component-rl-8h100-stageb-advcheck3-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64"
        self.assertEqual(items[stageb_advcheck3_name]["status"], "cancelled")
        stageb_advcheck4_name = "r33-component-rl-8h100-stageb-advcheck4-rb30-n16-gb480-lr5e6-rollout3-mb16-v64t64"
        self.assertEqual(items[stageb_advcheck4_name]["status"], "pending")
        for name in expected:
            if name not in {
                fit_name,
                probe_name,
                n16_name,
                n16_mb4_name,
                n16_mb8_name,
                n16_mb16_name,
                n16_medium_name,
                stageb_advcheck_name,
                stageb_advcheck2_name,
                stageb_advcheck3_name,
                stageb_advcheck4_name,
            }:
                self.assertEqual(items[name]["status"], "blocked")

        spec = queue.build_run_spec(queue_doc, items[probe_name], queue_path=queue_path)
        command = spec["command"]
        self.assertEqual(spec["resource_total_gpus"], 8)
        self.assertEqual(spec["env"]["ACTOR_GPUS"], "4")
        self.assertEqual(spec["env"]["CRITIC_GPUS"], "2")
        self.assertEqual(spec["env"]["ROLLOUT_GPUS"], "2")
        self.assertEqual(spec["rollout_batch_plan"]["generated_samples"], 512)
        self.assertTrue(spec["rollout_batch_plan"]["global_batch_matches_rollout"])
        self.assertEqual(spec["env"]["NLA_ROLLOUT_GLOBAL_MATCH"], "1")
        self.assertEqual(command[command.index("--rollout-batch-size") + 1], "64")
        self.assertEqual(command[command.index("--n-samples-per-prompt") + 1], "8")
        self.assertEqual(command[command.index("--global-batch-size") + 1], "512")
        self.assertIn("--rollout-num-gpus-per-engine", command)
        self.assertEqual(command[command.index("--rollout-num-gpus-per-engine") + 1], "2")
        addr_index = command.index("--rollout-external-engine-addrs")
        self.assertEqual(command[addr_index + 1], "127.0.0.1:31000")
        self.assertEqual(spec["sglang_service"]["engine_addrs"], ["127.0.0.1:31000"])
        post_eval = queue.build_roundtrip_post_eval_spec(queue_doc, items[probe_name], spec, queue_path=queue_path)
        assert post_eval is not None
        self.assertIn("v256_t256", post_eval["report_json"])
        self.assertIn("--validation-limit", post_eval["score_command"])
        self.assertEqual(post_eval["score_command"][post_eval["score_command"].index("--validation-limit") + 1], "256")
        self.assertNotIn("--test-limit", post_eval["score_command"])
        self.assertNotIn("--test-parquet", post_eval["score_command"])

        tp1_name = "r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp1-actor5"
        tp1_spec = queue.build_run_spec(queue_doc, items[tp1_name], queue_path=queue_path)
        tp1_command = tp1_spec["command"]
        self.assertEqual(tp1_spec["resource_total_gpus"], 8)
        self.assertEqual(tp1_spec["env"]["ACTOR_GPUS"], "5")
        self.assertEqual(tp1_spec["env"]["CRITIC_GPUS"], "2")
        self.assertEqual(tp1_spec["env"]["ROLLOUT_GPUS"], "1")
        self.assertEqual(tp1_spec["env"]["NLA_ROLLOUT_GPUS_PER_ENGINE"], "1")
        self.assertEqual(tp1_spec["env"]["NLA_SGLANG_TP_SIZE"], "1")
        self.assertEqual(tp1_command[tp1_command.index("--rollout-num-gpus-per-engine") + 1], "1")
        tp1_start = tp1_spec["sglang_service"]["start_commands"][0]
        self.assertEqual(tp1_start[tp1_start.index("--tp-size") + 1], "1")
        self.assertEqual(tp1_start[tp1_start.index("--base-gpu-id") + 1], "5")

        n16_spec = queue.build_run_spec(queue_doc, items[n16_name], queue_path=queue_path)
        n16_command = n16_spec["command"]
        self.assertEqual(n16_spec["resource_total_gpus"], 8)
        self.assertEqual(n16_spec["env"]["ACTOR_GPUS"], "5")
        self.assertEqual(n16_spec["env"]["CRITIC_GPUS"], "2")
        self.assertEqual(n16_spec["env"]["ROLLOUT_GPUS"], "1")
        self.assertEqual(n16_spec["env"]["NLA_ROLLOUT_GPUS_PER_ENGINE"], "1")
        self.assertEqual(n16_spec["env"]["NLA_SGLANG_TP_SIZE"], "1")
        self.assertEqual(n16_spec["rollout_batch_plan"]["generated_samples"], 512)
        self.assertTrue(n16_spec["rollout_batch_plan"]["global_batch_matches_rollout"])
        self.assertEqual(n16_command[n16_command.index("--rollout-batch-size") + 1], "32")
        self.assertEqual(n16_command[n16_command.index("--n-samples-per-prompt") + 1], "16")
        self.assertEqual(n16_command[n16_command.index("--global-batch-size") + 1], "512")
        self.assertEqual(n16_command[n16_command.index("--rollout-num-gpus-per-engine") + 1], "1")

        n16_medium_spec = queue.build_run_spec(queue_doc, items[n16_medium_name], queue_path=queue_path)
        n16_medium_command = n16_medium_spec["command"]
        self.assertEqual(n16_medium_spec["resource_total_gpus"], 8)
        self.assertEqual(n16_medium_spec["env"]["ACTOR_GPUS"], "5")
        self.assertEqual(n16_medium_spec["env"]["CRITIC_GPUS"], "2")
        self.assertEqual(n16_medium_spec["env"]["ROLLOUT_GPUS"], "1")
        self.assertEqual(n16_medium_spec["rollout_batch_plan"]["generated_samples"], 480)
        self.assertTrue(n16_medium_spec["rollout_batch_plan"]["global_batch_matches_rollout"])
        self.assertEqual(n16_medium_spec["env"]["NLA_ROLLOUT_GLOBAL_BATCH"], "480")
        self.assertEqual(n16_medium_spec["env"]["NLA_ROLLOUT_GENERATED_SAMPLES"], "480")
        self.assertEqual(n16_medium_command[n16_medium_command.index("--rollout-batch-size") + 1], "30")
        self.assertEqual(n16_medium_command[n16_medium_command.index("--n-samples-per-prompt") + 1], "16")
        self.assertEqual(n16_medium_command[n16_medium_command.index("--global-batch-size") + 1], "480")
        self.assertEqual(n16_medium_command[n16_medium_command.index("--rollout-num-gpus-per-engine") + 1], "1")
        n16_medium_eval = queue.build_roundtrip_post_eval_spec(
            queue_doc,
            items[n16_medium_name],
            n16_medium_spec,
            queue_path=queue_path,
        )
        assert n16_medium_eval is not None
        self.assertFalse(n16_medium_eval["cleanup_actor_checkpoint"])
        self.assertIn("v64_t64", n16_medium_eval["report_json"])

        cpuoffload_name = (
            "r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-tp1-actor5-cpuoffload-nockpt-mb2"
        )
        cpuoffload_spec = queue.build_run_spec(queue_doc, items[cpuoffload_name], queue_path=queue_path)
        cpuoffload_command = cpuoffload_spec["command"]
        self.assertEqual(cpuoffload_spec["env"]["ACTOR_GPUS"], "5")
        self.assertEqual(cpuoffload_spec["env"]["CRITIC_GPUS"], "2")
        self.assertEqual(cpuoffload_spec["env"]["ROLLOUT_GPUS"], "1")
        self.assertEqual(cpuoffload_spec["env"]["GRADIENT_CHECKPOINTING"], "0")
        self.assertEqual(cpuoffload_spec["env"]["FSDP_CPU_OFFLOAD"], "1")
        self.assertEqual(cpuoffload_spec["env"]["FSDP_CPU_BACKEND"], "gloo")
        self.assertEqual(cpuoffload_command[cpuoffload_command.index("--rollout-num-gpus-per-engine") + 1], "1")

        async_name = "r33-component-rl-8h100-tier1-fit-rb64-n8-gb512-lr5e6-async-tp2"
        async_spec = queue.build_run_spec(queue_doc, items[async_name], queue_path=queue_path)
        async_command = async_spec["command"]
        self.assertTrue(async_spec["env"]["TRAIN_ENTRYPOINT"].endswith("/train_async.py"))
        self.assertEqual(async_spec["env"]["GRADIENT_CHECKPOINTING"], "1")
        self.assertEqual(async_command[async_command.index("--rollout-num-gpus-per-engine") + 1], "2")

    def test_kl_ref_path_is_preflighted_before_launch(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            rl_parquet = root / "rl.parquet"
            rl_parquet.write_text("placeholder")
            actor = root / "actor"
            critic = root / "critic" / "hf"
            model = root / "model"
            for path in (actor, critic, model):
                path.mkdir(parents=True)
            missing_ref = root / "missing-hf-ref"
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  training:
                    kl_loss_coef: 0.0001
                items:
                  - name: missing-ref
                    status: pending
                    rl_parquet: {rl_parquet}
                    instruct_model: {model}
                    actor_sft_ckpt: {actor}
                    actor_ref_ckpt: {missing_ref}
                    critic_sl_ckpt: {critic}
                    run_dir: {root / "runs" / "missing-ref"}
                """,
            )

            queue.set_item_approval(queue_path, "missing-ref", approved=True, approved_by="test")
            dry = queue.process_next_item(queue_path, dry_run=True)
            queue_doc = queue.load_queue(queue_path)
            spec = queue.build_run_spec(queue_doc, queue_doc["items"][0], queue_path=queue_path)

        self.assertEqual(dry["status"], "dry_run")
        self.assertIn("ACTOR_REF_CKPT", {entry["label"] for entry in dry["preflight_missing_paths"]})
        with self.assertRaisesRegex(queue.QueueError, "ACTOR_REF_CKPT"):
            queue.preflight_run_spec(spec)

    def test_rl_shell_uses_train_entrypoint_and_env_defaults(self):
        rl_script = ROOT / "external" / "natural_language_autoencoders" / "configs" / "rl.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            fake_train = root / "train.py"
            output_json = root / "argv.json"
            fake_train.write_text(
                textwrap.dedent(
                    f"""
                    import json
                    import os
                    import sys
                    pathlib_safe = {str(output_json)!r}
                    with open(pathlib_safe, "w") as f:
                        json.dump({{"argv": sys.argv, "env": dict(os.environ)}}, f, sort_keys=True)
                    """
                )
            )
            env = {
                **os.environ,
                "PYTHON": sys.executable,
                "TRAIN_ENTRYPOINT": str(fake_train),
                "RL_PARQUET": "/data/rl.parquet",
                "INSTRUCT_MODEL": "/models/base",
                "ACTOR_SFT_CKPT": "/ckpts/actor",
                "CRITIC_SL_CKPT": "/ckpts/critic/hf",
                "RUN_DIR": "/runs/rl",
                "ACTOR_GPUS": "1",
                "CRITIC_GPUS": "1",
                "ROLLOUT_GPUS": "2",
                "ACTOR_MICRO": "1",
                "ACTOR_LR": "3e-6",
                "CRITIC_LR": "2e-5",
                "ATTN_IMPLEMENTATION": "eager",
                "FSDP_REDUCE_DTYPE": "bfloat16",
                "FSDP_DISABLE_BACKWARD_PREFETCH": "1",
                "KL_LOSS_COEF": "0",
                "SAVE_INTERVAL": "7",
                "NLA_EMBED_DUMP_DIR": str(root / "shm"),
                "ROLLOUT_BATCH_SIZE": "8",
                "GLOBAL_BATCH_SIZE": "16",
                "N_SAMPLES_PER_PROMPT": "2",
                "ROLLOUT_MAX_RESPONSE_LEN": "64",
                "ROLLOUT_MAX_CONTEXT_LEN": "256",
            }

            subprocess.run(["bash", str(rl_script), "--num-rollout", "1"], env=env, check=True)
            payload = json.loads(output_json.read_text())

        argv = payload["argv"]
        self.assertEqual(argv[0], str(fake_train))
        self.assertIn("--actor-num-gpus-per-node", argv)
        self.assertEqual(argv[argv.index("--actor-num-gpus-per-node") + 1], "1")
        self.assertEqual(argv[argv.index("--rollout-num-gpus") + 1], "2")
        self.assertEqual(argv[argv.index("--rollout-batch-size") + 1], "8")
        self.assertEqual(argv[argv.index("--global-batch-size") + 1], "16")
        self.assertEqual(argv[argv.index("--n-samples-per-prompt") + 1], "2")
        self.assertEqual(argv[argv.index("--attn-implementation") + 1], "eager")
        self.assertEqual(argv[argv.index("--lr") + 1], "3e-6")
        self.assertEqual(argv[argv.index("--critic-lr") + 1], "2e-5")
        self.assertEqual(argv[argv.index("--fsdp-reduce-dtype") + 1], "bfloat16")
        self.assertIn("--fsdp-disable-backward-prefetch", argv)
        self.assertNotIn("--sglang-context-length", argv)
        self.assertNotIn("--use-kl-loss", argv)
        train_env_vars = json.loads(argv[argv.index("--train-env-vars") + 1])
        self.assertEqual(train_env_vars["HF_MODULES_CACHE"], "/dev/shm/nano30b-nla-pilot/hf_modules_cache")

    def test_rl_shell_emits_signal_control_flags(self):
        rl_script = ROOT / "external" / "natural_language_autoencoders" / "configs" / "rl.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            fake_train = root / "train.py"
            output_json = root / "argv.json"
            fake_train.write_text(
                textwrap.dedent(
                    f"""
                    import json
                    import sys
                    with open({str(output_json)!r}, "w") as f:
                        json.dump(sys.argv, f)
                    """
                )
            )
            env = {
                **os.environ,
                "PYTHON": sys.executable,
                "TRAIN_ENTRYPOINT": str(fake_train),
                "RL_PARQUET": "/data/rl.parquet",
                "INSTRUCT_MODEL": "/models/base",
                "ACTOR_SFT_CKPT": "/ckpts/actor",
                "ACTOR_LOAD_CKPT": "/ckpts/actor/checkpoints",
                "ACTOR_REF_CKPT": "/ckpts/actor/checkpoints",
                "ACTOR_SIDECAR_SOURCE": "/ckpts/actor/checkpoints/iter_0000001",
                "CRITIC_SL_CKPT": "/ckpts/critic/hf",
                "RUN_DIR": "/runs/rl",
                "KL_LOSS_COEF": "0.0003",
                "ADVANTAGE_ESTIMATOR": "grpo",
                "NORMALIZE_ADVANTAGES": "1",
                "REWARDS_NORMALIZATION": "0",
                "GRPO_STD_NORMALIZATION": "0",
                "WANDB_MODE": "disabled",
                "SAVE_INTERVAL": "",
                "NLA_EMBED_DUMP_DIR": str(root / "shm"),
            }

            subprocess.run(["bash", str(rl_script), "--num-rollout", "1"], env=env, check=True)
            argv = json.loads(output_json.read_text())

        self.assertEqual(argv[argv.index("--advantage-estimator") + 1], "grpo")
        self.assertIn("--use-kl-loss", argv)
        self.assertEqual(argv[argv.index("--kl-loss-coef") + 1], "0.0003")
        self.assertIn("--normalize-advantages", argv)
        self.assertIn("--disable-rewards-normalization", argv)
        self.assertIn("--disable-grpo-std-normalization", argv)

    def test_rl_shell_emits_throughput_control_flags(self):
        rl_script = ROOT / "external" / "natural_language_autoencoders" / "configs" / "rl.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            fake_train = root / "train.py"
            output_json = root / "argv.json"
            fake_train.write_text(
                textwrap.dedent(
                    f"""
                    import json
                    import sys
                    with open({str(output_json)!r}, "w") as f:
                        json.dump(sys.argv, f)
                    """
                )
            )
            env = {
                **os.environ,
                "PYTHON": sys.executable,
                "TRAIN_ENTRYPOINT": str(fake_train),
                "RL_PARQUET": "/data/rl.parquet",
                "INSTRUCT_MODEL": "/models/base",
                "ACTOR_SFT_CKPT": "/ckpts/actor",
                "CRITIC_SL_CKPT": "/ckpts/critic/hf",
                "RUN_DIR": "/runs/rl",
                "KL_LOSS_COEF": "0",
                "WANDB_MODE": "disabled",
                "SAVE_INTERVAL": "",
                "NLA_EMBED_DUMP_DIR": str(root / "shm"),
                "GRADIENT_CHECKPOINTING": "0",
                "OFFLOAD_TRAIN": "1",
                "OFFLOAD_ROLLOUT": "1",
                "OFFLOAD_ROLLOUT_LEVEL": "kv_cache",
                "FSDP_CPU_OFFLOAD": "1",
                "FSDP_CPU_BACKEND": "gloo",
                "COLOCATE": "1",
            }

            subprocess.run(["bash", str(rl_script), "--num-rollout", "1"], env=env, check=True)
            argv = json.loads(output_json.read_text())

        self.assertNotIn("--gradient-checkpointing", argv)
        self.assertIn("--offload-train", argv)
        self.assertIn("--offload-rollout", argv)
        self.assertEqual(argv[argv.index("--offload-rollout-level") + 1 : argv.index("--offload-rollout-level") + 2], ["kv_cache"])
        self.assertIn("--fsdp-cpu-offload", argv)
        self.assertEqual(argv[argv.index("--fsdp-cpu-backend") + 1], "gloo")
        self.assertIn("--colocate", argv)


    def test_promotes_blocked_dependency_only_after_passing_gate(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            queue_path = self._write_queue(
                root,
                """
                schema_version: nano_rl_queue.v1
                items:
                  - name: confirmation
                    status: complete
                    gate_passed: false
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic
                    run_dir: /runs/confirmation
                  - name: medium
                    status: blocked
                    depends_on:
                      item: confirmation
                      require_gate_pass: true
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /ckpts/actor
                    critic_sl_ckpt: /ckpts/critic
                    run_dir: /runs/medium
                """,
            )
            queue_doc = queue.load_queue(queue_path)

        self.assertEqual(queue.promote_ready_blocked_items(queue_doc), [])
        self.assertEqual(queue_doc["items"][1]["status"], "blocked")

        queue_doc["items"][0]["gate_passed"] = True
        queue_doc["items"][1]["launch"] = {
            "requires_approval": True,
            "approved": True,
        }
        self.assertEqual(queue.promote_ready_blocked_items(queue_doc), ["medium"])
        self.assertEqual(queue_doc["items"][1]["status"], "pending")
        self.assertEqual(queue_doc["items"][1]["dependency_gate_passed"], True)

    def test_checked_in_r33_medium_scaleup_queue_uses_selected_hpo_config(self):
        queue = load_script("nano_rl_queue")
        queue_path = ROOT / "configs" / "nano_rl" / "r33_component_medium_scaleup_queue_8h100.yaml"

        queue_doc = queue.load_queue(queue_path)
        items = {item["name"]: item for item in queue_doc["items"]}
        self.assertEqual(
            set(items),
            {
                "r33-component-rl-8h100-confirm-lr2e5-kl1e3-rollout16-mb32",
                "r33-component-rl-8h100-medium-lr2e5-kl1e3-rollout32-mb32",
            },
        )

        for item in items.values():
            spec = queue.build_run_spec(queue_doc, item, queue_path=queue_path)
            self.assertEqual(spec["resource_total_gpus"], 8)
            self.assertEqual(spec["env"]["ACTOR_GPUS"], "6")
            self.assertEqual(spec["env"]["CRITIC_GPUS"], "1")
            self.assertEqual(spec["env"]["ROLLOUT_GPUS"], "1")
            self.assertEqual(spec["env"]["ACTOR_MICRO"], "32")
            self.assertEqual(spec["env"]["ACTOR_LR"], "2e-5")
            self.assertEqual(spec["env"]["KL_LOSS_COEF"], "0.001")
            self.assertEqual(spec["rollout_batch_plan"]["generated_samples"], 480)
            self.assertTrue(spec["rollout_batch_plan"]["global_batch_matches_rollout"])

        confirmation = items["r33-component-rl-8h100-confirm-lr2e5-kl1e3-rollout16-mb32"]
        medium = items["r33-component-rl-8h100-medium-lr2e5-kl1e3-rollout32-mb32"]
        self.assertEqual(confirmation["status"], "pending")
        self.assertEqual(confirmation["rollout"]["num_rollout"], 16)
        self.assertEqual(confirmation["post_eval"]["roundtrip"]["validation_limit"], 256)
        self.assertEqual(confirmation["post_eval"]["roundtrip"]["test_limit"], 256)
        self.assertEqual(medium["status"], "blocked")
        self.assertEqual(medium["depends_on"], {"item": confirmation["name"], "require_gate_pass": True})
        self.assertEqual(medium["rollout"]["num_rollout"], 32)
        self.assertEqual(medium["post_eval"]["roundtrip"]["validation_limit"], 512)
        self.assertEqual(medium["post_eval"]["roundtrip"]["test_limit"], 512)

    def test_builds_reward_gate_correlation_post_eval_spec(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            queue_path = root / "queue.yaml"
            queue_doc = {
                "defaults": {
                    "post_eval_scoring_python": "/venv/bin/python",
                    "post_eval": {
                        "reward_gate_correlation": {
                            "enabled": True,
                            "critic_checkpoint_dir": "/ckpts/critic/hf",
                            "validation_parquet": "/data/validation.parquet",
                            "test_parquet": "/data/test.parquet",
                            "batch_size": 8,
                            "device": "cuda:0",
                        }
                    },
                }
            }
            run_spec = {"code_root": str(root), "run_dir": str(root / "run")}
            roundtrip_spec = {
                "report_json": str(root / "run" / "roundtrip_report.json"),
                "generated_jsonl": str(root / "run" / "generated.jsonl"),
            }

            spec = queue.build_reward_gate_correlation_post_eval_spec(
                queue_doc,
                {},
                run_spec,
                roundtrip_spec,
                queue_path=queue_path,
            )

        assert spec is not None
        self.assertEqual(spec["command"][0], "/venv/bin/python")
        self.assertIn("scripts/analyze_rl_reward_gate_correlation.py", spec["command"][1])
        self.assertIn("--roundtrip-report-json", spec["command"])
        self.assertEqual(spec["command"][spec["command"].index("--batch-size") + 1], "8")
        self.assertEqual(spec["command"][spec["command"].index("--device") + 1], "cuda:0")
        self.assertTrue(spec["output_json"].endswith("reward_gate_correlation.json"))

    def test_rejects_global_batch_not_divisible_by_actor_dp_times_microbatch(self):
        queue = load_script("nano_rl_queue")

        with self.assertRaisesRegex(
            queue.QueueError,
            "actor_gpus=6.*actor_micro_batch=32",
        ):
            queue._validate_actor_batch_plan(
                generated_samples=480,
                global_batch_size=480,
                actor_gpus=6,
                actor_micro_batch=32,
                required=True,
            )

    def test_accepts_exact_384_sample_actor_batch(self):
        queue = load_script("nano_rl_queue")

        plan = queue._validate_actor_batch_plan(
            generated_samples=384,
            global_batch_size=384,
            actor_gpus=6,
            actor_micro_batch=32,
            required=True,
        )

        self.assertEqual(plan["effective_trained_samples"], 384)
        self.assertEqual(plan["samples_per_actor"], 64)
        self.assertEqual(plan["actor_batch_divisor"], 192)
        self.assertNotIn("warning", plan)

    def test_legacy_nondivisible_actor_batch_is_warning_only(self):
        queue = load_script("nano_rl_queue")

        plan = queue._validate_actor_batch_plan(
            generated_samples=480,
            global_batch_size=480,
            actor_gpus=6,
            actor_micro_batch=32,
            required=False,
        )

        self.assertEqual(plan["effective_trained_samples"], 384)
        self.assertIn("warning", plan)

    def test_accepts_exact_192_sample_critic_batch(self):
        queue = load_script("nano_rl_queue")

        plan = queue._validate_critic_batch_plan(
            global_batch_size=192,
            critic_gpus=3,
            critic_micro_batch=2,
            required=True,
        )

        self.assertEqual(plan["critic_batch_divisor"], 6)
        self.assertEqual(plan["full_usable_critic_retained_samples"], 192)
        self.assertEqual(plan["full_usable_critic_alignment_drop"], 0)
        self.assertEqual(plan["full_usable_critic_retained_fraction"], 1.0)

    def test_rejects_nondivisible_exact_critic_batch(self):
        queue = load_script("nano_rl_queue")

        with self.assertRaisesRegex(
            queue.QueueError,
            "critic_gpus=3.*shared_micro_batch=2",
        ):
            queue._validate_critic_batch_plan(
                global_batch_size=191,
                critic_gpus=3,
                critic_micro_batch=2,
                required=True,
            )

    def test_checked_in_clean_online_hpo_queue_has_balanced_medium_batches(self):
        queue = load_script("nano_rl_queue")
        queue_path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "hpo"
            / "r33_family_clean_online_joint_hpo8_queue_8h100.yaml"
        )
        queue_doc = queue.load_queue(queue_path)
        items = {item["name"]: item for item in queue_doc["items"]}
        h1_name = (
            "r33-family-clean-online-joint-hpo8-h1-a1e5-c2e6-k3e3-retry5"
        )

        self.assertEqual(len(items), 4)
        self.assertTrue(items[h1_name]["launch"]["approved"])
        for name, item in items.items():
            spec = queue.build_run_spec(queue_doc, item, queue_path=queue_path)
            plan = spec["rollout_batch_plan"]
            self.assertEqual(spec["resource_total_gpus"], 8, name)
            self.assertEqual(plan["generated_samples"], 192, name)
            self.assertEqual(plan["effective_trained_samples"], 192, name)
            self.assertEqual(plan["full_usable_critic_retained_samples"], 192, name)
            self.assertEqual(plan["full_usable_critic_alignment_drop"], 0, name)
            self.assertEqual(
                spec["env"]["NLA_MIN_CRITIC_RETAINED_FRACTION"],
                "0.95",
                name,
            )
            self.assertEqual(spec["env"]["WANDB_MODE"], "offline", name)
            self.assertEqual(spec["env"]["QKV_FORMAT"], "bshd", name)
            staged_by_env = {
                entry["env_key"]: entry for entry in spec["input_staging"]
            }
            self.assertEqual(
                staged_by_env["ACTOR_REF_CKPT"]["target_path"],
                spec["sglang_service"]["model_staging"]["target_path"],
                name,
            )

        h1 = queue.build_run_spec(
            queue_doc,
            items[h1_name],
            queue_path=queue_path,
        )
        self.assertEqual(h1["env"]["ACTOR_LR"], "1e-5")
        self.assertEqual(h1["env"]["CRITIC_LR"], "2e-6")
        self.assertEqual(h1["env"]["KL_LOSS_COEF"], "0.003")
        post_eval = queue.build_roundtrip_post_eval_spec(
            queue_doc,
            items[h1_name],
            h1,
            queue_path=queue_path,
        )
        assert post_eval is not None
        generation_command = post_eval["generation_command"]
        self.assertEqual(
            generation_command[generation_command.index("--seed") + 1],
            "20260709",
        )
        self.assertEqual(
            generation_command[generation_command.index("--stop-text") + 1],
            "</explanation>",
        )
        self.assertTrue(
            generation_command[
                generation_command.index("--train-parquet") + 1
            ].endswith("/splits/train_padded.parquet")
        )
        contracts = {
            contract["name"]: contract for contract in h1["runtime_contracts"]
        }
        self.assertEqual(
            contracts["fsdp_bshd_runtime_support"]["contains"],
            [
                'if args.qkv_format == "bshd":',
                "assert args.train_backend in (",
                '"megatron", "fsdp"',
                "bshd format is only supported for megatron and FSDP backends.",
            ],
        )
        rollout_rules = {
            rule["metric"]: rule
            for rule in json.loads(h1["env"]["NLA_ROLLOUT_GUARD_RULES_JSON"])
        }
        self.assertEqual(
            rollout_rules["rollout/nla_parse/closed_frac"]["threshold"],
            0.8,
        )
        self.assertEqual(
            rollout_rules["rollout/nla_parse/usable_frac"]["consecutive_steps"],
            2,
        )
        self.assertEqual(
            rollout_rules["rollout/nla_generation/truncated_frac"]["threshold"],
            0.2,
        )

    def test_corrected_k3_queue_has_two_probes_and_blocked_confirmation(self):
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_corrected_k3_hpo_queue_8h100.yaml"
        )
        queue = yaml.safe_load(path.read_text())
        items = queue["items"]

        self.assertEqual(
            queue["defaults"]["python"],
            queue["defaults"]["sglang"]["python"],
        )
        self.assertEqual(
            queue["defaults"]["env"]["NLA_ASSERT_PACKED_EQUIV"],
            "0",
        )
        self.assertEqual(
            queue["defaults"]["env"]["NLA_WEIGHT_UPDATE_LOG_EVERY"],
            "64",
        )
        self.assertEqual(
            queue["defaults"]["sglang"]["model_staging"]["release_after_health"],
            {"enabled": True, "globs": ["model-*.safetensors"]},
        )
        self.assertTrue(
            queue["defaults"]["post_eval"]["roundtrip"]["resume_generated"]
        )
        self.assertEqual(
            [item["training"]["actor_lr"] for item in items[:2]],
            ["1e-5", "2e-5"],
        )
        self.assertTrue(all(item["training"]["kl_loss_type"] == "k3" for item in items))
        self.assertTrue(all(item["rollout"]["rollout_batch_size"] == 48 for item in items))
        self.assertTrue(all(item["rollout"]["n_samples_per_prompt"] == 8 for item in items))
        self.assertTrue(all(item["rollout"]["global_batch_size"] == 384 for item in items))
        self.assertEqual(items[2]["status"], "blocked")
        self.assertEqual(
            items[2]["name"],
            "r33-corrected-k3-confirm-lr1e5-update32-runtimefix-retry1",
        )
        self.assertFalse(items[2]["launch"]["approved"])
        self.assertEqual(items[2]["training"]["save_interval"], 16)
        self.assertEqual(items[2]["checkpoint_retention"]["keep_iterations"], [16, 32])

        queue_module = load_script("nano_rl_queue")
        queue_doc = queue_module.load_queue(path)
        # These are frozen historical items; this test covers command rendering,
        # while source-policy enforcement has dedicated provenance tests.
        historical_queue_doc = copy.deepcopy(queue_doc)
        historical_queue_doc["defaults"]["source"] = {}
        for historical_item in historical_queue_doc["items"][:2]:
            historical_spec = queue_module.build_run_spec(
                historical_queue_doc,
                historical_item,
                queue_path=path,
            )
            self.assertNotIn("--rollout-shuffle", historical_spec["command"])
            self.assertNotIn("--rollout-seed", historical_spec["command"])
            self.assertIn("--nla-skip-grad-norm", historical_spec["command"])
            historical_train_rules = json.loads(
                historical_spec["env"]["NLA_TRAIN_GUARD_RULES_JSON"]
            )
            self.assertEqual(
                [rule["metric"] for rule in historical_train_rules],
                ["train/train_rollout_logprob_abs_diff"],
            )
            self.assertNotIn("NLA_ROLLOUT_GUARD_RULES_JSON", historical_spec["env"])

        confirmation_queue_doc = copy.deepcopy(queue_doc)
        frozen_commit = confirmation_queue_doc["defaults"]["source"]["frozen_git_commit"]
        confirmation_queue_doc["defaults"]["source"] = {}
        spec = queue_module.build_run_spec(
            confirmation_queue_doc,
            confirmation_queue_doc["items"][2],
            queue_path=path,
        )
        self.assertEqual(
            spec["sglang_service"]["model_staging"]["release_after_health"],
            {"enabled": True, "globs": ["model-*.safetensors"]},
        )
        self.assertIn("--rollout-shuffle", spec["command"])
        self.assertNotIn("--nla-skip-grad-norm", spec["command"])
        self.assertEqual(frozen_commit, "0974ca3317db333f2355c1faaa09c583a0e09e42")
        self.assertEqual(len(spec["runtime_contracts"]), 2)
        self.assertEqual(
            spec["runtime_contracts"][0]["sha256"],
            "7db9b4acfbc7af734dee736c8a549cdd5a3f6d31c46e4d7d53f8028b62357479",
        )
        self.assertEqual(spec["checkpoint_retention"]["expected_saved_iterations"], [16, 32])
        eval_specs = queue_module.build_roundtrip_post_eval_specs(
            queue_doc,
            queue_doc["items"][2],
            spec,
            queue_path=path,
        )
        self.assertEqual([item["iteration"] for item in eval_specs], [16, 32, 32])
        self.assertTrue(eval_specs[-1]["require_previous_gate_pass"])

    def test_checkpoint_retention_applies_declared_keep_set(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            actor = root / "actor"
            for iteration in (8, 16, 24, 32):
                (actor / f"iter_{iteration:07d}").mkdir(parents=True)
            spec = {
                "checkpoint_retention": {
                    "enabled": True,
                    "checkpoint_root": str(actor),
                    "keep_iterations": [16, 32],
                    "manifest_path": str(root / "retention.json"),
                    "apply": True,
                }
            }

            result = queue.apply_checkpoint_retention(spec)

            self.assertIsNotNone(result)
            self.assertFalse((actor / "iter_0000008").exists())
            self.assertTrue((actor / "iter_0000016").exists())
            self.assertFalse((actor / "iter_0000024").exists())
            self.assertTrue((actor / "iter_0000032").exists())
            self.assertTrue((root / "retention.json").is_file())

    def test_process_retains_post_eval_input_before_cleanup(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            actor_root = root / "run" / "actor"
            evaluated_checkpoint = actor_root / "iter_0000001"
            transient_checkpoint = actor_root / "iter_0000002"
            evaluated_checkpoint.mkdir(parents=True)
            transient_checkpoint.mkdir()
            queue_path = self._write_queue(
                root,
                """
                schema_version: nano_rl_queue.v1
                defaults:
                  launch:
                    requires_approval: false
                items:
                  - name: ordered-lifecycle
                    status: pending
                    rl_parquet: /data/rl.parquet
                    instruct_model: /models/base
                    actor_sft_ckpt: /runs/actor
                    critic_sl_ckpt: /runs/critic
                    run_dir: /runs/ordered-lifecycle
                """,
            )
            events = []
            spec = {
                "item_name": "ordered-lifecycle",
                "run_dir": str(root / "run"),
                "log_path": str(root / "run" / "train.log"),
                "cwd": str(root),
                "command": ["true"],
                "env": {},
                "resource_total_gpus": 0,
                "source_provenance": None,
                "checkpoint_retention": {
                    "enabled": True,
                    "checkpoint_root": str(actor_root),
                    "keep_iterations": [1],
                    "manifest_path": str(root / "retention.json"),
                    "apply": True,
                },
            }
            service = mock.MagicMock()
            apply_retention = queue.apply_checkpoint_retention

            def retain_before_eval(*args, **kwargs):
                events.append("retention")
                return apply_retention(*args, **kwargs)

            def eval_then_cleanup(*args, **kwargs):
                events.append("eval")
                self.assertTrue(evaluated_checkpoint.exists())
                self.assertFalse(transient_checkpoint.exists())
                shutil.rmtree(evaluated_checkpoint)
                return {}

            with (
                mock.patch.object(queue, "build_run_spec", return_value=spec),
                mock.patch.object(
                    queue,
                    "build_roundtrip_post_eval_specs",
                    return_value=[{"required_paths": {}, "iteration": 1}],
                ),
                mock.patch.object(queue, "preflight_run_spec"),
                mock.patch.object(queue, "freeze_launch_contract", return_value={}),
                mock.patch.object(queue, "managed_sglang_service", return_value=service),
                mock.patch.object(
                    queue, "_run_logged", side_effect=lambda *args, **kwargs: events.append("train")
                ),
                mock.patch.object(
                    queue,
                    "run_roundtrip_post_eval",
                    side_effect=eval_then_cleanup,
                ),
                mock.patch.object(
                    queue,
                    "apply_checkpoint_retention",
                    side_effect=retain_before_eval,
                ),
            ):
                result = queue.process_next_item(queue_path)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(events, ["train", "retention", "eval"])
            self.assertFalse(evaluated_checkpoint.exists())
            self.assertTrue((root / "retention.json").is_file())

    def test_corrected_k3_live_sync_canary_uses_unified_runtime_and_one_update(self):
        queue_module = load_script("nano_rl_queue")
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_corrected_k3_live_sync_canary_8h100.yaml"
        )

        queue_doc = queue_module.load_queue(path)
        item = queue_doc["items"][0]
        spec = queue_module.build_run_spec(queue_doc, item, queue_path=path)

        self.assertEqual(
            queue_doc["defaults"]["python"],
            queue_doc["defaults"]["sglang"]["python"],
        )
        self.assertIn(item["status"], {"pending", "running", "complete", "failed"})
        self.assertEqual(item["rollout"]["num_rollout"], 1)
        self.assertEqual(spec["rollout_batch_plan"]["generated_samples"], 384)
        self.assertEqual(spec["rollout_batch_plan"]["global_batch_size"], 384)
        self.assertEqual(spec["env"]["ACTOR_MICRO"], "32")
        self.assertEqual(spec["env"]["KL_LOSS_TYPE"], "k3")
        self.assertEqual(spec["env"]["NLA_FREEZE_CRITIC_TRAIN"], "1")
        self.assertEqual(spec["env"]["NLA_ASSERT_PACKED_EQUIV"], "0")
        self.assertEqual(spec["env"]["NLA_WEIGHT_UPDATE_LOG_EVERY"], "1")
        self.assertEqual(spec["env"]["TORCH_NCCL_DESYNC_DEBUG"], "1")
        self.assertEqual(spec["env"]["TORCH_NCCL_TRACE_BUFFER_SIZE"], "2000")
        self.assertIsNone(spec.get("post_eval"))

    def test_staged_model_cache_is_rebuilt_when_declared_files_are_missing(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "source"
            target = root / "staged"
            run_dir = root / "run"
            source.mkdir()
            run_dir.mkdir()
            (source / "config.json").write_text("{}")
            (source / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            nested = source / "nested"
            nested.mkdir()
            (nested / "tokenizer.json").write_bytes(b"tokenizer")
            service = {
                "model_staging": {
                    "enabled": True,
                    "source_model_path": str(source),
                    "target_path": str(target),
                    "reuse_existing": True,
                    "clean": True,
                    "copy_workers": 3,
                    "copy_chunk_bytes": 2,
                },
                "start_commands": [
                    ["python", "-m", "sglang.launch_server", "--model-path", str(source)]
                ],
            }

            queue._stage_sglang_model(service, run_dir)
            (target / "model-00001-of-00001.safetensors").unlink()
            rebuilt = queue._stage_sglang_model(service, run_dir)

            self.assertTrue((target / "model-00001-of-00001.safetensors").is_file())
            self.assertEqual(
                (target / "nested" / "tokenizer.json").read_bytes(),
                b"tokenizer",
            )
            self.assertFalse(rebuilt["model_staging"]["reused_existing"])

    def test_run_input_staging_rewrites_env_and_reuses_verified_tree(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "critic-source"
            target = root / "critic-staged"
            run_dir = root / "run"
            source.mkdir()
            (source / "config.json").write_text("{}")
            (source / "model.safetensors").write_bytes(b"0123456789")
            spec = {
                "env": {"CRITIC_SL_CKPT": str(source)},
                "run_dir": str(run_dir),
                "input_staging": [
                    {
                        "name": "critic_load",
                        "env_key": "CRITIC_SL_CKPT",
                        "source_path": None,
                        "target_path": str(target),
                        "reuse_existing": True,
                        "clean": True,
                        "copy_workers": 4,
                        "copy_chunk_bytes": 3,
                    }
                ],
            }

            staged = queue._stage_run_inputs(spec)

            self.assertEqual(staged["env"]["CRITIC_SL_CKPT"], str(target))
            self.assertEqual((target / "model.safetensors").read_bytes(), b"0123456789")
            self.assertEqual(staged["input_staging_report"]["status"], "complete")
            self.assertFalse(
                staged["input_staging_report"]["entries"][0]["reused_existing"]
            )

            reused = queue._stage_run_inputs(spec)

            self.assertTrue(
                reused["input_staging_report"]["entries"][0]["reused_existing"]
            )
            report = json.loads((run_dir / "input_staging.json").read_text())
            self.assertEqual(report["status"], "complete")

    def test_input_and_sglang_staging_share_a_verified_target(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "actor-source"
            target = root / "actor-staged"
            run_dir = root / "run"
            source.mkdir()
            (source / "config.json").write_text("{}")
            (source / "model.safetensors").write_bytes(b"actor-weights")
            spec = {
                "env": {"ACTOR_REF_CKPT": str(source)},
                "run_dir": str(run_dir),
                "input_staging": [
                    {
                        "name": "actor_reference",
                        "env_key": "ACTOR_REF_CKPT",
                        "source_path": None,
                        "target_path": str(target),
                        "reuse_existing": True,
                        "clean": True,
                        "copy_workers": 2,
                        "copy_chunk_bytes": 4,
                    }
                ],
            }
            service = {
                "model_staging": {
                    "enabled": True,
                    "source_model_path": str(source),
                    "target_path": str(target),
                    "reuse_existing": True,
                    "clean": True,
                    "copy_workers": 2,
                    "copy_chunk_bytes": 4,
                },
                "start_commands": [
                    ["python", "-m", "sglang.launch_server", "--model-path", str(source)]
                ],
            }

            queue._stage_run_inputs(spec)
            staged_service = queue._stage_sglang_model(service, run_dir)

            self.assertTrue(staged_service["model_staging"]["reused_existing"])
            self.assertEqual(staged_service["start_commands"][0][-1], str(target))
            self.assertTrue((target / ".nla_input_stage.json").is_file())
            self.assertEqual((target / "model.safetensors").read_bytes(), b"actor-weights")

    def test_release_staged_model_files_preserves_runtime_metadata(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = root / "staged"
            run_dir = root / "run"
            target.mkdir()
            run_dir.mkdir()
            shard = target / "model-00001-of-00001.safetensors"
            shard.write_bytes(b"weights")
            config = target / "config.json"
            config.write_text("{}")
            service = {
                "model_staging": {
                    "enabled": True,
                    "target_path": str(target),
                    "release_after_health": {
                        "enabled": True,
                        "globs": ["model-*.safetensors"],
                    },
                }
            }

            report = queue._release_sglang_staged_files(service, run_dir)

            self.assertFalse(shard.exists())
            self.assertTrue(config.is_file())
            self.assertEqual(report["released_files"], [str(shard)])
            self.assertEqual(report["released_bytes"], len(b"weights"))
            self.assertTrue((run_dir / "sglang_model_release.json").is_file())

    def test_runtime_contracts_bind_external_trainer_content(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            runtime_file = pathlib.Path(tmp) / "actor.py"
            runtime_file.write_text(
                "from nla.audit_runtime import clip_grad_norm_local_shards\n"
                "clip_grad_norm_local_shards(parameters, 1.0)\n"
            )
            digest = hashlib.sha256(runtime_file.read_bytes()).hexdigest()
            contract = {
                "name": "miles_local_grad_norm",
                "path": str(runtime_file),
                "sha256": digest,
                "contains": [
                    "from nla.audit_runtime import clip_grad_norm_local_shards",
                    "clip_grad_norm_local_shards(",
                ],
                "forbids": ["grad_norm.full_tensor()"],
            }

            queue._validate_runtime_contracts([contract])
            runtime_file.write_text(runtime_file.read_text() + "grad_norm.full_tensor()\n")

            with self.assertRaisesRegex(queue.QueueError, "runtime contract.*sha256"):
                queue._validate_runtime_contracts([contract])

    def test_hero_queue_is_blocked_on_validity_gate(self):
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_fixed_ar_hero_queue_8h100.yaml"
        )
        queue = yaml.safe_load(path.read_text())

        self.assertTrue(all(item["status"] == "blocked" for item in queue["items"]))
        self.assertEqual(queue["items"][-1]["rollout"]["num_rollout"], 256)
        self.assertTrue(
            all("validity_gate_report" in item for item in queue["items"])
        )
        self.assertEqual(
            queue["defaults"]["required_gate_reports"],
            [
                {
                    "path": "/workspace/interp/outputs/nano30b-nla-pilot/validity/r33-corrected-cross-critic-gate.json",
                    "field": "passed",
                    "expected": True,
                }
            ],
        )
        self.assertTrue(
            queue["defaults"]["source"]["require_expected_fingerprint"]
        )
        self.assertEqual(
            queue["defaults"]["post_eval"]["roundtrip"]["baseline_report_json"],
            "/workspace/interp/outputs/nano30b-nla-pilot/validity/r33-sft/roundtrip_v512_t512_hardened_report.json",
        )

    def test_corrected_342_hero_queue_is_armed_but_fail_closed(self):
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_corrected_k3_hero_342_queue_8h100.yaml"
        )
        queue = yaml.safe_load(path.read_text())
        defaults = queue["defaults"]
        item = queue["items"][0]

        self.assertEqual(
            defaults["code_root"],
            "/workspace/interp/code/nano30b-nla-pilot-hero-current",
        )
        self.assertEqual(defaults["workspace_gpus"], 8)
        self.assertEqual(defaults["resources"]["actor_gpus"], 6)
        self.assertEqual(defaults["resources"]["critic_gpus"], 1)
        self.assertEqual(defaults["resources"]["rollout_gpus"], 1)
        self.assertEqual(defaults["training"]["actor_lr"], "1e-5")
        self.assertEqual(defaults["training"]["actor_micro_batch"], 32)
        self.assertEqual(defaults["training"]["kl_loss_type"], "k3")
        self.assertEqual(defaults["training"]["kl_loss_coef"], 0.001)
        self.assertEqual(defaults["rollout"]["global_batch_size"], 384)
        self.assertEqual(defaults["rollout"]["n_samples_per_prompt"], 8)
        self.assertEqual(item["rollout"]["num_rollout"], 342)
        self.assertEqual(item["checkpoint_retention"]["keep_iterations"], [171, 342])
        self.assertIn(item["status"], {"pending", "running", "complete", "failed"})
        self.assertTrue(item["launch"]["requires_approval"])
        self.assertTrue(item["launch"]["approved"])
        self.assertEqual(
            [report["field"] for report in defaults["required_gate_reports"]],
            ["passed", "passed", "gate.passed"],
        )
        self.assertIn(
            "diagnostics/update16_v512_hardened/report.json",
            defaults["required_gate_reports"][-1]["path"],
        )
        self.assertEqual(
            defaults["source"]["expected_code_sha256"],
            "aef659279c9306f4818812b0b9eb0cbd24df0d857c562d323f4da221524c32a4",
        )
        self.assertEqual(
            defaults["source"]["frozen_git_commit"],
            "25bb58c84c3dc7276f6f8ec6d28630dd747f2894",
        )
        self.assertEqual(len(defaults["runtime_contracts"]), 2)
        self.assertTrue(
            defaults["sglang"]["model_staging"]["release_after_health"]["enabled"]
        )

        evals = item["post_eval"]["roundtrip"]["checkpoints"]
        self.assertEqual([entry["iteration"] for entry in evals], [171, 342, 342])
        self.assertEqual(
            [(entry["validation_limit"], entry["test_limit"]) for entry in evals],
            [(64, 64), (64, 64), (512, 512)],
        )
        self.assertTrue(evals[-1]["require_previous_gate_pass"])

    def test_corrected_342_hero_guard3_retry_is_storage_safe_and_fail_closed(self):
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_corrected_k3_hero_342_guard3_retry1_queue_8h100.yaml"
        )
        queue = yaml.safe_load(path.read_text())
        defaults = queue["defaults"]
        item = queue["items"][0]

        self.assertEqual(defaults["workspace_gpus"], 8)
        self.assertEqual(defaults["resources"]["actor_gpus"], 6)
        self.assertEqual(defaults["resources"]["critic_gpus"], 1)
        self.assertEqual(defaults["resources"]["rollout_gpus"], 1)
        self.assertEqual(defaults["training"]["actor_lr"], "1e-5")
        self.assertEqual(defaults["training"]["kl_loss_type"], "k3")
        self.assertEqual(defaults["training"]["kl_loss_coef"], 0.001)

        guards = {
            entry["metric"]: entry for entry in defaults["training"]["guard_rules"]
        }
        self.assertEqual(guards["train/kl_loss"]["threshold"], 5.0)
        self.assertEqual(guards["train/kl_loss"]["consecutive_steps"], 3)
        self.assertEqual(guards["train/grad_norm"]["threshold"], 100.0)
        self.assertEqual(guards["train/grad_norm"]["consecutive_steps"], 2)
        self.assertEqual(
            defaults["training"]["drift_guard"]["max_logprob_abs_diff"], 0.75
        )
        self.assertEqual(
            defaults["training"]["drift_guard"]["consecutive_steps"], 2
        )

        self.assertEqual(item["rollout"]["num_rollout"], 342)
        self.assertEqual(item["training"]["save_interval"], 114)
        self.assertEqual(
            list(range(
                item["training"]["save_interval"],
                item["rollout"]["num_rollout"] + 1,
                item["training"]["save_interval"],
            )),
            [114, 228, 342],
        )
        self.assertEqual(
            item["checkpoint_retention"]["keep_iterations"], [114, 342]
        )
        self.assertEqual(item["checkpoint_retention"]["max_transient_checkpoints"], 3)
        self.assertIn(item["status"], {"pending", "running", "complete", "failed"})
        self.assertTrue(item["launch"]["approved"])
        self.assertIn("guard3_retry1", item["run_dir"])

        evals = item["post_eval"]["roundtrip"]["checkpoints"]
        self.assertEqual([entry["iteration"] for entry in evals], [114, 342, 342])
        self.assertEqual(
            [(entry["validation_limit"], entry["test_limit"]) for entry in evals],
            [(64, 64), (64, 64), (512, 512)],
        )
        self.assertTrue(evals[-1]["require_previous_gate_pass"])

    def test_corrected_342_hero_lengthcap_retry_replaces_relative_length_guard(self):
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_corrected_k3_hero_342_guard3_lengthcap_retry2_queue_8h100.yaml"
        )
        queue = yaml.safe_load(path.read_text())
        defaults = queue["defaults"]
        item = queue["items"][0]

        rollout_guards = {
            entry["metric"]: entry for entry in defaults["rollout"]["guard_rules"]
        }
        self.assertFalse(
            any(entry["comparison"] == "increasing" for entry in rollout_guards.values())
        )
        self.assertEqual(
            rollout_guards["rollout/nla_response_length/p95"],
            {
                "metric": "rollout/nla_response_length/p95",
                "comparison": "max",
                "threshold": 230.0,
                "consecutive_steps": 2,
                "role_prefixes": ["rollout"],
            },
        )
        self.assertEqual(
            rollout_guards["rollout/nla_status/truncated_frac"],
            {
                "metric": "rollout/nla_status/truncated_frac",
                "comparison": "max",
                "threshold": 0.05,
                "consecutive_steps": 2,
                "role_prefixes": ["rollout"],
            },
        )

        train_guards = {
            entry["metric"]: entry for entry in defaults["training"]["guard_rules"]
        }
        self.assertEqual(train_guards["train/kl_loss"]["consecutive_steps"], 3)
        self.assertEqual(train_guards["train/grad_norm"]["threshold"], 100.0)
        self.assertEqual(item["training"]["save_interval"], 114)
        self.assertEqual(item["checkpoint_retention"]["keep_iterations"], [114, 342])
        self.assertIn(item["status"], {"pending", "running", "complete", "failed"})
        self.assertTrue(item["launch"]["approved"])
        self.assertIn("lengthcap_retry2", item["run_dir"])

    def test_corrected_342_hero_resume228_removes_length_abort_only(self):
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_corrected_k3_hero_342_resume228_retry3_queue_8h100.yaml"
        )
        queue = yaml.safe_load(path.read_text())
        defaults = queue["defaults"]
        item = queue["items"][0]

        rollout_guards = {
            entry["metric"]: entry for entry in defaults["rollout"]["guard_rules"]
        }
        self.assertNotIn("rollout/nla_response_length/p95", rollout_guards)
        self.assertEqual(
            rollout_guards["rollout/nla_status/truncated_frac"],
            {
                "metric": "rollout/nla_status/truncated_frac",
                "comparison": "max",
                "threshold": 0.05,
                "consecutive_steps": 2,
                "role_prefixes": ["rollout"],
            },
        )
        self.assertIn("rollout/nla_parse/closed_frac", rollout_guards)
        self.assertIn("rollout/nla_parse/usable_frac", rollout_guards)

        previous_actor = (
            "/workspace/interp/outputs/nano30b-nla-pilot/rl_hero/"
            "r33_corrected_k3_hero_lr1e5_update342_guard3_lengthcap_retry2/actor"
        )
        self.assertEqual(item["actor_load_ckpt"], previous_actor)
        self.assertEqual(
            item["actor_sft_ckpt"], f"{previous_actor}/iter_0000228"
        )
        self.assertEqual(
            item["actor_sidecar_source"], f"{previous_actor}/iter_0000228"
        )
        self.assertFalse(defaults["training"]["finetune"])
        self.assertTrue(defaults["training"]["no_load_optim"])
        self.assertEqual(item["rollout"]["num_rollout"], 342)
        self.assertEqual(item["training"]["save_interval"], 114)
        self.assertEqual(item["checkpoint_retention"]["keep_iterations"], [342])
        self.assertIn(item["status"], {"pending", "running", "complete", "failed"})
        self.assertTrue(item["launch"]["approved"])

        evals = item["post_eval"]["roundtrip"]["checkpoints"]
        self.assertEqual([entry["iteration"] for entry in evals], [342, 342])
        self.assertEqual(
            [(entry["validation_limit"], entry["test_limit"]) for entry in evals],
            [(64, 64), (512, 512)],
        )
        self.assertTrue(evals[-1]["require_previous_gate_pass"])

    def test_hero_retry_cleanup_policy_protects_selected_checkpoints(self):
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_hero_retry_prelaunch_cleanup_policy.json"
        )
        policy = json.loads(path.read_text())

        self.assertEqual(
            policy["output_root"],
            "/workspace/interp/outputs/nano30b-nla-pilot",
        )
        self.assertEqual(len(policy["candidates"]), 3)
        self.assertTrue(all(path.endswith("/model") for path in policy["candidates"]))
        self.assertIn(
            "r33_corrected_k3_confirm_lr1e5_update32_runtimefix_retry4_keyedloss_mem768",
            policy["current_best"],
        )
        self.assertGreaterEqual(len(policy["protected"]), 5)
        self.assertTrue(
            any("actor_sft_hf_iter_0001291" in path for path in policy["protected"])
        )
        self.assertTrue(
            any("independent" in path and "iter_0001289" in path for path in policy["protected"])
        )

    def test_corrected_confirmation_has_distinct_keyed_loss_retry(self):
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_corrected_k3_hpo_queue_8h100.yaml"
        )
        queue = yaml.safe_load(path.read_text())
        items = {item["name"]: item for item in queue["items"]}
        item = items[
            "r33-corrected-k3-confirm-lr1e5-update32-runtimefix-retry4-keyedloss-mem768"
        ]

        self.assertIn(item["status"], {"pending", "running", "complete", "failed"})
        self.assertTrue(item["launch"]["approved"])
        self.assertEqual(item["rollout"]["num_rollout"], 32)
        self.assertEqual(item["training"]["actor_lr"], "1e-5")
        self.assertEqual(item["training"]["kl_loss_type"], "k3")
        self.assertIn("retry4_keyedloss_mem768", item["run_dir"])
        self.assertIn("retry4-keyedloss-mem768", item["wandb"]["run_id"])
        self.assertNotIn("depends_on", item)
        self.assertEqual(
            queue["defaults"]["code_root"],
            "/workspace/interp/code/nano30b-nla-pilot-hero-current",
        )

    def test_irregular_save_iterations_are_config_driven(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            code_root = root / "code"
            nla_root = code_root / "external" / "natural_language_autoencoders"
            miles_root = root / "miles"
            (nla_root / "configs").mkdir(parents=True)
            miles_root.mkdir()
            (nla_root / "configs" / "rl.sh").write_text("#!/usr/bin/env bash\n")
            (miles_root / "train.py").write_text("print('train')\n")
            rl_parquet = root / "rl.parquet"
            rl_parquet.write_text("placeholder")
            actor = root / "actor"
            critic = root / "critic"
            model = root / "model"
            for path in (actor, critic, model):
                path.mkdir()
            queue_path = self._write_queue(
                root,
                f"""
                schema_version: nano_rl_queue.v1
                defaults:
                  code_root: {code_root}
                  miles_root: {miles_root}
                  rl_script: external/natural_language_autoencoders/configs/rl.sh
                  workspace_gpus: 3
                  resources:
                    actor_gpus: 1
                    critic_gpus: 1
                    rollout_gpus: 1
                  training:
                    actor_micro_batch: 1
                    clip_grad: 0.8
                    save_interval: null
                    save_iterations: [2, 5]
                  rollout:
                    rollout_batch_size: 1
                    global_batch_size: 1
                    n_samples_per_prompt: 1
                    num_rollout: 5
                items:
                  - name: irregular-saves
                    rl_parquet: {rl_parquet}
                    instruct_model: {model}
                    actor_sft_ckpt: {actor}
                    critic_sl_ckpt: {critic}
                    run_dir: {root / "run"}
                    checkpoint_retention:
                      enabled: true
                      keep_iterations: [2, 5]
                      max_transient_checkpoints: 2
                """,
            )
            queue_doc = queue.load_queue(queue_path)
            spec = queue.build_run_spec(
                queue_doc,
                queue_doc["items"][0],
                queue_path=queue_path,
            )

        self.assertEqual(spec["env"]["SAVE_INTERVAL"], "")
        self.assertEqual(spec["env"]["NLA_SAVE_ITERATIONS"], "2,5")
        self.assertEqual(spec["env"]["CLIP_GRAD"], "0.8")
        self.assertEqual(
            spec["checkpoint_retention"]["expected_saved_iterations"],
            [2, 5],
        )
        self.assertIsNone(spec["checkpoint_retention"]["save_interval"])

    def test_preregistration_seals_test_and_hashes_commitments(self):
        queue = load_script("nano_rl_queue")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            artifacts = {}
            for name, payload in (
                ("registration", "registered\n"),
                ("family", "families\n"),
                ("coverage", "coverage\n"),
                ("baseline", "{}\n"),
                ("family_seal", '{"passed": true}\n'),
                ("kernel_compatibility", '{"passed": true}\n'),
                ("power", '{"passed": true, "power": 0.9}\n'),
            ):
                path = root / f"{name}.json"
                path.write_text(payload)
                artifacts[name] = path

            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            training = {
                "save_interval": None,
                "save_iterations": [2],
                "actor_lr": "1e-5",
                "kl_loss_type": "k3",
                "kl_loss_coef": 0.001,
                "actor_micro_batch": 1,
                "drift_guard": {
                    "enabled": True,
                    "max_logprob_abs_diff": 0.75,
                    "consecutive_steps": 2,
                },
                "guard_rules": [
                    {
                        "metric": "train/kl_loss",
                        "comparison": "max",
                        "threshold": 5.0,
                        "consecutive_steps": 2,
                    },
                    {
                        "metric": "train/grad_norm",
                        "comparison": "max",
                        "threshold": 10.0,
                        "consecutive_steps": 2,
                    },
                ],
            }
            rollout = {
                "seed": 314159,
                "num_rollout": 2,
                "global_batch_size": 8,
                "n_samples_per_prompt": 2,
                "max_response_len": 256,
                "max_context_len": 512,
                "guard_rules": [
                    {
                        "metric": "rollout/nla_parse/closed_frac",
                        "comparison": "min",
                        "threshold": 0.95,
                        "consecutive_steps": 1,
                    },
                    {
                        "metric": "rollout/nla_parse/usable_frac",
                        "comparison": "min",
                        "threshold": 0.99,
                        "consecutive_steps": 1,
                    },
                    {
                        "metric": "rollout/nla_status/truncated_frac",
                        "comparison": "max",
                        "threshold": 0.05,
                        "consecutive_steps": 1,
                    },
                ],
            }
            preregistration = {
                "schema_version": "nano_rl_preregistration.v1",
                "phase": "stability_probe",
                "registration_path": str(artifacts["registration"]),
                "registration_sha256": digest(artifacts["registration"]),
                "registered_seed": 314159,
                "selection_split": "validation",
                "allowed_eval_splits": ["validation"],
                "test_policy": "sealed",
                "content_family_manifest": str(artifacts["family"]),
                "content_family_manifest_sha256": digest(artifacts["family"]),
                "content_family_coverage": str(artifacts["coverage"]),
                "content_family_coverage_sha256": digest(artifacts["coverage"]),
                "sft_baseline_report": str(artifacts["baseline"]),
                "sft_baseline_report_sha256": digest(artifacts["baseline"]),
                "family_seal_report": str(artifacts["family_seal"]),
                "family_seal_report_sha256": digest(artifacts["family_seal"]),
                "kernel_compatibility_report": str(
                    artifacts["kernel_compatibility"]
                ),
                "kernel_compatibility_report_sha256": digest(
                    artifacts["kernel_compatibility"]
                ),
                "guard_failure_action": "abort",
                "required_actor_guard_metrics": [
                    "train/train_rollout_logprob_abs_diff",
                    "train/kl_loss",
                    "train/grad_norm",
                ],
                "required_rollout_guard_metrics": [
                    "rollout/nla_parse/closed_frac",
                    "rollout/nla_parse/usable_frac",
                    "rollout/nla_status/truncated_frac",
                ],
                "checkpoint_iterations": [2],
                "primary_endpoint": "validation directional gain",
                "secondary_endpoints": ["raw centered R2"],
            }
            defaults = {
                "preregistration": preregistration,
                "post_eval": {
                    "roundtrip": {
                        "enabled": True,
                        "eval_splits": ["validation"],
                    }
                },
            }
            item = {}
            rules = queue._metric_guard_rules(training, rollout)
            contract = queue._preregistration_config(
                defaults,
                item,
                queue_path=root / "queue.yaml",
                code_root=root,
                training=training,
                rollout=rollout,
                runtime_env={
                    "NLA_FAILED_EXTRACTION_REWARD": "-2.0",
                    "NLA_ASSERT_PACKED_EQUIV": "1",
                    "NLA_ASSERT_ACTOR_PACKED_EQUIV": "1",
                    "NLA_TRAIN_MAMBA_KERNEL_MODE": "unfused_torch_conv",
                },
                metric_guard_rules=rules,
                roundtrip_configs=queue._roundtrip_post_eval_configs(
                    defaults,
                    item,
                ),
            )

            self.assertEqual(contract["test_policy"], "sealed")
            self.assertEqual(contract["allowed_eval_splits"], ["validation"])
            self.assertEqual(contract["checkpoint_iterations"], [2])
            self.assertEqual(
                contract["registration_sha256"],
                digest(artifacts["registration"]),
            )
            self.assertEqual(len(contract["guard_policy_sha256"]), 64)

            preregistration["phase"] = "confirmatory_train"
            preregistration["guard_policy_sha256"] = contract[
                "guard_policy_sha256"
            ]
            preregistration["power_report"] = str(artifacts["power"])
            preregistration["power_report_sha256"] = digest(artifacts["power"])
            confirmatory = queue._preregistration_config(
                defaults,
                item,
                queue_path=root / "queue.yaml",
                code_root=root,
                training=training,
                rollout=rollout,
                runtime_env={
                    "NLA_FAILED_EXTRACTION_REWARD": "-2.0",
                    "NLA_ASSERT_PACKED_EQUIV": "1",
                    "NLA_ASSERT_ACTOR_PACKED_EQUIV": "1",
                    "NLA_TRAIN_MAMBA_KERNEL_MODE": "unfused_torch_conv",
                },
                metric_guard_rules=rules,
                roundtrip_configs=queue._roundtrip_post_eval_configs(
                    defaults,
                    item,
                ),
            )
            self.assertIn("power_report", confirmatory["artifacts"])

            defaults["post_eval"]["roundtrip"]["eval_splits"] = [
                "validation",
                "test",
            ]
            with self.assertRaisesRegex(queue.QueueError, "test split is sealed"):
                queue._preregistration_config(
                    defaults,
                    item,
                    queue_path=root / "queue.yaml",
                    code_root=root,
                    training=training,
                    rollout=rollout,
                    runtime_env={
                        "NLA_FAILED_EXTRACTION_REWARD": "-2.0",
                        "NLA_ASSERT_PACKED_EQUIV": "1",
                        "NLA_ASSERT_ACTOR_PACKED_EQUIV": "1",
                        "NLA_TRAIN_MAMBA_KERNEL_MODE": "unfused_torch_conv",
                    },
                    metric_guard_rules=rules,
                    roundtrip_configs=queue._roundtrip_post_eval_configs(
                        defaults,
                        item,
                    ),
                )

    def test_retention_keeps_every_evaluated_and_registered_iteration(self):
        queue = load_script("nano_rl_queue")
        run_spec = {
            "checkpoint_retention": {
                "enabled": True,
                "keep_iterations": [16],
            },
            "post_eval_specs": [{"iteration": 32}],
            "preregistration": {"checkpoint_iterations": [16, 64]},
        }

        with self.assertRaisesRegex(
            queue.QueueError,
            r"evaluated or registered iterations: \[32, 64\]",
        ):
            queue.validate_retention_protected_iterations(run_spec)

        run_spec["checkpoint_retention"]["keep_iterations"] = [16, 32, 64]
        queue.validate_retention_protected_iterations(run_spec)

    def test_actor_schedule_duration_configs_couple_save_and_rollout_horizons(self):
        config_root = ROOT / "configs" / "nano_rl" / "hpo"
        paths = [
            config_root / "r33_family_clean_online_joint_actor_schedule_hpo_dev_queue_8h100.yaml",
            config_root
            / "r33_family_clean_online_joint_actor_schedule_a3e5_u24_promotion_dev_queue_8h100.yaml",
        ]

        duration_items = []
        for path in paths:
            queue_doc = yaml.safe_load(path.read_text())
            duration_items.extend(
                item
                for item in queue_doc["items"]
                if item["name"].endswith("u24")
            )

        self.assertEqual(len(duration_items), 3)
        for item in duration_items:
            with self.subTest(name=item["name"]):
                self.assertEqual(item["training"]["save_iterations"], [24])
                self.assertEqual(item["rollout"]["num_rollout"], 24)
                self.assertEqual(
                    item["checkpoint_retention"]["keep_iterations"], [24]
                )

    def test_internal_family_clean_hero_matches_promoted_regime(self):
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "hero"
            / "r33_family_clean_online_joint_a3e5_internal_hero_u342_queue_8h100.yaml"
        )
        queue_doc = yaml.safe_load(path.read_text())
        self.assertEqual(queue_doc["schema_version"], "nano_rl_queue.v1")
        self.assertEqual(len(queue_doc["items"]), 1)

        defaults = queue_doc["defaults"]
        item = queue_doc["items"][0]
        roundtrip = item["post_eval"]["roundtrip"]

        self.assertEqual(item["name"], "r33-family-clean-online-joint-a3e5-internal-hero-u342")
        self.assertEqual(item["status"], "pending")
        self.assertFalse(item["launch"]["approved"])
        self.assertEqual(item["training"]["actor_lr"], "3e-5")
        self.assertEqual(item["training"]["save_iterations"], [342])
        self.assertEqual(item["rollout"]["num_rollout"], 342)
        self.assertEqual(item["rollout"]["seed"], 20260719)
        self.assertEqual(defaults["rollout"]["rollout_batch_size"], 24)
        self.assertEqual(defaults["rollout"]["global_batch_size"], 192)
        self.assertEqual(defaults["rollout"]["n_samples_per_prompt"], 8)
        self.assertEqual(item["checkpoint_retention"]["keep_iterations"], [342])
        self.assertFalse(roundtrip["cleanup_actor_checkpoint"])
        self.assertEqual(roundtrip["eval_splits"], ["validation"])
        self.assertEqual(roundtrip["validation_limit"], 122)
        self.assertNotIn("preregistration", item)


if __name__ == "__main__":
    unittest.main()
