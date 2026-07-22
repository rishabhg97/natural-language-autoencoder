import importlib.util
import io
import pathlib
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "external" / "natural_language_autoencoders" / "nla" / "system_metrics.py"
    spec = importlib.util.spec_from_file_location("nla_system_metrics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SystemMetricsTests(unittest.TestCase):
    def test_disabled_logger_returns_no_metrics(self):
        module = load_module()
        logger = module.SystemMetricsLogger(enabled=False)

        self.assertEqual(logger.collect(step_id=1), {})

    def test_step_interval_controls_collection(self):
        module = load_module()
        logger = module.SystemMetricsLogger(enabled=True, interval_steps=4, include_nvidia_smi=False)

        self.assertFalse(logger.should_collect(1))
        self.assertTrue(logger.should_collect(4))

    def test_logger_can_be_configured_from_train_env_mapping(self):
        module = load_module()

        logger = module.SystemMetricsLogger.from_env_mapping(
            {
                "NLA_SYSTEM_METRICS": "1",
                "NLA_SYSTEM_METRICS_INTERVAL_STEPS": "3",
                "NLA_SYSTEM_METRICS_NVSMI_INTERVAL_STEPS": "2",
            },
            rank=7,
            local_rank=1,
            role="actor",
        )

        self.assertTrue(logger.enabled)
        self.assertEqual(logger.interval_steps, 3)
        self.assertEqual(logger.nvidia_smi_interval_steps, 2)
        self.assertEqual(logger.rank, 7)
        self.assertEqual(logger.local_rank, 1)
        self.assertEqual(logger.role, "actor")
        self.assertTrue(logger.include_nvidia_smi)

    def test_collect_includes_static_topology_metrics_from_env(self):
        module = load_module()
        logger = module.SystemMetricsLogger(enabled=True, include_nvidia_smi=False)

        with mock.patch.dict(
            "os.environ",
            {
                "NLA_WORKSPACE_GPUS": "8",
                "NLA_ACTOR_GPUS": "5",
                "NLA_CRITIC_GPUS": "2",
                "NLA_ROLLOUT_GPUS": "1",
                "NLA_ROLLOUT_GPUS_PER_ENGINE": "1",
                "NLA_SGLANG_TP_SIZE": "1",
                "NLA_SGLANG_BASE_GPU_ID": "5",
            },
            clear=False,
        ):
            metrics = logger.collect(step_id=1)

        self.assertEqual(metrics["nla/system/topology_workspace_gpus"], 8)
        self.assertEqual(metrics["nla/system/topology_actor_gpus"], 5)
        self.assertEqual(metrics["nla/system/topology_critic_gpus"], 2)
        self.assertEqual(metrics["nla/system/topology_rollout_gpus"], 1)
        self.assertEqual(metrics["nla/system/topology_rollout_gpus_per_engine"], 1)
        self.assertEqual(metrics["nla/system/topology_sglang_tp_size"], 1)
        self.assertEqual(metrics["nla/system/topology_sglang_base_gpu_id"], 5)

    def test_appends_metrics_to_miles_structured_loss_dict(self):
        module = load_module()
        try:
            torch = __import__("torch")
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this local test environment")
        loss_dict = {
            "keys": ["loss"],
            "values": torch.tensor([4.0, 8.0], dtype=torch.float32),
        }

        updated = module.append_metrics_to_miles_loss_dict(
            loss_dict,
            {
                "nla/system/cuda_memory_allocated_gib": 12.5,
                "nla/system/rank": 2,
            },
        )

        self.assertIs(updated, loss_dict)
        self.assertEqual(
            updated["keys"],
            ["loss", "nla/system/cuda_memory_allocated_gib", "nla/system/rank"],
        )
        self.assertTrue(
            torch.equal(
                updated["values"],
                torch.tensor([4.0, 8.0, 50.0, 8.0], dtype=torch.float32),
            )
        )

    def test_appends_metrics_without_torch_falls_back_to_flat_scalars(self):
        module = load_module()
        loss_dict = {
            "keys": ["loss"],
            "values": [4.0, 8.0],
        }

        updated = module.append_metrics_to_miles_loss_dict(
            loss_dict,
            {
                "nla/system/cuda_memory_allocated_gib": 50.0,
                "nla/system/rank": 8,
            },
        )

        self.assertEqual(updated["keys"], ["loss"])
        self.assertEqual(updated["values"], [4.0, 8.0])
        self.assertEqual(updated["nla/system/cuda_memory_allocated_gib"], 50.0)
        self.assertEqual(updated["nla/system/rank"], 8)

    def test_router_entropy_tracker_collects_integer_router_outputs(self):
        module = load_module()
        try:
            torch = __import__("torch")
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this local test environment")

        tracker = module.RouterEntropyTracker(expert_count=4)
        tracker._hook(None, (), {"indices": torch.tensor([[0, 1], [1, 3]], dtype=torch.long)})

        metrics = tracker.collect()

        self.assertEqual(metrics["nla/router/expert_count"], 4)
        self.assertEqual(metrics["nla/router/active_expert_count"], 3)
        self.assertEqual(metrics["nla/router/token_assignments"], 4)
        self.assertGreater(metrics["nla/router/router_entropy_normalized"], 0.0)

    def test_router_entropy_tracker_reports_per_layer_activity(self):
        module = load_module()
        try:
            torch = __import__("torch")
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in the local test environment")
        tracker = module.RouterEntropyTracker(expert_count=4)
        tracker._expert_count_by_layer["model.layers.0.router"] = 4
        tracker._hook(
            None,
            (),
            {"indices": torch.tensor([[0, 1], [1, 3]], dtype=torch.long)},
            layer_name="model.layers.0.router",
        )

        metrics = tracker.collect()

        prefix = "nla/router/layers/model/layers/0/router"
        self.assertEqual(metrics[f"{prefix}/active_expert_count"], 3)
        self.assertEqual(metrics[f"{prefix}/token_assignments"], 4)

    def test_router_entropy_tracker_is_noop_without_torch(self):
        module = load_module()

        with mock.patch.object(module, "_load_torch", return_value=None):
            tracker = module.RouterEntropyTracker()
            self.assertIsNone(tracker._find_index_tensor(object()))
            tracker._hook(None, (), object())

        self.assertEqual(tracker.collect(), {})

    def test_ru_maxrss_unit_conversion_is_platform_aware(self):
        module = load_module()

        self.assertAlmostEqual(
            module.ru_maxrss_to_gib(1024 * 1024, platform="linux"),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            module.ru_maxrss_to_gib(1024 * 1024 * 1024, platform="darwin"),
            1.0,
            places=6,
        )

    def test_phase_metrics_are_prefixed_and_keep_numeric_extras(self):
        module = load_module()
        logger = module.SystemMetricsLogger(enabled=True, include_nvidia_smi=False, role="actor")

        with mock.patch.object(
            logger,
            "collect",
            return_value={
                "nla/system/rank": 0,
                "nla/system/cuda_memory_allocated_gib": 12.5,
            },
        ):
            metrics = logger.collect_phase(
                step_id=3,
                phase="actor train",
                event="before backward",
                extra={"rollout_id": 7, "ignored_text": "not numeric"},
                force=False,
            )

        self.assertEqual(metrics["nla/phase/actor_train/rank"], 0)
        self.assertEqual(metrics["nla/phase/actor_train/cuda_memory_allocated_gib"], 12.5)
        self.assertEqual(metrics["nla/phase/actor_train/rollout_id"], 7)
        self.assertEqual(metrics["nla/phase/actor_train/event_code"], 1)
        self.assertNotIn("nla/phase/actor_train/ignored_text", metrics)

    def test_all_gpu_nvidia_smi_metrics_include_per_gpu_and_aggregates(self):
        module = load_module()
        sample = "\n".join(
            [
                "0, 1024, 143771, 10, 5, 120.50, 35",
                "1, 2048, 143771, 20, 6, 130.50, 36",
            ]
        )

        metrics = module.parse_all_gpu_nvidia_smi_csv(sample)

        self.assertEqual(metrics["nla/system/gpu0/nvidia_smi_memory_used_mib"], 1024)
        self.assertEqual(metrics["nla/system/gpu1/nvidia_smi_gpu_util_pct"], 20)
        self.assertEqual(metrics["nla/system/all_gpu_count"], 2)
        self.assertEqual(metrics["nla/system/all_gpu_memory_used_mib"], 3072)
        self.assertEqual(metrics["nla/system/all_gpu_util_pct_mean"], 15.0)
        self.assertEqual(metrics["nla/system/all_gpu_util_pct_max"], 20)

    def test_emit_phase_snapshot_prints_json_and_wandb_logs_when_active(self):
        module = load_module()
        logger = module.SystemMetricsLogger(enabled=True, include_nvidia_smi=False, role="critic")

        fake_wandb = mock.Mock()
        fake_wandb.run = object()
        with (
            mock.patch.object(
                logger,
                "collect_phase",
                return_value={
                    "nla/phase/critic_reward/cuda_memory_allocated_gib": 42.0,
                },
            ),
            mock.patch.dict(sys.modules, {"wandb": fake_wandb}),
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                metrics = logger.emit_phase_snapshot(
                    step_id=11,
                    phase="critic_reward",
                    event="end",
                    extra={"rollout_id": 2},
                )

        self.assertEqual(metrics["nla/phase/critic_reward/cuda_memory_allocated_gib"], 42.0)
        self.assertIn("[NLA OBS]", buffer.getvalue())
        self.assertIn('"phase": "critic_reward"', buffer.getvalue())
        fake_wandb.log.assert_called_once_with(
            {"nla/phase/critic_reward/cuda_memory_allocated_gib": 42.0},
            step=11,
        )
