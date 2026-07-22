from __future__ import annotations

import importlib.util
import io
import pathlib
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import pytest

pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"


def load_module():
    path = NLA_ROOT / "nla" / "rollout" / "rl_metrics.py"
    spec = importlib.util.spec_from_file_location("nla_rollout_rl_metrics", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(NLA_ROOT))
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class Sample:
    def __init__(
        self,
        reward: float | None,
        response: str,
        length: int,
        status: str = "COMPLETED",
        metadata: dict | None = None,
    ) -> None:
        self._reward = reward
        self.response = response
        self.effective_response_length = length
        self.status = types.SimpleNamespace(name=status)
        self.metadata = metadata or {}

    def get_reward_value(self, _args):
        return self._reward


class NLARLMetricsTests(unittest.TestCase):
    def test_injects_reward_distribution_and_parse_metrics(self):
        module = load_module()
        metrics: dict[str, float | int] = {}
        samples = [
            Sample(-0.3, "<explanation>a</explanation>", 10),
            Sample(-0.7, "<explanation>b</explanation>", 20),
            Sample(-1.1, "missing close", 30, "FAILED"),
        ]

        skip_default = module.log_rollout_data(0, object(), samples, metrics, 1.0)

        self.assertFalse(skip_default)
        self.assertEqual(metrics["rollout/nla_reward/count"], 3)
        self.assertAlmostEqual(metrics["rollout/nla_reward/mean"], -0.7)
        self.assertAlmostEqual(metrics["rollout/nla_reward/p50"], -0.7)
        self.assertGreater(metrics["rollout/nla_reward/std"], 0.0)
        self.assertEqual(metrics["rollout/nla_usable_reward/count"], 2)
        self.assertAlmostEqual(metrics["rollout/nla_usable_reward/mean"], -0.5)
        self.assertAlmostEqual(metrics["rollout/nla_parse/closed_frac"], 2 / 3)
        self.assertAlmostEqual(metrics["rollout/nla_parse/usable_frac"], 2 / 3)
        self.assertAlmostEqual(metrics["rollout/nla_status/failed_frac"], 1 / 3)
        self.assertIn("rollout/nla_reward/length_corr", metrics)

    def test_none_metrics_keeps_default_logger(self):
        module = load_module()
        sample = Sample(-0.3, "<explanation>a</explanation>", 10)

        self.assertFalse(module.log_rollout_data(0, object(), [sample], None, 1.0))

    def test_preserves_engine_truncation_after_parse_failure_relabel(self):
        module = load_module()
        metrics: dict[str, float | int] = {}
        sample = Sample(-2.0, "<explanation>unfinished", 256, "FAILED")
        sample.nla_generation_status = "TRUNCATED"

        module.log_rollout_data(0, object(), [sample], metrics, 1.0)

        self.assertEqual(metrics["rollout/nla_status/failed_frac"], 1.0)
        self.assertEqual(metrics["rollout/nla_status/truncated_frac"], 0.0)
        self.assertEqual(metrics["rollout/nla_generation/truncated_frac"], 1.0)

    def test_prints_summary_before_guard_failure(self):
        module = load_module()
        metrics: dict[str, float | int] = {}
        sample = Sample(-2.0, "missing tags", 256, "FAILED")

        buffer = io.StringIO()
        with mock.patch.object(
            module, "check_rollout_metrics", side_effect=RuntimeError("guard")
        ):
            with self.assertRaisesRegex(RuntimeError, "guard"):
                with redirect_stdout(buffer):
                    module.log_rollout_data(0, object(), [sample], metrics, 1.0)

        self.assertIn("[NLA ROLLOUT]", buffer.getvalue())
        self.assertIn('"usable_frac": 0.0', buffer.getvalue())

    def test_prints_compact_rollout_summary_to_stdout(self):
        module = load_module()
        metrics: dict[str, float | int] = {}
        samples = [
            Sample(-0.2, "<explanation>a</explanation>", 10),
            Sample(-0.6, "<explanation>b</explanation>", 11),
        ]

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            module.log_rollout_data(4, object(), samples, metrics, 1.0)

        output = buffer.getvalue()
        self.assertIn("[NLA ROLLOUT]", output)
        self.assertIn('"rollout_id": 4', output)
        self.assertIn('"reward_mean": -0.4', output)
        self.assertIn('"usable_reward_mean": -0.4', output)
        self.assertIn('"usable_frac": 1.0', output)

    def test_advantage_stats_from_rollout_data_tensor(self):
        module = load_module()
        import torch

        metrics = module.advantage_stats_from_rollout_data({"advantages": torch.tensor([0.0, 1.0, -1.0, 0.0])})

        self.assertEqual(metrics["rollout/nla_advantage/count"], 4)
        self.assertGreater(metrics["rollout/nla_advantage/std"], 0.0)
        self.assertAlmostEqual(metrics["rollout/nla_advantage/frac_zero"], 0.5)

    def test_advantage_stats_from_rollout_data_tensor_list(self):
        module = load_module()
        import torch

        metrics = module.advantage_stats_from_rollout_data(
            {"advantages": [torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])]}
        )

        self.assertEqual(metrics["rollout/nla_advantage/count"], 4)
        self.assertGreater(metrics["rollout/nla_advantage/std"], 0.0)
        self.assertAlmostEqual(metrics["rollout/nla_advantage/frac_zero"], 0.5)

    def test_advantage_stats_empty_when_missing(self):
        module = load_module()

        self.assertEqual(module.advantage_stats_from_rollout_data({}), {})


if __name__ == "__main__":
    unittest.main()
