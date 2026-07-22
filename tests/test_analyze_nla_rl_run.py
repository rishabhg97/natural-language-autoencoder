from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "analyze_nla_rl_run.py"
    spec = importlib.util.spec_from_file_location("analyze_nla_rl_run_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnalyzeNLARLRunTests(unittest.TestCase):
    def test_analyzes_old_and_new_train_log_lines(self):
        module = load_module()
        log = textwrap.dedent(
            """
            [x] log_utils.py:52 - rollout 0: {'rollout/response_lengths': 122.0, 'rollout/raw_reward': -0.48, 'rollout/truncated': 0.0, 'rollout/advantages': 0.0}
            [x] train_metric_utils.py:44 - perf 0: {'perf/actor_train_time': 500.0, 'perf/train_time': 510.0, 'perf/step_time': 600.0}
            [x] train_metric_utils.py:44 - perf 0: {'perf/actor_train_time': 1051.8, 'perf/ref_log_probs_time': 283.4, 'perf/log_probs_time': 242.6, 'perf/step_time': 1729.4}
            [x] log_utils.py:429 - step 0: {'train/loss': 0.3, 'train/fve_nrm': 0.4, 'train/grad_norm': 0.5, 'train/lr-pg_0': 2e-6}
            [x] log_utils.py:429 - step 0: {'train/ppo_kl': 0.0, 'train/kl_loss': 0.0, 'train/pg_clipfrac': 0.0, 'train/grad_norm': 0.0}
            [NLA ROLLOUT] {"rollout_id": 1, "reward_mean": -0.4, "usable_reward_mean": -0.4, "usable_frac": 1.0, "failed_frac": 0.0}
            [NLA ADVANTAGE] scope=rank0_shard rollout/nla_advantage/std=0.7 rollout/nla_advantage/frac_zero=0.1
            [x] WARNING something benign
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "train.log"
            path.write_text(log)
            summary = module.analyze_train_log(path)

        self.assertEqual(summary["rollout_count"], 1)
        self.assertEqual(summary["perf_count"], 2)
        self.assertEqual(summary["step_count"], 2)
        self.assertEqual(summary["nla_rollout_count"], 1)
        self.assertEqual(summary["nla_advantage_count"], 1)
        self.assertEqual(summary["latest_rollout"]["rollout/raw_reward"], -0.48)
        self.assertEqual(summary["latest_perf"]["perf/step_time"], 1729.4)
        self.assertEqual(summary["latest_step"]["train/ppo_kl"], 0.0)
        self.assertEqual(summary["usable_fraction"], 1.0)
        self.assertEqual(summary["failed_fraction"], 0.0)
        self.assertEqual(summary["latest_nla_advantage"]["rollout/nla_advantage/std"], 0.7)
        self.assertEqual(summary["warning_count"], 1)
        self.assertEqual(
            summary["nla_rollout_trajectory"],
            [
                {
                    "rollout_id": 1,
                    "reward_mean": -0.4,
                    "usable_reward_mean": -0.4,
                    "usable_frac": 1.0,
                    "failed_frac": 0.0,
                }
            ],
        )
        self.assertEqual(summary["actor_step_trajectory"][0]["step"], 0)
        self.assertEqual(summary["actor_step_trajectory"][0]["train/ppo_kl"], 0.0)
        self.assertEqual(summary["critic_step_trajectory"][0]["train/fve_nrm"], 0.4)
        self.assertEqual(summary["actor_perf_trajectory"][0]["perf/step_time"], 1729.4)
        self.assertEqual(summary["critic_perf_trajectory"][0]["perf/train_time"], 510.0)

        compact = module.compact_summary(summary)
        self.assertNotIn("latest_perf", compact)
        self.assertEqual(compact["critic_step_trajectory"], summary["critic_step_trajectory"])


if __name__ == "__main__":
    unittest.main()
