from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_rl_reward_gate_correlation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_rl_reward_gate_correlation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RewardGateCorrelationTests(unittest.TestCase):
    def test_spearman_positive_and_negative(self):
        module = load_module()

        self.assertAlmostEqual(module.spearman_corr([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(module.spearman_corr([1, 2, 3], [30, 20, 10]), -1.0)

    def test_pairs_report_losses_with_reward_rows(self):
        module = load_module()
        split = {
            "row_indices": [10, 11, 12],
            "rowwise_normalized_mse": {"av_real": [0.3, 0.2, 0.1]},
        }
        rewards = {12: -0.1, 10: -0.5}

        pairs = module.pair_rewards_with_gate_losses(split, rewards)

        self.assertEqual(pairs["row_indices"], [10, 12])
        self.assertEqual(pairs["rewards"], [-0.5, -0.1])
        self.assertEqual(pairs["gate_losses"], [0.3, 0.1])

    def test_run_analysis_with_stubbed_rewards(self):
        module = load_module()
        report = {
            "splits": {
                "validation": {
                    "row_indices": [10, 11, 12],
                    "rowwise_normalized_mse": {"av_real": [0.3, 0.2, 0.1]},
                },
                "test": {
                    "row_indices": [20, 21, 22],
                    "rowwise_normalized_mse": {"av_real": [0.1, 0.2, 0.3]},
                },
            }
        }
        rewards = {
            "validation": {10: -0.5, 11: -0.3, 12: -0.1},
            "test": {20: -0.5, 21: -0.3, 22: -0.1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            report_path = pathlib.Path(tmp) / "report.json"
            output_path = pathlib.Path(tmp) / "out.json"
            report_path.write_text(json.dumps(report))

            summary = module.run_analysis(
                roundtrip_report_json=report_path,
                reward_loader=lambda: rewards,
                output_json=output_path,
            )

            self.assertTrue(output_path.exists())

        self.assertEqual(summary["interpretation"], "fixed_policy_correlation_not_policy_gradient_proof")
        self.assertEqual(summary["splits"]["validation"]["paired_row_count"], 3)
        self.assertAlmostEqual(summary["splits"]["validation"]["spearman_reward_vs_gate_loss"], -1.0)
        self.assertAlmostEqual(summary["splits"]["test"]["spearman_reward_vs_gate_loss"], 1.0)


if __name__ == "__main__":
    unittest.main()
