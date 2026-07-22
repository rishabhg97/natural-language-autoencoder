import pathlib
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class R33CorrectedStage2QueueTests(unittest.TestCase):
    def test_queue_contains_complete_fail_closed_promotion_evidence(self):
        queue_path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_corrected_stage2_analysis_queue_8h100.yaml"
        )
        queue = yaml.safe_load(queue_path.read_text())
        items = {item["name"]: item for item in queue["items"]}
        expected = {
            "enrich-sft-v512",
            "enrich-confirm16-v512",
            "independent-sft-roundtrip-v512",
            "independent-confirm16-roundtrip-v512",
            "confirm16-invariance-v512",
            "confirm16-functional-v512",
            "confirm16-response-closure",
            "confirm16-cross-critic-gate",
            "confirm16-qualitative-panel-unreviewed",
            "confirm16-qualitative-structural-decisions",
            "confirm16-qualitative-panel-reviewed",
            "confirm16-composite-validity-gate",
        }

        self.assertEqual(queue["schema_version"], "nano_ar_layer_sweep_queue.v1")
        self.assertEqual(
            queue["defaults"]["code_root"],
            "/workspace/interp/code/nano30b-nla-pilot-hero-current",
        )
        self.assertEqual(set(items), expected)
        independent = items["independent-confirm16-roundtrip-v512"]["command"]
        self.assertIn("--reuse-generated", independent)
        self.assertTrue(any("mb48-4gpu-cudablock-resume393" in part for part in independent))
        independent_sft = items["independent-sft-roundtrip-v512"]["command"]
        self.assertTrue(any("mb48-4gpu-cudablock-resume393" in part for part in independent_sft))
        self.assertIn("--require-clustered-baseline-ci", independent)
        self.assertIn("--require-baseline-dataset-match", independent)
        self.assertIn("0.05", independent)
        cross = items["confirm16-cross-critic-gate"]["command"]
        self.assertIn("scripts/eval_nano_cross_critic_gate.py", cross)
        reviewed = items["confirm16-qualitative-panel-reviewed"]["command"]
        self.assertIn("--reviews-json", reviewed)
        structural = items["confirm16-qualitative-structural-decisions"]["command"]
        self.assertIn("scripts/auto_review_nano_qualitative_panel.py", structural)
        rendered = queue_path.read_text()
        self.assertNotIn("train_actor", rendered)
        self.assertNotIn("nano_rl_queue.py", rendered)

    def test_validity_config_points_to_corrected_confirmation(self):
        path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_corrected_validity_eval.yaml"
        )
        config = yaml.safe_load(path.read_text())
        candidate = config["evaluations"]["candidates"]["confirmation16"]

        self.assertEqual(config["thresholds"]["required_rows"], 512)
        self.assertEqual(config["thresholds"]["roundtrip_relative_improvement"], 0.10)
        self.assertIn(
            "diagnostics/update16_v512_hardened/report.json",
            candidate["roundtrip_report"],
        )
        self.assertIn("r33-corrected-confirm16", candidate["invariance_report"])


if __name__ == "__main__":
    unittest.main()
