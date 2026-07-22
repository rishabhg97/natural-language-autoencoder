import pathlib
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class R33Stage1AnalysisQueueTests(unittest.TestCase):
    def test_checked_in_functional_configs_are_full_512_by_512(self):
        config_root = ROOT / "configs" / "nano_functional"
        for candidate in ("update16", "update32"):
            with self.subTest(candidate=candidate):
                config = yaml.safe_load(
                    (config_root / f"r33_stage1_{candidate}_v512.yaml").read_text()
                )
                self.assertEqual(config["schema_version"], "nano_functional_eval.v1")
                self.assertEqual(config["eval"]["validation_limit"], 512)
                self.assertEqual(config["eval"]["test_limit"], 512)
                self.assertEqual(config["eval"]["boundary"], 33)
                self.assertEqual(config["eval"]["batch_size"], 8)
                self.assertIn("sft_generated_jsonl", config["paths"])

        canary = yaml.safe_load(
            (config_root / "r33_stage1_update16_v16_batch8_canary.yaml").read_text()
        )
        self.assertEqual(canary["eval"]["validation_limit"], 16)
        self.assertEqual(canary["eval"]["test_limit"], 16)
        self.assertEqual(canary["eval"]["batch_size"], 8)
        self.assertEqual(canary["eval"]["selection_strategy"], "longest_prefix")

    def test_analysis_queue_contains_all_evidence_and_no_training(self):
        queue_path = (
            ROOT
            / "configs"
            / "nano_rl"
            / "r33_component_stage1_analysis_queue_8h100.yaml"
        )
        queue = yaml.safe_load(queue_path.read_text())
        names = {item["name"] for item in queue["items"]}

        self.assertEqual(queue["schema_version"], "nano_ar_layer_sweep_queue.v1")
        for candidate in ("update16", "update32"):
            self.assertIn(f"{candidate}-invariance-v512", names)
            self.assertIn(f"{candidate}-functional-v512", names)
        self.assertIn("update16-qualitative-panel-pending", names)
        self.assertIn("update32-qualitative-panel-reviewed", names)
        self.assertIn("update16-functional-batch8-canary", names)
        self.assertIn("update32-response-closure-audit", names)
        closure_item = next(
            item for item in queue["items"]
            if item["name"] == "update32-response-closure-audit"
        )
        self.assertIn("256", closure_item["command"])
        update32_panel = next(
            item for item in queue["items"]
            if item["name"] == "update32-qualitative-panel-reviewed"
        )
        self.assertIn("--reviews-json", update32_panel["command"])
        rendered = queue_path.read_text()
        self.assertNotIn("train_actor", rendered)
        self.assertNotIn("nano_rl_queue.py watch", rendered)


if __name__ == "__main__":
    unittest.main()
