import copy
import importlib.util
import pathlib
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def functional_rows(kl: float, overlap: float):
    return [
        {
            "kl_original_to_patched": kl,
            "top_10_overlap": overlap,
            "top_50_overlap": overlap,
        }
        for _ in range(8)
    ]


def passing_fixture():
    splits = {}
    for split in ("validation", "test"):
        keys = [f"{split}-{index}" for index in range(8)]
        splits[split] = {
            "content_family_ids": [f"{split}-family-{index}" for index in range(8)],
            "row_keys": {
                "sft": keys,
                "candidate": keys,
                "invariance": keys,
                "functional": keys,
            },
            "roundtrip": {
                "sft_nmse": [1.0] * 8,
                "candidate_nmse": [0.8] * 8,
            },
            "functional": {
                "stored_gold": functional_rows(0.05, 0.98),
                "sft": functional_rows(0.4, 0.90),
                "candidate": functional_rows(0.2, 0.895),
            },
            "invariance_retention": {
                "format_normalize": 0.95,
                "unit_reorder": 0.92,
            },
            "control_win_fractions": {
                "mean": 0.95,
                "none": 0.96,
                "shuffled": 0.97,
                "zero": 0.98,
            },
            "parse": {"usable_fraction": 1.0, "closed_fraction": 0.98},
            "leakage": {"injection_marker_count": 0, "cjk_count": 0},
            "qualitative": {
                "row_count": 50,
                "reviewed_count": 50,
                "flagged_count": 2,
            },
        }
    return {
        "candidate_name": "update32",
        "eval_splits": ["validation"],
        "splits": splits,
        "thresholds": {
            "required_rows": 8,
            "roundtrip_relative_improvement": 0.05,
            "functional_topk_max_regression": 0.01,
            "invariance_fve_retention": 0.90,
            "control_win_fraction": 0.90,
            "usable_fraction": 0.99,
            "closed_fraction": 0.95,
            "qualitative_flag_fraction": 0.05,
            "bootstrap_resamples": 200,
            "bootstrap_seed": 7,
        },
        "evidence": {"fixture": "memory"},
    }


class NanoR33ValidityGateTests(unittest.TestCase):
    def test_passing_fixture_passes(self):
        module = load_script("eval_nano_r33_validity_gate")

        report = module.evaluate_gate(passing_fixture())

        self.assertTrue(report["passed"])
        self.assertEqual(report["blockers"], [])
        gold_checks = [
            check
            for check in report["checks"]
            if check["name"].startswith("functional_gold_reference:")
        ]
        self.assertEqual(len(gold_checks), 1)
        self.assertAlmostEqual(
            gold_checks[0]["observed"]["candidate_excess_kl"],
            0.15,
        )

    def test_each_hard_gate_blocks_promotion(self):
        module = load_script("eval_nano_r33_validity_gate")

        def mutate(name, function):
            fixture = copy.deepcopy(passing_fixture())
            function(fixture["splits"]["validation"])
            with self.subTest(name=name):
                report = module.evaluate_gate(fixture)
                self.assertFalse(report["passed"])
                self.assertTrue(any(name in blocker for blocker in report["blockers"]))

        mutate("row_identity", lambda split: split["row_keys"].__setitem__("functional", ["wrong"]))
        mutate(
            "independent_families",
            lambda split: split.__setitem__(
                "content_family_ids", [f"family-{index // 2}" for index in range(8)]
            ),
        )
        mutate("roundtrip_relative_improvement", lambda split: split["roundtrip"].__setitem__("candidate_nmse", [0.97] * 8))
        mutate("roundtrip_paired_ci", lambda split: split["roundtrip"].__setitem__("candidate_nmse", [1.1, 0.7] * 4))
        mutate("functional_kl", lambda split: split["functional"].__setitem__("candidate", functional_rows(0.5, 0.90)))
        mutate("functional_top_10_overlap", lambda split: split["functional"].__setitem__("candidate", functional_rows(0.2, 0.88)))
        mutate("invariance_fve_retention", lambda split: split["invariance_retention"].__setitem__("unit_reorder", 0.89))
        mutate("control_win_fraction", lambda split: split["control_win_fractions"].__setitem__("zero", 0.89))
        mutate("parse_health", lambda split: split.__setitem__("parse", {"usable_fraction": 0.98, "closed_fraction": 0.94}))
        mutate("leakage", lambda split: split.__setitem__("leakage", {"injection_marker_count": 1, "cjk_count": 0}))
        mutate("qualitative_flag_fraction", lambda split: split.__setitem__("qualitative", {"row_count": 50, "flagged_count": 3}))
        mutate(
            "qualitative_review_complete",
            lambda split: split.__setitem__(
                "qualitative",
                {"row_count": 50, "reviewed_count": 49, "flagged_count": 0},
            ),
        )

    def test_report_has_explicit_observed_threshold_split_and_evidence(self):
        module = load_script("eval_nano_r33_validity_gate")

        report = module.evaluate_gate(passing_fixture())

        for check in report["checks"]:
            self.assertIn("observed", check)
            self.assertIn("threshold", check)
            self.assertIn("split", check)
            self.assertIn("evidence", check)

    def test_checked_in_config_names_all_stage1_candidates_and_tmpfs_roots(self):
        config_path = (
            ROOT / "configs" / "nano_rl" / "r33_component_validity_eval.yaml"
        )
        config = yaml.safe_load(config_path.read_text())

        self.assertEqual(config["schema_version"], "nano_r33_validity_eval.v1")
        self.assertEqual(config["thresholds"]["required_rows"], 512)
        self.assertEqual(
            set(config["evaluations"]["candidates"]), {"update16", "update32"}
        )
        self.assertTrue(config["runtime"]["temporary_hf_root"].startswith("/dev/shm/"))
        self.assertTrue(
            config["evaluations"]["sft"]["temporary_hf_dir"].startswith("/dev/shm/")
        )

    def test_leakage_scanner_flags_marker_and_cjk_rows(self):
        evidence = load_script("nano_validity_evidence")
        records = [
            {"controls": {"real": {"generated": "plain explanation"}}},
            {"controls": {"real": {"generated": "NLA_ACTIVATION_MARKER leaked"}}},
            {"controls": {"real": {"generated": "unexpected 漢 text"}}},
        ]

        result = evidence.scan_generated_leakage(records)

        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["injection_marker_count"], 1)
        self.assertEqual(result["cjk_count"], 1)


if __name__ == "__main__":
    unittest.main()
