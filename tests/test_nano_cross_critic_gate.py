from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "eval_nano_cross_critic_gate.py"
    spec = importlib.util.spec_from_file_location("eval_nano_cross_critic_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def report(
    *,
    relative: float | None,
    win: float = 0.75,
    ci_low: float = 0.01,
    independent_unit: str = "content_family_id",
    length_gain: float | None = 0.02,
    generated_sha: str = "generated-sha",
    protocol_sha: str = "protocol-sha",
):
    provenance = {
        name: {"sha256": f"{name}-sha"}
        for name in ("train", "validation", "test")
    }
    splits = {}
    gates = {}
    for split in ("validation", "test"):
        splits[split] = {
            "row_indices": [1, 2, 3, 4],
            "row_keys": [
                {"doc_id": "doc-a", "token_position": 1},
                {"doc_id": "doc-a", "token_position": 2},
                {"doc_id": "doc-b", "token_position": 1},
                {"doc_id": "doc-b", "token_position": 2},
            ],
            "length_analysis": {
                "best_length_matched_relative_improvement": length_gain,
            },
        }
        gates[split] = {
            "baseline_row_identity_match": True,
            "baseline_dataset_hash_match": True,
            "baseline_rowwise_win_rate": {"candidate_better_fraction": win},
            "baseline_paired_improvement": {
                "relative_improvement": relative,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_low + 0.02,
                "independent_unit": independent_unit,
                "independent_unit_count": 2,
            },
            "parse_health": {"passed": True},
        }
    return {
        "generated_jsonl_provenance": {"sha256": generated_sha},
        "generation_protocol_sha256": protocol_sha,
        "dataset_provenance": provenance,
        "splits": splits,
        "gate": {"splits": gates},
    }


class NanoCrossCriticGateTests(unittest.TestCase):
    def test_passes_when_both_critics_show_distributed_improvement(self):
        gate = load_script()
        primary_candidate = report(relative=0.12)
        independent_candidate = report(relative=0.10)

        result = gate.build_cross_critic_gate(
            primary_candidate=primary_candidate,
            independent_candidate=independent_candidate,
            primary_sft=report(relative=None),
            independent_sft=report(relative=None),
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["splits"]["test"]["passed"])
        self.assertAlmostEqual(result["splits"]["test"]["gain_ratio"], 5 / 6)

    def test_requires_family_clustered_and_length_controlled_gain(self):
        gate = load_script()

        document_clustered = gate.build_cross_critic_gate(
            primary_candidate=report(relative=0.12),
            independent_candidate=report(relative=0.10, independent_unit="doc_id"),
            primary_sft=report(relative=None),
            independent_sft=report(relative=None),
        )
        no_length_gain = gate.build_cross_critic_gate(
            primary_candidate=report(relative=0.12),
            independent_candidate=report(relative=0.10, length_gain=0.0),
            primary_sft=report(relative=None),
            independent_sft=report(relative=None),
        )

        self.assertFalse(document_clustered["passed"])
        self.assertFalse(
            document_clustered["splits"]["test"]["checks"][
                "independent_family_clustered_ci_positive"
            ]
        )
        self.assertFalse(no_length_gain["passed"])
        self.assertFalse(
            no_length_gain["splits"]["test"]["checks"][
                "independent_length_control_gain"
            ]
        )

    def test_fails_closed_for_each_cross_critic_requirement(self):
        gate = load_script()
        cases = {
            "weak_independent_gain": report(relative=0.04),
            "nonpositive_ci": report(relative=0.08, ci_low=0.0),
            "row_wins": report(relative=0.08, win=0.50),
            "gain_ratio": report(relative=0.02),
            "missing_metric": report(relative=None),
        }
        for name, independent in cases.items():
            with self.subTest(name=name):
                result = gate.build_cross_critic_gate(
                    primary_candidate=report(relative=0.12),
                    independent_candidate=independent,
                    primary_sft=report(relative=None),
                    independent_sft=report(relative=None),
                )
                self.assertFalse(result["passed"])

    def test_fails_closed_on_dataset_or_row_identity_mismatch(self):
        gate = load_script()
        mismatched_hash = report(relative=0.08)
        mismatched_hash["dataset_provenance"]["test"]["sha256"] = "different"
        result = gate.build_cross_critic_gate(
            primary_candidate=report(relative=0.12),
            independent_candidate=mismatched_hash,
            primary_sft=report(relative=None),
            independent_sft=report(relative=None),
        )
        self.assertFalse(result["passed"])

    def test_fails_closed_on_generated_text_or_protocol_mismatch(self):
        gate = load_script()
        mismatched_text = gate.build_cross_critic_gate(
            primary_candidate=report(relative=0.12),
            independent_candidate=report(
                relative=0.10,
                generated_sha="other-generated-sha",
            ),
            primary_sft=report(relative=None),
            independent_sft=report(relative=None),
        )
        mismatched_protocol = gate.build_cross_critic_gate(
            primary_candidate=report(relative=0.12),
            independent_candidate=report(
                relative=0.10,
                protocol_sha="other-protocol-sha",
            ),
            primary_sft=report(relative=None),
            independent_sft=report(relative=None),
        )

        self.assertFalse(mismatched_text["passed"])
        self.assertFalse(
            mismatched_text["splits"]["test"]["checks"][
                "candidate_generated_text_identity"
            ]
        )
        self.assertFalse(mismatched_protocol["passed"])
        self.assertFalse(
            mismatched_protocol["splits"]["test"]["checks"][
                "generation_protocol_identity"
            ]
        )

    def test_optional_row_key_enrichment_does_not_change_row_identity(self):
        gate = load_script()
        primary_candidate = report(relative=0.12)
        independent_candidate = report(relative=0.10)
        primary_sft = report(relative=None)
        independent_sft = report(relative=None)
        for baseline in (primary_sft, independent_sft):
            for split in ("validation", "test"):
                baseline["splits"][split]["row_keys"] = [
                    {"doc_id": row["doc_id"]}
                    for row in baseline["splits"][split]["row_keys"]
                ]

        result = gate.build_cross_critic_gate(
            primary_candidate=primary_candidate,
            independent_candidate=independent_candidate,
            primary_sft=primary_sft,
            independent_sft=independent_sft,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["splits"]["validation"]["checks"]["row_identity"])

    def test_fails_closed_on_document_identity_mismatch(self):
        gate = load_script()
        independent = report(relative=0.08)
        independent["splits"]["test"]["row_keys"][-1]["doc_id"] = "other-doc"

        result = gate.build_cross_critic_gate(
            primary_candidate=report(relative=0.12),
            independent_candidate=independent,
            primary_sft=report(relative=None),
            independent_sft=report(relative=None),
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["splits"]["test"]["checks"]["row_identity"])

        mismatched_rows = report(relative=0.08)
        mismatched_rows["splits"]["validation"]["row_indices"][-1] = 99
        result = gate.build_cross_critic_gate(
            primary_candidate=report(relative=0.12),
            independent_candidate=mismatched_rows,
            primary_sft=report(relative=None),
            independent_sft=report(relative=None),
        )
        self.assertFalse(result["passed"])

    def test_fails_closed_when_a_split_is_missing(self):
        gate = load_script()
        independent = report(relative=0.08)
        del independent["gate"]["splits"]["test"]
        result = gate.build_cross_critic_gate(
            primary_candidate=report(relative=0.12),
            independent_candidate=independent,
            primary_sft=report(relative=None),
            independent_sft=report(relative=None),
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
