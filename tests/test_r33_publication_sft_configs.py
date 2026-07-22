import pathlib
import sys
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import nano_ar_hpo_queue  # noqa: E402
import nano_av_probe_queue  # noqa: E402
import nano_av_runner  # noqa: E402


class R33PublicationSFTConfigTests(unittest.TestCase):
    def test_confirmatory_protocol_locks_pair_and_fails_closed_on_exposure(self):
        protocol = yaml.safe_load(
            (
                ROOT
                / "configs/nano_roundtrip/publication/"
                "r33_clean_sft_confirmatory_protocol.yaml"
            ).read_text()
        )

        self.assertEqual(
            protocol["status"],
            "in_corpus_boundary_infeasible_external_boundary_required",
        )
        self.assertTrue(protocol["checkpoint_pair"]["selection_locked"])
        self.assertFalse(protocol["checkpoint_pair"]["additional_hpo_allowed"])
        self.assertEqual(
            protocol["confirmatory_boundary"]["max_unmapped_prior_documents"],
            0,
        )
        self.assertEqual(
            protocol["confirmatory_boundary"]["max_prior_exposure_families"],
            0,
        )
        self.assertTrue(protocol["confirmatory_boundary"]["one_shot"])
        self.assertFalse(protocol["confirmatory_boundary"]["audit_passed"])
        self.assertTrue(
            protocol["confirmatory_boundary"]["external_boundary_required"]
        )
        self.assertEqual(
            protocol["confirmatory_boundary"]["candidate_family_count"],
            protocol["confirmatory_boundary"]["exposed_candidate_family_count"],
        )
        self.assertIn(
            "r33_content_families_v6_selected_pair_exposure",
            protocol["confirmatory_boundary"]["family_manifest"],
        )
        self.assertFalse(
            protocol["superseded_validation_test_only_audit"][
                "may_be_used_for_confirmatory_scoring"
            ]
        )
        self.assertEqual(
            protocol["roundtrip_endpoint"]["primary_metric"],
            "directional_mse",
        )
        self.assertFalse(
            protocol["roundtrip_endpoint"]["require_raw_magnitude_claim"]
        )
        self.assertTrue(
            protocol["functional_endpoint"][
                "require_fresh_forward_activation_fidelity_for_fresh_forward_claim"
            ]
        )

    def test_clean_sft_configs_are_full_resumeable_and_validation_only(self):
        configs = [
            ROOT / "configs/nano_ar/publication/r33_family_clean_sft.yaml",
            ROOT
            / "configs/nano_ar/publication/r33_family_clean_independent_sft.yaml",
            ROOT / "configs/nano_av/publication/r33_family_clean_sft.yaml",
        ]

        specs = [nano_av_runner.load_and_validate_spec(path) for path in configs]

        for spec in specs:
            self.assertEqual(spec["run"]["experiment_class"], "complete-performance")
            self.assertEqual(spec["dataset"]["row_limit"], 275396)
            self.assertEqual(spec["eval"]["eval_splits"], ["validation"])
            self.assertFalse(spec["checkpoint"]["no_save_optim"])
            self.assertTrue(spec["checkpoint"]["require_optimizer_state_for_hero"])
            self.assertEqual(spec["checkpoint"]["save_interval"], 1291)
            self.assertEqual(spec["eval"]["interval"], 1291)
            self.assertTrue(
                spec["run"]["output_root"].startswith(
                    "/workspace/models/nano30b-nla-pilot/publication/"
                )
            )
            source_paths = " ".join(
                str(value) for value in spec["paths"].values()
            )
            self.assertIn("r33_deterministic_full275396", source_paths)
            self.assertNotIn("r33_frozen_runtime_full275396", source_paths)

        self.assertEqual(
            specs[0]["paths"]["critic_init_model_id"],
            "/workspace/models/nano30b-nla-pilot/publication/"
            "r33_deterministic_critic_init",
        )
        self.assertEqual(
            specs[1]["paths"]["critic_init_model_id"],
            "/workspace/models/nano30b-nla-pilot/publication/"
            "r33_deterministic_independent_critic_init_seed314159",
        )
        self.assertIn(
            "/publication/runtime_deterministic/",
            specs[1]["paths"]["critic_initialization_verification_report"],
        )

    def test_clean_sft_queues_cannot_request_test_evaluation(self):
        ar_queue = nano_ar_hpo_queue.load_queue(
            ROOT / "configs/nano_ar/publication/r33_family_clean_sft_queue.yaml"
        )
        av_queue = nano_av_probe_queue.load_queue(
            ROOT / "configs/nano_av/publication/r33_family_clean_sft_queue.yaml"
        )

        self.assertEqual(ar_queue["defaults"]["eval_splits"], ["validation"])
        self.assertEqual(av_queue["defaults"]["eval_splits"], ["validation"])
        self.assertTrue(all(item["status"] == "pending" for item in ar_queue["items"]))
        self.assertTrue(all(item["status"] == "pending" for item in av_queue["items"]))

    def test_independent_ar_replication_reuses_frozen_validation_generations(self):
        component_verifier = yaml.safe_load(
            (
                ROOT
                / "configs/nano_ar/publication/"
                "r33_family_clean_independent_sft_eval_verify.yaml"
            ).read_text()
        )
        score = yaml.safe_load(
            (
                ROOT
                / "configs/nano_roundtrip/publication/"
                "r33_clean_sft_independent_ar_validation_score.yaml"
            ).read_text()
        )
        verifier = yaml.safe_load(
            (
                ROOT
                / "configs/nano_roundtrip/publication/"
                "r33_clean_sft_independent_ar_validation_verify.yaml"
            ).read_text()
        )

        self.assertEqual(score["eval"]["eval_splits"], ["validation"])
        self.assertEqual(
            component_verifier["expected"]["eval_splits"], ["validation"]
        )
        self.assertIn(
            "independent-seed314159",
            component_verifier["report_json"],
        )
        self.assertFalse(
            component_verifier["expected"]["require_raw_magnitude_claim"]
        )
        self.assertTrue(score["eval"]["reuse_generated"])
        self.assertIn(
            "r33_clean_sft_roundtrip/validation_generated.jsonl",
            score["paths"]["generated_jsonl"],
        )
        self.assertIn(
            "independent-seed314159",
            score["paths"]["ar_checkpoint_dir"],
        )
        self.assertEqual(
            verifier["expected"]["ar_checkpoint_dir"],
            score["paths"]["ar_checkpoint_dir"],
        )
        self.assertFalse(
            verifier["expected"]["require_raw_magnitude_claim"]
        )
        self.assertEqual(
            set(verifier["expected"]["generation_identity"]["dataset_sha256"]),
            {"train", "validation"},
        )


if __name__ == "__main__":
    unittest.main()
