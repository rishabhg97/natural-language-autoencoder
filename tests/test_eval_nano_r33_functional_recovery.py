import importlib.util
import hashlib
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def identity_row(
    split: str,
    key: str,
    *,
    stored_relative_l2: float = 0.0,
    reinjection_relative_l2: float = 0.0,
):
    stored_metrics = {
        "relative_l2": stored_relative_l2,
        "max_abs": 0.0,
        "one_minus_cos": 0.0,
    }
    reinjection_metrics = {
        "relative_l2": reinjection_relative_l2,
        "max_abs": 0.0,
        "one_minus_cos": 0.0,
    }
    return {
        "split": split,
        "provenance_key": ["uuid", key],
        "stored_activation_drift": stored_metrics,
        "logit_identity": reinjection_metrics,
    }


def functional_row(split: str, key: str, variant: str, kl: float, overlap: float):
    return {
        "split": split,
        "provenance_key": ["uuid", key],
        "content_family_id": f"family-{key}",
        "variant": variant,
        "metrics": {
            "kl_original_to_patched": kl,
            "js_divergence": kl / 2,
            "logit_pearson": 1.0 - kl,
            "top_10_overlap": overlap,
            "top_50_overlap": overlap,
            "original_top1_rank": 1,
        },
    }


class NanoR33FunctionalRecoveryTests(unittest.TestCase):
    def test_generation_identity_rejects_prefix_and_accepts_bound_rows(self):
        module = load_script("eval_nano_r33_functional_recovery")
        protocol = {
            "schema_version": "nano_generation_protocol.v1",
            "backend": "legacy_batch",
            "prefix": "",
        }
        provenance = {
            "model_fingerprint": "dcp_model_sha256:" + "a" * 64,
            "tokenizer_fingerprint": "tokenizer_files_sha256:" + "b" * 64,
            "datasets": {
                "validation": {"path": "/data/validation", "sha256": "c" * 64}
            },
        }
        record = {
            "row_index": 0,
            "generation_protocol": protocol,
            "generation_protocol_sha256": hashlib.sha256(
                json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "generation_provenance": provenance,
            "generation_provenance_sha256": hashlib.sha256(
                json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }

        identity = module.validate_generation_identity([record], label="candidate")
        self.assertEqual(identity["provenance"], provenance)

        prefixed = json.loads(json.dumps(record))
        prefixed["generation_protocol"]["prefix"] = "<explanation>"
        prefixed["generation_protocol_sha256"] = hashlib.sha256(
            json.dumps(
                prefixed["generation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(
            module.FunctionalEvaluationError,
            "empty prefix",
        ):
            module.validate_generation_identity([prefixed], label="candidate")

    def test_select_exact_split_rows_is_deterministic_and_strict(self):
        module = load_script("eval_nano_r33_functional_recovery")
        records = [
            {"split": "test", "row_index": 3, "sample_uuid": "t3"},
            {"split": "validation", "row_index": 2, "sample_uuid": "v2"},
            {"split": "validation", "row_index": 1, "sample_uuid": "v1"},
            {"split": "test", "row_index": 1, "sample_uuid": "t1"},
        ]

        selected = module.select_exact_split_rows(records, 2, 2)

        self.assertEqual(
            [(row["split"], row["row_index"]) for row in selected],
            [("validation", 1), ("validation", 2), ("test", 1), ("test", 3)],
        )
        with self.assertRaisesRegex(module.FunctionalEvaluationError, "requested 3"):
            module.select_exact_split_rows(records, 3, 2)

    def test_longest_prefix_selection_stresses_each_split_deterministically(self):
        module = load_script("eval_nano_r33_functional_recovery")
        records = [
            {
                "split": split,
                "row_index": offset + index,
                "sample_uuid": f"{split}-{index}",
                "n_raw_tokens": length,
            }
            for split, offset in (("validation", 0), ("test", 100))
            for index, length in enumerate((10, 50, 30))
        ]

        selected = module.select_exact_split_rows(
            records,
            2,
            2,
            selection_strategy="longest_prefix",
        )

        self.assertEqual(
            [(row["split"], row["n_raw_tokens"]) for row in selected],
            [("validation", 50), ("validation", 30), ("test", 50), ("test", 30)],
        )

    def test_stored_activation_drift_is_reported_without_blocking(self):
        module = load_script("eval_nano_r33_functional_recovery")
        report = module.build_functional_report(
            identity_rows=[
                identity_row("validation", "a", stored_relative_l2=0.02),
                identity_row("test", "b"),
            ],
            functional_rows=[functional_row("validation", "a", "candidate", 0.1, 1.0)],
            identity_tolerances={
                "relative_l2": 0.01,
                "max_abs": 0.01,
                "one_minus_cos": 0.0001,
            },
            metadata={"config_hash": "fixture"},
            bootstrap_resamples=100,
        )

        self.assertTrue(report["gate"]["identity_passed"])
        self.assertFalse(report["gate"]["stored_activation_replay_within_tolerance"])
        self.assertEqual(report["gate"]["stored_drift_outlier_count"], 1)
        self.assertIn("validation", report["splits"])

    def test_reinjection_identity_failure_blocks_candidate_metrics(self):
        module = load_script("eval_nano_r33_functional_recovery")
        report = module.build_functional_report(
            identity_rows=[
                identity_row("validation", "a", reinjection_relative_l2=0.02),
                identity_row("test", "b"),
            ],
            functional_rows=[functional_row("validation", "a", "candidate", 0.1, 1.0)],
            identity_tolerances={
                "relative_l2": 0.01,
                "max_abs": 0.01,
                "one_minus_cos": 0.0001,
            },
            metadata={"config_hash": "fixture"},
            bootstrap_resamples=100,
        )

        self.assertFalse(report["gate"]["identity_passed"])
        self.assertEqual(report["splits"], {})
        self.assertEqual(report["gate"]["failing_row_count"], 1)

    def test_report_summarizes_variants_and_paired_candidate_gain(self):
        module = load_script("eval_nano_r33_functional_recovery")
        identities = [identity_row("validation", key) for key in ("a", "b")]
        rows = []
        for key, sft_kl, candidate_kl in (("a", 0.4, 0.2), ("b", 0.3, 0.1)):
            rows.append(functional_row("validation", key, "stored_gold", 0.05, 0.95))
            rows.append(functional_row("validation", key, "sft", sft_kl, 0.8))
            rows.append(functional_row("validation", key, "candidate", candidate_kl, 0.9))

        report = module.build_functional_report(
            identity_rows=identities,
            functional_rows=rows,
            identity_tolerances={
                "relative_l2": 0.01,
                "max_abs": 0.01,
                "one_minus_cos": 0.0001,
            },
            metadata={"config_hash": "fixture"},
            bootstrap_resamples=100,
        )

        split = report["splits"]["validation"]
        self.assertTrue(report["gate"]["identity_passed"])
        self.assertAlmostEqual(
            split["variants"]["candidate"]["means"]["kl_original_to_patched"],
            0.15,
        )
        paired = split["paired_vs_sft"]["candidate"]["kl_original_to_patched"]
        self.assertAlmostEqual(paired["mean_improvement"], 0.2)
        self.assertEqual(paired["unit"], "cluster")
        self.assertEqual(paired["cluster_count"], 2)
        overlap = split["paired_vs_sft"]["candidate"]["top_10_overlap"]
        self.assertAlmostEqual(overlap["mean_improvement"], 0.1)
        gold = split["paired_vs_stored_gold"]["candidate"][
            "kl_original_to_patched"
        ]
        self.assertAlmostEqual(gold["mean_improvement"], -0.1)
        candidate_vs_sft = split["paired_candidate_vs_variants"]["sft"][
            "kl_original_to_patched"
        ]
        self.assertAlmostEqual(candidate_vs_sft["mean_improvement"], 0.2)


if __name__ == "__main__":
    unittest.main()
