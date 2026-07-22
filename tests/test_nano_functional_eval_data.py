import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoFunctionalEvalDataTests(unittest.TestCase):
    def test_content_families_merge_exact_and_shifted_duplicates_across_docs(self):
        module = load_script("nano_functional_eval_data")
        rows = [
            {
                "row_index": 0,
                "doc_id": "doc-a",
                "source_text": "Alpha beta gamma delta epsilon zeta eta theta.",
            },
            {
                "row_index": 1,
                "doc_id": "doc-b",
                "source_text": "ALPHA beta gamma delta epsilon zeta eta theta!",
            },
            {
                "row_index": 2,
                "doc_id": "doc-c",
                "source_text": "preface alpha beta gamma delta epsilon zeta eta theta",
            },
            {
                "row_index": 3,
                "doc_id": "doc-d",
                "source_text": "completely unrelated material about another subject",
            },
        ]

        manifest = module.build_content_families(
            rows,
            text_field="source_text",
            shingle_width=3,
            similarity_threshold=0.80,
            signature_size=16,
            candidate_min_shared=3,
        )

        assignments = manifest["doc_assignments"]
        self.assertEqual(assignments["doc-a"], assignments["doc-b"])
        self.assertEqual(assignments["doc-a"], assignments["doc-c"])
        self.assertNotEqual(assignments["doc-a"], assignments["doc-d"])
        self.assertEqual(manifest["algorithm"]["normalization_version"], "unicode_nfkc_casefold_words_v1")
        self.assertEqual(manifest["algorithm"]["shingle_width"], 3)
        self.assertEqual(manifest["stats"]["document_count"], 4)
        self.assertEqual(manifest["stats"]["family_count"], 2)

    def test_family_split_assignment_keeps_whole_families_disjoint(self):
        module = load_script("nano_functional_eval_data")
        family_manifest = {
            "families": [
                {"content_family_id": f"family-{index}", "row_count": index + 1}
                for index in range(12)
            ]
        }

        assigned = module.assign_family_splits(
            family_manifest,
            split_weights={"train": 0.8, "validation": 0.1, "test": 0.1},
            seed=20260708,
        )

        self.assertEqual(len(assigned["family_splits"]), 12)
        self.assertEqual(
            set(assigned["family_splits"].values()),
            {"train", "validation", "test"},
        )
        self.assertEqual(assigned["overlap"]["train_validation"], [])
        self.assertEqual(assigned["overlap"]["train_test"], [])
        self.assertEqual(assigned["overlap"]["validation_test"], [])

    def test_family_split_assignment_respects_forbidden_test_families(self):
        module = load_script("nano_functional_eval_data")
        family_manifest = {
            "families": [
                {"content_family_id": f"family-{index}", "row_count": 10}
                for index in range(30)
            ]
        }
        forbidden = {
            f"family-{index}": {"test"}
            for index in range(12)
        }

        assigned = module.assign_family_splits(
            family_manifest,
            split_weights={"train": 0.8, "validation": 0.1, "test": 0.1},
            seed=20260709,
            forbidden_splits_by_family=forbidden,
        )

        self.assertFalse(
            {
                family_id
                for family_id, split in assigned["family_splits"].items()
                if split == "test"
            }
            & set(forbidden)
        )
        self.assertEqual(assigned["split_assignment"]["constraint_family_count"], 12)
        self.assertEqual(set(assigned["family_splits"].values()), {"train", "validation", "test"})

    def test_family_stratified_selection_is_seeded_and_maximizes_families(self):
        module = load_script("nano_functional_eval_data")
        records = [
            {
                "split": split,
                "row_index": row_index,
                "doc_id": f"doc-{family}",
                "content_family_id": f"family-{family}",
                "n_raw_tokens": row_index + 1,
            }
            for split in ("validation", "test")
            for family in range(4)
            for row_index in (family * 10, family * 10 + 1)
        ]

        selected = module.select_exact_split_rows(
            list(reversed(records)),
            4,
            4,
            selection_strategy="family_stratified",
            selection_seed=17,
        )
        selected_again = module.select_exact_split_rows(
            records,
            4,
            4,
            selection_strategy="family_stratified",
            selection_seed=17,
        )

        self.assertEqual(
            [(row["split"], row["row_index"]) for row in selected],
            [(row["split"], row["row_index"]) for row in selected_again],
        )
        for split in ("validation", "test"):
            split_rows = [row for row in selected if row["split"] == split]
            self.assertEqual(len({row["content_family_id"] for row in split_rows}), 4)

    def test_exact_split_selection_can_keep_test_sealed(self):
        module = load_script("nano_functional_eval_data")
        records = [
            {
                "split": split,
                "row_index": row_index,
                "doc_id": f"{split}-doc-{row_index}",
                "n_raw_tokens": row_index + 1,
            }
            for split in ("validation", "test")
            for row_index in range(3)
        ]

        selected = module.select_exact_split_rows(
            records,
            2,
            2,
            eval_splits=("validation",),
        )

        self.assertEqual([row["split"] for row in selected], ["validation"] * 2)

    def test_attaching_family_manifest_rejects_cross_split_family_overlap(self):
        module = load_script("nano_functional_eval_data")
        manifest = {
            "schema_version": "nano_content_family_manifest.v1",
            "doc_assignments": {
                "train-doc": "family-shared",
                "validation-doc": "family-shared",
            },
        }
        rows = [
            {"doc_id": "train-doc", "split": "train"},
            {"doc_id": "validation-doc", "split": "validation"},
        ]

        report = module.attach_content_family_ids(rows, manifest)

        self.assertFalse(report["passed"])
        self.assertEqual(report["overlap"]["train_validation"], ["family-shared"])
        with self.assertRaisesRegex(
            module.FunctionalEvaluationError, "overlap dataset splits"
        ):
            module.attach_content_family_ids(
                rows,
                manifest,
                require_disjoint_splits=True,
            )

    def test_holdout_coverage_marks_only_eligible_disjoint_rows(self):
        module = load_script("nano_functional_eval_data")
        rows = [
            {"doc_id": "train", "split": "train", "content_family_id": "family-train"},
            {"doc_id": "val-good", "split": "validation", "content_family_id": "family-val"},
            {"doc_id": "val-drop", "split": "validation", "content_family_id": "family-drop"},
            {"doc_id": "test-good", "split": "test", "content_family_id": "family-test"},
        ]
        coverage = {
            "schema_version": "nano_content_family_exposure_report.v1",
            "splits": {
                "validation": {
                    "eligible_doc_ids": ["val-good"],
                    "eligible_family_ids": ["family-val"],
                },
                "test": {
                    "eligible_doc_ids": ["test-good"],
                    "eligible_family_ids": ["family-test"],
                },
            },
            "retain_existing_sft_checkpoints": True,
        }

        report = module.apply_family_holdout_coverage(rows, coverage)

        self.assertTrue(report["passed"])
        self.assertEqual(report["eligible_row_counts"], {"test": 1, "validation": 1})
        self.assertTrue(rows[1]["publication_holdout_eligible"])
        self.assertFalse(rows[2]["publication_holdout_eligible"])

    def test_exposure_audit_identifies_only_never_trained_holdout_families(self):
        module = load_script("nano_functional_eval_data")
        manifest = {
            "doc_assignments": {
                "train-a": "family-a",
                "near-duplicate-a": "family-a",
                "fresh-b": "family-b",
                "fresh-c": "family-c",
            }
        }
        candidates = {
            "validation": [
                {"doc_id": "near-duplicate-a"},
                {"doc_id": "fresh-b"},
            ],
            "test": [{"doc_id": "fresh-c"}],
        }
        exposures = {"av_train": [{"doc_id": "train-a"}]}

        report = module.build_family_exposure_report(
            manifest,
            candidate_rows_by_split=candidates,
            exposure_rows_by_source=exposures,
            minimum_holdout_rows=1,
        )

        self.assertEqual(report["splits"]["validation"]["eligible_row_count"], 1)
        self.assertEqual(report["splits"]["validation"]["excluded_exposed_row_count"], 1)
        self.assertEqual(report["splits"]["test"]["eligible_row_count"], 1)
        self.assertTrue(report["retain_existing_sft_checkpoints"])

    def test_exposure_audit_conservatively_removes_cross_holdout_families(self):
        module = load_script("nano_functional_eval_data")
        manifest = {
            "doc_assignments": {
                "val-shared": "family-shared",
                "test-shared": "family-shared",
                "val-fresh": "family-validation",
                "test-fresh": "family-test",
            }
        }

        report = module.build_family_exposure_report(
            manifest,
            candidate_rows_by_split={
                "validation": [
                    {"doc_id": "val-shared"},
                    {"doc_id": "val-fresh"},
                ],
                "test": [
                    {"doc_id": "test-shared"},
                    {"doc_id": "test-fresh"},
                ],
            },
            exposure_rows_by_source={"train": []},
            minimum_holdout_rows=1,
        )

        self.assertEqual(
            report["raw_holdout_family_overlap"]["test_validation"],
            ["family-shared"],
        )
        self.assertEqual(report["final_holdout_family_overlap"]["test_validation"], [])
        self.assertEqual(report["splits"]["validation"]["eligible_row_count"], 1)
        self.assertEqual(report["splits"]["test"]["eligible_row_count"], 1)
        self.assertEqual(
            report["splits"]["validation"]["excluded_cross_split_family_row_count"],
            1,
        )
        self.assertTrue(report["retain_existing_sft_checkpoints"])

    def test_config_driven_family_manifest_builder_writes_auditable_outputs(self):
        builder = load_script("build_nano_content_family_manifest")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            rows_by_split = {
                "train": [
                    {
                        "doc_id": "train-a",
                        "source_text": "alpha beta gamma delta epsilon zeta eta theta",
                    }
                ],
                "validation": [
                    {
                        "doc_id": "near-a",
                        "source_text": "preface alpha beta gamma delta epsilon zeta eta theta",
                    },
                    {
                        "doc_id": "fresh-b",
                        "source_text": "fresh content about a separate validation topic entirely",
                    },
                ],
                "test": [
                    {
                        "doc_id": "fresh-c",
                        "source_text": "another independent test document with distinct words",
                    }
                ],
            }
            sources = []
            for split, rows in rows_by_split.items():
                path = root / f"{split}.parquet"
                pq.write_table(pa.Table.from_pylist(rows), path)
                sources.append(
                    {
                        "name": f"av_{split}",
                        "path": str(path),
                        "split": split,
                        "text_field": "source_text",
                    }
                )
            config = {
                "schema_version": "nano_content_family_build.v1",
                "family_sources": sources,
                "exposure_sources": [
                    {"name": "av_train", "path": str(root / "train.parquet")}
                ],
                "candidate_sources": [
                    {
                        "name": "av_validation",
                        "split": "validation",
                        "path": str(root / "validation.parquet"),
                    },
                    {
                        "name": "av_test",
                        "split": "test",
                        "path": str(root / "test.parquet"),
                    },
                ],
                "algorithm": {
                    "shingle_width": 3,
                    "similarity_threshold": 0.8,
                    "signature_size": 16,
                    "candidate_min_shared": 3,
                },
                "split_assignment": {
                    "seed": 17,
                    "weights": {"train": 0.8, "validation": 0.1, "test": 0.1},
                },
                "holdout": {"minimum_rows_per_split": 1},
                "outputs": {
                    "manifest_json": str(root / "manifest.json"),
                    "coverage_json": str(root / "coverage.json"),
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(__import__("yaml").safe_dump(config, sort_keys=False))

            result = builder.run_build(config_path)

            self.assertTrue((root / "manifest.json").is_file())
            self.assertTrue((root / "coverage.json").is_file())
            self.assertEqual(result["coverage"]["splits"]["validation"]["eligible_row_count"], 1)
            self.assertTrue(result["coverage"]["retain_existing_sft_checkpoints"])
            self.assertGreater(
                result["manifest"]["observed_split_overlap"]["overlap_family_count"],
                0,
            )

    def test_extracts_teacher_text_from_ar_sft_prompt(self):
        module = load_script("nano_functional_eval_data")

        explanation = module.extract_teacher_text(
            {
                "doc_id": "doc-1",
                "token_position": 4,
                "prompt": "Summary: <text>  a teacher-backed explanation  </text> <summary>",
            }
        )

        self.assertEqual(explanation, "a teacher-backed explanation")

    def test_extracts_nested_generated_control_and_respects_fallback(self):
        module = load_script("nano_functional_eval_data")
        record = {
            "controls": {
                "real": {"generated": "<explanation>recovered meaning</explanation>"}
            }
        }

        self.assertEqual(
            module.extract_generated_text(record, "real", "empty"),
            "recovered meaning",
        )
        self.assertEqual(
            module.extract_generated_text({"generated": "raw text"}, "real", "raw"),
            "raw text",
        )

    def test_shuffle_is_within_document_and_reports_ineligible_rows(self):
        module = load_script("nano_functional_eval_data")
        selected = [
            {"doc_id": "a"},
            {"doc_id": "a"},
            {"doc_id": "b"},
        ]
        sources = [
            {"doc_id": "a"},
            {"doc_id": "a"},
            {"doc_id": "b"},
        ]

        mapping, stats = module.within_document_shuffle(selected, sources)

        self.assertEqual(mapping, {0: 1, 1: 0})
        self.assertEqual(stats, {"eligible_rows": 2, "ineligible_rows": 1})

    def test_variant_entries_include_unmodified_stored_gold(self):
        module = load_script("nano_functional_eval_data")
        selected = [
            {
                "split": "validation",
                "row_index": 3,
                "doc_id": "doc-a",
                "n_raw_tokens": 2,
            }
        ]
        sources = [
            {
                "doc_id": "doc-a",
                "n_raw_tokens": 2,
                "token_ids_prefix": [10, 11],
                "activation_vector": [3.0, 4.0],
            }
        ]

        entries, _ = module.build_variant_entries(
            selected,
            sources,
            {"candidate": np.asarray([[1.0, 0.0]], dtype=np.float32)},
            np.asarray([0.5, 0.5], dtype=np.float32),
        )

        gold = next(entry for entry in entries if entry["variant"] == "stored_gold")
        self.assertTrue(torch.equal(gold["replacement"], torch.tensor([3.0, 4.0])))

    def test_source_mean_uses_only_train_rows_for_fixed_size_vectors(self):
        module = load_script("nano_functional_eval_data")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "train.parquet"
            vectors = pa.array(
                [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
                type=pa.list_(pa.float32(), 2),
            )
            pq.write_table(
                pa.table(
                    {
                        "activation_vector": vectors,
                        "split": ["train", "train", "validation"],
                    }
                ),
                path,
            )

            mean = module.source_mean_activation(path, batch_size=2)

        np.testing.assert_allclose(mean, np.asarray([2.0, 4.0], dtype=np.float32))

    def test_fixed_size_list_matrix_uses_flat_arrow_values(self):
        module = load_script("nano_functional_eval_data")
        vectors = pa.array(
            [[1.0, 2.0], [3.0, 4.0]],
            type=pa.list_(pa.float32(), 2),
        )

        matrix = module.fixed_size_list_matrix(vectors)

        np.testing.assert_allclose(
            matrix,
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
