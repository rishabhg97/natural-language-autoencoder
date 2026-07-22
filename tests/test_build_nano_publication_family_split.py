import importlib.util
import hashlib
import json
import pathlib
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "build_nano_publication_family_split.py"
    spec = importlib.util.spec_from_file_location("build_nano_publication_family_split", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildNanoPublicationFamilySplitTests(unittest.TestCase):
    def test_exposure_audit_config_is_fail_closed_and_writes_joint_inventory(self):
        config = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "nano_data"
                / "publication"
                / "r33_confirmatory_family_joint_exposure_audit_v4.yaml"
            ).read_text()
        )

        self.assertEqual(config["requirements"]["max_unmapped_prior_documents"], 0)
        self.assertGreaterEqual(config["requirements"]["min_test_families"], 100)
        self.assertTrue(config["joint_family_resolution"]["enabled"])
        self.assertIn("exposure_inventory_json", config["outputs"])
        self.assertIn("joint_family_manifest_json", config["outputs"])

    def test_full_exposure_config_includes_all_selected_checkpoint_splits(self):
        config = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "nano_data"
                / "publication"
                / "r33_confirmatory_selected_pair_exposure_audit_v6.yaml"
            ).read_text()
        )

        patterns = config["prior_exposure_globs"]
        self.assertEqual(config["requirements"]["max_unmapped_prior_documents"], 0)
        for component in ("miles-fsdp2-av-sft", "miles-fsdp2-ar-sft"):
            component_paths = [path for path in patterns if component in path]
            self.assertEqual(
                {path.rsplit("/", 1)[-1] for path in component_paths},
                {"train.parquet", "validation.parquet", "test.parquet"},
            )

    def test_joint_family_resolution_separates_near_duplicate_from_outside(self):
        builder = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            anchor_tokens = [f"token-{index}" for index in range(40)]
            anchor_text = " ".join(anchor_tokens)
            near_text = " ".join([*anchor_tokens[:-1], "changed-tail"])
            base_texts = [anchor_text] + [
                f"independent document number {index} with distinct publication content"
                for index in range(1, 30)
            ]
            base_manifest = root / "base.json"
            base_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_content_family_manifest.v1",
                        "doc_assignments": {
                            f"doc-{index}": f"family-{index}" for index in range(30)
                        },
                        "families": [
                            {
                                "content_family_id": f"family-{index}",
                                "row_count": 10,
                                "doc_ids": [f"doc-{index}"],
                                "normalized_text_sha256": [
                                    hashlib.sha256(text.encode("utf-8")).hexdigest()
                                ],
                            }
                            for index, text in enumerate(base_texts)
                        ],
                    }
                )
            )
            base_parquet = root / "base.parquet"
            pq.write_table(
                pa.table(
                    {
                        "doc_id": [f"doc-{index}" for index in range(30)],
                        "source_text": base_texts,
                    }
                ),
                base_parquet,
            )
            prior_parquet = root / "prior-validation.parquet"
            pq.write_table(
                pa.table(
                    {
                        "doc_id": ["historical-near", "historical-outside"],
                        "source_text": [
                            near_text,
                            "wholly unrelated archival source with separate vocabulary",
                        ],
                    }
                ),
                prior_parquet,
            )
            output_manifest = root / "frozen" / "manifest.json"
            report_json = root / "frozen" / "report.json"
            inventory_json = root / "frozen" / "exposure.json"
            joint_manifest_json = root / "frozen" / "joint.json"
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "nano_publication_family_split.v1",
                        "base_manifest": str(base_manifest),
                        "prior_evaluation_globs": [str(prior_parquet)],
                        "joint_family_resolution": {
                            "enabled": True,
                            "base_content_sources": [
                                {
                                    "path": str(base_parquet),
                                    "text_column": "source_text",
                                }
                            ],
                            "algorithm": {
                                "shingle_width": 5,
                                "similarity_threshold": 0.8,
                                "signature_size": 32,
                                "candidate_min_shared": 4,
                                "max_signature_bucket_size": 256,
                            },
                        },
                        "split_assignment": {
                            "seed": 20260715,
                            "weights": {"train": 0.8, "validation": 0.1, "test": 0.1},
                        },
                        "requirements": {
                            "min_test_rows": 10,
                            "min_test_families": 1,
                            "max_unmapped_prior_documents": 0,
                        },
                        "outputs": {
                            "manifest_json": str(output_manifest),
                            "report_json": str(report_json),
                            "exposure_inventory_json": str(inventory_json),
                            "joint_family_manifest_json": str(joint_manifest_json),
                        },
                    },
                    sort_keys=False,
                )
            )

            report = builder.run_build(config_path)
            inventory = json.loads(inventory_json.read_text())
            joint_manifest_exists = joint_manifest_json.is_file()

        documents = {row["doc_id"]: row for row in inventory["documents"]}
        self.assertTrue(report["passed"])
        self.assertEqual(report["unmapped_prior_document_count"], 0)
        self.assertEqual(documents["historical-near"]["status"], "near_duplicate")
        self.assertEqual(
            documents["historical-near"]["near_duplicate_family_ids"],
            ["family-0"],
        )
        self.assertEqual(
            documents["historical-outside"]["status"],
            "outside_candidate_universe",
        )
        self.assertTrue(joint_manifest_exists)

    def test_infeasible_exposure_boundary_preserves_failure_evidence(self):
        builder = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base_manifest = root / "base.json"
            base_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_content_family_manifest.v1",
                        "doc_assignments": {
                            f"doc-{index}": f"family-{index}" for index in range(3)
                        },
                        "families": [
                            {
                                "content_family_id": f"family-{index}",
                                "row_count": 10,
                                "doc_ids": [f"doc-{index}"],
                                "normalized_text_sha256": [],
                            }
                            for index in range(3)
                        ],
                    }
                )
            )
            prior = root / "selected-train.parquet"
            pq.write_table(
                pa.table({"doc_id": [f"doc-{index}" for index in range(3)]}),
                prior,
            )
            manifest_json = root / "frozen" / "manifest.json"
            report_json = root / "frozen" / "report.json"
            inventory_json = root / "frozen" / "inventory.json"
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "nano_publication_family_split.v1",
                        "base_manifest": str(base_manifest),
                        "prior_exposure_globs": [str(prior)],
                        "split_assignment": {
                            "seed": 20260715,
                            "weights": {"train": 0.8, "validation": 0.1, "test": 0.1},
                        },
                        "requirements": {
                            "min_test_rows": 1,
                            "min_test_families": 1,
                            "max_unmapped_prior_documents": 0,
                        },
                        "outputs": {
                            "manifest_json": str(manifest_json),
                            "report_json": str(report_json),
                            "exposure_inventory_json": str(inventory_json),
                        },
                    },
                    sort_keys=False,
                )
            )

            with self.assertRaisesRegex(
                builder.PublicationFamilySplitError,
                "split assignment is infeasible",
            ):
                builder.run_build(config_path)
            report = json.loads(report_json.read_text())
            inventory = json.loads(inventory_json.read_text())
            manifest_exists = manifest_json.exists()

        self.assertFalse(report["passed"])
        self.assertEqual(report["split_summary"], {})
        self.assertEqual(report["test_forbidden_family_count"], 3)
        self.assertEqual(report["prior_exposure_source_count"], 1)
        self.assertEqual(inventory["summary"]["status_counts"], {"direct_doc_id": 3})
        self.assertFalse(manifest_exists)

    def test_production_config_freezes_a_distinct_confirmatory_manifest(self):
        config_path = (
            ROOT
            / "configs"
            / "nano_data"
            / "publication"
            / "r33_confirmatory_family_split.yaml"
        )
        config = yaml.safe_load(config_path.read_text())
        expected_manifest = config["outputs"]["manifest_json"]

        self.assertEqual(config["split_assignment"]["seed"], 20260709)
        self.assertGreaterEqual(len(config["prior_evaluation_globs"]), 2)
        self.assertTrue(expected_manifest.endswith("r33_confirmatory_family_manifest.json"))
        for relative in (
            "configs/nano_ar/publication/r33_family_clean_sft.yaml",
            "configs/nano_ar/publication/r33_family_clean_independent_sft.yaml",
            "configs/nano_av/publication/r33_family_clean_sft.yaml",
        ):
            training_config = yaml.safe_load((ROOT / relative).read_text())
            self.assertEqual(
                training_config["dataset"]["content_family_manifest"],
                expected_manifest,
            )

    def test_freezes_test_away_from_previously_evaluated_families(self):
        builder = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base_manifest = root / "base.json"
            base_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_content_family_manifest.v1",
                        "doc_assignments": {
                            f"doc-{index}": f"family-{index}" for index in range(30)
                        },
                        "families": [
                            {
                                "content_family_id": f"family-{index}",
                                "row_count": 10,
                                "doc_ids": [f"doc-{index}"],
                                "normalized_text_sha256": (
                                    [hashlib.sha256(b"shared text").hexdigest()]
                                    if index == 29
                                    else []
                                ),
                            }
                            for index in range(30)
                        ],
                    }
                )
            )
            split_dir = root / "old-run" / "splits"
            split_dir.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "doc_id": [f"doc-{index}" for index in range(12)]
                        + ["outside-r33-family-universe"],
                        "source_text": [f"unique text {index}" for index in range(12)]
                        + ["shared text"],
                    }
                ),
                split_dir / "validation.parquet",
            )
            output_manifest = root / "frozen" / "manifest.json"
            report_json = root / "frozen" / "report.json"
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "nano_publication_family_split.v1",
                        "base_manifest": str(base_manifest),
                        "prior_evaluation_globs": [
                            str(root / "**" / "splits" / "validation.parquet")
                        ],
                        "split_assignment": {
                            "seed": 20260709,
                            "weights": {"train": 0.8, "validation": 0.1, "test": 0.1},
                        },
                        "requirements": {"min_test_rows": 10, "min_test_families": 1},
                        "outputs": {
                            "manifest_json": str(output_manifest),
                            "report_json": str(report_json),
                        },
                    },
                    sort_keys=False,
                )
            )

            report = builder.run_build(config_path)
            frozen = json.loads(output_manifest.read_text())

            with self.assertRaisesRegex(builder.PublicationFamilySplitError, "already exists"):
                builder.run_build(config_path)

        test_families = {
            family_id
            for family_id, split in frozen["family_splits"].items()
            if split == "test"
        }
        forbidden = {f"family-{index}" for index in range(12)} | {"family-29"}
        self.assertFalse(test_families & forbidden)
        self.assertTrue(report["passed"])
        self.assertEqual(report["prior_evaluation_source_count"], 1)
        self.assertEqual(report["unmapped_prior_document_count"], 0)
        self.assertEqual(frozen["publication_split_provenance"]["base_manifest_sha256"], report["base_manifest_sha256"])

    def test_exposure_inventory_reconciles_documents_across_sources(self):
        builder = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shared_hash = hashlib.sha256(b"shared historical text").hexdigest()
            base_manifest = root / "base.json"
            base_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "nano_content_family_manifest.v1",
                        "doc_assignments": {
                            f"doc-{index}": f"family-{index}" for index in range(20)
                        },
                        "families": [
                            {
                                "content_family_id": f"family-{index}",
                                "row_count": 10,
                                "doc_ids": [f"doc-{index}"],
                                "normalized_text_sha256": (
                                    [shared_hash] if index == 19 else []
                                ),
                            }
                            for index in range(20)
                        ],
                    }
                )
            )
            source_without_text = root / "old-validation.parquet"
            pq.write_table(
                pa.table({"doc_id": ["cross-source", "still-unmapped"]}),
                source_without_text,
            )
            source_with_text = root / "old-test.parquet"
            pq.write_table(
                pa.table(
                    {
                        "doc_id": ["cross-source", "still-unmapped"],
                        "source_text": ["shared historical text", "unknown text"],
                    }
                ),
                source_with_text,
            )
            output_manifest = root / "frozen" / "manifest.json"
            report_json = root / "frozen" / "report.json"
            inventory_json = root / "frozen" / "exposure.json"
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "nano_publication_family_split.v1",
                        "base_manifest": str(base_manifest),
                        "prior_evaluation_globs": [
                            str(source_without_text),
                            str(source_with_text),
                        ],
                        "split_assignment": {
                            "seed": 20260709,
                            "weights": {"train": 0.8, "validation": 0.1, "test": 0.1},
                        },
                        "requirements": {
                            "min_test_rows": 10,
                            "min_test_families": 1,
                            "max_unmapped_prior_documents": 0,
                        },
                        "outputs": {
                            "manifest_json": str(output_manifest),
                            "report_json": str(report_json),
                            "exposure_inventory_json": str(inventory_json),
                        },
                    },
                    sort_keys=False,
                )
            )

            with self.assertRaisesRegex(
                builder.PublicationFamilySplitError,
                "unmapped prior document count exceeds 0: 1",
            ):
                builder.run_build(config_path)

            report = json.loads(report_json.read_text())
            inventory = json.loads(inventory_json.read_text())

        documents = {row["doc_id"]: row for row in inventory["documents"]}
        self.assertEqual(report["unmapped_prior_document_count"], 1)
        self.assertEqual(len(report["prior_evaluation_sources"]), 2)
        self.assertEqual(documents["cross-source"]["status"], "content_hash")
        self.assertEqual(
            documents["cross-source"]["content_hash_family_ids"],
            ["family-19"],
        )
        self.assertEqual(documents["still-unmapped"]["status"], "unmapped")
        self.assertEqual(
            inventory["summary"]["status_counts"],
            {"content_hash": 1, "unmapped": 1},
        )
        self.assertTrue(report["exposure_inventory_sha256"])
        self.assertFalse(output_manifest.exists())


if __name__ == "__main__":
    unittest.main()
