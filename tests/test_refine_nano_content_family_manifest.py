import importlib.util
import json
import pathlib
import tempfile
import textwrap
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "refine_nano_content_family_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "refine_nano_content_family_manifest",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RefineNanoContentFamilyManifestTests(unittest.TestCase):
    def test_real_publication_configs_chain_refinement_into_split_assignment(self):
        refinement = yaml.safe_load(
            (
                ROOT
                / "configs/nano_data/publication/r33_exact_prefix_family_refinement.yaml"
            ).read_text()
        )
        split = yaml.safe_load(
            (
                ROOT
                / "configs/nano_data/publication/r33_confirmatory_family_split.yaml"
            ).read_text()
        )

        self.assertEqual(
            refinement["outputs"]["manifest_json"],
            split["base_manifest"],
        )
        self.assertTrue(
            refinement["exact_content_sources"][0].endswith(
                "base_R33_r33_frozen_runtime_full275396.parquet"
            )
        )

    def _write_base_manifest(self, path):
        path.write_text(
            json.dumps(
                {
                    "schema_version": "nano_content_family_manifest.v1",
                    "doc_assignments": {
                        "doc-a": "family-a",
                        "doc-b": "family-b",
                        "doc-c": "family-c",
                    },
                    "families": [
                        {
                            "content_family_id": "family-a",
                            "doc_ids": ["doc-a"],
                            "document_count": 1,
                            "row_count": 1,
                            "normalized_text_sha256": ["a" * 64],
                        },
                        {
                            "content_family_id": "family-b",
                            "doc_ids": ["doc-b"],
                            "document_count": 1,
                            "row_count": 1,
                            "normalized_text_sha256": ["b" * 64],
                        },
                        {
                            "content_family_id": "family-c",
                            "doc_ids": ["doc-c"],
                            "document_count": 1,
                            "row_count": 1,
                            "normalized_text_sha256": ["c" * 64],
                        },
                    ],
                    "stats": {
                        "row_count": 3,
                        "document_count": 3,
                        "family_count": 3,
                    },
                    "family_splits": {
                        "family-a": "train",
                        "family-b": "validation",
                        "family-c": "test",
                    },
                }
            )
        )

    def _write_source(self, path, *, include_doc_c=True):
        docs = ["doc-a", "doc-b"] + (["doc-c"] if include_doc_c else [])
        prefixes = [[101, 102], [101, 102]] + ([[201, 202]] if include_doc_c else [])
        pq.write_table(
            pa.table(
                {
                    "doc_id": pa.array(docs),
                    "token_ids_prefix": pa.array(prefixes, type=pa.list_(pa.int32())),
                }
            ),
            path,
        )

    def _write_config(self, path, *, base_manifest, source, manifest_out, report_out):
        path.write_text(
            textwrap.dedent(
                f"""
                schema_version: nano_content_family_refinement.v1
                base_manifest: {base_manifest}
                exact_content_sources:
                  - {source}
                requirements:
                  exact_document_coverage: true
                outputs:
                  manifest_json: {manifest_out}
                  report_json: {report_out}
                """
            )
        )

    def test_exact_prefix_connectivity_merges_preexisting_families(self):
        refiner = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.json"
            source = root / "source.parquet"
            config = root / "config.yaml"
            manifest_out = root / "refined.json"
            report_out = root / "report.json"
            self._write_base_manifest(base)
            self._write_source(source)
            self._write_config(
                config,
                base_manifest=base,
                source=source,
                manifest_out=manifest_out,
                report_out=report_out,
            )

            report = refiner.run_refinement(config)
            refined = json.loads(manifest_out.read_text())

        self.assertTrue(report["passed"])
        self.assertEqual(report["family_count_before"], 3)
        self.assertEqual(report["family_count_after"], 2)
        self.assertEqual(report["cross_family_duplicate_key_count"], 1)
        self.assertEqual(
            refined["doc_assignments"]["doc-a"],
            refined["doc_assignments"]["doc-b"],
        )
        self.assertNotEqual(
            refined["doc_assignments"]["doc-a"],
            refined["doc_assignments"]["doc-c"],
        )
        self.assertNotIn("family_splits", refined)
        self.assertEqual(refined["stats"]["row_count"], 3)

    def test_exact_document_coverage_rejects_missing_source_docs(self):
        refiner = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.json"
            source = root / "source.parquet"
            config = root / "config.yaml"
            self._write_base_manifest(base)
            self._write_source(source, include_doc_c=False)
            self._write_config(
                config,
                base_manifest=base,
                source=source,
                manifest_out=root / "refined.json",
                report_out=root / "report.json",
            )

            with self.assertRaisesRegex(ValueError, "document coverage mismatch"):
                refiner.run_refinement(config)


if __name__ == "__main__":
    unittest.main()
