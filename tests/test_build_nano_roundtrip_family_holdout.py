from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "build_nano_roundtrip_family_holdout.py"
    spec = importlib.util.spec_from_file_location("roundtrip_family_holdout", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RoundtripFamilyHoldoutTests(unittest.TestCase):
    def _write_config(self, root: pathlib.Path, *, manifest: pathlib.Path) -> pathlib.Path:
        source = root / "validation.parquet"
        pq.write_table(
            pa.table(
                {
                    "doc_id": ["doc-a", "doc-b", "doc-c", "doc-d"],
                    "response": ["a", "b", "c", "d"],
                }
            ),
            source,
        )
        (root / "validation.parquet.nla_meta.yaml").write_text(
            yaml.safe_dump(
                {
                    "kind": "nla_dataset",
                    "schema_version": 1,
                    "dataset_id": "source_validation",
                    "row_count": 4,
                    "extraction": {"d_model": 2688, "injection_scale": 75.0},
                    "tokens": {"injection_char": "X", "injection_token_id": 1},
                    "prompt_templates": {"actor": "{injection_char}"},
                },
                sort_keys=False,
            )
        )
        sealed = root / "sealed_report.json"
        sealed.write_text(
            json.dumps(
                {
                    "splits": {
                        "validation": {
                            "row_indices": [0, 1],
                            "content_family_ids": ["family-a", "family-b"],
                        }
                    },
                    "generation_protocol_sha256": "protocol-sha",
                }
            )
        )
        config = root / "holdout.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "nano_roundtrip_family_holdout.v1",
                    "role": "hpo_development",
                    "inputs": {
                        "source_validation_parquet": str(source),
                        "content_family_manifest": str(manifest),
                        "exclusion_reports": [
                            {"report_json": str(sealed), "split": "validation"}
                        ],
                    },
                    "outputs": {
                        "validation_parquet": str(root / "dev.parquet"),
                        "boundary_json": str(root / "boundary.json"),
                        "report_json": str(root / "report.json"),
                    },
                },
                sort_keys=False,
            )
        )
        return config

    def test_builds_an_idempotent_family_disjoint_parquet(self):
        builder = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = root / "families.json"
            manifest.write_text(
                json.dumps(
                    {
                        "doc_assignments": {
                            "doc-a": "family-a",
                            "doc-b": "family-b",
                            "doc-c": "family-c",
                            "doc-d": "family-d",
                        }
                    }
                )
            )
            config = self._write_config(root, manifest=manifest)

            report = builder.run_build(config)
            repeated = builder.run_build(config)
            output = pq.read_table(root / "dev.parquet")
            boundary = json.loads((root / "boundary.json").read_text())
            sidecar = yaml.safe_load((root / "dev.parquet.nla_meta.yaml").read_text())

            self.assertTrue(report["passed"])
            self.assertEqual(report, repeated)
            self.assertEqual(output.column("doc_id").to_pylist(), ["doc-c", "doc-d"])
            self.assertEqual(boundary["excluded_content_family_ids"], ["family-a", "family-b"])
            self.assertEqual(report["output_excluded_family_overlap_count"], 0)
            self.assertEqual(report["output_content_family_count"], 2)
            self.assertEqual(report["output_validation_nla_meta_yaml_sha256"], builder._sha256_file(root / "dev.parquet.nla_meta.yaml"))
            self.assertEqual(sidecar["dataset_id"], "dev")
            self.assertEqual(sidecar["row_count"], 2)
            self.assertEqual(sidecar["lineage"]["family_holdout_excluded_content_family_count"], 2)

    def test_requires_a_source_dataset_sidecar(self):
        builder = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = root / "families.json"
            manifest.write_text(
                json.dumps(
                    {
                        "doc_assignments": {
                            "doc-a": "family-a",
                            "doc-b": "family-b",
                            "doc-c": "family-c",
                            "doc-d": "family-d",
                        }
                    }
                )
            )
            config = self._write_config(root, manifest=manifest)
            (root / "validation.parquet.nla_meta.yaml").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "source validation sidecar"):
                builder.run_build(config)

    def test_rejects_source_rows_missing_from_the_family_manifest(self):
        builder = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = root / "families.json"
            manifest.write_text(
                json.dumps(
                    {
                        "doc_assignments": {
                            "doc-a": "family-a",
                            "doc-b": "family-b",
                            "doc-c": "family-c",
                        }
                    }
                )
            )
            config = self._write_config(root, manifest=manifest)

            with self.assertRaisesRegex(
                builder.RoundtripFamilyHoldoutError,
                "not fully covered",
            ):
                builder.run_build(config)


if __name__ == "__main__":
    unittest.main()
