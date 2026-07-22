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
    path = ROOT / "scripts" / "nano_av_materialize_splits.py"
    spec = importlib.util.spec_from_file_location("nano_av_materialize_splits", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoAVMaterializeSplitsTests(unittest.TestCase):
    def _write_source(self, root: pathlib.Path) -> pathlib.Path:
        source = root / "source.parquet"
        pq.write_table(
            pa.table(
                {
                    "doc_id": ["doc:a", "doc:a", "doc:b", "doc:c", "doc:d"],
                    "prompt": ["a0", "a1", "b0", "c0", "d0"],
                }
            ),
            source,
        )
        (root / "source.parquet.nla_meta.yaml").write_text(
            yaml.safe_dump({"kind": "nla_dataset", "row_count": 5})
        )
        return source

    def _write_manifest(self, root: pathlib.Path, *, include_doc_d: bool = True) -> pathlib.Path:
        assignments = {
            "doc:a": "family_ab",
            "doc:b": "family_ab",
            "doc:c": "family_c",
        }
        if include_doc_d:
            assignments["doc:d"] = "family_d"
        manifest = root / "families.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "nano_content_family_manifest.v1",
                    "doc_assignments": assignments,
                    "family_splits": {
                        "family_ab": "train",
                        "family_c": "validation",
                        "family_d": "test",
                    },
                    "split_assignment": {
                        "seed": 20260708,
                        "weights": {"train": 0.5, "validation": 0.25, "test": 0.25},
                    },
                }
            )
        )
        return manifest

    def test_frozen_family_manifest_controls_all_split_assignments(self):
        splitter = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = self._write_source(root)
            family_manifest = self._write_manifest(root)

            report = splitter.materialize_splits(
                source,
                root / "splits",
                train_fraction=0.5,
                validation_fraction=0.25,
                test_fraction=0.25,
                seed=20260708,
                split_mode="content_family_manifest",
                content_family_manifest=family_manifest,
            )
            train = pq.read_table(root / "splits" / "train.parquet")
            validation = pq.read_table(root / "splits" / "validation.parquet")
            test = pq.read_table(root / "splits" / "test.parquet")

        self.assertEqual(report["schema_version"], "nano_split_manifest.v2")
        self.assertEqual(report["split_unit_column"], "split_unit_id")
        self.assertEqual(report["split_unit_kind"], "content_family")
        self.assertEqual(set(report["splits"]["train"]["docs"]), {"doc:a", "doc:b"})
        self.assertEqual(report["splits"]["train"]["split_unit_ids"], ["family_ab"])
        self.assertEqual(report["splits"]["validation"]["docs"], ["doc:c"])
        self.assertEqual(report["splits"]["validation"]["split_unit_ids"], ["family_c"])
        self.assertEqual(report["splits"]["test"]["docs"], ["doc:d"])
        self.assertEqual(report["splits"]["test"]["split_unit_ids"], ["family_d"])
        for table in (train, validation, test):
            self.assertIn("split_unit_id", table.column_names)
            self.assertIn("content_family_id", table.column_names)
            self.assertEqual(
                table.column("split_unit_id").to_pylist(),
                table.column("content_family_id").to_pylist(),
            )
        self.assertEqual(set(train.column("split_unit_id").to_pylist()), {"family_ab"})
        self.assertEqual(validation.column("split_unit_id").to_pylist(), ["family_c"])
        self.assertEqual(test.column("split_unit_id").to_pylist(), ["family_d"])
        self.assertEqual(report["family_overlap_count"], 0)
        self.assertEqual(report["content_family_manifest"]["sha256"], report["content_family_manifest_sha256"])

    def test_frozen_family_manifest_rejects_unassigned_documents(self):
        splitter = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = self._write_source(root)
            family_manifest = self._write_manifest(root, include_doc_d=False)

            with self.assertRaisesRegex(ValueError, "missing family assignments"):
                splitter.materialize_splits(
                    source,
                    root / "splits",
                    train_fraction=0.5,
                    validation_fraction=0.25,
                    test_fraction=0.25,
                    seed=20260708,
                    split_mode="content_family_manifest",
                    content_family_manifest=family_manifest,
                )


if __name__ == "__main__":
    unittest.main()
