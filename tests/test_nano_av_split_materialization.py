import importlib.util
import json
import pathlib
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoAVSplitMaterializationTests(unittest.TestCase):
    def _write_dataset(self, path):
        docs = ["doc-a", "doc-a", "doc-b", "doc-b", "doc-c", "doc-c", "doc-d", "doc-d"]
        table = pa.table(
            {
                "prompt": pa.array([f"prompt {i} <concept><INJECT></concept>" for i in range(len(docs))]),
                "response": pa.array([f"<explanation>row {i}</explanation>" for i in range(len(docs))]),
                "activation_vector": pa.array(
                    [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(docs))],
                    type=pa.list_(pa.float32(), 4),
                ),
                "doc_id": pa.array(docs),
            }
        )
        pq.write_table(table, path)
        sidecar = {
            "kind": "nla_dataset",
            "schema_version": 1,
            "dataset_id": "tiny-av",
            "stage": "av_sft",
            "row_count": len(docs),
            "extraction": {"d_model": 4, "layer_index": 27},
            "tokens": {
                "injection_char": "@",
                "injection_token_id": 64,
                "injection_left_neighbor_id": 62,
                "injection_right_neighbor_id": 60,
            },
        }
        path.with_name(path.name + ".nla_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))

    def test_materializes_doc_splits_and_padded_train_manifest(self):
        materializer = load_script("nano_av_materialize_splits")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "av_sft.parquet"
            out_dir = root / "splits"
            self._write_dataset(parquet)

            manifest = materializer.materialize_splits(
                parquet,
                out_dir,
                train_fraction=0.5,
                validation_fraction=0.25,
                test_fraction=0.25,
                seed=7,
                pad_train_to_multiple=3,
            )

            train = pq.read_table(out_dir / "train.parquet")
            train_padded = pq.read_table(out_dir / "train_padded.parquet")
            validation = pq.read_table(out_dir / "validation.parquet")
            test = pq.read_table(out_dir / "test.parquet")
            manifest_file = json.loads((out_dir / "split_manifest.json").read_text())
            train_sidecar = yaml.safe_load((out_dir / "train.parquet.nla_meta.yaml").read_text())
            padded_sidecar = yaml.safe_load((out_dir / "train_padded.parquet.nla_meta.yaml").read_text())

        self.assertEqual(manifest, manifest_file)
        self.assertEqual(train.num_rows, 4)
        self.assertEqual(validation.num_rows, 2)
        self.assertEqual(test.num_rows, 2)
        self.assertEqual(train_padded.num_rows, 6)
        self.assertEqual(manifest["train"]["padding_duplicate_count"], 2)
        self.assertEqual(manifest["train"]["padded_row_count"], 6)
        self.assertEqual(manifest["doc_overlap_count"], 0)
        self.assertEqual(set(train.column("doc_id").to_pylist()) & set(validation.column("doc_id").to_pylist()), set())
        self.assertEqual(set(train.column("doc_id").to_pylist()) & set(test.column("doc_id").to_pylist()), set())

        self.assertEqual(train_sidecar["row_count"], 4)
        self.assertEqual(padded_sidecar["row_count"], 6)
        self.assertEqual(padded_sidecar["split"]["padding_duplicate_count"], 2)

    def test_content_component_split_keeps_duplicate_docs_together_without_dropping_rows(self):
        materializer = load_script("nano_av_materialize_splits")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "av_sft.parquet"
            out_dir = root / "splits"
            docs = ["doc:0", "doc:1", "doc:2", "doc:3", "doc:4", "doc:5"]
            source_texts = ["shared-a", "shared-a", "shared-b", "shared-b", "unique-c", "unique-d"]
            table = pa.table(
                {
                    "prompt": pa.array([f"prompt {i} <concept><INJECT></concept>" for i in range(len(docs))]),
                    "response": pa.array([f"<explanation>row {i}</explanation>" for i in range(len(docs))]),
                    "source_text": pa.array(source_texts),
                    "activation_vector": pa.array(
                        [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(docs))],
                        type=pa.list_(pa.float32(), 4),
                    ),
                    "doc_id": pa.array(docs),
                }
            )
            pq.write_table(table, parquet)
            parquet.with_name(parquet.name + ".nla_meta.yaml").write_text(
                yaml.safe_dump(
                    {
                        "kind": "nla_dataset",
                        "schema_version": 1,
                        "dataset_id": "tiny-av",
                        "stage": "av_sft",
                        "row_count": len(docs),
                        "extraction": {"d_model": 4, "layer_index": 33},
                    },
                    sort_keys=False,
                )
            )

            manifest = materializer.materialize_splits(
                parquet,
                out_dir,
                train_fraction=0.5,
                validation_fraction=0.25,
                test_fraction=0.25,
                seed=3,
                split_mode="content_component",
            )

            split_for_doc = {}
            total_rows = 0
            for split_name in ("train", "validation", "test"):
                split_table = pq.read_table(out_dir / f"{split_name}.parquet")
                total_rows += split_table.num_rows
                for doc_id in split_table.column("doc_id").to_pylist():
                    split_for_doc[doc_id] = split_name

        self.assertEqual(total_rows, len(docs))
        self.assertEqual(manifest["split_mode"], "content_component")
        self.assertEqual(manifest["source_doc_count"], len(docs))
        self.assertEqual(manifest["source_split_unit_count"], 4)
        self.assertEqual(manifest["content_components"]["duplicate_component_count"], 2)
        self.assertEqual(manifest["content_components"]["duplicate_doc_count"], 4)
        self.assertEqual(split_for_doc["doc:0"], split_for_doc["doc:1"])
        self.assertEqual(split_for_doc["doc:2"], split_for_doc["doc:3"])


if __name__ == "__main__":
    unittest.main()
