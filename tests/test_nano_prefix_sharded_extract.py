import importlib.util
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "nano_prefix_sharded_extract.py"
    spec = importlib.util.spec_from_file_location("nano_prefix_sharded_extract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoPrefixShardedExtractTests(unittest.TestCase):
    def test_worker_command_forwards_deterministic_execution_profile(self):
        sharded = load_script()
        args = SimpleNamespace(
            source_parquet=pathlib.Path("source.parquet"),
            layers="R33",
            source_batch_size=4096,
            batch_size=2,
            model_id="model",
            model_revision=None,
            tokenizer_revision=None,
            torch_dtype="bfloat16",
            attn_implementation=None,
            local_files_only=True,
            trust_remote_code=True,
            publication_mode=True,
            model_fingerprint_json=pathlib.Path("model.json"),
            runtime_provenance_json=pathlib.Path("runtime.json"),
            overwrite=True,
            deterministic_algorithms=True,
            allow_tf32=False,
            cudnn_benchmark=False,
            float32_matmul_precision="highest",
            cublas_workspace_config=":4096:8",
            seed=20260709,
        )

        command = sharded._extract_command(
            args,
            shard_root=pathlib.Path("shard"),
            row_start=10,
            row_count=20,
        )

        self.assertIn("--deterministic-algorithms", command)
        self.assertIn("--no-allow-tf32", command)
        self.assertIn("--no-cudnn-benchmark", command)
        self.assertIn(":4096:8", command)
        self.assertIn("highest", command)
        self.assertIn("20260709", command)

    def test_plans_balanced_contiguous_row_shards(self):
        sharded = load_script()

        plan = sharded.plan_row_shards(
            total_rows=103,
            row_start=3,
            row_limit=97,
            devices=["0", "1", "2", "3"],
        )

        self.assertEqual(sum(item["row_count"] for item in plan), 97)
        self.assertEqual(plan[0]["row_start"], 3)
        self.assertEqual(plan[-1]["row_start"] + plan[-1]["row_count"], 100)
        self.assertLessEqual(
            max(item["row_count"] for item in plan)
            - min(item["row_count"] for item in plan),
            1,
        )
        self.assertEqual([item["device"] for item in plan], ["0", "1", "2", "3"])

    def test_rejects_duplicate_devices_and_out_of_range_selection(self):
        sharded = load_script()

        with self.assertRaisesRegex(ValueError, "devices must be unique"):
            sharded.plan_row_shards(
                total_rows=10,
                row_start=0,
                row_limit=10,
                devices=["0", "0"],
            )
        with self.assertRaisesRegex(ValueError, "exceeds source row count"):
            sharded.plan_row_shards(
                total_rows=10,
                row_start=9,
                row_limit=2,
                devices=["0"],
            )

    def test_plans_document_and_batch_aligned_shards(self):
        sharded = load_script()
        doc_ids = (
            ["doc-a"] * 4
            + ["doc-b"] * 3
            + ["doc-c"] * 5
            + ["doc-d"] * 2
            + ["doc-e"] * 4
            + ["doc-f"] * 3
            + ["doc-g"] * 2
        )

        plan = sharded.plan_document_aligned_shards(
            doc_ids=doc_ids,
            row_start=0,
            row_limit=len(doc_ids),
            devices=["0", "1", "2"],
            document_batch_size=2,
        )

        self.assertEqual(sum(item["row_count"] for item in plan), len(doc_ids))
        self.assertEqual(plan[0]["row_start"], 0)
        self.assertEqual(plan[-1]["row_end_exclusive"], len(doc_ids))
        self.assertTrue(all(item["doc_count"] % 2 == 0 for item in plan[:-1]))
        for item in plan[:-1]:
            boundary = item["row_end_exclusive"]
            self.assertNotEqual(doc_ids[boundary - 1], doc_ids[boundary])

    def test_document_aligned_plan_rejects_selection_that_splits_a_doc(self):
        sharded = load_script()

        with self.assertRaisesRegex(ValueError, "starts inside document"):
            sharded.plan_document_aligned_shards(
                doc_ids=["doc-a", "doc-a", "doc-b"],
                row_start=1,
                row_limit=2,
                devices=["0"],
                document_batch_size=1,
            )

    def test_merges_shards_in_source_order_and_preserves_provenance(self):
        sharded = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shard_paths = []
            for shard_index, values in enumerate(([0, 1], [2, 3, 4])):
                path = root / "shards" / f"shard-{shard_index:02d}" / "R_33" / "base.parquet"
                path.parent.mkdir(parents=True)
                pq.write_table(pa.table({"row_index": values, "activation_vector": [[1.0]] * len(values)}), path)
                pathlib.Path(str(path) + ".nla_meta.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "dataset_id": f"base_prefix_nano_R33_rows{values[0]}_{len(values)}",
                            "kind": "nla_dataset",
                            "row_count": len(values),
                            "extraction": {
                                "base_model": "/models/nano",
                                "row_start": values[0],
                                "row_limit": len(values),
                            },
                            "publication_provenance": {"runtime": {"sha256": "runtime-sha"}},
                        }
                    )
                )
                path.with_suffix(path.suffix + ".metadata.json").write_text(
                    json.dumps(
                        {
                            "row_count": len(values),
                            "publication_provenance": {"runtime": {"sha256": "runtime-sha"}},
                        }
                    )
                )
                shard_paths.append(path)
            output = root / "R_33" / "base.parquet"

            report = sharded.merge_layer_shards(
                shard_paths,
                output=output,
                layer=33,
                expected_rows=5,
                shard_plan=[
                    {"device": "0", "row_start": 0, "row_count": 2},
                    {"device": "1", "row_start": 2, "row_count": 3},
                ],
            )

            merged = pq.read_table(output)
            sidecar = yaml.safe_load(pathlib.Path(str(output) + ".nla_meta.yaml").read_text())

        self.assertEqual(merged.column("row_index").to_pylist(), [0, 1, 2, 3, 4])
        self.assertEqual(report["row_count"], 5)
        self.assertEqual(sidecar["row_count"], 5)
        self.assertEqual(sidecar["publication_provenance"]["runtime"]["sha256"], "runtime-sha")
        self.assertEqual(sidecar["extraction"]["sharding"]["shard_count"], 2)
        self.assertEqual(sidecar["dataset_id"], "base_prefix_nano_R33_rows0_5")
        self.assertEqual(
            [item["device"] for item in sidecar["extraction"]["sharding"]["shards"]],
            ["0", "1"],
        )

    def test_merge_rejects_mismatched_publication_provenance(self):
        sharded = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shard_paths = []
            for shard_index, runtime_sha in enumerate(("runtime-a", "runtime-b")):
                path = root / "shards" / f"shard-{shard_index:02d}" / "R_33" / "base.parquet"
                path.parent.mkdir(parents=True)
                pq.write_table(pa.table({"row_index": [shard_index]}), path)
                provenance = {"runtime": {"sha256": runtime_sha}}
                pathlib.Path(str(path) + ".nla_meta.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "row_count": 1,
                            "publication_provenance": provenance,
                        }
                    )
                )
                path.with_suffix(path.suffix + ".metadata.json").write_text(
                    json.dumps(
                        {
                            "row_count": 1,
                            "publication_provenance": provenance,
                        }
                    )
                )
                shard_paths.append(path)

            with self.assertRaisesRegex(ValueError, "publication provenance mismatch"):
                sharded.merge_layer_shards(
                    shard_paths,
                    output=root / "R_33" / "base.parquet",
                    layer=33,
                    expected_rows=2,
                )

    def test_merge_rejects_noncontiguous_declared_row_ranges(self):
        sharded = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            shard_paths = []
            for shard_index, row_start in enumerate((0, 3)):
                path = root / "shards" / f"shard-{shard_index:02d}" / "R_33" / "base.parquet"
                path.parent.mkdir(parents=True)
                pq.write_table(pa.table({"row_index": [row_start]}), path)
                sidecar = {
                    "row_count": 1,
                    "extraction": {"row_start": row_start, "row_limit": 1},
                }
                pathlib.Path(str(path) + ".nla_meta.yaml").write_text(
                    yaml.safe_dump(sidecar)
                )
                path.with_suffix(path.suffix + ".metadata.json").write_text(
                    json.dumps({"row_count": 1, "publication_provenance": None})
                )
                shard_paths.append(path)

            with self.assertRaisesRegex(ValueError, "not contiguous"):
                sharded.merge_layer_shards(
                    shard_paths,
                    output=root / "R_33" / "base.parquet",
                    layer=33,
                    expected_rows=2,
                )


if __name__ == "__main__":
    unittest.main()
