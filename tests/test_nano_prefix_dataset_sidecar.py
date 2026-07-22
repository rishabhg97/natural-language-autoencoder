import importlib.util
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


class NanoPrefixDatasetSidecarTests(unittest.TestCase):
    def test_ar_sidecar_uses_training_loader_schema(self):
        sidecars = load_script("nano_prefix_dataset_sidecar")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.parquet"
            ar = root / "ar.parquet"
            contract = root / "contract.yaml"

            pq.write_table(pa.table({"activation_vector": [[[1.0, 2.0]]]}), base)
            pq.write_table(pa.table({"prompt": ["p"], "activation_vector": [[[1.0, 2.0]]]}), ar)
            pathlib.Path(str(base) + ".nla_meta.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "nla_dataset_meta.v1",
                        "dataset_id": "base-r33",
                        "stage": "base_explained",
                        "row_count": 1,
                        "created_at": "2026-06-08T00:00:00Z",
                        "extraction": {
                            "base_model": "nano",
                            "d_model": 2,
                            "layer_index": 33,
                            "norm": "none",
                            "source_mode": "token_ids_prefix",
                        },
                        "parent_datasets": ["teacher.parquet"],
                    },
                    sort_keys=False,
                )
            )
            contract.write_text(
                yaml.safe_dump(
                    {
                        "kind": "nla_dataset",
                        "schema_version": 1,
                        "tokens": {
                            "injection_char": "x",
                            "injection_token_id": 7,
                            "injection_left_neighbor_id": 6,
                            "injection_right_neighbor_id": 8,
                            "critic_suffix_ids": [1, 2, 3],
                        },
                        "prompt_templates": {
                            "actor": "actor {injection_char}",
                            "critic": "critic {explanation}",
                        },
                        "extraction": {
                            "mse_scale": "sqrt_d_model",
                            "injection_scale": 75,
                        },
                    },
                    sort_keys=False,
                )
            )

            meta = sidecars.write_ar_sidecar(
                base_path=base,
                ar_path=ar,
                contract_path=contract,
                layer=33,
                slug="r33_prefix",
            )

            written = yaml.safe_load(pathlib.Path(str(ar) + ".nla_meta.yaml").read_text())

        self.assertEqual(meta, written)
        self.assertEqual(written["kind"], "nla_dataset")
        self.assertEqual(written["schema_version"], 1)
        self.assertTrue(written["keep_debug_metadata"])
        self.assertEqual(written["stage"], "ar_sft")
        self.assertEqual(written["row_count"], 1)
        self.assertEqual(written["extraction"]["d_model"], 2)
        self.assertEqual(written["extraction"]["layer_index"], 33)
        self.assertEqual(written["extraction"]["mse_scale"], "sqrt_d_model")
        self.assertEqual(written["extraction"]["injection_scale"], 75)
        self.assertEqual(written["critic"]["extraction_layer_index"], 33)
        self.assertEqual(written["tokens"]["critic_suffix_ids"], [1, 2, 3])
        self.assertEqual(written["prompt_templates"]["critic"], "critic {explanation}")
        self.assertEqual(
            written["parent_datasets"],
            ["teacher.parquet", "base-r33", str(contract)],
        )

    def test_av_sidecar_uses_training_loader_schema(self):
        sidecars = load_script("nano_prefix_dataset_sidecar")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            base = root / "base.parquet"
            av = root / "av.parquet"
            contract = root / "contract.yaml"

            pq.write_table(pa.table({"activation_vector": [[[1.0, 2.0]]]}), base)
            pq.write_table(
                pa.table(
                    {
                        "prompt": [[{"role": "user", "content": "<INJECT>"}]],
                        "response": ["<explanation>x</explanation>"],
                        "activation_vector": [[[1.0, 2.0]]],
                    }
                ),
                av,
            )
            pathlib.Path(str(base) + ".nla_meta.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": "nla_dataset_meta.v1",
                        "dataset_id": "base-r33",
                        "stage": "base_explained",
                        "row_count": 1,
                        "created_at": "2026-06-08T00:00:00Z",
                        "extraction": {
                            "base_model": "nano",
                            "d_model": 2,
                            "layer_index": 33,
                            "norm": "none",
                            "source_mode": "token_ids_prefix",
                        },
                        "parent_datasets": ["teacher.parquet"],
                    },
                    sort_keys=False,
                )
            )
            contract.write_text(
                yaml.safe_dump(
                    {
                        "tokens": {
                            "injection_char": "x",
                            "injection_token_id": 7,
                            "injection_left_neighbor_id": 6,
                            "injection_right_neighbor_id": 8,
                            "critic_suffix_ids": [1, 2, 3],
                        },
                        "prompt_templates": {
                            "actor": "actor {injection_char}",
                            "critic": "critic {explanation}",
                        },
                    },
                    sort_keys=False,
                )
            )

            meta = sidecars.write_av_sidecar(
                base_path=base,
                av_path=av,
                contract_path=contract,
                layer=33,
                slug="r33_prefix",
            )

            written = yaml.safe_load(pathlib.Path(str(av) + ".nla_meta.yaml").read_text())

        self.assertEqual(meta, written)
        self.assertEqual(written["kind"], "nla_dataset")
        self.assertEqual(written["schema_version"], 1)
        self.assertEqual(written["stage"], "av_sft")
        self.assertEqual(written["row_count"], 1)
        self.assertEqual(written["extraction"]["layer_index"], 33)
        self.assertEqual(written["tokens"]["injection_char"], "x")
        self.assertIsNone(written["tokens"]["critic_suffix_ids"])
        self.assertEqual(written["prompt_templates"]["actor"], "actor {injection_char}")


if __name__ == "__main__":
    unittest.main()
