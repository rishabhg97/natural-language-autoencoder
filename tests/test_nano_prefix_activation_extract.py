import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoPrefixActivationExtractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_explanation_from_prompt_uses_text_tags(self):
        extract = load_script("nano_prefix_activation_extract")

        prompt = "Summary of the following text: <text>hello world</text> <summary>"

        self.assertEqual(extract.extract_explanation_from_prompt(prompt), "hello world")

    def test_publication_provenance_requires_complete_runtime_and_model_binding(self):
        extract = load_script("nano_prefix_activation_extract")
        model = self.root / "model"
        model.mkdir()
        source = self.root / "source.parquet"
        source.write_bytes(b"source-rows")
        model_report = self.root / "model_fingerprint.json"
        model_report.write_text(
            json.dumps(
                {
                    "root": str(model.resolve()),
                    "sha256": "model-sha",
                    "file_count": 73,
                    "total_bytes": 123,
                }
            )
        )
        runtime_report = self.root / "runtime.json"
        runtime_report.write_text(
            json.dumps(
                {
                    "runtime": {
                        "complete": True,
                        "sha256": "runtime-sha",
                    }
                }
            )
        )
        args = SimpleNamespace(
            publication_mode=True,
            model_id=str(model),
            source_parquet=source,
            model_fingerprint_json=model_report,
            runtime_provenance_json=runtime_report,
        )
        execution_profile = {
            "deterministic_algorithms": True,
            "allow_tf32": False,
            "cudnn_benchmark": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
            "seed": 20260709,
        }

        provenance = extract.publication_provenance_from_args(
            args,
            execution_profile=execution_profile,
        )

        self.assertEqual(provenance["model"]["sha256"], "model-sha")
        self.assertEqual(provenance["runtime"]["sha256"], "runtime-sha")
        self.assertEqual(provenance["source_parquet"]["size_bytes"], 11)
        self.assertEqual(provenance["execution"], execution_profile)

        runtime_report.write_text(
            json.dumps({"runtime": {"complete": False, "sha256": "runtime-sha"}})
        )
        with self.assertRaisesRegex(ValueError, "complete runtime fingerprint"):
            extract.publication_provenance_from_args(
                args,
                execution_profile=execution_profile,
            )

    def test_publication_execution_rejects_nondeterministic_profile(self):
        extract = load_script("nano_prefix_activation_extract")
        args = SimpleNamespace(
            publication_mode=True,
            deterministic_algorithms=False,
            allow_tf32=False,
            cudnn_benchmark=False,
            float32_matmul_precision="highest",
            cublas_workspace_config=":4096:8",
            seed=20260709,
        )

        with self.assertRaisesRegex(ValueError, "deterministic_algorithms"):
            extract.validate_execution_profile(args)

    def test_configure_execution_profile_applies_all_controls(self):
        extract = load_script("nano_prefix_activation_extract")
        state = {}
        fake_torch = SimpleNamespace(
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=True, benchmark=True),
            ),
            cuda=SimpleNamespace(
                manual_seed_all=lambda seed: state.setdefault("cuda_seed", seed)
            ),
            manual_seed=lambda seed: state.setdefault("seed", seed),
            use_deterministic_algorithms=lambda enabled: state.setdefault(
                "deterministic", enabled
            ),
            set_float32_matmul_precision=lambda value: state.setdefault(
                "matmul_precision", value
            ),
        )
        args = SimpleNamespace(
            publication_mode=True,
            deterministic_algorithms=True,
            allow_tf32=False,
            cudnn_benchmark=False,
            float32_matmul_precision="highest",
            cublas_workspace_config=":4096:8",
            seed=20260709,
        )

        with mock.patch.dict(os.environ, {}, clear=False):
            profile = extract.configure_extraction_execution(args, fake_torch)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

        self.assertEqual(state["seed"], 20260709)
        self.assertEqual(state["cuda_seed"], 20260709)
        self.assertTrue(state["deterministic"])
        self.assertEqual(state["matmul_precision"], "highest")
        self.assertFalse(fake_torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(fake_torch.backends.cudnn.allow_tf32)
        self.assertFalse(fake_torch.backends.cudnn.benchmark)
        self.assertEqual(profile["seed"], 20260709)

    def test_collect_source_records_validates_prefix_keys(self):
        extract = load_script("nano_prefix_activation_extract")
        source = self.root / "source.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "prompt": "Summary of the following text: <text>alpha</text> <summary>",
                        "doc_id": "HuggingFaceFW/fineweb:train:10500",
                        "token_ids_prefix": [11, 12, 13],
                        "n_raw_tokens": 3,
                        "token_position": 2,
                        "token_id": 13,
                        "detokenized_text_truncated": "alpha",
                    },
                    {
                        "prompt": "Summary of the following text: <text>beta</text> <summary>",
                        "doc_id": "HuggingFaceFW/fineweb:train:10501",
                        "token_ids_prefix": [21, 22],
                        "n_raw_tokens": 2,
                        "token_position": 1,
                        "token_id": 99,
                        "detokenized_text_truncated": "beta",
                    },
                ]
            ),
            source,
        )

        records, report = extract.collect_source_records(source, row_limit=1)

        self.assertEqual(report["rows_read"], 1)
        self.assertEqual(report["rows_kept"], 1)
        self.assertEqual(records[0]["api_explanation"], "alpha")
        self.assertEqual(records[0]["selected_position"], 2)

        with self.assertRaisesRegex(ValueError, "token_id mismatch"):
            extract.collect_source_records(source)

    def test_plan_prefix_batch_pads_and_selects_last_prefix_token(self):
        extract = load_script("nano_prefix_activation_extract")

        groups = extract.group_records_by_doc(
            [
                {"doc_id": "doc-a", "token_ids_prefix": [3, 4, 5], "selected_position": 2},
                {"doc_id": "doc-b", "token_ids_prefix": [8], "selected_position": 0},
            ]
        )
        batch = extract.plan_group_batch(
            groups,
            pad_token_id=0,
        )

        self.assertEqual(batch["input_ids"], [[3, 4, 5], [8, 0, 0]])
        self.assertEqual(batch["attention_mask"], [[1, 1, 1], [1, 0, 0]])
        self.assertEqual(batch["selected_positions"], [(0, 2), (1, 0)])

    def test_group_records_by_doc_reuses_longest_compatible_prefix(self):
        extract = load_script("nano_prefix_activation_extract")

        groups = extract.group_records_by_doc(
            [
                {"doc_id": "doc-a", "token_ids_prefix": [1, 2, 3], "selected_position": 2},
                {"doc_id": "doc-a", "token_ids_prefix": [1, 2, 3, 4, 5], "selected_position": 4},
                {"doc_id": "doc-b", "token_ids_prefix": [9, 10], "selected_position": 1},
            ]
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["doc_id"], "doc-a")
        self.assertEqual(groups[0]["token_ids"], [1, 2, 3, 4, 5])
        self.assertEqual([record["selected_position"] for record in groups[0]["records"]], [2, 4])

        with self.assertRaisesRegex(ValueError, "not a prefix"):
            extract.group_records_by_doc(
                [
                    {"doc_id": "doc-a", "token_ids_prefix": [1, 2, 3], "selected_position": 2},
                    {"doc_id": "doc-a", "token_ids_prefix": [1, 7, 3, 4], "selected_position": 3},
                ]
            )


if __name__ == "__main__":
    unittest.main()
