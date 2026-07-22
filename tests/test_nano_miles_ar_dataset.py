import importlib.util
import contextlib
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock
from collections.abc import Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CharTokenizer:
    class Encoding(Mapping):
        def __init__(self, input_ids):
            self._data = {"input_ids": input_ids}

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    def __call__(self, text, **kwargs):
        ids = [ord(ch) for ch in text]
        if kwargs.get("add_special_tokens"):
            ids = [1] + ids
        return self.Encoding(ids)


class NanoMilesARDatasetVerifierTests(unittest.TestCase):
    def _write_ar_dataset(
        self,
        path,
        *,
        prompts=None,
        vectors=None,
        docs=None,
        token_prefixes=None,
        sidecar_updates=None,
    ):
        prompts = prompts or [
            CRITIC_TEMPLATE.format(explanation="alpha feature"),
            CRITIC_TEMPLATE.format(explanation="beta feature"),
            CRITIC_TEMPLATE.format(explanation="gamma feature"),
            CRITIC_TEMPLATE.format(explanation="delta feature"),
        ]
        vectors = vectors or [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(prompts))]
        docs = docs or ["doc-a", "doc-a", "doc-b", "doc-c"][: len(prompts)]
        columns = {
            "prompt": pa.array(prompts, type=pa.string()),
            "activation_vector": pa.array(vectors, type=pa.list_(pa.float32(), 4)),
            "doc_id": pa.array(docs),
            "activation_layer": pa.array([27] * len(prompts), type=pa.int64()),
        }
        if token_prefixes is not None:
            columns["token_ids_prefix"] = pa.array(token_prefixes, type=pa.list_(pa.int32()))
        table = pa.table(columns)
        pq.write_table(table, path)
        sidecar = {
            "kind": "nla_dataset",
            "schema_version": 1,
            "dataset_id": "tiny-ar",
            "stage": "ar_sft",
            "row_count": len(prompts),
            "extraction": {"d_model": 4, "layer_index": 27, "norm": "none"},
            "tokens": {"critic_suffix_ids": [ord(ch) for ch in "</text> <summary>"]},
            "prompt_templates": {"critic": CRITIC_TEMPLATE},
        }
        if sidecar_updates:
            sidecar.update(sidecar_updates)
        path.with_name(path.name + ".nla_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))

    def _write_family_manifest(self, path):
        path.write_text(
            json.dumps(
                {
                    "schema_version": "nano_content_family_manifest.v1",
                    "doc_assignments": {
                        "doc-a": "family-ab",
                        "doc-b": "family-ab",
                        "doc-c": "family-c",
                        "doc-d": "family-d",
                    },
                    "family_splits": {
                        "family-ab": "train",
                        "family-c": "validation",
                        "family-d": "test",
                    },
                    "overlap": {
                        "train_validation": [],
                        "train_test": [],
                        "validation_test": [],
                    },
                }
            )
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_verifier_accepts_valid_ar_sft_contract_and_reports_doc_splits(self):
        verifier = load_script("verify_nano_miles_ar_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            parquet = pathlib.Path(tmp) / "ar_sft.parquet"
            self._write_ar_dataset(parquet)

            report = verifier.verify_dataset(
                parquet,
                tokenizer=CharTokenizer(),
                expected_rows=4,
                expected_d_model=4,
                split_specs=((0.8, 0.1, 0.1),),
            )

        self.assertEqual(report["row_count"], 4)
        self.assertEqual(report["stage"], "ar_sft")
        self.assertEqual(report["activation"]["d_model"], 4)
        self.assertEqual(report["prompts"]["empty_explanation_count"], 0)
        self.assertEqual(report["critic_suffix"]["bad_count"], 0)
        self.assertEqual(report["splits"]["80/10/10"]["doc_overlap_count"], 0)

    def test_verifier_rejects_prompt_without_explanation_text(self):
        verifier = load_script("verify_nano_miles_ar_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            parquet = pathlib.Path(tmp) / "ar_sft.parquet"
            self._write_ar_dataset(parquet, prompts=[CRITIC_TEMPLATE.format(explanation="")], docs=["doc-a"])

            with self.assertRaisesRegex(ValueError, "empty explanation"):
                verifier.verify_dataset(parquet, tokenizer=CharTokenizer(), expected_d_model=4)

    def test_verifier_rejects_missing_critic_sidecar_fields(self):
        verifier = load_script("verify_nano_miles_ar_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            parquet = pathlib.Path(tmp) / "ar_sft.parquet"
            self._write_ar_dataset(parquet, sidecar_updates={"tokens": {}})

            with self.assertRaisesRegex(ValueError, "critic_suffix_ids"):
                verifier.verify_dataset(parquet, tokenizer=CharTokenizer(), expected_d_model=4)

    def test_verifier_rejects_cross_split_duplicate_token_prefixes(self):
        verifier = load_script("verify_nano_miles_ar_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            parquet = pathlib.Path(tmp) / "ar_sft.parquet"
            self._write_ar_dataset(
                parquet,
                prompts=[
                    CRITIC_TEMPLATE.format(explanation="alpha feature"),
                    CRITIC_TEMPLATE.format(explanation="beta feature"),
                ],
                docs=["doc-a", "doc-b"],
                token_prefixes=[[101, 102, 103, 104], [101, 102, 103, 104]],
            )

            with self.assertRaisesRegex(ValueError, "content-hash cross-split overlap"):
                verifier.verify_dataset(
                    parquet,
                    tokenizer=CharTokenizer(),
                    expected_rows=2,
                    expected_d_model=4,
                    split_specs=((0.5, 0.5, 0.0),),
                )

    def test_verifier_accepts_frozen_content_family_manifest(self):
        verifier = load_script("verify_nano_miles_ar_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "ar_sft.parquet"
            manifest = root / "families.json"
            self._write_ar_dataset(
                parquet,
                prompts=[
                    CRITIC_TEMPLATE.format(explanation="alpha feature"),
                    CRITIC_TEMPLATE.format(explanation="beta feature"),
                    CRITIC_TEMPLATE.format(explanation="gamma feature"),
                    CRITIC_TEMPLATE.format(explanation="delta feature"),
                ],
                docs=["doc-a", "doc-b", "doc-c", "doc-d"],
                token_prefixes=[
                    [101, 102, 103],
                    [101, 102, 103],
                    [201, 202, 203],
                    [301, 302, 303],
                ],
            )
            manifest_sha256 = self._write_family_manifest(manifest)

            report = verifier.verify_dataset(
                parquet,
                tokenizer=CharTokenizer(),
                expected_rows=4,
                expected_d_model=4,
                split_specs=(),
                content_family_manifest=manifest,
                content_family_manifest_sha256=manifest_sha256,
            )

        self.assertEqual(
            report["content_family_manifest_split"]["split_family_counts"],
            {"train": 1, "validation": 1, "test": 1},
        )

    def test_cli_can_skip_synthetic_split_checks_for_materialized_component_splits(self):
        verifier = load_script("verify_nano_miles_ar_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "ar_sft.parquet"
            report_json = root / "report.json"
            self._write_ar_dataset(
                parquet,
                prompts=[
                    CRITIC_TEMPLATE.format(explanation="alpha feature"),
                    CRITIC_TEMPLATE.format(explanation="beta feature"),
                ],
                docs=["doc-a", "doc-b"],
                token_prefixes=[[101, 102, 103, 104], [101, 102, 103, 104]],
            )

            argv = [
                "verify_nano_miles_ar_dataset.py",
                str(parquet),
                "--expected-rows",
                "2",
                "--expected-d-model",
                "4",
                "--skip-tokenizer-check",
                "--skip-synthetic-split-checks",
                "--report-json",
                str(report_json),
            ]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with mock.patch.object(sys, "argv", argv):
                    rc = verifier.main()

            report = json.loads(report_json.read_text())

        self.assertEqual(rc, 0)
        self.assertEqual(report["splits"], {})


if __name__ == "__main__":
    unittest.main()
