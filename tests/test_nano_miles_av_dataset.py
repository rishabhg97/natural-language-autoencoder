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


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CharTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        rendered = "".join(message["content"] for message in messages)
        if kwargs.get("add_generation_prompt"):
            rendered += "<assistant>"
        if kwargs.get("tokenize"):
            return [ord(ch) for ch in rendered]
        return rendered


class MappingTokenizer(CharTokenizer):
    class Encoding(Mapping):
        def __init__(self, input_ids):
            self._data = {"input_ids": input_ids}

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    def apply_chat_template(self, messages, **kwargs):
        ids = super().apply_chat_template(messages, **kwargs)
        if kwargs.get("tokenize"):
            return self.Encoding(ids)
        return ids


class NanoMilesAVDatasetVerifierTests(unittest.TestCase):
    def _write_av_dataset(self, path, prompts, *, docs=None, responses=None, token_prefixes=None):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import yaml
        except ImportError as exc:
            self.skipTest(f"missing parquet/yaml dependency: {exc}")

        vectors = [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(prompts))]
        columns = {
            "prompt": pa.array(
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                type=pa.list_(
                    pa.struct(
                        [
                            ("role", pa.string()),
                            ("content", pa.string()),
                        ]
                    )
                ),
            ),
            "response": pa.array(
                responses or [f"<explanation>feature {i}</explanation>" for i in range(len(prompts))],
                type=pa.string(),
            ),
            "activation_vector": pa.array(vectors, type=pa.list_(pa.float32(), 4)),
            "doc_id": pa.array(docs or ["doc-a", "doc-a", "doc-b", "doc-c"][: len(prompts)]),
        }
        if token_prefixes is not None:
            columns["token_ids_prefix"] = pa.array(token_prefixes, type=pa.list_(pa.int32()))
        table = pa.table(columns)
        pq.write_table(table, path)

        sidecar = {
            "kind": "nla_dataset",
            "schema_version": 1,
            "dataset_id": "tiny-av",
            "stage": "av_sft",
            "row_count": len(prompts),
            "extraction": {
                "base_model": "fake-nano",
                "d_model": 4,
                "layer_index": 27,
                "norm": "none",
                "corpus": "unit-test",
                "corpus_slice": {"start": 0, "length": len(prompts)},
                "positions_per_doc": 1,
            },
            "tokens": {
                "injection_char": "@",
                "injection_token_id": ord("@"),
                "injection_left_neighbor_id": ord(">"),
                "injection_right_neighbor_id": ord("<"),
            },
            "prompt_templates": {"actor": "probe <concept>{injection_char}</concept>"},
        }
        path.with_name(path.name + ".nla_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))

    def _write_family_manifest(self, path, *, split_duplicate_family=False):
        if split_duplicate_family:
            assignments = {
                "doc-a": "family-a",
                "doc-b": "family-b",
                "doc-c": "family-c",
                "doc-d": "family-d",
            }
            family_splits = {
                "family-a": "train",
                "family-b": "validation",
                "family-c": "validation",
                "family-d": "test",
            }
        else:
            assignments = {
                "doc-a": "family-ab",
                "doc-b": "family-ab",
                "doc-c": "family-c",
                "doc-d": "family-d",
            }
            family_splits = {
                "family-ab": "train",
                "family-c": "validation",
                "family-d": "test",
            }
        path.write_text(
            json.dumps(
                {
                    "schema_version": "nano_content_family_manifest.v1",
                    "doc_assignments": assignments,
                    "family_splits": family_splits,
                    "overlap": {
                        "train_validation": [],
                        "train_test": [],
                        "validation_test": [],
                    },
                }
            )
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_verifier_accepts_valid_av_sft_contract_and_reports_doc_splits(self):
        verifier = load_script("verify_nano_miles_av_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            parquet = pathlib.Path(tmp) / "av_sft.parquet"
            self._write_av_dataset(
                parquet,
                [
                    "probe <concept><INJECT></concept>",
                    "probe <concept><INJECT></concept>",
                    "probe <concept><INJECT></concept>",
                    "probe <concept><INJECT></concept>",
                ],
            )

            report = verifier.verify_dataset(
                parquet,
                tokenizer=CharTokenizer(),
                expected_rows=4,
                expected_d_model=4,
                split_specs=((0.8, 0.1, 0.1), (0.9, 0.05, 0.05)),
            )

        self.assertEqual(report["row_count"], 4)
        self.assertEqual(report["activation"]["d_model"], 4)
        self.assertEqual(report["responses"]["malformed_count"], 0)
        self.assertEqual(report["prompt_markers"]["bad_count"], 0)
        self.assertEqual(report["splits"]["80/10/10"]["doc_overlap_count"], 0)
        self.assertEqual(report["splits"]["90/5/5"]["row_count"], 4)

    def test_verifier_accepts_mapping_like_tokenizer_outputs(self):
        verifier = load_script("verify_nano_miles_av_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            parquet = pathlib.Path(tmp) / "av_sft.parquet"
            self._write_av_dataset(parquet, ["probe <concept><INJECT></concept>"])

            report = verifier.verify_dataset(
                parquet,
                tokenizer=MappingTokenizer(),
                expected_rows=1,
                expected_d_model=4,
            )

        self.assertEqual(report["prompt_markers"]["bad_count"], 0)

    def test_verifier_rejects_prompt_marker_without_canonical_neighbors(self):
        verifier = load_script("verify_nano_miles_av_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            parquet = pathlib.Path(tmp) / "av_sft.parquet"
            self._write_av_dataset(parquet, ["probe <INJECT>"])

            with self.assertRaisesRegex(ValueError, "valid injection marker"):
                verifier.verify_dataset(
                    parquet,
                    tokenizer=CharTokenizer(),
                    expected_rows=1,
                    expected_d_model=4,
                )

    def test_verifier_rejects_cross_split_duplicate_token_prefixes(self):
        verifier = load_script("verify_nano_miles_av_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            parquet = pathlib.Path(tmp) / "av_sft.parquet"
            self._write_av_dataset(
                parquet,
                ["probe <concept><INJECT></concept>", "probe <concept><INJECT></concept>"],
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

    def test_family_manifest_keeps_duplicate_prefix_family_in_one_split(self):
        verifier = load_script("verify_nano_miles_av_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "av_sft.parquet"
            manifest = root / "families.json"
            self._write_av_dataset(
                parquet,
                ["probe <concept><INJECT></concept>"] * 4,
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

        family_report = report["content_family_manifest_split"]
        self.assertEqual(family_report["doc_overlap_count"], 0)
        self.assertEqual(family_report["family_overlap_count"], 0)
        self.assertEqual(family_report["content_cross_split_overlap_count"], 0)
        self.assertEqual(family_report["split_row_counts"], {"train": 2, "validation": 1, "test": 1})

    def test_family_manifest_rejects_duplicate_prefix_across_splits(self):
        verifier = load_script("verify_nano_miles_av_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "av_sft.parquet"
            manifest = root / "families.json"
            self._write_av_dataset(
                parquet,
                ["probe <concept><INJECT></concept>"] * 4,
                docs=["doc-a", "doc-b", "doc-c", "doc-d"],
                token_prefixes=[
                    [101, 102, 103],
                    [101, 102, 103],
                    [201, 202, 203],
                    [301, 302, 303],
                ],
            )
            manifest_sha256 = self._write_family_manifest(
                manifest,
                split_duplicate_family=True,
            )

            with self.assertRaisesRegex(ValueError, "content-hash cross-split overlap"):
                verifier.verify_dataset(
                    parquet,
                    tokenizer=CharTokenizer(),
                    expected_rows=4,
                    expected_d_model=4,
                    split_specs=(),
                    content_family_manifest=manifest,
                    content_family_manifest_sha256=manifest_sha256,
                )

    def test_cli_can_skip_synthetic_split_checks_for_materialized_component_splits(self):
        verifier = load_script("verify_nano_miles_av_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parquet = root / "av_sft.parquet"
            report_json = root / "report.json"
            self._write_av_dataset(
                parquet,
                ["probe <concept><INJECT></concept>", "probe <concept><INJECT></concept>"],
                docs=["doc-a", "doc-b"],
                token_prefixes=[[101, 102, 103, 104], [101, 102, 103, 104]],
            )

            argv = [
                "verify_nano_miles_av_dataset.py",
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
