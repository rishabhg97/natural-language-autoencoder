import importlib.util
import pathlib
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_rows(path: pathlib.Path, rows: dict) -> None:
    pq.write_table(pa.table(rows), path)


class NanoR33SourceRowsTests(unittest.TestCase):
    def test_batch_key_scan_returns_only_matching_row_offsets(self):
        module = load_script("nano_r33_source_rows")
        batch = pa.record_batch(
            {
                "doc_id": ["a", "b", "c"],
                "token_position": [2, 4, 6],
                "n_raw_tokens": [3, 5, 7],
                "activation_vector": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            }
        )

        matches = module.matching_batch_rows(
            batch,
            {
                ("position", "b", 4),
                ("raw_tokens", "b", 5),
                ("position", "missing", 8),
            },
        )

        self.assertEqual(
            matches,
            [(1, [("position", "b", 4), ("raw_tokens", "b", 5)])],
        )

    def test_provenance_key_priority(self):
        module = load_script("nano_r33_source_rows")

        self.assertEqual(
            module.provenance_key({"sample_uuid": "u", "doc_id": "d"}),
            ("uuid", "u"),
        )
        self.assertEqual(
            module.provenance_key({"doc_id": "d", "token_position": 8}),
            ("position", "d", 8),
        )
        self.assertEqual(
            module.provenance_key({"doc_id": "d", "n_raw_tokens": 9}),
            ("raw_tokens", "d", 9),
        )

    def test_duplicate_source_key_fails(self):
        module = load_script("nano_r33_source_rows")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "base.parquet"
            write_rows(
                path,
                {
                    "doc_id": ["d", "d"],
                    "token_position": [8, 8],
                    "n_raw_tokens": [9, 9],
                    "token_id": [11, 11],
                    "token_ids_prefix": [[1, 11], [1, 11]],
                    "activation_vector": [[0.0, 1.0], [0.0, 1.0]],
                },
            )

            with self.assertRaisesRegex(module.SourceRowError, "duplicate source provenance key"):
                module.resolve_source_rows(
                    path,
                    [{"doc_id": "d", "token_position": 8}],
                )

    def test_streaming_lookup_returns_only_requested_rows(self):
        module = load_script("nano_r33_source_rows")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "base.parquet"
            write_rows(
                path,
                {
                    "doc_id": ["a", "b", "c"],
                    "token_position": [2, 4, 6],
                    "n_raw_tokens": [3, 5, 7],
                    "token_id": [12, 14, 16],
                    "token_ids_prefix": [[1, 2, 12], [1, 2, 3, 4, 14], [1, 2, 3, 4, 5, 6, 16]],
                    "activation_vector": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                    "detokenized_text_truncated": ["source a", "source b", "source c"],
                    "api_explanation": ["alpha", "beta", "gamma"],
                },
            )

            rows = module.resolve_source_rows(
                path,
                [
                    {"doc_id": "b", "token_position": 4},
                    {"doc_id": "c", "n_raw_tokens": 7},
                ],
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[("position", "b", 4)]["token_id"], 14)
        self.assertEqual(rows[("position", "b", 4)]["api_explanation"], "beta")
        self.assertEqual(
            rows[("position", "b", 4)]["detokenized_text_truncated"],
            "source b",
        )
        self.assertEqual(rows[("raw_tokens", "c", 7)]["token_id"], 16)

    def test_missing_source_row_and_missing_required_columns_fail(self):
        module = load_script("nano_r33_source_rows")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            valid = root / "valid.parquet"
            write_rows(
                valid,
                {
                    "doc_id": ["a"],
                    "token_position": [2],
                    "n_raw_tokens": [3],
                    "token_ids_prefix": [[1, 2, 3]],
                    "activation_vector": [[1.0, 0.0]],
                },
            )
            with self.assertRaisesRegex(module.SourceRowError, "missing source rows"):
                module.resolve_source_rows(valid, [{"doc_id": "missing", "token_position": 2}])

            invalid = root / "invalid.parquet"
            write_rows(invalid, {"doc_id": ["a"], "token_position": [2]})
            with self.assertRaisesRegex(module.SourceRowError, "required columns"):
                module.resolve_source_rows(invalid, [{"doc_id": "a", "token_position": 2}])


if __name__ == "__main__":
    unittest.main()
