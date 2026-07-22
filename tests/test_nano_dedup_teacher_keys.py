import importlib.util
import pathlib
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = pathlib.Path(__file__).resolve().parents[1]
CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoDedupTeacherKeysTests(unittest.TestCase):
    def test_dedup_keeps_lowest_doc_suffix_and_extracts_prompt_text(self):
        dedup = load_script("nano_dedup_teacher_keys")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "ar_sft.parquet"
            output = root / "teacher_keys.parquet"
            table = pa.table(
                {
                    "doc_id": pa.array(
                        [
                            "HuggingFaceFW/fineweb:train:10500",
                            "HuggingFaceFW/fineweb:train:10500",
                            "HuggingFaceFW/fineweb:train:15500",
                            "HuggingFaceFW/fineweb:train:15500",
                            "HuggingFaceFW/fineweb:train:10501",
                        ],
                        type=pa.string(),
                    ),
                    "token_ids_prefix": pa.array(
                        [
                            [1, 2],
                            [1, 2, 3, 4],
                            [1, 2],
                            [1, 2, 3, 4],
                            [8, 9],
                        ],
                        type=pa.list_(pa.int32()),
                    ),
                    "n_raw_tokens": pa.array([2, 4, 2, 4, 2], type=pa.int64()),
                    "token_position": pa.array([1, 3, 1, 3, 1], type=pa.int64()),
                    "token_id": pa.array([2, 4, 2, 4, 9], type=pa.int64()),
                    "prompt": pa.array(
                        [
                            CRITIC_TEMPLATE.format(explanation="alpha"),
                            CRITIC_TEMPLATE.format(explanation="beta"),
                            CRITIC_TEMPLATE.format(explanation="alpha duplicate"),
                            CRITIC_TEMPLATE.format(explanation="beta duplicate"),
                            CRITIC_TEMPLATE.format(explanation="gamma"),
                        ],
                        type=pa.string(),
                    ),
                }
            )
            pq.write_table(table, source)

            report = dedup.build_dedup_teacher_keys(source=source, output=output)
            out = pq.read_table(output)

        self.assertEqual(report["source_doc_count"], 3)
        self.assertEqual(report["kept_doc_count"], 2)
        self.assertEqual(report["duplicate_group_count"], 2)
        self.assertEqual(report["duplicate_component_count"], 1)
        self.assertEqual(report["output_rows"], 3)
        self.assertEqual(set(out.column("doc_id").to_pylist()), {
            "HuggingFaceFW/fineweb:train:10500",
            "HuggingFaceFW/fineweb:train:10501",
        })
        self.assertEqual(out.column("api_explanation").to_pylist(), ["alpha", "beta", "gamma"])

    def test_dedup_drops_docs_connected_by_any_row_prefix(self):
        dedup = load_script("nano_dedup_teacher_keys")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "ar_sft.parquet"
            output = root / "teacher_keys.parquet"
            table = pa.table(
                {
                    "doc_id": pa.array(
                        [
                            "HuggingFaceFW/fineweb:train:10500",
                            "HuggingFaceFW/fineweb:train:10500",
                            "HuggingFaceFW/fineweb:train:15500",
                            "HuggingFaceFW/fineweb:train:15500",
                            "HuggingFaceFW/fineweb:train:20500",
                        ],
                        type=pa.string(),
                    ),
                    "token_ids_prefix": pa.array(
                        [
                            [1, 2],
                            [1, 2, 3, 4],
                            [1, 2],
                            [1, 2, 8, 9],
                            [7, 8],
                        ],
                        type=pa.list_(pa.int32()),
                    ),
                    "n_raw_tokens": pa.array([2, 4, 2, 4, 2], type=pa.int64()),
                    "token_position": pa.array([1, 3, 1, 3, 1], type=pa.int64()),
                    "prompt": pa.array(
                        [
                            CRITIC_TEMPLATE.format(explanation="alpha"),
                            CRITIC_TEMPLATE.format(explanation="beta"),
                            CRITIC_TEMPLATE.format(explanation="alpha duplicate"),
                            CRITIC_TEMPLATE.format(explanation="diverged duplicate"),
                            CRITIC_TEMPLATE.format(explanation="gamma"),
                        ],
                        type=pa.string(),
                    ),
                }
            )
            pq.write_table(table, source)

            report = dedup.build_dedup_teacher_keys(source=source, output=output)
            out = pq.read_table(output)

        self.assertEqual(report["source_doc_count"], 3)
        self.assertEqual(report["kept_doc_count"], 2)
        self.assertEqual(report["duplicate_group_count"], 1)
        self.assertEqual(report["duplicate_component_count"], 1)
        self.assertEqual(report["output_rows"], 3)
        self.assertEqual(
            set(out.column("doc_id").to_pylist()),
            {
                "HuggingFaceFW/fineweb:train:10500",
                "HuggingFaceFW/fineweb:train:20500",
            },
        )

    def test_dedup_drops_docs_connected_by_text_content(self):
        dedup = load_script("nano_dedup_teacher_keys")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "ar_sft.parquet"
            output = root / "teacher_keys.parquet"
            table = pa.table(
                {
                    "doc_id": pa.array(
                        [
                            "HuggingFaceFW/fineweb:train:10500",
                            "HuggingFaceFW/fineweb:train:15500",
                            "HuggingFaceFW/fineweb:train:20500",
                        ],
                        type=pa.string(),
                    ),
                    "token_ids_prefix": pa.array(
                        [
                            [1, 2],
                            [9, 10],
                            [20, 21],
                        ],
                        type=pa.list_(pa.int32()),
                    ),
                    "detokenized_text_truncated": pa.array(
                        [
                            "same visible source",
                            "same visible source",
                            "different visible source",
                        ],
                        type=pa.string(),
                    ),
                    "n_raw_tokens": pa.array([2, 2, 2], type=pa.int64()),
                    "token_position": pa.array([1, 1, 1], type=pa.int64()),
                    "prompt": pa.array(
                        [
                            CRITIC_TEMPLATE.format(explanation="alpha"),
                            CRITIC_TEMPLATE.format(explanation="alpha duplicate"),
                            CRITIC_TEMPLATE.format(explanation="gamma"),
                        ],
                        type=pa.string(),
                    ),
                }
            )
            pq.write_table(table, source)

            report = dedup.build_dedup_teacher_keys(source=source, output=output)
            out = pq.read_table(output)

        self.assertEqual(report["content_columns"], ["token_ids_prefix", "detokenized_text_truncated"])
        self.assertEqual(report["kept_doc_count"], 2)
        self.assertEqual(report["duplicate_group_count"], 1)
        self.assertEqual(
            set(out.column("doc_id").to_pylist()),
            {
                "HuggingFaceFW/fineweb:train:10500",
                "HuggingFaceFW/fineweb:train:20500",
            },
        )


if __name__ == "__main__":
    unittest.main()
