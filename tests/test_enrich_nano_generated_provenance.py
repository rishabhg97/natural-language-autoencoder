import importlib.util
import json
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


def write_split(path: pathlib.Path, doc_ids: list[str], raw_tokens: list[int]) -> None:
    pq.write_table(
        pa.table(
            {
                "doc_id": doc_ids,
                "n_raw_tokens": raw_tokens,
                "token_position": [value - 1 for value in raw_tokens],
                "token_id": [100 + value for value in raw_tokens],
            }
        ),
        path,
    )


class EnrichNanoGeneratedProvenanceTests(unittest.TestCase):
    def test_enriches_global_row_indices_without_changing_controls(self):
        module = load_script("enrich_nano_generated_provenance")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            train = root / "train.parquet"
            validation = root / "validation.parquet"
            test = root / "test.parquet"
            write_split(train, ["train-a", "train-b"], [2, 3])
            write_split(validation, ["validation-a", "validation-b"], [4, 5])
            write_split(test, ["test-a"], [6])
            generated = root / "generated.jsonl"
            controls = {"real": {"generated": "<explanation>kept</explanation>"}}
            generated.write_text(
                json.dumps(
                    {
                        "split": "validation",
                        "row_index": 3,
                        "source_row_index": 3,
                        "controls": controls,
                    }
                )
                + "\n"
            )
            output = root / "enriched.jsonl"

            report = module.enrich_generated_jsonl(
                generated_jsonl=generated,
                train_parquet=train,
                validation_parquet=validation,
                test_parquet=test,
                output_jsonl=output,
            )
            row = json.loads(output.read_text())

        self.assertEqual(row["doc_id"], "validation-b")
        self.assertEqual(row["n_raw_tokens"], 5)
        self.assertEqual(row["token_position"], 4)
        self.assertEqual(row["controls"], controls)
        self.assertEqual(report["rows"], 1)
        self.assertEqual(report["missing_rows"], 0)

    def test_rejects_out_of_range_or_split_mismatched_rows(self):
        module = load_script("enrich_nano_generated_provenance")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            paths = [root / name for name in ("train.parquet", "validation.parquet", "test.parquet")]
            for path in paths:
                write_split(path, [path.stem], [2])
            generated = root / "generated.jsonl"
            generated.write_text(
                json.dumps({"split": "validation", "row_index": 99, "controls": {}})
                + "\n"
            )

            with self.assertRaisesRegex(module.ProvenanceEnrichmentError, "out of range"):
                module.enrich_generated_jsonl(
                    generated_jsonl=generated,
                    train_parquet=paths[0],
                    validation_parquet=paths[1],
                    test_parquet=paths[2],
                    output_jsonl=root / "out.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
