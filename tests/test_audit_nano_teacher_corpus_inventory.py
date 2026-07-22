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
    path = ROOT / "scripts/audit_nano_teacher_corpus_inventory.py"
    spec = importlib.util.spec_from_file_location(
        "audit_nano_teacher_corpus_inventory",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AuditNanoTeacherCorpusInventoryTests(unittest.TestCase):
    def test_production_inventory_covers_all_historical_numeric_docs(self):
        config = yaml.safe_load(
            (
                ROOT
                / "configs/nano_data/publication/"
                "teacher_corpus_external_boundary_inventory.yaml"
            ).read_text()
        )

        self.assertEqual(config["known_exposed_numeric_doc_ranges"], [[0, 38161]])
        self.assertTrue(config["outputs"]["report_json"].endswith("_v2.json"))

    def test_extracts_prompt_teacher_text_and_external_coverage(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "candidate_ar_sft.parquet"
            pq.write_table(
                pa.table(
                    {
                        "doc_id": ["dataset:10", "dataset:11", "dataset:30"],
                        "n_raw_tokens": [4, 5, 6],
                        "prompt": [
                            "Summary: <text>alpha</text> <summary>",
                            "Summary: <text>beta</text> <summary>",
                            "Summary: <text>gamma</text> <summary>",
                        ],
                    }
                ),
                source,
            )
            output = root / "report.json"
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": module.SCHEMA_VERSION,
                        "roots": [str(root)],
                        "patterns": ["**/*ar_sft*.parquet"],
                        "known_exposed_numeric_doc_ranges": [[10, 20]],
                        "outputs": {"report_json": str(output)},
                    }
                )
            )

            report = module.run_audit(config_path)
            frozen = json.loads(output.read_text())

        table = report["tables"][0]
        self.assertEqual(table["text_mode"], "prompt_text_tags")
        self.assertEqual(table["empty_explanation_count"], 0)
        self.assertEqual(table["external_numeric_doc_count"], 1)
        self.assertEqual(table["numeric_doc_suffix_min"], 10)
        self.assertEqual(table["numeric_doc_suffix_max"], 30)
        self.assertTrue(table["usable_teacher_text"])
        self.assertTrue(table["usable_join_keys"])
        self.assertEqual(frozen["summary"]["external_teacher_table_count"], 1)

    def test_empty_prompt_explanation_is_not_usable(self):
        module = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "empty.parquet"
            pq.write_table(
                pa.table(
                    {
                        "doc_id": ["dataset:1"],
                        "token_position": [3],
                        "prompt": ["Summary: <text> </text> <summary>"],
                    }
                ),
                path,
            )

            report = module.audit_table(
                path,
                known_exposed_ranges=[(1, 1)],
                batch_size=16,
            )

        self.assertEqual(report["empty_explanation_count"], 1)
        self.assertFalse(report["usable_teacher_text"])


if __name__ == "__main__":
    unittest.main()
