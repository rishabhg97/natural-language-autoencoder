import importlib.util
import pathlib
import tempfile
import textwrap
import unittest
from unittest import mock

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


class NanoARLayerSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_boundary_spec_expands_ranges_and_commas(self):
        sweep = load_script("nano_ar_layer_sweep")

        self.assertEqual(sweep.parse_boundary_spec("R25-R27,R_30,34"), [25, 26, 27, 30, 34])

    def test_teacher_position_requests_filter_slice_and_derive_position(self):
        sweep = load_script("nano_ar_layer_sweep")
        teacher_path = self.root / "teacher_keys.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "doc_id": "HuggingFaceFW/fineweb:train:99",
                        "n_raw_tokens": 56,
                        "token_id": 1,
                        "api_explanation": "outside",
                    },
                    {
                        "doc_id": "HuggingFaceFW/fineweb:train:100",
                        "n_raw_tokens": 57,
                        "token_id": 2,
                        "api_explanation": "inside",
                    },
                    {
                        "doc_id": "HuggingFaceFW/fineweb:train:102",
                        "n_raw_tokens": 59,
                        "token_id": 3,
                        "api_explanation": "inside gap",
                    },
                ]
            ),
            teacher_path,
        )

        requests, summary = sweep._load_teacher_position_requests(
            teacher_path,
            corpus_start=100,
            corpus_length=3,
        )

        self.assertEqual(summary["requested_rows"], 2)
        self.assertEqual(summary["outside_slice_rows"], 1)
        self.assertEqual(summary["missing_numeric_suffix_count"], 1)
        self.assertEqual(requests["HuggingFaceFW/fineweb:train:100"][0]["position"], 56)
        self.assertEqual(requests["HuggingFaceFW/fineweb:train:102"][0]["token_id"], 3)

    def test_positions_from_teacher_requests_validate_token_keys(self):
        sweep = load_script("nano_ar_layer_sweep")
        payload = {"skipped": {}}

        positions = sweep._positions_from_teacher_requests(
            [10, 11, 12, 13, 14],
            [
                {"position": 2, "token_id": 12, "n_raw_tokens": 3},
                {"position": 3, "token_id": 99, "n_raw_tokens": 4},
                {"position": 8, "token_id": 14, "n_raw_tokens": 9},
                {"position": 4, "token_id": 14, "n_raw_tokens": 99},
            ],
            payload,
        )

        self.assertEqual(positions, [2])
        self.assertEqual(payload["skipped"]["teacher_token_mismatch"], 1)
        self.assertEqual(payload["skipped"]["teacher_position_oob"], 1)
        self.assertEqual(payload["skipped"]["teacher_n_raw_mismatch"], 1)

    def test_build_queue_doc_creates_extract_and_score_items(self):
        sweep = load_script("nano_ar_layer_sweep")
        queue = sweep.build_queue_doc(
            code_root=pathlib.Path("/repo"),
            python_bin="/venv/bin/python",
            layers=[25, 26, 27],
            output_root=pathlib.Path("/out/layer_sweep"),
            teacher_table=pathlib.Path("/out/teacher.parquet"),
            model_id="/models/nano",
            corpus_start=10500,
            corpus_length=2048,
            positions_per_doc=10,
            score_top_k=5,
        )

        self.assertEqual(queue["schema_version"], "nano_ar_layer_sweep_queue.v1")
        self.assertEqual([item["name"] for item in queue["items"]], ["extract-r25-r27", "score-r25-r27"])
        self.assertIn("extract", queue["items"][0]["command"])
        self.assertIn("score", queue["items"][1]["command"])
        self.assertIn("R25-R27", queue["items"][0]["command"])

    def test_queue_process_next_item_runs_command_and_updates_status(self):
        sweep = load_script("nano_ar_layer_sweep")
        queue_path = self.root / "queue.yaml"
        queue_path.write_text(
            textwrap.dedent(
                """
                schema_version: nano_ar_layer_sweep_queue.v1
                defaults: {}
                items:
                  - name: unit
                    status: pending
                    command: [python, -c, "print('ok')"]
                """
            )
        )

        with mock.patch.object(sweep, "_run_logged") as run_logged:
            result = sweep.process_next_item(queue_path)

        self.assertEqual(result["status"], "complete")
        run_logged.assert_called_once()
        updated = yaml.safe_load(queue_path.read_text())
        self.assertEqual(updated["items"][0]["status"], "complete")

    def test_score_layers_prefers_text_predictable_layer_over_mean_control(self):
        sweep = load_script("nano_ar_layer_sweep")
        output_root = self.root / "sweep"
        layer_dir = output_root / "R_25"
        layer_dir.mkdir(parents=True)
        teacher_path = self.root / "teacher.parquet"
        rows = []
        teacher_rows = []
        for i in range(40):
            label = "alpha" if i % 2 == 0 else "beta"
            sign = 1.0 if label == "alpha" else -1.0
            doc_id = f"doc-{i:03d}"
            row = {
                "doc_id": doc_id,
                "token_position": 10,
                "token_id": 100 + i,
                "n_raw_tokens": 11,
                "activation_layer": 25,
                "activation_vector": [sign, 0.0],
            }
            rows.append(row)
            teacher_rows.append(
                {
                    "doc_id": doc_id,
                    "token_position": 10,
                    "token_id": 100 + i,
                    "n_raw_tokens": 11,
                    "api_explanation": label,
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), layer_dir / "base.parquet")
        pq.write_table(pa.Table.from_pylist(teacher_rows), teacher_path)

        report = sweep.score_layers(
            layers=[25],
            output_root=output_root,
            teacher_table=teacher_path,
            report_json=self.root / "score.json",
            validation_limit=8,
            test_limit=8,
            hash_dim=64,
            knn_k=3,
            seed=0,
        )

        layer_report = report["layers"][0]
        self.assertEqual(layer_report["layer"], 25)
        self.assertEqual(layer_report["matched_rows"], 40)
        self.assertLess(
            layer_report["validation"]["teacher_knn_normalized_mse"],
            layer_report["validation"]["mean_normalized_mse"],
        )


if __name__ == "__main__":
    unittest.main()
