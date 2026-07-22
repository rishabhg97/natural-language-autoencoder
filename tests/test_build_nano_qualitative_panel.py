import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "build_nano_qualitative_panel.py"
    spec = importlib.util.spec_from_file_location("build_nano_qualitative_panel", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def generated_row(split: str, row_index: int, text: str) -> dict:
    return {
        "split": split,
        "row_index": row_index,
        "doc_id": f"source-{row_index % 2}:{row_index}",
        "token_position": row_index * 3,
        "n_raw_tokens": row_index * 3 + 1,
        "target_explanation": f"teacher reference {row_index}",
        "controls": {
            "real": {
                "generated": f"<explanation>{text}</explanation>",
                "parsed": {"explanation": text, "closed": True, "usable": True},
            }
        },
    }


class BuildNanoQualitativePanelTests(unittest.TestCase):
    def test_stratified_selection_is_deterministic_and_covers_document_types(self):
        module = load_script()
        rows = [
            {
                "row_index": index,
                "doc_type": f"type-{index % 3}",
                "token_position": index * 5,
                "activation_norm": float(index + 1),
                "explanation_length": 10 + index,
            }
            for index in range(30)
        ]

        first = module.select_stratified_panel(rows, panel_size=12, seed=17)
        second = module.select_stratified_panel(rows, panel_size=12, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual({row["doc_type"] for row in first}, {"type-0", "type-1", "type-2"})
        self.assertEqual(len({row["row_index"] for row in first}), 12)

    def test_automatic_flags_find_encoded_repetition_and_length_regression(self):
        module = load_script()

        encoded = module.automatic_flag_reasons(
            "QWxhZGRpbjpvcGVuIHNlc2FtZV9sb25nX2VuY29kZWRfc3RyaW5n",
            "A readable explanation with several ordinary words for comparison.",
        )
        repeated = module.automatic_flag_reasons(
            "same phrase repeated same phrase repeated same phrase repeated same phrase repeated",
            "A readable explanation with several ordinary words for comparison.",
        )
        short = module.automatic_flag_reasons(
            "brief",
            "This reference explanation contains enough words to make a severe shortening obvious and reviewable.",
        )

        self.assertIn("encoded_looking", encoded)
        self.assertIn("repetition", repeated)
        self.assertIn("length_regression", short)

    def test_structured_feature_explanation_is_not_mislabeled_as_repetition(self):
        module = load_script()
        text = (
            "Syntax feature: The token requires a noun phrase that completes the clause. "
            "Discourse feature: The token requires context from the reported delivery problem. "
            "Genre feature: The register is informal, conversational, and suitable for support. "
            "Final-token constraint: The token requires continuation with a grammatical complement."
        )

        reasons = module.automatic_flag_reasons(text, text)

        self.assertNotIn("repetition", reasons)

    def test_report_stays_pending_until_every_selected_row_is_reviewed(self):
        module = load_script()
        candidate = [
            generated_row(split, offset + index, f"candidate explanation {index}")
            for split, offset in (("validation", 0), ("test", 100))
            for index in range(6)
        ]
        sft = [
            generated_row(split, offset + index, f"sft explanation {index}")
            for split, offset in (("validation", 0), ("test", 100))
            for index in range(6)
        ]
        source = {
            index: {
                "activation_vector": [float(index), 1.0],
                "detokenized_text_truncated": f"source text {index}",
            }
            for index in list(range(6)) + list(range(100, 106))
        }

        pending = module.build_panel_report(
            candidate,
            sft,
            source_rows_by_index=source,
            panel_size=4,
            seed=17,
        )
        self.assertEqual(pending["splits"]["validation"]["row_count"], 4)
        self.assertEqual(pending["splits"]["validation"]["reviewed_count"], 0)
        self.assertEqual(pending["splits"]["validation"]["flagged_count"], -1)

        decisions = {
            f"{split}:{row['row_index']}": {"flagged": False, "notes": "readable"}
            for split, split_report in pending["splits"].items()
            for row in split_report["rows"]
        }
        reviewed = module.build_panel_report(
            candidate,
            sft,
            source_rows_by_index=source,
            panel_size=4,
            seed=17,
            review_decisions=decisions,
        )
        self.assertEqual(reviewed["splits"]["validation"]["reviewed_count"], 4)
        self.assertEqual(reviewed["splits"]["validation"]["flagged_count"], 0)

    def test_target_explanation_mode_does_not_require_second_generation_file(self):
        module = load_script()
        candidate = [
            generated_row(split, offset + index, f"candidate explanation {index}")
            for split, offset in (("validation", 0), ("test", 100))
            for index in range(6)
        ]
        source = {
            index: {
                "activation_vector": [float(index), 1.0],
                "detokenized_text_truncated": f"source text {index}",
            }
            for index in list(range(6)) + list(range(100, 106))
        }

        report = module.build_panel_report(
            candidate,
            None,
            source_rows_by_index=source,
            panel_size=4,
            seed=17,
            reference_mode="target_explanation",
        )

        self.assertEqual(report["reference_mode"], "target_explanation")
        row = report["splits"]["validation"]["rows"][0]
        self.assertTrue(row["reference_text"].startswith("teacher reference"))
        self.assertTrue(row["source_text"].startswith("source text"))

    def test_empty_source_text_fails_closed(self):
        module = load_script()
        candidate = [
            generated_row(split, offset, "candidate explanation")
            for split, offset in (("validation", 0), ("test", 100))
        ]
        source = {
            0: {"activation_vector": [1.0, 0.0]},
            100: {"activation_vector": [1.0, 0.0]},
        }

        with self.assertRaisesRegex(
            module.QualitativePanelError,
            "detokenized source text",
        ):
            module.build_panel_report(
                candidate,
                None,
                source_rows_by_index=source,
                panel_size=1,
                seed=17,
                reference_mode="target_explanation",
            )


if __name__ == "__main__":
    unittest.main()
