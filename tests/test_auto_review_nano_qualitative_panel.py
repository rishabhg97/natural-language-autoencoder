from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "auto_review_nano_qualitative_panel.py"
    spec = importlib.util.spec_from_file_location("auto_review_nano_qualitative_panel", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AutoReviewNanoQualitativePanelTests(unittest.TestCase):
    def test_structural_review_flags_only_rows_with_automatic_reasons(self):
        module = load_script()
        panel = {
            "review_complete": False,
            "splits": {
                "validation": {
                    "rows": [
                        {"row_index": 11, "automatic_flag_reasons": []},
                        {"row_index": 12, "automatic_flag_reasons": ["repetition"]},
                    ]
                },
                "test": {
                    "rows": [
                        {"row_index": 21, "automatic_flag_reasons": ["empty"]},
                    ]
                },
            },
        }

        reviews = module.build_structural_reviews(panel)

        self.assertFalse(reviews["decisions"]["validation:11"]["flagged"])
        self.assertTrue(reviews["decisions"]["validation:12"]["flagged"])
        self.assertTrue(reviews["decisions"]["test:21"]["flagged"])
        self.assertEqual(reviews["review_mode"], "automatic_structural_v1")
        self.assertIn("not a semantic human review", reviews["limitations"])


if __name__ == "__main__":
    unittest.main()
