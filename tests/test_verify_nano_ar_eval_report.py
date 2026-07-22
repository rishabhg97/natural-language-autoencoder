from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "verify_nano_ar_eval_report.py"
    spec = importlib.util.spec_from_file_location("verify_nano_ar_eval_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class VerifyNanoArEvalReportTests(unittest.TestCase):
    def test_directional_pass_does_not_imply_raw_magnitude_pass(self):
        verifier = load_script()
        controls = {}
        for name, nmse in {
            "teacher": 0.28,
            "teacher_shuffled": 0.96,
            "blank": 0.75,
            "generic": 0.78,
            "mean": 0.67,
            "source_context": 0.30,
            "source_raw": 0.08,
        }.items():
            controls[name] = {
                "row_count": 512,
                "normalized_mse": nmse,
                "raw_mse": 8.0,
                "cosine_mean": 0.86,
                "fve_nrm": 0.58,
                "centered_raw_r2": -0.2 if name == "teacher" else 0.0,
            }
        wins = {
            f"teacher_vs_{name}": {"teacher_better_fraction": 1.0}
            for name in ("teacher_shuffled", "blank", "generic", "mean", "source_context")
        }
        report = {
            "eval_splits": ["validation"],
            "splits": {
                "validation": {
                    "row_count": 512,
                    "controls": controls,
                    "rowwise_win_rates": wins,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            report_path = pathlib.Path(tmp) / "report.json"
            report_path.write_text(json.dumps(report))
            result = verifier.verify(
                {
                    "report_json": str(report_path),
                    "expected": {
                        "eval_splits": ["validation"],
                        "counts": {"validation": 512},
                        "required_controls": list(controls),
                        "teacher_must_beat": [
                            "teacher_shuffled",
                            "blank",
                            "generic",
                            "mean",
                            "source_context",
                        ],
                        "max_teacher_normalized_mse": 0.30,
                        "min_teacher_cosine": 0.85,
                        "min_teacher_fve_nrm": 0.55,
                        "min_teacher_rowwise_win_fraction": {},
                        "forbid_unrequested_rows": True,
                    },
                }
            )

        self.assertTrue(result["passed"])
        self.assertFalse(
            result["split_results"]["validation"]["raw_magnitude_claim_supported"]
        )


if __name__ == "__main__":
    unittest.main()
