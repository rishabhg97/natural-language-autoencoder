import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "nano_ar_r33_scaling_pipeline.sh"


class NanoARR33ScalingPipelineTests(unittest.TestCase):
    def test_default_slice_matches_reused_teacher_contract(self):
        text = SCRIPT.read_text()

        self.assertIn('EXPECTED_ROWS="${EXPECTED_ROWS:-99570}"', text)
        self.assertIn('CORPUS_START="${CORPUS_START:-500}"', text)
        self.assertIn('CORPUS_LENGTH="${CORPUS_LENGTH:-10000}"', text)

    def test_teacher_overlap_preflight_runs_before_expensive_extraction(self):
        text = SCRIPT.read_text()

        self.assertIn("preflight_teacher_overlap", text)
        self.assertIn("teacher_overlap_rows", text)
        self.assertIn("teacher_overlap_rows < expected_rows", text)

        preflight = text.index("preflight_teacher_overlap")
        extraction = text.index("scripts/nano_ar_layer_sweep.py extract")

        self.assertLess(preflight, extraction)


if __name__ == "__main__":
    unittest.main()
