from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NanoMilesEvalLoggingTests(unittest.TestCase):
    def test_checkpoint_eval_logs_heldout_wandb_scalars(self):
        text = (ROOT / "scripts" / "eval_nano_av_miles_checkpoint.py").read_text()

        required = [
            "add_wandb_args(parser)",
            "init_wandb(",
            "build_wandb_eval_metrics",
            "build_checkpoint_eval_control_vectors",
            "mean_vector = vectors[mean_source].mean(dim=0)",
            "eval/validation/real_nll",
            "eval/test/real_nll",
            "eval/validation/gap_vs_shuffled",
            "eval/test/gap_vs_none",
        ]
        for needle in required:
            with self.subTest(needle=needle):
                self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
