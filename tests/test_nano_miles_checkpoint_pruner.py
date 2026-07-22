from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class NanoMilesCheckpointPrunerTest(unittest.TestCase):
    def test_checkpoint_pruner_script_exists(self):
        script = ROOT / "scripts" / "prune_nano_miles_checkpoints.py"

        self.assertTrue(script.exists(), "checkpoint pruner script should exist")

    def test_keep_latest_completed_checkpoint_and_leave_in_progress_newer_checkpoint(self):
        pruner = load_script("prune_nano_miles_checkpoints")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "latest_checkpointed_iteration.txt").write_text("200")
            for iteration in (100, 200, 300):
                path = root / f"iter_{iteration:07d}"
                path.mkdir()
                (path / "marker.txt").write_text(str(iteration))
            rollout = root / "rollout"
            rollout.mkdir()
            for rollout_id in (99, 199, 299):
                (rollout / f"global_dataset_state_dict_{rollout_id}.pt").write_text(str(rollout_id))

            plan = pruner.run_once(root, keep_full=1, dry_run=False)

            self.assertEqual(plan.latest_iteration, 200)
            self.assertEqual(plan.keep_iterations, (200,))
            self.assertFalse((root / "iter_0000100").exists())
            self.assertTrue((root / "iter_0000200").exists())
            self.assertTrue((root / "iter_0000300").exists(), "newer-than-tracker checkpoint may be in-progress")
            self.assertFalse((rollout / "global_dataset_state_dict_99.pt").exists())
            self.assertTrue((rollout / "global_dataset_state_dict_199.pt").exists())
            self.assertTrue((rollout / "global_dataset_state_dict_299.pt").exists())


if __name__ == "__main__":
    unittest.main()
