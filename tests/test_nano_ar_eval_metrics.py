import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoAREvalMetricTests(unittest.TestCase):
    def test_metric_summary_reports_fve_against_mean_control(self):
        evaluator = load_script("eval_nano_ar_miles_checkpoint")
        targets = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        train_targets = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        summary = evaluator.metric_summary(
            predictions=targets.copy(),
            targets=targets,
            train_targets_for_mean=train_targets,
        )

        self.assertAlmostEqual(summary["normalized_mse"], 0.0)
        self.assertAlmostEqual(summary["cosine_mean"], 1.0)
        self.assertGreater(summary["mean_control_normalized_mse"], 0.0)
        self.assertAlmostEqual(summary["fve_nrm"], 1.0)

    def test_deranged_indices_do_not_leave_rows_in_place_when_possible(self):
        evaluator = load_script("eval_nano_ar_miles_checkpoint")

        indices = evaluator.deranged_indices(8, seed=3)

        self.assertEqual(sorted(indices), list(range(8)))
        self.assertTrue(all(idx != source_idx for idx, source_idx in enumerate(indices)))

    def test_direct_device_map_resolves_single_gpu_config_values(self):
        evaluator = load_script("eval_nano_ar_miles_checkpoint")

        self.assertEqual(evaluator._direct_device_map("single_gpu"), {"": 0})
        self.assertEqual(evaluator._direct_device_map("cuda:3"), {"": 3})
        self.assertEqual(evaluator._direct_device_map("auto"), "auto")
        with self.assertRaisesRegex(ValueError, "cuda:<non-negative-integer>"):
            evaluator._direct_device_map("cuda:bad")

    def test_remote_code_patch_uses_checkpoint_directory_api(self):
        evaluator = load_script("eval_nano_ar_miles_checkpoint")
        checkpoint_dir = pathlib.Path("/tmp/nano-ar-checkpoint")
        calls = []
        nla = types.ModuleType("nla")
        patches = types.ModuleType("nla.remote_code_patches")
        patches.prepare_nemotron_h_checkpoint_for_load = calls.append

        original_modules = {
            name: sys.modules.get(name)
            for name in ("nla", "nla.remote_code_patches")
        }
        try:
            sys.modules["nla"] = nla
            sys.modules["nla.remote_code_patches"] = patches
            evaluator._patch_remote_code_for_eval(checkpoint_dir)
        finally:
            for name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(calls, [checkpoint_dir])

    def test_rowwise_win_summary_reports_teacher_control_fraction(self):
        evaluator = load_script("eval_nano_ar_miles_checkpoint")
        teacher_losses = np.array([0.1, 0.2, 0.5, 0.7], dtype=np.float32)
        control_losses = np.array([0.3, 0.1, 0.5, 0.9], dtype=np.float32)

        summary = evaluator.rowwise_win_summary(teacher_losses, control_losses)

        self.assertEqual(summary["row_count"], 4)
        self.assertAlmostEqual(summary["teacher_better_fraction"], 0.5)
        self.assertAlmostEqual(summary["tie_fraction"], 0.25)
        self.assertAlmostEqual(summary["control_better_fraction"], 0.25)
        self.assertAlmostEqual(summary["mean_loss_delta_control_minus_teacher"], 0.075)


if __name__ == "__main__":
    unittest.main()
