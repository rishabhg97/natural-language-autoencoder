import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    miles = types.ModuleType("miles")
    miles_utils = types.ModuleType("miles.utils")
    miles_misc = types.ModuleType("miles.utils.misc")

    def periodic(rollout_id, interval, num_rollout_per_epoch=None, num_rollout=None):
        if interval is None:
            return False
        if num_rollout is not None and rollout_id == num_rollout - 1:
            return True
        step = rollout_id + 1
        return step % interval == 0 or (
            num_rollout_per_epoch is not None
            and step % num_rollout_per_epoch == 0
        )

    miles_misc.should_run_periodic_action = periodic
    modules = {
        "miles": miles,
        "miles.utils": miles_utils,
        "miles.utils.misc": miles_misc,
    }
    path = (
        ROOT
        / "external"
        / "natural_language_autoencoders"
        / "nla"
        / "save_schedule.py"
    )
    spec = importlib.util.spec_from_file_location("nla_save_schedule_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class SaveScheduleTest(unittest.TestCase):
    def test_explicit_updates_override_fixed_interval(self):
        module = load_module()
        with mock.patch.dict(os.environ, {"NLA_SAVE_ITERATIONS": "2,5"}):
            observed = [
                module.should_save_rollout(index, 1, num_rollout=5)
                for index in range(5)
            ]
        self.assertEqual(observed, [False, True, False, False, True])

    def test_falls_back_to_miles_periodic_schedule(self):
        module = load_module()
        with mock.patch.dict(os.environ, {}, clear=True):
            observed = [
                module.should_save_rollout(index, 2, num_rollout=5)
                for index in range(5)
            ]
        self.assertEqual(observed, [False, True, False, True, True])

    def test_explicit_schedule_requires_final_update(self):
        module = load_module()
        with mock.patch.dict(os.environ, {"NLA_SAVE_ITERATIONS": "2,4"}):
            with self.assertRaisesRegex(ValueError, "final configured rollout"):
                module.should_save_rollout(0, None, num_rollout=5)

    def test_parser_rejects_unsorted_or_duplicate_updates(self):
        module = load_module()
        for raw in ("4,2", "2,2"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    module.parse_save_iterations(raw)


if __name__ == "__main__":
    unittest.main()
