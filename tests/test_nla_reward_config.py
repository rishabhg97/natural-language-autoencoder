from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
import unittest

import pytest

pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"


def _install_runtime_stubs() -> dict[str, types.ModuleType | None]:
    names = [
        "ray",
        "miles",
        "miles.utils",
        "miles.utils.processing_utils",
        "miles.utils.types",
    ]
    old_modules = {name: sys.modules.get(name) for name in names}

    ray = types.ModuleType("ray")
    miles = types.ModuleType("miles")
    miles_utils = types.ModuleType("miles.utils")
    processing_utils = types.ModuleType("miles.utils.processing_utils")
    processing_utils.load_tokenizer = lambda *_args, **_kwargs: None
    miles_types = types.ModuleType("miles.utils.types")
    miles_types.Sample = object

    sys.modules.update(
        {
            "ray": ray,
            "miles": miles,
            "miles.utils": miles_utils,
            "miles.utils.processing_utils": processing_utils,
            "miles.utils.types": miles_types,
        }
    )
    return old_modules


def _restore_modules(old_modules: dict[str, types.ModuleType | None]) -> None:
    for name, module in old_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def load_reward_module(env: dict[str, str] | None = None):
    path = NLA_ROOT / "nla" / "reward.py"
    spec = importlib.util.spec_from_file_location("nla_reward_under_test", path)
    module = importlib.util.module_from_spec(spec)
    old_env = os.environ.copy()
    old_modules = _install_runtime_stubs()
    sys.path.insert(0, str(NLA_ROOT))
    try:
        os.environ.clear()
        os.environ.update(old_env)
        if env:
            os.environ.update(env)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        sys.path.pop(0)
        _restore_modules(old_modules)


class NLARewardConfigTests(unittest.TestCase):
    def test_default_failure_penalty_stays_minus_two(self):
        module = load_reward_module()

        self.assertEqual(module.failed_extraction_reward(), -2.0)
        self.assertEqual(module.FAILED_EXTRACTION_REWARD, -2.0)

    def test_env_overrides_failure_penalty(self):
        module = load_reward_module({"NLA_FAILED_EXTRACTION_REWARD": "-2.5"})

        self.assertEqual(module.failed_extraction_reward(), -2.5)
        self.assertEqual(module.FAILED_EXTRACTION_REWARD, -2.5)

    def test_bad_env_value_raises(self):
        with self.assertRaises(ValueError):
            load_reward_module({"NLA_FAILED_EXTRACTION_REWARD": "not-a-number"})

    def test_unsafe_softened_penalty_raises(self):
        with self.assertRaises(ValueError):
            load_reward_module({"NLA_FAILED_EXTRACTION_REWARD": "-1.0"})


if __name__ == "__main__":
    unittest.main()
