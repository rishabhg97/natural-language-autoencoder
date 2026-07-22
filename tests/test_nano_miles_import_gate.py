import importlib.util
import pathlib
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoMilesImportGateTests(unittest.TestCase):
    def test_check_imports_reports_required_symbols(self):
        gate = load_script("check_miles_nla_imports")

        modules = {
            "miles": types.SimpleNamespace(__file__="/fake/miles/__init__.py"),
            "train": types.SimpleNamespace(train=lambda *args: None),
            "miles.ray.rollout": types.SimpleNamespace(RolloutManager=object),
            "nla.train_actor": types.SimpleNamespace(NLAFSDPActor=object),
            "nla.rollout.sft_actor": types.SimpleNamespace(generate_rollout=lambda *args: None),
            "nla.injection": types.SimpleNamespace(inject_at_marked_positions=lambda *args: None),
        }

        report = gate.check_imports(import_module=modules.__getitem__)

        self.assertTrue(report["ok"])
        self.assertEqual(
            [check["target"] for check in report["checks"]],
            [
                "miles",
                "train.train",
                "miles.ray.rollout.RolloutManager",
                "nla.train_actor.NLAFSDPActor",
                "nla.rollout.sft_actor.generate_rollout",
                "nla.injection.inject_at_marked_positions",
            ],
        )

    def test_check_imports_reports_missing_module_without_raising(self):
        gate = load_script("check_miles_nla_imports")

        def missing_import(name):
            raise ModuleNotFoundError(name)

        report = gate.check_imports(import_module=missing_import)

        self.assertFalse(report["ok"])
        self.assertEqual(report["checks"][0]["status"], "missing")


if __name__ == "__main__":
    unittest.main()
