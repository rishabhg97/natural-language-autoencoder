import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoCheckpointEvalSplitIsolationTests(unittest.TestCase):
    def test_selects_only_requested_split_payloads(self):
        evaluator = load_script("nano_eval_core")

        selected = evaluator.select_requested_eval_splits(
            ["validation"],
            validation={"path": "validation"},
            test={"path": "test"},
        )

        self.assertEqual(selected, {"validation": {"path": "validation"}})

    def test_zero_limit_on_unrequested_test_is_never_sampled(self):
        evaluator = load_script("nano_eval_core")

        selected = evaluator.select_requested_eval_splits(
            ["validation"],
            validation=[1, 2],
            test=[4, 5, 6],
        )

        self.assertEqual(selected, {"validation": [1, 2]})


if __name__ == "__main__":
    unittest.main()
