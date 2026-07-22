import importlib.util
import pathlib
import unittest

import pytest

torch = pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NanoR33FunctionalCoreTests(unittest.TestCase):
    def test_rescale_direction_matches_gold_norm(self):
        module = load_script("nano_r33_functional_core")
        prediction = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        gold = torch.tensor([[0.0, 10.0], [0.0, 2.0]])

        scaled = module.rescale_direction(prediction, gold)

        self.assertTrue(torch.allclose(scaled.norm(dim=-1), gold.norm(dim=-1)))
        self.assertEqual(scaled.dtype, torch.float32)

    def test_rescale_direction_rejects_zero_or_nonfinite_prediction(self):
        module = load_script("nano_r33_functional_core")
        gold = torch.tensor([[0.0, 1.0]])

        with self.assertRaisesRegex(module.FunctionalRecoveryError, "zero-norm"):
            module.rescale_direction(torch.zeros_like(gold), gold)
        with self.assertRaisesRegex(module.FunctionalRecoveryError, "finite"):
            module.rescale_direction(torch.tensor([[float("nan"), 0.0]]), gold)

    def test_hook_replaces_only_requested_positions_in_tuple_output(self):
        module = load_script("nano_r33_functional_core")
        hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        original = hidden.clone()
        replacement = torch.tensor([[100.0] * 4, [200.0] * 4])
        hook = module.make_boundary_replacement_hook(
            replacement,
            positions=torch.tensor([2, 1]),
        )

        output = hook(None, None, (hidden, "cache"))

        self.assertTrue(torch.equal(output[0][0, :2], original[0, :2]))
        self.assertTrue(torch.equal(output[0][0, 2], replacement[0]))
        self.assertTrue(torch.equal(output[0][1, 1], replacement[1]))
        self.assertTrue(torch.equal(output[0][1, 0], original[1, 0]))
        self.assertTrue(torch.equal(output[0][1, 2], original[1, 2]))
        self.assertEqual(output[1], "cache")
        self.assertTrue(torch.equal(hidden, original))

    def test_hook_supports_tensor_output_and_rejects_shape_mismatch(self):
        module = load_script("nano_r33_functional_core")
        hidden = torch.zeros((1, 2, 3))
        hook = module.make_boundary_replacement_hook(
            torch.ones((1, 3)),
            positions=torch.tensor([1]),
        )
        self.assertTrue(torch.equal(hook(None, None, hidden)[0, 1], torch.ones(3)))

        bad = module.make_boundary_replacement_hook(
            torch.ones((2, 3)),
            positions=torch.tensor([1, 1]),
        )
        with self.assertRaisesRegex(module.FunctionalRecoveryError, "batch"):
            bad(None, None, hidden)

    def test_gather_position_logits(self):
        module = load_script("nano_r33_functional_core")
        logits = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)

        selected = module.gather_position_logits(logits, torch.tensor([2, 1]))

        self.assertTrue(torch.equal(selected[0], logits[0, 2]))
        self.assertTrue(torch.equal(selected[1], logits[1, 1]))


if __name__ == "__main__":
    unittest.main()
