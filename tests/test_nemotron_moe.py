import copy
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ModuleNotFoundError:  # Local lightweight test environments omit Torch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
sys.path.insert(0, str(NLA_ROOT))


@unittest.skipIf(torch is None, "Torch is required for numerical MoE equivalence")
class NemotronMoETests(unittest.TestCase):
    def test_expert_scan_matches_segmented_output_and_gradients(self):
        from nla.nemotron_moe import expert_scan_moe, segmented_moe

        torch.manual_seed(7)
        hidden_segmented = torch.randn(7, 5, dtype=torch.float32, requires_grad=True)
        hidden_scan = hidden_segmented.detach().clone().requires_grad_(True)
        indices = torch.tensor(
            [[0, 2], [1, 0], [2, 1], [0, 1], [2, 0], [1, 2], [0, 2]],
            dtype=torch.long,
        )
        weights_segmented = torch.softmax(torch.randn(7, 2), dim=-1).requires_grad_(True)
        weights_scan = weights_segmented.detach().clone().requires_grad_(True)
        experts_segmented = torch.nn.ModuleList([torch.nn.Linear(5, 5, bias=False) for _ in range(3)])
        experts_scan = copy.deepcopy(experts_segmented)

        out_segmented = segmented_moe(hidden_segmented, indices, weights_segmented, experts_segmented)
        out_scan = expert_scan_moe(hidden_scan, indices, weights_scan, experts_scan)
        out_segmented.square().sum().backward()
        out_scan.square().sum().backward()

        torch.testing.assert_close(out_scan, out_segmented)
        torch.testing.assert_close(hidden_scan.grad, hidden_segmented.grad)
        torch.testing.assert_close(weights_scan.grad, weights_segmented.grad)
        for scan_expert, segmented_expert in zip(experts_scan, experts_segmented, strict=True):
            torch.testing.assert_close(scan_expert.weight.grad, segmented_expert.weight.grad)

    def test_dispatch_uses_expert_scan_from_environment(self):
        from nla import nemotron_moe

        hidden = torch.randn(3, 4)
        indices = torch.tensor([[0], [1], [0]], dtype=torch.long)
        weights = torch.ones(3, 1)
        experts = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(2)])

        with mock.patch.dict(os.environ, {"NLA_MOE_ROUTING_IMPL": "expert_scan"}):
            expected = nemotron_moe.expert_scan_moe(hidden, indices, weights, experts)
            actual = nemotron_moe.segmented_moe(hidden, indices, weights, experts)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
