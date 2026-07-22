import sys
import types

import pytest
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from external.natural_language_autoencoders.nla.mamba_runtime import (
    parse_mamba_train_kernel_mode,
    resolve_mamba_train_kernel_mode,
    temporarily_select_mamba_training_kernel,
    torch_causal_conv1d,
)


def test_torch_causal_conv1d_matches_grouped_conv_and_gradients():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(3, 5, 11, generator=generator, requires_grad=True)
    weight = torch.randn(5, 4, generator=generator, requires_grad=True)
    bias = torch.randn(5, generator=generator, requires_grad=True)

    actual = torch_causal_conv1d(x, weight, bias, activation="silu")
    expected = F.silu(
        F.conv1d(x, weight.unsqueeze(1), bias, padding=3, groups=5)[..., :11]
    )
    torch.testing.assert_close(actual, expected)

    actual.sum().backward(retain_graph=True)
    actual_gradients = (x.grad.clone(), weight.grad.clone(), bias.grad.clone())
    x.grad = None
    weight.grad = None
    bias.grad = None
    expected.sum().backward()
    expected_gradients = (x.grad, weight.grad, bias.grad)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_unfused_torch_conv_mode_restores_remote_module_state():
    module_name = "test_remote_nemotron_h"
    remote_module = types.ModuleType(module_name)
    original_conv = object()
    remote_module.is_fast_path_available = True
    remote_module.causal_conv1d_fn = original_conv

    class FakeMixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = torch.nn.Conv1d(2, 2, 3, groups=2)
            self.in_proj = torch.nn.Linear(2, 2)

        def cuda_kernels_forward(self):
            raise NotImplementedError

        def torch_forward(self):
            raise NotImplementedError

    FakeMixer.__module__ = module_name
    remote_module.FakeMixer = FakeMixer
    sys.modules[module_name] = remote_module
    try:
        mixer = FakeMixer()
        model = torch.nn.Sequential(mixer)
        assert mixer.training

        with temporarily_select_mamba_training_kernel(
            model, "unfused_torch_conv"
        ):
            assert not mixer.training
            assert remote_module.is_fast_path_available
            assert remote_module.causal_conv1d_fn is torch_causal_conv1d

        assert mixer.training
        assert remote_module.is_fast_path_available
        assert remote_module.causal_conv1d_fn is original_conv
    finally:
        sys.modules.pop(module_name, None)


def test_mamba_train_kernel_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="Mamba training kernel mode"):
        parse_mamba_train_kernel_mode("mystery")


def test_role_kernel_mode_overrides_shared_default():
    env = {
        "NLA_TRAIN_MAMBA_KERNEL_MODE": "unfused_torch_conv",
        "NLA_CRITIC_TRAIN_MAMBA_KERNEL_MODE": "torch",
    }

    assert resolve_mamba_train_kernel_mode("actor", env) == "unfused_torch_conv"
    assert resolve_mamba_train_kernel_mode("critic", env) == "torch"
