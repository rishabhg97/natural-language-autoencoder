"""Runtime-selectable Mamba training kernels for stable Nano experiments."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator, Mapping
from typing import Any

import torch
import torch.nn.functional as F


MAMBA_TRAIN_KERNEL_MODE_ENV = "NLA_TRAIN_MAMBA_KERNEL_MODE"
MAMBA_TRAIN_KERNEL_MODES = frozenset({"auto", "torch", "unfused_torch_conv"})


def parse_mamba_train_kernel_mode(value: str | None) -> str:
    mode = (value or "auto").strip().lower()
    if mode not in MAMBA_TRAIN_KERNEL_MODES:
        allowed = ", ".join(sorted(MAMBA_TRAIN_KERNEL_MODES))
        raise ValueError(f"Mamba training kernel mode must be one of {allowed}, got {value!r}")
    return mode


def resolve_mamba_train_kernel_mode(
    role: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve a role override before falling back to the shared kernel mode."""

    env = os.environ if environ is None else environ
    role_key = f"NLA_{str(role).strip().upper()}_TRAIN_MAMBA_KERNEL_MODE"
    return parse_mamba_train_kernel_mode(
        env.get(role_key, env.get(MAMBA_TRAIN_KERNEL_MODE_ENV))
    )


def torch_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Exact PyTorch equivalent of causal_conv1d_fn for full-sequence training."""

    if x.ndim != 3 or weight.ndim != 2 or x.shape[1] != weight.shape[0]:
        raise ValueError(
            "causal conv expects x=[batch, channels, sequence] and "
            "weight=[channels, width]"
        )
    sequence_length = x.shape[-1]
    output = F.conv1d(
        x,
        weight.unsqueeze(1),
        bias,
        padding=weight.shape[-1] - 1,
        groups=x.shape[1],
    )[..., :sequence_length]
    if activation is None:
        return output
    if activation.lower() not in {"silu", "swish"}:
        raise ValueError(f"unsupported causal conv activation: {activation!r}")
    return F.silu(output)


def _remote_mamba_state(model: torch.nn.Module) -> tuple[list[Any], list[torch.nn.Module]]:
    remote_modules: list[Any] = []
    mixers: list[torch.nn.Module] = []
    seen: set[int] = set()
    for module in model.modules():
        remote_module = sys.modules.get(module.__class__.__module__)
        if remote_module is None or not hasattr(remote_module, "is_fast_path_available"):
            continue
        if id(remote_module) not in seen:
            seen.add(id(remote_module))
            remote_modules.append(remote_module)
        if all(
            hasattr(module, attribute)
            for attribute in ("cuda_kernels_forward", "torch_forward", "conv1d", "in_proj")
        ):
            mixers.append(module)
    return remote_modules, mixers


@contextlib.contextmanager
def temporarily_disable_mamba_fast_path(model: torch.nn.Module) -> Iterator[None]:
    remote_modules, _mixers = _remote_mamba_state(model)
    previous = [module.is_fast_path_available for module in remote_modules]
    for module in remote_modules:
        module.is_fast_path_available = False
    try:
        yield
    finally:
        for module, old_value in zip(remote_modules, previous, strict=True):
            module.is_fast_path_available = old_value


@contextlib.contextmanager
def temporarily_select_mamba_training_kernel(
    model: torch.nn.Module,
    mode: str,
) -> Iterator[None]:
    """Select a Mamba training path for both forward and checkpoint recomputation."""

    mode = parse_mamba_train_kernel_mode(mode)
    if mode == "auto":
        yield
        return
    if mode == "torch":
        with temporarily_disable_mamba_fast_path(model):
            yield
        return

    remote_modules, mixers = _remote_mamba_state(model)
    previous_functions: list[Any] = []
    patched_modules: list[Any] = []
    for remote_module in remote_modules:
        if not hasattr(remote_module, "causal_conv1d_fn"):
            continue
        previous_functions.append(remote_module.causal_conv1d_fn)
        patched_modules.append(remote_module)
        remote_module.causal_conv1d_fn = torch_causal_conv1d
    previous_training = [module.training for module in mixers]
    for module in mixers:
        # Nemotron-H uses this flag only to select its fused training kernel.
        module.training = False
    try:
        yield
    finally:
        for module, old_training in zip(mixers, previous_training, strict=True):
            module.training = old_training
        for remote_module, old_function in zip(
            patched_modules, previous_functions, strict=True
        ):
            remote_module.causal_conv1d_fn = old_function
