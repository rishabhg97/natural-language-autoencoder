#!/usr/bin/env python3
"""Tensor helpers for Nano R33 functional activation-recovery evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class FunctionalRecoveryError(ValueError):
    """Raised when a functional-recovery input violates evaluator invariants."""


def _require_finite(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FunctionalRecoveryError(f"{name} must contain only finite values")


def rescale_direction(
    prediction: torch.Tensor,
    gold: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Give each predicted direction the corresponding gold activation norm."""

    if prediction.ndim != 2 or gold.ndim != 2:
        raise FunctionalRecoveryError("prediction and gold must both be rank-2 tensors")
    if prediction.shape != gold.shape:
        raise FunctionalRecoveryError("prediction and gold must have matching shapes")
    if epsilon <= 0.0:
        raise FunctionalRecoveryError("epsilon must be positive")

    prediction_float = prediction.to(dtype=torch.float32)
    gold_float = gold.to(device=prediction.device, dtype=torch.float32)
    _require_finite(prediction_float, name="prediction")
    _require_finite(gold_float, name="gold")

    prediction_norm = torch.linalg.vector_norm(prediction_float, dim=-1, keepdim=True)
    if torch.any(prediction_norm <= epsilon).item():
        raise FunctionalRecoveryError("prediction contains a zero-norm direction")
    gold_norm = torch.linalg.vector_norm(gold_float, dim=-1, keepdim=True)
    return prediction_float * (gold_norm / prediction_norm)


def make_boundary_replacement_hook(
    replacement: torch.Tensor,
    *,
    positions: torch.Tensor,
) -> Callable[[Any, Any, Any], Any]:
    """Build a forward hook that replaces one sequence position per batch row."""

    if replacement.ndim != 2:
        raise FunctionalRecoveryError("replacement must be a rank-2 tensor")
    if positions.ndim != 1:
        raise FunctionalRecoveryError("positions must be a rank-1 tensor")
    if replacement.shape[0] != positions.shape[0]:
        raise FunctionalRecoveryError("replacement and positions batch sizes must match")

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            if not output or not isinstance(output[0], torch.Tensor):
                raise FunctionalRecoveryError(
                    "tuple module output must contain a tensor in its first element"
                )
            hidden = output[0]
        elif isinstance(output, torch.Tensor):
            hidden = output
        else:
            raise FunctionalRecoveryError("module output must be a tensor or tuple")

        if hidden.ndim != 3:
            raise FunctionalRecoveryError("hidden state must be a rank-3 tensor")
        batch_size, sequence_length, hidden_size = hidden.shape
        if replacement.shape[0] != batch_size:
            raise FunctionalRecoveryError("replacement batch size must match hidden batch")
        if replacement.shape[1] != hidden_size:
            raise FunctionalRecoveryError("replacement hidden size must match hidden state")

        hook_positions = positions.to(device=hidden.device, dtype=torch.long)
        if torch.any(hook_positions < 0).item() or torch.any(
            hook_positions >= sequence_length
        ).item():
            raise FunctionalRecoveryError("replacement positions are out of bounds")
        hook_replacement = replacement.to(device=hidden.device, dtype=hidden.dtype)
        _require_finite(hook_replacement, name="replacement")

        patched_hidden = hidden.clone()
        batch_indices = torch.arange(batch_size, device=hidden.device)
        patched_hidden[batch_indices, hook_positions] = hook_replacement
        if is_tuple:
            return (patched_hidden, *output[1:])
        return patched_hidden

    return hook


def gather_position_logits(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Select one vocabulary-logit vector per batch row."""

    if logits.ndim != 3:
        raise FunctionalRecoveryError("logits must be a rank-3 tensor")
    if positions.ndim != 1:
        raise FunctionalRecoveryError("positions must be a rank-1 tensor")
    batch_size, sequence_length, _ = logits.shape
    if positions.shape[0] != batch_size:
        raise FunctionalRecoveryError("positions batch size must match logits batch")

    selected_positions = positions.to(device=logits.device, dtype=torch.long)
    if torch.any(selected_positions < 0).item() or torch.any(
        selected_positions >= sequence_length
    ).item():
        raise FunctionalRecoveryError("logit positions are out of bounds")
    batch_indices = torch.arange(batch_size, device=logits.device)
    return logits[batch_indices, selected_positions]
