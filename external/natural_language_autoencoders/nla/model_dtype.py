"""Model dtype utilities shared by NLA training integrations."""

from __future__ import annotations

import torch


def normalize_floating_module_dtype(module: torch.nn.Module, dtype: torch.dtype) -> int:
    """Cast floating parameters and buffers in-place to ``dtype``."""

    changed = 0
    for parameter in module.parameters():
        if parameter.is_floating_point() and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)
            changed += 1
    for buffer in module.buffers():
        if buffer.is_floating_point() and buffer.dtype != dtype:
            buffer.data = buffer.data.to(dtype=dtype)
            changed += 1
    return changed
