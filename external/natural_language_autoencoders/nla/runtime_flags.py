"""Runtime flag helpers shared by NLA training/runtime entrypoints."""

from __future__ import annotations

import math
import os


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean environment flag with explicit true/false parsing."""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {value!r}"
    )


def env_float(name: str, default: float) -> float:
    """Return a finite floating-point environment setting or its default."""

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite float, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite float, got {value!r}")
    return parsed
