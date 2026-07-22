"""Learning-rate policy helpers shared by patched training backends."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}
_CONSTANT_STYLES = {"", "constant", "none"}


def _env_flag(name: str, environ: Mapping[str, str]) -> bool:
    return environ.get(name, "").strip().lower() in _TRUE_VALUES


def _optimizer_param_groups(optimizer: Any) -> list[dict[str, Any]]:
    inner = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    groups = list(inner.param_groups)
    if not groups:
        raise ValueError("optimizer must have at least one parameter group")
    return groups


def apply_fsdp_live_lr_policy(
    lr_scheduler: Any,
    optimizer: Any,
    args: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply config bounds without discarding a restored scheduler position.

    FSDP checkpoints restore ``last_epoch`` and the scheduler's most recent LR,
    but optimizer-state loading can leave the live parameter-group LR at the
    configured maximum. Recompute the live LR after refreshing the requested
    bounds so the next optimizer step continues the saved schedule exactly.

    Constant LR is selected only by the configured schedule or the explicit
    ``NLA_FORCE_CONSTANT_LR`` compatibility override. Merely resuming from a
    checkpoint is not evidence that its scheduler is stale.
    """

    env = os.environ if environ is None else environ
    max_lr = float(args.lr)
    requested_style = str(getattr(args, "lr_decay_style", "constant") or "constant")
    force_constant = _env_flag("NLA_FORCE_CONSTANT_LR", env)
    if requested_style.lower() in _CONSTANT_STYLES or force_constant:
        style = "constant"
        min_lr = max_lr
    else:
        style = requested_style
        min_lr = float(getattr(args, "min_lr", 0.0) or 0.0)

    lr_scheduler.max_lr = max_lr
    lr_scheduler.min_lr = min_lr
    lr_scheduler.lr_decay_style = style

    groups = _optimizer_param_groups(optimizer)
    for group in groups:
        group["max_lr"] = max_lr
        group["min_lr"] = min_lr

    live_lrs = list(lr_scheduler.get_lr())
    if len(live_lrs) != len(groups):
        raise ValueError(
            "scheduler LR count does not match optimizer parameter groups: "
            f"{len(live_lrs)} != {len(groups)}"
        )
    for group, live_lr in zip(groups, live_lrs):
        group["lr"] = float(live_lr)

    resumed_optimizer = bool(getattr(args, "load", None)) and not bool(
        getattr(args, "finetune", False) or getattr(args, "no_load_optim", False)
    )
    return {
        "style": style,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "live_lrs": [float(lr) for lr in live_lrs],
        "last_epoch": int(getattr(lr_scheduler, "last_epoch", 0)),
        "resumed_optimizer": resumed_optimizer,
        "force_constant": force_constant,
    }
