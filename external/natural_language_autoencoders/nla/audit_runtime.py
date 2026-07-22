"""Runtime helpers for Nano NLA audit remediations.

The functions here are intentionally small and side-effect light so the Miles
patches and the NLA actor can share them without growing another monolith.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

try:  # Torch versions without DTensor should still import this module.
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover - depends on the cluster torch build.
    DTensor = ()  # type: ignore[assignment]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if DTensor != () and isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


@dataclass(frozen=True)
class MseRatioAgreement:
    """Per-sample MSE-ratio diagnostics for two nominally equivalent paths."""

    ratios: tuple[float, ...]
    mean_ratio: float
    max_abs_deviation: float
    p95_abs_deviation: float


def mse_ratio_agreement(
    numerator_mse: torch.Tensor,
    denominator_mse: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> MseRatioAgreement:
    """Summarize ``numerator_mse / denominator_mse`` with finite safeguards.

    Relative error is meaningful only when every denominator is positive.  Fail
    rather than silently clamping a zero MSE, since that would hide a malformed
    consistency check.
    """

    numerator = numerator_mse.detach().float().cpu().reshape(-1)
    denominator = denominator_mse.detach().float().cpu().reshape(-1)
    if numerator.numel() == 0:
        raise ValueError("MSE agreement requires at least one sample")
    if numerator.shape != denominator.shape:
        raise ValueError(
            "MSE agreement requires matching shapes, got "
            f"{tuple(numerator.shape)} and {tuple(denominator.shape)}"
        )
    if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all():
        raise ValueError("MSE agreement received a non-finite value")
    if (denominator <= eps).any():
        raise ValueError("MSE agreement denominator must be greater than eps")

    ratios = numerator / denominator
    if not torch.isfinite(ratios).all():
        raise ValueError("MSE agreement ratio is non-finite")
    deviations = (ratios - 1.0).abs()
    return MseRatioAgreement(
        ratios=tuple(float(value) for value in ratios.tolist()),
        mean_ratio=float(ratios.mean().item()),
        max_abs_deviation=float(deviations.max().item()),
        p95_abs_deviation=float(torch.quantile(deviations, 0.95).item()),
    )


def _ordered_metric_union(key_groups: Iterable[Iterable[str]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for keys in key_groups:
        for key in keys:
            if not isinstance(key, str):
                raise TypeError(f"metric key must be a string, got {type(key).__name__}")
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def aggregate_train_losses_by_key(
    losses_reduced: list[dict[str, list[str] | torch.Tensor]],
    parallel_state: Any,
) -> dict[str, float]:
    """Aggregate weighted microbatch metrics without assuming a fixed schema.

    Miles stores a normalizer at ``values[0]`` followed by weighted metric
    numerators. Observability metrics can be absent on individual microbatches
    or ranks, so align by key and track a denominator for each metric before
    reducing across the data/context-parallel group.
    """

    if not losses_reduced:
        return {}

    normalized: list[tuple[list[str], torch.Tensor]] = []
    for index, log_dict in enumerate(losses_reduced):
        keys = log_dict.get("keys")
        values = log_dict.get("values")
        if not isinstance(keys, list) or not isinstance(values, torch.Tensor):
            raise TypeError(f"microbatch {index} must contain list keys and tensor values")
        if values.ndim != 1 or values.numel() != len(keys) + 1:
            raise ValueError(
                f"microbatch {index} has {len(keys)} keys but {values.numel()} values"
            )
        if len(set(keys)) != len(keys):
            raise ValueError(f"microbatch {index} contains duplicate metric keys")
        normalized.append((keys, values))

    local_keys = _ordered_metric_union(keys for keys, _values in normalized)
    group = getattr(parallel_state, "dp_cp_group", None)
    world_size = 1
    gathered_key_groups: list[list[str]] = [local_keys]
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group=group)
        if world_size > 1:
            gathered: list[Any] = [None] * world_size
            dist.all_gather_object(gathered, local_keys, group=group)
            gathered_key_groups = gathered
    keys = _ordered_metric_union(gathered_key_groups)

    template = normalized[0][1]
    numerators = torch.zeros(len(keys), dtype=template.dtype, device=template.device)
    denominators = torch.zeros_like(numerators)
    key_indices = {key: index for index, key in enumerate(keys)}
    for microbatch_keys, values in normalized:
        normalizer = values[0]
        for local_index, key in enumerate(microbatch_keys):
            global_index = key_indices[key]
            numerators[global_index] += values[local_index + 1]
            denominators[global_index] += normalizer

    packed = torch.stack((numerators, denominators))
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)

    cp_size = float(getattr(parallel_state, "cp_size", 1))
    valid = packed[1] != 0
    averages = torch.zeros_like(packed[0])
    averages[valid] = packed[0, valid] * cp_size / packed[1, valid]
    valid_values = valid.detach().cpu().tolist()
    average_values = averages.detach().cpu().tolist()
    return {
        key: float(average_values[index])
        for index, key in enumerate(keys)
        if valid_values[index]
    }


def critic_last_token_indices(unconcat_tokens: list[torch.Tensor], device: torch.device | int | str) -> torch.Tensor:
    """Return flat packed-stream indices for each sample's last token."""

    last_idx = torch.empty(len(unconcat_tokens), dtype=torch.long, device=device)
    offset = 0
    for i, tokens in enumerate(unconcat_tokens):
        last_idx[i] = offset + int(tokens.shape[0]) - 1
        offset += int(tokens.shape[0])
    return last_idx


def padded_critic_inputs_from_tokens(
    unconcat_tokens: list[torch.Tensor],
    device: torch.device | int | str,
    *,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build right-padded critic input IDs, attention mask, and last-token indices."""

    if not unconcat_tokens:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty.reshape(0, 0), empty.reshape(0, 0), empty

    lengths = torch.tensor([int(tokens.shape[0]) for tokens in unconcat_tokens], dtype=torch.long)
    max_len = int(lengths.max().item())
    ids = torch.full((len(unconcat_tokens), max_len), pad_id, dtype=unconcat_tokens[0].dtype, device=device)
    mask = torch.zeros((len(unconcat_tokens), max_len), dtype=torch.long, device=device)
    for index, tokens in enumerate(unconcat_tokens):
        length = int(tokens.shape[0])
        ids[index, :length] = tokens.to(device)
        mask[index, :length] = 1
    last_idx = lengths.to(device) - 1
    return ids, mask, last_idx


def clip_grad_norm_local_shards(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    *,
    norm_type: float = 2.0,
    process_group: Any | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip gradients using local DTensor shards plus one scalar all-reduce.

    FSDP2 gradients are DTensors whose local shards partition the global tensor.
    L2 norm is decomposable, so computing local squared norms and reducing one
    scalar is equivalent to materializing every gradient's global tensor through
    ``torch.nn.utils.clip_grad_norm_``.
    """

    local_grads: list[torch.Tensor] = []
    norm_grads: list[torch.Tensor] = []
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        local = _as_local_tensor(grad.detach())
        if local.is_sparse:
            local = local.coalesce().values()
        local_grads.append(local)
        norm_grads.append(local.float())

    if not local_grads:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        return torch.zeros((), device=device)

    device = local_grads[0].device
    if norm_type == float("inf"):
        local_norm = torch.stack([g.abs().max() for g in norm_grads]).max()
        total_norm = local_norm.to(device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=process_group)
    else:
        if hasattr(torch, "_foreach_norm"):
            local_norms = torch._foreach_norm(norm_grads, norm_type)
        else:  # pragma: no cover - older torch fallback.
            local_norms = [torch.linalg.vector_norm(g, ord=norm_type) for g in norm_grads]
        local_power = torch.stack([n.pow(norm_type) for n in local_norms]).sum().to(device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_power, op=dist.ReduceOp.SUM, group=process_group)
        total_norm = local_power.pow(1.0 / norm_type)

    clip_coef = (float(max_norm) / (total_norm + eps)).clamp(max=1.0)
    if hasattr(torch, "_foreach_mul_"):
        torch._foreach_mul_(local_grads, clip_coef)
    else:  # pragma: no cover - older torch fallback.
        for grad in local_grads:
            grad.mul_(clip_coef)
    return total_norm.detach()


def should_synchronize_microbatch(model: torch.nn.Module, *, default_threshold_params: int = 1_000_000_000) -> bool:
    """Decide whether to keep the actor microbatch boundary sync.

    The old unconditional sync was needed for Gemma-scale tied embeddings. Nano's
    untied embedding is much smaller, so default to syncing only for very large
    embedding tables. ``NLA_SYNC_MICROBATCH`` can force either behavior.
    """

    if "NLA_SYNC_MICROBATCH" in os.environ:
        return _env_bool("NLA_SYNC_MICROBATCH")
    threshold = int(os.environ.get("NLA_SYNC_EMBED_PARAM_THRESHOLD", default_threshold_params))
    try:
        weight = model.get_input_embeddings().weight
    except Exception:
        return True
    return int(weight.numel()) >= threshold


def keep_router_fp32(model: torch.nn.Module) -> int:
    """Restore Nano/Nemotron router weights and correction biases to fp32 storage."""

    count = 0
    for module in model.modules():
        name = type(module).__name__.lower()
        looks_like_router = "router" in name or "gate" in name
        if not looks_like_router:
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.nn.Parameter) and weight.dtype != torch.float32:
            module.weight = torch.nn.Parameter(weight.detach().float(), requires_grad=weight.requires_grad)
            count += 1
        bias = getattr(module, "e_score_correction_bias", None)
        if isinstance(bias, torch.Tensor) and bias.dtype != torch.float32:
            module.e_score_correction_bias = bias.detach().float()
            count += 1
    return count
