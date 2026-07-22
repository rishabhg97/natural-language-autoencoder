"""Segmented MoE helper for patched Nemotron-H remote code."""

from __future__ import annotations

import os

import torch


def expert_scan_moe(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    experts,
) -> torch.Tensor:
    """Route tokens with the stock-style per-expert scan implementation."""

    original_shape = hidden_states.shape
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    routed_experts = topk_indices.reshape(hidden.shape[0], -1).to(torch.long)
    routed_weights = topk_weights.reshape(hidden.shape[0], -1)
    final_hidden_states = torch.zeros_like(hidden, dtype=topk_weights.dtype)

    for expert_idx, expert in enumerate(experts):
        token_indices, route_slots = torch.where(routed_experts == expert_idx)
        if token_indices.numel() == 0:
            final_hidden_states = final_hidden_states + expert(hidden[:1]).sum() * 0.0
            continue
        weights = routed_weights[token_indices, route_slots].unsqueeze(-1)
        expert_output = expert(hidden.index_select(0, token_indices))
        final_hidden_states.index_add_(
            0,
            token_indices,
            expert_output.to(final_hidden_states.dtype) * weights,
        )

    return final_hidden_states.reshape(original_shape).to(hidden_states.dtype)


def segmented_moe(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    experts,
) -> torch.Tensor:
    """Route top-k tokens through experts with one sort/segment pass.

    This preserves the stock per-expert arithmetic while avoiding the old
    ``torch.where(mask)`` scan for every expert.
    """

    routing_impl = os.environ.get("NLA_MOE_ROUTING_IMPL", "segmented").strip().lower()
    if routing_impl == "expert_scan":
        return expert_scan_moe(hidden_states, topk_indices, topk_weights, experts)
    if routing_impl != "segmented":
        raise ValueError(
            "NLA_MOE_ROUTING_IMPL must be 'segmented' or 'expert_scan', "
            f"got {routing_impl!r}"
        )

    original_shape = hidden_states.shape
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_experts = topk_indices.reshape(-1).to(torch.long)
    flat_weights = topk_weights.reshape(-1)
    flat_tokens = torch.arange(hidden.shape[0], device=hidden.device).repeat_interleave(topk_indices.shape[-1])

    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts.index_select(0, order)
    sorted_tokens = flat_tokens.index_select(0, order)
    sorted_weights = flat_weights.index_select(0, order)
    counts = torch.bincount(sorted_experts, minlength=len(experts))

    final_hidden_states = torch.zeros_like(hidden, dtype=topk_weights.dtype)
    start = 0
    for expert_idx, count_tensor in enumerate(counts.tolist()):
        count = int(count_tensor)
        if count == 0:
            # Keep unused expert parameters in the autograd graph, matching the
            # stock dummy-forward behavior without routing any real tokens.
            final_hidden_states = final_hidden_states + experts[expert_idx](hidden[:1]).sum() * 0.0
            continue
        end = start + count
        token_indices = sorted_tokens[start:end]
        weights = sorted_weights[start:end].unsqueeze(-1)
        expert_output = experts[expert_idx](hidden.index_select(0, token_indices))
        final_hidden_states.index_add_(0, token_indices, expert_output.to(final_hidden_states.dtype) * weights)
        start = end
    return final_hidden_states.reshape(original_shape).to(hidden_states.dtype)
