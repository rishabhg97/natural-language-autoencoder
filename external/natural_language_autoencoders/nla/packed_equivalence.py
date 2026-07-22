"""Utilities for checking packed actor forwards against padded references."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PackedPaddedInputs:
    tokens: tuple[torch.Tensor, ...]
    packed_input_ids: torch.Tensor
    packed_position_ids: torch.Tensor
    padded_input_ids: torch.Tensor
    padded_position_ids: torch.Tensor
    padded_attention_mask: torch.Tensor


def build_bshd_attention_mask(
    token_rows: list[torch.Tensor],
    padded_input_ids: torch.Tensor,
) -> torch.Tensor:
    """Build a length-derived mask for Miles' padded FSDP batch format."""

    if padded_input_ids.ndim != 2:
        raise ValueError(
            f"bshd input_ids must be rank 2, got {tuple(padded_input_ids.shape)}"
        )
    if len(token_rows) != padded_input_ids.shape[0]:
        raise ValueError(
            "bshd token-row count must match padded batch size: "
            f"rows={len(token_rows)} batch={padded_input_ids.shape[0]}"
        )
    lengths = torch.tensor(
        [int(row.numel()) for row in token_rows],
        dtype=torch.long,
        device=padded_input_ids.device,
    )
    width = int(padded_input_ids.shape[1])
    if lengths.numel() == 0 or int(lengths.min().item()) <= 0:
        raise ValueError("bshd batches require non-empty token rows")
    if int(lengths.max().item()) > width:
        raise ValueError(
            f"bshd row length exceeds padded width: max={int(lengths.max().item())} "
            f"width={width}"
        )
    positions = torch.arange(width, device=padded_input_ids.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def build_bshd_max_seq_lens(
    token_rows: list[torch.Tensor],
    *,
    pad_size: int,
) -> list[int]:
    """Return Miles-compatible padded widths for a rewritten token batch."""

    if pad_size <= 0:
        raise ValueError(f"bshd pad_size must be positive, got {pad_size}")
    if not token_rows:
        raise ValueError("bshd batches require non-empty token rows")
    lengths = [int(row.numel()) for row in token_rows]
    if any(length <= 0 for length in lengths):
        raise ValueError("bshd batches require non-empty token rows")
    width = ((max(lengths) + pad_size - 1) // pad_size) * pad_size
    return [width] * len(token_rows)


def build_packed_padded_inputs(
    token_rows: list[torch.Tensor],
    *,
    sample_limit: int = 2,
    pad_token_id: int = 0,
) -> PackedPaddedInputs:
    if sample_limit < 2:
        raise ValueError("packed equivalence requires at least two samples")
    selected = tuple(row.flatten() for row in token_rows[:sample_limit])
    if len(selected) < 2 or any(row.numel() < 2 for row in selected):
        raise ValueError("packed equivalence requires two token rows of length >= 2")

    device = selected[0].device
    if any(row.device != device for row in selected):
        raise ValueError("all token rows must be on the same device")
    max_length = max(int(row.numel()) for row in selected)
    padded_ids = torch.full(
        (len(selected), max_length),
        pad_token_id,
        dtype=selected[0].dtype,
        device=device,
    )
    padded_positions = torch.zeros_like(padded_ids)
    padded_mask = torch.zeros_like(padded_ids)
    packed_positions: list[torch.Tensor] = []
    for index, row in enumerate(selected):
        length = int(row.numel())
        positions = torch.arange(length, dtype=torch.long, device=device)
        padded_ids[index, :length] = row
        padded_positions[index, :length] = positions
        padded_mask[index, :length] = 1
        packed_positions.append(positions)

    return PackedPaddedInputs(
        tokens=selected,
        packed_input_ids=torch.cat(selected).unsqueeze(0),
        packed_position_ids=torch.cat(packed_positions).unsqueeze(0),
        padded_input_ids=padded_ids,
        padded_position_ids=padded_positions,
        padded_attention_mask=padded_mask,
    )


def response_mean_nlls(
    logits: torch.Tensor,
    token_rows: tuple[torch.Tensor, ...],
    response_lengths: list[int],
    *,
    packed: bool,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"expected logits [batch, sequence, vocab], got {logits.shape}")
    if len(token_rows) != len(response_lengths):
        raise ValueError("token and response-length counts must match")

    sample_nlls: list[torch.Tensor] = []
    packed_offset = 0
    for index, (tokens, raw_response_length) in enumerate(
        zip(token_rows, response_lengths, strict=True)
    ):
        total_length = int(tokens.numel())
        response_length = int(raw_response_length)
        prompt_length = total_length - response_length
        if response_length <= 0 or prompt_length <= 0:
            raise ValueError("each sample needs non-empty prompt and response tokens")
        positions = torch.arange(
            prompt_length - 1,
            total_length - 1,
            dtype=torch.long,
            device=logits.device,
        )
        if packed:
            selected_logits = logits[0, positions + packed_offset]
        else:
            selected_logits = logits[index, positions]
        targets = tokens[prompt_length:].to(logits.device)
        sample_nlls.append(
            F.cross_entropy(selected_logits.float(), targets, reduction="mean")
        )
        packed_offset += total_length
    return torch.stack(sample_nlls)


def packed_equivalence_metrics(
    packed_nlls: torch.Tensor,
    padded_nlls: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float | bool]:
    if packed_nlls.shape != padded_nlls.shape or packed_nlls.numel() < 2:
        raise ValueError("packed and padded NLLs must have the same multi-sample shape")
    absolute = (packed_nlls.float() - padded_nlls.float()).abs()
    relative = absolute / padded_nlls.float().abs().clamp_min(1e-8)
    passed = torch.all(absolute <= atol + rtol * padded_nlls.float().abs())
    return {
        "passed": bool(passed.item()),
        "sample_count": float(packed_nlls.numel()),
        "packed_mean_nll": float(packed_nlls.float().mean().item()),
        "padded_mean_nll": float(padded_nlls.float().mean().item()),
        "max_abs_diff": float(absolute.max().item()),
        "max_rel_diff": float(relative.max().item()),
    }
