"""Pure injection-hook logic — extracted for testability.

The most correctness-critical path in NLA: if injection fails or hits the wrong
position, the model sees the literal ㊗ character and outputs Chinese. This
function is the one place that must be right, so it's pure and unit-testable.
"""

import torch


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
    seq_slice: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Overwrite embedding rows at injection-marker positions with activation vectors.

    input_ids: [B, S] — or [1, T_packed] for thd layout. The FULL token stream
        (broadcast across TP ranks — identical everywhere).
    embeddings: [B, S, d] (unsharded) or [B, S_local, d] (seq_slice set). The
        embedding layer output. Cloned; original unchanged.
    vectors: [N, d] — activation vectors in microbatch order. N = number of
        injection sites expected GLOBALLY. Must equal the count of valid matches
        found in the FULL input_ids (regardless of seq_slice).
    inj_id, left_id, right_id: the injection token + its canonical neighbors.
    seq_slice: (start, end) if embeddings holds only positions [start:end) of
        the sequence dim. For Megatron with --sequence-parallel: each TP rank's
        LanguageModelEmbedding output covers [tp_rank * S/TP : (tp_rank+1) * S/TP).
        The scan still runs over FULL input_ids (count + vec_idx are global),
        writes skip positions outside the slice.

    A match is valid iff input_ids[b, p] == inj_id AND input_ids[b, p-1] == left_id
    AND input_ids[b, p+1] == right_id. The neighbor check rejects false positives
    from ㊗ appearing in response text (user pasted it, multi-turn context).

    Raises:
        AssertionError if GLOBAL count of valid matches != vectors.shape[0] —
        means prompt template drift, tokenizer version mismatch, or data corruption.
        Fires identically on every TP rank (scan is over full input_ids).
    """
    seq_len = input_ids.shape[-1]
    if seq_slice is None:
        start, end = 0, seq_len
        assert input_ids.shape == embeddings.shape[:-1], (
            f"input_ids {tuple(input_ids.shape)} and embeddings "
            f"{tuple(embeddings.shape[:-1])} batch dims must match"
        )
    else:
        start, end = seq_slice
        assert input_ids.shape[0] == embeddings.shape[0], (
            f"batch dim mismatch: input_ids {input_ids.shape[0]}, "
            f"embeddings {embeddings.shape[0]}"
        )
        assert embeddings.shape[1] == end - start, (
            f"seq_slice={seq_slice} spans {end - start} positions but "
            f"embeddings seq dim is {embeddings.shape[1]}. SP shard layout "
            f"mismatch — check tp_rank/tp_size computation."
        )
    assert vectors.ndim == 2 and vectors.shape[1] == embeddings.shape[-1], (
        f"vectors must be [N, d_model], got {tuple(vectors.shape)}, "
        f"d_model={embeddings.shape[-1]}"
    )
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    matches = (input_ids == inj_id).nonzero()  # [M, 2] — (batch_idx, seq_idx), row-major sorted
    if matches.numel() == 0:
        valid_matches = matches
    else:
        batch_positions = matches[:, 0]
        valid_positions = matches[:, 1]
        interior = (valid_positions > 0) & (valid_positions < seq_len - 1)
        valid_matches = matches[interior]
        if valid_matches.numel() > 0:
            batch_positions = valid_matches[:, 0]
            valid_positions = valid_matches[:, 1]
            neighbor_ok = (
                (input_ids[batch_positions, valid_positions - 1] == left_id)
                & (input_ids[batch_positions, valid_positions + 1] == right_id)
            )
            valid_matches = valid_matches[neighbor_ok]

    vec_idx = int(valid_matches.shape[0])
    expected = vectors.shape[0]
    if vec_idx != expected:
        msg = (
            f"found {vec_idx} injection sites with correct neighbors, expected {expected}. "
            f"Check prompt template drift, tokenizer version, cp accidentally >1, "
            f"or (RL) rollout samples with multimodal_train_inputs=None skipped in concat."
        )
        # Under PP, this hook only runs on stage 0. Bare assert leaves stage 1
        # hanging on P2P recv → 10min NCCL timeout with no error. Abort the
        # whole world so the real error surfaces.
        if torch.distributed.is_initialized():
            print(f"[inject_at_marked_positions] FATAL: {msg}", flush=True)
            torch.distributed.destroy_process_group()
        raise RuntimeError(msg)
    if vec_idx:
        valid_positions = valid_matches[:, 1]
        local_mask = (valid_positions >= start) & (valid_positions < end)
        local_matches = valid_matches[local_mask]
        if local_matches.numel() > 0:
            index_device = out.device
            local_batch_indices = local_matches[:, 0].to(index_device)
            local_positions = (local_matches[:, 1] - start).to(index_device)
            local_vector_indices = local_mask.nonzero(as_tuple=False).flatten().to(index_device)
            out[local_batch_indices, local_positions] = vectors.index_select(0, local_vector_indices)
    return out
