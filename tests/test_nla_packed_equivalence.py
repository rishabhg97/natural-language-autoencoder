import pytest

torch = pytest.importorskip("torch")

from external.natural_language_autoencoders.nla.packed_equivalence import (
    build_bshd_attention_mask,
    build_bshd_max_seq_lens,
    build_packed_padded_inputs,
    packed_equivalence_metrics,
    response_mean_nlls,
)


def test_build_bshd_attention_mask_uses_true_row_lengths():
    rows = [torch.tensor([10, 11, 12]), torch.tensor([20, 21])]
    padded = torch.tensor([[10, 11, 12, 0], [20, 21, 0, 0]])

    mask = build_bshd_attention_mask(rows, padded)

    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, True, False], [True, True, False, False]]


def test_build_bshd_attention_mask_rejects_batch_mismatch():
    with pytest.raises(ValueError, match="row count"):
        build_bshd_attention_mask(
            [torch.tensor([10, 11])],
            torch.tensor([[10, 11], [20, 21]]),
        )


def test_build_bshd_max_seq_lens_rounds_rewritten_critic_rows():
    rows = [torch.arange(125), torch.arange(67), torch.arange(128)]

    assert build_bshd_max_seq_lens(rows, pad_size=128) == [128, 128, 128]


def test_build_bshd_max_seq_lens_rejects_empty_rows():
    with pytest.raises(ValueError, match="non-empty"):
        build_bshd_max_seq_lens([torch.tensor([], dtype=torch.long)], pad_size=128)


def test_build_packed_padded_inputs_resets_positions_and_masks_padding():
    rows = [torch.tensor([10, 11, 12, 13]), torch.tensor([20, 21, 22])]

    inputs = build_packed_padded_inputs(rows)

    assert inputs.packed_input_ids.tolist() == [[10, 11, 12, 13, 20, 21, 22]]
    assert inputs.packed_position_ids.tolist() == [[0, 1, 2, 3, 0, 1, 2]]
    assert inputs.padded_input_ids.tolist() == [[10, 11, 12, 13], [20, 21, 22, 0]]
    assert inputs.padded_attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]


def test_response_mean_nlls_align_packed_and_padded_response_targets():
    rows = (torch.tensor([10, 11, 12, 13]), torch.tensor([20, 21, 22]))
    packed_logits = torch.zeros(1, 7, 32)
    padded_logits = torch.zeros(2, 4, 32)
    for packed_position, padded_row, padded_position, target in (
        (1, 0, 1, 12),
        (2, 0, 2, 13),
        (5, 1, 1, 22),
    ):
        packed_logits[0, packed_position, target] = 5
        padded_logits[padded_row, padded_position, target] = 5

    packed = response_mean_nlls(
        packed_logits, rows, [2, 1], packed=True
    )
    padded = response_mean_nlls(
        padded_logits, rows, [2, 1], packed=False
    )

    torch.testing.assert_close(packed, padded)


def test_packed_equivalence_metrics_fail_closed_on_excess_drift():
    passed = packed_equivalence_metrics(
        torch.tensor([1.001, 2.001]),
        torch.tensor([1.0, 2.0]),
        rtol=0.01,
        atol=0.001,
    )
    failed = packed_equivalence_metrics(
        torch.tensor([1.0, 2.2]),
        torch.tensor([1.0, 2.0]),
        rtol=0.01,
        atol=0.001,
    )

    assert passed["passed"]
    assert not failed["passed"]
    assert failed["max_abs_diff"] > 0.19
