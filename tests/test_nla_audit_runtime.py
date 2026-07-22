from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytest

torch = pytest.importorskip("torch")

from nla.audit_runtime import (
    aggregate_train_losses_by_key,
    clip_grad_norm_local_shards,
    mse_ratio_agreement,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_local_shard_clip_scales_original_gradients(dtype: torch.dtype) -> None:
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
    parameter.grad = torch.tensor([3.0, 4.0], dtype=dtype)

    total_norm = clip_grad_norm_local_shards([parameter], 1.0)

    assert float(total_norm) == pytest.approx(5.0)
    assert parameter.grad is not None
    assert float(torch.linalg.vector_norm(parameter.grad.float())) == pytest.approx(1.0, abs=0.01)


def test_local_shard_clip_reports_norm_without_scaling_below_limit() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    parameter.grad = torch.tensor([0.3, 0.4], dtype=torch.bfloat16)
    before = parameter.grad.clone()

    total_norm = clip_grad_norm_local_shards([parameter], 1.0)

    assert float(total_norm) == pytest.approx(0.5, abs=0.01)
    assert torch.equal(parameter.grad, before)


def test_keyed_loss_aggregation_handles_dynamic_microbatch_metrics() -> None:
    losses = [
        {
            "keys": ["policy_loss", "system/gpu_memory"],
            "values": torch.tensor([2.0, 4.0, 20.0]),
        },
        {
            "keys": ["policy_loss", "router/entropy"],
            "values": torch.tensor([1.0, 3.0, 5.0]),
        },
    ]
    parallel_state = SimpleNamespace(dp_cp_group=None, cp_size=1)

    aggregated = aggregate_train_losses_by_key(losses, parallel_state)

    assert aggregated == pytest.approx(
        {
            "policy_loss": 7.0 / 3.0,
            "system/gpu_memory": 10.0,
            "router/entropy": 5.0,
        }
    )


def test_mse_ratio_agreement_reports_tail_drift() -> None:
    agreement = mse_ratio_agreement(
        torch.tensor([1.0, 1.98, 4.2, 7.6]),
        torch.tensor([1.0, 2.0, 4.0, 8.0]),
    )

    assert agreement.mean_ratio == pytest.approx(0.9975)
    assert agreement.max_abs_deviation == pytest.approx(0.05)
    assert agreement.p95_abs_deviation == pytest.approx(0.05)
    assert agreement.ratios == pytest.approx((1.0, 0.99, 1.05, 0.95))


def test_mse_ratio_agreement_rejects_nonpositive_denominator() -> None:
    with pytest.raises(ValueError, match="denominator"):
        mse_ratio_agreement(torch.tensor([1.0]), torch.tensor([0.0]))
