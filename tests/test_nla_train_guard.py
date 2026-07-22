import math

import pytest

from nla.train_guard import CompositeMetricGuard, DriftGuard, DriftGuardTriggered, MetricRule


METRIC = "train/train_rollout_logprob_abs_diff"


def test_guard_requires_two_consecutive_exceedances():
    guard = DriftGuard(max_logprob_abs_diff=0.75, consecutive_steps=2)

    guard.check({METRIC: 0.8}, role="actor", step=3)
    with pytest.raises(DriftGuardTriggered) as exc_info:
        guard.check({METRIC: 0.9}, role="actor", step=4)

    assert exc_info.value.details["reason"] == "consecutive_threshold_exceeded"
    assert exc_info.value.details["consecutive_exceedances"] == 2
    assert exc_info.value.details["step"] == 4


def test_guard_resets_after_healthy_step():
    guard = DriftGuard(max_logprob_abs_diff=0.75, consecutive_steps=2)

    guard.check({METRIC: 0.8}, role="actor", step=3)
    result = guard.check({METRIC: 0.2}, role="actor", step=4)
    guard.check({METRIC: 0.8}, role="actor", step=5)

    assert result["consecutive_exceedances"] == 0


def test_guard_ignores_critic_metrics():
    guard = DriftGuard(max_logprob_abs_diff=0.75, consecutive_steps=1)

    result = guard.check({METRIC: math.inf}, role="critic", step=1)

    assert result["ignored"] is True


def test_guard_rejects_any_nonfinite_actor_metric_immediately():
    guard = DriftGuard(max_logprob_abs_diff=0.75, consecutive_steps=3)

    with pytest.raises(DriftGuardTriggered) as exc_info:
        guard.check({METRIC: 0.1, "train/loss": float("nan")}, role="actor", step=1)

    assert exc_info.value.details["reason"] == "nonfinite_metric"
    assert exc_info.value.details["metric"] == "train/loss"


def test_guard_accepts_unprefixed_miles_metric_name():
    guard = DriftGuard(max_logprob_abs_diff=0.75, consecutive_steps=1)

    with pytest.raises(DriftGuardTriggered):
        guard.check({"train_rollout_logprob_abs_diff": 0.8}, role="actor", step=1)


def test_composite_guard_stops_repeated_kl_spikes():
    guard = CompositeMetricGuard(
        [MetricRule("train/kl_loss", "max", threshold=5.0, consecutive_steps=2)]
    )

    guard.check({"train/kl_loss": 6.0}, role="actor", step=1)
    with pytest.raises(DriftGuardTriggered, match="metric_guard_triggered"):
        guard.check({"train/kl_loss": 5.5}, role="actor", step=2)


def test_composite_guard_rejects_nonfinite_unlisted_metric():
    guard = CompositeMetricGuard(
        [MetricRule("train/kl_loss", "max", threshold=5.0, consecutive_steps=2)]
    )

    with pytest.raises(DriftGuardTriggered) as exc_info:
        guard.check(
            {"train/kl_loss": 0.2, "train/grad_norm": float("nan")},
            role="actor",
            step=1,
        )

    assert exc_info.value.details["reason"] == "nonfinite_metric"
    assert exc_info.value.details["metric"] == "train/grad_norm"


def test_increasing_metric_rule_tracks_consecutive_rises():
    guard = CompositeMetricGuard(
        [
            MetricRule(
                "rollout/nla_response_length/p95",
                "increasing",
                consecutive_steps=3,
                role_prefixes=("rollout",),
                min_delta=1.0,
            )
        ]
    )

    for step, value in enumerate((100.0, 101.0, 102.0), start=1):
        guard.check({"rollout/nla_response_length/p95": value}, role="rollout", step=step)
    with pytest.raises(DriftGuardTriggered):
        guard.check({"rollout/nla_response_length/p95": 103.0}, role="rollout", step=4)
