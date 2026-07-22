"""NLA-specific rollout metrics for Miles RL logging.

Miles calls this hook before its default rollout logger. We mutate the
``rollout_extra_metrics`` dict and return ``False`` so the stock logger still
records response length, throughput, and other built-in metrics.
"""

from __future__ import annotations

import math
import json
from collections.abc import Mapping
from typing import Any

from nla.schema import extract_explanation
from nla.train_guard import check_rollout_metrics


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count

    def percentile(q: float) -> float:
        if count == 1:
            return ordered[0]
        position = (count - 1) * q
        lo = int(math.floor(position))
        hi = int(math.ceil(position))
        if lo == hi:
            return ordered[lo]
        weight = position - lo
        return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": ordered[0],
        "max": ordered[-1],
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
    }


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denom = math.sqrt(x_var * y_var)
    return 0.0 if denom == 0.0 else cov / denom


def _prefix(prefix: str, values: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}/{key}": value for key, value in values.items()}


def _flatten_float_values(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            return [
                float(item)
                for item in value.detach().float().cpu().view(-1).tolist()
                if math.isfinite(float(item))
            ]
        except Exception:
            return []
    if isinstance(value, Mapping):
        values: list[float] = []
        for item in value.values():
            values.extend(_flatten_float_values(item))
        return values
    if isinstance(value, (str, bytes)):
        scalar = _as_float(value)
        return [scalar] if scalar is not None and math.isfinite(scalar) else []
    if isinstance(value, (list, tuple)):
        values: list[float] = []
        for item in value:
            values.extend(_flatten_float_values(item))
        return values
    if hasattr(value, "tolist"):
        try:
            return _flatten_float_values(value.tolist())
        except Exception:
            return []
    scalar = _as_float(value)
    return [scalar] if scalar is not None and math.isfinite(scalar) else []


def advantage_stats_from_rollout_data(rollout_data: dict[str, Any]) -> dict[str, float | int]:
    values_obj = rollout_data.get("advantages")
    if values_obj is None:
        return {}
    values = _flatten_float_values(values_obj)
    if not values:
        return {}
    stats = _prefix("rollout/nla_advantage", _stats(values))
    stats["rollout/nla_advantage/frac_zero"] = sum(abs(value) <= 1e-12 for value in values) / len(values)
    return stats


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: list[Any],
    rollout_extra_metrics: dict[str, float | int] | None,
    _rollout_time: float,
) -> bool:
    """Add NLA reward/parse scalars, then let Miles perform default logging."""

    if rollout_extra_metrics is None:
        return False

    rewards: list[float] = []
    usable_rewards: list[float] = []
    response_lengths: list[float] = []
    usable = 0
    closed = 0
    completed = 0
    truncated = 0
    failed = 0
    generation_completed = 0
    generation_truncated = 0
    generation_failed = 0

    for sample in samples:
        reward = _as_float(sample.get_reward_value(args))
        if reward is not None:
            rewards.append(reward)
        length = _as_float(getattr(sample, "effective_response_length", getattr(sample, "response_length", 0)))
        if length is not None:
            response_lengths.append(length)

        status_name = getattr(getattr(sample, "status", None), "name", "")
        completed += int(status_name == "COMPLETED")
        truncated += int(status_name == "TRUNCATED")
        failed += int(status_name == "FAILED")
        generation_status_name = str(
            getattr(sample, "nla_generation_status", status_name)
        )
        generation_completed += int(generation_status_name == "COMPLETED")
        generation_truncated += int(generation_status_name == "TRUNCATED")
        generation_failed += int(generation_status_name == "FAILED")

        response = getattr(sample, "response", "") or ""
        closed += int("</explanation>" in response)
        explanation = extract_explanation(response)
        is_usable = explanation is not None and bool(explanation.strip())
        usable += int(is_usable)
        if is_usable and reward is not None:
            usable_rewards.append(reward)

    n = max(1, len(samples))
    rollout_extra_metrics.update(_prefix("rollout/nla_reward", _stats(rewards)))
    rollout_extra_metrics.update(
        _prefix("rollout/nla_usable_reward", _stats(usable_rewards))
    )
    rollout_extra_metrics.update(_prefix("rollout/nla_response_length", _stats(response_lengths)))
    rollout_extra_metrics.update(
        {
            "rollout/nla_parse/closed_frac": closed / n,
            "rollout/nla_parse/usable_frac": usable / n,
            "rollout/nla_status/completed_frac": completed / n,
            "rollout/nla_status/truncated_frac": truncated / n,
            "rollout/nla_status/failed_frac": failed / n,
            "rollout/nla_generation/completed_frac": generation_completed / n,
            "rollout/nla_generation/truncated_frac": generation_truncated / n,
            "rollout/nla_generation/failed_frac": generation_failed / n,
        }
    )
    if rewards and len(response_lengths) == len(rewards):
        rollout_extra_metrics["rollout/nla_reward/length_corr"] = _corr(response_lengths, rewards)
    payload = {
        "rollout_id": rollout_id,
        "sample_count": len(samples),
        "reward_count": len(rewards),
        "reward_mean": rollout_extra_metrics.get("rollout/nla_reward/mean"),
        "reward_std": rollout_extra_metrics.get("rollout/nla_reward/std"),
        "reward_min": rollout_extra_metrics.get("rollout/nla_reward/min"),
        "reward_max": rollout_extra_metrics.get("rollout/nla_reward/max"),
        "usable_reward_mean": rollout_extra_metrics.get("rollout/nla_usable_reward/mean"),
        "usable_reward_std": rollout_extra_metrics.get("rollout/nla_usable_reward/std"),
        "closed_frac": rollout_extra_metrics.get("rollout/nla_parse/closed_frac"),
        "usable_frac": rollout_extra_metrics.get("rollout/nla_parse/usable_frac"),
        "failed_frac": rollout_extra_metrics.get("rollout/nla_status/failed_frac"),
        "truncated_frac": rollout_extra_metrics.get("rollout/nla_status/truncated_frac"),
        "generation_truncated_frac": rollout_extra_metrics.get(
            "rollout/nla_generation/truncated_frac"
        ),
        "length_corr": rollout_extra_metrics.get("rollout/nla_reward/length_corr"),
        "response_length_p95": rollout_extra_metrics.get("rollout/nla_response_length/p95"),
    }
    print("[NLA ROLLOUT] " + json.dumps(payload, sort_keys=True), flush=True)
    check_rollout_metrics(rollout_extra_metrics, step=rollout_id)
    return False
