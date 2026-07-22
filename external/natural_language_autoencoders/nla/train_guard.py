"""Fail-fast guardrails for unstable NLA actor training metrics."""

from __future__ import annotations

import json
import math
import numbers
import os
from dataclasses import dataclass, field
from typing import Any


DEFAULT_METRIC = "train/train_rollout_logprob_abs_diff"


class DriftGuardTriggered(RuntimeError):
    """Raised on every actor rank when training violates a guard invariant."""

    def __init__(self, details: dict[str, Any]):
        self.details = details
        super().__init__(f"NLA train guard triggered: {json.dumps(details, sort_keys=True)}")


@dataclass
class MetricRule:
    metric_name: str
    comparison: str
    threshold: float = 0.0
    consecutive_steps: int = 1
    role_prefixes: tuple[str, ...] = ("actor",)
    min_delta: float = 0.0
    _consecutive_exceedances: int = field(default=0, init=False)
    _previous_value: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.metric_name = str(self.metric_name).strip()
        self.comparison = str(self.comparison).strip().lower()
        self.threshold = float(self.threshold)
        self.consecutive_steps = int(self.consecutive_steps)
        self.role_prefixes = tuple(str(value).strip().lower() for value in self.role_prefixes)
        self.min_delta = float(self.min_delta)
        if not self.metric_name:
            raise ValueError("metric_name must not be empty")
        if self.comparison not in {"max", "min", "increasing"}:
            raise ValueError("comparison must be max, min, or increasing")
        if not math.isfinite(self.threshold) or not math.isfinite(self.min_delta):
            raise ValueError("metric rule thresholds must be finite")
        if self.consecutive_steps <= 0:
            raise ValueError("consecutive_steps must be positive")
        if not self.role_prefixes:
            raise ValueError("role_prefixes must not be empty")

    def applies_to(self, role: str) -> bool:
        role_name = str(role or "").lower()
        return any(role_name.startswith(prefix) for prefix in self.role_prefixes)

    def metric_candidates(self) -> tuple[str, ...]:
        if "/" in self.metric_name:
            return self.metric_name, self.metric_name.split("/", 1)[1]
        return self.metric_name, f"train/{self.metric_name}", f"rollout/{self.metric_name}"

    def check(self, metrics: dict[str, Any], *, role: str, step: int) -> dict[str, Any]:
        if not self.applies_to(role):
            return {"ignored": True, "metric": self.metric_name, "role": role, "step": int(step)}
        metric_key = next((name for name in self.metric_candidates() if name in metrics), None)
        if metric_key is None:
            return {
                "ignored": False,
                "metric_missing": True,
                "metric": self.metric_name,
                "role": role,
                "step": int(step),
            }
        value = float(metrics[metric_key])
        if not math.isfinite(value):
            raise DriftGuardTriggered(
                {
                    "reason": "nonfinite_metric",
                    "metric": metric_key,
                    "value": str(value),
                    "role": role,
                    "step": int(step),
                }
            )
        if self.comparison == "max":
            exceeded = value > self.threshold
        elif self.comparison == "min":
            exceeded = value < self.threshold
        else:
            exceeded = self._previous_value is not None and value >= self._previous_value + self.min_delta
            self._previous_value = value
        self._consecutive_exceedances = self._consecutive_exceedances + 1 if exceeded else 0
        result = {
            "ignored": False,
            "metric": metric_key,
            "value": value,
            "comparison": self.comparison,
            "threshold": self.threshold,
            "min_delta": self.min_delta,
            "role": role,
            "step": int(step),
            "consecutive_exceedances": self._consecutive_exceedances,
            "required_consecutive_steps": self.consecutive_steps,
        }
        if self._consecutive_exceedances >= self.consecutive_steps:
            raise DriftGuardTriggered({"reason": "metric_guard_triggered", **result})
        return result


@dataclass
class CompositeMetricGuard:
    rules: list[MetricRule]

    def check(self, metrics: dict[str, Any], *, role: str, step: int) -> dict[str, Any]:
        if not isinstance(metrics, dict):
            raise DriftGuardTriggered(
                {"reason": "invalid_metrics", "role": role, "step": int(step), "type": type(metrics).__name__}
            )
        for name, value in metrics.items():
            if isinstance(value, numbers.Number) and not math.isfinite(float(value)):
                raise DriftGuardTriggered(
                    {
                        "reason": "nonfinite_metric",
                        "role": role,
                        "step": int(step),
                        "metric": str(name),
                        "value": str(value),
                    }
                )
        return {
            "role": role,
            "step": int(step),
            "rules": [rule.check(metrics, role=role, step=step) for rule in self.rules],
        }


def metric_rules_from_json(value: str) -> list[MetricRule]:
    payload = json.loads(value)
    if not isinstance(payload, list) or not payload:
        raise ValueError("metric guard rules must be a non-empty JSON list")
    rules: list[MetricRule] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each metric guard rule must be an object")
        rules.append(
            MetricRule(
                metric_name=item["metric"],
                comparison=item.get("comparison", "max"),
                threshold=item.get("threshold", 0.0),
                consecutive_steps=item.get("consecutive_steps", 1),
                role_prefixes=tuple(item.get("role_prefixes", ("actor",))),
                min_delta=item.get("min_delta", 0.0),
            )
        )
    return rules


@dataclass
class DriftGuard:
    max_logprob_abs_diff: float
    consecutive_steps: int = 2
    metric_name: str = DEFAULT_METRIC
    _consecutive_exceedances: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.max_logprob_abs_diff = float(self.max_logprob_abs_diff)
        self.consecutive_steps = int(self.consecutive_steps)
        if not math.isfinite(self.max_logprob_abs_diff) or self.max_logprob_abs_diff < 0.0:
            raise ValueError("max_logprob_abs_diff must be finite and non-negative")
        if self.consecutive_steps <= 0:
            raise ValueError("consecutive_steps must be positive")
        if not str(self.metric_name).strip():
            raise ValueError("metric_name must not be empty")
        self.metric_name = str(self.metric_name).strip()

    def _metric_candidates(self) -> tuple[str, ...]:
        if self.metric_name.startswith("train/"):
            return self.metric_name, self.metric_name[len("train/") :]
        return self.metric_name, f"train/{self.metric_name}"

    def check(
        self,
        metrics: dict[str, Any],
        *,
        role: str,
        step: int,
    ) -> dict[str, Any]:
        role_name = str(role or "").lower()
        if not role_name.startswith("actor"):
            return {
                "ignored": True,
                "role": role,
                "step": int(step),
                "consecutive_exceedances": self._consecutive_exceedances,
            }
        if not isinstance(metrics, dict):
            raise DriftGuardTriggered(
                {
                    "reason": "invalid_metrics",
                    "role": role,
                    "step": int(step),
                    "type": type(metrics).__name__,
                }
            )

        for name, value in metrics.items():
            if isinstance(value, numbers.Number) and not math.isfinite(float(value)):
                raise DriftGuardTriggered(
                    {
                        "reason": "nonfinite_metric",
                        "role": role,
                        "step": int(step),
                        "metric": str(name),
                        "value": str(value),
                    }
                )

        metric_key = next(
            (name for name in self._metric_candidates() if name in metrics),
            None,
        )
        if metric_key is None:
            return {
                "ignored": False,
                "metric_missing": True,
                "role": role,
                "step": int(step),
                "consecutive_exceedances": self._consecutive_exceedances,
            }
        value = float(metrics[metric_key])
        if value > self.max_logprob_abs_diff:
            self._consecutive_exceedances += 1
        else:
            self._consecutive_exceedances = 0

        result = {
            "ignored": False,
            "metric": metric_key,
            "value": value,
            "threshold": self.max_logprob_abs_diff,
            "role": role,
            "step": int(step),
            "consecutive_exceedances": self._consecutive_exceedances,
            "required_consecutive_steps": self.consecutive_steps,
        }
        if self._consecutive_exceedances >= self.consecutive_steps:
            raise DriftGuardTriggered(
                {"reason": "consecutive_threshold_exceeded", **result}
            )
        return result


_PROCESS_GUARD: DriftGuard | CompositeMetricGuard | None = None


def _process_guard() -> DriftGuard | CompositeMetricGuard:
    global _PROCESS_GUARD
    if _PROCESS_GUARD is None:
        rules_json = os.environ.get("NLA_TRAIN_GUARD_RULES_JSON")
        if rules_json:
            _PROCESS_GUARD = CompositeMetricGuard(metric_rules_from_json(rules_json))
        else:
            _PROCESS_GUARD = DriftGuard(
                max_logprob_abs_diff=float(
                    os.environ.get("NLA_TRAIN_GUARD_MAX_LOGPROB_ABS_DIFF", "0.75")
                ),
                consecutive_steps=int(
                    os.environ.get("NLA_TRAIN_GUARD_CONSECUTIVE_STEPS", "2")
                ),
                metric_name=os.environ.get("NLA_TRAIN_GUARD_METRIC", DEFAULT_METRIC),
            )
    return _PROCESS_GUARD


def reset_process_guard() -> None:
    """Reset process-local state for tests and explicit worker reinitialization."""

    global _PROCESS_GUARD
    _PROCESS_GUARD = None


def check_train_metrics(
    *,
    args: Any,
    metrics: dict[str, Any],
    role: str,
    step: int,
) -> dict[str, Any]:
    del args
    return _process_guard().check(metrics, role=role, step=step)


_PROCESS_ROLLOUT_GUARD: CompositeMetricGuard | None = None


def reset_rollout_guard() -> None:
    global _PROCESS_ROLLOUT_GUARD
    _PROCESS_ROLLOUT_GUARD = None


def check_rollout_metrics(metrics: dict[str, Any], *, step: int) -> dict[str, Any] | None:
    global _PROCESS_ROLLOUT_GUARD
    rules_json = os.environ.get("NLA_ROLLOUT_GUARD_RULES_JSON")
    if not rules_json:
        return None
    if _PROCESS_ROLLOUT_GUARD is None:
        _PROCESS_ROLLOUT_GUARD = CompositeMetricGuard(metric_rules_from_json(rules_json))
    return _PROCESS_ROLLOUT_GUARD.check(metrics, role="rollout", step=step)
