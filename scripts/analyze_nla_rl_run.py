#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")

ROLLOUT_TRAJECTORY_KEYS = (
    "rollout_id",
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_max",
    "usable_reward_mean",
    "usable_reward_std",
    "closed_frac",
    "usable_frac",
    "failed_frac",
    "generation_truncated_frac",
    "response_length_p95",
    "length_corr",
)
ACTOR_TRAJECTORY_KEYS = (
    "train/loss",
    "train/pg_loss",
    "train/kl_loss",
    "train/ppo_kl",
    "train/pg_clipfrac",
    "train/grad_norm",
    "train/lr-pg_0",
    "train/train_rollout_logprob_abs_diff",
    "train/train_rollout_logprob_abs_diff_mean",
)
CRITIC_TRAJECTORY_KEYS = (
    "train/loss",
    "train/fve_nrm",
    "train/grad_norm",
    "train/lr-pg_0",
    "train/cosine_sum",
    "train/pred_norm_ratio_sum",
)
PERF_TRAJECTORY_KEYS = (
    "perf/update_weights_time",
    "perf/train_wait_time",
    "perf/ref_log_probs_time",
    "perf/log_probs_time",
    "perf/actor_train_time",
    "perf/train_time",
    "perf/step_time",
    "perf/wait_time_ratio",
)


def _clean(line: str) -> str:
    return ANSI_RE.sub("", line)


def _parse_dict_after(pattern: str, line: str) -> tuple[int, dict[str, Any]] | None:
    match = re.search(pattern, line)
    if not match:
        return None
    try:
        return int(match.group(1)), ast.literal_eval(match.group(2))
    except (SyntaxError, ValueError):
        return None


def _parse_key_values(text: str) -> dict[str, float | int | str]:
    parsed: dict[str, float | int | str] = {}
    for part in text.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            number = float(value)
        except ValueError:
            parsed[key] = value
            continue
        parsed[key] = int(number) if number.is_integer() else number
    return parsed


def _parse_nla_rollout(line: str) -> dict[str, Any] | None:
    if "[NLA ROLLOUT]" not in line:
        return None
    raw = line.split("[NLA ROLLOUT]", 1)[1].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _parse_nla_advantage(line: str) -> dict[str, Any] | None:
    if "[NLA ADVANTAGE]" not in line:
        return None
    return _parse_key_values(line.split("[NLA ADVANTAGE]", 1)[1].strip())


def _metric_subset(metrics: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: metrics[key] for key in keys if key in metrics}


def _step_role(metrics: dict[str, Any]) -> str | None:
    if "train/fve_nrm" in metrics:
        return "critic"
    if "train/ppo_kl" in metrics or "train/pg_loss" in metrics:
        return "actor"
    return None


def _perf_role(metrics: dict[str, Any]) -> str | None:
    if "perf/ref_log_probs_time" in metrics or "perf/log_probs_time" in metrics:
        return "actor"
    if "perf/actor_train_time" in metrics:
        return "critic"
    return None


def analyze_train_log(path: Path) -> dict[str, Any]:
    rollouts: list[tuple[int, dict[str, Any]]] = []
    perfs: list[tuple[int, dict[str, Any]]] = []
    steps: list[tuple[int, dict[str, Any]]] = []
    nla_rollouts: list[dict[str, Any]] = []
    nla_advantages: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    for raw in path.read_text(errors="ignore").splitlines():
        line = _clean(raw)
        parsed_rollout = _parse_nla_rollout(line)
        if parsed_rollout is not None:
            nla_rollouts.append(parsed_rollout)
        parsed_advantage = _parse_nla_advantage(line)
        if parsed_advantage is not None:
            nla_advantages.append(parsed_advantage)
        for pattern, target in (
            (r"rollout (\d+): (\{.*\})", rollouts),
            (r"perf (\d+): (\{.*\})", perfs),
            (r"step (\d+): (\{.*\})", steps),
        ):
            parsed = _parse_dict_after(pattern, line)
            if parsed is not None:
                target.append(parsed)
                break
        lower = line.lower()
        if "warning" in lower:
            warnings.append(line[:500])
        if any(token in lower for token in ("traceback", "error", "oom", "out of memory")):
            errors.append(line[:500])

    latest_nla_rollout = nla_rollouts[-1] if nla_rollouts else None
    actor_steps = [
        {"step": step, **_metric_subset(metrics, ACTOR_TRAJECTORY_KEYS)}
        for step, metrics in steps
        if _step_role(metrics) == "actor"
    ]
    critic_steps = [
        {"step": step, **_metric_subset(metrics, CRITIC_TRAJECTORY_KEYS)}
        for step, metrics in steps
        if _step_role(metrics) == "critic"
    ]
    actor_perfs = [
        {"step": step, **_metric_subset(metrics, PERF_TRAJECTORY_KEYS)}
        for step, metrics in perfs
        if _perf_role(metrics) == "actor"
    ]
    critic_perfs = [
        {"step": step, **_metric_subset(metrics, PERF_TRAJECTORY_KEYS)}
        for step, metrics in perfs
        if _perf_role(metrics) == "critic"
    ]
    return {
        "path": str(path),
        "rollout_count": len(rollouts),
        "perf_count": len(perfs),
        "step_count": len(steps),
        "nla_rollout_count": len(nla_rollouts),
        "nla_advantage_count": len(nla_advantages),
        "latest_rollout": rollouts[-1][1] if rollouts else None,
        "latest_perf": perfs[-1][1] if perfs else None,
        "latest_step": steps[-1][1] if steps else None,
        "latest_nla_rollout": latest_nla_rollout,
        "latest_nla_advantage": nla_advantages[-1] if nla_advantages else None,
        "nla_rollout_trajectory": [
            _metric_subset(metrics, ROLLOUT_TRAJECTORY_KEYS) for metrics in nla_rollouts
        ],
        "actor_step_trajectory": actor_steps,
        "critic_step_trajectory": critic_steps,
        "actor_perf_trajectory": actor_perfs,
        "critic_perf_trajectory": critic_perfs,
        "usable_fraction": latest_nla_rollout.get("usable_frac") if latest_nla_rollout else None,
        "failed_fraction": latest_nla_rollout.get("failed_frac") if latest_nla_rollout else None,
        "warning_count": len(warnings),
        "error_count": len(errors),
        "warnings_tail": warnings[-10:],
        "errors_tail": errors[-10:],
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "path",
        "rollout_count",
        "perf_count",
        "step_count",
        "nla_rollout_count",
        "nla_advantage_count",
        "nla_rollout_trajectory",
        "actor_step_trajectory",
        "critic_step_trajectory",
        "actor_perf_trajectory",
        "critic_perf_trajectory",
        "latest_nla_advantage",
        "warning_count",
        "error_count",
        "warnings_tail",
        "errors_tail",
    )
    return {key: summary[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit role-aware HPO trajectories without full router/system payloads.",
    )
    args = parser.parse_args()
    summary = analyze_train_log(args.train_log)
    if args.compact:
        summary = compact_summary(summary)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
