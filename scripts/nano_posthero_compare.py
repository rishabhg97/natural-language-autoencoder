#!/usr/bin/env python3
"""Build post-hero Nano NLA comparability and reward-dry-run reports."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def split_gate(report: dict[str, Any], split: str) -> dict[str, Any]:
    return (((report.get("gate") or {}).get("splits") or {}).get(split) or {})


def split_payload(report: dict[str, Any], split: str) -> dict[str, Any]:
    return (report.get("splits") or {}).get(split) or {}


def primary_nmse(report: dict[str, Any], split: str) -> float:
    value = split_gate(report, split).get("primary_normalized_mse")
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing primary_normalized_mse for split={split}")
    return float(value)


def teacher_nmse(report: dict[str, Any], split: str) -> float:
    value = split_gate(report, split).get("teacher_normalized_mse")
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing teacher_normalized_mse for split={split}")
    return float(value)


def row_losses(report: dict[str, Any], split: str, variant: str = "av_real") -> list[float]:
    losses = (split_payload(report, split).get("rowwise_normalized_mse") or {}).get(variant)
    if not isinstance(losses, list) or not losses:
        raise ValueError(f"missing rowwise losses for split={split} variant={variant}")
    return [float(item) for item in losses]


def parse_health(report: dict[str, Any], split: str) -> dict[str, Any]:
    return split_gate(report, split).get("parse_health") or {}


def controls(report: dict[str, Any], split: str) -> dict[str, Any]:
    return split_gate(report, split).get("controls") or {}


def reward_summary(losses: list[float]) -> dict[str, float | int]:
    rewards = [-value for value in losses]
    return {
        "count": len(rewards),
        "reward_mean": statistics.fmean(rewards),
        "reward_median": statistics.median(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "nmse_mean": statistics.fmean(losses),
        "nmse_median": statistics.median(losses),
        "nmse_p10": percentile(losses, 0.10),
        "nmse_p90": percentile(losses, 0.90),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile input")
    index = q * (len(ordered) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    weight = index - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def extract_qwen_qc(markdown: str) -> dict[str, Any]:
    patterns = {
        "correct_mean_mse": r"\| correct \|\s*56\s*\|\s*([0-9.]+)",
        "correct_median_mse": r"\| correct \|\s*56\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)",
        "correct_mean_cosine": r"\| correct \|\s*56\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)",
        "mean_target_mean_mse": r"\| mean_target \|\s*56\s*\|\s*([0-9.]+)",
        "shuffled_mean_mse": r"\| shuffled_text \|\s*56\s*\|\s*([0-9.]+)",
        "random_mean_mse": r"\| random_text \|\s*56\s*\|\s*([0-9.]+)",
    }
    result: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, markdown)
        result[key] = float(match.group(1)) if match else None
    pass_match = re.search(r"\| Scientific pass \|\s*(true|false)", markdown)
    result["scientific_pass"] = None if not pass_match else pass_match.group(1) == "true"
    return result


def report_block(name: str, report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "gate_passed": bool((report.get("gate") or {}).get("passed")),
        "baseline_required": bool((report.get("gate") or {}).get("baseline_required")),
        "splits": {},
    }
    for split in ("validation", "test"):
        result["splits"][split] = {
            "primary_nmse": primary_nmse(report, split),
            "teacher_nmse": teacher_nmse(report, split),
            "parse": parse_health(report, split),
            "controls": controls(report, split),
            "reward_dry_run": reward_summary(row_losses(report, split, "av_real")),
        }
    result["name"] = name
    return result


def improvement(candidate: float, baseline: float) -> float:
    return 1.0 - candidate / baseline


def render_markdown(payload: dict[str, Any]) -> str:
    r33 = payload["r33"]
    r27 = payload["r27"]
    qwen = payload["qwen_qc"]
    lines = [
        "# R33 Post-Hero Comparability And Reward Dry Run",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Summary",
        "",
        "- R33 component-full AV+AR SFT is the selected internal hero milestone.",
        "- The R27 256/256 baseline report is useful baseline evidence, but it is not a freshly retrained post-fix row-identical R27 pair.",
        "- The reward dry run uses the frozen AR scoring already present in the round-trip reports: `reward = -rowwise_normalized_mse` for AV-real generated text.",
        "- Do not start RL from this report alone unless the operator accepts the non-row-identical R27 caveat and the RL smoke is explicitly guarded.",
        "",
        "## R33 Vs R27 Round-Trip",
        "",
        "| Split | R33 AV-real NMSE | R27 AV-real NMSE | R33 relative improvement | R33 teacher NMSE | R27 teacher NMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        r33_split = r33["splits"][split]
        r27_split = r27["splits"][split]
        lines.append(
            "| {split} | `{r33_primary:.9f}` | `{r27_primary:.9f}` | `{improve:.1%}` | `{r33_teacher:.9f}` | `{r27_teacher:.9f}` |".format(
                split=split,
                r33_primary=r33_split["primary_nmse"],
                r27_primary=r27_split["primary_nmse"],
                improve=improvement(r33_split["primary_nmse"], r27_split["primary_nmse"]),
                r33_teacher=r33_split["teacher_nmse"],
                r27_teacher=r27_split["teacher_nmse"],
            )
        )
    lines.extend(
        [
            "",
            "## Reward Dry Run",
            "",
            "| Split | Reward mean | Reward median | NMSE p10 | NMSE p90 | Count |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("validation", "test"):
        reward = r33["splits"][split]["reward_dry_run"]
        lines.append(
            "| {split} | `{reward_mean:.9f}` | `{reward_median:.9f}` | `{nmse_p10:.9f}` | `{nmse_p90:.9f}` | `{count}` |".format(
                split=split,
                **reward,
            )
        )
    lines.extend(
        [
            "",
            "## Qwen Reference QC",
            "",
            "| Metric | Qwen released QC |",
            "|---|---:|",
            f"| Correct mean MSE | `{qwen.get('correct_mean_mse')}` |",
            f"| Correct median MSE | `{qwen.get('correct_median_mse')}` |",
            f"| Correct mean cosine | `{qwen.get('correct_mean_cosine')}` |",
            f"| Mean-target mean MSE | `{qwen.get('mean_target_mean_mse')}` |",
            f"| Shuffled-text mean MSE | `{qwen.get('shuffled_mean_mse')}` |",
            f"| Random-text mean MSE | `{qwen.get('random_mean_mse')}` |",
            f"| Scientific pass | `{qwen.get('scientific_pass')}` |",
            "",
            "Read: Nano R33 is now comparable to Qwen at the AV+AR SFT contract level, but not yet a Qwen-level claim. Qwen QC uses released Qwen checkpoints and a different activation/model dimension; the right use here is contract sanity and qualitative gap framing, not a direct metric threshold.",
            "",
            "## RL Readiness",
            "",
            "- RL should be a guarded smoke, not a hero promotion.",
            "- Required before launch: preserve the selected SFT checkpoints, keep frozen-AR reward audit outputs, set strict parse/length/KL guardrails, and define rollback/stop criteria.",
            "- Success criterion for any RL smoke: improve heldout AV-generated-text round-trip against the frozen SFT AV checkpoint without degrading parse health or AV real-vs-control specificity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r33-roundtrip", type=Path, required=True)
    parser.add_argument("--r27-roundtrip", type=Path, required=True)
    parser.add_argument("--qwen-report-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--generated-at", default="2026-06-21T16:30:00Z")
    args = parser.parse_args()

    payload = {
        "generated_at": args.generated_at,
        "r33": report_block("r33_component_full_hero", load_json(args.r33_roundtrip)),
        "r27": report_block("r27_baseline_v256_t256", load_json(args.r27_roundtrip)),
        "qwen_qc": extract_qwen_qc(args.qwen_report_md.read_text()),
        "inputs": {
            "r33_roundtrip": str(args.r33_roundtrip),
            "r27_roundtrip": str(args.r27_roundtrip),
            "qwen_report_md": str(args.qwen_report_md),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(payload))
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
