#!/usr/bin/env python3
"""Bounded Nano frozen-AR smoke grid runner.

This runner launches small `nano_ar_frozen_baseline.py` configs sequentially and
aggregates their JSON results. It is intentionally conservative: one process at
a time, explicit max-runs cap, no PEFT/LoRA/AV/RL/serving.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nano_introspection import DEFAULT_MODEL_ID, DEFAULT_OUTPUT_ROOT, utc_timestamp, write_json  # noqa: E402


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in _csv_strings(value)]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in _csv_strings(value)]


def expand_grid(
    *,
    boundaries: list[str],
    max_records: list[int],
    train_fractions: list[float],
    split_strategies: list[str],
    explanation_templates: list[str],
    lrs: list[float],
    max_steps: list[int],
    seeds: list[int],
    max_runs: int,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for boundary, records, train_fraction, split_strategy, template, lr, steps, seed in itertools.product(
        boundaries,
        max_records,
        train_fractions,
        split_strategies,
        explanation_templates,
        lrs,
        max_steps,
        seeds,
    ):
        configs.append(
            {
                "boundaries": boundary,
                "max_records": int(records),
                "train_fraction": float(train_fraction),
                "split_strategy": split_strategy,
                "explanation_template": template,
                "lr": float(lr),
                "max_steps": int(steps),
                "random_seed": int(seed),
            }
        )
        if len(configs) >= max_runs:
            return configs
    return configs


def build_child_command(
    *,
    python_exe: str,
    baseline_script: Path,
    config: dict[str, Any],
    timestamp: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python_exe,
        str(baseline_script),
        "--model-id",
        args.model_id,
        "--device-map",
        args.device_map,
        "--torch-dtype",
        args.torch_dtype,
        "--boundaries",
        config["boundaries"],
        "--prompt-names",
        args.prompt_names,
        "--prompt-max-length",
        str(args.prompt_max_length),
        "--explanation-max-length",
        str(args.explanation_max_length),
        "--source-tau",
        str(args.source_tau),
        "--ar-tau",
        str(args.ar_tau),
        "--max-records",
        str(config["max_records"]),
        "--train-fraction",
        str(config["train_fraction"]),
        "--split-strategy",
        config["split_strategy"],
        "--explanation-template",
        config["explanation_template"],
        "--critic-template",
        args.critic_template,
        "--max-steps",
        str(config["max_steps"]),
        "--lr",
        str(config["lr"]),
        "--weight-decay",
        str(args.weight_decay),
        "--random-seed",
        str(config["random_seed"]),
        "--mse-margin",
        str(args.mse_margin),
        "--cosine-margin",
        str(args.cosine_margin),
        "--min-rri",
        str(args.min_rri),
        "--output-root",
        str(args.output_root),
        "--timestamp",
        timestamp,
    ]
    if args.model_revision:
        command.extend(["--model-revision", args.model_revision])
    if args.tokenizer_revision:
        command.extend(["--tokenizer-revision", args.tokenizer_revision])
    if args.attn_implementation:
        command.extend(["--attn-implementation", args.attn_implementation])
    if args.local_files_only:
        command.append("--local-files-only")
    if not args.trust_remote_code:
        command.append("--no-trust-remote-code")
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-names", default="raw,reasoning_off_chat,av_marker,ar_critic")
    parser.add_argument("--prompt-max-length", type=int, default=256)
    parser.add_argument("--explanation-max-length", type=int, default=128)
    parser.add_argument("--source-tau", type=int, default=-1)
    parser.add_argument("--ar-tau", type=int, default=-1)
    parser.add_argument("--boundaries-grid", type=_csv_strings, default=["R_34", "R_27"])
    parser.add_argument("--max-records-grid", type=_csv_ints, default=[8])
    parser.add_argument("--train-fractions-grid", type=_csv_floats, default=[0.5])
    parser.add_argument("--split-strategies-grid", type=_csv_strings, default=["alternating"])
    parser.add_argument("--explanation-templates-grid", type=_csv_strings, default=["generic", "prompt_label"])
    parser.add_argument("--lrs-grid", type=_csv_floats, default=[2e-5, 5e-5])
    parser.add_argument("--max-steps-grid", type=_csv_ints, default=[50])
    parser.add_argument("--seeds-grid", type=_csv_ints, default=[1234])
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--critic-template", default="Summary of the following text: <text>{explanation}</text> <summary>")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--mse-margin", type=float, default=0.05)
    parser.add_argument("--cosine-margin", type=float, default=0.02)
    parser.add_argument("--min-rri", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    grid_timestamp = args.timestamp or f"ar-smoke-grid-{utc_timestamp()}"
    grid_dir = args.output_root / grid_timestamp
    grid_dir.mkdir(parents=True, exist_ok=True)
    baseline_script = SCRIPT_DIR / "nano_ar_frozen_baseline.py"
    configs = expand_grid(
        boundaries=args.boundaries_grid,
        max_records=args.max_records_grid,
        train_fractions=args.train_fractions_grid,
        split_strategies=args.split_strategies_grid,
        explanation_templates=args.explanation_templates_grid,
        lrs=args.lrs_grid,
        max_steps=args.max_steps_grid,
        seeds=args.seeds_grid,
        max_runs=args.max_runs,
    )
    aggregate: dict[str, Any] = {
        "schema_version": "nano_ar_smoke_grid.v1",
        "run_dir": str(grid_dir),
        "max_runs": args.max_runs,
        "dry_run": args.dry_run,
        "runs": [],
    }

    for idx, config in enumerate(configs):
        child_timestamp = f"{grid_timestamp}-run{idx:03d}"
        child_dir = args.output_root / child_timestamp
        command = build_child_command(
            python_exe=args.python,
            baseline_script=baseline_script,
            config=config,
            timestamp=child_timestamp,
            args=args,
        )
        record: dict[str, Any] = {
            "index": idx,
            "config": config,
            "timestamp": child_timestamp,
            "command": command,
            "result_path": str(child_dir / "ar_frozen_baseline.json"),
        }
        if not args.dry_run:
            completed = subprocess.run(command, check=False)
            record["returncode"] = completed.returncode
            result_path = child_dir / "ar_frozen_baseline.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                record["passed"] = result.get("passed")
                record["scientific_passed"] = result.get("scientific_passed")
                record["heldout_controls"] = result.get("eval", {}).get("heldout_controls")
                record["train_after"] = result.get("training", {}).get("train_after")
                record["blockers"] = result.get("blockers")
            else:
                record["blockers"] = [{"kind": "missing_output", "path": str(result_path)}]
        aggregate["runs"].append(record)
        write_json(grid_dir / "ar_smoke_grid.json", aggregate)

    print(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"\nwrote {grid_dir / 'ar_smoke_grid.json'}")
    if args.dry_run:
        return 0
    return 0 if any(run.get("scientific_passed") for run in aggregate["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
