#!/usr/bin/env python3
"""Build a matched SFT-versus-RL domain-canary comparison artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "nano_domain_canary_comparison.v1"


class DomainComparisonError(ValueError):
    """Raised when matched canary evidence is incompatible."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _description_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_jsonl(path)
    indexed = {(row["row_id"], row["position_name"]): row for row in rows}
    if len(indexed) != len(rows):
        raise DomainComparisonError(f"duplicate description rows in {path}")
    return indexed


def _token_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_counts = []
    by_position: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        parsed = row["controls"]["real"]["parsed"]
        if not parsed.get("usable"):
            raise DomainComparisonError("comparison includes unusable real explanation")
        count = int(parsed["token_count"])
        all_counts.append(count)
        by_position[row["position_name"]].append(count)
    return {
        "mean": mean(all_counts),
        "median": median(all_counts),
        "minimum": min(all_counts),
        "maximum": max(all_counts),
        "mean_by_position": {
            position: mean(values) for position, values in sorted(by_position.items())
        },
    }


def compare(config: Mapping[str, Any]) -> dict[str, Any]:
    roots = {name: Path(path) for name, path in config["inputs"].items()}
    if set(roots) != {"sft", "rl"}:
        raise DomainComparisonError("inputs must contain exactly sft and rl roots")
    descriptions = {
        name: root / "model_outputs" / "nla_descriptions.jsonl"
        for name, root in roots.items()
    }
    indexed = {name: _description_index(path) for name, path in descriptions.items()}
    if set(indexed["sft"]) != set(indexed["rl"]):
        raise DomainComparisonError("SFT and RL description keys differ")
    identity_fields = (
        "rendered_prompt_sha256",
        "causal_prefix_sha256",
        "token_index",
        "token_id",
    )
    for key in indexed["sft"]:
        if any(
            indexed["sft"][key].get(field) != indexed["rl"][key].get(field)
            for field in identity_fields
        ):
            raise DomainComparisonError(f"matched cell identity differs: {key}")

    analyses = {
        name: json.loads((root / "analysis" / "domain_canary_report.json").read_text())
        for name, root in roots.items()
    }
    behaviors = {
        name: json.loads((root / "model_outputs" / "behavior_report.json").read_text())
        for name, root in roots.items()
    }
    paired_effects = {}
    for key in sorted(analyses["sft"]["paired_effects"]):
        sft = analyses["sft"]["paired_effects"][key]
        rl = analyses["rl"]["paired_effects"][key]
        paired_effects[key] = {
            "pairs": sft["pairs"],
            "sft_positive_hit_rate": sft["positive_hit_rate"],
            "sft_negative_hit_rate": sft["negative_hit_rate"],
            "rl_positive_hit_rate": rl["positive_hit_rate"],
            "rl_negative_hit_rate": rl["negative_hit_rate"],
            "rl_minus_sft_positive_hit_rate": (
                rl["positive_hit_rate"] - sft["positive_hit_rate"]
            ),
            "rl_minus_sft_negative_hit_rate": (
                rl["negative_hit_rate"] - sft["negative_hit_rate"]
            ),
        }

    exact_matches = sum(
        indexed["sft"][key]["controls"]["real"]["parsed"]["explanation"]
        == indexed["rl"][key]["controls"]["real"]["parsed"]["explanation"]
        for key in indexed["sft"]
    )
    token_summaries = {
        name: _token_summary(list(rows.values())) for name, rows in indexed.items()
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "claim_scope": "exploratory_matched_checkpoint_comparison",
        "matched_cells": len(indexed["sft"]),
        "matched_identity_fields": list(identity_fields),
        "exact_sft_rl_explanation_matches": exact_matches,
        "real_explanation_tokens": token_summaries,
        "rl_to_sft_mean_token_ratio": (
            token_summaries["rl"]["mean"] / token_summaries["sft"]["mean"]
        ),
        "paired_lexicon_effects": paired_effects,
        "behavior_repeat": {
            name: {
                "decision_parse_rate": value["decision_parse_rate"],
                "decision_accuracy": value["decision_accuracy"],
                "sha256": analyses[name]["behavior_sha256"],
            }
            for name, value in behaviors.items()
        },
        "inputs": {
            name: {
                "root": str(roots[name]),
                "description_sha256": _sha256(descriptions[name]),
                "config_sha256": analyses[name]["config_sha256"],
            }
            for name in sorted(roots)
        },
        "limitations": [
            "Lexicon hits are descriptive and not a held-out semantic classifier.",
            "RL explanations are longer, so raw lexicon hit differences are length-confounded.",
            "Behavior generation is checkpoint-independent here; repeat differences are not an AV effect.",
            "Human semantic ratings remain blinded and pending.",
        ],
    }
    output = Path(config["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        print(f"ERROR: config must use schema_version {SCHEMA_VERSION}")
        return 2
    try:
        report = compare(config)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
