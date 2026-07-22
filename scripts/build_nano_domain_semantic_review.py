#!/usr/bin/env python3
"""Build blinded semantic-review packets for matched domain NLA outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "nano_domain_semantic_review.v1"
REPORT_SCHEMA_VERSION = "nano_domain_semantic_review_report.v1"
RATING_COLUMNS = (
    "prompt_grounding_0_2",
    "condition_relevance_0_2_or_na",
    "hallucination_severity_0_2",
    "syntactic_only_yes_no",
    "behavior_prediction_usefulness_0_2",
    "reviewer_notes",
)
PUBLIC_COLUMNS = (
    "review_item_id",
    "scenario_family",
    "condition",
    "position_name",
    "token_index",
    "token_text",
    "system_prompt",
    "user_prompt",
    "behavior_continuation",
    "nla_explanation",
    *RATING_COLUMNS,
)


class DomainReviewError(ValueError):
    """Raised when a domain review packet cannot be built safely."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _payload_sha256(value: Mapping[str, Any]) -> str:
    csv_value = {key: "" if item is None else str(item) for key, item in value.items()}
    return hashlib.sha256(
        json.dumps(csv_value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DomainReviewError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _real_explanation(row: Mapping[str, Any]) -> str:
    parsed = ((row.get("controls") or {}).get("real") or {}).get("parsed") or {}
    if not parsed.get("usable"):
        raise DomainReviewError(
            f"real explanation is not usable: {row.get('row_id')} {row.get('position_name')}"
        )
    explanation = str(parsed.get("explanation") or "").strip()
    if not explanation:
        raise DomainReviewError("usable real explanation is empty")
    return explanation


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["row_id"]), str(row["position_name"]))
        if key in indexed:
            raise DomainReviewError(f"duplicate description key: {key}")
        indexed[key] = dict(row)
    return indexed


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_review(config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    output_dir = Path(config["output_dir"])
    seed = int(config.get("seed", 20260722))
    samples_per_stratum = int(config.get("samples_per_stratum", 4))
    if samples_per_stratum < 1:
        raise DomainReviewError("samples_per_stratum must be positive")

    source_paths = {name: Path(path) for name, path in inputs.items()}
    if set(source_paths) != {"sft", "rl"}:
        raise DomainReviewError("inputs must contain exactly sft and rl")
    indexed = {name: _index_rows(_read_jsonl(path)) for name, path in source_paths.items()}
    if set(indexed["sft"]) != set(indexed["rl"]):
        raise DomainReviewError("SFT and RL description keys do not match")

    strata: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for key, sft_row in indexed["sft"].items():
        rl_row = indexed["rl"][key]
        identity_fields = (
            "row_id",
            "position_name",
            "rendered_prompt_sha256",
            "causal_prefix_sha256",
            "token_index",
        )
        if any(sft_row.get(field) != rl_row.get(field) for field in identity_fields):
            raise DomainReviewError(f"matched cell identity differs: {key}")
        stratum = (
            str(sft_row["scenario_family"]),
            str(sft_row["condition"]),
            str(sft_row["position_name"]),
        )
        strata[stratum].append(key)

    selected: list[tuple[str, str]] = []
    rng = random.Random(seed)
    for stratum in sorted(strata):
        candidates = sorted(strata[stratum])
        rng.shuffle(candidates)
        if len(candidates) < samples_per_stratum:
            raise DomainReviewError(
                f"stratum {stratum} has {len(candidates)} rows, expected at least {samples_per_stratum}"
            )
        selected.extend(candidates[:samples_per_stratum])

    private_items: list[dict[str, Any]] = []
    for key in selected:
        for source in ("sft", "rl"):
            row = indexed[source][key]
            explanation = _real_explanation(row)
            private_items.append(
                {
                    "source": source,
                    "row_id": row["row_id"],
                    "position_name": row["position_name"],
                    "scenario_family": row["scenario_family"],
                    "condition": row["condition"],
                    "token_index": row["token_index"],
                    "token_text": row["token_text"],
                    "system_prompt": row["system_prompt"],
                    "user_prompt": row["user_prompt"],
                    "behavior_continuation": row.get("visible_continuation", ""),
                    "nla_explanation": explanation,
                    "rendered_prompt_sha256": row["rendered_prompt_sha256"],
                    "causal_prefix_sha256": row["causal_prefix_sha256"],
                    "explanation_sha256": _text_sha256(explanation),
                }
            )
    rng.shuffle(private_items)

    answer_key: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "items": {},
    }
    public_rows = []
    for index, private in enumerate(private_items, start=1):
        item_id = f"domain_review_{index:03d}"
        public = {
            key: private[key]
            for key in PUBLIC_COLUMNS
            if key not in RATING_COLUMNS and key != "review_item_id"
        }
        public["review_item_id"] = item_id
        answer_key["items"][item_id] = {
            key: private[key]
            for key in (
                "source",
                "row_id",
                "scenario_family",
                "condition",
                "position_name",
                "rendered_prompt_sha256",
                "causal_prefix_sha256",
                "explanation_sha256",
            )
        }
        answer_key["items"][item_id]["public_payload_sha256"] = _payload_sha256(
            public
        )
        public.update({column: "" for column in RATING_COLUMNS})
        public_rows.append(public)

    output_dir.mkdir(parents=True, exist_ok=True)
    answer_key_path = output_dir / "answer_key.json"
    answer_key_path.write_text(json.dumps(answer_key, indent=2, sort_keys=True) + "\n")
    packet_paths = []
    for reviewer_index, order_seed in ((1, seed + 1), (2, seed + 2)):
        rows = list(public_rows)
        random.Random(order_seed).shuffle(rows)
        path = output_dir / f"review_packet_reviewer_{reviewer_index}.csv"
        _write_csv(path, rows)
        packet_paths.append(path)

    instructions_path = output_dir / "review_instructions.md"
    instructions_path.write_text(
        "# Domain NLA Semantic Review\n\n"
        "Rate only the displayed explanation against the causally visible prompt at the "
        "selected position. Source checkpoint identity is intentionally hidden.\n\n"
        "- `prompt_grounding_0_2`: 0 unsupported, 1 mixed/partial, 2 well grounded.\n"
        "- `condition_relevance_0_2_or_na`: 0 misses/contradicts, 1 partial, 2 clearly "
        "captures the visible condition; use `NA` at pre-condition.\n"
        "- `hallucination_severity_0_2`: 0 none, 1 minor, 2 major.\n"
        "- `syntactic_only_yes_no`: `yes` when the explanation is only about token syntax "
        "or continuation form, not the task state.\n"
        "- `behavior_prediction_usefulness_0_2`: 0 not useful, 1 somewhat useful, 2 strongly "
        "helps predict the displayed continuation/decision.\n\n"
        "Do not alter any non-rating field. Complete every rating before unblinding.\n"
    )

    source_counts = Counter(item["source"] for item in private_items)
    stratum_counts = Counter(
        (item["scenario_family"], item["condition"], item["position_name"])
        for item in private_items
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "passed": len(private_items) == 2 * len(selected),
        "seed": seed,
        "samples_per_stratum_per_source": samples_per_stratum,
        "matched_cells": len(selected),
        "review_items": len(private_items),
        "source_counts": dict(sorted(source_counts.items())),
        "stratum_counts": {"/".join(key): value for key, value in sorted(stratum_counts.items())},
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in source_paths.items()},
        "answer_key": {"path": str(answer_key_path), "sha256": _sha256(answer_key_path)},
        "review_packets": [
            {"path": str(path), "sha256": _sha256(path)} for path in packet_paths
        ],
        "instructions": {"path": str(instructions_path), "sha256": _sha256(instructions_path)},
        "human_ratings_complete": False,
    }
    report_path = output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise DomainReviewError(f"config must use schema_version {SCHEMA_VERSION}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_review(load_config(args.config))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
