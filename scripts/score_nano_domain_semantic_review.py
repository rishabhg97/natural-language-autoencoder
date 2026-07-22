#!/usr/bin/env python3
"""Validate, unblind, and summarize matched domain NLA semantic reviews."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml

from build_nano_domain_semantic_review import PUBLIC_COLUMNS, RATING_COLUMNS


SCHEMA_VERSION = "nano_domain_semantic_review_score.v1"
REPORT_SCHEMA_VERSION = "nano_domain_semantic_review_score_report.v1"


class DomainReviewScoreError(ValueError):
    """Raised when ratings or blinded packet payloads are invalid."""


def _payload_sha256(row: Mapping[str, Any]) -> str:
    payload = {
        key: row[key]
        for key in PUBLIC_COLUMNS
        if key not in RATING_COLUMNS
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rating_int(row: Mapping[str, str], field: str) -> int:
    raw = row[field].strip()
    if raw not in {"0", "1", "2"}:
        raise DomainReviewScoreError(
            f"{row.get('review_item_id')} {field} must be 0, 1, or 2"
        )
    return int(raw)


def _load_packet(
    path: Path, answer_key: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = set(answer_key["items"])
    ids = [row.get("review_item_id", "") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise DomainReviewScoreError(f"review item IDs changed in {path}")
    parsed = {}
    for row in rows:
        item_id = row["review_item_id"]
        private = answer_key["items"][item_id]
        if _payload_sha256(row) != private["public_payload_sha256"]:
            raise DomainReviewScoreError(f"non-rating payload changed: {item_id}")
        condition_raw = row["condition_relevance_0_2_or_na"].strip().upper()
        if condition_raw == "NA":
            if row["position_name"] != "pre_condition":
                raise DomainReviewScoreError(
                    f"condition relevance may be NA only at pre_condition: {item_id}"
                )
            condition_rating: int | None = None
        else:
            condition_rating = _rating_int(
                row, "condition_relevance_0_2_or_na"
            )
        syntax = row["syntactic_only_yes_no"].strip().lower()
        if syntax not in {"yes", "no"}:
            raise DomainReviewScoreError(
                f"{item_id} syntactic_only_yes_no must be yes or no"
            )
        parsed[item_id] = {
            **private,
            "prompt_grounding": _rating_int(row, "prompt_grounding_0_2"),
            "condition_relevance": condition_rating,
            "hallucination_severity": _rating_int(
                row, "hallucination_severity_0_2"
            ),
            "syntactic_only": syntax == "yes",
            "behavior_prediction_usefulness": _rating_int(
                row, "behavior_prediction_usefulness_0_2"
            ),
            "reviewer_notes": row["reviewer_notes"].strip(),
        }
    return parsed


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    condition = [row["condition_relevance"] for row in rows]
    condition = [value for value in condition if value is not None]
    return {
        "items": len(rows),
        "prompt_grounding_mean": mean(row["prompt_grounding"] for row in rows),
        "condition_relevance_mean": mean(condition) if condition else None,
        "hallucination_severity_mean": mean(
            row["hallucination_severity"] for row in rows
        ),
        "syntactic_only_fraction": mean(row["syntactic_only"] for row in rows),
        "behavior_prediction_usefulness_mean": mean(
            row["behavior_prediction_usefulness"] for row in rows
        ),
    }


def score_reviews(config: Mapping[str, Any]) -> dict[str, Any]:
    answer_key = json.loads(Path(config["answer_key"]).read_text())
    packets = {
        name: _load_packet(Path(path), answer_key)
        for name, path in config["review_packets"].items()
    }
    if len(packets) < 2:
        raise DomainReviewScoreError("at least two completed review packets are required")

    reviewer_summaries = {}
    for reviewer, values in packets.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in values.values():
            grouped[f"source/{row['source']}"] .append(row)
            grouped[f"source/{row['source']}/position/{row['position_name']}"] .append(row)
            grouped[
                f"source/{row['source']}/family/{row['scenario_family']}"
            ].append(row)
        reviewer_summaries[reviewer] = {
            key: _summarize(rows) for key, rows in sorted(grouped.items())
        }

    reviewer_names = sorted(packets)
    first, second = packets[reviewer_names[0]], packets[reviewer_names[1]]
    rating_fields = (
        "prompt_grounding",
        "condition_relevance",
        "hallucination_severity",
        "syntactic_only",
        "behavior_prediction_usefulness",
    )
    agreement = {}
    for field in rating_fields:
        comparable = [
            item_id
            for item_id in first
            if first[item_id][field] is not None and second[item_id][field] is not None
        ]
        agreement[field] = {
            "items": len(comparable),
            "exact_agreement_fraction": mean(
                first[item_id][field] == second[item_id][field]
                for item_id in comparable
            ),
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "passed": True,
        "human_ratings_complete": True,
        "reviewers": reviewer_names,
        "items_per_reviewer": len(first),
        "reviewer_summaries": reviewer_summaries,
        "inter_reviewer_agreement": agreement,
    }
    output_path = Path(config["output_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise DomainReviewScoreError(f"config must use schema_version {SCHEMA_VERSION}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        report = score_reviews(load_config(args.config))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
