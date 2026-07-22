#!/usr/bin/env python3
"""Validate and score completed blinded Nano NLA qualitative reviews."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import yaml

from build_nano_blinded_review_packets import (
    ANSWER_SCHEMA_VERSION,
    PACKET_SCHEMA_VERSION,
    _text_sha256,
)


SCHEMA_VERSION = "nano_blinded_qualitative_review_score.v1"


class ReviewScoreError(ValueError):
    """Raised when a completed review is missing, altered, or invalid."""


def _cohen_kappa(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[value] * right_counts[value] for value in set(left_counts) | set(right_counts)
    ) / (len(left) ** 2)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def _linear_weighted_kappa(left: Sequence[int], right: Sequence[int], *, levels: int = 5) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed_disagreement = mean(abs(a - b) / (levels - 1) for a, b in zip(left, right))
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected_disagreement = sum(
        (left_counts[a] / len(left))
        * (right_counts[b] / len(right))
        * abs(a - b)
        / (levels - 1)
        for a in range(1, levels + 1)
        for b in range(1, levels + 1)
    )
    if math.isclose(expected_disagreement, 0.0):
        return 1.0 if math.isclose(observed_disagreement, 0.0) else None
    return 1.0 - observed_disagreement / expected_disagreement


def _validate_rating(rating: Mapping[str, Any], dimensions: Sequence[str], identity: str) -> None:
    scores = rating.get("scores") or {}
    if set(scores) != set(dimensions):
        raise ReviewScoreError(f"{identity} score dimensions do not match protocol")
    for dimension, value in scores.items():
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
            raise ReviewScoreError(f"{identity} {dimension} must be an integer in [1, 5]")
    for field in (
        "unsupported_specific_claim",
        "privacy_or_sensitive_content",
        "release_appropriate",
    ):
        if not isinstance(rating.get(field), bool):
            raise ReviewScoreError(f"{identity} {field} must be boolean")


def _load_reviewer(
    packet_path: Path,
    reviewer_answers: Mapping[str, Any],
    dimensions: Sequence[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    packet = json.loads(packet_path.read_text())
    if packet.get("schema_version") != PACKET_SCHEMA_VERSION:
        raise ReviewScoreError(f"invalid packet schema: {packet_path}")
    reviewer_id = str(packet["reviewer_id"])
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in packet.get("rows") or []:
        review_id = str(row["review_id"])
        answer = reviewer_answers.get(review_id)
        if not answer:
            raise ReviewScoreError(f"unknown review_id {review_id!r} for {reviewer_id}")
        if _text_sha256(str(row["text_a"])) != answer["text_a_sha256"]:
            raise ReviewScoreError(f"text A changed for {review_id}")
        if _text_sha256(str(row["text_b"])) != answer["text_b_sha256"]:
            raise ReviewScoreError(f"text B changed for {review_id}")
        _validate_rating(row["ratings_a"], dimensions, f"{review_id}:A")
        _validate_rating(row["ratings_b"], dimensions, f"{review_id}:B")
        preference = row.get("preference")
        if preference not in {"A", "B", "tie"}:
            raise ReviewScoreError(f"{review_id} preference must be A, B, or tie")
        by_role = {
            answer["role_a"]: row["ratings_a"],
            answer["role_b"]: row["ratings_b"],
        }
        preferred_role = "tie" if preference == "tie" else answer[f"role_{preference.lower()}"]
        identity = (str(answer["split"]), int(answer["row_index"]))
        if identity in rows:
            raise ReviewScoreError(f"duplicate completed identity {identity!r}")
        rows[identity] = {
            "reviewer_id": reviewer_id,
            "doc_id": answer.get("doc_id"),
            "candidate": by_role["candidate"],
            "reference": by_role["reference"],
            "preferred_role": preferred_role,
        }
    if set(rows) != {
        (str(answer["split"]), int(answer["row_index"]))
        for answer in reviewer_answers.values()
    }:
        raise ReviewScoreError(f"completed packet is missing rows for {reviewer_id}")
    return rows


def _role_summary(rows: Iterable[Mapping[str, Any]], role: str, dimensions: Sequence[str]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "row_ratings": len(rows),
        "mean_scores": {
            dimension: mean(row[role]["scores"][dimension] for row in rows)
            for dimension in dimensions
        },
        "unsupported_specific_claim_fraction": mean(
            row[role]["unsupported_specific_claim"] for row in rows
        ),
        "privacy_or_sensitive_content_fraction": mean(
            row[role]["privacy_or_sensitive_content"] for row in rows
        ),
        "release_appropriate_fraction": mean(row[role]["release_appropriate"] for row in rows),
    }


def score_reviews(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ReviewScoreError(f"schema_version must be {SCHEMA_VERSION!r}")
    paths = config.get("paths") or {}
    thresholds = config.get("thresholds") or {}
    answer_key = json.loads(Path(paths["answer_key"]).read_text())
    if answer_key.get("schema_version") != ANSWER_SCHEMA_VERSION:
        raise ReviewScoreError("invalid answer-key schema")
    dimensions = list(answer_key["dimensions"])
    completed = {str(key): Path(value) for key, value in (paths.get("completed_packets") or {}).items()}
    if set(completed) != set(answer_key["reviewers"]):
        raise ReviewScoreError("completed_packets must contain every registered reviewer")

    reviewer_rows = {
        reviewer_id: _load_reviewer(
            path,
            answer_key["reviewers"][reviewer_id]["answers"],
            dimensions,
        )
        for reviewer_id, path in completed.items()
    }
    identities = set(next(iter(reviewer_rows.values())))
    if any(set(rows) != identities for rows in reviewer_rows.values()):
        raise ReviewScoreError("reviewer row identities do not match")
    all_rows = [row for rows in reviewer_rows.values() for row in rows.values()]

    candidate = _role_summary(all_rows, "candidate", dimensions)
    reference = _role_summary(all_rows, "reference", dimensions)
    preferences = Counter(row["preferred_role"] for row in all_rows)
    candidate["preference_fraction"] = preferences["candidate"] / len(all_rows)
    reference["preference_fraction"] = preferences["reference"] / len(all_rows)
    preference_tie_fraction = preferences["tie"] / len(all_rows)

    reviewer_ids = sorted(reviewer_rows)
    agreement: dict[str, Any] = {}
    for left_index, left_id in enumerate(reviewer_ids):
        for right_id in reviewer_ids[left_index + 1 :]:
            left = reviewer_rows[left_id]
            right = reviewer_rows[right_id]
            pair: dict[str, Any] = {
                "preference_kappa": _cohen_kappa(
                    [left[key]["preferred_role"] for key in sorted(identities)],
                    [right[key]["preferred_role"] for key in sorted(identities)],
                ),
                "candidate_release_appropriate_kappa": _cohen_kappa(
                    [left[key]["candidate"]["release_appropriate"] for key in sorted(identities)],
                    [right[key]["candidate"]["release_appropriate"] for key in sorted(identities)],
                ),
                "candidate_unsupported_claim_kappa": _cohen_kappa(
                    [left[key]["candidate"]["unsupported_specific_claim"] for key in sorted(identities)],
                    [right[key]["candidate"]["unsupported_specific_claim"] for key in sorted(identities)],
                ),
                "candidate_score_weighted_kappa": {
                    dimension: _linear_weighted_kappa(
                        [left[key]["candidate"]["scores"][dimension] for key in sorted(identities)],
                        [right[key]["candidate"]["scores"][dimension] for key in sorted(identities)],
                    )
                    for dimension in dimensions
                },
            }
            agreement[f"{left_id}__{right_id}"] = pair

    checks: dict[str, bool] = {}
    for dimension in dimensions:
        key = f"min_candidate_mean_{dimension}"
        if key in thresholds:
            checks[key] = candidate["mean_scores"][dimension] >= float(thresholds[key])
    if "max_candidate_unsupported_specific_claim_fraction" in thresholds:
        key = "max_candidate_unsupported_specific_claim_fraction"
        checks[key] = candidate["unsupported_specific_claim_fraction"] <= float(thresholds[key])
    if "max_candidate_privacy_or_sensitive_content_fraction" in thresholds:
        key = "max_candidate_privacy_or_sensitive_content_fraction"
        checks[key] = candidate["privacy_or_sensitive_content_fraction"] <= float(thresholds[key])
    if "min_candidate_release_appropriate_fraction" in thresholds:
        key = "min_candidate_release_appropriate_fraction"
        checks[key] = candidate["release_appropriate_fraction"] >= float(thresholds[key])

    return {
        "schema_version": SCHEMA_VERSION,
        "passed": bool(checks) and all(checks.values()),
        "review_complete": True,
        "row_count": len(identities),
        "reviewer_count": len(reviewer_ids),
        "dimensions": dimensions,
        "candidate": candidate,
        "reference": reference,
        "preference_tie_fraction": preference_tie_fraction,
        "agreement": agreement,
        "threshold_checks": checks,
        "claim_boundary": "Human source-grounded ratings; not evidence of activation faithfulness or causal model reasoning.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    report = score_reviews(config)
    output = Path(config["paths"]["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "output_json": str(output)}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
