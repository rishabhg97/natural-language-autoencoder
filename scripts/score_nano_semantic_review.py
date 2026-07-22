#!/usr/bin/env python3
"""Validate, unblind, and summarize Nano semantic-transform reviews."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "nano_semantic_meaning_review_score.v1"
REPORT_SCHEMA_VERSION = "nano_semantic_meaning_review_score_report.v1"


class SemanticReviewScoreError(ValueError):
    """Raised when blinded semantic ratings are incomplete or inconsistent."""


def _parse_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise SemanticReviewScoreError(f"{label} must be true or false")


def _parse_int(value: Any, *, minimum: int, maximum: int, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SemanticReviewScoreError(f"{label} must be an integer") from exc
    if not minimum <= result <= maximum:
        raise SemanticReviewScoreError(
            f"{label} must be between {minimum} and {maximum}"
        )
    return result


def _load_json_review(path: Path) -> list[dict[str, Any]]:
    packet = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for raw in packet.get("rows") or []:
        rating = raw.get("rating") or {}
        rows.append(
            {
                "review_id": raw.get("review_id"),
                "original_explanation": raw.get("original_explanation"),
                "transformed_explanation": raw.get("transformed_explanation"),
                **rating,
            }
        )
    return rows


def _load_csv_review(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    return [
        {
            "review_id": row.get("review_id"),
            "original_explanation": row.get("original_explanation"),
            "transformed_explanation": row.get("transformed_explanation"),
            "meaning_preservation": row.get("meaning_preservation_1_to_5"),
            "omission_severity": row.get("omission_severity_0_to_3"),
            "unsupported_addition_severity": row.get(
                "unsupported_addition_severity_0_to_3"
            ),
            "contradiction_present": row.get("contradiction_present_true_false"),
            "fluent_and_interpretable": row.get(
                "fluent_and_interpretable_true_false"
            ),
            "notes": row.get("notes") or "",
        }
        for row in raw_rows
    ]


def _load_review(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return _load_csv_review(path)
    return _load_json_review(path)


def _quadratic_weighted_kappa(left: Sequence[int], right: Sequence[int]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    categories = range(1, 6)
    observed = Counter(zip(left, right))
    left_counts = Counter(left)
    right_counts = Counter(right)
    denominator = float((5 - 1) ** 2)
    observed_disagreement = sum(
        ((a - b) ** 2 / denominator) * count for (a, b), count in observed.items()
    ) / len(left)
    expected_disagreement = sum(
        ((a - b) ** 2 / denominator)
        * left_counts[a]
        * right_counts[b]
        / (len(left) ** 2)
        for a in categories
        for b in categories
    )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1.0 - observed_disagreement / expected_disagreement


def _validate_review_rows(
    reviewer_id: str,
    rows: Sequence[Mapping[str, Any]],
    answers: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in rows:
        review_id = str(raw.get("review_id") or "")
        if not review_id or review_id in indexed or review_id not in answers:
            raise SemanticReviewScoreError(
                f"{reviewer_id} has invalid or duplicate review_id {review_id!r}"
            )
        answer = answers[review_id]
        original = str(raw.get("original_explanation") or "")
        transformed = str(raw.get("transformed_explanation") or "")
        import hashlib

        if hashlib.sha256(original.encode()).hexdigest() != answer["original_sha256"]:
            raise SemanticReviewScoreError(f"{reviewer_id}/{review_id} original text changed")
        if hashlib.sha256(transformed.encode()).hexdigest() != answer["transformed_sha256"]:
            raise SemanticReviewScoreError(
                f"{reviewer_id}/{review_id} transformed text changed"
            )
        indexed[review_id] = {
            "meaning_preservation": _parse_int(
                raw.get("meaning_preservation"),
                minimum=1,
                maximum=5,
                label=f"{reviewer_id}/{review_id}/meaning_preservation",
            ),
            "omission_severity": _parse_int(
                raw.get("omission_severity"),
                minimum=0,
                maximum=3,
                label=f"{reviewer_id}/{review_id}/omission_severity",
            ),
            "unsupported_addition_severity": _parse_int(
                raw.get("unsupported_addition_severity"),
                minimum=0,
                maximum=3,
                label=f"{reviewer_id}/{review_id}/unsupported_addition_severity",
            ),
            "contradiction_present": _parse_bool(
                raw.get("contradiction_present"),
                label=f"{reviewer_id}/{review_id}/contradiction_present",
            ),
            "fluent_and_interpretable": _parse_bool(
                raw.get("fluent_and_interpretable"),
                label=f"{reviewer_id}/{review_id}/fluent_and_interpretable",
            ),
            "notes": str(raw.get("notes") or ""),
            **answer,
        }
    if set(indexed) != set(answers):
        raise SemanticReviewScoreError(
            f"{reviewer_id} completed {len(indexed)}/{len(answers)} rows"
        )
    return indexed


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "ratings": len(rows),
        "mean_meaning_preservation": mean(
            float(row["meaning_preservation"]) for row in rows
        ),
        "meaning_preservation_at_least_4_fraction": sum(
            int(row["meaning_preservation"]) >= 4 for row in rows
        )
        / len(rows),
        "mean_omission_severity": mean(float(row["omission_severity"]) for row in rows),
        "mean_unsupported_addition_severity": mean(
            float(row["unsupported_addition_severity"]) for row in rows
        ),
        "contradiction_fraction": sum(bool(row["contradiction_present"]) for row in rows)
        / len(rows),
        "fluent_fraction": sum(bool(row["fluent_and_interpretable"]) for row in rows)
        / len(rows),
    }


def score(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SemanticReviewScoreError(f"schema_version must be {SCHEMA_VERSION}")
    paths = config.get("paths") or {}
    protocol = config.get("protocol") or {}
    answer_key = json.loads(Path(paths["answer_key"]).read_text())
    review_paths = {name: Path(path) for name, path in (paths.get("reviews") or {}).items()}
    if len(review_paths) < int(protocol.get("minimum_reviewers", 2)):
        raise SemanticReviewScoreError("not enough reviewer files")

    validated: dict[str, dict[str, dict[str, Any]]] = {}
    for reviewer_id, path in sorted(review_paths.items()):
        reviewer_answers = (answer_key.get("reviewers") or {}).get(reviewer_id)
        if reviewer_answers is None:
            raise SemanticReviewScoreError(f"answer key has no reviewer {reviewer_id}")
        validated[reviewer_id] = _validate_review_rows(
            reviewer_id,
            _load_review(path),
            reviewer_answers["answers"],
        )

    all_rows = [row for reviewer in validated.values() for row in reviewer.values()]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_transform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_transform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_source[str(row["source"])].append(row)
        by_transform[str(row["transform"])].append(row)
        by_source_transform[f"{row['source']}/{row['transform']}"].append(row)

    reviewer_ids = sorted(validated)
    agreement: dict[str, Any] = {}
    for left_index, left_id in enumerate(reviewer_ids):
        for right_id in reviewer_ids[left_index + 1 :]:
            left_by_identity = {
                (row["source"], row["transform"], row["row_key"]): row
                for row in validated[left_id].values()
            }
            right_by_identity = {
                (row["source"], row["transform"], row["row_key"]): row
                for row in validated[right_id].values()
            }
            identities = sorted(set(left_by_identity) & set(right_by_identity))
            if not identities:
                raise SemanticReviewScoreError(
                    f"reviewers {left_id} and {right_id} have no common rows"
                )
            left_scores = [left_by_identity[key]["meaning_preservation"] for key in identities]
            right_scores = [right_by_identity[key]["meaning_preservation"] for key in identities]
            agreement[f"{left_id}/{right_id}"] = {
                "common_rows": len(identities),
                "meaning_preservation_quadratic_weighted_kappa": _quadratic_weighted_kappa(
                    left_scores, right_scores
                ),
                "contradiction_exact_agreement": sum(
                    left_by_identity[key]["contradiction_present"]
                    == right_by_identity[key]["contradiction_present"]
                    for key in identities
                )
                / len(identities),
                "fluent_exact_agreement": sum(
                    left_by_identity[key]["fluent_and_interpretable"]
                    == right_by_identity[key]["fluent_and_interpretable"]
                    for key in identities
                )
                / len(identities),
            }

    thresholds = protocol.get("thresholds") or {}
    gated_transforms = [str(value) for value in protocol.get("gated_transforms") or []]
    transform_summaries = {
        key: _summary(rows) for key, rows in sorted(by_transform.items())
    }
    gates: dict[str, Any] = {}
    for transform in gated_transforms:
        summary = transform_summaries.get(transform)
        if summary is None:
            raise SemanticReviewScoreError(f"gated transform is absent: {transform}")
        checks = {
            "mean_meaning_preservation": summary["mean_meaning_preservation"]
            >= float(thresholds.get("minimum_meaning_preservation", 4.0)),
            "contradiction_fraction": summary["contradiction_fraction"]
            <= float(thresholds.get("maximum_contradiction_fraction", 0.05)),
            "mean_unsupported_addition_severity": summary[
                "mean_unsupported_addition_severity"
            ]
            <= float(thresholds.get("maximum_unsupported_addition_severity", 0.5)),
            "fluent_fraction": summary["fluent_fraction"]
            >= float(thresholds.get("minimum_fluent_fraction", 0.90)),
        }
        gates[transform] = {"passed": all(checks.values()), "checks": checks}

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "passed": True,
        "review_complete": True,
        "meaning_preservation_gate_passed": bool(gates)
        and all(value["passed"] for value in gates.values()),
        "reviewers": reviewer_ids,
        "unique_review_items": len(
            {
                (row["source"], row["transform"], row["row_key"])
                for row in all_rows
            }
        ),
        "ratings": len(all_rows),
        "overall": _summary(all_rows),
        "by_source": {key: _summary(rows) for key, rows in sorted(by_source.items())},
        "by_transform": transform_summaries,
        "by_source_transform": {
            key: _summary(rows) for key, rows in sorted(by_source_transform.items())
        },
        "agreement": agreement,
        "gates": gates,
        "thresholds": thresholds,
        "notes": [
            "Two-sentence summaries are reported as information-removal controls and are not gated as semantic invariance transforms.",
            "A complete review validates the packet protocol; the separate meaning_preservation_gate_passed field reports transform quality.",
        ],
    }
    output_path = Path(paths["output_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    result = score(yaml.safe_load(args.config.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
