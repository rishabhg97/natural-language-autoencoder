#!/usr/bin/env python3
"""Build deterministic blinded packets for semantic-transform review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "nano_semantic_meaning_review.v1"
PACKET_SCHEMA_VERSION = "nano_semantic_meaning_review_packet.v1"
ANSWER_SCHEMA_VERSION = "nano_semantic_meaning_review_answer_key.v1"


class SemanticReviewError(ValueError):
    """Raised when a semantic-review packet cannot be built safely."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _text_sha256(value: str) -> str:
    return _sha256_bytes(value.encode())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SemanticReviewError(f"{path}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise SemanticReviewError(f"{path} is empty")
    return rows


def _unwrap_explanation(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("<explanation>"):
        text = text[len("<explanation>") :]
    if text.endswith("</explanation>"):
        text = text[: -len("</explanation>")]
    return text.strip()


def _source_explanation(row: Mapping[str, Any]) -> str:
    controls = row.get("controls") or {}
    real = controls.get("real") or {}
    parsed = real.get("parsed") or {}
    explanation = _unwrap_explanation(parsed.get("explanation"))
    if not explanation:
        raise SemanticReviewError(
            f"missing real parsed explanation for {row.get('split')}:{row.get('row_index')}"
        )
    return explanation


def _source_generation(row: Mapping[str, Any]) -> str:
    controls = row.get("controls") or {}
    real = controls.get("real") or {}
    generated = str(real.get("generated") or "")
    if generated:
        return generated
    return _source_explanation(row)


def _length_boundaries(lengths: Sequence[int], bins: int) -> list[int]:
    if bins < 1:
        raise SemanticReviewError("protocol.length_bins must be positive")
    ordered = sorted(lengths)
    if not ordered:
        raise SemanticReviewError("cannot derive length bins from no rows")
    boundaries: list[int] = []
    for index in range(1, bins):
        offset = min(len(ordered) - 1, math.ceil(index * len(ordered) / bins) - 1)
        boundaries.append(ordered[offset])
    return boundaries


def _length_bin(length: int, boundaries: Sequence[int]) -> int:
    return sum(length > boundary for boundary in boundaries)


def _review_id(seed: int, reviewer_id: str, row_key: str) -> str:
    material = f"{seed}\0{reviewer_id}\0{row_key}".encode()
    return _sha256_bytes(material)[:20]


def _load_candidates(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = config.get("paths") or {}
    sources = paths.get("sources") or {}
    if len(sources) < 2:
        raise SemanticReviewError("paths.sources must contain at least two text sources")

    candidates: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for source_name, source_config in sorted(sources.items()):
        generated_path = Path(source_config["generated_jsonl"])
        generated_rows = _read_jsonl(generated_path)
        generated_by_key: dict[str, dict[str, Any]] = {}
        for row in generated_rows:
            split = str(row.get("split") or "")
            row_index = row.get("row_index")
            row_key = f"{split}:{row_index}"
            if not split or row_index is None or row_key in generated_by_key:
                raise SemanticReviewError(
                    f"invalid or duplicate generated row identity in {generated_path}: {row_key}"
                )
            generated_by_key[row_key] = row

        transform_paths = source_config.get("transforms") or {}
        if not transform_paths:
            raise SemanticReviewError(f"no transforms configured for source {source_name}")
        provenance[source_name] = {
            "generated_jsonl": str(generated_path),
            "generated_sha256": _sha256_file(generated_path),
            "transforms": {},
        }
        for transform_name, raw_path in sorted(transform_paths.items()):
            transform_path = Path(raw_path)
            transform_rows = _read_jsonl(transform_path)
            provenance[source_name]["transforms"][transform_name] = {
                "path": str(transform_path),
                "sha256": _sha256_file(transform_path),
            }
            seen: set[str] = set()
            for transformed in transform_rows:
                row_key = str(transformed.get("row_key") or "")
                if row_key in seen:
                    raise SemanticReviewError(
                        f"duplicate row_key {row_key!r} in {transform_path}"
                    )
                seen.add(row_key)
                source_row = generated_by_key.get(row_key)
                if source_row is None:
                    raise SemanticReviewError(
                        f"transform row {row_key!r} is absent from {generated_path}"
                    )
                original = _source_explanation(source_row)
                expected_source_hash = str(transformed.get("source_sha256") or "")
                raw_source = _source_generation(source_row)
                if expected_source_hash and expected_source_hash != _text_sha256(raw_source):
                    raise SemanticReviewError(
                        f"source hash mismatch for {source_name}/{transform_name}/{row_key}"
                    )
                changed = _unwrap_explanation(transformed.get("transformed_text"))
                if not changed:
                    raise SemanticReviewError(
                        f"empty transform for {source_name}/{transform_name}/{row_key}"
                    )
                candidates.append(
                    {
                        "source": str(source_name),
                        "transform": str(transform_name),
                        "row_key": row_key,
                        "split": str(source_row["split"]),
                        "row_index": int(source_row["row_index"]),
                        "doc_id": source_row.get("doc_id"),
                        "content_family_id": source_row.get("content_family_id"),
                        "original_text": original,
                        "transformed_text": changed,
                        "original_word_count": len(original.split()),
                        "transformed_word_count": len(changed.split()),
                        "prompt_sha256": transformed.get("prompt_sha256"),
                        "transform_model": transformed.get("model"),
                    }
                )
            if seen != set(generated_by_key):
                missing = sorted(set(generated_by_key) - seen)
                raise SemanticReviewError(
                    f"{transform_path} is missing {len(missing)} generated rows"
                )
    return candidates, provenance


def _select_candidates(
    candidates: list[dict[str, Any]], *, sample_size: int, length_bins: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    if sample_size < 1:
        raise SemanticReviewError("protocol.sample_size must be positive")
    families = {str(row.get("content_family_id") or "") for row in candidates}
    families.discard("")
    if len(families) < sample_size:
        raise SemanticReviewError(
            f"need {sample_size} unique content families, found only {len(families)}"
        )

    source_lengths: dict[str, dict[str, int]] = defaultdict(dict)
    for row in candidates:
        source_lengths[str(row["source"])][str(row["row_key"])] = int(
            row["original_word_count"]
        )
    boundaries_by_source = {
        source: _length_boundaries(list(lengths.values()), length_bins)
        for source, lengths in sorted(source_lengths.items())
    }
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        row = dict(row)
        row["length_bin"] = _length_bin(
            int(row["original_word_count"]), boundaries_by_source[row["source"]]
        )
        buckets[(row["source"], row["transform"], row["length_bin"])].append(row)

    rng = random.Random(seed)
    for key, rows in buckets.items():
        rng_for_bucket = random.Random(f"{seed}:{key}")
        rng_for_bucket.shuffle(rows)

    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    ordered_keys = sorted(buckets)
    rng.shuffle(ordered_keys)
    while len(selected) < sample_size:
        progress = False
        for key in ordered_keys:
            rows = buckets[key]
            while rows:
                row = rows.pop()
                family = str(row["content_family_id"])
                if family in used_families:
                    continue
                selected.append(row)
                used_families.add(family)
                progress = True
                break
            if len(selected) == sample_size:
                break
        if not progress:
            raise SemanticReviewError(
                f"could select only {len(selected)} unique-family rows from requested strata"
            )
    return selected, boundaries_by_source


def _empty_rating() -> dict[str, Any]:
    return {
        "meaning_preservation": None,
        "omission_severity": None,
        "unsupported_addition_severity": None,
        "contradiction_present": None,
        "fluent_and_interpretable": None,
        "notes": "",
    }


def build_packets(config: Mapping[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SemanticReviewError(f"schema_version must be {SCHEMA_VERSION!r}")
    protocol = config.get("protocol") or {}
    reviewer_ids = [str(value) for value in protocol.get("reviewer_ids") or []]
    if len(reviewer_ids) < 2 or len(set(reviewer_ids)) != len(reviewer_ids):
        raise SemanticReviewError("at least two unique reviewer_ids are required")
    seed = int(protocol["seed"])
    sample_size = int(protocol.get("sample_size", 50))
    length_bins = int(protocol.get("length_bins", 3))
    output_dir = Path((config.get("paths") or {})["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates, provenance = _load_candidates(config)
    selected, boundaries_by_source = _select_candidates(
        candidates, sample_size=sample_size, length_bins=length_bins, seed=seed
    )
    selected_by_identity = {
        (row["source"], row["transform"], row["row_key"]): row for row in selected
    }

    answer_key: dict[str, Any] = {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "seed": seed,
        "sample_size": sample_size,
        "length_bin_upper_boundaries_words_by_source": boundaries_by_source,
        "input_provenance": provenance,
        "reviewers": {},
    }
    packet_paths: dict[str, str] = {}
    csv_paths: dict[str, str] = {}
    for reviewer_id in reviewer_ids:
        packet_rows: list[dict[str, Any]] = []
        answers: dict[str, Any] = {}
        for identity, row in selected_by_identity.items():
            opaque_identity = "\0".join(str(value) for value in identity)
            review_id = _review_id(seed, reviewer_id, opaque_identity)
            packet_rows.append(
                {
                    "review_id": review_id,
                    "original_explanation": row["original_text"],
                    "transformed_explanation": row["transformed_text"],
                    "rating": _empty_rating(),
                }
            )
            answers[review_id] = {
                "source": row["source"],
                "transform": row["transform"],
                "row_key": row["row_key"],
                "split": row["split"],
                "row_index": row["row_index"],
                "doc_id": row["doc_id"],
                "content_family_id": row["content_family_id"],
                "length_bin": row["length_bin"],
                "original_word_count": row["original_word_count"],
                "transformed_word_count": row["transformed_word_count"],
                "original_sha256": _text_sha256(row["original_text"]),
                "transformed_sha256": _text_sha256(row["transformed_text"]),
                "prompt_sha256": row["prompt_sha256"],
                "transform_model": row["transform_model"],
            }
        random.Random(f"{seed}:{reviewer_id}:order").shuffle(packet_rows)
        packet = {
            "schema_version": PACKET_SCHEMA_VERSION,
            "reviewer_id": reviewer_id,
            "seed": seed,
            "instructions": [
                "Judge whether the transformed explanation preserves the meaning of the original explanation.",
                "Do not infer the model checkpoint or transformation that produced either text.",
                "Score meaning preservation from 1 (meaning substantially changed) to 5 (all substantive meaning preserved).",
                "Score omission and unsupported-addition severity from 0 (none) to 3 (severe).",
                "Mark contradiction only when the transformed explanation conflicts with the original.",
                "Compression may be fluent while omitting details; record those dimensions separately.",
            ],
            "scales": {
                "meaning_preservation": {"minimum": 1, "maximum": 5},
                "omission_severity": {"minimum": 0, "maximum": 3},
                "unsupported_addition_severity": {"minimum": 0, "maximum": 3},
            },
            "rows": packet_rows,
        }
        packet_path = output_dir / f"review_packet_{reviewer_id}.json"
        packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
        packet_paths[reviewer_id] = str(packet_path)

        csv_path = output_dir / f"review_packet_{reviewer_id}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "review_id",
                    "original_explanation",
                    "transformed_explanation",
                    "meaning_preservation_1_to_5",
                    "omission_severity_0_to_3",
                    "unsupported_addition_severity_0_to_3",
                    "contradiction_present_true_false",
                    "fluent_and_interpretable_true_false",
                    "notes",
                ),
            )
            writer.writeheader()
            for row in packet_rows:
                writer.writerow(
                    {
                        "review_id": row["review_id"],
                        "original_explanation": row["original_explanation"],
                        "transformed_explanation": row["transformed_explanation"],
                    }
                )
        csv_paths[reviewer_id] = str(csv_path)
        answer_key["reviewers"][reviewer_id] = {
            "packet_path": str(packet_path),
            "packet_sha256": _sha256_file(packet_path),
            "csv_path": str(csv_path),
            "csv_sha256": _sha256_file(csv_path),
            "answers": answers,
        }

    answer_path = output_dir / "answer_key.json"
    answer_path.write_text(json.dumps(answer_key, indent=2, sort_keys=True) + "\n")
    strata = Counter(
        (row["source"], row["transform"], int(row["length_bin"])) for row in selected
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "config": str(config_path) if config_path else None,
        "passed": (
            len(selected) == sample_size
            and len({row["content_family_id"] for row in selected}) == sample_size
            and all(str(row["original_text"]).strip() for row in selected)
            and all(str(row["transformed_text"]).strip() for row in selected)
        ),
        "sample_size": len(selected),
        "unique_content_families": len(
            {row["content_family_id"] for row in selected}
        ),
        "source_counts": dict(sorted(Counter(row["source"] for row in selected).items())),
        "transform_counts": dict(
            sorted(Counter(row["transform"] for row in selected).items())
        ),
        "length_bin_counts": {
            str(key): value
            for key, value in sorted(Counter(row["length_bin"] for row in selected).items())
        },
        "strata_counts": {
            f"{source}/{transform}/length_{length_bin}": count
            for (source, transform, length_bin), count in sorted(strata.items())
        },
        "length_bin_upper_boundaries_words_by_source": boundaries_by_source,
        "packet_paths": packet_paths,
        "csv_paths": csv_paths,
        "answer_key": str(answer_path),
        "answer_key_sha256": _sha256_file(answer_path),
    }
    report_path = output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    result = build_packets(config, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
