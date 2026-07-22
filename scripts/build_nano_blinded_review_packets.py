#!/usr/bin/env python3
"""Build deterministic blinded A/B packets for Nano NLA semantic review."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "nano_blinded_qualitative_review.v1"
PACKET_SCHEMA_VERSION = "nano_blinded_qualitative_review_packet.v1"
ANSWER_SCHEMA_VERSION = "nano_blinded_qualitative_review_answer_key.v1"
DEFAULT_DIMENSIONS = ("factuality", "coverage", "coherence")


class BlindedReviewError(ValueError):
    """Raised when the blinded-review protocol is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _text_sha256(text: str) -> str:
    return _sha256_bytes(text.encode())


def _review_id(seed: int, reviewer_id: str, split: str, row_index: int) -> str:
    material = f"{seed}\0{reviewer_id}\0{split}\0{row_index}".encode()
    return _sha256_bytes(material)[:20]


def _candidate_first(seed: int, reviewer_id: str, split: str, row_index: int) -> bool:
    material = f"swap\0{seed}\0{reviewer_id}\0{split}\0{row_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2 == 0


def _empty_ratings(dimensions: Sequence[str]) -> dict[str, Any]:
    return {
        "scores": {dimension: None for dimension in dimensions},
        "unsupported_specific_claim": None,
        "privacy_or_sensitive_content": None,
        "release_appropriate": None,
        "notes": "",
    }


def build_packets(config: Mapping[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise BlindedReviewError(f"schema_version must be {SCHEMA_VERSION!r}")
    paths = config.get("paths") or {}
    protocol = config.get("protocol") or {}
    panel_path = Path(paths["panel_json"])
    output_dir = Path(paths["output_dir"])
    reviewer_ids = [str(value) for value in protocol.get("reviewer_ids") or []]
    if len(reviewer_ids) < 2 or len(set(reviewer_ids)) != len(reviewer_ids):
        raise BlindedReviewError("at least two unique reviewer_ids are required")
    seed = int(protocol["seed"])
    dimensions = tuple(protocol.get("dimensions") or DEFAULT_DIMENSIONS)
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise BlindedReviewError("dimensions must be unique and non-empty")

    panel = json.loads(panel_path.read_text())
    panel_rows: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for split, split_payload in sorted((panel.get("splits") or {}).items()):
        for raw in split_payload.get("rows") or []:
            row = dict(raw)
            identity = (str(split), int(row["row_index"]))
            if identity in identities:
                raise BlindedReviewError(f"duplicate panel identity {identity!r}")
            identities.add(identity)
            for field in ("source_text", "candidate_text", "reference_text"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise BlindedReviewError(f"{identity!r} has invalid {field}")
            row["split"] = str(split)
            panel_rows.append(row)
    if not panel_rows:
        raise BlindedReviewError("panel contains no rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    answer_key: dict[str, Any] = {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "source_panel": str(panel_path),
        "source_panel_sha256": _sha256_file(panel_path),
        "seed": seed,
        "dimensions": list(dimensions),
        "reviewers": {},
    }
    packet_paths: dict[str, str] = {}

    for reviewer_id in reviewer_ids:
        packet_rows: list[dict[str, Any]] = []
        answers: dict[str, Any] = {}
        for row in panel_rows:
            split = row["split"]
            row_index = int(row["row_index"])
            review_id = _review_id(seed, reviewer_id, split, row_index)
            candidate_first = _candidate_first(seed, reviewer_id, split, row_index)
            text_a = row["candidate_text"] if candidate_first else row["reference_text"]
            text_b = row["reference_text"] if candidate_first else row["candidate_text"]
            role_a = "candidate" if candidate_first else "reference"
            role_b = "reference" if candidate_first else "candidate"
            packet_rows.append(
                {
                    "review_id": review_id,
                    "split": split,
                    "row_index": row_index,
                    "doc_id": row.get("doc_id"),
                    "doc_type": row.get("doc_type"),
                    "source_text": row["source_text"],
                    "text_a": text_a,
                    "text_b": text_b,
                    "ratings_a": _empty_ratings(dimensions),
                    "ratings_b": _empty_ratings(dimensions),
                    "preference": None,
                    "overall_notes": "",
                }
            )
            answers[review_id] = {
                "split": split,
                "row_index": row_index,
                "doc_id": row.get("doc_id"),
                "role_a": role_a,
                "role_b": role_b,
                "text_a_sha256": _text_sha256(text_a),
                "text_b_sha256": _text_sha256(text_b),
            }
        rng = random.Random(f"{seed}:{reviewer_id}:order")
        rng.shuffle(packet_rows)
        packet = {
            "schema_version": PACKET_SCHEMA_VERSION,
            "reviewer_id": reviewer_id,
            "seed": seed,
            "dimensions": list(dimensions),
            "score_scale": {
                "minimum": 1,
                "maximum": 5,
                "anchors": {
                    "1": "seriously deficient",
                    "3": "mixed or adequate",
                    "5": "strong and well supported",
                },
            },
            "instructions": [
                "Judge A and B independently against the source before choosing a preference.",
                "Factuality measures source support, not stylistic plausibility.",
                "Coverage measures whether the important source content is represented.",
                "Coherence measures clarity and internal readability.",
                "Mark unsupported_specific_claim when an output adds a concrete claim not supported by the source.",
                "Do not infer which output is the model candidate or teacher reference.",
            ],
            "preference_values": ["A", "B", "tie"],
            "rows": packet_rows,
        }
        packet_path = output_dir / f"review_packet_{reviewer_id}.json"
        packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
        packet_paths[reviewer_id] = str(packet_path)
        answer_key["reviewers"][reviewer_id] = {
            "packet_path": str(packet_path),
            "packet_sha256": _sha256_file(packet_path),
            "answers": answers,
        }

    answer_path = output_dir / "answer_key.json"
    answer_path.write_text(json.dumps(answer_key, indent=2, sort_keys=True) + "\n")
    return {
        "schema_version": SCHEMA_VERSION,
        "config": str(config_path) if config_path else None,
        "panel_rows": len(panel_rows),
        "reviewer_count": len(reviewer_ids),
        "packet_paths": packet_paths,
        "answer_key": str(answer_path),
        "answer_key_sha256": _sha256_file(answer_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    result = build_packets(config, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
