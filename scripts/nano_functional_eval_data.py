#!/usr/bin/env python3
"""Data preparation helpers for Nano functional-recovery evaluations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
NLA_ROOT = SCRIPT_DIR.parent / "external" / "natural_language_autoencoders"
for candidate in (SCRIPT_DIR, NLA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from nano_r33_source_rows import provenance_key  # noqa: E402


DEFAULT_CRITIC_TEMPLATE = (
    "Summary of the following text: <text>{explanation}</text> <summary>"
)
TEXT_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL)
EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)
CONTENT_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
CONTENT_NORMALIZATION_VERSION = "unicode_nfkc_casefold_words_v1"
CONTENT_FAMILY_ALGORITHM_VERSION = (
    "bottomk_plus_deterministic_prefix_jaccard_union_find_v2"
)


class FunctionalEvaluationError(ValueError):
    """Raised when functional-evaluation inputs are incomplete or ambiguous."""


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def normalized_content_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return tuple(CONTENT_WORD_RE.findall(normalized))


def normalized_content_text(text: str) -> str:
    return " ".join(normalized_content_tokens(text))


def token_shingles(tokens: tuple[str, ...], width: int) -> frozenset[str]:
    if width <= 0:
        raise FunctionalEvaluationError("shingle_width must be positive")
    if not tokens:
        return frozenset()
    if len(tokens) < width:
        return frozenset({"\x1f".join(tokens)})
    return frozenset(
        "\x1f".join(tokens[start : start + width])
        for start in range(len(tokens) - width + 1)
    )


def _stable_hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_hash_int(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 1.0 if left == right else 0.0
    return len(left & right) / len(left | right)


def _deterministic_jaccard_closure(
    documents: list[dict[str, Any]],
    *,
    similarity_threshold: float,
    union_find: _UnionFind,
) -> int:
    """Evaluate every possible threshold match using exact prefix filtering."""

    token_frequency: dict[str, int] = defaultdict(int)
    for document in documents:
        for token in document["shingles"]:
            token_frequency[token] += 1

    prefix_index: dict[str, list[int]] = defaultdict(list)
    evaluated_pairs: set[tuple[int, int]] = set()
    for right, document in enumerate(documents):
        shingles = document["shingles"]
        ordered = sorted(
            shingles,
            key=lambda token: (token_frequency[token], token),
        )
        prefix_length = max(
            1,
            len(ordered) - math.ceil(similarity_threshold * len(ordered)) + 1,
        )
        prefix = ordered[:prefix_length]
        candidates = {
            left
            for token in prefix
            for left in prefix_index.get(token, ())
        }
        for left in candidates:
            left_size = len(documents[left]["shingles"])
            right_size = len(shingles)
            if min(left_size, right_size) < similarity_threshold * max(
                left_size, right_size
            ):
                continue
            pair = (left, right)
            evaluated_pairs.add(pair)
            if union_find.find(left) == union_find.find(right):
                continue
            if _jaccard(documents[left]["shingles"], shingles) >= similarity_threshold:
                union_find.union(left, right)
        for token in prefix:
            prefix_index[token].append(right)
    return len(evaluated_pairs)


def build_content_families(
    rows: Iterable[dict[str, Any]],
    *,
    text_field: str = "detokenized_text_truncated",
    doc_id_field: str = "doc_id",
    shingle_width: int = 5,
    similarity_threshold: float = 0.80,
    signature_size: int = 32,
    candidate_min_shared: int = 4,
    max_signature_bucket_size: int = 256,
) -> dict[str, Any]:
    """Cluster documents by normalized source content, never by doc ID alone."""

    if not 0.0 <= similarity_threshold <= 1.0:
        raise FunctionalEvaluationError("similarity_threshold must be in [0, 1]")
    if signature_size <= 0 or candidate_min_shared <= 0:
        raise FunctionalEvaluationError("signature sizes must be positive")
    if max_signature_bucket_size < 2:
        raise FunctionalEvaluationError("max_signature_bucket_size must be at least 2")

    document_state: dict[str, dict[str, Any]] = {}
    row_count = 0
    for row_index, row in enumerate(rows):
        row_count += 1
        doc_id = str(row.get(doc_id_field) or "").strip()
        if not doc_id:
            raise FunctionalEvaluationError(
                f"row {row_index} has no non-empty {doc_id_field}"
            )
        tokens = normalized_content_tokens(str(row.get(text_field) or ""))
        if not tokens:
            raise FunctionalEvaluationError(
                f"row {row_index} doc_id={doc_id!r} has empty normalized source text"
            )
        normalized_text = " ".join(tokens)
        state = document_state.setdefault(
            doc_id,
            {"row_count": 0, "representative_text": "", "token_count": -1},
        )
        state["row_count"] += 1
        if (len(tokens), normalized_text) > (
            int(state["token_count"]),
            str(state["representative_text"]),
        ):
            state["representative_text"] = normalized_text
            state["token_count"] = len(tokens)

    if row_count == 0:
        raise FunctionalEvaluationError("content-family construction requires rows")

    documents: list[dict[str, Any]] = []
    for doc_id in sorted(document_state):
        state = document_state[doc_id]
        representative_text = str(state["representative_text"])
        tokens = tuple(representative_text.split())
        shingles = token_shingles(tokens, shingle_width)
        documents.append(
            {
                "doc_id": doc_id,
                "row_count": int(state["row_count"]),
                "normalized_text_sha256": _stable_hash_text(representative_text),
                "shingles": shingles,
                "signature": tuple(
                    sorted(_stable_hash_int(shingle) for shingle in shingles)[
                        :signature_size
                    ]
                ),
            }
        )

    union_find = _UnionFind(len(documents))
    exact_text_index: dict[str, int] = {}
    for index, document in enumerate(documents):
        content_hash = document["normalized_text_sha256"]
        if content_hash in exact_text_index:
            union_find.union(index, exact_text_index[content_hash])
        else:
            exact_text_index[content_hash] = index

    signature_buckets: dict[int, list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        for signature_value in document["signature"]:
            signature_buckets[signature_value].append(index)

    pair_shared_counts: dict[tuple[int, int], int] = defaultdict(int)
    evaluated_pairs: set[tuple[int, int]] = set()
    skipped_oversized_buckets = 0
    for members in signature_buckets.values():
        if len(members) > max_signature_bucket_size:
            skipped_oversized_buckets += 1
            continue
        for left_position, left in enumerate(members):
            for right in members[left_position + 1 :]:
                pair = (left, right) if left < right else (right, left)
                if pair in evaluated_pairs or union_find.find(left) == union_find.find(right):
                    continue
                pair_shared_counts[pair] += 1
                required_shared = min(
                    candidate_min_shared,
                    len(documents[left]["signature"]),
                    len(documents[right]["signature"]),
                )
                if pair_shared_counts[pair] < required_shared:
                    continue
                evaluated_pairs.add(pair)
                pair_shared_counts.pop(pair, None)
                if _jaccard(
                    documents[left]["shingles"],
                    documents[right]["shingles"],
                ) >= similarity_threshold:
                    union_find.union(left, right)

    deterministic_pairs_evaluated = _deterministic_jaccard_closure(
        documents,
        similarity_threshold=similarity_threshold,
        union_find=union_find,
    )

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(documents)):
        members_by_root[union_find.find(index)].append(index)

    doc_assignments: dict[str, str] = {}
    families: list[dict[str, Any]] = []
    for member_indices in members_by_root.values():
        content_hashes = sorted(
            {documents[index]["normalized_text_sha256"] for index in member_indices}
        )
        family_id = "cf_" + _stable_hash_text("\n".join(content_hashes))[:20]
        doc_ids = sorted(documents[index]["doc_id"] for index in member_indices)
        family_row_count = sum(documents[index]["row_count"] for index in member_indices)
        for doc_id in doc_ids:
            doc_assignments[doc_id] = family_id
        families.append(
            {
                "content_family_id": family_id,
                "document_count": len(doc_ids),
                "row_count": family_row_count,
                "doc_ids": doc_ids,
                "normalized_text_sha256": content_hashes,
            }
        )
    families.sort(key=lambda family: family["content_family_id"])
    return {
        "schema_version": "nano_content_family_manifest.v1",
        "algorithm": {
            "algorithm_version": CONTENT_FAMILY_ALGORITHM_VERSION,
            "normalization_version": CONTENT_NORMALIZATION_VERSION,
            "text_field": text_field,
            "doc_id_field": doc_id_field,
            "shingle_width": shingle_width,
            "similarity_threshold": similarity_threshold,
            "signature_size": signature_size,
            "candidate_min_shared": candidate_min_shared,
            "max_signature_bucket_size": max_signature_bucket_size,
            "exact_threshold_closure": "deterministic_prefix_filter",
        },
        "doc_assignments": dict(sorted(doc_assignments.items())),
        "families": families,
        "stats": {
            "row_count": row_count,
            "document_count": len(documents),
            "family_count": len(families),
            "near_duplicate_pairs_evaluated": len(evaluated_pairs),
            "deterministic_threshold_pairs_evaluated": (
                deterministic_pairs_evaluated
            ),
            "skipped_oversized_signature_buckets": skipped_oversized_buckets,
        },
    }


def assign_family_splits(
    family_manifest: dict[str, Any],
    *,
    split_weights: dict[str, float],
    seed: int,
    forbidden_splits_by_family: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    families = family_manifest.get("families") or []
    if not families:
        raise FunctionalEvaluationError("family manifest contains no families")
    if not split_weights or any(float(weight) <= 0 for weight in split_weights.values()):
        raise FunctionalEvaluationError("split_weights must all be positive")
    weight_total = float(sum(split_weights.values()))
    split_names = list(split_weights)
    family_ids = {str(family["content_family_id"]) for family in families}
    normalized_constraints = {
        str(family_id): sorted({str(split) for split in forbidden_splits})
        for family_id, forbidden_splits in (forbidden_splits_by_family or {}).items()
        if forbidden_splits
    }
    unknown_families = sorted(set(normalized_constraints) - family_ids)
    if unknown_families:
        raise FunctionalEvaluationError(
            f"split constraints reference unknown families: {unknown_families[:5]}"
        )
    unknown_splits = sorted(
        {
            split
            for forbidden_splits in normalized_constraints.values()
            for split in forbidden_splits
            if split not in split_weights
        }
    )
    if unknown_splits:
        raise FunctionalEvaluationError(
            f"split constraints reference unknown splits: {unknown_splits}"
        )
    ordered_families = sorted(
        families,
        key=lambda family: _stable_hash_text(
            f"{int(seed)}|{family['content_family_id']}"
        ),
    )
    total_rows = sum(int(family.get("row_count") or 0) for family in ordered_families)
    targets = {
        split: total_rows * float(weight) / weight_total
        for split, weight in split_weights.items()
    }
    assigned_rows = {split: 0 for split in split_names}
    assigned_family_counts = {split: 0 for split in split_names}
    family_splits: dict[str, str] = {}
    for family in ordered_families:
        family_id = str(family["content_family_id"])
        forbidden = set(normalized_constraints.get(family_id, []))
        allowed_splits = [split for split in split_names if split not in forbidden]
        if not allowed_splits:
            raise FunctionalEvaluationError(
                f"family {family_id} is forbidden from every split"
            )
        empty_allowed = [
            split for split in allowed_splits if assigned_family_counts[split] == 0
        ]
        candidates = empty_allowed or allowed_splits
        split = max(
            candidates,
            key=lambda name: (
                targets[name] - assigned_rows[name],
                -split_names.index(name),
            ),
        )
        family_splits[family_id] = split
        assigned_rows[split] += int(family.get("row_count") or 0)
        assigned_family_counts[split] += 1
    empty_splits = [split for split, count in assigned_family_counts.items() if count == 0]
    if empty_splits:
        raise FunctionalEvaluationError(
            f"split constraints leave splits without families: {empty_splits}"
        )
    split_families = {
        split: {family for family, assigned in family_splits.items() if assigned == split}
        for split in split_names
    }
    overlap: dict[str, list[str]] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap[f"{left}_{right}"] = sorted(split_families[left] & split_families[right])
    return {
        **family_manifest,
        "split_assignment": {
            "seed": int(seed),
            "weights": {name: float(weight) for name, weight in split_weights.items()},
            "constraint_family_count": len(normalized_constraints),
            "forbidden_splits_by_family": dict(sorted(normalized_constraints.items())),
            "constraint_sha256": _stable_hash_text(
                json.dumps(normalized_constraints, sort_keys=True, separators=(",", ":"))
            ),
        },
        "family_splits": dict(sorted(family_splits.items())),
        "split_summary": {
            split: {
                "family_count": len(split_families[split]),
                "row_count": assigned_rows[split],
            }
            for split in split_names
        },
        "overlap": overlap,
    }


def select_family_stratified_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_id = str(row.get("content_family_id") or "").strip()
        if not family_id:
            raise FunctionalEvaluationError(
                "family_stratified selection requires content_family_id on every row"
            )
        rows_by_family[family_id].append(row)
    ordered_families = sorted(
        rows_by_family,
        key=lambda family_id: _stable_hash_text(f"{seed}|{split}|{family_id}"),
    )
    for family_id, family_rows in rows_by_family.items():
        family_rows.sort(
            key=lambda row: _stable_hash_text(
                f"{seed}|{split}|{family_id}|{repr(provenance_key(row))}"
            )
        )
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for family_id in ordered_families:
            family_rows = rows_by_family[family_id]
            if depth < len(family_rows):
                selected.append(family_rows[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def load_content_family_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text())
    else:
        manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise FunctionalEvaluationError("content-family manifest must be an object")
    if manifest.get("schema_version") != "nano_content_family_manifest.v1":
        raise FunctionalEvaluationError(
            "content-family manifest must use schema_version nano_content_family_manifest.v1"
        )
    assignments = manifest.get("doc_assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise FunctionalEvaluationError("content-family manifest has no doc_assignments")
    return manifest


def content_family_overlap_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row.get("split") or "")
        family_id = str(row.get("content_family_id") or "")
        if split and family_id:
            split_families[split].add(family_id)
    overlap: dict[str, list[str]] = {}
    split_names = sorted(split_families)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap[f"{left}_{right}"] = sorted(
                split_families[left] & split_families[right]
            )
    return {
        "split_family_counts": {
            split: len(families) for split, families in sorted(split_families.items())
        },
        "overlap": overlap,
        "overlap_family_count": sum(len(families) for families in overlap.values()),
        "passed": all(not families for families in overlap.values()),
    }


def attach_content_family_ids(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    require_disjoint_splits: bool = False,
) -> dict[str, Any]:
    assignments = manifest.get("doc_assignments") or {}
    missing_docs: set[str] = set()
    for row in rows:
        doc_id = str(row.get("doc_id") or "")
        family_id = assignments.get(doc_id)
        if not family_id:
            missing_docs.add(doc_id or "<missing>")
            continue
        row["content_family_id"] = str(family_id)
    if missing_docs:
        raise FunctionalEvaluationError(
            f"content-family manifest is missing {len(missing_docs)} docs: "
            f"{sorted(missing_docs)[:10]}"
        )
    report = content_family_overlap_report(rows)
    if require_disjoint_splits and not report["passed"]:
        raise FunctionalEvaluationError(
            "content families overlap dataset splits: "
            f"{report['overlap']}"
        )
    return report


def load_content_family_coverage(path: str | Path) -> dict[str, Any]:
    coverage = json.loads(Path(path).read_text())
    if not isinstance(coverage, dict) or coverage.get("schema_version") != (
        "nano_content_family_exposure_report.v1"
    ):
        raise FunctionalEvaluationError(
            "content-family coverage must use schema_version "
            "nano_content_family_exposure_report.v1"
        )
    return coverage


def apply_family_holdout_coverage(
    rows: list[dict[str, Any]],
    coverage: dict[str, Any],
    *,
    required_splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    coverage_splits = coverage.get("splits") or {}
    requested = (
        tuple(str(split) for split in required_splits)
        if required_splits is not None
        else tuple(str(split) for split in coverage_splits)
    )
    if not requested or len(set(requested)) != len(requested):
        raise FunctionalEvaluationError(
            "required coverage splits must be non-empty and unique"
        )
    missing_coverage = sorted(set(requested) - set(coverage_splits))
    if missing_coverage:
        raise FunctionalEvaluationError(
            f"coverage report is missing required splits: {missing_coverage}"
        )
    eligible_docs_by_split = {
        str(split): {str(doc_id) for doc_id in (summary.get("eligible_doc_ids") or [])}
        for split, summary in coverage_splits.items()
        if str(split) in requested
    }
    eligible_families_by_split = {
        str(split): {
            str(family_id) for family_id in (summary.get("eligible_family_ids") or [])
        }
        for split, summary in coverage_splits.items()
        if str(split) in requested
    }
    eligible_row_counts = {split: 0 for split in eligible_docs_by_split}
    observed_families = {split: set() for split in eligible_docs_by_split}
    for row in rows:
        split = str(row.get("split") or "")
        eligible = False
        if split in eligible_docs_by_split:
            doc_id = str(row.get("doc_id") or "")
            family_id = str(row.get("content_family_id") or "")
            eligible = (
                doc_id in eligible_docs_by_split[split]
                and family_id in eligible_families_by_split[split]
            )
            if eligible:
                eligible_row_counts[split] += 1
                observed_families[split].add(family_id)
        row["publication_holdout_eligible"] = eligible
    overlap: dict[str, list[str]] = {}
    split_names = sorted(observed_families)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap[f"{left}_{right}"] = sorted(
                observed_families[left] & observed_families[right]
            )
    missing_splits = [
        split for split, count in eligible_row_counts.items() if count == 0
    ]
    return {
        "eligible_row_counts": dict(sorted(eligible_row_counts.items())),
        "eligible_family_counts": {
            split: len(families) for split, families in sorted(observed_families.items())
        },
        "overlap": overlap,
        "missing_splits": missing_splits,
        "passed": not missing_splits and all(not values for values in overlap.values()),
    }


def build_family_exposure_report(
    manifest: dict[str, Any],
    *,
    candidate_rows_by_split: dict[str, Iterable[dict[str, Any]]],
    exposure_rows_by_source: dict[str, Iterable[dict[str, Any]]],
    minimum_holdout_rows: int = 512,
) -> dict[str, Any]:
    if minimum_holdout_rows <= 0:
        raise FunctionalEvaluationError("minimum_holdout_rows must be positive")
    assignments = manifest.get("doc_assignments") or {}
    if not assignments:
        raise FunctionalEvaluationError("family manifest has no doc_assignments")

    exposed_families: set[str] = set()
    exposure_summary: dict[str, Any] = {}
    for source_name, rows in exposure_rows_by_source.items():
        source_row_count = 0
        source_families: set[str] = set()
        source_docs: set[str] = set()
        missing_docs: set[str] = set()
        for row in rows:
            source_row_count += 1
            doc_id = str(row.get("doc_id") or "")
            family_id = assignments.get(doc_id)
            if not family_id:
                missing_docs.add(doc_id or "<missing>")
                continue
            source_docs.add(doc_id)
            source_families.add(str(family_id))
        if missing_docs:
            raise FunctionalEvaluationError(
                f"exposure source {source_name!r} has docs missing from the family manifest: "
                f"{sorted(missing_docs)[:10]}"
            )
        exposed_families.update(source_families)
        exposure_summary[source_name] = {
            "row_count": source_row_count,
            "document_count": len(source_docs),
            "family_count": len(source_families),
        }

    split_summary: dict[str, Any] = {}
    eligible_families_by_split: dict[str, set[str]] = {}
    eligible_row_counts_by_split_family: dict[str, dict[str, int]] = {}
    eligible_docs_by_split_family: dict[str, dict[str, set[str]]] = {}
    for split, rows in candidate_rows_by_split.items():
        candidate_row_count = 0
        eligible_row_count = 0
        excluded_rows = 0
        candidate_families: set[str] = set()
        eligible_families: set[str] = set()
        eligible_docs: set[str] = set()
        eligible_row_counts_by_family: dict[str, int] = defaultdict(int)
        eligible_docs_by_family: dict[str, set[str]] = defaultdict(set)
        missing_docs: set[str] = set()
        for row in rows:
            candidate_row_count += 1
            doc_id = str(row.get("doc_id") or "")
            family_id = assignments.get(doc_id)
            if not family_id:
                missing_docs.add(doc_id or "<missing>")
                continue
            family_id = str(family_id)
            candidate_families.add(family_id)
            if family_id in exposed_families:
                excluded_rows += 1
                continue
            eligible_row_count += 1
            eligible_families.add(family_id)
            eligible_docs.add(doc_id)
            eligible_row_counts_by_family[family_id] += 1
            eligible_docs_by_family[family_id].add(doc_id)
        if missing_docs:
            raise FunctionalEvaluationError(
                f"candidate split {split!r} has docs missing from the family manifest: "
                f"{sorted(missing_docs)[:10]}"
            )
        eligible_families_by_split[split] = eligible_families
        eligible_row_counts_by_split_family[split] = dict(
            eligible_row_counts_by_family
        )
        eligible_docs_by_split_family[split] = dict(eligible_docs_by_family)
        split_summary[split] = {
            "candidate_row_count": candidate_row_count,
            "candidate_family_count": len(candidate_families),
            "pre_disjoint_eligible_row_count": eligible_row_count,
            "pre_disjoint_eligible_document_count": len(eligible_docs),
            "pre_disjoint_eligible_family_count": len(eligible_families),
            "excluded_exposed_row_count": excluded_rows,
            "minimum_holdout_rows": int(minimum_holdout_rows),
        }

    raw_holdout_overlap: dict[str, list[str]] = {}
    split_names = sorted(eligible_families_by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            raw_holdout_overlap[f"{left}_{right}"] = sorted(
                eligible_families_by_split[left] & eligible_families_by_split[right]
            )
    cross_split_families = {
        family_id
        for families in raw_holdout_overlap.values()
        for family_id in families
    }
    final_families_by_split: dict[str, set[str]] = {}
    for split in split_names:
        final_families = eligible_families_by_split[split] - cross_split_families
        final_families_by_split[split] = final_families
        row_counts = eligible_row_counts_by_split_family[split]
        docs_by_family = eligible_docs_by_split_family[split]
        eligible_row_count = sum(row_counts[family_id] for family_id in final_families)
        excluded_cross_split_rows = sum(
            row_counts[family_id]
            for family_id in cross_split_families
            if family_id in row_counts
        )
        eligible_docs = {
            doc_id
            for family_id in final_families
            for doc_id in docs_by_family[family_id]
        }
        split_summary[split].update(
            {
                "eligible_row_count": eligible_row_count,
                "eligible_document_count": len(eligible_docs),
                "eligible_family_count": len(final_families),
                "excluded_cross_split_family_row_count": excluded_cross_split_rows,
                "eligible_doc_ids": sorted(eligible_docs),
                "eligible_family_ids": sorted(final_families),
                "enough_rows": eligible_row_count >= minimum_holdout_rows,
            }
        )
    final_holdout_overlap: dict[str, list[str]] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            final_holdout_overlap[f"{left}_{right}"] = sorted(
                final_families_by_split[left] & final_families_by_split[right]
            )
    retain_existing = all(
        summary["enough_rows"] for summary in split_summary.values()
    ) and all(not families for families in final_holdout_overlap.values())
    return {
        "schema_version": "nano_content_family_exposure_report.v1",
        "minimum_holdout_rows": int(minimum_holdout_rows),
        "exposure_sources": exposure_summary,
        "exposed_family_count": len(exposed_families),
        "splits": split_summary,
        "raw_holdout_family_overlap": raw_holdout_overlap,
        "final_holdout_family_overlap": final_holdout_overlap,
        "holdout_family_overlap": final_holdout_overlap,
        "retain_existing_sft_checkpoints": bool(retain_existing),
        "clean_sft_retraining_required": not retain_existing,
    }


def read_generated_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FunctionalEvaluationError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            records.append(value)
    if not records:
        raise FunctionalEvaluationError(f"generated JSONL is empty: {path}")
    return records


def select_exact_split_rows(
    records: list[dict[str, Any]],
    validation_limit: int,
    test_limit: int,
    *,
    eval_splits: Sequence[str] = ("validation", "test"),
    selection_strategy: str = "row_order",
    selection_seed: int = 0,
) -> list[dict[str, Any]]:
    splits = tuple(str(split) for split in eval_splits)
    if not splits:
        raise FunctionalEvaluationError("at least one evaluation split is required")
    if len(splits) != len(set(splits)):
        raise FunctionalEvaluationError("evaluation splits must be unique")
    unknown = sorted(set(splits) - {"validation", "test"})
    if unknown:
        raise FunctionalEvaluationError(f"unsupported evaluation splits: {unknown}")
    limits = {"validation": validation_limit, "test": test_limit}
    if any(limits[split] <= 0 for split in splits):
        raise FunctionalEvaluationError("selected split limits must be positive")
    selected: list[dict[str, Any]] = []
    for split in splits:
        limit = limits[split]
        split_rows = [row for row in records if str(row.get("split")) == split]
        if selection_strategy == "row_order":
            split_rows.sort(
                key=lambda row: (int(row.get("row_index", -1)), repr(provenance_key(row)))
            )
        elif selection_strategy == "longest_prefix":
            def prefix_length(row: dict[str, Any]) -> int:
                if row.get("n_raw_tokens") is not None:
                    return int(row["n_raw_tokens"])
                if row.get("token_position") is not None:
                    return int(row["token_position"]) + 1
                raise FunctionalEvaluationError(
                    "longest_prefix selection requires n_raw_tokens or token_position"
                )

            split_rows.sort(
                key=lambda row: (
                    -prefix_length(row),
                    int(row.get("row_index", -1)),
                    repr(provenance_key(row)),
                )
            )
        elif selection_strategy == "family_stratified":
            split_rows = select_family_stratified_rows(
                split_rows,
                split=split,
                limit=limit,
                seed=selection_seed,
            )
        else:
            raise FunctionalEvaluationError(
                f"unknown selection_strategy: {selection_strategy}"
            )
        if len(split_rows) < limit:
            raise FunctionalEvaluationError(
                f"split {split!r} has {len(split_rows)} rows; requested {limit}"
            )
        chosen = split_rows[:limit]
        keys = [provenance_key(row) for row in chosen]
        if len(keys) != len(set(keys)):
            raise FunctionalEvaluationError(
                f"split {split!r} selection contains duplicate provenance keys"
            )
        selected.extend(chosen)
    return selected


def extract_generated_text(record: dict[str, Any], control: str, fallback: str) -> str:
    control_value = (record.get("controls") or {}).get(control)
    if isinstance(control_value, dict):
        raw = control_value.get("generated")
        if raw is None:
            raw = control_value.get("explanation")
    else:
        raw = record.get("generated")
        if raw is None:
            raw = record.get("explanation")
    text = str(raw or "")
    match = EXPLANATION_RE.search(text)
    if match:
        return match.group(1).strip()
    if fallback == "raw":
        return text.strip()
    if fallback != "empty":
        raise FunctionalEvaluationError(f"unknown generated-text fallback: {fallback}")
    return ""


def extract_teacher_text(source: dict[str, Any]) -> str:
    for name in ("api_explanation", "explanation", "teacher_explanation"):
        value = str(source.get(name) or "").strip()
        if value:
            return value
    prompt = str(source.get("prompt") or "")
    match = TEXT_RE.search(prompt)
    if match and match.group(1).strip():
        return match.group(1).strip()
    response = str(source.get("response") or "")
    match = EXPLANATION_RE.search(response)
    if match and match.group(1).strip():
        return match.group(1).strip()
    raise FunctionalEvaluationError(
        f"source row {provenance_key(source)!r} has no teacher explanation"
    )


def records_by_key(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = provenance_key(record)
        if key in output:
            raise FunctionalEvaluationError(f"duplicate generated provenance key: {key}")
        output[key] = record
    return output


def source_schema(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema_arrow.names)


def fixed_size_list_matrix(array: Any) -> np.ndarray:
    import pyarrow as pa

    if not pa.types.is_fixed_size_list(array.type):
        raise FunctionalEvaluationError(
            f"activation_vector must be fixed-size list, got {array.type}"
        )
    if array.null_count:
        raise FunctionalEvaluationError("activation_vector contains null rows")
    width = int(array.type.list_size)
    start = int(array.offset) * width
    flat = array.values.slice(start, len(array) * width).to_numpy(
        zero_copy_only=False
    )
    return np.asarray(flat, dtype=np.float32).reshape(len(array), width)


def source_mean_activation(path: Path, *, batch_size: int) -> np.ndarray:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = ["activation_vector"]
    has_split = "split" in parquet.schema_arrow.names
    if has_split:
        columns.append("split")
    total: np.ndarray | None = None
    count = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        vectors = batch.column(batch.schema.get_field_index("activation_vector"))
        if has_split:
            splits = batch.column(batch.schema.get_field_index("split"))
            vectors = pc.filter(vectors, pc.equal(splits, "train"))
        if len(vectors) == 0:
            continue
        values = fixed_size_list_matrix(vectors).astype(np.float64, copy=False)
        chunk_sum = values.sum(axis=0)
        total = chunk_sum if total is None else total + chunk_sum
        count += values.shape[0]
    if total is None or count == 0:
        raise FunctionalEvaluationError("source parquet has no rows for mean activation")
    return (total / count).astype(np.float32)


def _critic_prompts(texts: list[str], template: str) -> list[str]:
    if "{explanation}" not in template:
        raise FunctionalEvaluationError("critic template must contain {explanation}")
    return [template.format(explanation=text) for text in texts]


def load_ar_predictions(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], str, str]:
    from eval_nano_ar_miles_checkpoint import (
        _load_model_and_tokenizer,
        _resolve_hf_dir,
        _sidecar_template,
        predict_prompts,
    )

    hf_dir = _resolve_hf_dir(args.ar_checkpoint_dir)
    if args.critic_template:
        template = args.critic_template
    else:
        sidecar_sources = [hf_dir, args.critic_template_source, args.source_base_parquet]
        try:
            template = _sidecar_template(*sidecar_sources)
        except ValueError:
            template = DEFAULT_CRITIC_TEMPLATE

    texts_by_variant = {
        "candidate": [
            extract_generated_text(record, args.control, args.generated_text_fallback)
            for record in selected
        ],
        "teacher": [extract_teacher_text(source) for source in sources],
    }
    if args.sft_generated_jsonl:
        sft_by_key = records_by_key(read_generated_jsonl(args.sft_generated_jsonl))
        missing = [
            provenance_key(row)
            for row in selected
            if provenance_key(row) not in sft_by_key
        ]
        if missing:
            raise FunctionalEvaluationError(
                f"SFT generated JSONL is missing selected rows: {missing[:10]}"
            )
        texts_by_variant["sft"] = [
            extract_generated_text(
                sft_by_key[provenance_key(row)],
                args.control,
                args.generated_text_fallback,
            )
            for row in selected
        ]

    model, tokenizer = _load_model_and_tokenizer(
        hf_dir,
        torch_dtype=args.ar_torch_dtype,
        device_map=args.ar_device_map,
    )
    predictions: dict[str, np.ndarray] = {}
    try:
        for variant, texts in texts_by_variant.items():
            predictions[variant] = predict_prompts(
                model,
                tokenizer,
                _critic_prompts(texts, template),
                batch_size=args.ar_batch_size or args.batch_size,
                max_length=args.ar_max_length,
            )
    finally:
        del model, tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
    return predictions, template, str(hf_dir)


def within_document_shuffle(
    selected: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> tuple[dict[int, int], dict[str, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (record, source) in enumerate(zip(selected, sources, strict=True)):
        doc_id = source.get("doc_id", record.get("doc_id"))
        if doc_id not in {None, ""}:
            groups[str(doc_id)].append(index)
    mapping: dict[int, int] = {}
    for indices in groups.values():
        if len(indices) >= 2:
            for offset, index in enumerate(indices):
                mapping[index] = indices[(offset + 1) % len(indices)]
    return mapping, {
        "eligible_rows": len(mapping),
        "ineligible_rows": len(selected) - len(mapping),
    }


def build_variant_entries(
    selected: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    source_mean: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import torch

    from nano_r33_functional_core import rescale_direction

    gold = torch.tensor(
        np.asarray([row["activation_vector"] for row in sources], dtype=np.float32)
    )
    replacements = {
        variant: rescale_direction(torch.from_numpy(values), gold).cpu()
        for variant, values in predictions.items()
    }
    shuffle, shuffle_stats = within_document_shuffle(selected, sources)
    mean = torch.tensor(source_mean, dtype=torch.float32)
    entries: list[dict[str, Any]] = []
    for index, (record, source) in enumerate(zip(selected, sources, strict=True)):
        key = provenance_key(record)
        content_family_id = str(record.get("content_family_id") or "")
        prefix = [int(token) for token in source["token_ids_prefix"]]
        for variant, values in replacements.items():
            entries.append(
                {
                    "split": str(record["split"]),
                    "row_index": int(record["row_index"]),
                    "provenance_key": key,
                    "content_family_id": content_family_id,
                    "variant": variant,
                    "prefix": prefix,
                    "replacement": values[index],
                }
            )
        entries.extend(
            [
                {
                    "split": str(record["split"]),
                    "row_index": int(record["row_index"]),
                    "provenance_key": key,
                    "content_family_id": content_family_id,
                    "variant": "stored_gold",
                    "prefix": prefix,
                    "replacement": gold[index],
                },
                {
                    "split": str(record["split"]),
                    "row_index": int(record["row_index"]),
                    "provenance_key": key,
                    "content_family_id": content_family_id,
                    "variant": "mean",
                    "prefix": prefix,
                    "replacement": mean,
                },
                {
                    "split": str(record["split"]),
                    "row_index": int(record["row_index"]),
                    "provenance_key": key,
                    "content_family_id": content_family_id,
                    "variant": "zero",
                    "prefix": prefix,
                    "replacement": torch.zeros_like(gold[index]),
                },
            ]
        )
        if index in shuffle:
            entries.append(
                {
                    "split": str(record["split"]),
                    "row_index": int(record["row_index"]),
                    "provenance_key": key,
                    "content_family_id": content_family_id,
                    "variant": "shuffled",
                    "prefix": prefix,
                    "replacement": gold[shuffle[index]],
                }
            )
    return entries, shuffle_stats
